"""Expert list + detail lookup.

`list_experts` does a direct Postgres SELECT (read-only conn from app.db).
`get_expert` prefers `profile_builder.fetch_candidate_profile`; on
NotImplementedError it falls back to a minimal Postgres query so the endpoint
stays usable while the ingest worktree is being built.
"""
from __future__ import annotations

from typing import Any

import psycopg2.extras

from app.core.logging import get_logger
from app.db import source_conn
from app.models.api import ExpertDetail, ExpertListItem, ExpertListResponse


_log = get_logger("infoquest.experts_service")


def list_experts(
    *,
    offset: int,
    limit: int,
    country: str | None = None,
    industry: str | None = None,
    min_yoe: int | None = None,
) -> ExpertListResponse:
    """Paginated expert listing with optional filters."""
    filters_sql: list[str] = []
    params: list[Any] = []

    if country:
        filters_sql.append("UPPER(co.code) = UPPER(%s)")
        params.append(country)
    if min_yoe is not None:
        filters_sql.append("ca.years_of_experience >= %s")
        params.append(int(min_yoe))
    if industry:
        filters_sql.append("comp.industry = %s")
        params.append(industry)

    where_clause = ("WHERE " + " AND ".join(filters_sql)) if filters_sql else ""

    # Count query mirrors filters (but doesn't need current-role join to count
    # unless we're filtering by industry).
    count_sql = f"""
        SELECT COUNT(DISTINCT ca.id) AS n
        FROM candidates ca
        LEFT JOIN cities ci ON ci.id = ca.city_id
        LEFT JOIN countries co ON co.id = ci.country_id
        LEFT JOIN work_experience cur
            ON cur.candidate_id = ca.id AND cur.is_current = TRUE
        LEFT JOIN companies comp ON comp.id = cur.company_id
        {where_clause}
    """

    list_sql = f"""
        SELECT DISTINCT ON (ca.id)
            ca.id AS candidate_id,
            ca.first_name,
            ca.last_name,
            COALESCE(ca.headline, '') AS headline,
            COALESCE(ca.years_of_experience, 0) AS years_of_experience,
            co.name AS country_name,
            cur.job_title AS current_title,
            comp.name AS current_company
        FROM candidates ca
        LEFT JOIN cities ci ON ci.id = ca.city_id
        LEFT JOIN countries co ON co.id = ci.country_id
        LEFT JOIN work_experience cur
            ON cur.candidate_id = ca.id AND cur.is_current = TRUE
        LEFT JOIN companies comp ON comp.id = cur.company_id
        {where_clause}
        ORDER BY ca.id
        OFFSET %s LIMIT %s
    """

    with source_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(count_sql, tuple(params))
        total = int((cur.fetchone() or {}).get("n", 0))

        cur.execute(list_sql, tuple(params) + (int(offset), int(limit)))
        rows = cur.fetchall() or []

    items: list[ExpertListItem] = []
    for r in rows:
        first = r.get("first_name") or ""
        last = r.get("last_name") or ""
        full = f"{first} {last}".strip()
        items.append(
            ExpertListItem(
                candidate_id=str(r.get("candidate_id")),
                full_name=full,
                headline=r.get("headline") or "",
                current_title=r.get("current_title"),
                current_company=r.get("current_company"),
                country=r.get("country_name"),
                years_of_experience=int(r.get("years_of_experience") or 0),
            )
        )

    return ExpertListResponse(
        items=items,
        total=total,
        offset=int(offset),
        limit=int(limit),
    )


def _profile_to_detail(profile) -> ExpertDetail:
    roles = []
    for r in profile.roles or []:
        roles.append(
            {
                "role_id": r.role_id,
                "job_title": r.job_title,
                "company": r.company,
                "industry": r.industry,
                "seniority_tier": r.seniority_tier,
                "start_date": r.start_date.isoformat() if r.start_date else None,
                "end_date": r.end_date.isoformat() if r.end_date else None,
                "is_current": r.is_current,
                "description": r.description,
            }
        )
    return ExpertDetail(
        candidate_id=profile.candidate_id,
        full_name=profile.full_name,
        headline=profile.headline or "",
        years_of_experience=int(profile.years_of_experience or 0),
        country=profile.country,
        city=profile.city,
        nationality=profile.nationality,
        roles=roles,
        education=list(profile.education or []),
        skills=list(profile.skills or []),
        languages=list(profile.languages or []),
    )


def _stub_detail_from_db(candidate_id: str) -> ExpertDetail | None:
    """Fallback: minimal SELECT-based detail when profile_builder isn't ready."""
    header_sql = """
        SELECT ca.id, ca.first_name, ca.last_name,
               COALESCE(ca.headline, '') AS headline,
               COALESCE(ca.years_of_experience, 0) AS yoe,
               co.name AS country,
               ci.name AS city,
               nat.name AS nationality
        FROM candidates ca
        LEFT JOIN cities ci ON ci.id = ca.city_id
        LEFT JOIN countries co ON co.id = ci.country_id
        LEFT JOIN countries nat ON nat.id = ca.nationality_id
        WHERE ca.id = %s
    """
    work_sql = """
        SELECT we.id AS role_id, we.job_title, we.start_date, we.end_date,
               we.is_current, we.description,
               comp.name AS company, comp.industry AS industry
        FROM work_experience we
        LEFT JOIN companies comp ON comp.id = we.company_id
        WHERE we.candidate_id = %s
        ORDER BY we.is_current DESC, we.start_date DESC NULLS LAST
    """
    with source_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(header_sql, (candidate_id,))
        row = cur.fetchone()
        if row is None:
            return None
        cur.execute(work_sql, (candidate_id,))
        roles = cur.fetchall() or []

    full_name = f"{row.get('first_name') or ''} {row.get('last_name') or ''}".strip()
    role_dicts: list[dict[str, Any]] = []
    for r in roles:
        role_dicts.append(
            {
                "role_id": str(r.get("role_id")),
                "job_title": r.get("job_title") or "",
                "company": r.get("company") or "",
                "industry": r.get("industry"),
                "start_date": (
                    r.get("start_date").isoformat() if r.get("start_date") else None
                ),
                "end_date": (
                    r.get("end_date").isoformat() if r.get("end_date") else None
                ),
                "is_current": bool(r.get("is_current")),
                "description": r.get("description") or "",
            }
        )

    return ExpertDetail(
        candidate_id=str(row.get("id")),
        full_name=full_name,
        headline=row.get("headline") or "",
        years_of_experience=int(row.get("yoe") or 0),
        country=row.get("country"),
        city=row.get("city"),
        nationality=row.get("nationality"),
        roles=role_dicts,
    )


def get_expert(candidate_id: str) -> ExpertDetail | None:
    """Prefer profile_builder; fall back to minimal DB stub on NotImplementedError."""
    try:
        from app.services import profile_builder  # lazy import
    except Exception as e:  # pragma: no cover
        _log.warning(
            "profile_builder_import_failed",
            candidate_id=candidate_id,
            error=str(e),
        )
        profile_builder = None  # type: ignore

    if profile_builder is not None:
        try:
            profile = profile_builder.fetch_candidate_profile(candidate_id)
            if profile is None:
                return None
            return _profile_to_detail(profile)
        except NotImplementedError:
            pass
        except Exception as e:
            # Defensive: fall through to stub
            _log.warning(
                "profile_builder_fetch_failed",
                candidate_id=candidate_id,
                error=str(e),
            )

    try:
        return _stub_detail_from_db(candidate_id)
    except Exception as e:
        _log.warning(
            "expert_stub_lookup_failed",
            candidate_id=candidate_id,
            error=str(e),
        )
        return None
