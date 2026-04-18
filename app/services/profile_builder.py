"""Assemble CandidateProfile + RoleRecord[] from the normalized source Postgres.

Streams candidates in batches (ORDER BY id, LIMIT/OFFSET) and performs bulk
secondary lookups per batch. Domain objects are populated with denormalized
fields so downstream filters can be pushed into the vector-store metadata.

Key design choices:
  - Skill names in the source DB contain noise ("intent", "fullstory10") — we
    use the top-level skill_category name instead (all categories have
    parent_id IS NULL today).
  - Language names are inconsistent; normalized via taxonomies.languages.normalize.
  - country codes are ISO-3166 alpha-2 (countries.code), fallback nationality
    code when a candidate has no city->country.
  - Seniority is derived from the job title prefix via tier_from_title.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import date
from typing import Any

import psycopg2.extras

from app.db import source_conn
from app.models.domain import CandidateProfile, RoleRecord
from app.taxonomies.languages import normalize as normalize_language
from app.taxonomies.seniority import tier_from_title

# Candidates-per-batch for bulk secondary queries.
_BATCH_SIZE = 200


# ============================================================
#                       Public API
# ============================================================


def iter_candidate_profiles(
    limit: int | None = None,
) -> Iterable[CandidateProfile]:
    """Stream fully-denormalized CandidateProfile objects.

    Uses keyset-like OFFSET pagination in batches of 200 candidates; for each
    batch, secondary tables are fetched in a single round-trip per table with
    `WHERE candidate_id = ANY(%s)`.
    """
    with source_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        yielded = 0
        offset = 0
        while True:
            remaining = None if limit is None else max(0, limit - yielded)
            if remaining == 0:
                break
            batch_size = _BATCH_SIZE if remaining is None else min(_BATCH_SIZE, remaining)

            cands = _fetch_candidate_batch(cur, offset=offset, limit=batch_size)
            if not cands:
                break

            cand_ids = [c["id"] for c in cands]
            roles_by_cand = _fetch_roles(cur, cand_ids)
            edu_by_cand = _fetch_education(cur, cand_ids)
            skills_by_cand = _fetch_skills(cur, cand_ids)
            langs_by_cand = _fetch_languages(cur, cand_ids)

            for c in cands:
                profile = _assemble_profile(
                    c,
                    roles_by_cand.get(c["id"], []),
                    edu_by_cand.get(c["id"], []),
                    skills_by_cand.get(c["id"], []),
                    langs_by_cand.get(c["id"], []),
                )
                yield profile
                yielded += 1
                if limit is not None and yielded >= limit:
                    return

            offset += len(cands)
            if len(cands) < batch_size:
                break


def iter_role_records(limit: int | None = None) -> Iterable[RoleRecord]:
    """Flatten iter_candidate_profiles to yield one RoleRecord per work_experience row.

    `limit` is applied to the ROLE count (not candidate count), to make
    `--limit N` for ingest dev-runs produce exactly N embedded docs.
    """
    yielded = 0
    # Candidates average ~2.4 roles → loop until we hit the limit or run dry.
    for profile in iter_candidate_profiles(limit=None):
        for role in profile.roles:
            if limit is not None and yielded >= limit:
                return
            yield role
            yielded += 1
        if limit is not None and yielded >= limit:
            return


def fetch_candidate_profile(candidate_id: str) -> CandidateProfile | None:
    """Targeted single-candidate assembly used by GET /experts/{id}."""
    with source_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(_CANDIDATE_SELECT + " WHERE ca.id = %s", (candidate_id,))
        row = cur.fetchone()
        if not row:
            return None

        cand_ids = [row["id"]]
        roles = _fetch_roles(cur, cand_ids).get(row["id"], [])
        edu = _fetch_education(cur, cand_ids).get(row["id"], [])
        skills = _fetch_skills(cur, cand_ids).get(row["id"], [])
        langs = _fetch_languages(cur, cand_ids).get(row["id"], [])
        return _assemble_profile(row, roles, edu, skills, langs)


# ============================================================
#                       SQL fragments
# ============================================================


# Column list used by both the batch and single-fetch paths. Casts ids to text
# so we don't have to call str() in Python everywhere.
_CANDIDATE_SELECT = """
    SELECT
        ca.id::text       AS id,
        ca.first_name,
        ca.last_name,
        COALESCE(ca.headline, '')          AS headline,
        COALESCE(ca.years_of_experience, 0) AS years_of_experience,
        ci.name                             AS city,
        co.code                             AS country_code,
        nat.code                            AS nationality_code
    FROM candidates ca
    LEFT JOIN cities    ci  ON ci.id  = ca.city_id
    LEFT JOIN countries co  ON co.id  = ci.country_id
    LEFT JOIN countries nat ON nat.id = ca.nationality_id
"""


def _fetch_candidate_batch(cur, *, offset: int, limit: int) -> list[dict[str, Any]]:
    cur.execute(
        _CANDIDATE_SELECT + " ORDER BY ca.id LIMIT %s OFFSET %s",
        (limit, offset),
    )
    return list(cur.fetchall())


def _fetch_roles(cur, candidate_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    if not candidate_ids:
        return {}
    cur.execute(
        """
        SELECT
            we.id::text           AS role_id,
            we.candidate_id::text AS candidate_id,
            we.job_title,
            we.start_date,
            we.end_date,
            we.is_current,
            COALESCE(we.description, '') AS description,
            co.name  AS company,
            co.industry,
            cc.code  AS company_country_code
        FROM work_experience we
        JOIN companies  co ON co.id = we.company_id
        LEFT JOIN countries cc ON cc.id = co.country_id
        WHERE we.candidate_id = ANY(%s::uuid[])
        ORDER BY we.candidate_id, we.is_current DESC, we.start_date DESC
        """,
        (candidate_ids,),
    )
    out: dict[str, list[dict[str, Any]]] = {}
    for r in cur.fetchall():
        out.setdefault(r["candidate_id"], []).append(dict(r))
    return out


def _fetch_education(cur, candidate_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    if not candidate_ids:
        return {}
    cur.execute(
        """
        SELECT
            e.candidate_id::text AS candidate_id,
            d.name AS degree,
            f.name AS field_of_study,
            i.name AS institution,
            e.start_year,
            e.graduation_year,
            e.grade
        FROM education e
        JOIN degrees         d ON d.id = e.degree_id
        JOIN fields_of_study f ON f.id = e.field_of_study_id
        JOIN institutions    i ON i.id = e.institution_id
        WHERE e.candidate_id = ANY(%s::uuid[])
        ORDER BY e.candidate_id, e.graduation_year DESC NULLS LAST
        """,
        (candidate_ids,),
    )
    out: dict[str, list[dict[str, Any]]] = {}
    for r in cur.fetchall():
        out.setdefault(r["candidate_id"], []).append(dict(r))
    return out


def _fetch_skills(cur, candidate_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    if not candidate_ids:
        return {}
    # We walk skills -> skill_categories. Every category in the current DB has
    # parent_id IS NULL; if that changes, coalesce to the top-level by following
    # parent_id. We expose both skill_name and category for downstream use.
    cur.execute(
        """
        SELECT
            cs.candidate_id::text AS candidate_id,
            s.name  AS skill,
            sc.name AS category,
            cs.years_of_experience,
            cs.proficiency_level
        FROM candidate_skills cs
        JOIN skills           s  ON s.id  = cs.skill_id
        JOIN skill_categories sc ON sc.id = s.category_id
        WHERE cs.candidate_id = ANY(%s::uuid[])
        """,
        (candidate_ids,),
    )
    out: dict[str, list[dict[str, Any]]] = {}
    for r in cur.fetchall():
        out.setdefault(r["candidate_id"], []).append(dict(r))
    return out


def _fetch_languages(cur, candidate_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    if not candidate_ids:
        return {}
    cur.execute(
        """
        SELECT
            cl.candidate_id::text AS candidate_id,
            l.name  AS language,
            p.name  AS proficiency,
            p.rank  AS proficiency_rank
        FROM candidate_languages cl
        JOIN languages          l ON l.id = cl.language_id
        JOIN proficiency_levels p ON p.id = cl.proficiency_level_id
        WHERE cl.candidate_id = ANY(%s::uuid[])
        ORDER BY cl.candidate_id, p.rank DESC
        """,
        (candidate_ids,),
    )
    out: dict[str, list[dict[str, Any]]] = {}
    for r in cur.fetchall():
        out.setdefault(r["candidate_id"], []).append(dict(r))
    return out


# ============================================================
#                       Assembly
# ============================================================


def _assemble_profile(
    cand: dict[str, Any],
    roles_raw: list[dict[str, Any]],
    edu_raw: list[dict[str, Any]],
    skills_raw: list[dict[str, Any]],
    langs_raw: list[dict[str, Any]],
) -> CandidateProfile:
    """Build a CandidateProfile with all roles denormalized."""
    country = cand.get("country_code") or cand.get("nationality_code")
    nationality = cand.get("nationality_code")
    city = cand.get("city")
    headline = cand.get("headline") or ""
    yoe = int(cand.get("years_of_experience") or 0)

    # candidate-level skill_categories: DISTINCT top-level category names,
    # ordered by the MAX years_of_experience per category, truncated to 5.
    cat_max_yoe: dict[str, float] = {}
    for s in skills_raw:
        cat = s.get("category")
        if not cat:
            continue
        y = float(s.get("years_of_experience") or 0)
        if cat not in cat_max_yoe or y > cat_max_yoe[cat]:
            cat_max_yoe[cat] = y
    skill_categories = [
        c for c, _ in sorted(cat_max_yoe.items(), key=lambda kv: kv[1], reverse=True)
    ][:5]

    # Languages: distinct normalized ISO-639-1 codes (falls back to lowercase).
    seen_langs: set[str] = set()
    languages: list[str] = []
    for l in langs_raw:
        code = normalize_language(l.get("language") or "")
        if not code or code in seen_langs:
            continue
        seen_langs.add(code)
        languages.append(code)

    # Build RoleRecords.
    roles: list[RoleRecord] = []
    for r in roles_raw:
        roles.append(
            _build_role(
                r,
                candidate_id=cand["id"],
                candidate_headline=headline,
                candidate_yoe=yoe,
                candidate_country=country,
                candidate_city=city,
                candidate_nationality=nationality,
                skill_categories=skill_categories,
                languages=languages,
            )
        )

    # Preserve native row dicts for the Full-profile API payload
    # (GET /experts/{id} accepts list[dict]).
    education_list = [
        {
            "institution": e.get("institution"),
            "degree": e.get("degree"),
            "field_of_study": e.get("field_of_study"),
            "start_year": e.get("start_year"),
            "graduation_year": e.get("graduation_year"),
            "grade": e.get("grade"),
        }
        for e in edu_raw
    ]
    skills_list = [
        {
            "name": s.get("skill"),
            "category": s.get("category"),
            "years_of_experience": s.get("years_of_experience"),
            "proficiency_level": s.get("proficiency_level"),
        }
        for s in skills_raw
    ]
    languages_list = [
        {
            "name": l.get("language"),
            "code": normalize_language(l.get("language") or ""),
            "proficiency": l.get("proficiency"),
            "rank": l.get("proficiency_rank"),
        }
        for l in langs_raw
    ]

    return CandidateProfile(
        candidate_id=cand["id"],
        first_name=cand["first_name"],
        last_name=cand["last_name"],
        headline=headline,
        years_of_experience=yoe,
        city=city,
        country=country,
        nationality=nationality,
        roles=roles,
        education=education_list,
        skills=skills_list,
        languages=languages_list,
    )


def _build_role(
    row: dict[str, Any],
    *,
    candidate_id: str,
    candidate_headline: str,
    candidate_yoe: int,
    candidate_country: str | None,
    candidate_city: str | None,
    candidate_nationality: str | None,
    skill_categories: list[str],
    languages: list[str],
) -> RoleRecord:
    start: date = row["start_date"]
    end: date | None = row.get("end_date")
    ref_end = end or date.today()
    role_years = max(0.0, round((ref_end - start).days / 365.25, 1))

    return RoleRecord(
        role_id=row["role_id"],
        candidate_id=candidate_id,
        job_title=row["job_title"],
        company=row["company"],
        industry=row.get("industry"),
        seniority_tier=tier_from_title(row["job_title"]),
        start_date=start,
        end_date=end,
        is_current=bool(row["is_current"]),
        description=row.get("description") or "",
        role_years=role_years,
        candidate_headline=candidate_headline,
        candidate_yoe=candidate_yoe,
        candidate_country=candidate_country,
        candidate_city=candidate_city,
        candidate_nationality=candidate_nationality,
        skill_categories=list(skill_categories),
        languages=list(languages),
    )


__all__ = [
    "iter_candidate_profiles",
    "iter_role_records",
    "fetch_candidate_profile",
]
