"""LLM-generated match explanation + why-not counter-explanations.

Contract:
    def explain_matches(intent, candidates, llm) -> None:
        Mutates each ScoredCandidate in-place: sets match_explanation, highlights.

    def explain_why_not(intent, candidates, llm) -> None:
        Mutates candidates in-place: sets why_not.

Implementation uses a ThreadPoolExecutor (max_workers=5) to run LLM calls
concurrently. Per-call exceptions are swallowed so the pipeline never raises.
"""
from __future__ import annotations

import concurrent.futures as cf
from functools import lru_cache
from pathlib import Path

from app.models.domain import QueryIntent, ScoredCandidate
from app.ports.llm import LLMClient


_EXPLAIN_PATH = Path(__file__).resolve().parent.parent / "prompts" / "explain_match.md"
_WHY_NOT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "why_not.md"


@lru_cache(maxsize=1)
def _load_explain_prompt() -> str:
    try:
        return _EXPLAIN_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


@lru_cache(maxsize=1)
def _load_why_not_prompt() -> str:
    try:
        return _WHY_NOT_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _signal_table(c: ScoredCandidate) -> str:
    rows = sorted(c.signal_scores.items(), key=lambda kv: kv[1], reverse=True)
    return ", ".join(f"{name}={score:.2f}" for name, score in rows)


def _candidate_summary(c: ScoredCandidate) -> str:
    name = (c.candidate.full_name if c.candidate else "") or ""
    headline = (c.candidate.headline if c.candidate else "") or (c.best_role.candidate_headline if c.best_role else "")
    country = (c.candidate.country if c.candidate else "") or (c.best_role.candidate_country if c.best_role else "") or ""
    yoe = int(c.candidate.years_of_experience if c.candidate else (c.best_role.candidate_yoe if c.best_role else 0))
    br = c.best_role
    if br is not None and br.is_current:
        cur_title = br.job_title or ""
        cur_company = br.company or ""
        cur_industry = br.industry or ""
        current_line = f"{cur_title} at {cur_company} ({cur_industry})".strip(" ()")
    else:
        if br is not None:
            current_line = f"(not current) {br.job_title or ''} at {br.company or ''} ({br.industry or ''})".strip(" ()")
        else:
            current_line = ""
    parts = [
        f"name={name}" if name else "",
        f"headline={headline}" if headline else "",
        f"country={country}" if country else "",
        f"yoe={yoe}",
        f"current={current_line}" if current_line else "",
    ]
    return "; ".join(p for p in parts if p)


def _build_highlights(c: ScoredCandidate) -> list[str]:
    """Short deterministic highlight strings."""
    out: list[str] = []
    br = c.best_role
    yoe = int(c.candidate.years_of_experience if c.candidate else (br.candidate_yoe if br else 0))
    if yoe:
        out.append(f"{yoe} yrs experience")
    if br:
        tier = (br.seniority_tier or "").replace("_", " ").title()
        if tier and br.company:
            out.append(f"{tier} at {br.company}")
        elif br.company:
            out.append(f"At {br.company}")
    country = (c.candidate.country if c.candidate else None) or (br.candidate_country if br else None)
    if country:
        out.append(f"Based in {country}")
    return out[:3]


# ----------------------------------------------------------------------
# explain_matches
# ----------------------------------------------------------------------


def _explain_one_match(
    intent: QueryIntent, rank: int, c: ScoredCandidate, llm: LLMClient, system: str
) -> None:
    user = (
        f"query: {intent.raw_query}\n"
        f"candidate: {_candidate_summary(c)}\n"
        f"signal_breakdown: {_signal_table(c)}\n"
        f"rank: {rank}"
    )
    try:
        text = llm.chat(system=system, user=user, max_tokens=200, temperature=0.2)
        c.match_explanation = (text or "").strip()
    except Exception:
        c.match_explanation = (
            "(explanation unavailable — scoring is deterministic; see signal_breakdown)"
        )


def explain_matches(
    intent: QueryIntent, candidates: list[ScoredCandidate], llm: LLMClient
) -> None:
    if not candidates:
        return
    system = _load_explain_prompt()

    # Populate highlights deterministically, regardless of LLM success.
    for c in candidates:
        c.highlights = _build_highlights(c)

    with cf.ThreadPoolExecutor(max_workers=5) as pool:
        futures = [
            pool.submit(_explain_one_match, intent, i + 1, c, llm, system)
            for i, c in enumerate(candidates)
        ]
        for f in cf.as_completed(futures):
            # Exceptions are already swallowed inside _explain_one_match, but defend anyway.
            try:
                f.result()
            except Exception:
                pass


# ----------------------------------------------------------------------
# explain_why_not
# ----------------------------------------------------------------------


def _top5_summary(top_candidates: list[ScoredCandidate]) -> str:
    if not top_candidates:
        return "(no reference top-5)"
    bands = sorted({(c.best_role.seniority_tier or "").strip() for c in top_candidates if c.best_role})
    industries = sorted({(c.best_role.industry or "").strip() for c in top_candidates if c.best_role and c.best_role.industry})
    countries = sorted({(c.best_role.candidate_country or "").strip() for c in top_candidates if c.best_role and c.best_role.candidate_country})
    parts = []
    if bands:
        parts.append("seniority " + "/".join(b for b in bands if b))
    if industries:
        parts.append("industries " + ",".join(i for i in industries if i))
    if countries:
        parts.append("geo " + ",".join(c for c in countries if c))
    return "; ".join(parts) or "(no consistent pattern)"


def _explain_one_why_not(
    intent: QueryIntent,
    rank: int,
    c: ScoredCandidate,
    llm: LLMClient,
    system: str,
    top5_line: str,
) -> None:
    user = (
        f"query: {intent.raw_query}\n"
        f"top5_summary: {top5_line}\n"
        f"candidate: {_candidate_summary(c)}\n"
        f"signal_breakdown: {_signal_table(c)}\n"
        f"rank: {rank}"
    )
    try:
        text = llm.chat(system=system, user=user, max_tokens=100, temperature=0.2)
        c.why_not = (text or "").strip() or None
    except Exception:
        # Leave why_not as None on any failure.
        c.why_not = None


def explain_why_not(
    intent: QueryIntent,
    candidates: list[ScoredCandidate],
    llm: LLMClient,
    top_candidates: list[ScoredCandidate] | None = None,
) -> None:
    if not candidates:
        return
    system = _load_why_not_prompt()
    top5_line = _top5_summary(top_candidates or [])
    with cf.ThreadPoolExecutor(max_workers=5) as pool:
        futures = [
            pool.submit(_explain_one_why_not, intent, i + 1, c, llm, system, top5_line)
            for i, c in enumerate(candidates)
        ]
        for f in cf.as_completed(futures):
            try:
                f.result()
            except Exception:
                pass
