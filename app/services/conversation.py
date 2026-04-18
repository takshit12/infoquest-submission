"""Follow-up classification + QueryIntent merge + soft prior boost.

Key semantics:
  - If the follow-up is a REFINEMENT, merge its intent with the prior intent;
    then re-run retrieval GLOBALLY (not capped to prior shortlist).
  - After retrieval, apply a `prior_shortlist_boost` to any candidate that
    appeared in the prior top-K. Soft prior, not a filter.
  - If the follow-up is a NEW TOPIC, ignore prior intent.

Contract:
    def classify_followup(query, prior_intent, llm) -> Literal["refine", "new"]
    def merge_intent(prior: QueryIntent, current: QueryIntent) -> QueryIntent
    def apply_prior_boost(candidates, prior_ids, boost) -> None  # mutates
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from app.core.logging import get_logger
from app.models.domain import QueryIntent, ScoredCandidate
from app.ports.llm import LLMClient


_log = get_logger("infoquest.conversation")

_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "classify_followup.md"
)


@lru_cache(maxsize=1)
def _load_prompt() -> str:
    try:
        return _PROMPT_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


def _intent_summary(intent: QueryIntent) -> str:
    """One-liner summary for the LLM follow-up classifier."""
    bits: list[str] = []
    if intent.function:
        bits.append(f"function={intent.function}")
    if intent.industries:
        bits.append(f"industries={','.join(intent.industries)}")
    if intent.seniority_band:
        bits.append(f"seniority={intent.seniority_band}")
    if intent.geographies:
        bits.append(f"geo={','.join(intent.geographies)}")
    if intent.require_current is not None:
        bits.append(f"require_current={intent.require_current}")
    if intent.min_yoe is not None:
        bits.append(f"min_yoe={intent.min_yoe}")
    if not bits:
        bits.append(f'query="{intent.raw_query}"')
    return "; ".join(bits)


def classify_followup(
    query: str, prior_intent: QueryIntent, llm: LLMClient
) -> Literal["refine", "new"]:
    """Classify new query as 'refine' (same population) or 'new' (different)."""
    system = _load_prompt()
    prior_line = _intent_summary(prior_intent)
    user = (
        f"prior_intent: {prior_line}\n"
        f"new_query: {query}\n"
        "Return only the JSON object."
    )
    try:
        result = llm.chat_json(system=system, user=user)
        if isinstance(result, dict):
            cls = str(result.get("classification", "")).strip().lower()
            if cls in {"refine", "new"}:
                return cls  # type: ignore[return-value]
    except Exception as exc:
        _log.warning("classify_followup_error", error=str(exc))
    return "new"


def merge_intent(prior: QueryIntent, current: QueryIntent) -> QueryIntent:
    """Merge a refinement into the prior intent.

    Rules:
      - raw_query: current
      - rewritten_search: current or prior
      - keywords: UNION deduped
      - scalar fields (function, seniority_band, require_current, min_yoe):
            current if specified, else prior
      - geographies / industries: if current is a non-empty SUBSET of prior,
            treat as strict narrowing and use current; else UNION
      - exclude_candidate_ids: UNION
      - decomposer_source: 'merged'
    """

    def union_list(a: list, b: list) -> list:
        seen: set = set()
        out: list = []
        for x in list(a or []) + list(b or []):
            if x is None:
                continue
            key = x
            if key in seen:
                continue
            seen.add(key)
            out.append(x)
        return out

    def narrow_or_union(prior_list: list, current_list: list) -> list:
        pset = set(prior_list or [])
        cset = set(current_list or [])
        if pset and cset and cset.issubset(pset):
            # strict narrowing — keep current
            # preserve order from current
            return list(current_list or [])
        return union_list(prior_list or [], current_list or [])

    merged = QueryIntent(
        raw_query=current.raw_query,
        rewritten_search=(current.rewritten_search or prior.rewritten_search or ""),
        keywords=union_list(prior.keywords, current.keywords),
        geographies=narrow_or_union(prior.geographies, current.geographies),
        require_current=(
            current.require_current if current.require_current is not None else prior.require_current
        ),
        min_yoe=(current.min_yoe if current.min_yoe is not None else prior.min_yoe),
        exclude_candidate_ids=union_list(
            prior.exclude_candidate_ids, current.exclude_candidate_ids
        ),
        function=(current.function or prior.function),
        industries=narrow_or_union(prior.industries, current.industries),
        seniority_band=(current.seniority_band or prior.seniority_band),
        skill_categories=union_list(prior.skill_categories, current.skill_categories),
        decomposer_source="merged",
        warnings=list(prior.warnings or []) + list(current.warnings or []),
    )
    return merged


def apply_prior_boost(
    candidates: list[ScoredCandidate],
    prior_ids: list[str],
    boost: float,
) -> None:
    """Mutate candidates in place: add boost to any candidate in prior_ids, re-sort."""
    prior_set = set(prior_ids or [])
    if not prior_set:
        return
    for c in candidates:
        if c.candidate_id in prior_set:
            c.relevance_score = float(min(1.0, float(c.relevance_score) + float(boost)))
            c.prior_shortlist = True
    candidates.sort(key=lambda c: c.relevance_score, reverse=True)
