"""Per-query weight modulation.

The LLM-extracted QueryIntent already tells us WHICH signals the query
exercises (e.g., ``intent.industries`` populated → ``industry_match`` is
relevant). This resolver translates that "which" into weight shifts: boost
signals the query cares about, damp signals it doesn't, clamp drift,
renormalize to sum 1.

Pure function, no I/O. The reranker calls it at the start of ``rerank()`` to
derive per-query weights from the base weights loaded from env
(``SignalWeights``). Keeps the reranker itself stateless beyond its base.
"""
from __future__ import annotations

from app.models.domain import QueryIntent


# Semantic / lexical / recency signals are always relevant regardless of how
# structured the intent is — they carry match information that the intent
# fields can't fully substitute for. Never damp these.
_ALWAYS_ON = {"recency_decay", "dense_cosine", "bm25_score"}


def _presence_mask(intent: QueryIntent) -> dict[str, bool]:
    """True where the intent exercises the corresponding signal's dimension."""
    return {
        "industry_match":       bool(intent.industries),
        "function_match":       bool(intent.function),
        "seniority_match":      intent.seniority_band is not None,
        "skill_category_match": bool(intent.skill_categories),
        "trajectory_match":     intent.career_trajectory is not None,
        "recency_decay":        True,
        "dense_cosine":         True,
        "bm25_score":           True,
    }


def resolve_query_weights(
    intent: QueryIntent,
    base: dict[str, float],
    *,
    boost_present: float = 1.5,
    damp_absent: float = 0.5,
    max_drift: float = 2.0,
) -> dict[str, float]:
    """Base weights → per-query weights. Pure, deterministic, O(#signals).

    Contract:
      - Multiplicative boost/damp based on intent-field presence.
      - Always-on signals (semantic/lexical/recency) are never damped.
      - Per-weight clamp to [base/max_drift, base*max_drift] to prevent one
        signal from dominating even under pathological intents.
      - Output renormalized to sum = 1.
      - Low-confidence intents (``decomposer_source == "regex_fallback"``)
        bypass modulation — the regex parser is best-effort and a misparse
        shouldn't skew ranking. LLM and merged intents get full modulation.
    """
    if intent.decomposer_source == "regex_fallback":
        return dict(base)

    mask = _presence_mask(intent)
    shifted = {
        k: base[k] * (boost_present if (mask.get(k) or k in _ALWAYS_ON) else damp_absent)
        for k in base
    }
    clamped = {
        k: min(max_drift * base[k], max(base[k] / max_drift, shifted[k]))
        for k in shifted
    }
    total = sum(clamped.values())
    return {k: v / total for k, v in clamped.items()} if total > 0 else dict(base)
