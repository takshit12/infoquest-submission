"""Tests for the per-query weight resolver.

The resolver is a pure function of (QueryIntent, base weights) → dict of
per-query weights. Tests below cover:

  - empty LLM intent → all non-always-on signals damped uniformly, output
    still normalized (so relative proportions among always-on signals are
    preserved; semantic/lexical/recency stay dominant)
  - single-field intent → the matching signal's weight rises, sum stays 1
  - all-fields intent → every signal boosted, output ≈ base renormalized
  - max_drift clamp actually engages
  - regex-fallback source → modulation skipped, identity
  - sum invariant for randomly-chosen inputs
"""
from __future__ import annotations

from app.models.domain import QueryIntent
from app.services.weight_resolver import resolve_query_weights


BASE: dict[str, float] = {
    "industry_match":       0.25,
    "function_match":       0.20,
    "seniority_match":      0.20,
    "skill_category_match": 0.10,
    "recency_decay":        0.08,
    "dense_cosine":         0.09,
    "bm25_score":           0.03,
    "trajectory_match":     0.05,
}


def _intent(**overrides) -> QueryIntent:
    kwargs = {"raw_query": "test"}
    kwargs.update(overrides)
    return QueryIntent(**kwargs)


def _is_normalized(w: dict[str, float]) -> bool:
    return abs(sum(w.values()) - 1.0) < 1e-9


def test_empty_llm_intent_damps_structured_signals():
    """LLM emits a blank intent: every structured signal is damped, always-on
    signals keep their boost → always-on share rises relative to base."""
    w = resolve_query_weights(_intent(), BASE)

    assert _is_normalized(w)
    # always-on signals should end up a larger share than in base
    always_on = ["recency_decay", "dense_cosine", "bm25_score"]
    base_share = sum(BASE[k] for k in always_on)
    out_share = sum(w[k] for k in always_on)
    assert out_share > base_share
    # structured signals shrink
    for k in ("industry_match", "function_match", "seniority_match",
              "skill_category_match", "trajectory_match"):
        assert w[k] < BASE[k]


def test_single_field_intent_boosts_matching_signal():
    w = resolve_query_weights(_intent(industries=["fintech"]), BASE)

    assert _is_normalized(w)
    # industry_match is the only structured signal that gets boosted
    assert w["industry_match"] > BASE["industry_match"]
    # other structured signals are damped
    for k in ("function_match", "seniority_match",
              "skill_category_match", "trajectory_match"):
        assert w[k] < BASE[k]


def test_all_fields_intent_boosts_everything():
    """Every intent field populated → every signal gets the 1.5× boost.
    After renormalization, relative proportions stay ≈ the base."""
    w = resolve_query_weights(
        _intent(
            industries=["fintech"],
            function="product management",
            seniority_band="senior",
            skill_categories=["payments"],
            career_trajectory="current",
        ),
        BASE,
    )

    assert _is_normalized(w)
    for k, v in BASE.items():
        # relative share preserved within tight tolerance
        assert abs(w[k] - v) < 1e-6


def test_max_drift_clamp_engages():
    """boost_present=10, max_drift=2 → every boosted signal clamps to 2×base.
    Absent structured signals have damp=1.0 so they stay at their base."""
    always_on = {"recency_decay", "dense_cosine", "bm25_score"}
    w = resolve_query_weights(
        _intent(industries=["fintech"]),
        BASE,
        boost_present=10.0,
        damp_absent=1.0,
        max_drift=2.0,
    )

    # pre-normalization: present or always-on → clamped to 2*base; rest → base
    pre = {
        k: (2.0 * BASE[k] if (k == "industry_match" or k in always_on) else BASE[k])
        for k in BASE
    }
    total = sum(pre.values())
    for k in BASE:
        expected = pre[k] / total
        assert abs(w[k] - expected) < 1e-9, f"{k}: {w[k]} vs {expected}"


def test_regex_fallback_is_identity():
    """Regex-fallback intents skip modulation — avoid amplifying a misparse."""
    w = resolve_query_weights(
        _intent(industries=["fintech"], decomposer_source="regex_fallback"),
        BASE,
    )
    assert w == BASE


def test_sum_invariant_under_many_intents():
    """Whatever the intent, the output must be normalized."""
    intents = [
        _intent(),
        _intent(industries=["fintech"]),
        _intent(function="product management"),
        _intent(seniority_band="director"),
        _intent(skill_categories=["ml", "nlp"]),
        _intent(career_trajectory="former"),
        _intent(industries=["ai"], function="research"),
        _intent(
            industries=["fintech"],
            function="pm",
            seniority_band="vp",
            skill_categories=["payments"],
            career_trajectory="current",
        ),
    ]
    for i in intents:
        w = resolve_query_weights(i, BASE)
        assert _is_normalized(w), f"not normalized for intent: {i!r}"
        assert all(v >= 0 for v in w.values())
