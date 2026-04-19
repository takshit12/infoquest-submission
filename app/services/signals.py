"""The 8 ranking signals as pure functions + the SIGNALS registry.

Filled in by the feat/search worktree. Each signal returns a float in [0, 1].
Weights come from `app.core.config.SignalWeights` — the LLM does NOT choose
magnitudes; it only tells us which signals apply (via QueryIntent).

Expected signals:
  - industry_match      : role industry vs intent.industries (with aliases)
  - function_match      : job_title/description vs intent.function
  - seniority_match     : seniority_tier vs intent.seniority_band (banded distance)
  - skill_category_match: skill_categories overlap with intent.skill_categories
  - recency_decay       : exp decay on (today - end_date_or_today)
  - dense_cosine        : pass-through of ScoredRole.dense_score
  - bm25_score          : pass-through of normalized ScoredRole.sparse_score
  - trajectory_match    : per-role view of intent.career_trajectory
                          (current/former/transitioning/ascending)

Registry shape:
    SIGNALS: dict[str, SignalFn] = { "industry_match": industry_match_fn, ... }

The reranker pulls weights from settings at runtime; signals stay weight-free.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Callable

from app.models.domain import QueryIntent, ScoredRole
from app.taxonomies import industry_aliases
from app.taxonomies.seniority import band_distance


SignalFn = Callable[[ScoredRole, QueryIntent], float]


# ------------------------------------------------------------------
# Convention recap (see design notes):
#   - Intent doesn't specify the relevant field → return 0.5 (neutral).
#   - Intent specifies and role matches → 1.0.
#   - Intent specifies, partial match → 0.3–0.7.
#   - Intent specifies, no match → 0.0.
# ------------------------------------------------------------------


def industry_match(role: ScoredRole, intent: QueryIntent) -> float:
    if not intent.industries:
        return 0.5
    role_industry = role.role.industry or ""
    role_canonical = industry_aliases.canonicalize(role_industry)
    # also treat exact-case canonical as a canonical
    if role_canonical is None and role_industry:
        # fall back: compare raw lowercased against intent industries
        role_canonical_raw = role_industry.strip().lower()
    else:
        role_canonical_raw = (role_canonical or "").strip().lower()

    for target in intent.industries:
        tgt_canonical = industry_aliases.canonicalize(target) or target
        if tgt_canonical.strip().lower() == role_canonical_raw:
            return 1.0
    return 0.0


def function_match(role: ScoredRole, intent: QueryIntent) -> float:
    if not intent.function:
        return 0.5
    haystack = " ".join(
        [
            role.role.job_title or "",
            role.role.description or "",
            role.role.candidate_headline or "",
        ]
    ).lower()
    needle = intent.function.strip().lower()
    if not needle:
        return 0.5
    if needle in haystack:
        return 1.0
    # Partial match: any significant (len>3) token of a multi-word function
    tokens = [t for t in needle.split() if len(t) > 3]
    if len(tokens) > 1:
        for t in tokens:
            if t in haystack:
                return 0.5
    return 0.0


def seniority_match(role: ScoredRole, intent: QueryIntent) -> float:
    if intent.seniority_band is None:
        return 0.5
    tier = role.role.seniority_tier or "mid"
    dist = band_distance(tier, intent.seniority_band)
    if dist <= 0:
        base = 1.0
    elif dist == 1:
        base = 0.7
    elif dist == 2:
        base = 0.4
    else:
        base = 0.1

    # For senior/director/vp/head/cxo, scale by YoE (floor 0.6 multiplier).
    if intent.seniority_band in {"senior", "director", "vp", "head", "cxo"}:
        yoe = max(0, int(role.role.candidate_yoe or 0))
        yoe_scale = min(1.0, yoe / 15.0)
        multiplier = max(0.6, yoe_scale)
        base *= multiplier

    # clamp to [0, 1]
    return max(0.0, min(1.0, base))


def skill_category_match(role: ScoredRole, intent: QueryIntent) -> float:
    if not intent.skill_categories:
        return 0.5
    role_cats = {c.strip().lower() for c in (role.role.skill_categories or []) if c}
    intent_cats = {c.strip().lower() for c in intent.skill_categories if c}
    if not role_cats or not intent_cats:
        return 0.0
    inter = role_cats & intent_cats
    union = role_cats | intent_cats
    if not union:
        return 0.0
    return len(inter) / len(union)


def recency_decay(role: ScoredRole, intent: QueryIntent) -> float:
    # is_current OR end_date None → treat as current
    if role.role.is_current or role.role.end_date is None:
        return 1.0
    today = date.today()
    days = max(0, (today - role.role.end_date).days)
    years = days / 365.25
    val = math.exp(-years / 10.0)
    return max(0.0, min(1.0, val))


def dense_cosine(role: ScoredRole, intent: QueryIntent) -> float:
    return max(0.0, min(1.0, float(role.dense_score)))


def bm25_score(role: ScoredRole, intent: QueryIntent) -> float:
    return max(0.0, min(1.0, float(role.sparse_score)))


def trajectory_match(role: ScoredRole, intent: QueryIntent) -> float:
    """Per-role view of the intent's career trajectory.

    Limitation: a single-role view can't truly detect "transitioning" (that
    needs multi-role inspection). For transitioning queries we return neutral
    so the signal contributes a flat half-weight without distorting ranking.
    For current/former we mirror the binary `is_current`. For ascending we
    reward the 5–12 YoE / senior-tier sweet spot.
    """
    if intent.career_trajectory is None:
        return 0.5  # neutral: signal not exercised by this query

    r = role.role
    if intent.career_trajectory == "current":
        return 1.0 if r.is_current else 0.0
    if intent.career_trajectory == "former":
        # Even with `require_current=False` as a hard filter we still score
        # to differentiate within the historical-role set; treat any current
        # leak (filter bypass) as a soft penalty rather than zero.
        return 1.0 if not r.is_current else 0.2
    if intent.career_trajectory == "ascending":
        yoe = max(0, int(r.candidate_yoe or 0))
        tier = r.seniority_tier
        if 5 <= yoe <= 12 and tier in {"senior", "director", "vp", "head"}:
            return 1.0
        if yoe < 5 or yoe > 15:
            return 0.3  # too early or too late to be "ascending"
        return 0.6
    if intent.career_trajectory == "transitioning":
        # Single-role inspection can't see a transition; neutral is honest.
        # See DESIGN §11 known-limitations note.
        return 0.5
    return 0.5


SIGNALS: dict[str, SignalFn] = {
    "industry_match": industry_match,
    "function_match": function_match,
    "seniority_match": seniority_match,
    "skill_category_match": skill_category_match,
    "recency_decay": recency_decay,
    "dense_cosine": dense_cosine,
    "bm25_score": bm25_score,
    "trajectory_match": trajectory_match,
}
