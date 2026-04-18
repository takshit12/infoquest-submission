"""The 7 ranking signals as pure functions + the SIGNALS registry.

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

Registry shape:
    SIGNALS: dict[str, SignalFn] = { "industry_match": industry_match_fn, ... }

The reranker pulls weights from settings at runtime; signals stay weight-free.
"""
from __future__ import annotations

from typing import Callable

from app.models.domain import QueryIntent, ScoredRole


SignalFn = Callable[[ScoredRole, QueryIntent], float]


def industry_match(role: ScoredRole, intent: QueryIntent) -> float:
    raise NotImplementedError("signals.industry_match — feat/search")


def function_match(role: ScoredRole, intent: QueryIntent) -> float:
    raise NotImplementedError("signals.function_match — feat/search")


def seniority_match(role: ScoredRole, intent: QueryIntent) -> float:
    raise NotImplementedError("signals.seniority_match — feat/search")


def skill_category_match(role: ScoredRole, intent: QueryIntent) -> float:
    raise NotImplementedError("signals.skill_category_match — feat/search")


def recency_decay(role: ScoredRole, intent: QueryIntent) -> float:
    raise NotImplementedError("signals.recency_decay — feat/search")


def dense_cosine(role: ScoredRole, intent: QueryIntent) -> float:
    return role.dense_score


def bm25_score(role: ScoredRole, intent: QueryIntent) -> float:
    return role.sparse_score


SIGNALS: dict[str, SignalFn] = {
    "industry_match": industry_match,
    "function_match": function_match,
    "seniority_match": seniority_match,
    "skill_category_match": skill_category_match,
    "recency_decay": recency_decay,
    "dense_cosine": dense_cosine,
    "bm25_score": bm25_score,
}
