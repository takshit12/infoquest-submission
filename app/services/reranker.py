"""WeightedSignalReranker — composes signal scores and produces ScoredCandidate list.

Filled in by the feat/search worktree. Contract:

    class WeightedSignalReranker:
        def __init__(self, weights: dict[str, float], signals: dict[str, SignalFn],
                     maxp_bonus: float, maxp_cap: float): ...
        def rerank(self, intent, roles) -> list[ScoredCandidate]:
            # 1. compute per-role weighted score from signals
            # 2. group by candidate_id; candidate score = max + log-scaled bonus for multi-matches
            # 3. sort descending, return top-N
"""
from __future__ import annotations

from app.models.domain import QueryIntent, ScoredCandidate, ScoredRole


class WeightedSignalReranker:
    def __init__(
        self,
        weights: dict[str, float],
        signals: dict,
        maxp_bonus: float = 0.05,
        maxp_cap: float = 0.15,
    ) -> None:
        self.weights = weights
        self.signals = signals
        self.maxp_bonus = maxp_bonus
        self.maxp_cap = maxp_cap

    def rerank(
        self, intent: QueryIntent, roles: list[ScoredRole]
    ) -> list[ScoredCandidate]:
        raise NotImplementedError("WeightedSignalReranker.rerank — feat/search")
