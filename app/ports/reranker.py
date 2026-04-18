"""Reranker port: transforms a candidate list by applying ranking logic.

Intentionally separate from individual signal functions so that a cross-encoder
reranker or a learning-to-rank model can be dropped in as a different Reranker
implementation without touching the signals module.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.models.domain import QueryIntent, ScoredCandidate, ScoredRole


@runtime_checkable
class Reranker(Protocol):
    def rerank(
        self, intent: QueryIntent, roles: list[ScoredRole]
    ) -> list[ScoredCandidate]:
        """Consume role-level fused scores, produce candidate-level reranked list.

        Implementations are expected to:
          - group roles by candidate_id (MaxP)
          - compute per-signal scores
          - combine into final score
          - sort descending
        """
