"""Maximal Marginal Relevance (MMR) diversification.

Filled in by the feat/search worktree.
Primary diversity dimension: current_company. Secondary: industry.
MMR(d) = λ * rel(d) - (1-λ) * max_{d' in selected} sim(d, d')
"""
from __future__ import annotations

from app.models.domain import ScoredCandidate


def apply_mmr(
    candidates: list[ScoredCandidate],
    top_k: int,
    lambda_: float = 0.7,
) -> list[ScoredCandidate]:
    raise NotImplementedError("diversity.apply_mmr — feat/search")
