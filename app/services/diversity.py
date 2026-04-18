"""Maximal Marginal Relevance (MMR) diversification.

Filled in by the feat/search worktree.
Primary diversity dimension: current_company. Secondary: industry.
MMR(d) = λ * rel(d) - (1-λ) * max_{d' in selected} sim(d, d')
"""
from __future__ import annotations

from app.models.domain import ScoredCandidate


def _similarity(a: ScoredCandidate, b: ScoredCandidate) -> float:
    """Same-company match dominates, then industry overlap."""
    ar = a.best_role
    br = b.best_role
    if ar is None or br is None:
        return 0.0
    if ar.company and ar.company == br.company:
        return 1.0
    if ar.industry and ar.industry == br.industry:
        return 0.5
    return 0.0


def apply_mmr(
    candidates: list[ScoredCandidate],
    top_k: int,
    lambda_: float = 0.7,
) -> list[ScoredCandidate]:
    """Greedy MMR selection. Assumes candidates are already sorted desc by relevance."""
    if not candidates:
        return []
    if len(candidates) <= top_k:
        for idx, c in enumerate(candidates):
            c.mmr_rank = idx
        return candidates

    selected: list[ScoredCandidate] = [candidates[0]]
    remaining: list[ScoredCandidate] = list(candidates[1:])

    while len(selected) < top_k and remaining:

        def mmr_score(c: ScoredCandidate) -> float:
            rel = float(c.relevance_score)
            max_sim = max((_similarity(c, s) for s in selected), default=0.0)
            return lambda_ * rel - (1.0 - lambda_) * max_sim

        best = max(remaining, key=mmr_score)
        remaining.remove(best)
        selected.append(best)

    for idx, c in enumerate(selected):
        c.mmr_rank = idx
    return selected
