"""LLM-generated match explanation + why-not counter-explanations.

Filled in by the feat/search worktree.

Contract:
    def explain_matches(intent, candidates, llm) -> None:
        '''Mutates each ScoredCandidate in-place: sets match_explanation, highlights.
        One LLM call per top-K (can be parallelized with asyncio).'''

    def explain_why_not(intent, candidates, llm) -> None:
        '''Mutates candidates[5:] in-place: sets why_not.'''
"""
from __future__ import annotations

from app.models.domain import QueryIntent, ScoredCandidate
from app.ports.llm import LLMClient


def explain_matches(
    intent: QueryIntent, candidates: list[ScoredCandidate], llm: LLMClient
) -> None:
    raise NotImplementedError("explainer.explain_matches — feat/search")


def explain_why_not(
    intent: QueryIntent, candidates: list[ScoredCandidate], llm: LLMClient
) -> None:
    raise NotImplementedError("explainer.explain_why_not — feat/search")
