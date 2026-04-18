"""Follow-up classification + QueryIntent merge + soft prior boost.

Filled in by the feat/search worktree.

Key semantics (derived from critique):
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

from typing import Literal

from app.models.domain import QueryIntent, ScoredCandidate
from app.ports.llm import LLMClient


def classify_followup(
    query: str, prior_intent: QueryIntent, llm: LLMClient
) -> Literal["refine", "new"]:
    raise NotImplementedError("conversation.classify_followup — feat/search")


def merge_intent(prior: QueryIntent, current: QueryIntent) -> QueryIntent:
    raise NotImplementedError("conversation.merge_intent — feat/search")


def apply_prior_boost(
    candidates: list[ScoredCandidate],
    prior_ids: list[str],
    boost: float,
) -> None:
    raise NotImplementedError("conversation.apply_prior_boost — feat/search")
