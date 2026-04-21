"""Fine-grained ranking breakdown records — used for /chat?debug=true output."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SignalScore(BaseModel):
    name: str
    raw: float
    # Weights are query-adaptive: base_weight is the static env-driven value,
    # applied_weight is what the resolver produced for this specific query.
    # `weighted` uses applied_weight — that's the contribution to the final score.
    base_weight: float
    applied_weight: float
    weighted: float  # raw * applied_weight
    note: str | None = None


class RankingBreakdown(BaseModel):
    candidate_id: str
    signals: list[SignalScore]
    maxp_bonus: float = 0.0
    prior_shortlist_boost: float = 0.0
    final_score: float
    rank: int


class StageTiming(BaseModel):
    stage: str
    elapsed_ms: float


class DebugPayload(BaseModel):
    """Everything that goes into /chat?debug=true."""

    query_intent: dict = Field(default_factory=dict)
    hard_filters: dict = Field(default_factory=dict)
    dense_top: list[dict] = Field(default_factory=list)
    sparse_top: list[dict] = Field(default_factory=list)
    fused_roles: list[dict] = Field(default_factory=list)
    ranking_breakdown: list[RankingBreakdown] = Field(default_factory=list)
    mmr_selections: list[dict] = Field(default_factory=list)
    timings: list[StageTiming] = Field(default_factory=list)
