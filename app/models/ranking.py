"""Fine-grained ranking breakdown records — used for /chat?debug=true output."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SignalScore(BaseModel):
    name: str
    raw: float
    weight: float
    weighted: float  # raw * weight
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
