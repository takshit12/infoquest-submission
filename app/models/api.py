"""Wire-level Pydantic schemas for FastAPI request/response bodies."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.models.ranking import DebugPayload


# ============================================================
#                         /health
# ============================================================


class DependencyStatus(BaseModel):
    postgres: str  # "ok" | "error"
    vectorstore: str
    llm: str


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    version: str
    deps: DependencyStatus


class LivenessResponse(BaseModel):
    """Cheap process-up probe — no dependency pings."""

    status: str  # always "ok"
    version: str


# ============================================================
#                         /ingest
# ============================================================


class IngestRequest(BaseModel):
    reset: bool = False  # drop + re-create collection before ingesting
    limit: int | None = None  # for dev/testing: cap candidates ingested


class IngestResponse(BaseModel):
    candidates: int
    roles: int
    dense_docs: int
    sparse_docs: int
    elapsed_seconds: float


# ============================================================
#                          /chat
# ============================================================


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None
    top_k: int | None = None  # override server default
    include_why_not: bool = True


class ExpertHighlight(BaseModel):
    """Trimmed profile summary shown for ranked experts."""

    candidate_id: str
    full_name: str
    headline: str
    # The candidate's CURRENT role (is_current=true). Null if the matched
    # role was historical — see matched_role_* for what the ranking used.
    current_title: str | None = None
    current_company: str | None = None
    # The role that actually drove the ranking (== current when require_current
    # is not set; may be a historical role when the query asks for "former X").
    matched_role_title: str | None = None
    matched_role_company: str | None = None
    matched_role_is_current: bool | None = None
    seniority_tier: str | None = None  # derived from matched role's job title
    industry: str | None = None
    country: str | None = None
    city: str | None = None
    years_of_experience: int
    top_skills: list[str] = Field(default_factory=list)
    # Normalized ISO-639-1 language codes the candidate speaks. Populated
    # from role metadata; empty list if unknown.
    languages: list[str] = Field(default_factory=list)


class RankedExpert(BaseModel):
    rank: int
    expert: ExpertHighlight
    relevance_score: float = Field(ge=0.0, le=1.0)
    match_explanation: str
    why_not: str | None = None
    highlights: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    conversation_id: str
    query: str
    results: list[RankedExpert]
    returned_at: datetime
    # Bearer token returned only on the FIRST turn of a NEW conversation. The
    # caller must echo it as ``X-Session-Token`` on follow-up /chat calls and
    # on GET /conversations/{id}. Omitted on subsequent turns to keep the
    # secret out of repeated payloads.
    session_token: str | None = None
    debug: DebugPayload | None = None  # populated when ?debug=true

# ============================================================
#                        /experts
# ============================================================

class ExpertListItem(BaseModel):
    candidate_id: str
    full_name: str
    headline: str
    current_title: str | None = None
    current_company: str | None = None
    country: str | None = None
    years_of_experience: int


class ExpertListResponse(BaseModel):
    items: list[ExpertListItem]
    total: int
    offset: int
    limit: int


class ExpertDetail(BaseModel):
    """Full profile returned by GET /experts/{id}."""

    candidate_id: str
    full_name: str
    headline: str
    years_of_experience: int
    country: str | None = None
    city: str | None = None
    nationality: str | None = None
    roles: list[dict] = Field(default_factory=list)
    education: list[dict] = Field(default_factory=list)
    skills: list[dict] = Field(default_factory=list)
    languages: list[dict] = Field(default_factory=list)


# ============================================================
#                     /conversations
# ============================================================


class ConversationTurn(BaseModel):
    turn_index: int
    query: str
    query_intent: dict
    top_candidate_ids: list[str]
    timestamp: datetime


class ConversationHistory(BaseModel):
    conversation_id: str
    turns: list[ConversationTurn]
    created_at: datetime
    last_activity: datetime
