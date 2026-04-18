"""Core domain objects passed between services.

Distinct from `models.api` (wire-level request/response schemas) and
`models.ranking` (score-breakdown records). Domain models are what the
ingestion + search pipelines actually manipulate internally.
"""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


SeniorityTier = Literal[
    "junior", "mid", "senior", "director", "vp", "head", "cxo", "staff_principal"
]


class RoleRecord(BaseModel):
    """One row of work_experience, denormalized with candidate + company fields.

    This is the unit that gets embedded and stored in the vector DB.
    Metadata is intentionally rich because filters happen on this object.
    """

    # ---- identity ----
    role_id: str
    candidate_id: str

    # ---- role-level ----
    job_title: str
    company: str
    industry: str | None = None
    seniority_tier: SeniorityTier | None = None
    start_date: date
    end_date: date | None = None
    is_current: bool
    description: str = ""
    role_years: float = 0.0

    # ---- candidate-level (denormalized for filter pushdown) ----
    candidate_headline: str = ""
    candidate_yoe: int = 0
    candidate_country: str | None = None
    candidate_city: str | None = None
    candidate_nationality: str | None = None
    skill_categories: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)  # normalized ISO-639-1 where possible

    def to_embedding_text(self) -> str:
        """Concatenate the fields we want in the embedded semantic payload."""
        end = self.end_date.isoformat() if self.end_date else "present"
        pieces = [
            f"{self.job_title} at {self.company}",
            f"[{self.industry}]" if self.industry else "",
            f"({self.start_date.isoformat()}–{end})",
        ]
        header = " ".join(p for p in pieces if p)
        body = f"{self.description}".strip()
        cand = f"Candidate: {self.candidate_headline}. YoE: {self.candidate_yoe}."
        skills = (
            f"Skill categories: {', '.join(self.skill_categories[:5])}."
            if self.skill_categories
            else ""
        )
        return ". ".join(s for s in [header + (": " + body if body else ""), cand, skills] if s)


class CandidateProfile(BaseModel):
    """Full candidate profile assembled from the normalized tables.

    Used for GET /experts/{id} and as the "parent" when aggregating role hits.
    """

    candidate_id: str
    first_name: str
    last_name: str
    headline: str = ""
    years_of_experience: int = 0
    city: str | None = None
    country: str | None = None
    nationality: str | None = None
    roles: list[RoleRecord] = Field(default_factory=list)
    education: list[dict] = Field(default_factory=list)
    skills: list[dict] = Field(default_factory=list)
    languages: list[dict] = Field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    @property
    def current_role(self) -> RoleRecord | None:
        for r in self.roles:
            if r.is_current:
                return r
        return None


class QueryIntent(BaseModel):
    """Structured output of the LLM query decomposer + regex fallback.

    HARD constraints (geographies, require_current, min_yoe, exclude) become
    prefilters at retrieval time. SOFT signal flags guide the reranker —
    the LLM says *which* signals apply; magnitudes are config constants.
    """

    # ---- raw + rewritten ----
    raw_query: str
    rewritten_search: str = ""  # for dense retrieval
    keywords: list[str] = Field(default_factory=list)  # for BM25

    # ---- HARD constraints (prefilters) ----
    geographies: list[str] = Field(default_factory=list)  # ISO-3166 alpha-2 country codes
    require_current: bool | None = None  # True/False/None (unspecified)
    min_yoe: int | None = None
    exclude_candidate_ids: list[str] = Field(default_factory=list)

    # ---- SOFT signal flags (which signals apply) ----
    function: str | None = None  # e.g., "regulatory affairs", "product management"
    industries: list[str] = Field(default_factory=list)  # canonical industry names
    seniority_band: (
        Literal["junior", "mid", "senior", "director", "vp", "head", "cxo"] | None
    ) = None
    skill_categories: list[str] = Field(default_factory=list)

    # ---- meta ----
    decomposer_source: Literal["llm", "regex_fallback", "merged"] = "llm"
    warnings: list[str] = Field(default_factory=list)


class ScoredRole(BaseModel):
    """One role record after retrieval+reranking, with its signal breakdown."""

    role: RoleRecord
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0  # post-RRF
    signal_scores: dict[str, float] = Field(default_factory=dict)
    final_score: float = 0.0


class ScoredCandidate(BaseModel):
    """A candidate with its MaxP-aggregated score and matched roles."""

    candidate_id: str
    candidate: CandidateProfile
    best_role: RoleRecord
    matched_roles: list[ScoredRole] = Field(default_factory=list)
    signal_scores: dict[str, float] = Field(default_factory=dict)
    relevance_score: float = 0.0
    prior_shortlist: bool = False
    mmr_rank: int | None = None

    # populated by the explainer service
    match_explanation: str = ""
    why_not: str | None = None
    highlights: list[str] = Field(default_factory=list)
