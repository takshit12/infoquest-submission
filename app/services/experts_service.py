"""Expert list + detail lookup — implemented in feat/search worktree.

Uses profile_builder (from feat/ingest) to fetch from Postgres.
"""
from __future__ import annotations

from app.models.api import ExpertDetail, ExpertListResponse


def list_experts(
    *,
    offset: int,
    limit: int,
    country: str | None = None,
    industry: str | None = None,
    min_yoe: int | None = None,
) -> ExpertListResponse:
    raise NotImplementedError("experts_service.list_experts — feat/search")


def get_expert(candidate_id: str) -> ExpertDetail | None:
    raise NotImplementedError("experts_service.get_expert — feat/search")
