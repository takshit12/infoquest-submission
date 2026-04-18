"""GET /experts/{id}, GET /experts — direct access to candidate profiles.

Implementation in feat/search worktree.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.models.api import ExpertDetail, ExpertListResponse


router = APIRouter(tags=["experts"])


@router.get("/experts", response_model=ExpertListResponse)
def list_experts(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=25, ge=1, le=100),
    country: str | None = None,
    industry: str | None = None,
    min_yoe: int | None = Query(default=None, ge=0, le=50),
) -> ExpertListResponse:
    try:
        from app.services import experts_service
    except ImportError as e:  # pragma: no cover
        raise HTTPException(status_code=501, detail=f"experts not available: {e}")
    return experts_service.list_experts(
        offset=offset,
        limit=limit,
        country=country,
        industry=industry,
        min_yoe=min_yoe,
    )


@router.get("/experts/{candidate_id}", response_model=ExpertDetail)
def get_expert(candidate_id: str) -> ExpertDetail:
    try:
        from app.services import experts_service
    except ImportError as e:  # pragma: no cover
        raise HTTPException(status_code=501, detail=f"experts not available: {e}")
    result = experts_service.get_expert(candidate_id)
    if result is None:
        raise HTTPException(status_code=404, detail="expert not found")
    return result
