"""Admin endpoints for signal weights configuration."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.deps import invalidate_weights_cache, get_current_weights
from app.core.logging import get_logger
from app.db import fetch_signal_weights, update_signal_weights
from app.models.api import SignalWeightsResponse, SignalWeightsUpdateRequest


_log = get_logger("infoquest.api.admin")
router = APIRouter(tags=["admin"], prefix="/admin")


@router.get("/signal-weights", response_model=SignalWeightsResponse)
def get_signal_weights() -> SignalWeightsResponse:
    """Fetch current signal weights."""
    weights_dict = fetch_signal_weights()
    return SignalWeightsResponse(**weights_dict)


@router.post("/signal-weights", response_model=SignalWeightsResponse)
def update_signal_weights_endpoint(req: SignalWeightsUpdateRequest) -> SignalWeightsResponse:
    """Update signal weights (validates sum ≈ 1.0)."""
    weights_sum = sum([
        req.industry, req.function, req.seniority, req.skill_category,
        req.recency, req.dense, req.bm25, req.trajectory
    ])
    
    if abs(weights_sum - 1.0) > 0.01:
        raise HTTPException(
            status_code=400,
            detail=f"weights must sum to ~1.0 (got {weights_sum:.3f})",
        )
    
    success = update_signal_weights(
        industry=req.industry,
        function=req.function,
        seniority=req.seniority,
        skill_category=req.skill_category,
        recency=req.recency,
        dense=req.dense,
        bm25=req.bm25,
        trajectory=req.trajectory,
        changed_by="api",
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update weights")
    
    invalidate_weights_cache()
    _log.info("update_signal_weights.success", sum=weights_sum)
    
    return SignalWeightsResponse(**req.model_dump())
