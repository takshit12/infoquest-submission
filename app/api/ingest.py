"""POST /ingest — trigger the ingestion pipeline.

Implementation in feat/ingest worktree. This file contains the route skeleton
so the URL is registered in the scaffold commit.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.deps import EmbedderDep, SparseRetrieverDep, VectorStoreDep
from app.models.api import IngestRequest, IngestResponse


router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    req: IngestRequest,
    embedder: EmbedderDep,
    vs: VectorStoreDep,
    sparse: SparseRetrieverDep,
) -> IngestResponse:
    # Imported lazily so unit tests that never hit this route don't require
    # the feat/ingest implementation to be present.
    try:
        from app.services.ingestion import run_ingest
    except ImportError as e:  # pragma: no cover
        raise HTTPException(status_code=501, detail=f"ingest not available: {e}")
    return run_ingest(
        embedder=embedder,
        vector_store=vs,
        sparse=sparse,
        reset=req.reset,
        limit=req.limit,
    )
