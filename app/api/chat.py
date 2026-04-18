"""POST /chat — natural-language expert search with optional conversation state.

Implementation in feat/search worktree. This file contains the route skeleton.
The `?debug=true` query flag asks the search pipeline to return pipeline
internals via ChatResponse.debug.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.core.deps import (
    EmbedderDep,
    LLMDep,
    RerankerDep,
    SessionStoreDep,
    SettingsDep,
    SparseRetrieverDep,
    VectorStoreDep,
)
from app.models.api import ChatRequest, ChatResponse


router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    embedder: EmbedderDep,
    vs: VectorStoreDep,
    sparse: SparseRetrieverDep,
    llm: LLMDep,
    reranker: RerankerDep,
    sessions: SessionStoreDep,
    settings: SettingsDep,
    debug: bool = Query(default=False, description="Return pipeline internals"),
) -> ChatResponse:
    try:
        from app.services import search_pipeline  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise HTTPException(status_code=501, detail=f"chat not available: {e}")

    return search_pipeline.run_chat(
        req=req,
        debug=debug,
        embedder=embedder,
        vector_store=vs,
        sparse=sparse,
        llm=llm,
        reranker=reranker,
        sessions=sessions,
        settings=settings,
    )
