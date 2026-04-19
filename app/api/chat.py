"""POST /chat — natural-language expert search with optional conversation state.

Implementation in feat/search worktree. This file contains the route skeleton.
The `?debug=true` query flag asks the search pipeline to return pipeline
internals via ChatResponse.debug.

A new conversation (no ``conversation_id`` in the body) is created server-side;
the response carries a one-time ``session_token`` the caller must echo as
``X-Session-Token`` on follow-up turns. Without that header (or with the wrong
token) follow-ups are rejected with 401 / 403.
"""
from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Query

from app.core.deps import (
    EmbedderDep,
    LLMDep,
    RerankerDep,
    SessionStoreDep,
    SettingsDep,
    SparseRetrieverDep,
    VectorStoreDep,
)
from app.core.logging import get_logger
from app.models.api import ChatRequest, ChatResponse


_log = get_logger("infoquest.api.chat")
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
    x_session_token: str | None = Header(default=None, alias="X-Session-Token"),
) -> ChatResponse:
    # If the caller is continuing an existing conversation they MUST present
    # the bearer token issued on first turn. Brand-new conversations (no
    # conversation_id in the body) skip this check — they receive a token in
    # the response.
    if req.conversation_id and sessions.exists(req.conversation_id):
        if not x_session_token:
            raise HTTPException(status_code=401, detail="missing X-Session-Token")
        if not sessions.verify_token(req.conversation_id, x_session_token):
            raise HTTPException(status_code=403, detail="invalid session token")

    try:
        from app.services import search_pipeline  # noqa: F401
    except ImportError as e:  # pragma: no cover
        _log.error("chat_service_unavailable", error=str(e))
        raise HTTPException(status_code=503, detail="search service unavailable")

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
