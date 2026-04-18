"""Top-level orchestration of /chat — filled in by feat/search worktree.

Contract:
    def run_chat(
        *,
        req: ChatRequest,
        debug: bool,
        embedder, vector_store, sparse, llm, reranker, sessions, settings,
    ) -> ChatResponse:
        '''Decompose → retrieve → rerank → MMR → explain → persist session.'''
"""
from __future__ import annotations

from app.models.api import ChatRequest, ChatResponse


def run_chat(
    *,
    req: ChatRequest,
    debug: bool,
    embedder,
    vector_store,
    sparse,
    llm,
    reranker,
    sessions,
    settings,
) -> ChatResponse:
    raise NotImplementedError("search_pipeline.run_chat — feat/search")
