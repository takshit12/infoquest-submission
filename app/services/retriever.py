"""Dense + sparse retrieval, RRF fusion, MaxP candidate aggregation.

Filled in by the feat/search worktree.

Contract:
    def retrieve(
        intent: QueryIntent,
        *,
        embedder: Embedder,
        vector_store: VectorStore,
        sparse: SparseRetriever,
        settings: Settings,
    ) -> list[ScoredRole]:
        '''Compose hard-filter `where` dict from intent (geography, is_current,
        min_yoe). Dense top-K + sparse top-K. RRF fuse on role_id. Return ScoredRole
        list with dense_score / sparse_score / fused_score populated.'''

    def rrf_fuse(
        dense: list[tuple[id, score]],
        sparse: list[tuple[id, score]],
        k: int = 60,
    ) -> list[tuple[id, fused_score]]:
        '''Reciprocal Rank Fusion: 1 / (k + rank_dense) + 1 / (k + rank_sparse).'''
"""
from __future__ import annotations

from app.core.config import Settings
from app.models.domain import QueryIntent, ScoredRole
from app.ports.embedder import Embedder
from app.ports.sparse_retriever import SparseRetriever
from app.ports.vector_store import VectorStore


def retrieve(
    intent: QueryIntent,
    *,
    embedder: Embedder,
    vector_store: VectorStore,
    sparse: SparseRetriever,
    settings: Settings,
) -> list[ScoredRole]:
    raise NotImplementedError("retriever.retrieve — feat/search")


def rrf_fuse(
    dense: list[tuple[str, float]],
    sparse: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    raise NotImplementedError("retriever.rrf_fuse — feat/search")
