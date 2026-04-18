"""Orchestrates the full ingest pipeline.

Filled in by the feat/ingest worktree. Contract:

    def run_ingest(
        *,
        embedder: Embedder,
        vector_store: VectorStore,
        sparse: SparseRetriever,
        reset: bool = False,
        limit: int | None = None,
    ) -> IngestResponse:
        '''DB → RoleRecord[] → embed batches → vector_store.upsert → sparse.build.
        Returns counts + elapsed time.'''
"""
from __future__ import annotations

from app.models.api import IngestResponse
from app.ports.embedder import Embedder
from app.ports.sparse_retriever import SparseRetriever
from app.ports.vector_store import VectorStore


def run_ingest(
    *,
    embedder: Embedder,
    vector_store: VectorStore,
    sparse: SparseRetriever,
    reset: bool = False,
    limit: int | None = None,
) -> IngestResponse:
    raise NotImplementedError(
        "ingestion.run_ingest — implemented in feat/ingest worktree"
    )
