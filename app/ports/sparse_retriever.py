"""Sparse retriever port: BM25 / lexical retrieval over same doc corpus."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SparseRetriever(Protocol):
    """Contract for lexical retrievers (BM25, TF-IDF, etc.)."""

    def build(self, ids: list[str], documents: list[str]) -> None:
        """Build the index from scratch and persist it."""

    def load(self) -> None:
        """Load a persisted index. Called on server startup."""

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return (doc_id, score) tuples, top-k by BM25 score."""

    def count(self) -> int:
        ...
