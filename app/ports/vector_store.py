"""Vector store port: dense retrieval over embedded docs with metadata filters."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VectorStore(Protocol):
    """Contract for dense vector stores (Chroma, pgvector, Qdrant, ...)."""

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None:
        """Idempotent insert-or-update."""

    def search(
        self,
        query_embedding: list[float],
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Return (id, score, metadata) tuples. Score should be higher=better
        after whatever normalization the adapter applies (e.g., 1 - cosine_distance).
        """

    def get(self, ids: list[str]) -> list[dict[str, Any] | None]:
        """Fetch documents by id."""

    def count(self) -> int:
        """Doc count in the collection."""

    def reset(self) -> None:
        """Drop + recreate the collection (used by /ingest?reset=true)."""

    def ping(self) -> bool:
        """Cheap liveness probe for /health."""
