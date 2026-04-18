"""Embedder port: any encoder that maps text → float vectors."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Contract for text embedding back-ends."""

    dimension: int  # fixed output dimension (e.g., 384 for bge-small)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of document texts. Returns normalized vectors."""

    def embed_query(self, text: str) -> list[float]:
        """Encode a single query. May use a different prompt prefix than documents."""
