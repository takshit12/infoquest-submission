"""BGE embedder — wraps sentence-transformers with a unified query/doc API.

bge-small-en-v1.5 uses a recommended prefix for queries:
    "Represent this sentence for searching relevant passages: {query}"
Documents are encoded without a prefix. Embeddings are L2-normalized so that
dot product == cosine similarity; pair this with `hnsw:space=cosine` in Chroma.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class BGEEmbedder:
    """Implements the Embedder Protocol via sentence-transformers."""

    # filled in __init__
    dimension: int = 384

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 64,
        device: str = "cpu",
    ) -> None:
        # Lazy import keeps uvicorn start fast when the embedder isn't needed
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model: SentenceTransformer = SentenceTransformer(
            model_name, device=device
        )
        self.dimension = self._model.get_sentence_embedding_dimension() or 384

    _QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        prefixed = self._QUERY_PREFIX + text
        vector = self._model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return vector.tolist()
