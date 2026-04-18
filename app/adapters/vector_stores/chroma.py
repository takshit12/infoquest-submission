"""Chroma adapter implementing the VectorStore Protocol.

Pitfalls worth knowing:
  - `hnsw:space=cosine` is set at collection creation (IMMUTABLE after).
  - Uses `upsert()` not `add()` — add() duplicates in older Chroma versions.
  - Chroma returns `distance` (lower=better for cosine: 1 - cos_sim); we
    convert to `score` = 1 - distance so higher=better throughout the stack.
  - Chroma metadata filters use `where={"field": value}` for equality, and
    `$and`/`$or`/`$in` operators for composites. Not all Chroma versions
    support `$in` with strings cleanly — we build filter dicts defensively.
"""
from __future__ import annotations

import os
from typing import Any


class ChromaVectorStore:
    """Implements the VectorStore Protocol."""

    def __init__(
        self,
        persist_directory: str = "./chroma_data",
        collection_name: str = "roles_v1",
    ) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        os.makedirs(persist_directory, exist_ok=True)
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ---- VectorStore Protocol ----

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None:
        if not ids:
            return
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=[self._coerce_metadata(m) for m in metadatas],
            documents=documents,
        )

    def search(
        self,
        query_embedding: list[float],
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0] or [{}] * len(ids)
        out: list[tuple[str, float, dict[str, Any]]] = []
        for id_, dist, meta in zip(ids, distances, metadatas):
            # cosine distance in [0, 2] → score in [-1, 1]; we clip to [0, 1]
            score = max(0.0, min(1.0, 1.0 - float(dist)))
            out.append((id_, score, meta or {}))
        return out

    def get(self, ids: list[str]) -> list[dict[str, Any] | None]:
        if not ids:
            return []
        result = self._collection.get(ids=ids)
        got_ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        documents = result.get("documents") or []
        index = {i: (m, d) for i, m, d in zip(got_ids, metadatas, documents)}
        out: list[dict[str, Any] | None] = []
        for id_ in ids:
            if id_ in index:
                meta, doc = index[id_]
                out.append({"id": id_, "metadata": meta or {}, "document": doc})
            else:
                out.append(None)
        return out

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ping(self) -> bool:
        try:
            self._client.heartbeat()
            return True
        except Exception:
            return False

    # ---- helpers ----

    @staticmethod
    def _coerce_metadata(meta: dict[str, Any]) -> dict[str, Any]:
        """Chroma metadata values must be str/int/float/bool. Serialize lists."""
        out: dict[str, Any] = {}
        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                out[k] = v
            elif isinstance(v, (list, tuple, set)):
                # join lists as CSV so we can later split in the reranker
                out[k] = ",".join(str(x) for x in v)
            else:
                out[k] = str(v)
        return out
