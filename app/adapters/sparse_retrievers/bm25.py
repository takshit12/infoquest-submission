"""BM25 retriever using rank_bm25 (pure Python)."""
from __future__ import annotations

import pickle
import re
from pathlib import Path

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]{1,}")


def _tokenize(text: str) -> list[str]:
    """Simple lower-case word tokenizer. Good enough for BM25 on short prose."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class BM25Retriever:
    """Implements the SparseRetriever Protocol.

    Index is held in-process and pickled to disk on build(). Loaded lazily.
    """

    def __init__(self, index_path: str = "./bm25_index.pkl") -> None:
        self.index_path = index_path
        self._ids: list[str] = []
        self._bm25 = None  # rank_bm25.BM25Okapi, lazy

    def build(self, ids: list[str], documents: list[str]) -> None:
        from rank_bm25 import BM25Okapi

        tokenized = [_tokenize(d) for d in documents]
        self._ids = list(ids)
        self._bm25 = BM25Okapi(tokenized)
        self._persist()

    def load(self) -> None:
        path = Path(self.index_path)
        if not path.exists():
            return
        with path.open("rb") as f:
            payload = pickle.load(f)  # nosec B301 - operator-controlled BM25_INDEX_PATH, built locally
        self._ids = payload["ids"]
        self._bm25 = payload["bm25"]

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            self.load()
        if self._bm25 is None or not self._ids:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # pick top-k
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:k]
        return [(self._ids[i], float(s)) for i, s in ranked if s > 0]

    def count(self) -> int:
        return len(self._ids)

    # ---- internals ----

    def _persist(self) -> None:
        path = Path(self.index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"ids": self._ids, "bm25": self._bm25}, f)
