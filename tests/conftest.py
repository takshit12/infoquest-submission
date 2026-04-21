"""Shared fixtures.

Each worktree agent adds more fixtures as needed. This scaffold provides:
  - a FastAPI TestClient fixture (imports the app lazily)
  - no-network stubs for Embedder / VectorStore / SparseRetriever / LLMClient / SessionStore
    so unit tests of services can run without Postgres/LLM/Chroma.
"""
from __future__ import annotations

from typing import Any

import pytest


# ============================================================
#                Stub adapters (no external I/O)
# ============================================================


def _match_where(metadata: dict, where: dict) -> bool:
    """Minimal evaluator for a Chroma-style `where` dict used by the retriever."""
    if not where:
        return True
    for key, clause in where.items():
        if key == "$and":
            if not all(_match_where(metadata, sub) for sub in clause):
                return False
            continue
        if key == "$or":
            if not any(_match_where(metadata, sub) for sub in clause):
                return False
            continue
        value = metadata.get(key)
        if isinstance(clause, dict):
            for op, operand in clause.items():
                if op == "$in":
                    if value not in set(operand):
                        return False
                elif op == "$eq":
                    if value != operand:
                        return False
                elif op == "$ne":
                    if value == operand:
                        return False
                elif op == "$gte":
                    try:
                        if not (float(value or 0) >= float(operand)):
                            return False
                    except (TypeError, ValueError):
                        return False
                elif op == "$gt":
                    try:
                        if not (float(value or 0) > float(operand)):
                            return False
                    except (TypeError, ValueError):
                        return False
                elif op == "$lte":
                    try:
                        if not (float(value or 0) <= float(operand)):
                            return False
                    except (TypeError, ValueError):
                        return False
                elif op == "$lt":
                    try:
                        if not (float(value or 0) < float(operand)):
                            return False
                    except (TypeError, ValueError):
                        return False
                else:
                    return False
        else:
            if value != clause:
                return False
    return True


class FakeEmbedder:
    dimension = 4

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Deterministic pseudo-embeddings for reproducible tests
        return [[float(len(t) % 7), 1.0, 0.0, 0.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 7), 1.0, 0.0, 0.0]


class FakeVectorStore:
    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}

    def upsert(self, ids, embeddings, metadatas, documents) -> None:
        for i, e, m, d in zip(ids, embeddings, metadatas, documents):
            self._rows[i] = {"embedding": e, "metadata": m, "document": d}

    def search(self, query_embedding, k, where=None):
        # match + rank by first-dim closeness; supports a tiny subset of the
        # Chroma-style operator dialect that our retriever actually emits:
        # equality, {"$in": [...]}, {"$gte": n}, and a top-level {"$and": [...]}.
        items = list(self._rows.items())
        if where:
            items = [(i, r) for i, r in items if _match_where(r["metadata"], where)]

        def score(r):
            return 1.0 - abs(r["embedding"][0] - query_embedding[0]) * 0.1

        items.sort(key=lambda kv: score(kv[1]), reverse=True)
        return [(i, score(r), r["metadata"]) for i, r in items[:k]]

    def get(self, ids):
        return [
            {"id": i, "metadata": self._rows[i]["metadata"], "document": self._rows[i]["document"]}
            if i in self._rows
            else None
            for i in ids
        ]

    def count(self):
        return len(self._rows)

    def reset(self):
        self._rows.clear()

    def ping(self):
        return True


class FakeSparseRetriever:
    def __init__(self):
        self._rows: dict[str, str] = {}

    def build(self, ids, documents):
        self._rows = dict(zip(ids, documents))

    def load(self):
        pass

    def search(self, query, k):
        q = query.lower()
        scored = []
        for i, d in self._rows.items():
            score = sum(1 for t in q.split() if t in d.lower())
            if score:
                scored.append((i, float(score)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def count(self):
        return len(self._rows)


class FakeLLM:
    model = "fake-model"

    def __init__(self, canned_json: dict | None = None, canned_text: str = "ok"):
        self.canned_json = canned_json or {}
        self.canned_text = canned_text

    def chat(self, *, system, user, temperature=0.1, max_tokens=1024, response_format=None):
        return self.canned_text

    def chat_json(self, *, system, user, schema_hint=None, temperature=0.1, max_tokens=1024):
        return dict(self.canned_json)

    def ping(self):
        return True


class FakeSessionStore:
    def __init__(self):
        self._conv: dict[str, list[dict]] = {}
        self._tokens: dict[str, str] = {}

    def create(self):
        import secrets
        import uuid

        cid = str(uuid.uuid4())
        token = f"tok-{secrets.token_urlsafe(16)}"
        self._conv[cid] = []
        self._tokens[cid] = token
        return cid, token

    def exists(self, cid):
        return cid in self._conv

    def verify_token(self, cid, token):
        import secrets

        if not cid or not token:
            return False
        stored = self._tokens.get(cid) or ""
        if not stored:
            return False
        return secrets.compare_digest(stored, token)

    def append_turn(self, cid, query, intent, top_candidate_ids):
        from datetime import datetime, timezone

        self._conv.setdefault(cid, [])
        idx = len(self._conv[cid])
        self._conv[cid].append(
            {
                "turn_index": idx,
                "query": query,
                "intent": intent,
                "top_candidate_ids": top_candidate_ids,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        return idx

    def last_intent(self, cid):
        turns = self._conv.get(cid) or []
        return turns[-1]["intent"] if turns else None

    def last_candidate_ids(self, cid):
        turns = self._conv.get(cid) or []
        return turns[-1]["top_candidate_ids"] if turns else []

    def history(self, cid):
        # Mirror the SQLite adapter's output shape so the conversations
        # endpoint can render this without a separate code path.
        out = []
        for t in self._conv.get(cid, []):
            intent = t["intent"]
            try:
                intent_dict = (
                    intent.model_dump() if hasattr(intent, "model_dump") else dict(intent)
                )
            except Exception:
                intent_dict = {}
            out.append(
                {
                    "turn_index": t["turn_index"],
                    "query": t["query"],
                    "query_intent": intent_dict,
                    "top_candidate_ids": list(t["top_candidate_ids"]),
                    "timestamp": t.get(
                        "timestamp", __import__("datetime").datetime.now().isoformat()
                    ),
                }
            )
        return out

    def meta(self, cid):
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        return (now, now) if cid in self._conv else None

    def ping(self):
        return True



# ============================================================
#                        Fixtures
# ============================================================


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_vector_store():
    return FakeVectorStore()


@pytest.fixture
def fake_sparse():
    return FakeSparseRetriever()


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def fake_sessions():
    return FakeSessionStore()
