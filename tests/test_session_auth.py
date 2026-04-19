"""Tests for the per-conversation session-token IDOR fix.

Three covered surfaces:
  - POST /chat returns ``session_token`` on the FIRST turn of a NEW conversation.
  - POST /chat with a ``conversation_id`` in the body REQUIRES X-Session-Token
    (401 missing, 403 wrong, 200 correct, no fresh token in 200 response).
  - GET  /conversations/{id} REQUIRES X-Session-Token (same 401/403/200 matrix).

The chat route has a fat dep graph (embedder / vector store / sparse / LLM /
reranker / sessions / settings); we override each via ``app.dependency_overrides``
so no real network or DB I/O happens.
"""
from __future__ import annotations

import os
from pathlib import Path

# Match the .env loading dance from tests/test_middleware.py — pydantic-settings
# would otherwise blow up on the required DATABASE_URL when ``Settings()`` is
# imported transitively below.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("OPENROUTER_API_KEY", "")
os.environ.pop("API_KEY", None)

from datetime import datetime, timezone  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.api import chat as chat_router  # noqa: E402
from app.api import conversations as conversations_router  # noqa: E402
from app.core import deps as deps_module  # noqa: E402
from app.core.config import Settings, get_settings  # noqa: E402
from app.models.domain import QueryIntent  # noqa: E402
from tests.conftest import (  # noqa: E402
    FakeEmbedder,
    FakeLLM,
    FakeSessionStore,
    FakeSparseRetriever,
    FakeVectorStore,
)


# ============================================================
#                       Fixtures helpers
# ============================================================


def _make_canned_decompose_json() -> dict:
    return {
        "rewritten_search": "regulatory affairs pharma Middle East",
        "keywords": ["regulatory", "pharma"],
        "geographies": ["AE"],
        "require_current": None,
        "min_yoe": None,
        "exclude_candidate_ids": [],
        "function": "regulatory affairs",
        "industries": ["Pharmaceuticals"],
        "seniority_band": None,
        "skill_categories": [],
    }


def _build_test_app(sessions: FakeSessionStore) -> FastAPI:
    """Build a tiny FastAPI app with /chat + /conversations/{id} and FAKE deps.

    No DB / LLM / Chroma is touched: every dep returns a FakeXxx adapter from
    tests/conftest.py. The same FakeSessionStore instance is shared across the
    chat + conversation routers so the token issued on first /chat is
    verifiable on subsequent calls.
    """
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()
    sparse = FakeSparseRetriever()
    llm = FakeLLM(canned_json=_make_canned_decompose_json(), canned_text="ok match")

    # Provide a real reranker so explainer/diversity don't need stubbing — it's
    # the same module the live app uses, just fed our fake corpus.
    from app.services.reranker import WeightedSignalReranker
    from app.services.signals import SIGNALS

    reranker = WeightedSignalReranker(
        weights=Settings().weights.as_dict(),
        signals=SIGNALS,
        maxp_bonus=0.05,
        maxp_cap=0.15,
    )

    app = FastAPI()
    app.include_router(chat_router.router)
    app.include_router(conversations_router.router)

    app.dependency_overrides[deps_module.get_embedder] = lambda: embedder
    app.dependency_overrides[deps_module.get_vector_store] = lambda: vector_store
    app.dependency_overrides[deps_module.get_sparse_retriever] = lambda: sparse
    app.dependency_overrides[deps_module.get_llm] = lambda: llm
    app.dependency_overrides[deps_module.get_reranker] = lambda: reranker
    app.dependency_overrides[deps_module.get_session_store] = lambda: sessions
    app.dependency_overrides[get_settings] = lambda: Settings()

    return app


# ============================================================
#                         /chat tests
# ============================================================


def test_first_chat_returns_session_token() -> None:
    sessions = FakeSessionStore()
    app = _build_test_app(sessions)
    client = TestClient(app)

    resp = client.post(
        "/chat",
        json={"query": "find me regulatory affairs experts in pharma", "include_why_not": False},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["conversation_id"]
    assert body.get("session_token"), "first turn must include a session_token"
    # token persisted in the fake store
    assert sessions.verify_token(body["conversation_id"], body["session_token"])


def test_followup_chat_requires_token() -> None:
    sessions = FakeSessionStore()
    app = _build_test_app(sessions)
    client = TestClient(app)

    first = client.post(
        "/chat",
        json={"query": "find me regulatory affairs experts in pharma", "include_why_not": False},
    )
    assert first.status_code == 200, first.text
    cid = first.json()["conversation_id"]

    # Follow-up with NO X-Session-Token header → 401
    follow = client.post(
        "/chat",
        json={
            "query": "narrow to UAE",
            "conversation_id": cid,
            "include_why_not": False,
        },
    )
    assert follow.status_code == 401
    assert "missing X-Session-Token" in follow.text


def test_followup_chat_rejects_wrong_token() -> None:
    sessions = FakeSessionStore()
    app = _build_test_app(sessions)
    client = TestClient(app)

    first = client.post(
        "/chat",
        json={"query": "find me regulatory affairs experts in pharma", "include_why_not": False},
    )
    assert first.status_code == 200, first.text
    cid = first.json()["conversation_id"]

    # Bogus token → 403
    follow = client.post(
        "/chat",
        json={
            "query": "narrow to UAE",
            "conversation_id": cid,
            "include_why_not": False,
        },
        headers={"X-Session-Token": "tok-not-a-real-token"},
    )
    assert follow.status_code == 403
    assert "invalid session token" in follow.text


def test_followup_chat_accepts_correct_token() -> None:
    sessions = FakeSessionStore()
    app = _build_test_app(sessions)
    client = TestClient(app)

    first = client.post(
        "/chat",
        json={"query": "find me regulatory affairs experts in pharma", "include_why_not": False},
    )
    assert first.status_code == 200, first.text
    body = first.json()
    cid, token = body["conversation_id"], body["session_token"]

    follow = client.post(
        "/chat",
        json={
            "query": "narrow to UAE",
            "conversation_id": cid,
            "include_why_not": False,
        },
        headers={"X-Session-Token": token},
    )
    assert follow.status_code == 200, follow.text
    follow_body = follow.json()
    # follow-up keeps the same conversation
    assert follow_body["conversation_id"] == cid
    # follow-ups MUST NOT echo the token again — keeps the secret out of
    # repeated request/response payloads.
    assert follow_body.get("session_token") in (None, "")


# ============================================================
#                /conversations/{id} tests
# ============================================================


def _seed_conversation(sessions: FakeSessionStore) -> tuple[str, str]:
    """Create a conversation directly + record one turn so /conversations
    has something interesting to return."""
    cid, token = sessions.create()
    sessions.append_turn(
        cid,
        query="find regulatory affairs experts",
        intent=QueryIntent(raw_query="find regulatory affairs experts"),
        top_candidate_ids=["c1", "c2"],
    )
    return cid, token


def test_get_conversation_requires_token() -> None:
    sessions = FakeSessionStore()
    cid, token = _seed_conversation(sessions)
    app = _build_test_app(sessions)
    client = TestClient(app)

    # Missing header → 401
    no_header = client.get(f"/conversations/{cid}")
    assert no_header.status_code == 401
    assert "missing X-Session-Token" in no_header.text

    # Wrong token → 403
    wrong = client.get(
        f"/conversations/{cid}", headers={"X-Session-Token": "tok-wrong"}
    )
    assert wrong.status_code == 403
    assert "invalid session token" in wrong.text

    # Correct token → 200 with history
    good = client.get(
        f"/conversations/{cid}", headers={"X-Session-Token": token}
    )
    assert good.status_code == 200
    body = good.json()
    assert body["conversation_id"] == cid
    assert len(body["turns"]) == 1
    assert body["turns"][0]["query"] == "find regulatory affairs experts"


def test_get_unknown_conversation_returns_404_before_token_check() -> None:
    """An unknown conversation id is a 404; we don't leak whether it once
    existed by switching to a 401 only if the row is found."""
    sessions = FakeSessionStore()
    app = _build_test_app(sessions)
    client = TestClient(app)

    # Even with a random token, unknown id is a 404
    resp = client.get(
        "/conversations/00000000-0000-0000-0000-000000000000",
        headers={"X-Session-Token": "tok-anything"},
    )
    assert resp.status_code == 404
