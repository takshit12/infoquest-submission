"""Tests for the request-id, access-log, security-headers, API-key middleware
and the slowapi rate-limit wiring.

These tests construct a fresh ``FastAPI`` app per test (or rebuild settings via
``cache_clear``) so each scenario is hermetic. No real DB or LLM is required:
we hit ``/`` and ``/health``, plus a small ad-hoc protected route mounted onto
a throwaway app for the API-key tests.
"""
from __future__ import annotations

import importlib
import os
import re
from pathlib import Path

import pytest

# Load the repo's .env explicitly before any ``app.main`` import. pydantic-settings
# would normally read it when ``Settings()`` is instantiated, but ``setdefault``
# below would otherwise win and pin DATABASE_URL to a dummy value, bleeding into
# other tests in the same session (notably the ingestion tests that hit Postgres).
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

# Final fallback if .env is absent (e.g. running tests inside a fresh worktree).
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("OPENROUTER_API_KEY", "")
os.environ.pop("API_KEY", None)

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core import config as config_module  # noqa: E402
from app.core.middleware import (  # noqa: E402
    APIKeyMiddleware,
    AccessLogMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
)


_UUID_HEX_RE = re.compile(r"^[0-9a-f]{32}$")


def _reset_settings_cache(monkeypatch: pytest.MonkeyPatch, **env: str | None) -> None:
    """Reset the ``get_settings()`` singleton and apply ad-hoc env overrides."""
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    # The Settings singleton lives at module level, not as functools.lru_cache.
    config_module._settings = None


def _build_minimal_app() -> FastAPI:
    """A throwaway FastAPI app wired with our four middleware and a few routes.

    Avoids importing ``app.main`` (which pulls in routers that touch the DB at
    module load) so each middleware behavior can be verified in isolation.
    """
    app = FastAPI()

    # Innermost first → outermost last (FastAPI runs them in reverse order).
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(APIKeyMiddleware)
    app.add_middleware(RequestIDMiddleware)

    @app.get("/")
    def root():
        return {"ok": True}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/protected")
    def protected():
        return {"ok": True}

    @app.post("/protected")
    def protected_post():
        return {"ok": True}

    return app


# ----------------------------- Request ID ---------------------------------


def test_request_id_added_to_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_settings_cache(monkeypatch, API_KEY=None)
    app = _build_minimal_app()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    request_id = response.headers.get("x-request-id")
    assert request_id is not None
    assert _UUID_HEX_RE.match(request_id), f"unexpected request id: {request_id!r}"


def test_request_id_propagates_from_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_settings_cache(monkeypatch, API_KEY=None)
    app = _build_minimal_app()
    client = TestClient(app)

    response = client.get("/health", headers={"X-Request-ID": "my-trace-id"})

    assert response.status_code == 200
    assert response.headers.get("x-request-id") == "my-trace-id"


# --------------------------- Security headers -----------------------------


def test_security_headers_present(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_settings_cache(monkeypatch, API_KEY=None)
    app = _build_minimal_app()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "DENY"
    assert response.headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert "max-age=31536000" in (response.headers.get("strict-transport-security") or "")


# ------------------------------- API key ----------------------------------


def test_api_key_not_required_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_settings_cache(monkeypatch, API_KEY=None)
    app = _build_minimal_app()
    client = TestClient(app)

    response = client.get("/protected")

    assert response.status_code != 401
    assert response.status_code == 200


def test_api_key_enforced_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_settings_cache(monkeypatch, API_KEY="secret")
    app = _build_minimal_app()
    client = TestClient(app)

    # No header -> 401
    bad = client.get("/protected")
    assert bad.status_code == 401
    assert "invalid or missing API key" in bad.text

    # Wrong header -> 401
    wrong = client.get("/protected", headers={"X-API-Key": "nope"})
    assert wrong.status_code == 401

    # Correct header -> not 401 (fine to be 200/500 depending on route deps)
    good = client.get("/protected", headers={"X-API-Key": "secret"})
    assert good.status_code != 401
    assert good.status_code == 200


def test_health_exempt_from_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_settings_cache(monkeypatch, API_KEY="secret")
    app = _build_minimal_app()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200


def test_root_and_docs_exempt_from_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_settings_cache(monkeypatch, API_KEY="secret")
    app = _build_minimal_app()
    client = TestClient(app)

    assert client.get("/").status_code == 200
    # FastAPI auto-mounts /openapi.json; ensure it is reachable without a key.
    assert client.get("/openapi.json").status_code == 200


# ------------------------------ Rate limit --------------------------------


def test_rate_limit_kicks_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build a fresh app with a tiny per-minute ceiling and hit it past the limit.

    We rebuild ``app.main`` after toggling ``RATE_LIMIT_PER_MIN`` so the slowapi
    Limiter picks up the override.
    """
    _reset_settings_cache(
        monkeypatch,
        API_KEY=None,
        RATE_LIMIT_PER_MIN="2",
        # avoid pulling network deps via the lifespan handler at startup
        OPENROUTER_API_KEY="",
    )

    # Fresh import so the module-level Limiter sees the new setting.
    if "app.main" in list(globals()):
        pass
    main_module = importlib.import_module("app.main")
    main_module = importlib.reload(main_module)

    # Use a TestClient without firing the lifespan (don't need DB pings).
    client = TestClient(main_module.app, raise_server_exceptions=False)

    statuses = [client.get("/").status_code for _ in range(5)]
    assert any(code == 429 for code in statuses), f"expected a 429, got {statuses}"
