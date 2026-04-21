"""Tests for the admin signal-weights endpoints."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("OPENROUTER_API_KEY", "")
os.environ.setdefault("ENABLE_DYNAMIC_WEIGHTS", "true")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.api import admin  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from tests.conftest import FakeWeightsRepo  # noqa: E402


def _build_admin_app(weights_repo: FakeWeightsRepo) -> FastAPI:
    """Build a test app with the admin router."""
    app = FastAPI()
    app.include_router(admin.router)

    # Monkey-patch app.db module
    import app.db
    app.db.fetch_signal_weights = weights_repo.fetch_signal_weights
    app.db.update_signal_weights = lambda **kw: weights_repo.update_signal_weights(**{k: v for k, v in kw.items() if k in weights_repo.weights})

    return app


def test_get_signal_weights(fake_weights_repo):
    """GET /admin/signal-weights returns current weights."""
    app = _build_admin_app(fake_weights_repo)
    client = TestClient(app)

    resp = client.get("/admin/signal-weights")
    assert resp.status_code == 200
    body = resp.json()
    assert body["industry"] == 0.25
    assert abs(sum(body.values()) - 1.0) < 0.01


def test_update_signal_weights_success(fake_weights_repo):
    """POST /admin/signal-weights updates weights."""
    app = _build_admin_app(fake_weights_repo)
    client = TestClient(app)

    new_weights = {
        "industry": 0.30,
        "function": 0.25,
        "seniority": 0.20,
        "skill_category": 0.10,
        "recency": 0.05,
        "dense": 0.05,
        "bm25": 0.03,
        "trajectory": 0.02,
    }
    resp = client.post("/admin/signal-weights", json=new_weights)
    assert resp.status_code == 200
    body = resp.json()
    assert body["industry"] == 0.30


def test_update_signal_weights_invalid_sum(fake_weights_repo):
    """POST /admin/signal-weights rejects weights that don't sum to ~1.0."""
    app = _build_admin_app(fake_weights_repo)
    client = TestClient(app)

    bad_weights = {
        "industry": 0.50,
        "function": 0.50,
        "seniority": 0.50,
        "skill_category": 0.50,
        "recency": 0.50,
        "dense": 0.50,
        "bm25": 0.50,
        "trajectory": 0.50,
    }
    resp = client.post("/admin/signal-weights", json=bad_weights)
    assert resp.status_code == 400
    assert "sum to ~1.0" in resp.text
