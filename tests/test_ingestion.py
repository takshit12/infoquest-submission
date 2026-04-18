"""Tests for the ingestion pipeline.

Unit tests run without any external I/O (no Postgres, no Chroma, no model
weights). Integration tests hit the real stack — mark them with
`@pytest.mark.integration` so they can be skipped via `-m 'not integration'`.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from app.models.api import IngestResponse
from app.models.domain import RoleRecord
from app.services.ingestion import _METADATA_KEYS, _role_to_metadata, run_ingest


# ============================================================
#                        Fixtures
# ============================================================


def _sample_role(**overrides) -> RoleRecord:
    """Construct a minimal but schema-complete RoleRecord for unit tests."""
    base = dict(
        role_id="11111111-1111-1111-1111-111111111111",
        candidate_id="22222222-2222-2222-2222-222222222222",
        job_title="Head of Product",
        company="Acme Corp",
        industry="Financial Services",
        seniority_tier="vp",
        start_date=date(2020, 1, 1),
        end_date=date(2023, 6, 1),
        is_current=False,
        description="Owned product strategy across a 40-person org.",
        role_years=3.4,
        candidate_headline="Product leader",
        candidate_yoe=12,
        candidate_country="SA",
        candidate_city="Riyadh",
        candidate_nationality="SA",
        skill_categories=["Product", "Strategy"],
        languages=["en", "ar"],
    )
    base.update(overrides)
    return RoleRecord(**base)


# ============================================================
#                       Unit tests
# ============================================================


def test_role_metadata_shape():
    """`_role_to_metadata` must emit exactly the keys the search code reads."""
    role = _sample_role()
    meta = _role_to_metadata(role)

    # Exact key set — adding/removing keys is a breaking change for feat/search.
    assert set(meta.keys()) == set(_METADATA_KEYS)

    # Scalar types match what Chroma accepts + what downstream expects.
    assert meta["candidate_id"] == role.candidate_id
    assert meta["role_id"] == role.role_id
    assert meta["job_title"] == "Head of Product"
    assert meta["seniority_tier"] == "vp"
    assert meta["company"] == "Acme Corp"
    assert meta["industry"] == "Financial Services"
    assert meta["country"] == "SA"
    assert meta["city"] == "Riyadh"
    assert meta["nationality"] == "SA"
    assert meta["is_current"] is False
    assert meta["start_date"] == "2020-01-01"
    assert meta["end_date"] == "2023-06-01"
    assert isinstance(meta["role_years"], float)
    assert isinstance(meta["candidate_yoe"], int)
    assert meta["candidate_headline"] == "Product leader"
    assert meta["skill_categories"] == ["Product", "Strategy"]
    assert meta["languages"] == ["en", "ar"]


def test_role_metadata_handles_nullable_fields():
    """Nullable strings become empty strings; an empty end_date (current role) too."""
    role = _sample_role(
        industry=None,
        end_date=None,
        is_current=True,
        candidate_city=None,
        candidate_nationality=None,
    )
    meta = _role_to_metadata(role)
    assert meta["industry"] == ""
    assert meta["end_date"] == ""
    assert meta["city"] == ""
    assert meta["nationality"] == ""
    assert meta["is_current"] is True


def test_run_ingest_with_fakes(fake_embedder, fake_vector_store, fake_sparse, monkeypatch):
    """Smoke-test the orchestration with in-memory fakes and a patched role iterator."""
    roles = [
        _sample_role(role_id=f"role-{i}", candidate_id=f"cand-{i % 2}")
        for i in range(5)
    ]

    monkeypatch.setattr(
        "app.services.ingestion.iter_role_records",
        lambda limit=None: iter(roles if limit is None else roles[:limit]),
    )

    response = run_ingest(
        embedder=fake_embedder,
        vector_store=fake_vector_store,
        sparse=fake_sparse,
        reset=True,
        limit=None,
    )

    assert isinstance(response, IngestResponse)
    assert response.roles == 5
    assert response.candidates == 2  # cand-0, cand-1
    assert response.dense_docs == 5
    assert response.sparse_docs == 5
    assert response.elapsed_seconds >= 0

    # Round-trip one metadata dict.
    rows = fake_vector_store.get(["role-0"])
    assert rows and rows[0] is not None
    meta = rows[0]["metadata"]
    assert set(meta.keys()) == set(_METADATA_KEYS)


# ============================================================
#                    Integration tests
# ============================================================


@pytest.mark.integration
def test_iter_role_records_limit():
    """Real Postgres: `limit=5` yields exactly 5 RoleRecords."""
    from app.services.profile_builder import iter_role_records

    roles = list(iter_role_records(limit=5))
    assert len(roles) == 5
    for r in roles:
        assert r.role_id
        assert r.candidate_id
        assert r.job_title


@pytest.mark.integration
def test_run_ingest_smoke(tmp_path: Path):
    """End-to-end ingest of 10 roles against real Postgres + BGE + a tmp Chroma dir."""
    from app.adapters.embedders.bge import BGEEmbedder
    from app.adapters.sparse_retrievers.bm25 import BM25Retriever
    from app.adapters.vector_stores.chroma import ChromaVectorStore

    vector_store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_roles_v1",
    )
    sparse = BM25Retriever(index_path=str(tmp_path / "bm25.pkl"))
    embedder = BGEEmbedder(
        model_name="BAAI/bge-small-en-v1.5", batch_size=8, device="cpu"
    )

    response = run_ingest(
        embedder=embedder,
        vector_store=vector_store,
        sparse=sparse,
        reset=True,
        limit=10,
    )

    assert response.roles == 10
    assert response.dense_docs == 10
    assert response.sparse_docs == 10
    assert vector_store.count() == 10

    # Round-trip a metadata entry and confirm all required keys are present.
    any_role_id = None
    # We don't know the role IDs; ask Chroma for one.
    result = vector_store._collection.peek(limit=1)  # type: ignore[attr-defined]
    assert result and result.get("ids")
    any_role_id = result["ids"][0]

    rows = vector_store.get([any_role_id])
    assert rows and rows[0] is not None
    meta = rows[0]["metadata"]
    for key in _METADATA_KEYS:
        assert key in meta, f"missing metadata key: {key}"
