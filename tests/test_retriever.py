"""Tests for rrf_fuse determinism and retrieve() over fake adapters."""
from __future__ import annotations

from app.core.config import Settings
from app.models.domain import QueryIntent
from app.services import retriever


def test_rrf_fuse_basic_order():
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    sparse = [("b", 0.95), ("d", 0.6), ("a", 0.3)]
    fused = retriever.rrf_fuse(dense, sparse, k=60)
    ids = [id_ for id_, _ in fused]
    # b appears at ranks 2 and 1 → strongest combined
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_fuse_empty():
    assert retriever.rrf_fuse([], [], k=60) == []


def test_rrf_fuse_deterministic():
    d = [("x", 0.1), ("y", 0.1)]
    s = [("y", 0.1), ("x", 0.1)]
    assert retriever.rrf_fuse(d, s) == retriever.rrf_fuse(d, s)


def _seed_store(vs, sparse):
    """Populate FakeVectorStore + FakeSparseRetriever with a handful of roles."""
    rows = [
        {
            "role_id": "r1",
            "candidate_id": "c1",
            "job_title": "Regulatory Affairs VP",
            "company": "Pfizer MENA",
            "industry": "Pharmaceuticals",
            "seniority_tier": "vp",
            "country": "AE",
            "city": "Dubai",
            "is_current": True,
            "start_date": "2020-01-01",
            "end_date": "",
            "role_years": 5.0,
            "candidate_yoe": 18,
            "candidate_headline": "VP Regulatory Affairs",
            "skill_categories": "Regulatory,Clinical",
            "languages": "en,ar",
            "description": "regulatory submissions FDA EMA",
        },
        {
            "role_id": "r2",
            "candidate_id": "c2",
            "job_title": "Marketing Director",
            "company": "Unilever",
            "industry": "Consumer Goods",
            "seniority_tier": "director",
            "country": "SA",
            "city": "Riyadh",
            "is_current": True,
            "start_date": "2019-01-01",
            "end_date": "",
            "role_years": 6.0,
            "candidate_yoe": 14,
            "candidate_headline": "Marketing Director",
            "skill_categories": "Marketing,Brand",
            "languages": "en,ar",
            "description": "go to market consumer brands",
        },
        {
            "role_id": "r3",
            "candidate_id": "c3",
            "job_title": "Junior Data Engineer",
            "company": "Careem",
            "industry": "Technology, Information and Internet",
            "seniority_tier": "junior",
            "country": "AE",
            "city": "Dubai",
            "is_current": True,
            "start_date": "2023-03-01",
            "end_date": "",
            "role_years": 1.5,
            "candidate_yoe": 2,
            "candidate_headline": "Jr. data engineer",
            "skill_categories": "Data,Engineering",
            "languages": "en",
            "description": "ETL pipelines Python Airflow",
        },
    ]
    ids = []
    docs = []
    for r in rows:
        rid = r["role_id"]
        ids.append(rid)
        docs.append(
            f"{r['job_title']} at {r['company']} {r['industry']} "
            f"{r['description']} {r['candidate_headline']}"
        )
        # the FakeVectorStore uses first-dim closeness — embed by job-title length
        vs.upsert(
            ids=[rid],
            embeddings=[[float(len(r["job_title"]) % 7), 1.0, 0.0, 0.0]],
            metadatas=[r],
            documents=[docs[-1]],
        )
    sparse.build(ids, docs)


def test_retrieve_returns_scored_roles(fake_embedder, fake_vector_store, fake_sparse):
    _seed_store(fake_vector_store, fake_sparse)
    settings = Settings()
    intent = QueryIntent(
        raw_query="regulatory affairs in pharma UAE",
        rewritten_search="regulatory affairs pharma",
        keywords=["regulatory", "affairs", "pharma"],
        geographies=["AE"],
        industries=["Pharmaceuticals"],
    )
    scored = retriever.retrieve(
        intent,
        embedder=fake_embedder,
        vector_store=fake_vector_store,
        sparse=fake_sparse,
        settings=settings,
    )
    # All returned roles must satisfy hard filter (country AE)
    assert scored, "expected non-empty scored roles"
    for sr in scored:
        assert sr.role.candidate_country == "AE"
    # The top-ranked role should match the query context best — r1 is the
    # regulatory affairs role in AE.
    assert scored[0].role.role_id == "r1"


def test_retrieve_applies_require_current(fake_embedder, fake_vector_store, fake_sparse):
    _seed_store(fake_vector_store, fake_sparse)
    settings = Settings()
    intent = QueryIntent(
        raw_query="regulatory affairs former",
        keywords=["regulatory"],
        require_current=False,  # no role in the seed is not-current, so expect empty
    )
    scored = retriever.retrieve(
        intent,
        embedder=fake_embedder,
        vector_store=fake_vector_store,
        sparse=fake_sparse,
        settings=settings,
    )
    assert scored == []


def test_retrieve_applies_min_yoe(fake_embedder, fake_vector_store, fake_sparse):
    _seed_store(fake_vector_store, fake_sparse)
    settings = Settings()
    intent = QueryIntent(
        raw_query="senior",
        keywords=["senior"],
        min_yoe=10,
    )
    scored = retriever.retrieve(
        intent,
        embedder=fake_embedder,
        vector_store=fake_vector_store,
        sparse=fake_sparse,
        settings=settings,
    )
    for sr in scored:
        assert sr.role.candidate_yoe >= 10
    # r3 (2 yrs) should not be present
    assert all(sr.role.role_id != "r3" for sr in scored)
