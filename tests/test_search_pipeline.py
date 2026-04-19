"""End-to-end integration test for run_chat against fake adapters."""
from __future__ import annotations

import pytest

from app.core.config import Settings, SignalWeights
from app.models.api import ChatRequest
from app.services.reranker import WeightedSignalReranker
from app.services.signals import SIGNALS
from app.services import search_pipeline


def _seed(vs, sparse):
    rows = [
        {
            "role_id": "r1",
            "candidate_id": "c1",
            "job_title": "VP Regulatory Affairs",
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
            "description": "regulatory submissions FDA EMA pharma",
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
            "skill_categories": "Marketing",
            "languages": "en",
            "description": "consumer brands GTM strategy",
        },
        {
            "role_id": "r3",
            "candidate_id": "c3",
            "job_title": "Head of Regulatory Compliance",
            "company": "Sanofi",
            "industry": "Pharmaceuticals",
            "seniority_tier": "head",
            "country": "AE",
            "city": "Abu Dhabi",
            "is_current": True,
            "start_date": "2017-06-01",
            "end_date": "",
            "role_years": 7.0,
            "candidate_yoe": 16,
            "candidate_headline": "Head of Regulatory",
            "skill_categories": "Regulatory,Compliance",
            "languages": "en,ar",
            "description": "regulatory affairs compliance leadership",
        },
    ]
    ids = []
    docs = []
    for r in rows:
        rid = r["role_id"]
        ids.append(rid)
        docs.append(
            f"{r['job_title']} {r['company']} {r['industry']} {r['description']} "
            f"{r['candidate_headline']}"
        )
        vs.upsert(
            ids=[rid],
            embeddings=[[float(len(r["job_title"]) % 7), 1.0, 0.0, 0.0]],
            metadatas=[r],
            documents=[docs[-1]],
        )
    sparse.build(ids, docs)


@pytest.fixture
def reranker():
    return WeightedSignalReranker(
        weights=SignalWeights().as_dict(),
        signals=SIGNALS,
        maxp_bonus=0.05,
        maxp_cap=0.15,
    )


def test_run_chat_end_to_end(
    fake_embedder, fake_vector_store, fake_sparse, fake_llm, fake_sessions, reranker
):
    _seed(fake_vector_store, fake_sparse)
    # give the FakeLLM enough JSON so decompose() hits the merged path
    fake_llm.canned_json = {
        "rewritten_search": "regulatory affairs pharma Middle East",
        "keywords": ["regulatory", "pharma"],
        "geographies": ["AE"],
        "require_current": True,
        "min_yoe": None,
        "exclude_candidate_ids": [],
        "function": "regulatory affairs",
        "industries": ["Pharmaceuticals"],
        "seniority_band": None,
        "skill_categories": [],
    }
    fake_llm.canned_text = "Strong hit on industry and geography."

    settings = Settings()
    req = ChatRequest(
        query="Find regulatory affairs experts in pharma in the Middle East",
        top_k=2,
        include_why_not=False,
    )
    resp = search_pipeline.run_chat(
        req=req,
        debug=True,
        embedder=fake_embedder,
        vector_store=fake_vector_store,
        sparse=fake_sparse,
        llm=fake_llm,
        reranker=reranker,
        sessions=fake_sessions,
        settings=settings,
    )
    assert resp.query == req.query
    assert resp.conversation_id
    assert len(resp.results) <= 2
    # top candidate should be one of the pharma/UAE roles
    top_ids = [r.expert.candidate_id for r in resp.results]
    assert top_ids[0] in {"c1", "c3"}
    # debug payload populated
    assert resp.debug is not None
    assert resp.debug.timings, "timings should be recorded"
    # session persistence recorded the turn
    history = fake_sessions.history(resp.conversation_id)
    assert len(history) == 1
    assert history[0]["top_candidate_ids"] == top_ids


def test_run_chat_follow_up_refine_triggers_merge(
    fake_embedder, fake_vector_store, fake_sparse, fake_sessions, reranker
):
    _seed(fake_vector_store, fake_sparse)
    from tests.conftest import FakeLLM

    # First turn: decompose returns minimal canned JSON; classifier won't be called.
    llm1 = FakeLLM(
        canned_json={
            "rewritten_search": "pharma",
            "keywords": ["pharma"],
            "geographies": ["AE", "SA"],
            "require_current": None,
            "min_yoe": None,
            "exclude_candidate_ids": [],
            "function": "regulatory affairs",
            "industries": ["Pharmaceuticals"],
            "seniority_band": None,
            "skill_categories": [],
        },
        canned_text="exp",
    )
    settings = Settings()
    r1 = search_pipeline.run_chat(
        req=ChatRequest(
            query="regulatory affairs pharma Middle East", top_k=2, include_why_not=False
        ),
        debug=False,
        embedder=fake_embedder,
        vector_store=fake_vector_store,
        sparse=fake_sparse,
        llm=llm1,
        reranker=reranker,
        sessions=fake_sessions,
        settings=settings,
    )
    cid = r1.conversation_id

    # Second turn: classifier says refine.
    class RefineLLM:
        model = "refine"

        def __init__(self):
            self.calls = 0

        def chat(self, **kw):
            return "explanation"

        def chat_json(self, **kw):
            self.calls += 1
            # first call is decompose → return narrow JSON; second is classifier.
            if self.calls == 1:
                return {
                    "rewritten_search": "filter to AE",
                    "keywords": [],
                    "geographies": ["AE"],
                    "require_current": None,
                    "min_yoe": None,
                    "exclude_candidate_ids": [],
                    "function": None,
                    "industries": [],
                    "seniority_band": None,
                    "skill_categories": [],
                }
            return {"classification": "refine", "reason": "narrowing geography"}

        def ping(self):
            return True

    r2 = search_pipeline.run_chat(
        req=ChatRequest(
            query="now filter to the UAE only",
            conversation_id=cid,
            top_k=2,
            include_why_not=False,
        ),
        debug=False,
        embedder=fake_embedder,
        vector_store=fake_vector_store,
        sparse=fake_sparse,
        llm=RefineLLM(),
        reranker=reranker,
        sessions=fake_sessions,
        settings=settings,
    )
    # should preserve same conversation and return only AE-based candidates
    assert r2.conversation_id == cid
    for entry in r2.results:
        if entry.expert.country:
            assert entry.expert.country == "AE"


def test_refinement_filters_to_saudi_arabia_brief_phrase(
    fake_embedder, fake_vector_store, fake_sparse, fake_sessions, reranker
):
    """Brief example follow-up: 'Filter those to only people based in Saudi Arabia.'

    Turn-1 intent has the broad MENA country set (from "Middle East"); turn-2
    intent should narrow geographies to {SA} via the strict-subset rule in
    `merge_intent.narrow_or_union`. Verifies the conversational refinement
    path called out as the brief's example.
    """
    _seed(fake_vector_store, fake_sparse)
    from tests.conftest import FakeLLM

    settings = Settings()

    # Turn 1: broad Middle East query. LLM returns the Pharma-MENA intent.
    llm1 = FakeLLM(
        canned_json={
            "rewritten_search": "regulatory affairs in pharmaceuticals, MENA",
            "keywords": ["regulatory", "affairs", "pharma"],
            "geographies": ["AE", "SA", "QA", "EG"],
            "require_current": None,
            "min_yoe": None,
            "exclude_candidate_ids": [],
            "function": "regulatory affairs",
            "industries": ["Pharmaceuticals"],
            "seniority_band": None,
            "skill_categories": [],
        },
        canned_text="exp",
    )
    r1 = search_pipeline.run_chat(
        req=ChatRequest(
            query="Find regulatory affairs experts in pharma in the Middle East",
            top_k=2,
            include_why_not=False,
        ),
        debug=True,
        embedder=fake_embedder,
        vector_store=fake_vector_store,
        sparse=fake_sparse,
        llm=llm1,
        reranker=reranker,
        sessions=fake_sessions,
        settings=settings,
    )
    cid = r1.conversation_id
    token = r1.session_token
    assert token, "first turn must return session_token"
    turn1_geos = r1.debug.query_intent["geographies"]
    # Multi-country MENA set after merging LLM + regex Middle East expansion.
    assert "SA" in turn1_geos and len(turn1_geos) > 1

    # Turn 2: brief's exact phrase. LLM returns (a) decompose with only SA,
    # then (b) classify_followup -> "refine".
    class TwoCallLLM:
        model = "twocall"

        def __init__(self):
            self.calls = 0

        def chat(self, **kw):
            return "exp"

        def chat_json(self, **kw):
            self.calls += 1
            if self.calls == 1:
                return {
                    "rewritten_search": "Saudi Arabia only",
                    "keywords": ["Saudi"],
                    "geographies": ["SA"],
                    "require_current": None,
                    "min_yoe": None,
                    "exclude_candidate_ids": [],
                    "function": None,
                    "industries": [],
                    "seniority_band": None,
                    "skill_categories": [],
                }
            return {"classification": "refine", "reason": "narrowing geo"}

        def ping(self):
            return True

    r2 = search_pipeline.run_chat(
        req=ChatRequest(
            query="Filter those to only people based in Saudi Arabia.",
            conversation_id=cid,
            top_k=2,
            include_why_not=False,
        ),
        debug=True,
        embedder=fake_embedder,
        vector_store=fake_vector_store,
        sparse=fake_sparse,
        llm=TwoCallLLM(),
        reranker=reranker,
        sessions=fake_sessions,
        settings=settings,
    )

    # session_token NOT echoed on follow-ups (kept out of repeated payloads).
    assert r2.session_token is None
    # Conversation continuity preserved.
    assert r2.conversation_id == cid
    # Strict-subset narrowing: {SA} is a subset of the prior MENA set, so
    # merge_intent should keep just ["SA"], not re-union to MENA.
    turn2_geos = r2.debug.query_intent["geographies"]
    assert turn2_geos == ["SA"], f"expected ['SA'] after narrowing, got {turn2_geos}"
    # Every returned candidate (with a country attached) must be SA-based.
    for entry in r2.results:
        if entry.expert.country:
            assert entry.expert.country == "SA", (
                f"got non-SA country {entry.expert.country!r} after Saudi-only refinement"
            )
