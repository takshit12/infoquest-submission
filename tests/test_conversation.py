"""Tests for follow-up classify + intent merge + prior-shortlist boost."""
from __future__ import annotations

from datetime import date

from app.models.domain import (
    CandidateProfile,
    QueryIntent,
    RoleRecord,
    ScoredCandidate,
)
from app.services import conversation
from tests.conftest import FakeLLM


def _candidate(cid: str, rel: float = 0.5) -> ScoredCandidate:
    role = RoleRecord(
        role_id=f"r-{cid}",
        candidate_id=cid,
        job_title="VP",
        company="Acme",
        industry="Financial Services",
        seniority_tier="vp",
        start_date=date(2019, 1, 1),
        is_current=True,
        candidate_yoe=15,
    )
    prof = CandidateProfile(
        candidate_id=cid,
        first_name="F",
        last_name="L",
        years_of_experience=15,
        roles=[role],
    )
    return ScoredCandidate(
        candidate_id=cid,
        candidate=prof,
        best_role=role,
        relevance_score=rel,
    )


def test_merge_intent_strict_narrowing_on_subset():
    prior = QueryIntent(
        raw_query="q1",
        geographies=["SA", "AE", "QA"],
        industries=["Pharmaceuticals", "Banking"],
        function="regulatory affairs",
        seniority_band="senior",
        min_yoe=10,
    )
    current = QueryIntent(
        raw_query="q2",
        geographies=["SA"],  # strict subset
        industries=["Pharmaceuticals"],  # strict subset
    )
    merged = conversation.merge_intent(prior, current)
    assert merged.decomposer_source == "merged"
    # narrowing applied for both list fields
    assert merged.geographies == ["SA"]
    assert merged.industries == ["Pharmaceuticals"]
    # scalar fields from prior preserved when current is None
    assert merged.function == "regulatory affairs"
    assert merged.seniority_band == "senior"
    assert merged.min_yoe == 10
    # raw_query from current
    assert merged.raw_query == "q2"


def test_merge_intent_union_when_current_not_subset():
    prior = QueryIntent(
        raw_query="q1",
        geographies=["SA"],
        industries=["Pharmaceuticals"],
    )
    current = QueryIntent(
        raw_query="q2",
        geographies=["AE", "BR"],  # not subset
        industries=["Banking"],  # not subset
    )
    merged = conversation.merge_intent(prior, current)
    assert set(merged.geographies) == {"SA", "AE", "BR"}
    assert set(merged.industries) == {"Pharmaceuticals", "Banking"}


def test_merge_intent_current_scalars_take_precedence():
    prior = QueryIntent(raw_query="q1", function="operations", min_yoe=5)
    current = QueryIntent(raw_query="q2", function="marketing", min_yoe=20)
    merged = conversation.merge_intent(prior, current)
    assert merged.function == "marketing"
    assert merged.min_yoe == 20


def test_apply_prior_boost_mutates_and_resorts():
    a = _candidate("a", rel=0.4)
    b = _candidate("b", rel=0.6)
    c = _candidate("c", rel=0.5)
    cands = [b, c, a]  # already sorted desc
    conversation.apply_prior_boost(cands, prior_ids=["a"], boost=0.3)
    # a should have moved to top: 0.4 + 0.3 = 0.7 > 0.6 and > 0.5
    assert cands[0].candidate_id == "a"
    assert cands[0].prior_shortlist is True
    assert abs(cands[0].relevance_score - 0.7) < 1e-9


def test_apply_prior_boost_noop_if_no_prior_ids():
    a = _candidate("a", rel=0.4)
    b = _candidate("b", rel=0.6)
    cands = [b, a]
    conversation.apply_prior_boost(cands, prior_ids=[], boost=0.3)
    assert cands[0].candidate_id == "b"
    assert cands[0].relevance_score == 0.6
    assert a.prior_shortlist is False


def test_classify_followup_refine():
    llm = FakeLLM(canned_json={"classification": "refine", "reason": "narrowing"})
    prior = QueryIntent(raw_query="q1", function="operations")
    assert conversation.classify_followup("filter to SA", prior, llm) == "refine"


def test_classify_followup_new_on_unrecognized():
    llm = FakeLLM(canned_json={"classification": "bogus"})
    prior = QueryIntent(raw_query="q1")
    assert conversation.classify_followup("anything", prior, llm) == "new"


def test_classify_followup_default_new_on_error():
    class BrokenLLM:
        model = "b"

        def chat_json(self, **kw):
            raise RuntimeError("boom")

        def chat(self, **kw):
            return ""

        def ping(self):
            return False

    assert (
        conversation.classify_followup("hi", QueryIntent(raw_query="q1"), BrokenLLM())
        == "new"
    )
