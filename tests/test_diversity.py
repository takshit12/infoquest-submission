"""Tests for MMR diversity."""
from __future__ import annotations

from datetime import date

from app.models.domain import (
    CandidateProfile,
    RoleRecord,
    ScoredCandidate,
)
from app.services.diversity import apply_mmr


def _cand(cid: str, company: str, industry: str, rel: float) -> ScoredCandidate:
    role = RoleRecord(
        role_id=f"r-{cid}",
        candidate_id=cid,
        job_title="VP",
        company=company,
        industry=industry,
        seniority_tier="vp",
        start_date=date(2018, 1, 1),
        is_current=True,
    )
    prof = CandidateProfile(candidate_id=cid, first_name=cid, last_name="")
    return ScoredCandidate(
        candidate_id=cid,
        candidate=prof,
        best_role=role,
        relevance_score=rel,
    )


def test_mmr_dedupes_same_company():
    # Two same-company candidates at rank 0 and 1; a distinct one at rank 2.
    cands = [
        _cand("a", "Acme", "Finance", 0.95),
        _cand("b", "Acme", "Finance", 0.94),  # very similar (same company)
        _cand("c", "Beta", "Retail", 0.80),
    ]
    out = apply_mmr(cands, top_k=2, lambda_=0.5)
    ids = [c.candidate_id for c in out]
    # a should be first; c should displace b because b is same-company as a
    assert ids[0] == "a"
    assert ids[1] == "c"


def test_mmr_returns_all_when_below_top_k():
    cands = [
        _cand("a", "Acme", "Finance", 0.95),
        _cand("b", "Beta", "Finance", 0.94),
    ]
    out = apply_mmr(cands, top_k=5, lambda_=0.7)
    assert len(out) == 2
    assert {c.mmr_rank for c in out} == {0, 1}


def test_mmr_empty_input_is_empty():
    assert apply_mmr([], top_k=5) == []


def test_mmr_preserves_order_when_no_duplicates():
    cands = [
        _cand("a", "Acme", "Finance", 0.9),
        _cand("b", "Beta", "Tech", 0.8),
        _cand("c", "Gamma", "Retail", 0.7),
    ]
    out = apply_mmr(cands, top_k=3, lambda_=0.7)
    assert [c.candidate_id for c in out] == ["a", "b", "c"]
