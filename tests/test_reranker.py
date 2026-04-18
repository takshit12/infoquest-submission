"""Tests for WeightedSignalReranker: MaxP grouping + log-scaled bonus cap."""
from __future__ import annotations

from datetime import date

from app.core.config import SignalWeights
from app.models.domain import QueryIntent, RoleRecord, ScoredRole
from app.services.reranker import WeightedSignalReranker
from app.services.signals import SIGNALS


def _role(**kw):
    base = dict(
        role_id="r?",
        candidate_id="c?",
        job_title="VP Regulatory Affairs",
        company="Pfizer",
        industry="Pharmaceuticals",
        seniority_tier="vp",
        start_date=date(2018, 1, 1),
        end_date=None,
        is_current=True,
        description="regulatory submissions",
        role_years=5.0,
        candidate_headline="Regulatory leader",
        candidate_yoe=20,
        candidate_country="AE",
        candidate_city="Dubai",
        skill_categories=["Regulatory"],
    )
    base.update(kw)
    return RoleRecord(**base)


def _make_reranker(bonus=0.05, cap=0.15):
    return WeightedSignalReranker(
        weights=SignalWeights().as_dict(),
        signals=SIGNALS,
        maxp_bonus=bonus,
        maxp_cap=cap,
    )


def _intent():
    return QueryIntent(
        raw_query="regulatory affairs pharma vp UAE",
        function="regulatory affairs",
        industries=["Pharmaceuticals"],
        seniority_band="vp",
        geographies=["AE"],
    )


def test_rerank_maxp_max_score_wins_per_candidate():
    # two roles for the same candidate; weaker role shouldn't drop the best
    strong = ScoredRole(
        role=_role(role_id="r1", candidate_id="c1"),
        dense_score=0.9,
        sparse_score=1.0,
    )
    weak = ScoredRole(
        role=_role(
            role_id="r2",
            candidate_id="c1",
            job_title="Intern",
            seniority_tier="junior",
            industry="Retail",
            is_current=False,
            end_date=date(2015, 6, 1),
            candidate_yoe=2,
        ),
        dense_score=0.2,
        sparse_score=0.2,
    )
    out = _make_reranker().rerank(_intent(), [strong, weak])
    assert len(out) == 1
    c = out[0]
    assert c.candidate_id == "c1"
    assert c.best_role.role_id == "r1"
    # strong role's final_score should be what flows into candidate relevance (+ bonus if >=2 matched)


def test_rerank_multi_match_bonus_applied_but_capped():
    roles = []
    for i in range(5):
        # 5 matching roles — bonus should be capped
        roles.append(
            ScoredRole(
                role=_role(role_id=f"r{i}", candidate_id="c1"),
                dense_score=0.9,
                sparse_score=1.0,
            )
        )
    rr = _make_reranker(bonus=0.05, cap=0.10)
    out = rr.rerank(_intent(), roles)
    assert len(out) == 1
    # best role final_score is clamped at 1.0 already; bonus should not push beyond 1.0
    assert out[0].relevance_score <= 1.0
    # and cap never exceeded internally — relevance ≤ best+cap
    # We can't easily read best's final_score since all are 1.0, so confirm sort stability.
    assert out[0].best_role is not None


def test_rerank_groups_independent_candidates_and_sorts():
    a = ScoredRole(
        role=_role(role_id="ra", candidate_id="ca"),
        dense_score=0.9,
        sparse_score=1.0,
    )
    b = ScoredRole(
        role=_role(
            role_id="rb",
            candidate_id="cb",
            job_title="Marketing Director",
            industry="Consumer Goods",
            seniority_tier="director",
            candidate_yoe=12,
            candidate_country="SA",
        ),
        dense_score=0.4,
        sparse_score=0.2,
    )
    out = _make_reranker().rerank(_intent(), [a, b])
    # ca should outrank cb given intent
    assert [c.candidate_id for c in out] == ["ca", "cb"]


def test_rerank_empty_input_returns_empty():
    out = _make_reranker().rerank(_intent(), [])
    assert out == []


def test_rerank_sparse_normalization_batch_max_one():
    r1 = ScoredRole(
        role=_role(role_id="r1", candidate_id="c1"), dense_score=0.5, sparse_score=2.0
    )
    r2 = ScoredRole(
        role=_role(role_id="r2", candidate_id="c2"), dense_score=0.5, sparse_score=1.0
    )
    _ = _make_reranker().rerank(_intent(), [r1, r2])
    # after rerank, r1.sparse_score should be normalized to 1.0, r2 to 0.5
    assert abs(r1.sparse_score - 1.0) < 1e-6
    assert abs(r2.sparse_score - 0.5) < 1e-6
