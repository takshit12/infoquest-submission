"""Unit tests for the 7 ranking signals."""
from __future__ import annotations

import math
from datetime import date, timedelta

from app.models.domain import QueryIntent, RoleRecord, ScoredRole
from app.services import signals


def _role(**overrides):
    base = dict(
        role_id="r1",
        candidate_id="c1",
        job_title="Regulatory Affairs Manager",
        company="Pfizer MENA",
        industry="Pharmaceuticals",
        seniority_tier="senior",
        start_date=date(2018, 1, 1),
        end_date=None,
        is_current=True,
        description="Led regulatory submissions to FDA and EMA.",
        role_years=6.0,
        candidate_headline="VP Regulatory Affairs",
        candidate_yoe=18,
        candidate_country="AE",
        candidate_city="Dubai",
        skill_categories=["Regulatory", "Clinical", "Compliance"],
    )
    base.update(overrides)
    return RoleRecord(**base)


def _intent(**overrides):
    base = dict(raw_query="q")
    base.update(overrides)
    return QueryIntent(**base)


def _sr(role: RoleRecord, dense=0.5, sparse=0.5) -> ScoredRole:
    return ScoredRole(role=role, dense_score=dense, sparse_score=sparse)


# ------------------ industry_match ------------------


def test_industry_match_exact_alias():
    r = _sr(_role(industry="Pharmaceuticals"))
    it = _intent(industries=["Pharmaceuticals"])
    assert signals.industry_match(r, it) == 1.0


def test_industry_match_mismatch():
    r = _sr(_role(industry="Manufacturing"))
    it = _intent(industries=["Pharmaceuticals"])
    assert signals.industry_match(r, it) == 0.0


def test_industry_match_neutral_when_empty():
    r = _sr(_role(industry="Banking"))
    it = _intent(industries=[])
    assert signals.industry_match(r, it) == 0.5


def test_industry_match_alias_resolution():
    r = _sr(_role(industry="Pharmaceuticals"))
    it = _intent(industries=["pharma"])  # alias form
    assert signals.industry_match(r, it) == 1.0


# ------------------ function_match ------------------


def test_function_match_exact_substring():
    r = _sr(_role(job_title="Regulatory Affairs Manager"))
    it = _intent(function="regulatory affairs")
    assert signals.function_match(r, it) == 1.0


def test_function_match_partial_token():
    r = _sr(
        _role(
            job_title="Quality Manager",
            description="oversees affairs",
            candidate_headline="Quality Manager",
        )
    )
    it = _intent(function="regulatory affairs")
    # "affairs" (>3 chars) alone should give 0.5
    assert signals.function_match(r, it) == 0.5


def test_function_match_no_match():
    r = _sr(_role(job_title="Sales Lead", description="b2b deals", candidate_headline="Sales"))
    it = _intent(function="regulatory affairs")
    assert signals.function_match(r, it) == 0.0


def test_function_match_neutral_when_none():
    r = _sr(_role())
    it = _intent(function=None)
    assert signals.function_match(r, it) == 0.5


# ------------------ seniority_match ------------------


def test_seniority_match_exact_band_with_yoe():
    r = _sr(_role(seniority_tier="vp", candidate_yoe=20))
    it = _intent(seniority_band="vp")
    val = signals.seniority_match(r, it)
    # base 1.0 * min(1, 20/15)=1 multiplier → 1.0
    assert val == 1.0


def test_seniority_match_near_band():
    r = _sr(_role(seniority_tier="director", candidate_yoe=18))
    it = _intent(seniority_band="vp")
    val = signals.seniority_match(r, it)
    # dist=1 → base 0.7; yoe_scale=1.0; floor 0.6 => base*1.0 = 0.7
    assert abs(val - 0.7) < 1e-6


def test_seniority_match_low_yoe_scales_down():
    r = _sr(_role(seniority_tier="vp", candidate_yoe=3))
    it = _intent(seniority_band="vp")
    val = signals.seniority_match(r, it)
    # base 1 * max(0.6, 3/15=0.2) = 0.6
    assert abs(val - 0.6) < 1e-6


def test_seniority_match_neutral_when_none():
    r = _sr(_role(seniority_tier="senior"))
    it = _intent(seniority_band=None)
    assert signals.seniority_match(r, it) == 0.5


# ------------------ skill_category_match ------------------


def test_skill_category_match_partial():
    r = _sr(_role(skill_categories=["Data", "Engineering"]))
    it = _intent(skill_categories=["Data", "Design"])
    val = signals.skill_category_match(r, it)
    # jaccard(|{data}|, |{data, engineering, design}|) = 1/3
    assert abs(val - 1 / 3) < 1e-6


def test_skill_category_match_full_overlap():
    r = _sr(_role(skill_categories=["Data"]))
    it = _intent(skill_categories=["Data"])
    assert signals.skill_category_match(r, it) == 1.0


def test_skill_category_match_empty_intent_neutral():
    r = _sr(_role(skill_categories=["Data"]))
    it = _intent(skill_categories=[])
    assert signals.skill_category_match(r, it) == 0.5


def test_skill_category_match_no_overlap():
    r = _sr(_role(skill_categories=["Retail"]))
    it = _intent(skill_categories=["Data"])
    assert signals.skill_category_match(r, it) == 0.0


# ------------------ recency_decay ------------------


def test_recency_decay_is_current():
    r = _sr(_role(is_current=True, end_date=None))
    it = _intent()
    assert signals.recency_decay(r, it) == 1.0


def test_recency_decay_none_end_date_treated_as_current():
    r = _sr(_role(is_current=False, end_date=None))
    it = _intent()
    assert signals.recency_decay(r, it) == 1.0


def test_recency_decay_10_years_ago():
    ten_yrs = date.today() - timedelta(days=int(365.25 * 10))
    r = _sr(_role(is_current=False, end_date=ten_yrs))
    it = _intent()
    val = signals.recency_decay(r, it)
    # exp(-1) ≈ 0.3678794
    assert abs(val - math.exp(-1)) < 1e-3


# ------------------ dense / bm25 pass-through ------------------


def test_dense_cosine_passthrough_clamped():
    r = _sr(_role(), dense=0.73)
    it = _intent()
    assert abs(signals.dense_cosine(r, it) - 0.73) < 1e-9


def test_bm25_passthrough_clamped():
    r = _sr(_role(), sparse=0.42)
    it = _intent()
    assert abs(signals.bm25_score(r, it) - 0.42) < 1e-9
