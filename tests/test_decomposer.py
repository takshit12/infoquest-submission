"""Query decomposer tests — regex fallback coverage + LLM-path merge."""
from __future__ import annotations

from app.services import query_decomposer
from tests.conftest import FakeLLM


# ------------------------------------------------------------------
# regex_fallback — hand-written queries
# ------------------------------------------------------------------


def test_regex_fallback_region_middle_east():
    i = query_decomposer.regex_fallback(
        "Find regulatory affairs experts in the Middle East"
    )
    # Middle East expands to the MENA-style set
    assert "SA" in i.geographies
    assert "AE" in i.geographies
    assert i.function == "regulatory affairs"
    assert i.decomposer_source == "regex_fallback"


def test_regex_fallback_seniority_vp():
    i = query_decomposer.regex_fallback("VP of engineering, senior preferred")
    assert i.seniority_band == "vp"


def test_regex_fallback_seniority_junior():
    i = query_decomposer.regex_fallback("junior data engineers anywhere")
    assert i.seniority_band == "junior"


def test_regex_fallback_min_yoe_plus():
    i = query_decomposer.regex_fallback("Senior product leaders with 15+ years experience")
    assert i.min_yoe == 15
    assert i.seniority_band == "senior"


def test_regex_fallback_former():
    i = query_decomposer.regex_fallback("former CPO at a petrochemical company")
    assert i.require_current is False


def test_regex_fallback_currently():
    i = query_decomposer.regex_fallback(
        "currently active regulatory affairs leaders in pharma"
    )
    assert i.require_current is True


# ------------------------------------------------------------------
# regex_fallback — career trajectory extraction
# ------------------------------------------------------------------


def test_regex_fallback_trajectory_former_also_sets_require_current():
    i = query_decomposer.regex_fallback("former CPO at a petrochemical company")
    assert i.career_trajectory == "former"
    assert i.require_current is False  # consistency with hard filter


def test_regex_fallback_trajectory_current_also_sets_require_current():
    i = query_decomposer.regex_fallback("currently a regulatory affairs lead")
    assert i.career_trajectory == "current"
    assert i.require_current is True


def test_regex_fallback_trajectory_ascending_does_not_set_require_current():
    i = query_decomposer.regex_fallback("rising senior data scientist anywhere")
    assert i.career_trajectory == "ascending"
    # ascending is a soft preference; must not lock the hard filter
    assert i.require_current is None


def test_regex_fallback_trajectory_transitioning():
    i = query_decomposer.regex_fallback(
        "product manager transitioning into developer relations"
    )
    assert i.career_trajectory == "transitioning"
    assert i.require_current is None


def test_regex_fallback_trajectory_none_when_unmarked():
    i = query_decomposer.regex_fallback("VP of engineering at SaaS companies")
    assert i.career_trajectory is None


def test_regex_fallback_industry_alias_pharma():
    i = query_decomposer.regex_fallback("pharma regulatory leader")
    assert "Pharmaceuticals" in i.industries


def test_regex_fallback_country_code():
    i = query_decomposer.regex_fallback("find people in SA with banking experience")
    assert "SA" in i.geographies
    assert "Banking" in i.industries


def test_regex_fallback_keywords_exclude_stopwords():
    i = query_decomposer.regex_fallback("junior data engineers in pharma")
    # stopwords like "junior" shouldn't appear in distinctive keywords
    assert "junior" not in i.keywords
    # but content words should
    assert any("data" in k or "pharma" in k or "engineer" in k for k in i.keywords)


# ------------------------------------------------------------------
# decompose() — happy path with canned JSON
# ------------------------------------------------------------------


def test_decompose_merges_llm_and_regex():
    canned = {
        "rewritten_search": "regulatory affairs pharma MENA",
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
    llm = FakeLLM(canned_json=canned)
    intent = query_decomposer.decompose(
        "find former regulatory affairs experts in the Middle East with 10+ years", llm
    )
    assert intent.decomposer_source == "merged"
    # regex should fill in: require_current=False (former), min_yoe=10, geos union
    assert intent.require_current is False
    assert intent.min_yoe == 10
    # LLM's AE merged with regex's expanded middle-east countries
    assert "AE" in intent.geographies
    assert "SA" in intent.geographies
    # Industries union (both say Pharmaceuticals) — no duplicate
    assert intent.industries.count("Pharmaceuticals") == 1
    # raw_query preserved
    assert "Middle East" in intent.raw_query


def test_decompose_falls_back_on_llm_error(monkeypatch):
    class BrokenLLM:
        model = "broken"

        def chat(self, **kw):
            raise RuntimeError("boom")

        def chat_json(self, **kw):
            raise RuntimeError("boom")

        def ping(self):
            return False

    intent = query_decomposer.decompose(
        "junior data engineers anywhere", BrokenLLM()
    )
    assert intent.decomposer_source == "regex_fallback"
    assert intent.seniority_band == "junior"
