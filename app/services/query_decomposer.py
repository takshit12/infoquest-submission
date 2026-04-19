"""Natural-language query → structured QueryIntent.

Flow:
  1. Call LLM with prompts/decompose_query.md; parse JSON.
  2. Validate with QueryIntent; on ValidationError or JSON error, fall back.
  3. Regex fallback: scan query for country names, title prefixes, industry
     aliases (from taxonomies/). Merge with LLM output (LLM > regex when
     both agree; regex fills gaps / unions lists).
  4. Set `decomposer_source` to 'llm' | 'regex_fallback' | 'merged'.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from app.models.domain import QueryIntent
from app.ports.llm import LLMClient
from app.taxonomies import industry_aliases
from app.taxonomies.regions import REGION_ALIASES, REGIONS, resolve_region


_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "decompose_query.md"


@lru_cache(maxsize=1)
def _load_system_prompt() -> str:
    try:
        return _PROMPT_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


@lru_cache(maxsize=1)
def _all_country_codes() -> set[str]:
    codes: set[str] = set()
    for s in REGIONS.values():
        codes |= set(s)
    return codes


_STOPWORDS = {
    "a", "an", "and", "or", "the", "of", "in", "on", "at", "to", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "can", "could", "should", "may",
    "me", "my", "mine", "we", "our", "us", "you", "your", "they", "them", "their",
    "who", "what", "when", "where", "why", "how", "which", "that", "this", "these", "those",
    "find", "show", "give", "want", "need", "like", "please", "some", "any",
    "people", "person", "expert", "experts", "someone", "anyone",
    "work", "working", "worked", "role", "roles", "job", "jobs",
    "from", "by", "as", "about", "over", "under", "years", "year", "yrs", "yr",
    "experience", "experienced",
    "current", "currently", "former", "ex", "past", "present",
    "based", "currently", "now",
    "junior", "senior", "director", "vp", "head", "chief", "principal", "staff",
    "lead", "manager",
    "anywhere",
}


def _extract_keywords(query: str, max_n: int = 6) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]+", query)
    out: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        low = tok.lower()
        if low in _STOPWORDS:
            continue
        if len(low) < 3:
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(low)
        if len(out) >= max_n:
            break
    return out


_SENIORITY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(vice president|vp)\b", re.I), "vp"),
    (re.compile(r"\bhead of\b", re.I), "head"),
    (re.compile(r"\bdirector\b", re.I), "director"),
    (re.compile(r"\b(cxo|c[a-z]o|chief\s+[a-z]+\s+officer)\b", re.I), "cxo"),
    (re.compile(r"\b(senior|sr\.?)\b", re.I), "senior"),
    (re.compile(r"\b(junior|jr\.?)\b", re.I), "junior"),
]


def _detect_seniority(query: str) -> str | None:
    for pat, label in _SENIORITY_PATTERNS:
        if pat.search(query):
            return label
    return None


def _detect_require_current(query: str) -> bool | None:
    q = query.lower()
    if re.search(r"\b(former|ex-|past)\b", q):
        return False
    if re.search(r"\bcurrently\b", q):
        return True
    if re.search(r"\bcurrent\s+[a-z]+", q):
        return True
    return None


def _detect_min_yoe(query: str) -> int | None:
    m = re.search(r"\b(\d+)\s*\+?\s*(?:years?|yrs?)\b", query, flags=re.I)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


_FUNCTION_CANDIDATES: list[str] = [
    "regulatory affairs",
    "product management",
    "data engineering",
    "marketing",
    "operations",
    "design",
    "engineering",
    "consulting",
]


def _detect_function(query: str) -> str | None:
    low = query.lower()
    for fn in _FUNCTION_CANDIDATES:
        if fn in low:
            return fn
    return None


_COUNTRY_NAME_ALIASES: dict[str, str] = {
    # Full names / common aliases → ISO alpha-2.
    # Limited to what's actually present in the corpus (see inspect_data.py).
    "united states": "US", "usa": "US", "america": "US",
    "united kingdom": "GB", "britain": "GB", "england": "GB",
    "germany": "DE", "deutschland": "DE",
    "france": "FR",
    "italy": "IT",
    "spain": "ES",
    "netherlands": "NL", "holland": "NL",
    "switzerland": "CH",
    "belgium": "BE",
    "ireland": "IE",
    "poland": "PL",
    "portugal": "PT",
    "austria": "AT",
    "india": "IN",
    "pakistan": "PK",
    "china": "CN",
    "japan": "JP",
    "singapore": "SG",
    "south korea": "KR", "korea": "KR",
    "saudi arabia": "SA", "saudi": "SA",
    "united arab emirates": "AE", "uae": "AE", "emirates": "AE",
    "qatar": "QA",
    "egypt": "EG",
    "oman": "OM",
    "turkey": "TR",
    "mexico": "MX",
    "canada": "CA",
    "brazil": "BR",
    "argentina": "AR",
}


def _detect_geographies(query: str) -> list[str]:
    low = query.lower()
    geos: set[str] = set()

    # regions (multi-word substring match)
    for alias_key in REGION_ALIASES.keys():
        if alias_key in low:
            geos |= resolve_region(alias_key)

    # country names / common aliases (case-insensitive word-boundary match)
    for name, code in _COUNTRY_NAME_ALIASES.items():
        if re.search(rf"\b{re.escape(name)}\b", low):
            geos.add(code)

    # Uppercase alpha-2 code mentions. Deliberately case-SENSITIVE to avoid
    # false positives from common English words: "at" (AT=Austria),
    # "us" (US=United States, but also a pronoun), "in" (IN=India but also
    # a preposition), "or" (OR=Oregon region), "is" (IS=Iceland), etc.
    # Written query usage almost always spells country names out; explicit
    # codes appear only in UPPER, so UPPER-only is the safer filter.
    codes = _all_country_codes()
    pattern = re.compile(r"\b(" + "|".join(re.escape(c) for c in codes) + r")\b")
    for m in pattern.finditer(query):
        geos.add(m.group(1))

    return sorted(geos)


def regex_fallback(query: str) -> QueryIntent:
    """Best-effort structured intent without any LLM call."""
    industries = industry_aliases.industries_matching(query)
    geos = _detect_geographies(query)
    seniority = _detect_seniority(query)
    require_current = _detect_require_current(query)
    min_yoe = _detect_min_yoe(query)
    function = _detect_function(query)
    keywords = _extract_keywords(query)

    return QueryIntent(
        raw_query=query,
        rewritten_search=query.strip(),
        keywords=keywords,
        geographies=list(geos),
        require_current=require_current,
        min_yoe=min_yoe,
        exclude_candidate_ids=[],
        function=function,
        industries=industries,
        seniority_band=seniority,  # type: ignore[arg-type]
        skill_categories=[],
        decomposer_source="regex_fallback",
    )


def _merge(llm_intent: QueryIntent, regex_intent: QueryIntent) -> QueryIntent:
    """LLM wins on scalars; list fields are unioned with regex."""
    # Union list fields preserving LLM order first.
    def union_list(a: list[str], b: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in list(a) + list(b):
            if x is None:
                continue
            key = str(x)
            if key in seen:
                continue
            seen.add(key)
            out.append(x)
        return out

    merged = QueryIntent(
        raw_query=llm_intent.raw_query or regex_intent.raw_query,
        rewritten_search=llm_intent.rewritten_search or regex_intent.rewritten_search,
        keywords=union_list(llm_intent.keywords, regex_intent.keywords),
        geographies=union_list(llm_intent.geographies, regex_intent.geographies),
        require_current=(
            llm_intent.require_current
            if llm_intent.require_current is not None
            else regex_intent.require_current
        ),
        min_yoe=(
            llm_intent.min_yoe if llm_intent.min_yoe is not None else regex_intent.min_yoe
        ),
        exclude_candidate_ids=union_list(
            llm_intent.exclude_candidate_ids, regex_intent.exclude_candidate_ids
        ),
        function=llm_intent.function or regex_intent.function,
        industries=union_list(llm_intent.industries, regex_intent.industries),
        seniority_band=llm_intent.seniority_band or regex_intent.seniority_band,
        skill_categories=union_list(
            llm_intent.skill_categories, regex_intent.skill_categories
        ),
        decomposer_source="merged",
        warnings=list(llm_intent.warnings) + list(regex_intent.warnings),
    )
    return merged


def decompose(query: str, llm: LLMClient) -> QueryIntent:
    """LLM decomposition with regex fallback + merge."""
    system = _load_system_prompt()
    try:
        data = llm.chat_json(system=system, user=query)
        if not isinstance(data, dict):
            raise ValueError("chat_json did not return a dict")
        data.setdefault("raw_query", query)
        llm_intent = QueryIntent.model_validate(data)
        # also always compute regex fallback and merge
        reg = regex_fallback(query)
        merged = _merge(llm_intent, reg)
        merged.raw_query = query
        return merged
    except Exception:
        fallback = regex_fallback(query)
        fallback.raw_query = query
        fallback.decomposer_source = "regex_fallback"
        return fallback
