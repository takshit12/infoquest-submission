"""Industry canonical names + aliases used for matching.

Kept as a simple alias map. Filled in / extended by feat/search worktree.
Industry values observed in the source DB (from inspect_data.py) include:
  Financial Services, Software Development, Higher Education, IT Services and IT Consulting,
  Hospitals and Health Care, Retail, Technology-Information-and-Internet, Business Consulting and Services,
  Non-profit Organizations, Education, Real Estate, Advertising Services, Manufacturing,
  Telecommunications, Construction, Government Administration, Banking, Insurance,
  Hospitality, Venture Capital and Private Equity Principals, ...
"""
from __future__ import annotations

# Canonical name → set of lowercase aliases
INDUSTRY_ALIASES: dict[str, set[str]] = {
    "Financial Services": {"financial services", "finance", "fintech", "financial"},
    "Banking": {"banking", "banks", "investment banking"},
    "Insurance": {"insurance"},
    "Venture Capital and Private Equity Principals": {
        "vc", "venture capital", "private equity", "pe",
    },
    "Software Development": {
        "software development", "software", "software engineering", "saas",
    },
    "IT Services and IT Consulting": {
        "it services", "it consulting", "technology consulting",
    },
    "Technology, Information and Internet": {
        "technology", "internet", "tech", "information technology",
    },
    "Hospitals and Health Care": {
        "healthcare", "health care", "hospitals", "medical", "hospitals and health care",
    },
    "Pharmaceuticals": {
        "pharma", "pharmaceuticals", "pharmaceutical", "life sciences", "biotech", "biotechnology",
    },
    "Higher Education": {"higher education", "university", "academia"},
    "Education": {"education", "k-12", "schools"},
    "Retail": {"retail", "e-commerce", "ecommerce"},
    "Real Estate": {"real estate", "property", "proptech"},
    "Advertising Services": {"advertising", "ad-tech", "adtech"},
    "Manufacturing": {"manufacturing", "industrial"},
    "Telecommunications": {"telecommunications", "telecom"},
    "Construction": {"construction", "built environment"},
    "Government Administration": {"government", "public sector", "govtech"},
    "Hospitality": {"hospitality", "hotels", "travel"},
    "Business Consulting and Services": {
        "consulting", "business consulting", "management consulting",
    },
    "Non-profit Organizations": {"nonprofit", "non-profit", "ngo"},
    "Oil & Gas": {"oil and gas", "oil & gas", "petrochemical", "petrochemicals", "energy"},
}


def canonicalize(text: str) -> str | None:
    """Map an industry mention to the canonical DB name, or None."""
    lc = text.strip().lower()
    for canonical, aliases in INDUSTRY_ALIASES.items():
        if lc == canonical.lower() or lc in aliases:
            return canonical
    return None


def industries_matching(query: str) -> list[str]:
    """Return canonical industries mentioned in the free-text query."""
    lc = query.lower()
    out: list[str] = []
    for canonical, aliases in INDUSTRY_ALIASES.items():
        if canonical.lower() in lc:
            out.append(canonical)
            continue
        if any(a in lc for a in aliases):
            out.append(canonical)
    # dedupe preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for c in out:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped
