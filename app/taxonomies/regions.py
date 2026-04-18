"""Region → ISO-3166 alpha-2 country code mapping.

Used by the query decomposer (to convert "Middle East" in a query into
the correct country filter) and by tests/demo queries.

Country codes match what lives in `countries.code` in the source DB.
Populated based on actual country presence in the data — see
scripts/inspect_data.py output. Extend as needed from the feat/search worktree.
"""
from __future__ import annotations

# Broad geographic regions. Values are ISO-3166-1 alpha-2 codes.
REGIONS: dict[str, set[str]] = {
    "middle_east": {"SA", "AE", "QA", "EG", "OM", "PS", "JO", "LB", "KW", "BH", "SY", "IQ", "YE", "IL", "TR"},
    "gulf": {"SA", "AE", "QA", "OM", "KW", "BH"},
    "gcc": {"SA", "AE", "QA", "OM", "KW", "BH"},
    "north_africa": {"EG", "MA", "DZ", "TN", "LY", "SD"},
    "mena": {"SA", "AE", "QA", "EG", "OM", "PS", "JO", "LB", "KW", "BH", "MA", "DZ", "TN", "LY", "SD", "IL", "TR"},
    "emea": {"DE", "GB", "FR", "IT", "ES", "NL", "CH", "BE", "SE", "NO", "DK", "FI", "PL", "IE", "AT", "PT", "GR", "CZ", "HU", "RO"},
    "europe": {"DE", "GB", "FR", "IT", "ES", "NL", "CH", "BE", "SE", "NO", "DK", "FI", "PL", "IE", "AT", "PT", "GR", "CZ", "HU", "RO"},
    "apac": {"IN", "PK", "SG", "JP", "KR", "CN", "AU", "NZ", "MY", "ID", "TH", "VN", "PH"},
    "south_asia": {"IN", "PK", "BD", "LK", "NP"},
    "latam": {"MX", "BR", "AR", "CO", "CL", "PE", "VE", "UY"},
    "north_america": {"US", "CA"},
    "americas": {"US", "CA", "MX", "BR", "AR", "CO", "CL", "PE", "VE", "UY"},
}

# Common noun-phrase aliases the decomposer should map. Values lookup in REGIONS.
REGION_ALIASES: dict[str, str] = {
    "middle east": "middle_east",
    "mideast": "middle_east",
    "the gulf": "gulf",
    "gulf region": "gulf",
    "gcc": "gcc",
    "mena": "mena",
    "north africa": "north_africa",
    "emea": "emea",
    "europe": "europe",
    "european": "europe",
    "apac": "apac",
    "asia pacific": "apac",
    "south asia": "south_asia",
    "latin america": "latam",
    "latam": "latam",
    "north america": "north_america",
    "americas": "americas",
}


def resolve_region(text: str) -> set[str]:
    """Return the set of ISO-3166 country codes for a region name, or empty."""
    key = REGION_ALIASES.get(text.strip().lower())
    if not key:
        return set()
    return REGIONS.get(key, set())
