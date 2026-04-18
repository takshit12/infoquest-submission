"""Seniority extraction from job titles.

Filled in / extended by the feat/search worktree. The heuristic matches
title-prefix keywords; YoE is layered in by the seniority_match signal.

Observed top titles in the data are all leadership:
  Head of X, VP of X, Director of X, Principal X, Staff X.
"""
from __future__ import annotations

import re
from typing import Literal

SeniorityTier = Literal[
    "junior", "mid", "senior", "director", "vp", "head", "cxo", "staff_principal"
]

# Ordered from most specific / most senior to least.
_TITLE_PATTERNS: list[tuple[re.Pattern[str], SeniorityTier]] = [
    (re.compile(r"\b(chief|c[a-z]o)\b", re.I), "cxo"),
    (re.compile(r"\b(head of|vp of|vice president)\b", re.I), "vp"),
    (re.compile(r"\b(head)\b", re.I), "head"),
    (re.compile(r"\b(director)\b", re.I), "director"),
    (re.compile(r"\b(principal|staff)\b", re.I), "staff_principal"),
    (re.compile(r"\b(sr\.?|senior|lead)\b", re.I), "senior"),
    (re.compile(r"\b(jr\.?|junior|intern|associate)\b", re.I), "junior"),
]

# Bands used to compare to intent.seniority_band
_BAND_ORDER: list[str] = ["junior", "mid", "senior", "director", "vp", "head", "cxo"]


def tier_from_title(title: str) -> SeniorityTier:
    t = (title or "").strip()
    for pattern, tier in _TITLE_PATTERNS:
        if pattern.search(t):
            return tier
    return "mid"


def band_distance(tier: SeniorityTier, band: str) -> int:
    """Absolute distance between the role tier and the target seniority band.
    0 = exact, larger = further. Used by seniority_match signal.
    Treats 'staff_principal' as equivalent to 'senior' for band purposes."""
    tier_as_band = "senior" if tier == "staff_principal" else tier
    try:
        return abs(_BAND_ORDER.index(tier_as_band) - _BAND_ORDER.index(band))
    except ValueError:
        return len(_BAND_ORDER)  # max distance if unknown
