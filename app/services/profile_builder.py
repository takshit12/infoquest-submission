"""Assemble CandidateProfile + RoleRecord[] from the normalized source Postgres.

Filled in by the feat/ingest worktree. See `app/models/domain.py` for the
target shape.

Expected public API:

    def iter_candidate_profiles(
        limit: int | None = None,
    ) -> Iterable[CandidateProfile]:
        '''Stream candidates joined with work_experience, companies, education,
        candidate_skills, skill_categories, candidate_languages, cities, countries.
        Each yielded CandidateProfile has `.roles` fully populated (denormalized).'''

    def iter_role_records(
        limit: int | None = None,
    ) -> Iterable[RoleRecord]:
        '''Stream per-work_experience RoleRecords (what gets embedded).
        Typically implemented in terms of iter_candidate_profiles.'''
"""
from __future__ import annotations

from collections.abc import Iterable

from app.models.domain import CandidateProfile, RoleRecord


def iter_candidate_profiles(
    limit: int | None = None,
) -> Iterable[CandidateProfile]:
    raise NotImplementedError(
        "profile_builder.iter_candidate_profiles — implemented in feat/ingest worktree"
    )


def iter_role_records(limit: int | None = None) -> Iterable[RoleRecord]:
    raise NotImplementedError(
        "profile_builder.iter_role_records — implemented in feat/ingest worktree"
    )


def fetch_candidate_profile(candidate_id: str) -> CandidateProfile | None:
    """Used by GET /experts/{id}."""
    raise NotImplementedError(
        "profile_builder.fetch_candidate_profile — implemented in feat/ingest worktree"
    )
