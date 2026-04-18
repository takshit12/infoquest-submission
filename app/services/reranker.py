"""WeightedSignalReranker — composes signal scores and produces ScoredCandidate list.

Contract:

    class WeightedSignalReranker:
        def __init__(self, weights: dict[str, float], signals: dict[str, SignalFn],
                     maxp_bonus: float, maxp_cap: float): ...
        def rerank(self, intent, roles) -> list[ScoredCandidate]:
            # 1. compute per-role weighted score from signals
            # 2. group by candidate_id; candidate score = max + log-scaled bonus for multi-matches
            # 3. sort descending, return top-N
"""
from __future__ import annotations

import math
from collections import defaultdict

from app.models.domain import (
    CandidateProfile,
    QueryIntent,
    RoleRecord,
    ScoredCandidate,
    ScoredRole,
)


class WeightedSignalReranker:
    def __init__(
        self,
        weights: dict[str, float],
        signals: dict,
        maxp_bonus: float = 0.05,
        maxp_cap: float = 0.15,
    ) -> None:
        self.weights = dict(weights)
        self.signals = dict(signals)
        self.maxp_bonus = float(maxp_bonus)
        self.maxp_cap = float(maxp_cap)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_sparse_in_place(roles: list[ScoredRole]) -> None:
        """Normalize ScoredRole.sparse_score so batch max == 1 (in place)."""
        if not roles:
            return
        mx = max((float(r.sparse_score) for r in roles), default=0.0)
        if mx <= 0:
            for r in roles:
                r.sparse_score = 0.0
            return
        for r in roles:
            r.sparse_score = float(r.sparse_score) / mx

    def _stub_candidate_profile(self, role: RoleRecord) -> CandidateProfile:
        """Fallback profile built solely from RoleRecord metadata."""
        return CandidateProfile(
            candidate_id=role.candidate_id,
            first_name="",
            last_name="",
            headline=role.candidate_headline or "",
            years_of_experience=int(role.candidate_yoe or 0),
            city=role.candidate_city,
            country=role.candidate_country,
            nationality=role.candidate_nationality,
            roles=[role],
        )

    # ------------------------------------------------------------------
    # main
    # ------------------------------------------------------------------

    def rerank(
        self, intent: QueryIntent, roles: list[ScoredRole]
    ) -> list[ScoredCandidate]:
        if not roles:
            return []

        # 1. Normalize sparse scores batch-wide.
        self._normalize_sparse_in_place(roles)

        # 2. Compute per-role weighted final_score from signals.
        for role in roles:
            total = 0.0
            sig_scores: dict[str, float] = {}
            for sig_name, weight in self.weights.items():
                fn = self.signals.get(sig_name)
                if fn is None:
                    continue
                try:
                    raw = float(fn(role, intent))
                except Exception:
                    raw = 0.0
                sig_scores[sig_name] = raw
                total += raw * float(weight)
            role.signal_scores = sig_scores
            role.final_score = max(0.0, min(1.0, total))

        # 3. Group roles by candidate_id.
        by_cand: dict[str, list[ScoredRole]] = defaultdict(list)
        for role in roles:
            by_cand[role.role.candidate_id].append(role)

        candidates: list[ScoredCandidate] = []
        # lazy-import profile_builder to avoid side effects when unavailable
        try:
            from app.services import profile_builder  # type: ignore
        except Exception:  # pragma: no cover
            profile_builder = None  # type: ignore

        # Collect the best-of each candidate first, sorted by best role final_score
        bests: list[tuple[str, list[ScoredRole]]] = []
        for cid, group in by_cand.items():
            group.sort(key=lambda r: r.final_score, reverse=True)
            bests.append((cid, group))
        bests.sort(key=lambda kv: kv[1][0].final_score, reverse=True)

        # We only attempt profile fetch for the top `rerank_top_k` candidates to limit DB calls.
        # The reranker doesn't know rerank_top_k directly; use len(bests) — caller trims later.
        for cid, group in bests:
            best = group[0]
            # MaxP bonus: number of "matched enough" roles
            n_matched = sum(1 for r in group if r.final_score >= 0.3)
            extra = max(0, n_matched - 1)
            bonus = min(self.maxp_cap, self.maxp_bonus * math.log1p(extra))
            relevance = min(1.0, float(best.final_score) + float(bonus))

            # Try to fetch the full profile; tolerate NotImplementedError.
            profile: CandidateProfile | None = None
            if profile_builder is not None:
                try:
                    profile = profile_builder.fetch_candidate_profile(cid)
                except NotImplementedError:
                    profile = None
                except Exception:
                    profile = None
            if profile is None:
                profile = self._stub_candidate_profile(best.role)

            candidates.append(
                ScoredCandidate(
                    candidate_id=cid,
                    candidate=profile,
                    best_role=best.role,
                    matched_roles=[r for r in group if r.final_score > 0],
                    signal_scores=dict(best.signal_scores),
                    relevance_score=float(relevance),
                )
            )

        # 4. Sort candidates desc by relevance_score.
        candidates.sort(key=lambda c: c.relevance_score, reverse=True)
        return candidates
