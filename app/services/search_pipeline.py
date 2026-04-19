"""Top-level orchestration of /chat.

Contract:
    def run_chat(
        *,
        req: ChatRequest,
        debug: bool,
        embedder, vector_store, sparse, llm, reranker, sessions, settings,
    ) -> ChatResponse:
        Decompose → (classify_followup) → retrieve → rerank → prior_boost → MMR
          → explain → persist session → build response.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from app.core.logging import get_logger
from app.models.api import (
    ChatRequest,
    ChatResponse,
    ExpertHighlight,
    RankedExpert,
)
from app.models.ranking import (
    DebugPayload,
    RankingBreakdown,
    SignalScore,
    StageTiming,
)


_log = get_logger("infoquest.search_pipeline")


def _to_highlight(c) -> ExpertHighlight:
    r = c.best_role
    full_name = ""
    headline = ""
    if c.candidate is not None:
        full_name = (c.candidate.full_name or "").strip()
        headline = (c.candidate.headline or "").strip()
    if not full_name:
        full_name = (c.candidate_id or "")[:8]
    if not headline and r is not None:
        headline = r.candidate_headline or ""
    current_title = r.job_title if (r is not None and r.is_current) else None
    current_company = r.company if (r is not None and r.is_current) else None
    # Fall back to the candidate's actual current role (from the profile) so
    # the response still shows a title when the MATCHED role is historical.
    if current_title is None and c.candidate is not None:
        cur = c.candidate.current_role
        if cur is not None:
            current_title = cur.job_title
            current_company = cur.company
    return ExpertHighlight(
        candidate_id=c.candidate_id,
        full_name=full_name or c.candidate_id[:8],
        headline=headline or "",
        current_title=current_title,
        current_company=current_company,
        matched_role_title=(r.job_title if r is not None else None),
        matched_role_company=(r.company if r is not None else None),
        matched_role_is_current=(bool(r.is_current) if r is not None else None),
        seniority_tier=(r.seniority_tier if r is not None else None),
        industry=(r.industry if r is not None else None),
        country=(r.candidate_country if r is not None else None),
        city=(r.candidate_city if r is not None else None),
        years_of_experience=int(
            (c.candidate.years_of_experience if c.candidate is not None else (r.candidate_yoe if r else 0))
            or 0
        ),
        top_skills=list((r.skill_categories if r is not None else [])[:5]),
        languages=list((r.languages if r is not None else [])),
    )


def _hard_filters_dict(intent) -> dict:
    out: dict = {}
    if intent.geographies:
        out["geographies"] = list(intent.geographies)
    if intent.require_current is not None:
        out["require_current"] = bool(intent.require_current)
    if intent.min_yoe is not None:
        out["min_yoe"] = int(intent.min_yoe)
    return out


def run_chat(
    *,
    req: ChatRequest,
    debug: bool,
    embedder,
    vector_store,
    sparse,
    llm,
    reranker,
    sessions,
    settings,
) -> ChatResponse:
    from app.services import (
        conversation,
        diversity,
        explainer,
        query_decomposer,
        retriever,
    )

    timings: list[StageTiming] = []

    def stage(name: str, start: float) -> None:
        timings.append(
            StageTiming(stage=name, elapsed_ms=(time.time() - start) * 1000.0)
        )

    # Session setup
    if req.conversation_id:
        conv_id = req.conversation_id
        if not sessions.exists(conv_id):
            # Treat unknown conv_id as a new conversation to avoid 404s; create one.
            conv_id = sessions.create()
            prior_intent = None
            prior_ids: list[str] = []
        else:
            prior_intent = sessions.last_intent(conv_id)
            prior_ids = sessions.last_candidate_ids(conv_id) or []
    else:
        conv_id = sessions.create()
        prior_intent = None
        prior_ids = []

    # 1. Decompose
    t = time.time()
    current_intent = query_decomposer.decompose(req.query, llm)
    if not current_intent.raw_query:
        current_intent.raw_query = req.query
    stage("decompose", t)

    # 2. Follow-up classification / intent merge
    if prior_intent is not None:
        t = time.time()
        cls = conversation.classify_followup(req.query, prior_intent, llm)
        stage("classify_followup", t)
        if cls == "refine":
            current_intent = conversation.merge_intent(prior_intent, current_intent)
            # ensure raw_query is the current query
            current_intent.raw_query = req.query

    # 3. Retrieve
    t = time.time()
    roles = retriever.retrieve(
        current_intent,
        embedder=embedder,
        vector_store=vector_store,
        sparse=sparse,
        settings=settings,
    )
    stage("retrieve", t)

    # 4. Rerank
    t = time.time()
    candidates = reranker.rerank(current_intent, roles)
    stage("rerank", t)

    # 5. Prior boost (soft)
    if prior_intent is not None and prior_ids:
        conversation.apply_prior_boost(candidates, prior_ids, settings.prior_shortlist_boost)

    # 6. MMR
    t = time.time()
    final_k = int(req.top_k or settings.final_top_k)
    total = final_k + (settings.why_not_k if req.include_why_not else 0)
    candidates_top = candidates[: settings.rerank_top_k]
    candidates_top = diversity.apply_mmr(
        candidates_top, top_k=total, lambda_=settings.mmr_lambda
    )
    stage("mmr", t)

    # 7. Explain
    t = time.time()
    head = candidates_top[:final_k]
    explainer.explain_matches(current_intent, head, llm)
    if req.include_why_not and len(candidates_top) > final_k:
        tail = candidates_top[final_k : final_k + settings.why_not_k]
        explainer.explain_why_not(current_intent, tail, llm, top_candidates=head)
    stage("explain", t)

    # 8. Persist session (non-blocking — the response is still valid if this fails)
    try:
        sessions.append_turn(
            conv_id,
            req.query,
            current_intent,
            [c.candidate_id for c in head],
        )
    except Exception as exc:
        _log.warning(
            "session_persist_error",
            conversation_id=conv_id,
            error=str(exc),
        )

    # 9. Build response
    ranked: list[RankedExpert] = [
        RankedExpert(
            rank=i + 1,
            expert=_to_highlight(c),
            relevance_score=float(max(0.0, min(1.0, c.relevance_score))),
            match_explanation=c.match_explanation or "",
            why_not=c.why_not,
            highlights=list(c.highlights or []),
        )
        for i, c in enumerate(head)
    ]

    debug_payload: DebugPayload | None = None
    if debug:
        weights_map = settings.weights.as_dict()
        breakdown: list[RankingBreakdown] = []
        for i, c in enumerate(head):
            sigs = [
                SignalScore(
                    name=n,
                    raw=float(r),
                    weight=float(weights_map.get(n, 0.0)),
                    weighted=float(r) * float(weights_map.get(n, 0.0)),
                )
                for n, r in (c.signal_scores or {}).items()
            ]
            breakdown.append(
                RankingBreakdown(
                    candidate_id=c.candidate_id,
                    signals=sigs,
                    maxp_bonus=0.0,
                    prior_shortlist_boost=(
                        settings.prior_shortlist_boost if c.prior_shortlist else 0.0
                    ),
                    final_score=float(c.relevance_score),
                    rank=i + 1,
                )
            )
        debug_payload = DebugPayload(
            query_intent=current_intent.model_dump(),
            hard_filters=_hard_filters_dict(current_intent),
            ranking_breakdown=breakdown,
            timings=timings,
        )

    return ChatResponse(
        conversation_id=conv_id,
        query=req.query,
        results=ranked,
        returned_at=datetime.now(timezone.utc),
        debug=debug_payload,
    )
