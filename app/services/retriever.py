"""Dense + sparse retrieval, RRF fusion, MaxP candidate aggregation.

Filled in by the feat/search worktree.

Contract:
    def retrieve(
        intent: QueryIntent,
        *,
        embedder: Embedder,
        vector_store: VectorStore,
        sparse: SparseRetriever,
        settings: Settings,
    ) -> list[ScoredRole]:
        '''Compose hard-filter `where` dict from intent (geography, is_current,
        min_yoe). Dense top-K + sparse top-K. RRF fuse on role_id. Return ScoredRole
        list with dense_score / sparse_score / fused_score populated.'''

    def rrf_fuse(
        dense: list[tuple[id, score]],
        sparse: list[tuple[id, score]],
        k: int = 60,
    ) -> list[tuple[id, fused_score]]:
        '''Reciprocal Rank Fusion: 1 / (k + rank_dense) + 1 / (k + rank_sparse).'''
"""
from __future__ import annotations

from datetime import date
from typing import Any

from app.core.config import Settings
from app.models.domain import QueryIntent, RoleRecord, ScoredRole
from app.ports.embedder import Embedder
from app.ports.sparse_retriever import SparseRetriever
from app.ports.vector_store import VectorStore


def _build_where(intent: QueryIntent) -> dict[str, Any] | None:
    """Build a Chroma `where` filter from the hard constraints in the intent."""
    clauses: list[dict[str, Any]] = []
    if intent.geographies:
        clauses.append({"country": {"$in": list(intent.geographies)}})
    if intent.require_current is not None:
        clauses.append({"is_current": bool(intent.require_current)})
    if intent.min_yoe is not None:
        clauses.append({"candidate_yoe": {"$gte": int(intent.min_yoe)}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _metadata_matches_hard(meta: dict[str, Any], intent: QueryIntent) -> bool:
    """Post-filter for sparse hits (BM25 has no metadata filter)."""
    if intent.geographies:
        country = meta.get("country")
        if country not in intent.geographies:
            return False
    if intent.require_current is not None:
        is_current = meta.get("is_current")
        # metadata may round-trip through Chroma as bool or as other — coerce
        is_current_bool = bool(is_current)
        if is_current_bool != bool(intent.require_current):
            return False
    if intent.min_yoe is not None:
        try:
            yoe = int(meta.get("candidate_yoe") or 0)
        except (TypeError, ValueError):
            yoe = 0
        if yoe < int(intent.min_yoe):
            return False
    return True


def _parse_iso_date(value: Any) -> date | None:
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


def _parse_csv_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [s.strip() for s in str(value).split(",") if s.strip()]


def _role_from_metadata(role_id: str, meta: dict[str, Any]) -> RoleRecord:
    """Reconstruct a RoleRecord from stored Chroma metadata per the contract."""
    start = _parse_iso_date(meta.get("start_date")) or date(1970, 1, 1)
    end = _parse_iso_date(meta.get("end_date"))

    # is_current coercion
    is_current_raw = meta.get("is_current")
    if isinstance(is_current_raw, bool):
        is_current = is_current_raw
    elif isinstance(is_current_raw, (int, float)):
        is_current = bool(is_current_raw)
    elif isinstance(is_current_raw, str):
        is_current = is_current_raw.strip().lower() in {"true", "1", "yes"}
    else:
        is_current = False

    try:
        role_years = float(meta.get("role_years") or 0.0)
    except (TypeError, ValueError):
        role_years = 0.0

    try:
        candidate_yoe = int(meta.get("candidate_yoe") or 0)
    except (TypeError, ValueError):
        candidate_yoe = 0

    seniority = meta.get("seniority_tier") or None

    return RoleRecord(
        role_id=role_id,
        candidate_id=str(meta.get("candidate_id") or ""),
        job_title=str(meta.get("job_title") or ""),
        company=str(meta.get("company") or ""),
        industry=meta.get("industry") or None,
        seniority_tier=seniority if seniority else None,
        start_date=start,
        end_date=end,
        is_current=is_current,
        description=str(meta.get("description") or ""),
        role_years=role_years,
        candidate_headline=str(meta.get("candidate_headline") or ""),
        candidate_yoe=candidate_yoe,
        candidate_country=meta.get("country") or None,
        candidate_city=meta.get("city") or None,
        candidate_nationality=meta.get("nationality") or None,
        skill_categories=_parse_csv_list(meta.get("skill_categories")),
        languages=_parse_csv_list(meta.get("languages")),
    )


def rrf_fuse(
    dense: list[tuple[str, float]],
    sparse: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion (deterministic)."""
    scores: dict[str, float] = {}
    for rank, (id_, _score) in enumerate(dense, start=1):
        scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank)
    for rank, (id_, _score) in enumerate(sparse, start=1):
        scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def retrieve(
    intent: QueryIntent,
    *,
    embedder: Embedder,
    vector_store: VectorStore,
    sparse: SparseRetriever,
    settings: Settings,
) -> list[ScoredRole]:
    """Hybrid retrieval: dense (Chroma) + sparse (BM25), fused with RRF."""
    where = _build_where(intent)

    # ---- Dense side ----
    query_text = (intent.rewritten_search or intent.raw_query or "").strip()
    dense_embedding = embedder.embed_query(query_text)
    dense_raw = vector_store.search(
        dense_embedding,
        k=settings.retrieval_top_k_dense,
        where=where,
    )
    # dense_raw is list[(id, score, metadata)]
    dense_meta: dict[str, dict[str, Any]] = {}
    dense_score_map: dict[str, float] = {}
    dense_pairs: list[tuple[str, float]] = []
    for id_, score, meta in dense_raw:
        dense_pairs.append((id_, float(score)))
        dense_score_map[id_] = float(score)
        dense_meta[id_] = dict(meta or {})

    # ---- Sparse side ----
    sparse_query = (" ".join(intent.keywords).strip() or intent.raw_query or "").strip()
    sparse_raw = sparse.search(sparse_query, k=settings.retrieval_top_k_sparse)
    # sparse_raw is list[(id, score)]
    sparse_score_map: dict[str, float] = {}
    sparse_ids_needing_meta: list[str] = []
    for id_, score in sparse_raw:
        sparse_score_map[id_] = float(score)
        if id_ not in dense_meta:
            sparse_ids_needing_meta.append(id_)

    # fetch metadata for sparse-only hits once (batched)
    sparse_meta: dict[str, dict[str, Any]] = {}
    if sparse_ids_needing_meta:
        fetched = vector_store.get(sparse_ids_needing_meta)
        for id_, payload in zip(sparse_ids_needing_meta, fetched or []):
            if payload is None:
                continue
            sparse_meta[id_] = dict(payload.get("metadata") or {})

    # Filter sparse hits by hard constraints.
    sparse_pairs: list[tuple[str, float]] = []
    for id_, score in sparse_raw:
        meta = dense_meta.get(id_) or sparse_meta.get(id_)
        if meta is None:
            # No metadata available — drop if any hard filters exist.
            if where is None:
                sparse_pairs.append((id_, score))
            # else: drop
            continue
        if _metadata_matches_hard(meta, intent):
            sparse_pairs.append((id_, score))

    # ---- Fuse ----
    fused = rrf_fuse(dense_pairs, sparse_pairs, k=settings.rrf_k)

    # ---- Take top fused ids, build ScoredRole ----
    top_n = max(1, settings.rerank_top_k * 3)
    top_fused = fused[:top_n]

    # sparse-score normalization across the batch we return
    max_sparse = max(sparse_score_map.values(), default=0.0)

    # We need metadata for every id in top_fused; gather missing
    missing = [id_ for id_, _ in top_fused if id_ not in dense_meta and id_ not in sparse_meta]
    if missing:
        fetched = vector_store.get(missing)
        for id_, payload in zip(missing, fetched or []):
            if payload is None:
                continue
            sparse_meta[id_] = dict(payload.get("metadata") or {})

    scored: list[ScoredRole] = []
    for id_, fused_score in top_fused:
        meta = dense_meta.get(id_) or sparse_meta.get(id_)
        if meta is None:
            # If we can't reconstruct, skip.
            continue
        role_rec = _role_from_metadata(id_, meta)
        dense_s = dense_score_map.get(id_, 0.0)
        raw_sparse = sparse_score_map.get(id_, 0.0)
        norm_sparse = (raw_sparse / max_sparse) if max_sparse > 0 else 0.0
        scored.append(
            ScoredRole(
                role=role_rec,
                dense_score=float(dense_s),
                sparse_score=float(norm_sparse),
                fused_score=float(fused_score),
            )
        )
    return scored
