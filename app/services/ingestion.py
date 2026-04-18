"""Orchestrates the full ingest pipeline.

Contract:
    run_ingest(*, embedder, vector_store, sparse, reset=False, limit=None) -> IngestResponse

Pipeline:
    DB  -> RoleRecord iterator
        -> batches of embedding_batch_size
        -> embedder.embed_documents(texts)
        -> vector_store.upsert(ids, embeddings, metadatas, documents)
    After the loop: sparse.build(all_ids, all_texts) once (BM25 is global).

Metadata keys emitted are the exact contract the search worktree reads. Keep
them in sync with `_role_to_metadata`.
"""
from __future__ import annotations

import os
import time
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.api import IngestResponse
from app.models.domain import RoleRecord
from app.ports.embedder import Embedder
from app.ports.sparse_retriever import SparseRetriever
from app.ports.vector_store import VectorStore
from app.services.profile_builder import iter_role_records


_log = get_logger("infoquest.ingest")


# Metadata keys — the search worktree reads these exact names. Do NOT change
# without coordinating with feat/search.
_METADATA_KEYS = (
    "candidate_id",
    "role_id",
    "job_title",
    "seniority_tier",
    "company",
    "industry",
    "country",
    "city",
    "nationality",
    "is_current",
    "start_date",
    "end_date",
    "role_years",
    "candidate_yoe",
    "candidate_headline",
    "skill_categories",
    "languages",
)


def _role_to_metadata(r: RoleRecord) -> dict[str, Any]:
    """Project a RoleRecord onto the Chroma metadata dict.

    List values (skill_categories, languages) are left as lists here; the
    Chroma adapter's `_coerce_metadata` joins them to CSV on upsert. Empty
    strings are used (not None) for nullable scalars so Chroma keeps the keys.
    """
    return {
        "candidate_id": r.candidate_id,
        "role_id": r.role_id,
        "job_title": r.job_title,
        "seniority_tier": r.seniority_tier or "mid",
        "company": r.company,
        "industry": r.industry or "",
        "country": r.candidate_country or "",
        "city": r.candidate_city or "",
        "nationality": r.candidate_nationality or "",
        "is_current": bool(r.is_current),
        "start_date": r.start_date.isoformat(),
        "end_date": r.end_date.isoformat() if r.end_date else "",
        "role_years": float(r.role_years),
        "candidate_yoe": int(r.candidate_yoe),
        "candidate_headline": r.candidate_headline or "",
        "skill_categories": list(r.skill_categories),
        "languages": list(r.languages),
    }


def run_ingest(
    *,
    embedder: Embedder,
    vector_store: VectorStore,
    sparse: SparseRetriever,
    reset: bool = False,
    limit: int | None = None,
) -> IngestResponse:
    """Run the end-to-end ingest. Returns summary counts + wall time."""
    settings = get_settings()
    batch_size = max(1, int(settings.embedding_batch_size))

    t0 = time.perf_counter()

    if reset:
        _log.info("ingest.reset", msg="dropping existing dense + sparse indices")
        vector_store.reset()
        try:
            bm25_path = settings.bm25_index_path
            if bm25_path and os.path.exists(bm25_path):
                os.remove(bm25_path)
        except OSError as exc:
            _log.warning("ingest.reset.bm25_delete_failed", error=str(exc))

    # Accumulators for the final BM25 build + response.
    all_ids: list[str] = []
    all_texts: list[str] = []
    candidate_ids: set[str] = set()
    total_roles = 0
    batches = 0

    # Per-batch working lists (reused to avoid allocation churn).
    batch: list[RoleRecord] = []

    def flush() -> None:
        nonlocal batches
        if not batch:
            return
        texts = [r.to_embedding_text() for r in batch]
        ids = [r.role_id for r in batch]
        metadatas = [_role_to_metadata(r) for r in batch]
        embeddings = embedder.embed_documents(texts)
        vector_store.upsert(ids, embeddings, metadatas, texts)

        all_ids.extend(ids)
        all_texts.extend(texts)
        batches += 1
        _log.info(
            "ingest.batch",
            batch_index=batches,
            batch_size=len(batch),
            roles_seen=total_roles,
            candidates_seen=len(candidate_ids),
        )
        batch.clear()

    for role in iter_role_records(limit=limit):
        batch.append(role)
        candidate_ids.add(role.candidate_id)
        total_roles += 1
        if len(batch) >= batch_size:
            flush()
    flush()  # tail batch

    # BM25 is a global, single-pass index — build once over everything ingested
    # this run. (For incremental ingests, one would rebuild against the full
    # corpus; out of scope here.)
    _log.info("ingest.sparse_build", docs=len(all_ids))
    sparse.build(all_ids, all_texts)

    elapsed = time.perf_counter() - t0

    response = IngestResponse(
        candidates=len(candidate_ids),
        roles=total_roles,
        dense_docs=int(vector_store.count()),
        sparse_docs=int(sparse.count()),
        elapsed_seconds=round(elapsed, 2),
    )
    _log.info("ingest.done", **response.model_dump())
    return response


__all__ = ["run_ingest", "_role_to_metadata"]
