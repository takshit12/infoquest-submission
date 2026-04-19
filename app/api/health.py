"""Liveness + readiness probes.

- ``GET /live``  — process-level liveness, no upstream pings; always 200.
- ``GET /ready`` — readiness check that pings Postgres / Chroma / sessions / LLM.
- ``GET /health`` — back-compat alias for ``/ready`` (existing clients keep
  working, but new infra should split the two).
"""
from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.core.deps import LLMDep, SessionStoreDep, VectorStoreDep
from app.core.logging import get_logger
from app.db import ping as ping_postgres
from app.models.api import DependencyStatus, HealthResponse, LivenessResponse


_log = get_logger("infoquest.health")
router = APIRouter(tags=["health"])


@router.get("/live", response_model=LivenessResponse)
def live() -> LivenessResponse:
    """Minimal liveness probe — does not touch any dependency."""
    return LivenessResponse(status="ok", version=__version__)


@router.get("/health", response_model=HealthResponse)
@router.get("/ready", response_model=HealthResponse)
def health(
    vs: VectorStoreDep,
    llm: LLMDep,
    sessions: SessionStoreDep,
) -> HealthResponse:
    try:
        pg_ok = ping_postgres()
    except Exception as e:
        _log.warning("health_postgres_ping_failed", error=str(e))
        pg_ok = False
    try:
        vs_ok = vs.ping()
    except Exception as e:
        _log.warning("health_vectorstore_ping_failed", error=str(e))
        vs_ok = False
    # LLM ping is optional; some environments may not have the key in /health
    try:
        llm_ok = llm.ping()
    except Exception as e:
        _log.warning("health_llm_ping_failed", error=str(e))
        llm_ok = False
    try:
        sess_ok = sessions.ping()
    except Exception as e:
        _log.warning("health_sessions_ping_failed", error=str(e))
        sess_ok = False

    overall = "ok" if (pg_ok and vs_ok and sess_ok) else "degraded"

    return HealthResponse(
        status=overall,
        version=__version__,
        deps=DependencyStatus(
            postgres="ok" if pg_ok else "error",
            vectorstore="ok" if vs_ok else "error",
            llm="ok" if llm_ok else "error",
        ),
    )
