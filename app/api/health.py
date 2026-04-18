"""GET /health — liveness + dependency probes."""
from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.core.deps import LLMDep, SessionStoreDep, VectorStoreDep
from app.db import ping as ping_postgres
from app.models.api import DependencyStatus, HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    vs: VectorStoreDep,
    llm: LLMDep,
    sessions: SessionStoreDep,
) -> HealthResponse:
    pg_ok = ping_postgres()
    vs_ok = vs.ping()
    # LLM ping is optional; some environments may not have the key in /health
    try:
        llm_ok = llm.ping()
    except Exception:
        llm_ok = False
    sess_ok = sessions.ping()

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
