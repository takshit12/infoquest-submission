"""FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app import __version__
from app.api import chat as chat_router
from app.api import conversations as conversations_router
from app.api import experts as experts_router
from app.api import health as health_router
from app.api import ingest as ingest_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.middleware import (
    AccessLogMiddleware,
    APIKeyMiddleware,
    BodySizeLimitMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    log = get_logger("infoquest.startup")
    log.info(
        "starting",
        version=__version__,
        model=settings.openrouter_model,
        embedding_model=settings.embedding_model,
        chroma_dir=settings.chroma_dir,
    )
    yield
    log.info("shutdown")


_settings = get_settings()

app = FastAPI(
    title="InfoQuest Expert Network Search Copilot",
    version=__version__,
    description=(
        "FastAPI backend for natural-language search over a candidate/expert "
        "vector database with precision-oriented ranking."
    ),
    lifespan=lifespan,
)

# ---- middleware ----
# FastAPI invokes middleware in reverse-registration order on requests
# (last-added runs first). So register innermost-first: security headers wrap
# the response, then access log wraps everything, then API key gating, then
# request-id binding so all subsequent layers see the contextvar, finally
# the body-size limit so oversize requests short-circuit before any work.
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(AccessLogMiddleware)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(BodySizeLimitMiddleware)

# CORS only if explicitly configured (avoid accidental "*" wildcard).
if _settings.cors_allow_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(_settings.cors_allow_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        allow_credentials=False,
    )

# ---- rate limiting (slowapi) ----
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{_settings.rate_limit_per_min}/minute"],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ---- routers ----
app.include_router(health_router.router)
app.include_router(ingest_router.router)
app.include_router(chat_router.router)
app.include_router(experts_router.router)
app.include_router(conversations_router.router)


@app.get("/", include_in_schema=False)
def root():
    return {
        "name": "InfoQuest Expert Network Search Copilot",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
        "live": "/live",
        "ready": "/ready",
    }
