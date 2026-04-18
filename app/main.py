"""FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api import chat as chat_router
from app.api import conversations as conversations_router
from app.api import experts as experts_router
from app.api import health as health_router
from app.api import ingest as ingest_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger


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


app = FastAPI(
    title="InfoQuest Expert Network Search Copilot",
    version=__version__,
    description=(
        "FastAPI backend for natural-language search over a candidate/expert "
        "vector database with precision-oriented ranking."
    ),
    lifespan=lifespan,
)

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
    }
