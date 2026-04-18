"""Structured JSON logging for the app. Usable as a drop-in for stdlib logging."""
from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configure structlog + stdlib logging. Call once at app startup."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    timestamper = structlog.processors.TimeStamper(fmt="iso")

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # keep stdlib logging quiet-ish and uniform
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        stream=sys.stdout,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
