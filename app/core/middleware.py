"""ASGI middleware: request-id, access logging, security headers, optional API key.

Each middleware is implemented as a Starlette-style ASGI class so it can be
unit-tested in isolation and composed via ``app.add_middleware(...)``.
"""
from __future__ import annotations

import time
import uuid
from typing import Awaitable, Callable

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.config import get_settings
from app.core.logging import get_logger


# Routes that never require an API key (public surface + docs + healthcheck).
_API_KEY_EXEMPT_PATHS: frozenset[str] = frozenset(
    {"/", "/health", "/docs", "/redoc", "/openapi.json"}
)


def _is_http(scope: Scope) -> bool:
    return scope.get("type") == "http"


class RequestIDMiddleware:
    """Stamp ``X-Request-ID`` on every request and bind it to structlog context.

    - reads incoming ``X-Request-ID`` header if provided
    - else generates a fresh ``uuid.uuid4().hex``
    - exposes it on the response via the same header
    - clears the structlog contextvars after the response so values don't leak
      across requests on the same worker
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not _is_http(scope):
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
        request_id = headers.get("x-request-id") or uuid.uuid4().hex

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                raw_headers = list(message.get("headers") or [])
                # avoid duplicating if a downstream layer already set it
                if not any(h[0].lower() == b"x-request-id" for h in raw_headers):
                    raw_headers.append((b"x-request-id", request_id.encode("latin-1")))
                message["headers"] = raw_headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            structlog.contextvars.clear_contextvars()


class AccessLogMiddleware:
    """Emit structured ``request_started`` / ``request_completed`` log events."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._log = get_logger("infoquest.access")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not _is_http(scope):
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")
        method: str = scope.get("method", "")

        # Skip noisy health probes.
        if path == "/health":
            await self.app(scope, receive, send)
            return

        self._log.info("request_started", method=method, path=path)
        start = time.perf_counter()
        status_code = 500  # default if the app crashes before sending

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message.get("status", 500))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000.0, 2)
            self._log.info(
                "request_completed",
                method=method,
                path=path,
                status=status_code,
                duration_ms=duration_ms,
            )


class SecurityHeadersMiddleware:
    """Append a small set of conservative security headers to every response."""

    _HEADERS: tuple[tuple[bytes, bytes], ...] = (
        (b"strict-transport-security", b"max-age=31536000; includeSubDomains"),
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
        (b"referrer-policy", b"strict-origin-when-cross-origin"),
    )

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not _is_http(scope):
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                raw_headers = list(message.get("headers") or [])
                existing = {h[0].lower() for h in raw_headers}
                for name, value in self._HEADERS:
                    if name not in existing:
                        raw_headers.append((name, value))
                message["headers"] = raw_headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


class APIKeyMiddleware:
    """Optional ``X-API-Key`` enforcement.

    No-op when ``settings.api_key`` is empty/None. When set, every request to a
    non-exempt path must present a matching ``X-API-Key`` header; otherwise we
    short-circuit a 401 JSON response.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not _is_http(scope):
            await self.app(scope, receive, send)
            return

        # Read settings lazily so tests can mutate env / clear cache between calls.
        configured = (get_settings().api_key or "").strip()
        if not configured:
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")
        if path in _API_KEY_EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
        provided = headers.get("x-api-key", "")
        if provided != configured:
            await _send_json_response(
                send,
                status=401,
                payload=b'{"detail":"invalid or missing API key"}',
            )
            return

        await self.app(scope, receive, send)


async def _send_json_response(send: Send, *, status: int, payload: bytes) -> None:
    """Tiny ASGI JSON-response helper for short-circuited middleware responses."""
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode("latin-1")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload, "more_body": False})


__all__ = [
    "RequestIDMiddleware",
    "AccessLogMiddleware",
    "SecurityHeadersMiddleware",
    "APIKeyMiddleware",
]
