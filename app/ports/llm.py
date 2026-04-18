"""LLM client port: chat completion with optional structured-JSON coercion."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Contract for chat-LLM back-ends (OpenRouter, raw Anthropic, etc.)."""

    model: str

    def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Return the assistant message content. Raises on HTTP errors.

        `response_format={"type": "json_object"}` requests strict JSON
        (supported by OpenRouter for most providers).
        """

    def chat_json(
        self,
        *,
        system: str,
        user: str,
        schema_hint: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Convenience wrapper returning a parsed JSON object.

        Raises json.JSONDecodeError on invalid JSON. Callers should
        validate with a Pydantic model.
        """

    def ping(self) -> bool:
        ...
