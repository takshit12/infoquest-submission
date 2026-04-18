"""OpenRouter LLM adapter.

OpenRouter exposes an OpenAI-compatible API at
https://openrouter.ai/api/v1 — we use the `openai` SDK with a different
base_url. This also gets us JSON-mode response_format support for providers
that implement it (Anthropic/OpenAI/Google via OpenRouter all do as of 2025).
"""
from __future__ import annotations

import json
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class OpenRouterClient:
    """Implements the LLMClient Protocol."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "anthropic/claude-haiku-4.5",
        timeout: float = 30.0,
        max_retries: int = 2,
        referer: str = "https://infoquest.takehome.local",
        title: str = "InfoQuest Expert Search Copilot",
    ) -> None:
        from openai import OpenAI

        self.model = model
        self._max_retries = max_retries
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            default_headers={
                # OpenRouter-recommended headers for leaderboard + attribution
                "HTTP-Referer": referer,
                "X-Title": title,
            },
        )

    def _retrying(self):
        from openai import APIConnectionError, APITimeoutError, RateLimitError

        return retry(
            reraise=True,
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
            retry=retry_if_exception_type(
                (APIConnectionError, APITimeoutError, RateLimitError)
            ),
        )

    def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        @self._retrying()
        def _call() -> str:
            kwargs: dict[str, Any] = dict(
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            if response_format:
                kwargs["response_format"] = response_format
            resp = self._client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()

        return _call()

    def chat_json(
        self,
        *,
        system: str,
        user: str,
        schema_hint: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        prompt = user
        if schema_hint:
            prompt = (
                f"{user}\n\n"
                "Return a single JSON object matching this schema "
                f"(no prose, no markdown fences):\n{schema_hint}"
            )
        raw = self.chat(
            system=system,
            user=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        # tolerate accidental ```json fences from over-helpful models
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.removeprefix("json").strip()
        return json.loads(cleaned)

    def ping(self) -> bool:
        try:
            # /models is free to call, no tokens used
            self._client.models.list()
            return True
        except Exception:
            return False
