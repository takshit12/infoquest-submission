"""Session store port: persists conversation state across /chat calls."""
from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from app.models.domain import QueryIntent


@runtime_checkable
class SessionStore(Protocol):
    def create(self) -> tuple[str, str]:
        """Create a new conversation. Returns ``(conversation_id, auth_token)``.

        The ``auth_token`` is a per-conversation bearer secret that callers
        must present (via ``X-Session-Token``) to read back history or post
        follow-up turns. It is generated server-side; clients never choose it.
        """

    def exists(self, conversation_id: str) -> bool: ...

    def verify_token(self, conversation_id: str, token: str) -> bool:
        """Constant-time check that ``token`` matches the stored secret.

        Returns False on a missing conversation, an empty stored token, or a
        mismatch. Implementations MUST use a constant-time comparator.
        """

    def append_turn(
        self,
        conversation_id: str,
        query: str,
        intent: QueryIntent,
        top_candidate_ids: list[str],
    ) -> int:
        """Append a turn. Returns its turn_index."""

    def last_intent(self, conversation_id: str) -> QueryIntent | None: ...

    def last_candidate_ids(self, conversation_id: str) -> list[str]: ...

    def history(self, conversation_id: str) -> list[dict]:
        """Return full conversation history, oldest first."""

    def meta(
        self, conversation_id: str
    ) -> tuple[datetime, datetime] | None:
        """Return (created_at, last_activity)."""

    def ping(self) -> bool: ...
