"""Session store port: persists conversation state across /chat calls."""
from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from app.models.domain import QueryIntent


@runtime_checkable
class SessionStore(Protocol):
    def create(self) -> str:
        """Create a new conversation, return its id."""

    def exists(self, conversation_id: str) -> bool: ...

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
