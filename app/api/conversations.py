"""GET /conversations/{id} — inspect a conversation's turn history."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.core.deps import SessionStoreDep
from app.models.api import ConversationHistory, ConversationTurn


router = APIRouter(tags=["conversations"])


@router.get("/conversations/{conversation_id}", response_model=ConversationHistory)
def get_conversation(
    conversation_id: str, sessions: SessionStoreDep
) -> ConversationHistory:
    if not sessions.exists(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")
    meta = sessions.meta(conversation_id)
    history = sessions.history(conversation_id)
    created, last = meta if meta else (datetime.now(timezone.utc), datetime.now(timezone.utc))
    return ConversationHistory(
        conversation_id=conversation_id,
        turns=[
            ConversationTurn(
                turn_index=h["turn_index"],
                query=h["query"],
                query_intent=h["query_intent"],
                top_candidate_ids=h["top_candidate_ids"],
                timestamp=datetime.fromisoformat(h["timestamp"]),
            )
            for h in history
        ],
        created_at=created,
        last_activity=last,
    )
