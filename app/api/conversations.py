"""GET /conversations/{id} — inspect a conversation's turn history.

Bearer-protected: the caller must present the per-conversation
``X-Session-Token`` returned on first /chat. Missing → 401, mismatch → 403.
This is the IDOR fix that closes the gap where any client holding a
conversation UUID could pull another caller's turn history.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException

from app.core.deps import SessionStoreDep
from app.models.api import ConversationHistory, ConversationTurn


router = APIRouter(tags=["conversations"])


@router.get("/conversations/{conversation_id}", response_model=ConversationHistory)
def get_conversation(
    conversation_id: str,
    sessions: SessionStoreDep,
    x_session_token: str | None = Header(default=None, alias="X-Session-Token"),
) -> ConversationHistory:
    if not sessions.exists(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")
    if not x_session_token:
        raise HTTPException(status_code=401, detail="missing X-Session-Token")
    if not sessions.verify_token(conversation_id, x_session_token):
        raise HTTPException(status_code=403, detail="invalid session token")
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
