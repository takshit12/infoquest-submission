"""SQLite session store — conversation turns + QueryIntent history.

Each conversation row carries an ``auth_token`` — a per-conversation bearer
secret returned to the caller on ``create()`` and required (via constant-time
compare in :meth:`verify_token`) for any follow-up read or write. This is the
fix for the IDOR where any client holding a conversation UUID could fetch
another caller's turn history.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.models.domain import QueryIntent


_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    last_activity TEXT NOT NULL,
    auth_token    TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS turns (
    conversation_id  TEXT NOT NULL,
    turn_index       INTEGER NOT NULL,
    query            TEXT NOT NULL,
    intent_json      TEXT NOT NULL,
    top_candidate_ids TEXT NOT NULL,  -- JSON-encoded list[str]
    timestamp        TEXT NOT NULL,
    PRIMARY KEY (conversation_id, turn_index),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
"""


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class SQLiteSessionStore:
    """Implements the SessionStore Protocol."""

    def __init__(self, db_path: str = "./sessions.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            # Migrate older DBs that pre-date the auth_token column.
            cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(conversations)").fetchall()
            }
            if "auth_token" not in cols:
                conn.execute(
                    "ALTER TABLE conversations ADD COLUMN auth_token TEXT NOT NULL DEFAULT ''"
                )

    # ---- SessionStore Protocol ----

    def create(self) -> tuple[str, str]:
        conv_id = str(uuid.uuid4())
        token = secrets.token_urlsafe(32)
        now = _now_iso()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO conversations (id, created_at, last_activity, auth_token) "
                "VALUES (?, ?, ?, ?)",
                (conv_id, now, now, token),
            )
        return conv_id, token

    def exists(self, conversation_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        return row is not None

    def verify_token(self, conversation_id: str, token: str) -> bool:
        if not conversation_id or not token:
            return False
        with self._conn() as conn:
            row = conn.execute(
                "SELECT auth_token FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        if not row:
            return False
        stored = row["auth_token"] or ""
        if not stored:
            return False
        return secrets.compare_digest(stored, token)

    def append_turn(
        self,
        conversation_id: str,
        query: str,
        intent: QueryIntent,
        top_candidate_ids: list[str],
    ) -> int:
        now = _now_iso()
        with self._conn() as conn:
            # auto-create conversation if caller didn't (no token issued in this
            # auto-path; the caller already holds the cid so the IDOR check is
            # the caller's responsibility — we don't issue a fresh token here).
            conn.execute(
                "INSERT OR IGNORE INTO conversations (id, created_at, last_activity, auth_token) "
                "VALUES (?, ?, ?, ?)",
                (conversation_id, now, now, ""),
            )
            row = conn.execute(
                "SELECT COALESCE(MAX(turn_index), -1) + 1 AS next_idx FROM turns "
                "WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            turn_index = int(row["next_idx"])
            conn.execute(
                """
                INSERT INTO turns (conversation_id, turn_index, query, intent_json,
                                   top_candidate_ids, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    turn_index,
                    query,
                    intent.model_dump_json(),
                    json.dumps(top_candidate_ids),
                    now,
                ),
            )
            conn.execute(
                "UPDATE conversations SET last_activity = ? WHERE id = ?",
                (now, conversation_id),
            )
        return turn_index

    def last_intent(self, conversation_id: str) -> QueryIntent | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT intent_json FROM turns WHERE conversation_id = ? "
                "ORDER BY turn_index DESC LIMIT 1",
                (conversation_id,),
            ).fetchone()
        if not row:
            return None
        return QueryIntent.model_validate_json(row["intent_json"])

    def last_candidate_ids(self, conversation_id: str) -> list[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT top_candidate_ids FROM turns WHERE conversation_id = ? "
                "ORDER BY turn_index DESC LIMIT 1",
                (conversation_id,),
            ).fetchone()
        if not row:
            return []
        return list(json.loads(row["top_candidate_ids"]))

    def history(self, conversation_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT turn_index, query, intent_json, top_candidate_ids, timestamp "
                "FROM turns WHERE conversation_id = ? ORDER BY turn_index ASC",
                (conversation_id,),
            ).fetchall()
        return [
            {
                "turn_index": r["turn_index"],
                "query": r["query"],
                "query_intent": json.loads(r["intent_json"]),
                "top_candidate_ids": json.loads(r["top_candidate_ids"]),
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]

    def meta(self, conversation_id: str):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT created_at, last_activity FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        if not row:
            return None
        return (
            datetime.fromisoformat(row["created_at"]),
            datetime.fromisoformat(row["last_activity"]),
        )

    def ping(self) -> bool:
        try:
            with self._conn() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False
