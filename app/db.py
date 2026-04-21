"""Connection factory for the read-only source Postgres."""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import psycopg2
import psycopg2.extras

from app.core.config import get_settings
from app.core.logging import get_logger


_log = get_logger("infoquest.db")


@contextlib.contextmanager
def source_conn() -> Iterator[Any]:
    """Yield a read-only psycopg2 connection to the candidate_profiles DB."""
    settings = get_settings()
    conn = psycopg2.connect(settings.database_url, connect_timeout=15)
    conn.set_session(readonly=True, autocommit=True)
    try:
        yield conn
    finally:
        conn.close()


def ping() -> bool:
    """Cheap health probe."""
    try:
        with source_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            return cur.fetchone()[0] == 1
    except Exception:
        return False


# ============================================================
#               SIGNAL WEIGHTS (SQLite local store)
# ============================================================

import sqlite3
from datetime import datetime, timezone


_DEFAULTS: dict[str, float] = {
    "industry": 0.25,
    "function": 0.20,
    "seniority": 0.20,
    "skill_category": 0.10,
    "recency": 0.08,
    "dense": 0.09,
    "bm25": 0.03,
    "trajectory": 0.05,
}

_FIELDS = list(_DEFAULTS.keys())


def _weights_db_path() -> str:
    settings = get_settings()
    # Sit next to sessions.db in the same directory
    import os
    base = os.path.dirname(os.path.abspath(settings.sessions_db))
    return os.path.join(base, "signal_weights.db")


def _sqlite_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_weights_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_signal_weights_table() -> None:
    """Create signal_weights table in local SQLite if it doesn't exist."""
    try:
        conn = _sqlite_conn()
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signal_weights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    industry REAL NOT NULL,
                    function REAL NOT NULL,
                    seniority REAL NOT NULL,
                    skill_category REAL NOT NULL,
                    recency REAL NOT NULL,
                    dense REAL NOT NULL,
                    bm25 REAL NOT NULL,
                    trajectory REAL NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                )
                """
            )
            row = conn.execute("SELECT COUNT(*) FROM signal_weights").fetchone()
            if row[0] == 0:
                conn.execute(
                    """
                    INSERT INTO signal_weights
                      (industry, function, seniority, skill_category,
                       recency, dense, bm25, trajectory, is_active)
                    VALUES (0.25, 0.20, 0.20, 0.10, 0.08, 0.09, 0.03, 0.05, 1)
                    """
                )
        conn.close()
        _log.info("init_signal_weights_table.success")
    except Exception as exc:
        _log.error("init_signal_weights_table.error", error=str(exc))


def fetch_signal_weights() -> dict[str, float]:
    """Fetch current active signal weights from local SQLite."""
    try:
        conn = _sqlite_conn()
        row = conn.execute(
            """
            SELECT industry, function, seniority, skill_category,
                   recency, dense, bm25, trajectory
            FROM signal_weights
            WHERE is_active = 1
            ORDER BY updated_at DESC
            LIMIT 1
            """
        ).fetchone()
        conn.close()
        if row:
            return {k: float(row[k]) for k in _FIELDS}
        return dict(_DEFAULTS)
    except Exception as exc:
        _log.warning("fetch_signal_weights.error", error=str(exc))
        return dict(_DEFAULTS)


def update_signal_weights(
    industry: float,
    function: float,
    seniority: float,
    skill_category: float,
    recency: float,
    dense: float,
    bm25: float,
    trajectory: float,
    changed_by: str = "api",
) -> bool:
    """Update signal weights (deactivate old row, insert new) in local SQLite."""
    try:
        conn = _sqlite_conn()
        now = datetime.now(timezone.utc).isoformat()
        with conn:
            conn.execute("UPDATE signal_weights SET is_active = 0 WHERE is_active = 1")
            conn.execute(
                """
                INSERT INTO signal_weights
                  (industry, function, seniority, skill_category,
                   recency, dense, bm25, trajectory, is_active, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (industry, function, seniority, skill_category,
                 recency, dense, bm25, trajectory, now),
            )
        conn.close()
        _log.info("update_signal_weights.success", changed_by=changed_by)
        return True
    except Exception as exc:
        _log.error("update_signal_weights.error", error=str(exc))
        return False
