"""Connection factory for the read-only source Postgres."""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import psycopg2
import psycopg2.extras

from app.core.config import get_settings


@contextlib.contextmanager
def source_conn() -> Iterator[Any]:
    """Yield a read-only psycopg2 connection to the candidate_profiles DB.

    Use as:
        with source_conn() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(...)
    """
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
