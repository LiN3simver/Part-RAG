"""One native conversation turn at a time across API processes.

Negative bigint advisory keys are reserved for conversation IDs; migration and
scheduler keys are positive. A dedicated session owns the lock and both message
writes. Closing it releases the lock, including on process failure, without
holding an idle transaction or consuming a retrieval-pool connection.
"""
from __future__ import annotations

import os

import psycopg
from fastapi import HTTPException

from . import db
from .request_budget import checkpoint, current


class ConversationTurn:
    def __init__(self, connection, conv_id, owner_id):
        self.connection = connection
        self.conv_id, self.owner_id = conv_id, owner_id

    def add_message(self, role, content, payload=None):
        from .chat import _insert_message
        # The writes use the lock-owning session. A lost session cannot later
        # reconnect through the pool and append a reply to another turn.
        with self.connection.transaction():
            return _insert_message(self.connection, self.conv_id, role, content,
                                   payload, owner_id=self.owner_id)

    def repair_pending(self):
        """Pair an unfinished user turn after the previous session disappeared."""
        from .chat import _insert_message
        with self.connection.transaction():
            row = self.connection.execute(
                "SELECT role FROM partrag.messages WHERE conv_id=%s ORDER BY id DESC LIMIT 1",
                (self.conv_id,)).fetchone()
            if row and row[0] == "user":
                _insert_message(self.connection, self.conv_id, "assistant", "",
                                {"generation": {"status": "interrupted"}}, owner_id=self.owner_id)

    def delete(self):
        # Deletion is fenced by the same session as the lock, just like writes.
        with self.connection.transaction():
            result = self.connection.execute(
                "DELETE FROM partrag.conversations WHERE id=%s AND owner_id=%s",
                (self.conv_id, self.owner_id))
            return result.rowcount > 0

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def claim_turn(conv_id: int, *, owner_id: str) -> ConversationTurn:
    checkpoint()
    if conv_id <= 0:
        raise HTTPException(404, f"找不到對話 {conv_id}")
    timeout = int(os.getenv("PG_CONNECT_TIMEOUT_SECONDS", "5"))
    if timeout <= 0:
        raise ValueError("database connect timeout must be positive")
    deadline = current.get()
    if deadline is not None:
        timeout = max(1, min(timeout, int(deadline.remaining())))
    conn = psycopg.connect(db.PG_DSN, autocommit=True, connect_timeout=timeout,
                           options="-c statement_timeout=2000 -c lock_timeout=2000")
    try:
        checkpoint()
        sql = "SELECT 1 FROM partrag.conversations WHERE id=%s AND owner_id=%s"
        if not conn.execute(sql, (conv_id, owner_id)).fetchone():
            raise HTTPException(404, f"找不到對話 {conv_id}")
        if not conn.execute("SELECT pg_try_advisory_lock(%s)", (-conv_id,)).fetchone()[0]:
            raise HTTPException(409, "這段對話正在回答上一個問題，請稍後再送出。",
                                headers={"Retry-After": "1"})
        # A guarded delete could finish between the first ownership lookup and
        # this acquisition. Check again before any message is written.
        if not conn.execute(sql, (conv_id, owner_id)).fetchone():
            raise HTTPException(404, f"找不到對話 {conv_id}")
        checkpoint()
        return ConversationTurn(conn, conv_id, owner_id)
    except BaseException:
        conn.close()
        raise
