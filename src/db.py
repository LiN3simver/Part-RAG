"""Shared psycopg connection pool.

Lifted out of retrieval.py so src/values.py can query the corpus without
importing retrieval (retrieval imports values, which would be a cycle).
`retrieval.pool` is still the name the rest of the code uses -- it re-exports
this one.
"""
from __future__ import annotations

import os
import threading

from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

load_dotenv()
PG_DSN = os.environ["PG_DSN"]

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()
_api = False


def configure_api() -> None:
    """API budgets do not apply to long-running ingest/scheduler commands."""
    global _api
    if _pool is not None:
        raise RuntimeError("configure API budgets before opening the database pool")
    _api = True


def close() -> None:
    global _pool
    with _pool_lock:
        current, _pool = _pool, None
    if current is not None:
        current.close()


def stats() -> dict:
    """Do not open a DB connection to serve operational status."""
    current = _pool
    return {"initialized": current is not None,
            **(current.get_stats() if current is not None else {})}


def pool() -> ConnectionPool:
    """Postgres 連線池。第一次呼叫才建立。

    ★ 為什麼要 double-checked locking
      route handler 是同步 `def`，FastAPI 會把它們丟進 threadpool 跑 —— 服務
      剛起來、還沒有人碰過 DB 的時候，同時進來的兩個請求會**各建一個
      ConnectionPool**：一個留在全域變數上，另一個變成孤兒，帶著它已經開好的
      連線一起洩漏（max_size=8）。不會有錯誤訊息，只有 Postgres 那邊多出一批
      沒人用的連線。

      鎖只包建構 —— 池子建好之後第一個 if 就回去了，正常請求不會碰到鎖。
      RFQ-RAG 的 src/db.py 是同一個寫法（那邊先踩到的）。
    """
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is not None:       # 等鎖的期間別人已經建好了
            return _pool
        kwargs = {"connect_timeout": int(os.getenv("PG_CONNECT_TIMEOUT_SECONDS", "5"))}
        if _api:
            statement = int(os.getenv("PG_API_STATEMENT_TIMEOUT_MS", "15000"))
            lock = int(os.getenv("PG_API_LOCK_TIMEOUT_MS", "2000"))
            if statement <= 0 or lock <= 0:
                raise ValueError("API database timeouts must be positive")
            kwargs["options"] = f"-c statement_timeout={statement} -c lock_timeout={lock}"
        _pool = ConnectionPool(
            PG_DSN, min_size=1, max_size=int(os.getenv("PG_POOL_MAX_SIZE", "8")),
            timeout=float(os.getenv("PG_POOL_WAIT_SECONDS", "5")),
            max_waiting=int(os.getenv("PG_POOL_MAX_WAITING", "16")),
            kwargs=kwargs, configure=register_vector, open=True
        )
    return _pool
