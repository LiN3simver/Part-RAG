"""Readiness checks schema and real data, independently of the query pool/LLM."""
from __future__ import annotations

import os

import psycopg
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from . import db
from .probes import Probe

REQUIRED_COLUMNS = {
    "parts": {"item_number", "name", "embedding", "is_obsolete", "search_text"},
    "drawings": {"frame", "title_fields", "source_profile"},
    "drawing_items": {"table_tag", "unit", "remark", "color", "length_raw"},
    "conversations": {"owner_id"},
    "messages": {"conv_id"},
    "sync_state": {"stage", "last_success", "rows_failed", "rows_skipped", "last_finished"},
    "lessons": {"key"},
    "eval_runs": {"run_id"},
    "eval_cases": {"case_id"},
}


def check_database():
    # A reserved, short-lived connection avoids waiting behind user queries.
    with psycopg.connect(db.PG_DSN, connect_timeout=3, autocommit=True,
                         options="-c statement_timeout=2000 -c lock_timeout=1000") as conn:
        rows = conn.execute("SELECT table_name, column_name FROM information_schema.columns "
                            "WHERE table_schema = 'partrag'").fetchall()
        present: dict[str, set[str]] = {}
        for table, column in rows:
            present.setdefault(table, set()).add(column)
        missing = {table: sorted(columns - present.get(table, set()))
                   for table, columns in REQUIRED_COLUMNS.items()
                   if columns - present.get(table, set())}
        if missing:
            return "not_ready", {"db": True, "schema": False, "missing": missing}
        has_parts = conn.execute("SELECT EXISTS (SELECT 1 FROM partrag.parts)").fetchone()[0]
        if not has_parts:
            return "not_ready", {"db": True, "schema": True, "corpus": False}
    return "ready", {"db": True, "schema": True, "corpus": True}


def mount(app: FastAPI):
    probe = Probe(check_database, ttl=float(os.getenv("READINESS_TTL_SECONDS", "10")), wait=1)
    app.state.readiness_probe = probe

    def check_version():
        from .delivery import version
        try:
            return "ok", version(database=True)
        except Exception as exc:
            return "unavailable", {"error_type": type(exc).__name__}

    version_probe = Probe(check_version, ttl=10, wait=1)
    app.state.version_probe = version_probe

    @app.get("/version")
    async def service_version():
        status, detail = await version_probe.aget()
        return JSONResponse({"status": status, **detail}, status_code=200 if status == "ok" else 503)

    @app.get("/livez", include_in_schema=False)
    async def livez():
        return {"status": "alive"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz():
        from . import retrieval
        status, detail = await probe.aget()
        mounted = bool(getattr(app.state, "a2a_mounted", False))
        if not mounted:
            status = "not_ready"
        return JSONResponse({"status": status, "detail": {**detail, "a2a": mounted},
                             "capabilities": {"retrieval": status == "ready",
                                              "vector": retrieval.embedding_ready()}},
                            status_code=200 if status == "ready" else 503)
