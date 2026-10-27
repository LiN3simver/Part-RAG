"""Explicit, transactional upgrades. The base schema file is never an upgrade input."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import psycopg

from .delivery import ROOT, git_revision, json_digest, text_digest

# Same order as Compose initialization; the base schema remains initialization only.
# 展示版：實際的 SQL migration 檔未收錄，以下檔名僅示意 runner 的用法（依序套用、記錄 checksum）。
MIGRATIONS = (
    "migrate_0001_example.sql", "migrate_0002_example.sql",
)
HISTORY_SQL = """
CREATE TABLE IF NOT EXISTS partrag.schema_migrations (
    name text PRIMARY KEY, sha256 text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now(),
    applied_by text NOT NULL DEFAULT current_user, app_commit text
)
"""


def catalog(root: Path = ROOT) -> dict[str, str]:
    return {name: text_digest(root / name) for name in MIGRATIONS}


def status(conn, root: Path = ROOT) -> dict:
    expected = catalog(root)
    exists = conn.execute("SELECT to_regclass('partrag.schema_migrations')").fetchone()[0]
    records = conn.execute("SELECT name, sha256 FROM partrag.schema_migrations ORDER BY name").fetchall() if exists else []
    recorded = dict(records)
    changed = [name for name in expected if name in recorded and expected[name] != recorded[name]]
    pending = [name for name in expected if name not in recorded]
    extra = sorted(set(recorded) - set(expected))
    state = "drift" if changed else "pending" if pending else "ahead" if extra else "current"
    return {"status": state, "applied": len(recorded), "required": len(expected),
            "pending": pending, "changed": changed, "extra": extra,
            "revision": json_digest(expected) if not pending and not changed else None}


def connection(dsn: str | None = None, *, write: bool = False):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    configured = dsn or os.getenv("PG_DSN")
    if not configured:
        raise ValueError("PG_DSN is required")
    return psycopg.connect(configured, autocommit=True, connect_timeout=3,
                           options=f"-c statement_timeout={'120000' if write else '2000'} -c lock_timeout=5000")


def database_status(dsn: str | None = None, root: Path = ROOT) -> dict:
    with connection(dsn) as conn:
        return status(conn, root)


def migration_sql(path: Path) -> str:
    sql = path.read_text(encoding="utf-8")
    # Some legacy scripts have their own outer BEGIN/COMMIT. The runner owns the transaction.
    sql = re.sub(r"(?m)^(?:BEGIN|COMMIT);[ \t]*$", "", sql)
    if re.search(r"(?im)^\s*(?:COMMIT|ROLLBACK|BEGIN\s*;)", sql):
        raise ValueError(f"unsupported transaction control in {path.name}")
    return sql


def apply(dsn: str | None = None, root: Path = ROOT) -> dict:
    with connection(dsn, write=True) as conn, conn.transaction():
        if not conn.execute("SELECT pg_try_advisory_xact_lock(72724720261002)").fetchone()[0]:
            raise ValueError("another schema upgrade is running")
        if not conn.execute("SELECT to_regclass('partrag.parts')").fetchone()[0]:
            raise ValueError("base schema missing; initialize a NEW database first, never replay the base schema on existing data")
        before = status(conn, root)
        if before["changed"]:
            raise ValueError("applied migration checksum changed: " + ", ".join(before["changed"]))
        conn.execute(HISTORY_SQL)
        expected = catalog(root)
        revision = git_revision(root)
        if revision is None and (root / "build-info.json").exists():
            revision = json.loads((root / "build-info.json").read_text(encoding="utf-8")).get("app_commit")
        for name in before["pending"]:
            conn.execute(migration_sql(root / name))
            conn.execute("INSERT INTO partrag.schema_migrations(name, sha256, app_commit) VALUES (%s,%s,%s)",
                         (name, expected[name], revision))
        return {**status(conn, root), "newly_applied": before["pending"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "apply"))
    args = parser.parse_args()
    result = apply() if args.command == "apply" else database_status()
    print(json.dumps(result, ensure_ascii=False))
    if args.command == "status" and result["status"] not in {"current", "ahead"}:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except (psycopg.Error, OSError, ValueError) as exc:
        # DSN and PostgreSQL connection diagnostics must not leak credentials.
        message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        print("migration: " + message, file=sys.stderr)
        raise SystemExit(1)
