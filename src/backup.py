"""Backup a quiescent Compose database/data tree; restore only to a new isolated container."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .delivery import ROOT, local_version, now, run, write_json


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree_files(root: Path) -> dict[str, str]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("backup data must not contain symlinks")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = file_hash(path)
    return files


def sql(container: str, query: str) -> str:
    return run(["docker", "exec", container, "psql", "-U", "postgres", "-d", "part_rag",
                "-At", "-v", "ON_ERROR_STOP=1", "-c", query], timeout=120)


def row_counts(container: str) -> dict[str, int]:
    tables = sql(container, "SELECT tablename FROM pg_tables WHERE schemaname='partrag' ORDER BY tablename").splitlines()
    if not tables or any(not re.fullmatch(r"[a-z_][a-z_0-9]*", name) for name in tables):
        raise ValueError("no valid partrag tables to verify")
    query = " UNION ALL ".join(f"SELECT '{name}', count(*) FROM partrag.\"{name}\"" for name in tables)
    return {name: int(count) for name, count in (line.split("|") for line in sql(container, query).splitlines())}


def database_container(root: Path) -> str:
    cid = run(["docker", "compose", "ps", "-q", "db"], cwd=root)
    if not cid or len(cid.splitlines()) != 1:
        raise ValueError("exactly one running Compose db is required")
    return cid


def require_quiescent(root: Path):
    for service in ("api", "scheduler"):
        ids = run(["docker", "compose", "--profile", "scheduler", "ps", "-aq", service], cwd=root).splitlines()
        for cid in ids:
            if json.loads(run(["docker", "inspect", cid]))[0]["State"]["Running"]:
                raise ValueError("stop API/scheduler and all other DB/file writers before backup")


def attachments(container: str, data_root: Path) -> dict:
    encoded = sql(container, "SELECT coalesce(json_agg(json_build_object('profile',source_profile,"
                  "'file',local_file)), '[]'::json) FROM partrag.drawings")
    references = json.loads(encoded)
    resolved = []
    for reference in references:
        # 展示版：公司系統對接已移除（原本由來源設定模組解析每個 profile 的 PDF 目錄）
        directory = data_root / "pdf" / reference["profile"]
        path = (directory / reference["file"]).resolve()
        if not path.is_relative_to(data_root.resolve()) or not path.is_file():
            raise ValueError("a referenced PDF is missing or outside data/; include it before creating the backup")
        resolved.append(path.relative_to(data_root.resolve()).as_posix())
    return {"referenced_files": sorted(set(resolved)), "drawing_records": len(references)}


def create(output: Path, root: Path = ROOT) -> dict:
    output = output.resolve()
    data = (root / "data").resolve()
    if output == root.resolve() or output.is_relative_to(data) or output.exists():
        raise ValueError("backup output must be a new directory outside data/")
    require_quiescent(root)
    cid = database_container(root)
    before = row_counts(cid)
    references = attachments(cid, data)
    inventory = tree_files(data)
    output.mkdir(parents=True)
    # A failed attempt stays as an incomplete directory: no success manifest is written.
    dump = output / "database.dump"
    with dump.open("wb") as stream:
        result = subprocess.run(["docker", "exec", cid, "pg_dump", "-U", "postgres", "-d", "part_rag", "-Fc"],
                                stdout=stream, stderr=subprocess.PIPE, timeout=600)
    if result.returncode:
        raise RuntimeError("pg_dump failed; backup is incomplete")
    for name in inventory:
        target = output / "data" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(data / name, target)
    if inventory != tree_files(data) or inventory != tree_files(output / "data") or before != row_counts(cid):
        raise ValueError("source changed during backup; keep writers stopped and retry to a new directory")
    require_quiescent(root)
    image_id = json.loads(run(["docker", "inspect", cid]))[0]["Image"]
    image = json.loads(run(["docker", "image", "inspect", image_id]))[0]
    # Secret-bearing .env is deliberately outside this artifact.
    # 展示版：設定範本檔已移除，備份不再附帶 env 範本。
    manifest = {"format": 1, "created_at": now(), "database_image_id": image_id,
                "database_repo_digests": image.get("RepoDigests", []),
                "database_sha256": file_hash(dump), "tables": before, "data_files": inventory,
                "attachments": references, "version": local_version(root)}
    write_json(output / "manifest.json", manifest)
    verify(output)
    return manifest


def verify(directory: Path) -> dict:
    if directory.is_symlink():
        raise ValueError("backup directory must not be a symlink")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != 1:
        raise ValueError("unsupported backup format")
    for name in ("database.dump", "data"):
        if (directory / name).is_symlink():
            raise ValueError("backup contains a symlink")
    if file_hash(directory / "database.dump") != manifest["database_sha256"]:
        raise ValueError("database dump checksum mismatch")
    if tree_files(directory / "data") != manifest["data_files"]:
        raise ValueError("attachment/data inventory differs from backup manifest")
    for name in manifest["attachments"]["referenced_files"]:
        if name not in manifest["data_files"]:
            raise ValueError("missing referenced attachment")
        with (directory / "data" / name).open("rb") as stream:
            if b"%PDF-" not in stream.read(1024):
                raise ValueError("referenced file is not a PDF")
    return manifest


def drill(directory: Path) -> dict:
    manifest = verify(directory)
    image = manifest["database_image_id"]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("invalid database image identity")
    # Exact local image; no pulls, ports, existing DB targets or initialization SQL.
    run(["docker", "image", "inspect", image])
    started = time.monotonic()
    cid = run(["docker", "run", "--detach", "--rm", "--pull=never", "--network=none",
               "--name", "partrag-restore-drill-" + secrets.token_hex(6),
               "--env", "POSTGRES_PASSWORD=" + secrets.token_hex(24), "--env", "POSTGRES_DB=part_rag", image])
    try:
        deadline = time.monotonic() + 45
        while True:
            try:
                # initdb uses a temporary Unix-socket-only server. Wait for final TCP readiness.
                run(["docker", "exec", cid, "pg_isready", "-h", "127.0.0.1", "-U", "postgres", "-d", "part_rag"])
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("restore drill database did not start")
                time.sleep(.25)
        with (directory / "database.dump").open("rb") as stream:
            result = subprocess.run(["docker", "exec", "-i", cid, "pg_restore", "-U", "postgres", "-d", "part_rag",
                                     "--exit-on-error", "--single-transaction", "--no-owner", "--no-privileges"],
                                    stdin=stream, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        if result.returncode:
            raise RuntimeError("pg_restore failed; restore drill did not pass")
        if row_counts(cid) != manifest["tables"]:
            raise ValueError("restored table counts differ from backup")
        # Exercise the primary key lookup, keeping private content out of the report.
        count = sql(cid, "SELECT count(*) FROM partrag.parts WHERE item_number = "
                    "(SELECT item_number FROM partrag.parts ORDER BY item_number LIMIT 1)")
        if manifest["tables"].get("parts", 0) and count != "1":
            raise ValueError("representative part lookup failed")
        verify(directory)
        report = {"status": "passed", "verified_at": now(), "seconds": round(time.monotonic() - started, 2),
                  "tables": manifest["tables"], "data_files": len(manifest["data_files"]),
                  "referenced_pdfs": len(manifest["attachments"]["referenced_files"]),
                  "database_image_id": image}
        write_json(directory / "restore-check.json", report)
        return report
    finally:
        # Only this freshly created container is stopped; --rm also removes its anonymous volume.
        run(["docker", "stop", "--time", "5", cid])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "verify", "drill"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "create": result = create(args.directory)
    elif args.command == "verify":
        manifest = verify(args.directory)
        result = {"status": "verified", "data_files": len(manifest["data_files"])}
    else: result = drill(args.directory)
    print(json.dumps({k: v for k, v in result.items() if k not in {"data_files", "version"}}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"backup: {exc}", file=sys.stderr)
        raise SystemExit(1)
