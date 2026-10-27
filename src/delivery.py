"""Pinned build inputs and release evidence. Never serialize runtime secrets."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = "vendor/agentkit.lock.json"
BUILD = "build-info.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_digest(path: Path) -> str:
    return digest(path.read_bytes().replace(b"\r\n", b"\n"))


def json_digest(value) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run(args, *, cwd=None, timeout=30) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                            encoding="utf-8", timeout=timeout)
    if result.returncode:
        # Docker/DB errors can contain environment values. Show the operation only.
        raise RuntimeError(f"command failed ({result.returncode}): {args[0]}")
    return result.stdout.strip()


def git_revision(root: Path) -> str | None:
    try:
        return run(["git", "rev-parse", "HEAD"], cwd=root)
    except (OSError, RuntimeError):
        return None


def package_files(package: Path) -> dict[str, str]:
    files = {}
    for path in sorted(package.rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError("agentkit symlinks are not supported")
        if path.is_file():
            files[path.relative_to(package).as_posix()] = text_digest(path)
    if "__init__.py" not in files or "envelope.py" not in files:
        raise ValueError("incomplete agentkit package")
    return files


def validate_graph(package: Path):
    tree = ast.parse((package / "registry.py").read_text(encoding="utf-8"))
    graph = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "AgentRef":
            values = {k.arg: k.value for k in node.keywords}
            name = ast.literal_eval(values["agent_id"])
            if name in graph:
                raise ValueError("duplicate registry agent: " + name)
            graph[name] = tuple(ast.literal_eval(values["downstream"])) if "downstream" in values else ()
    if not graph:
        raise ValueError("registry contains no agents")
    visiting, visited = set(), set()

    def visit(name):
        if name not in graph:
            raise ValueError("unknown downstream agent: " + name)
        if name in visiting:
            raise ValueError("registry delegation cycle: " + name)
        if name in visited:
            return
        visiting.add(name)
        for peer in graph[name]:
            visit(peer)
        visiting.remove(name)
        visited.add(name)
    for name in graph:
        visit(name)


def verify_vendor(root: Path = ROOT, package: Path | None = None) -> dict:
    lock = json.loads((root / LOCK).read_text(encoding="utf-8"))
    actual = package_files(package or root / "vendor/agentkit")
    if actual != lock["files"] or json_digest(actual) != lock["sha256"]:
        raise ValueError("agentkit differs from the pinned snapshot; run vendor-pin and review it")
    validate_graph(package or root / "vendor/agentkit")
    return lock


def pin_vendor(source: Path, root: Path = ROOT) -> dict:
    source = source.resolve()
    files = package_files(source)
    validate_graph(source)
    tree = ast.parse((source / "envelope.py").read_text(encoding="utf-8"))
    protocol = next(ast.literal_eval(n.value) for n in tree.body
                    if isinstance(n, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == "PROTOCOL" for t in n.targets))
    if protocol != "a2a/1":
        raise ValueError("unsupported A2A protocol")
    destination = (root / "vendor/agentkit").resolve()
    if source == destination or not destination.is_relative_to(root.resolve() / "vendor"):
        raise ValueError("invalid vendor destination")
    # Only replace files within this exact package directory. Never alter the source.
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise ValueError("vendor destination contains a symlink")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            if path.relative_to(destination).as_posix() not in files:
                path.unlink()
    for name in files:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source / name).read_bytes().replace(b"\r\n", b"\n"))
    lock = {"format": 1, "protocol": protocol, "source_commit": git_revision(source.parent),
            "sha256": json_digest(files), "files": files}
    write_json(root / LOCK, lock)
    return verify_vendor(root)


def build_files(root: Path = ROOT) -> dict[str, str]:
    paths = [root / p for p in ("main.py", "Dockerfile", "requirements-api.txt",
             "requirements.txt", "requirements-api-linux.lock", "requirements-worker-linux.lock", LOCK)]
    for directory in ("src", "vendor/agentkit"):
        paths += [p for p in (root / directory).rglob("*") if p.is_file()
                  and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    return {p.relative_to(root).as_posix(): text_digest(p) for p in sorted(set(paths))}


def dependency_lock(path: Path) -> dict[str, str]:
    versions = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line)
        if not match:
            raise ValueError("invalid Linux dependency lock")
        versions[re.sub(r"[-_.]+", "-", match[1]).lower()] = match[2]
    return versions


def check_dependencies(root: Path, target: str | None = None):
    api = dependency_lock(root / "requirements-api-linux.lock")
    worker = dependency_lock(root / "requirements-worker-linux.lock")
    if any(worker.get(name) != ver for name, ver in api.items()):
        raise ValueError("API/worker shared dependency versions differ")
    if target:
        expected = api if target == "api" else worker
        installed = {re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower(): d.version
                     for d in importlib.metadata.distributions()
                     if d.metadata["Name"].lower() not in {"pip", "setuptools", "wheel"}}
        if installed != expected:
            raise ValueError("installed Python dependencies differ from the complete Linux lock")


def prepare(root: Path = ROOT) -> dict:
    lock = verify_vendor(root)
    check_dependencies(root)
    files = build_files(root)
    revision = git_revision(root)
    dirty = None
    if revision:
        dirty = bool(run(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root))
    info = {"format": 1, "prepared_at": now(), "app_commit": revision, "source_dirty": dirty,
            "source_sha256": json_digest(files), "files": files,
            "agentkit": {k: lock[k] for k in ("source_commit", "sha256", "protocol")}}
    write_json(root / BUILD, info)
    return info


def verify_build(root: Path = ROOT, target: str | None = None) -> dict:
    lock = verify_vendor(root)
    check_dependencies(root, target)
    info = json.loads((root / BUILD).read_text(encoding="utf-8"))
    files = build_files(root)
    if files != info["files"] or json_digest(files) != info["source_sha256"]:
        raise ValueError("build inputs changed after prepare; run python -m src.delivery prepare")
    if info["agentkit"]["sha256"] != lock["sha256"]:
        raise ValueError("build agentkit identity mismatch")
    if target:
        info["target"] = target
        info["python"] = platform.python_version()
        info["packages"] = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
        write_json(root / BUILD, info)
    return info


def local_version(root: Path = ROOT) -> dict:
    lock = verify_vendor(root)
    if (root / BUILD).exists():
        info = verify_build(root)
    else:
        files = build_files(root)
        info = {"app_commit": git_revision(root), "source_sha256": json_digest(files),
                "source_dirty": True, "target": "development"}
    result = {k: info.get(k) for k in ("app_commit", "source_sha256", "source_dirty", "target",
                                     "prepared_at", "python")}
    result["agentkit"] = {k: lock[k] for k in ("source_commit", "sha256", "protocol")}
    module = sys.modules.get("agentkit")
    if module is not None:
        verify_vendor(root, Path(module.__file__).parent)
    # Defaults come from the same module AST, without importing the model/DB pool.
    module = ast.parse((root / "src/retrieval.py").read_text(encoding="utf-8"))
    defaults = {}
    for n in module.body:
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            for name in n.targets:
                if isinstance(name, ast.Name) and name.id in {"EMBED_MODEL", "EMBED_REVISION"}:
                    defaults[name.id] = ast.literal_eval(n.value.args[1])
    result["embedding"] = {"model": os.getenv("EMBED_MODEL", defaults["EMBED_MODEL"]),
                           "revision": os.getenv("EMBED_REVISION", defaults["EMBED_REVISION"]),
                           "max_seq": int(os.getenv("EMBED_MAX_SEQ", "256"))}
    result["image_id"] = None  # Actual Docker identities are collected by release(), outside the image.
    return result


def version(root: Path = ROOT, database: bool = False) -> dict:
    result = local_version(root)
    if database:
        from . import migrate
        result["schema"] = migrate.database_status(root=root)
    return result


def release(output: Path, services: list[str], root: Path = ROOT) -> dict:
    evidence = {"format": 1, "recorded_at": now(), "services": {}}
    identities = []
    for service in services:
        if service not in {"api", "scheduler"}:
            raise ValueError("unsupported release service")
        cid = run(["docker", "compose", "--profile", "scheduler", "ps", "-aq", service], cwd=root)
        if not cid or len(cid.splitlines()) != 1:
            raise ValueError(f"missing or ambiguous service: {service}")
        container = json.loads(run(["docker", "inspect", cid]))[0]
        if not container["State"]["Running"]:
            raise ValueError(f"service is stopped: {service}")
        image_id = container["Image"]
        image = json.loads(run(["docker", "image", "inspect", image_id]))[0]
        built = json.loads(run(["docker", "exec", cid, "python", "-m", "src.delivery", "version", "--database"]))
        if built["target"] != ("api" if service == "api" else "worker"):
            raise ValueError(f"unexpected image target: {service}")
        if built["schema"]["status"] not in {"current", "ahead"}:
            raise ValueError(f"schema history is incomplete: {service}")
        identities.append((built["source_sha256"], built["agentkit"]["sha256"],
                           built["embedding"]["model"], built["embedding"]["revision"],
                           built["embedding"]["max_seq"]))
        evidence["services"][service] = {"image_id": image_id,
                                        "repo_digests": image.get("RepoDigests", []), "version": built}
    if len(set(identities)) != 1:
        raise ValueError("API and scheduler are running different source/model/agentkit versions")
    write_json(output, evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pin = commands.add_parser("vendor-pin")
    pin.add_argument("--source", type=Path, required=True)
    commands.add_parser("vendor-check")
    commands.add_parser("prepare")
    check = commands.add_parser("verify-build")
    check.add_argument("--target", choices=("api", "worker"))
    ver = commands.add_parser("version")
    ver.add_argument("--database", action="store_true")
    rel = commands.add_parser("release")
    rel.add_argument("--output", type=Path, required=True)
    rel.add_argument("--services", default="api")
    args = parser.parse_args()
    if args.command == "vendor-pin": result = pin_vendor(args.source)
    elif args.command == "vendor-check": result = verify_vendor()
    elif args.command == "prepare": result = prepare()
    elif args.command == "verify-build": result = verify_build(target=args.target)
    elif args.command == "version": result = version(database=args.database)
    else: result = release(args.output, args.services.split(","))
    # Full file inventories are on disk, not in normal CLI output.
    print(json.dumps({k: v for k, v in result.items() if k not in {"files", "packages"}}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"delivery: {exc}", file=sys.stderr)
        raise SystemExit(1)
