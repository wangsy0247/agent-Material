"""评测结果的可复现信息收集（不记录密钥和密码）。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SECRET_MARKERS = ("password", "secret", "token", "api_key", "apikey", "connection")
PACKAGES = (
    "langchain",
    "langchain-core",
    "langchain-postgres",
    "langgraph",
    "psycopg",
    "pgvector",
    "ragas",
    "rank-bm25",
    "sentence-transformers",
)


def sha256_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _run_git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def redact(value: Any, key: str = "") -> Any:
    lowered = key.lower()
    if any(marker in lowered for marker in SECRET_MARKERS):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def repo_relative_path(path: Path, root: Path) -> str:
    """Return a portable repository-relative path without exposing the host path."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _portable_command(root: Path) -> list[str]:
    command = ["python"]
    for argument in sys.argv:
        candidate = Path(argument)
        if candidate.is_absolute():
            command.append(repo_relative_path(candidate, root))
        else:
            command.append(argument.replace("\\", "/"))
    return command


def collect(
    *,
    root: Path,
    dataset: Path,
    config: dict,
    seed: int,
    artifacts: list[Path] | None = None,
) -> dict:
    status = _run_git(root, "status", "--porcelain")
    packages: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    artifact_hashes = {
        repo_relative_path(path, root): sha256_file(path)
        for path in artifacts or []
    }
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": _portable_command(root),
        "git_commit": _run_git(root, "rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "git_changed_file_count": len(status.splitlines()) if status else 0,
        "python_version": platform.python_version(),
        "random_seed": seed,
        "dataset": repo_relative_path(dataset, root),
        "dataset_sha256": sha256_file(dataset),
        "artifact_sha256": artifact_hashes,
        "package_versions": packages,
        "config_snapshot": redact(json.loads(json.dumps(config, default=str))),
    }
