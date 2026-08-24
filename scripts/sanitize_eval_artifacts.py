#!/usr/bin/env python3
"""清除公开评测 JSON 中的本机路径和主机环境信息。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = ROOT / "evaluation_data"
LOCAL_VALUE = "<local-value-removed>"
REDACTED_PATH = "<redacted-local-path>"


def _local_path_pattern() -> re.Pattern[str]:
    roots = {str(ROOT.resolve()), str(Path.home().resolve())}
    root_posix = ROOT.resolve().as_posix()
    wsl_match = re.match(r"^/mnt/([a-z])/(.+)$", root_posix, flags=re.IGNORECASE)
    if wsl_match:
        roots.add(f"{wsl_match.group(1)}:/{wsl_match.group(2)}")
    alternatives = []
    for root in sorted(roots, key=len, reverse=True):
        normalized = root.replace("\\", "/").rstrip("/")
        alternatives.append(re.escape(normalized).replace("/", r"[\\/]"))
    suffix = r"(?:[\\/][^\s\"'<>，。；、（）\[\]{}]*)?"
    return re.compile(rf"(?i)(?:{'|'.join(alternatives)}){suffix}")


LOCAL_PATH_PATTERN = _local_path_pattern()
PORTABLE_MARKERS = (
    "evaluation_data/",
    "scripts/",
    "logs/",
    "config/",
)
BANNED_REPRO_KEYS = {
    "working_directory",
    "platform",
    "wsl_distro",
    "conda_environment",
}
CANONICAL_PATHS = {
    "ablation_results_v2.json": {
        "dataset": "evaluation_data/fixed_eval_dataset_v2.json",
    },
    "weight_sweep_results_v1.json": {
        "dataset": "evaluation_data/fixed_eval_dataset_v2.json",
    },
    "ragas_v3_scores.json": {
        "dataset": "evaluation_data/fixed_eval_dataset_v2.json",
        "filled": "evaluation_data/ragas_v3_filled.json",
    },
}


def _portable_text(value: str) -> str:
    normalized = value.replace("\\", "/")
    lowered = normalized.lower()
    if not LOCAL_PATH_PATTERN.search(normalized):
        return value
    if LOCAL_PATH_PATTERN.match(normalized.lstrip()):
        for marker in PORTABLE_MARKERS:
            index = lowered.find(marker)
            if index >= 0:
                return normalized[index:]
    return LOCAL_PATH_PATTERN.sub(REDACTED_PATH, value)


def _sanitize(node: Any) -> Any:
    if isinstance(node, dict):
        result = {str(key): _sanitize(value) for key, value in node.items()}
        for key in ("source", "target_source"):
            if result.get(key) in {LOCAL_VALUE, REDACTED_PATH}:
                result[key] = "abacus.pdf"
        return result
    if isinstance(node, list):
        return [_sanitize(value) for value in node]
    if isinstance(node, str):
        return _portable_text(node)
    return node


def _sanitize_reproducibility(repro: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: value for key, value in repro.items() if key not in BANNED_REPRO_KEYS
    }
    python = result.pop("python", None)
    if "python_version" not in result and isinstance(python, str):
        result["python_version"] = python.split()[0]
    command = result.get("command")
    if isinstance(command, list) and command:
        result["command"] = ["python", *command[1:]]
    return _sanitize(result)


def sanitize_file(path: Path, *, check: bool) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    data = _sanitize(data)
    if isinstance(data, dict):
        repro = data.get("reproducibility")
        if isinstance(repro, dict):
            data["reproducibility"] = _sanitize_reproducibility(repro)
        for key, value in CANONICAL_PATHS.get(path.name, {}).items():
            data[key] = value
        data.pop("log_file", None)

    rendered = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    current = path.read_text(encoding="utf-8")
    changed = rendered != current
    if changed and not check:
        path.write_text(rendered, encoding="utf-8")
    return changed


def _privacy_issues(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rendered = json.dumps(data, ensure_ascii=False)
    issues: list[str] = []
    if LOCAL_PATH_PATTERN.search(rendered):
        issues.append("local path or identifier")
    if LOCAL_VALUE in rendered:
        issues.append("unresolved privacy placeholder")
    repro = data.get("reproducibility") if isinstance(data, dict) else None
    if isinstance(repro, dict):
        remaining = BANNED_REPRO_KEYS & set(repro)
        if remaining:
            issues.append(f"banned reproducibility keys: {sorted(remaining)}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    files = sorted(args.directory.rglob("*.json"))
    changed = [path for path in files if sanitize_file(path, check=args.check)]
    issues = {
        path.relative_to(ROOT).as_posix(): found
        for path in files
        if (found := _privacy_issues(path))
    }
    if issues:
        for path, found in issues.items():
            print(f"{path}: {', '.join(found)}")
        return 1
    if args.check and changed:
        for path in changed:
            print(f"需要重新清洗: {path.relative_to(ROOT).as_posix()}")
        return 1
    print(f"隐私检查通过: {len(files)} 个 JSON 文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
