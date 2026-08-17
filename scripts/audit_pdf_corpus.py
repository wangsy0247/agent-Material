"""只读审计 PDF 语料的文本层与 OCR 路由分布。

用法：
    python scripts/audit_pdf_corpus.py data
    python scripts/audit_pdf_corpus.py data --json output/pdf_audit.json

脚本不会调用 MinerU、不会写入数据库，也不会修改 PDF。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "backend" / "packages"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from agent_material.rag_engine.mineru_loader import inspect_pdf  # noqa: E402


def audit(directory: Path) -> dict:
    files = sorted(directory.rglob("*.pdf"))
    items = []
    modes: Counter[str] = Counter()
    for path in files:
        try:
            profile = inspect_pdf(str(path))
            modes[profile.pdf_mode] += 1
            items.append(
                {
                    "path": str(path.resolve()),
                    **profile.to_metadata(),
                    "error": "",
                }
            )
        except Exception as exc:
            modes["invalid"] += 1
            items.append(
                {
                    "path": str(path.resolve()),
                    "total_pages": 0,
                    "text_layer_pages": 0,
                    "image_only_pages": 0,
                    "inspection_failed_pages": 0,
                    "pdf_mode": "invalid",
                    "ocr_required": False,
                    "error": str(exc),
                }
            )
    return {
        "directory": str(directory.resolve()),
        "total_files": len(items),
        "modes": dict(sorted(modes.items())),
        "ocr_required_files": sum(item["ocr_required"] for item in items),
        "inspection_failed_pages": sum(
            item["inspection_failed_pages"] for item in items
        ),
        "items": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="待审计的 PDF 根目录")
    parser.add_argument("--json", type=Path, dest="json_path", help="可选 JSON 输出")
    args = parser.parse_args()

    if not args.directory.is_dir():
        parser.error(f"目录不存在: {args.directory}")

    report = audit(args.directory)
    print(
        f"PDF files={report['total_files']} modes={report['modes']} "
        f"ocr_required={report['ocr_required_files']} "
        f"inspection_failed_pages={report['inspection_failed_pages']}"
    )
    for item in report["items"]:
        if item["pdf_mode"] != "native":
            print(
                f"[{item['pdf_mode']}] {item['path']} | "
                f"pages={item['total_pages']} text={item['text_layer_pages']} "
                f"image_only={item['image_only_pages']} "
                f"failed={item['inspection_failed_pages']} "
                f"error={item['error']}"
            )

    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
