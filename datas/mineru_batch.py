#!/usr/bin/env python3
"""
PDF 按 20 页切分 → MinerU Flash 并发批处理 → Markdown 合并

原理:
  - Flash 模式限制单文件 ≤ 20 页 / ≤ 10MB，大 PDF 先切分为 20 页/份
  - 切分后使用 ThreadPoolExecutor 并发调用 flash_extract
  - 全部完成后按页码顺序合并为单个 Markdown 文件

依赖: pip install pypdf mineru-open-sdk

用法: python mineru_batch.py input.pdf [output.md] [--workers 3]
      python mineru_batch.py paper.pdf result.md --workers 5
"""

from __future__ import annotations

import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pypdf import PdfReader, PdfWriter

CHUNK_PAGES = 20
DEFAULT_WORKERS = 3


# ─── PDF 切分 ────────────────────────────────────────────────────────────────


def split_pdf(pdf_path: str, chunk_size: int = CHUNK_PAGES) -> list[tuple[Path, int, int]]:
    """
    将 PDF 按每 chunk_size 页切分成多个临时文件。

    Returns:
        [(临时文件路径, 起始页码, 结束页码), ...]，页码从 1 开始
    """
    reader = PdfReader(pdf_path)
    total = len(reader.pages)
    temp_dir = Path(tempfile.mkdtemp(prefix="mineru_batch_"))
    chunks: list[tuple[Path, int, int]] = []

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])

        path = temp_dir / f"part_{start // chunk_size + 1:03d}_p{start + 1}-{end}.pdf"
        writer.write(str(path))
        chunks.append((path, start + 1, end))

    reader.close()
    return chunks


# ─── 并发批处理 (含重试) ──────────────────────────────────────────────────────


def _parse_one(
    path: Path, start: int, end: int, idx: int, total: int, max_retries: int = 3
) -> tuple[int, int, int, str | None]:
    """单个切片的 flash_extract 调用，含指数退避重试。"""
    from mineru import MinerU

    mb = path.stat().st_size / (1024 * 1024)
    header = f"[{idx}/{total}] p{start}-{end} ({mb:.1f}MB)"

    for attempt in range(max_retries + 1):
        try:
            client = MinerU()
            result = client.flash_extract(str(path))
        except Exception as exc:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  {header} ⚠️ 异常重试({attempt + 1}/{max_retries}) {wait}s后: {exc}")
                time.sleep(wait)
                continue
            print(f"  {header} ❌ 异常: {exc}")
            return (idx, start, end, None)

        if result.state == "done" and result.markdown is not None:
            print(f"  {header} ✅ {len(result.markdown)} 字符")
            return (idx, start, end, result.markdown)

        if attempt < max_retries:
            wait = 2 ** attempt
            print(f"  {header} ⚠️ state={result.state} 重试({attempt + 1}/{max_retries}) {wait}s后")
            time.sleep(wait)
            continue

        print(f"  {header} ❌ state={result.state} error={result.error}")
        return (idx, start, end, None)

    return (idx, start, end, None)


def batch_extract(
    chunks: list[tuple[Path, int, int]], max_workers: int = DEFAULT_WORKERS
) -> list[tuple[int, int, str | None]]:
    """
    使用线程池并发调用 MinerU Flash 解析所有切片。

    Returns:
        [(start, end, markdown_or_None), ...] 按 start 排序
    """
    total = len(chunks)
    print(f"\n🚀 并发批处理: {total} 个切片, 最大并发 {max_workers}\n")

    results: list[tuple[int, int, int, str | None]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_parse_one, path, start, end, i, total): i
            for i, (path, start, end) in enumerate(chunks, start=1)
        }
        for future in as_completed(futures):
            results.append(future.result())

    # 按页码顺序排序
    results.sort(key=lambda x: x[1])
    return [(start, end, md) for _, start, end, md in results]


# ─── Markdown 合并 ───────────────────────────────────────────────────────────


def merge_markdown(
    parsed: list[tuple[int, int, str | None]],
    total_pages: int,
    output_path: str,
    source_name: str,
) -> None:
    """合并各切片结果为单个 Markdown 文件。"""
    lines: list[str] = [
        f"# {source_name}",
        "",
        f"> MinerU Flash 批处理解析  |  共 {total_pages} 页  |  {len(parsed)} 个切片",
        "",
        "---",
        "",
    ]

    ok = sum(1 for _, _, md in parsed if md is not None)
    fail = len(parsed) - ok

    for start, end, md in parsed:
        lines.append(f"## 第 {start}–{end} 页")
        lines.append("")
        if md is None:
            lines.append(f"> ⚠️ 此切片解析失败")
        else:
            lines.append(md)
        lines.append("")
        lines.append("---")
        lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📊 合并完成: {ok}/{len(parsed)} 成功, {fail} 失败 → {output_path}")


# ─── 清理 ────────────────────────────────────────────────────────────────────


def cleanup(chunks: list[tuple[Path, int, int]]) -> None:
    """删除临时 PDF 切片文件。"""
    if not chunks:
        return
    temp_dir = chunks[0][0].parent
    for path, _, _ in chunks:
        if path.exists():
            path.unlink()
    temp_dir.rmdir()
    print(f"🧹 已清理临时文件: {temp_dir}")


# ─── 主入口 ──────────────────────────────────────────────────────────────────


def batch_parse(
    pdf_path: str,
    output_md: str | None = None,
    max_workers: int = DEFAULT_WORKERS,
    keep_temp: bool = False,
) -> str:
    """
    切分 → 批处理 → 合并，一站式处理。

    Args:
        pdf_path:   输入 PDF 路径
        output_md:  输出 Markdown 路径（默认与 PDF 同名 + _mineru.md）
        max_workers: 最大并发数
        keep_temp:  是否保留临时文件

    Returns:
        合并后的 Markdown 文本
    """
    src = Path(pdf_path)
    if not src.exists():
        raise FileNotFoundError(f"找不到文件: {pdf_path}")

    if output_md is None:
        output_md = str(src.with_suffix("")) + "_mineru.md"

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    total_mb = src.stat().st_size / (1024 * 1024)
    reader.close()

    print(f"📂 {src.name}")
    print(f"   总页数: {total_pages}  总大小: {total_mb:.1f} MB")
    print(f"   策略: 每 {CHUNK_PAGES} 页/片 → {(total_pages - 1) // CHUNK_PAGES + 1} 片, {max_workers} 并发")

    # Step 1: 切分
    print("\n✂️  切分 PDF...")
    chunks = split_pdf(pdf_path)
    for path, s, e in chunks:
        mb = path.stat().st_size / (1024 * 1024)
        print(f"   {path.name}  第 {s}–{e} 页 ({mb:.1f} MB)")

    # Step 2: 并发批处理
    parsed = batch_extract(chunks, max_workers)

    # Step 3: 合并
    print("\n🔗 合并 Markdown...")
    merge_markdown(parsed, total_pages, output_md, src.name)

    # Step 4: 清理
    if not keep_temp:
        cleanup(chunks)
    else:
        print(f"📁 临时文件保留在: {chunks[0][0].parent}")

    return Path(output_md).read_text(encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_pdf = sys.argv[1]
    output_md_val: str | None = None
    workers = DEFAULT_WORKERS
    keep_temp_val = False

    for arg in sys.argv[2:]:
        if arg == "--keep-temp":
            keep_temp_val = True
        elif arg.startswith("--workers="):
            workers = int(arg.split("=")[1])
        elif not arg.startswith("--"):
            output_md_val = arg

    try:
        batch_parse(input_pdf, output_md_val, workers, keep_temp_val)
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
