"""MinerU PDF 文档加载器。

使用 MinerU (mineru-open-sdk) 将 PDF 解析为结构化 Markdown，支持文本、表格和图片提取。
对于超过 20 页的 PDF：全量切分 → ThreadPoolExecutor 并发调用 → 按页排序合并。

MinerU Flash 模式限制：
- 单文件 ≤ 10MB
- 单文件 ≤ 20 页
- 大文件需使用 Precision 模式（需 API Token）或手动拆分
"""

from __future__ import annotations

import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_core.documents import Document

from rag_mcp_server.utils.logger import logger

# MinerU Flash 模式限制
FLASH_MAX_SIZE_MB = 10
FLASH_MAX_PAGES = 20

# 批处理并发数和重试次数
BATCH_MAX_WORKERS = 3
BATCH_MAX_RETRIES = 3


def _get_pdf_page_count(pdf_path: str) -> int:
    """获取 PDF 总页数（优先 pypdf，回退 pdfplumber / PyPDF2）。"""
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        count = len(reader.pages)
        reader.close()
        return count
    except Exception:
        pass
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception:
        pass
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except Exception:
        logger.warning(f"[MinerU] 无法获取 PDF 页数: {pdf_path}，假设为 1 页")
        return 1


def _split_pdf_page_ranges(total_pages: int, max_pages: int = FLASH_MAX_PAGES) -> list[tuple[int, int]]:
    """将总页数拆分为 (start, end) 区间（1-indexed, 闭区间）。"""
    ranges: list[tuple[int, int]] = []
    start = 1
    while start <= total_pages:
        end = min(start + max_pages - 1, total_pages)
        ranges.append((start, end))
        start = end + 1
    return ranges


# ════════════════════════════════════════════════════════════════════════════════
#  Phase 1: PDF 全量切分
# ════════════════════════════════════════════════════════════════════════════════


def _split_all_chunks(
    pdf_path: str, page_ranges: list[tuple[int, int]],
) -> list[tuple[str, int, int]]:
    """一次性将 PDF 按所有页范围切分为临时文件。

    Returns:
        [(临时文件路径, start_page, end_page), ...] 按页码排序
    """
    # 获取 reader
    reader = None
    for lib in ["pypdf", "PyPDF2"]:
        try:
            if lib == "pypdf":
                from pypdf import PdfReader
            else:
                from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            break
        except (ImportError, Exception):
            continue

    if reader is None:
        logger.error("[MinerU] PDF 切分失败: 未找到可用的 PDF 库 (pip install pypdf)")
        return []

    tmp_dir = tempfile.mkdtemp(prefix="mineru_batch_")
    chunks: list[tuple[str, int, int]] = []

    for idx, (start, end) in enumerate(page_ranges, 1):
        try:
            writer = None
            for lib in ["pypdf", "PyPDF2"]:
                try:
                    if lib == "pypdf":
                        from pypdf import PdfWriter
                    else:
                        from PyPDF2 import PdfWriter
                    writer = PdfWriter()
                    break
                except ImportError:
                    continue

            if writer is None:
                logger.error(f"[MinerU] 未找到 PdfWriter")
                continue

            for i in range(start - 1, min(end, len(reader.pages))):
                writer.add_page(reader.pages[i])

            tmp_path = os.path.join(
                tmp_dir, f"part_{idx:03d}_p{start}-{end}.pdf"
            )
            with open(tmp_path, "wb") as f:
                writer.write(f)

            mb = os.path.getsize(tmp_path) / (1024 * 1024)
            logger.debug(f"[MinerU] 切分: part_{idx:03d}_p{start}-{end} ({mb:.1f}MB)")
            chunks.append((tmp_path, start, end))

        except Exception as e:
            logger.error(f"[MinerU] 切分失败 p{start}-p{end}: {e}")

    if hasattr(reader, 'close'):
        reader.close()
    return chunks


# ════════════════════════════════════════════════════════════════════════════════
#  Phase 2: 并发批处理
# ════════════════════════════════════════════════════════════════════════════════


def _parse_one(
    tmp_path: str, start: int, end: int, idx: int, total: int,
    max_retries: int = BATCH_MAX_RETRIES,
) -> tuple[int, int, int, str | None]:
    """单个切片的 flash_extract 调用，含指数退避重试。

    Returns:
        (索引, start_page, end_page, markdown_or_None)
    """
    from mineru import MinerU

    mb = os.path.getsize(tmp_path) / (1024 * 1024)
    prefix = f"[{idx}/{total}] p{start}-{end} ({mb:.1f}MB)"

    for attempt in range(max_retries + 1):
        try:
            client = MinerU()
            result = client.flash_extract(tmp_path)
        except Exception as exc:
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(
                    f"[MinerU] {prefix} 异常重试({attempt + 1}/{max_retries}) {wait}s: {exc}"
                )
                time.sleep(wait)
                continue
            logger.error(f"[MinerU] {prefix} 异常: {exc}")
            return (idx, start, end, None)

        if result.state == "done" and result.markdown is not None:
            logger.info(f"[MinerU] {prefix} ✅ {len(result.markdown)} 字符")
            return (idx, start, end, result.markdown)

        if attempt < max_retries:
            wait = 2 ** attempt
            logger.warning(
                f"[MinerU] {prefix} state={result.state} 重试({attempt + 1}/{max_retries}) {wait}s"
            )
            time.sleep(wait)
            continue

        logger.error(f"[MinerU] {prefix} state={result.state} error={result.error}")
        return (idx, start, end, None)

    return (idx, start, end, None)


def _batch_extract(
    chunks: list[tuple[str, int, int]],
    max_workers: int = BATCH_MAX_WORKERS,
) -> list[tuple[int, int, str | None]]:
    """使用线程池并发调用 MinerU Flash 解析所有切片。

    Returns:
        [(start_page, end_page, markdown_or_None), ...] 按 start_page 排序
    """
    total = len(chunks)
    logger.info(
        f"[MinerU] 并发批处理: {total} 个切片, 最大并发 {max_workers}"
    )

    results: list[tuple[int, int, int, str | None]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _parse_one, path, start, end, i, total,
            ): i
            for i, (path, start, end) in enumerate(chunks, start=1)
        }
        for future in as_completed(futures):
            results.append(future.result())

    # 按页码排序
    results.sort(key=lambda x: x[1])
    return [(start, end, md) for _, start, end, md in results]


# ════════════════════════════════════════════════════════════════════════════════
#  Phase 3: 合并 & 清理
# ════════════════════════════════════════════════════════════════════════════════


def _merge_and_cleanup(
    parsed: list[tuple[int, int, str | None]],
    chunks: list[tuple[str, int, int]],
) -> tuple[str, int, int]:
    """合并解析结果为完整 Markdown，清理临时文件。

    Returns:
        (merged_markdown, success_count, fail_count)
    """
    parts: list[str] = []
    ok = 0
    fail = 0

    for start, end, md in parsed:
        if md is not None:
            header = f"<!-- part p{start}-p{end} -->\n\n"
            parts.append(header + md)
            ok += 1
        else:
            parts.append(f"<!-- part p{start}-p{end} FAILED -->\n\n> ⚠️ 此段解析失败\n")
            fail += 1

    # 清理临时文件
    if chunks:
        tmp_dir = os.path.dirname(chunks[0][0])
        for path, _, _ in chunks:
            if os.path.exists(path):
                os.unlink(path)
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    full_md = "\n\n---\n\n".join(parts)
    return full_md, ok, fail


# ════════════════════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════════════════════


def mineru_pdf_loader(file_path: str) -> list[Document]:
    """使用 MinerU 加载 PDF 为 Markdown 格式，支持文本、表格和图片。

    小文件（≤20 页）：直接 API 调用。
    大文件（>20 页）：全量切分 → 并发批处理 → 按页合并。

    Args:
        file_path: PDF 文件路径

    Returns:
        包含完整 Markdown 内容的 Document 列表（通常为单元素）。
    """
    from mineru import MinerU

    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        logger.error(f"[MinerU] 文件不存在: {abs_path}")
        return []

    file_size_mb = os.path.getsize(abs_path) / (1024 * 1024)
    total_pages = _get_pdf_page_count(abs_path)
    source_name = Path(abs_path).name
    logger.info(f"[MinerU] 开始解析: {source_name} ({total_pages} 页, {file_size_mb:.1f} MB)")

    if file_size_mb > FLASH_MAX_SIZE_MB:
        logger.warning(
            f"[MinerU] 文件 ({file_size_mb:.1f} MB) 超过 Flash 限制 ({FLASH_MAX_SIZE_MB} MB)，"
            f"将尝试解析，可能失败。建议使用 Precision 模式或压缩 PDF。"
        )

    client = MinerU()

    # ── 小文件：直接解析 ──
    if total_pages <= FLASH_MAX_PAGES:
        return _extract_single(abs_path, client, 1, total_pages)

    # ── 大文件：全量切分 → 并发批处理 → 按页合并 ──
    logger.info(
        f"[MinerU] PDF ({total_pages} 页) 超过 {FLASH_MAX_PAGES} 页限制，"
        f"拆分为多段并发处理"
    )
    page_ranges = _split_pdf_page_ranges(total_pages, FLASH_MAX_PAGES)

    # Phase 1: 全量切分
    logger.info(f"[MinerU] Phase 1/3: 切分 PDF → {len(page_ranges)} 个临时文件")
    chunks = _split_all_chunks(abs_path, page_ranges)
    if not chunks:
        logger.error(f"[MinerU] PDF 切分全部失败: {abs_path}")
        return []

    # Phase 2: 并发批处理
    logger.info(f"[MinerU] Phase 2/3: 并发解析 {len(chunks)} 个切片 (max_workers={BATCH_MAX_WORKERS})")
    parsed = _batch_extract(chunks, max_workers=BATCH_MAX_WORKERS)

    # Phase 3: 合并 + 清理
    logger.info(f"[MinerU] Phase 3/3: 合并结果 & 清理临时文件")
    full_md, ok, fail = _merge_and_cleanup(parsed, chunks)

    metadata = {
        "source": abs_path,
        "file_type": "pdf",
        "loader": "mineru",
        "total_pages": total_pages,
        "parts": len(page_ranges),
        "parts_ok": ok,
        "parts_failed": fail,
        "content_format": "markdown",
    }

    logger.info(
        f"[MinerU] 解析完成: {source_name} | "
        f"{len(full_md)} 字符 | {ok}/{ok + fail} 段成功"
    )
    return [Document(page_content=full_md, metadata=metadata)]


def _extract_single(
    pdf_path: str, client, start_page: int, end_page: int,
) -> list[Document]:
    """调用 MinerU API 解析单个 PDF 文件（小文件或已切分的片段）。

    Args:
        pdf_path: 待解析的 PDF 文件路径
        client: MinerU 客户端实例
        start_page: 起始页码（1-indexed）
        end_page: 结束页码（1-indexed）

    Returns:
        Document 列表
    """
    try:
        result = client.flash_extract(pdf_path)
    except Exception as e:
        logger.error(f"[MinerU] API 调用异常: {e}")
        return []

    if result.state != "done":
        logger.error(
            f"[MinerU] 解析未完成 (state={result.state})"
            f"{f': {result.error}' if result.error else ''}"
        )
        return []

    md_content = result.markdown
    if md_content is None:
        logger.error("[MinerU] 解析结果为空 (markdown=None)")
        return []

    # 添加段落标记
    if start_page != 1 or end_page != 1:
        header = f"<!-- part p{start_page}-p{end_page} -->\n\n"
    else:
        header = ""

    md_content = header + md_content

    return [Document(
        page_content=md_content,
        metadata={
            "source": pdf_path,
            "start_page": start_page,
            "end_page": end_page,
            "content_format": "markdown",
        },
    )]
