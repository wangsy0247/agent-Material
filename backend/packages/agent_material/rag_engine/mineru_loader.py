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
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from agent_material.utils.logger import logger


def _extract_result_markdown(result: Any) -> str | None:
    if isinstance(result, str):
        return result
    return getattr(result, "markdown", None)


def _call_mineru_extract(client: Any, pdf_path: str, *, ocr_required: bool) -> Any:
    """结构化索引使用标准 extract；旧客户端/关闭开关时保留 Flash 路径。"""
    from agent_material.rag_engine.indexing.structured_artifacts import (
        structured_index_enabled,
    )

    if structured_index_enabled() and hasattr(client, "extract"):
        return client.extract(pdf_path, ocr=ocr_required, table=True)
    return client.flash_extract(pdf_path)

# MinerU Flash 模式限制
FLASH_MAX_SIZE_MB = 10
FLASH_MAX_PAGES = 20

# 批处理并发数和重试次数
BATCH_MAX_WORKERS = 3
BATCH_MAX_RETRIES = 3

# PDF 预检与解析质量门禁。阈值有意保守：只拦截明显为空或明显缺页的结果，
# 不用“字符越多质量越高”这类容易误伤表格、图纸和演示文稿的启发式规则。
TEXT_LAYER_MIN_CHARS_PER_PAGE = 1
PARSED_MIN_CHARS_PER_PAGE = 8
PARSED_MIN_CHARS = 20


@dataclass(frozen=True)
class PdfProfile:
    """PDF 文本层画像，用于决定是否必须依赖 OCR。"""

    total_pages: int
    text_layer_pages: int
    image_only_pages: int
    inspection_failed_pages: int
    pdf_mode: str
    ocr_required: bool

    def to_metadata(self) -> dict:
        return asdict(self)


def _has_pdf_signature(pdf_path: str) -> bool:
    """只接受真正以 PDF 魔数开头的文件，拒绝 HTML/multipart 伪装文件。"""
    try:
        with open(pdf_path, "rb") as stream:
            return stream.read(5) == b"%PDF-"
    except OSError:
        return False


def inspect_pdf(pdf_path: str) -> PdfProfile:
    """按页检查 PDF 文本层，分类为 native/scanned/mixed/unknown。

    这是路由信号而不是 OCR 结果：扫描页交给 MinerU 解析；若预检自身失败，
    按 ``unknown + ocr_required`` 保守处理，禁止静默回退成不完整的纯文本。
    """
    if not _has_pdf_signature(pdf_path):
        raise ValueError(f"文件不是有效 PDF（缺少 %PDF- 文件头）: {pdf_path}")

    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        text_pages = 0
        failed_pages = 0
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
                visible_chars = len(re.sub(r"\s+", "", text))
                if visible_chars >= TEXT_LAYER_MIN_CHARS_PER_PAGE:
                    text_pages += 1
            except Exception:
                failed_pages += 1
        if hasattr(reader, "close"):
            reader.close()
    except Exception as exc:
        total_pages = _get_pdf_page_count(pdf_path)
        logger.warning(f"[MinerU] PDF 文本层预检失败，按 unknown/OCR 处理: {pdf_path} | {exc}")
        return PdfProfile(
            total_pages=total_pages,
            text_layer_pages=0,
            image_only_pages=total_pages,
            inspection_failed_pages=total_pages,
            pdf_mode="unknown",
            ocr_required=True,
        )

    image_only_pages = max(total_pages - text_pages - failed_pages, 0)
    if total_pages <= 0 or failed_pages == total_pages:
        mode = "unknown"
    elif text_pages == total_pages:
        mode = "native"
    elif text_pages == 0 and failed_pages == 0:
        mode = "scanned"
    else:
        mode = "mixed"
    return PdfProfile(
        total_pages=total_pages,
        text_layer_pages=text_pages,
        image_only_pages=image_only_pages,
        inspection_failed_pages=failed_pages,
        pdf_mode=mode,
        ocr_required=mode != "native",
    )


def _markdown_visible_char_count(markdown: str) -> int:
    """计算 Markdown 中可检索正文的近似字符数。"""
    text = re.sub(r"<!--[\s\S]*?-->", "", markdown or "")
    text = re.sub(r"!\[[^]]*\]\([^)]*\)", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return len(re.findall(r"[\w\u4e00-\u9fff]", text, flags=re.UNICODE))


def _quality_metadata(markdown: str, total_pages: int, ok: int, fail: int) -> dict:
    visible_chars = _markdown_visible_char_count(markdown)
    total_parts = ok + fail
    success_ratio = ok / total_parts if total_parts else 0.0
    minimum_chars = max(PARSED_MIN_CHARS, total_pages * PARSED_MIN_CHARS_PER_PAGE)
    passed = fail == 0 and ok > 0 and visible_chars >= minimum_chars
    if fail:
        reason = f"存在 {fail} 个失败分片"
    elif ok == 0:
        reason = "没有成功解析的分片"
    elif visible_chars < minimum_chars:
        reason = f"有效正文过少: {visible_chars} < {minimum_chars}"
    else:
        reason = ""
    return {
        "parse_quality_status": "passed" if passed else "rejected",
        "parse_quality_reason": reason,
        "parse_visible_chars": visible_chars,
        "parse_success_ratio": success_ratio,
    }


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
    pending = list(page_ranges)
    sequence = 0

    def _cleanup() -> None:
        for path, _, _ in chunks:
            try:
                os.unlink(path)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    if not pending:
        _cleanup()
        if hasattr(reader, "close"):
            reader.close()
        return []

    while pending:
        start, end = pending.pop(0)
        sequence += 1
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
                logger.error("[MinerU] 未找到 PdfWriter")
                _cleanup()
                if hasattr(reader, "close"):
                    reader.close()
                return []

            for i in range(start - 1, min(end, len(reader.pages))):
                writer.add_page(reader.pages[i])

            tmp_path = os.path.join(tmp_dir, f"part_{sequence:03d}_p{start}-{end}.pdf")
            with open(tmp_path, "wb") as f:
                writer.write(f)

            mb = os.path.getsize(tmp_path) / (1024 * 1024)
            if mb > FLASH_MAX_SIZE_MB:
                os.unlink(tmp_path)
                if start == end:
                    logger.error(
                        f"[MinerU] 单页 p{start} 为 {mb:.1f}MB，超过 Flash "
                        f"{FLASH_MAX_SIZE_MB}MB 限制，拒绝不完整入库"
                    )
                    _cleanup()
                    return []
                middle = (start + end) // 2
                pending[0:0] = [(start, middle), (middle + 1, end)]
                continue

            logger.debug(f"[MinerU] 切分: p{start}-{end} ({mb:.1f}MB)")
            chunks.append((tmp_path, start, end))

        except Exception as e:
            logger.error(f"[MinerU] 切分失败 p{start}-p{end}: {e}")
            _cleanup()
            if hasattr(reader, "close"):
                reader.close()
            return []

    if hasattr(reader, 'close'):
        reader.close()
    chunks.sort(key=lambda item: item[1])
    return chunks


# ════════════════════════════════════════════════════════════════════════════════
#  Phase 2: 并发批处理
# ════════════════════════════════════════════════════════════════════════════════


def _parse_one(
    tmp_path: str, start: int, end: int, idx: int, total: int,
    max_retries: int = BATCH_MAX_RETRIES,
    ocr_required: bool = False,
) -> tuple[int, int, int, Any | None]:
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
            result = _call_mineru_extract(
                client, tmp_path, ocr_required=ocr_required
            )
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
            return (idx, start, end, result)

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
    ocr_required: bool = False,
) -> list[tuple[int, int, Any | None]]:
    """使用线程池并发调用 MinerU Flash 解析所有切片。

    Returns:
        [(start_page, end_page, markdown_or_None), ...] 按 start_page 排序
    """
    total = len(chunks)
    logger.info(
        f"[MinerU] 并发批处理: {total} 个切片, 最大并发 {max_workers}"
    )

    results: list[tuple[int, int, int, Any | None]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _parse_one, path, start, end, i, total,
                ocr_required=ocr_required,
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
    parsed: list[tuple[int, int, Any | None]],
    chunks: list[tuple[str, int, int]],
) -> tuple[str, int, int]:
    """合并解析结果为完整 Markdown，清理临时文件。

    Returns:
        (merged_markdown, success_count, fail_count)
    """
    parts: list[str] = []
    ok = 0
    fail = 0

    for start, end, result in parsed:
        markdown = _extract_result_markdown(result)
        if markdown is not None:
            header = f"<!-- part p{start}-p{end} -->\n\n"
            parts.append(header + markdown)
            ok += 1
        else:
            # 失败分片绝不写入占位文本；上层质量门禁会拒绝整份版本。
            fail += 1

    # 清理临时文件
    if chunks:
        tmp_dir = os.path.dirname(chunks[0][0])
        for path, _, _ in chunks:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError as exc:
                logger.warning(f"[MinerU] 临时分片清理失败: {path} | {exc}")
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    full_md = "\n\n---\n\n".join(parts)
    return full_md, ok, fail


# ════════════════════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════════════════════


def mineru_pdf_loader(
    file_path: str, *, version_context: dict | None = None
) -> list[Document]:
    """使用 MinerU 加载 PDF 为 Markdown 格式，支持文本、表格和图片。

    小文件（≤20 页）：直接 API 调用。
    大文件（>20 页）：全量切分 → 并发批处理 → 按页合并。

    Args:
        file_path: PDF 文件路径

    Returns:
        包含完整 Markdown 内容的 Document 列表（通常为单元素）。
    """
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        logger.error(f"[MinerU] 文件不存在: {abs_path}")
        return []
    if not _has_pdf_signature(abs_path):
        logger.error(f"[MinerU] 文件格式校验失败（缺少 %PDF- 文件头）: {abs_path}")
        return []

    file_size_mb = os.path.getsize(abs_path) / (1024 * 1024)
    profile = inspect_pdf(abs_path)
    total_pages = profile.total_pages
    source_name = Path(abs_path).name
    logger.info(f"[MinerU] 开始解析: {source_name} ({total_pages} 页, {file_size_mb:.1f} MB)")

    if file_size_mb > FLASH_MAX_SIZE_MB:
        logger.warning(
            f"[MinerU] 文件 ({file_size_mb:.1f} MB) 超过 Flash 限制 ({FLASH_MAX_SIZE_MB} MB)，"
            f"将尝试解析，可能失败。建议使用 Precision 模式或压缩 PDF。"
        )

    # ── 同时满足页数和体积限制才直接解析 ──
    if total_pages <= FLASH_MAX_PAGES and file_size_mb <= FLASH_MAX_SIZE_MB:
        from mineru import MinerU

        client = MinerU()
        return _extract_single(
            abs_path,
            client,
            1,
            total_pages,
            profile=profile,
            version_context=version_context,
        )

    # ── 大文件：全量切分 → 并发批处理 → 按页合并 ──
    logger.info(
        f"[MinerU] PDF 需要安全拆分 ({total_pages} 页, {file_size_mb:.1f}MB)，"
        f"单分片上限 {FLASH_MAX_PAGES} 页/{FLASH_MAX_SIZE_MB}MB"
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
    parsed = _batch_extract(
        chunks,
        max_workers=BATCH_MAX_WORKERS,
        ocr_required=profile.ocr_required,
    )

    # Phase 3: 合并 + 清理
    logger.info("[MinerU] Phase 3/3: 合并结果 & 清理临时文件")
    full_md, ok, fail = _merge_and_cleanup(parsed, chunks)
    quality = _quality_metadata(full_md, total_pages, ok, fail)
    if quality["parse_quality_status"] != "passed":
        logger.error(
            f"[MinerU] 解析质量门禁拒绝: {source_name} | "
            f"{quality['parse_quality_reason']}"
        )
        return []

    metadata = {
        "source": abs_path,
        "file_type": "pdf",
        "loader": "mineru",
        "total_pages": total_pages,
        "parts": len(chunks),
        "parts_ok": ok,
        "parts_failed": fail,
        "content_format": "markdown",
        "ocr_route_selected": profile.ocr_required,
        **profile.to_metadata(),
        **quality,
    }

    logger.info(
        f"[MinerU] 解析完成: {source_name} | "
        f"{len(full_md)} 字符 | {ok}/{ok + fail} 段成功"
    )
    artifact_docs: list[Document] = []
    ordinal_offset = 0
    if version_context:
        from agent_material.rag_engine.indexing.structured_artifacts import (
            build_artifact_documents,
        )

        for start, _, result in parsed:
            if result is None or isinstance(result, str):
                continue
            batch_artifacts = build_artifact_documents(
                result,
                source=abs_path,
                version_context=version_context,
                page_offset=start - 1,
                ordinal_offset=ordinal_offset,
            )
            ordinal_offset += len(batch_artifacts)
            artifact_docs.extend(batch_artifacts)
        for artifact in artifact_docs:
            artifact.metadata = {
                **metadata,
                **artifact.metadata,
                "content_format": "structured_artifact",
            }

    return [Document(page_content=full_md, metadata=metadata), *artifact_docs]


def _extract_single(
    pdf_path: str, client, start_page: int, end_page: int,
    profile: PdfProfile | None = None,
    version_context: dict | None = None,
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
    profile = profile or inspect_pdf(pdf_path)
    try:
        result = _call_mineru_extract(
            client, pdf_path, ocr_required=profile.ocr_required
        )
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

    total_pages = max(end_page - start_page + 1, 1)
    quality = _quality_metadata(md_content, total_pages, ok=1, fail=0)
    if quality["parse_quality_status"] != "passed":
        logger.error(
            f"[MinerU] 解析质量门禁拒绝 p{start_page}-p{end_page}: "
            f"{quality['parse_quality_reason']}"
        )
        return []

    # 添加段落标记
    if start_page != 1 or end_page != 1:
        header = f"<!-- part p{start_page}-p{end_page} -->\n\n"
    else:
        header = ""

    md_content = header + md_content

    metadata = {
        "source": pdf_path,
        "file_type": "pdf",
        "loader": "mineru",
        "start_page": start_page,
        "end_page": end_page,
        "parts": 1,
        "parts_ok": 1,
        "parts_failed": 0,
        "content_format": "markdown",
        "ocr_route_selected": profile.ocr_required,
        **profile.to_metadata(),
        **quality,
    }
    main_document = Document(
        page_content=md_content,
        metadata=metadata,
    )
    artifact_docs: list[Document] = []
    if version_context:
        from agent_material.rag_engine.indexing.structured_artifacts import (
            build_artifact_documents,
        )

        artifact_docs = build_artifact_documents(
            result,
            source=pdf_path,
            version_context=version_context,
            page_offset=start_page - 1,
        )
        for artifact in artifact_docs:
            artifact.metadata = {
                **metadata,
                **artifact.metadata,
                "content_format": "structured_artifact",
            }
    return [main_document, *artifact_docs]
