"""文件加载器 — 独立于主项目。"""

import os
import hashlib
from typing import Union

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_core.documents import Document
from agent_material.utils.path_tools import get_abs_path
from agent_material.utils.logger import logger


def get_file_md5_hex(file_path: str) -> str | None:
    abs_path = get_abs_path(file_path) if not os.path.isabs(file_path) else file_path
    if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
        return None
    chunk_size = 4096
    md5_obj = hashlib.md5()
    try:
        with open(abs_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                md5_obj.update(chunk)
        return md5_obj.hexdigest()
    except Exception:
        return None


def filelist_with_allow_type(path: str, allow_type: Union[tuple, list]) -> list[str]:
    abs_path = get_abs_path(path) if not os.path.isabs(path) else path
    files = []
    if not os.path.isdir(abs_path):
        logger.error(f"不是目录: {path}")
        return files
    for root, _dirs, filenames in os.walk(abs_path):
        for filename in filenames:
            if filename.endswith(tuple(allow_type)):
                files.append(os.path.join(root, filename))
    return files


def pdf_text_loader(path: str, password=None) -> list:
    """纯文本提取（不经过手册清洗），用于论文等需要原始文本的场景。

    优先使用 PyMuPDF（fitz），缺失则回退到 PyPDFLoader。
    """
    try:
        import fitz

        doc = fitz.open(path)
        return [
            Document(
                page_content=page.get_text(),
                metadata={"source": path, "page": i + 1},
            )
            for i, page in enumerate(doc)
        ]
    except ImportError:
        logger.debug(f"PyMuPDF 未安装，回退到 PyPDFLoader: {path}")
    except Exception as e:
        logger.debug(f"PyMuPDF 提取失败，回退到 PyPDFLoader: {path} | {e}")

    try:
        return PyPDFLoader(path, password).load()
    except Exception as e:
        logger.error(f"PDF加载失败 {path}: {e}")
        return []




def html_loader(path: str) -> list:
    """加载并清洗 HTML 文件，返回 Markdown 格式文本。"""
    try:
        from agent_material.rag_engine.cleaner.html_cleaner import HTMLCleaner

        _html_cleaner = HTMLCleaner()
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        cleaned_text = _html_cleaner.clean_html(content)
        if not cleaned_text:
            logger.warning(f"HTML清洗后为空: {path}")
            return []
        return [Document(page_content=cleaned_text, metadata={"source": path})]
    except Exception as e:
        logger.error(f"HTML加载失败 {path}: {e}")
        return []


def txt_loader(path: str) -> list:
    try:
        return TextLoader(path, autodetect_encoding=True).load()
    except Exception as e:
        logger.error(f"文本加载失败 {path}: {e}")
        return []


def md_loader(path: str) -> list:
    try:
        return UnstructuredMarkdownLoader(path).load()
    except Exception as e:
        logger.error(f"Markdown加载失败 {path}: {e}")
        return []


def docx_loader(path: str) -> list:
    try:
        return Docx2txtLoader(path).load()
    except Exception as e:
        logger.error(f"DOCX加载失败 {path}: {e}")
        return []


def pptx_loader(path: str) -> list:
    try:
        return UnstructuredPowerPointLoader(path).load()
    except Exception as e:
        logger.error(f"PPTX加载失败 {path}: {e}")
        return []


def mineru_pdf_loader(
    path: str, *, version_context: dict | None = None
) -> list[Document]:
    """使用 MinerU 加载 PDF → 清洗 Markdown → 返回结构化 Document。

    流程: MinerU API → Markdown → MarkdownCleaner → list[Document]
    原生文本 PDF 失败时可回退到纯文本提取；扫描版、混合型或无法判断的 PDF
    必须依赖 OCR，禁止回退为可能缺页的纯文本结果。
    """
    def _native_text_fallback(reason: str) -> list[Document]:
        from agent_material.rag_engine.mineru_loader import (
            PARSED_MIN_CHARS,
            inspect_pdf,
        )

        try:
            profile = inspect_pdf(path)
        except (OSError, ValueError) as exc:
            logger.error(f"[MinerU] PDF 预检失败，拒绝降级: {path} | {exc}")
            return []
        if profile.ocr_required:
            logger.error(
                f"[MinerU] {profile.pdf_mode} PDF 需要 OCR，{reason}后拒绝纯文本回退: {path}"
            )
            return []

        fallback_docs = pdf_text_loader(path)
        visible_chars = sum(
            len("".join((doc.page_content or "").split())) for doc in fallback_docs
        )
        if visible_chars < PARSED_MIN_CHARS:
            logger.error(f"[MinerU] 纯文本回退内容过少，拒绝入库: {path}")
            return []

        profile_meta = profile.to_metadata()
        for doc in fallback_docs:
            doc.metadata.update(
                {
                    "file_type": "pdf",
                    "loader": "pdf_text_fallback",
                    "content_format": "text",
                    "ocr_route_selected": False,
                    "parse_quality_status": "fallback_text",
                    "parse_quality_reason": reason,
                    "parse_visible_chars": visible_chars,
                    "parse_success_ratio": 1.0,
                    "parts": 1,
                    "parts_ok": 1,
                    "parts_failed": 0,
                    **profile_meta,
                }
            )
        logger.warning(f"[MinerU] 原生文本 PDF 已降级到纯文本加载: {path} | {reason}")
        return fallback_docs

    try:
        from agent_material.rag_engine.mineru_loader import mineru_pdf_loader as _mineru_load
        from agent_material.rag_engine.cleaner.markdown_cleaner import MarkdownCleaner

        docs = (
            _mineru_load(path, version_context=version_context)
            if version_context is not None
            else _mineru_load(path)
        )
        if not docs:
            return _native_text_fallback("MinerU 解析失败或质量门禁未通过")

        from agent_material.rag_engine.indexing.structured_artifacts import (
            is_artifact_document,
        )

        cleaner = MarkdownCleaner()
        artifacts = [doc for doc in docs if is_artifact_document(doc)]
        cleaned = cleaner.clean([doc for doc in docs if not is_artifact_document(doc)])
        cleaned.extend(artifacts)
        if cleaned:
            logger.info(f"[MinerU] 加载+清洗成功: {path} | {len(cleaned)} 篇文档")
            return cleaned

        return _native_text_fallback("MinerU 清洗后为空")

    except ImportError as e:
        return _native_text_fallback(f"MinerU 依赖不可用: {e}")
    except Exception as e:
        logger.error(f"[MinerU] 加载异常: {e}")
        return _native_text_fallback(f"MinerU 加载异常: {e}")
