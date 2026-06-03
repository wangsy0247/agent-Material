"""文件加载器 — 独立于主项目。"""

import os
import hashlib
from typing import Union

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_core.documents import Document
from rag_mcp_server.utils.path import get_abs_path
from rag_mcp_server.utils.logger import logger


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


def pdf_loader(path: str, password=None) -> list:
    """加载 PDF，优先使用软件手册清洗器，失败则回退到 PyPDFLoader。"""
    # 尝试软件手册清洗
    try:
        from rag_mcp_server.cleaner.pdf_manual_cleaner import SoftwareManualCleaner

        cleaner = SoftwareManualCleaner()
        pages = cleaner.extract_pages(path)
        if pages:
            cleaned_pages = cleaner.clean(pages)
            docs = []
            for page in cleaned_pages:
                text = page.get("cleaned_text", "")
                tags = page.get("tags", [])
                if not text or "toc_page" in tags:
                    continue
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": path,
                            "page": page.get("page_num", 0),
                        },
                    )
                )
            if docs:
                logger.info(f"PDF手册清洗成功: {path}，共 {len(docs)} 页")
                return docs
    except ImportError:
        logger.debug(f"pdfplumber 未安装，跳过 PDF 清洗: {path}")
    except Exception as e:
        logger.debug(f"PDF软件手册清洗失败，回退到 PyPDFLoader: {path} | {e}")

    # 回退到纯文本提取
    return pdf_text_loader(path, password)


def html_loader(path: str) -> list:
    """加载并清洗 HTML 文件，返回 Markdown 格式文本。"""
    try:
        from rag_mcp_server.cleaner.html_cleaner import HTMLCleaner

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
