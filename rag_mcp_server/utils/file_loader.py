"""文件加载器 — 独立于主项目。"""

import os
import hashlib
from typing import Union

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from rag_mcp_server.utils.path import get_abs_path
from rag_mcp_server.utils.logger import logger

# 可选依赖：Markdown
try:
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
    MD_AVAILABLE = True
except ImportError:
    MD_AVAILABLE = False
    logger.warning("UnstructuredMarkdownLoader 不可用，将无法加载 .md 文件")

# 可选依赖：Word
try:
    from langchain_community.document_loaders import Docx2txtLoader
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("Docx2txtLoader 不可用，将无法加载 .docx 文件")

# 可选依赖：PPT
try:
    from langchain_community.document_loaders import UnstructuredPowerPointLoader
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False
    logger.warning("UnstructuredPowerPointLoader 不可用，将无法加载 .pptx 文件")


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
    for filename in os.listdir(abs_path):
        if filename.endswith(tuple(allow_type)):
            files.append(os.path.join(abs_path, filename))
    return files


def pdf_loader(path: str, password=None) -> list:
    try:
        return PyPDFLoader(path, password).load()
    except Exception as e:
        logger.error(f"PDF加载失败 {path}: {e}")
        return []


def txt_loader(path: str) -> list:
    try:
        return TextLoader(path, autodetect_encoding=True).load()
    except Exception as e:
        logger.error(f"文本加载失败 {path}: {e}")
        return []


def md_loader(path: str) -> list:
    if not MD_AVAILABLE:
        return txt_loader(path)
    try:
        return UnstructuredMarkdownLoader(path).load()
    except Exception as e:
        logger.error(f"Markdown加载失败 {path}: {e}")
        return []


def docx_loader(path: str) -> list:
    if not DOCX_AVAILABLE:
        logger.error(f"Docx2txtLoader 未安装，无法加载 {path}")
        return []
    try:
        return Docx2txtLoader(path).load()
    except Exception as e:
        logger.error(f"DOCX加载失败 {path}: {e}")
        return []


def pptx_loader(path: str) -> list:
    if not PPTX_AVAILABLE:
        logger.error(f"UnstructuredPowerPointLoader 未安装，无法加载 {path}")
        return []
    try:
        return UnstructuredPowerPointLoader(path).load()
    except Exception as e:
        logger.error(f"PPTX加载失败 {path}: {e}")
        return []
