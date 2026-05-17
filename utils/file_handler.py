import os
import hashlib
from typing import Union
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core import documents
from utils.path_tools import get_abs_path
from utils.logger import logger

# 可选依赖：Markdown 文档
try:
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
    MD_AVAILABLE = True
except ImportError:
    MD_AVAILABLE = False
    logger.warning(
        "UnstructuredMarkdownLoader 不可用，将无法加载 .md 文件 "
        "（pip install unstructured）"
    )

# 可选依赖：Word 文档
try:
    from langchain_community.document_loaders import Docx2txtLoader
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("Docx2txtLoader 不可用，将无法加载 .docx 文件（pip install docx2txt）")

# 可选依赖：PPT 文档
try:
    from langchain_community.document_loaders import UnstructuredPowerPointLoader
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False
    logger.warning(
        "UnstructuredPowerPointLoader 不可用，将无法加载 .pptx 文件 "
        "（pip install python-pptx unstructured）"
    )


def get_file_md5_hex(file_path: str) -> str | None:
    abs_path = get_abs_path(file_path)
    if not os.path.exists(abs_path):
        logger.error(f"文件不存在: {file_path}")
        return None

    if not os.path.isfile(abs_path):
        logger.error(f"不是文件: {file_path}")
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
            md5_hex = md5_obj.hexdigest()
            return md5_hex

    except Exception as e:
        logger.error(f"md5计算失败: {file_path}, 错误: {e}")
        return None


def filelist_with_allow_type(path: str, allow_type: Union[tuple, list]) -> list[str]:
    files = []

    if not os.path.isdir(get_abs_path(path)):
        logger.error(f"不是目录: {path}")
        return files

    for filename in os.listdir(get_abs_path(path)):
        if filename.endswith(tuple(allow_type)):
            files.append(os.path.join(get_abs_path(path), filename))

    return files


def pdf_loader(path: str, password=None) -> list:
    try:
        return PyPDFLoader(path, password).load()
    except Exception as e:
        logger.error(f"PDF加载失败 {path}: {e}")
        return []


def txt_loader(path: str) -> list:
    """加载纯文本文件（.txt）。"""
    try:
        return TextLoader(path, autodetect_encoding=True).load()
    except Exception as e:
        logger.error(f"文本加载失败 {path}: {e}")
        return []


def md_loader(path: str) -> list:
    """加载 Markdown 文件（.md），保留标题层级等结构信息。"""
    if not MD_AVAILABLE:
        # fallback: 用 TextLoader 兜底
        logger.warning(f"UnstructuredMarkdownLoader 未安装，使用 TextLoader 兜底加载 {path}")
        return txt_loader(path)
    try:
        return UnstructuredMarkdownLoader(path).load()
    except Exception as e:
        logger.error(f"Markdown加载失败 {path}: {e}")
        return []


def docx_loader(path: str) -> list:
    """加载 Word 文档（.docx）。"""
    if not DOCX_AVAILABLE:
        logger.error(f"Docx2txtLoader 未安装，无法加载 {path}")
        return []
    try:
        return Docx2txtLoader(path).load()
    except Exception as e:
        logger.error(f"DOCX加载失败 {path}: {e}")
        return []


def pptx_loader(path: str) -> list:
    """加载 PowerPoint 文档（.pptx）。"""
    if not PPTX_AVAILABLE:
        logger.error(f"UnstructuredPowerPointLoader 未安装，无法加载 {path}")
        return []
    try:
        return UnstructuredPowerPointLoader(path).load()
    except Exception as e:
        logger.error(f"PPTX加载失败 {path}: {e}")
        return []


if __name__ == "__main__":
    pass
