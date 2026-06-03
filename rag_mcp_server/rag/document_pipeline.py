"""文档处理管道 — 极简架构：扩展名直接映射切分器。

流程：Loader（按扩展名加载）→ 切分器（按扩展名选择）→ list[Document]

支持的扩展名：.txt, .md, .pdf, .html, .docx, .pptx
支持的切分器：recursive, semantic, parent, markdown_header, html_header
"""

from __future__ import annotations

import os
from typing import Callable

from langchain_core.documents import Document
from langchain_text_splitters import (
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag_mcp_server.config import vector_store_config
from rag_mcp_server.utils.logger import logger

# ── embedding_model 延迟导入 ─────────────────────────────────────
_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from rag_mcp_server.models import embedding_model

            _embedding_model = embedding_model
        except Exception as e:
            logger.warning(f"Embedding 模型加载失败: {e}，语义切分将退化为标准切分")
            _embedding_model = None
    return _embedding_model


# ════════════════════════════════════════════════════════════════
# 阶段 1：DocumentLoader（按扩展名加载 + 清洗）
# ════════════════════════════════════════════════════════════════


class DocumentLoader:
    """按扩展名选择对应 Loader，执行加载与清洗。

    所有 Loader 统一返回 ``list[Document]``，保证下游接口一致。
    """

    _LOADERS: dict[str, Callable[[str], list[Document]]] = {}

    @classmethod
    def _init_loaders(cls):
        """惰性初始化，避免模块顶层导入失败。"""
        if cls._LOADERS:
            return

        from rag_mcp_server.utils.file_loader import (
            txt_loader,
            md_loader,
            pdf_loader,
            html_loader,
            docx_loader,
            pptx_loader,
        )

        cls._LOADERS = {
            ".txt": txt_loader,
            ".md": md_loader,
            ".pdf": pdf_loader,
            ".html": html_loader,
            ".docx": docx_loader,
            ".pptx": pptx_loader,
        }

    @classmethod
    def load(cls, file_path: str) -> list[Document]:
        cls._init_loaders()
        ext = os.path.splitext(file_path)[1].lower()
        loader = cls._LOADERS.get(ext)
        if loader is None:
            raise ValueError(
                f"不支持的文件类型: {ext}。"
                f"支持的类型: {list(cls._LOADERS.keys())}"
            )
        docs = loader(file_path)
        logger.info(f"[Loader] {file_path} 加载完成 | {len(docs)} 页/块")
        return docs


# ════════════════════════════════════════════════════════════════
# 阶段 2：切分器集合
# ════════════════════════════════════════════════════════════════


class BaseSplitter:
    """切分器抽象基类，统一接口 ``split_documents(docs) -> list[Document]``。"""

    def split_documents(self, documents: list[Document]) -> list[Document]:
        raise NotImplementedError


class RecursiveSplitter(BaseSplitter):
    """递归字符切分器 — 通用、安全、不依赖 embedding。"""

    def __init__(self):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=vector_store_config.get("chunk_size", 800),
            chunk_overlap=vector_store_config.get("chunk_overlap", 100),
            separators=vector_store_config.get(
                "separator", ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
            ),
            length_function=len,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        return self._splitter.split_documents(documents)


class SemanticSplitter(BaseSplitter):
    """语义切分器 — 优先 LangChain SemanticChunker，缺失则回退 SemanticTextSplitter。"""

    def __init__(self):
        self._fallback = RecursiveSplitter()

    def split_documents(self, documents: list[Document]) -> list[Document]:
        try:
            from langchain_experimental.text_splitter import SemanticChunker

            splitter = SemanticChunker(
                embeddings=_get_embedding_model(),
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=90,
                min_chunk_size=100,
            )
            result = splitter.split_documents(documents)
            logger.info(f"[Splitter] SemanticChunker 切分完成 | {len(result)} chunks")
            return result
        except ImportError:
            logger.warning(
                "langchain_experimental 未安装，回退到 SemanticTextSplitter。"
                "安装命令: pip install langchain-experimental"
            )
            return self._fallback_semantic(documents)
        except Exception as e:
            logger.error(f"SemanticChunker 失败: {e}，回退到 SemanticTextSplitter")
            return self._fallback_semantic(documents)

    def _fallback_semantic(self, documents: list[Document]) -> list[Document]:
        try:
            from rag_mcp_server.rag.text_spliter import SemanticTextSplitter

            splitter = SemanticTextSplitter(
                chunk_size=vector_store_config.get("chunk_size", 800),
                chunk_overlap=vector_store_config.get("chunk_overlap", 100),
                separator=vector_store_config.get("separator"),
                embedding_model=_get_embedding_model(),
                similarity_threshold=vector_store_config.get("similarity_threshold", 0.75),
            )
            result = splitter.split_documents(documents)
            logger.info(f"[Splitter] SemanticTextSplitter 切分完成 | {len(result)} chunks")
            return result
        except Exception as e:
            logger.error(f"SemanticTextSplitter 也失败: {e}，最终回退到 RecursiveSplitter")
            return self._fallback.split_documents(documents)


class ParentDocumentSplitter(BaseSplitter):
    """父子文档切分器（ParentDocumentRetriever 兼容实现）。

    先用大 chunk_size 切出"父文档"，再用小 chunk_size 切出"子文档"，
    子文档 metadata 保留 ``parent_doc`` / ``parent_id`` 引用。
    """

    def __init__(
        self,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 400,
    ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=200,
            separators=vector_store_config.get(
                "separator", ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
            ),
            length_function=len,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=50,
            separators=vector_store_config.get(
                "separator", ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
            ),
            length_function=len,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        result: list[Document] = []
        for doc in documents:
            parents = self.parent_splitter.split_documents([doc])
            for parent in parents:
                children = self.child_splitter.split_documents([parent])
                parent_id = str(hash(parent.page_content) & 0xFFFFFFFF)
                for child in children:
                    child.metadata["parent_doc"] = parent.page_content
                    child.metadata["parent_id"] = parent_id
                    result.append(child)
        logger.info(f"[Splitter] ParentDocumentSplitter 切分完成 | {len(result)} chunks")
        return result


class MarkdownHeaderSplitter(BaseSplitter):
    """Markdown 标题切分器 — 按 #/##/###/#### 切分，大 chunk 二次细切分。"""

    def __init__(self):
        self._md = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ],
            strip_headers=False,
        )
        self._fallback = RecursiveSplitter()

    def split_documents(self, documents: list[Document]) -> list[Document]:
        result: list[Document] = []
        for doc in documents:
            chunks = self._md.split_text(doc.page_content)
            if not chunks:
                chunks = [doc]
            for chunk in chunks:
                sub_docs = self._fallback.split_documents([chunk])
                for sub in sub_docs:
                    sub.metadata = {**doc.metadata, **sub.metadata}
                    result.append(sub)
        logger.info(f"[Splitter] MarkdownHeaderSplitter 切分完成 | {len(result)} chunks")
        return result


class HTMLHeaderSplitter(BaseSplitter):
    """HTML 标题切分器 — 按 h1/h2/h3 标签切分，大 chunk 二次细切分。"""

    def __init__(self):
        self._html = HTMLHeaderTextSplitter(
            headers_to_split_on=[
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
                ("h4", "Header 4"),
            ],
        )
        self._fallback = RecursiveSplitter()

    def split_documents(self, documents: list[Document]) -> list[Document]:
        result: list[Document] = []
        for doc in documents:
            chunks = self._html.split_text(doc.page_content)
            if not chunks:
                chunks = [doc]
            for chunk in chunks:
                sub_docs = self._fallback.split_documents([chunk])
                for sub in sub_docs:
                    sub.metadata = {**doc.metadata, **sub.metadata}
                    result.append(sub)
        logger.info(f"[Splitter] HTMLHeaderSplitter 切分完成 | {len(result)} chunks")
        return result


# 切分器工厂（供 Pipeline 和外部使用）
SPLITTER_FACTORY: dict[str, Callable[[], BaseSplitter]] = {
    "recursive": RecursiveSplitter,
    "semantic": SemanticSplitter,
    "parent": ParentDocumentSplitter,
    "markdown_header": MarkdownHeaderSplitter,
    "html_header": HTMLHeaderSplitter,
}


# ════════════════════════════════════════════════════════════════
# ProcessingPipeline — 扩展名直接映射切分器
# ════════════════════════════════════════════════════════════════


class ProcessingPipeline:
    """完整处理管道：Loader（按扩展名）→ 切分器（按扩展名）。"""

    def __init__(self):
        self.loader = DocumentLoader()
        self._ext_map: dict[str, str] = vector_store_config.get(
            "extension_splitter_map", {}
        )

    def _get_splitter(self, file_path: str) -> BaseSplitter:
        """根据扩展名获取切分器实例，未配置则默认 recursive。"""
        ext = os.path.splitext(file_path)[1].lower()
        splitter_name = self._ext_map.get(ext, "recursive")
        factory = SPLITTER_FACTORY.get(splitter_name)
        if factory is None:
            logger.warning(f"未知切分器: {splitter_name}，回退到 recursive")
            factory = RecursiveSplitter
        return factory()

    def  process(self, file_path: str) -> list[Document]:
        """处理单个文件。

        Returns:
            切分后的 Document 列表（标准字段：page_content, metadata）
        """
        # 阶段 1：加载 + 清洗
        docs = self.loader.load(file_path)
        if not docs:
            logger.warning(f"文件加载为空: {file_path}")
            return []

        # 阶段 2：按扩展名选择切分器并切分
        splitter = self._get_splitter(file_path)
        chunks = splitter.split_documents(docs)

        logger.info(
            f"[Pipeline] {file_path} → {splitter.__class__.__name__} | 产出 {len(chunks)} chunks"
        )
        return chunks

    def process_batch(self, file_paths: list[str]) -> list[Document]:
        """批量处理多个文件（串行，如需并发请在外层用 ThreadPoolExecutor）。"""
        all_chunks: list[Document] = []
        for fp in file_paths:
            try:
                chunks = self.process(fp)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"文件处理失败: {fp} | {e}")
        return all_chunks


# ════════════════════════════════════════════════════════════════
# DocumentSplitter — 按名称切分（供外部/手动添加文档使用）
# ════════════════════════════════════════════════════════════════


class DocumentSplitter:
    """文档切分器 — 根据切分器名称字符串创建对应实例并执行切分。

    供 ``VectorStore.add_documents`` 等外部入口使用。
    """

    def split(
        self,
        documents: list[Document],
        splitter_name: str,
    ) -> list[Document]:
        """按切分器名称执行切分。

        Args:
            documents: 加载后的 Document 列表
            splitter_name: 切分器名称（如 "recursive"/"semantic" 等），
                          未知名称时回退到 recursive

        Returns:
            切分后的 Document 列表
        """
        factory = SPLITTER_FACTORY.get(splitter_name)
        if factory is None:
            logger.warning(f"未知切分器: {splitter_name}，回退到 recursive")
            factory = RecursiveSplitter

        splitter = factory()
        chunks = splitter.split_documents(documents)
        logger.info(f"[Splitter] {splitter_name} | 产出 {len(chunks)} chunks")
        return chunks
