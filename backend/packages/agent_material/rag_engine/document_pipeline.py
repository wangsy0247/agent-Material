"""文档处理管道 — 三阶段流水线。

流程：Loader → Cleaner → Splitter

PDF 处理流程（MinerU）：
  MinerU API（文本/表格/图片 → Markdown）
  → MarkdownCleaner（移除噪声、规范化格式）
  → MarkdownHeader + Recursive + SemanticMerge（标题切分 + 递归字符 + 语义合并）

支持的扩展名：.txt, .md, .pdf, .html, .docx, .pptx
支持的切分器：recursive, semantic, parent, markdown_header, html_header, sentence_boundary
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

from agent_material.utils.config_handler import vector_store_config
from agent_material.utils.logger import logger


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

        from agent_material.rag_engine.file_loader import (
            txt_loader,
            md_loader,
            mineru_pdf_loader,
            html_loader,
            docx_loader,
            pptx_loader,
        )

        cls._LOADERS = {
            ".txt": txt_loader,
            ".md": md_loader,
            ".pdf": mineru_pdf_loader,
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
#  切分器配置工具
# ════════════════════════════════════════════════════════════════

from agent_material.rag_engine.text_spliter import (
    BaseSplitter,
    RecursiveSplitter,
    SemanticSplitter,
    ParentDocumentSplitter,
    MarkdownHeaderSplitter,
    HTMLHeaderSplitter,
    SemanticMergeOnly,
    SentenceBoundarySplitter,
    HybridSplitter,
)

# 切分器工厂（供 Pipeline 和外部使用）
SPLITTER_FACTORY: dict[str, Callable[[], BaseSplitter]] = {
    "recursive": RecursiveSplitter,
    "sentence_boundary": SentenceBoundarySplitter,
    "semantic": SemanticSplitter,
    "semantic_merge": SemanticMergeOnly,
    "parent": ParentDocumentSplitter,
    "markdown_header": MarkdownHeaderSplitter,
    "html_header": HTMLHeaderSplitter,
}


# ════════════════════════════════════════════════════════════════
# ProcessingPipeline — 混合切分流水线
# ════════════════════════════════════════════════════════════════


class ProcessingPipeline:
    """完整处理管道：Loader → Cleaner → Splitter → Metadata 增强。

    PDF 处理（MinerU）：
      MinerU 提取 Markdown（文本+表格+图片）
      → MarkdownCleaner（去除噪声、规范化格式）
      → markdown_header + recursive + semantic_merge
      → MetadataEnhancer

    切分策略从 config 的 extension_pipeline 读取，按扩展名映射多步切分步骤。
    """

    # 需要执行 Markdown 清洗的扩展名（MinerU / Markdown 来源）
    _MD_CLEAN_EXTS: set[str] = {".pdf", ".md"}

    def __init__(self):
        self.loader = DocumentLoader()
        self._ext_pipeline: dict[str, list[str]] = vector_store_config.get(
            "extension_pipeline", {}
        )
        self._md_cleaner = None  # 延迟初始化

    def _get_cleaner(self):
        """延迟初始化 MarkdownCleaner。"""
        if self._md_cleaner is None:
            from agent_material.rag_engine.cleaner.markdown_cleaner import MarkdownCleaner
            self._md_cleaner = MarkdownCleaner()
        return self._md_cleaner

    def _get_splitter(self, file_path: str) -> BaseSplitter:
        """根据扩展名获取切分器实例（单步或多步混合）。"""
        ext = os.path.splitext(file_path)[1].lower()
        steps = self._ext_pipeline.get(ext, ["recursive"])
        return HybridSplitter(steps)

    def  process(
        self, file_path: str, enhance_metadata: bool = True,
        save_pre_split_dir: str = "",
    ) -> list[Document]:
        """处理单个文件。

        四阶段流水线：
        1. 加载（MinerU → Markdown / 传统 Loader）
        2. 数据清洗（MarkdownCleaner，仅 .pdf / .md）
        3. 切分（按 extension_pipeline 配置的步骤链）
        4. Metadata 增强（语言、类型、溯源等）

        Args:
            file_path: 文件路径
            enhance_metadata: 是否增强元数据
            save_pre_split_dir: 非空时，将切分前的 Document 保存到此目录（JSONL 格式）

        Returns:
            切分后的 Document 列表
        """
        # 阶段 1：加载（MinerU 已内建清洗）
        docs = self.loader.load(file_path)
        if not docs:
            logger.warning(f"文件加载为空: {file_path}")
            return []

        # 阶段 2：数据清洗 & 格式恢复（Markdown 来源的二次清洗）
        docs = self._clean_documents(docs, file_path)

        # 可选：保存切分前的内容（调试/检查用）
        if save_pre_split_dir:
            self._save_pre_split(docs, file_path, save_pre_split_dir)

        # 阶段 3：按扩展名选择切分器并切分
        splitter = self._get_splitter(file_path)
        chunks = splitter.split_documents(docs)

        # 阶段 4：metadata 增强
        if enhance_metadata:
            chunks = self._enhance_metadata(chunks)

        logger.info(
            f"[Pipeline] {file_path} → {splitter.__class__.__name__} | 产出 {len(chunks)} chunks"
        )
        return chunks

    def _clean_documents(
        self, docs: list[Document], file_path: str,
    ) -> list[Document]:
        """对 Markdown 来源的文档执行二次清洗。

        仅在 loader 未内建 Markdown 清洗时生效。
        MarkdownCleaner 是幂等的，重复清洗无害。
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self._MD_CLEAN_EXTS:
            return docs
        try:
            cleaner = self._get_cleaner()
            return cleaner.clean(docs)
        except Exception as e:
            logger.warning(f"[Pipeline] Markdown 清洗失败，跳过: {e}")
            return docs

    @staticmethod
    def _save_pre_split(docs: list[Document], file_path: str, output_dir: str) -> None:
        """将切分前的 Document 列表保存为 JSONL 文件。"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        filename = Path(file_path).stem + "_pre_split.jsonl"
        filepath = out / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }, ensure_ascii=False) + "\n")
        logger.info(
            f"[Pipeline] 切分前内容已保存: {filepath} "
            f"({len(docs)} docs, {sum(len(d.page_content) for d in docs)} chars)"
        )

    @staticmethod
    def _enhance_metadata(chunks: list[Document]) -> list[Document]:
        try:
            from agent_material.rag_engine.metadata_enhancer import MetadataEnhancer
            enhancer = MetadataEnhancer()
            return enhancer.enhance(chunks)
        except Exception as e:
            logger.warning(f"Metadata 增强失败，跳过: {e}")
            return chunks


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
