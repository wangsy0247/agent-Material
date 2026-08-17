"""文档处理管道 — 三阶段流水线。

流程：Loader → Cleaner → Splitter

PDF 处理流程（MinerU）：
  MinerU API（文本/表格/图片 → Markdown）
  → MarkdownCleaner（移除噪声、规范化格式）
  → MarkdownHeader + Recursive + SemanticMerge（标题切分 + 递归字符 + 语义合并）

支持的扩展名：.txt, .md, .pdf, .html, .docx, .pptx
支持的切分器：recursive, semantic, parent, parent_first, markdown_header, html_header, sentence_boundary
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
    ParentFirstSplitter,
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
    "parent_first": ParentFirstSplitter,
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

    # 父块优先切分适用的扩展名（富文本，含标题结构）
    _PARENT_FIRST_EXTS: set[str] = {".pdf", ".md", ".html"}

    def _get_splitter(self, file_path: str) -> BaseSplitter:
        """根据扩展名获取切分器实例（单步或多步混合）。

        feature_flags.parent_first_split 开启且为富文本格式时，
        直接使用 ParentFirstSplitter（OFF-005）；否则走 extension_pipeline
        配置的旧步骤链（回滚路径）。
        """
        ext = os.path.splitext(file_path)[1].lower()
        flags = vector_store_config.get("feature_flags", {})
        if flags.get("parent_first_split") and ext in self._PARENT_FIRST_EXTS:
            return ParentFirstSplitter()
        steps = self._ext_pipeline.get(ext, ["recursive"])
        return HybridSplitter(steps)

    def  process(
        self, file_path: str, enhance_metadata: bool = True,
        save_pre_split_dir: str = "",
        version_context: dict | None = None,
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
            version_context: 版本上下文（含 document_id / version_id 两个键）。
                非空时在切分后写入确定性 parent_id / chunk_id 及版本溯源
                元数据（OFF-001/OFF-006）；为空时保持旧行为不变

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

        # 阶段 5：版本上下文 → 确定性 ID 与溯源元数据（放在增强之后，避免被覆盖）
        if version_context:
            chunks = self._apply_version_context(chunks, version_context)

        logger.info(
            f"[Pipeline] {file_path} → {splitter.__class__.__name__} | 产出 {len(chunks)} chunks"
        )
        return chunks

    @staticmethod
    def _apply_version_context(
        chunks: list[Document], version_context: dict
    ) -> list[Document]:
        """写入确定性 parent_id / chunk_id 与版本溯源元数据（OFF-001/OFF-006）。

        - 父块按在文档中的出现顺序编号 position，用 compute_parent_id 重算
          parent_id 并同步更新其所有子块；
        - 子块按父块内顺序编号，用 compute_chunk_id 写 chunk_id；
        - 每个子块写 document_id / version_id / is_active=False / indexed_at；
        - parent_metadata 里同步写 parent_id / document_id / version_id。

        相同 (document_id, version_id, 内容) 重复处理生成完全相同的 ID（幂等）。
        """
        import hashlib
        from datetime import datetime, timezone

        from agent_material.rag_engine.versioning import (
            compute_chunk_id,
            compute_parent_id,
        )

        document_id = version_context["document_id"]
        version_id = version_context["version_id"]
        indexed_at = datetime.now(timezone.utc).isoformat()

        def _old_parent_key(chunk: Document) -> str:
            """切分器写入的占位 parent_id；缺失时按父块内容 hash 兜底分组。"""
            pid = chunk.metadata.get("parent_id")
            if pid:
                return pid
            content = chunk.metadata.get("parent_doc", chunk.page_content)
            return hashlib.md5(content.encode("utf-8")).hexdigest()[:16]

        # 第一遍：按出现顺序收集父块（旧 parent_id → 位置 / 内容）
        parents: dict[str, dict] = {}
        for chunk in chunks:
            key = _old_parent_key(chunk)
            if key not in parents:
                parents[key] = {
                    "position": len(parents),
                    "content": chunk.metadata.get("parent_doc", chunk.page_content),
                }

        # 重算确定性 parent_id
        new_parent_id = {
            key: compute_parent_id(document_id, version_id, info["position"], info["content"])
            for key, info in parents.items()
        }

        # 第二遍：写子块元数据（子块按父块内顺序编号）
        child_counters: dict[str, int] = {}
        for chunk in chunks:
            key = _old_parent_key(chunk)
            parent_id = new_parent_id[key]
            position = child_counters.get(key, 0)
            child_counters[key] = position + 1

            meta = chunk.metadata
            meta["parent_id"] = parent_id
            meta["chunk_id"] = compute_chunk_id(
                document_id, version_id, parent_id, position, chunk.page_content
            )
            meta["document_id"] = document_id
            meta["version_id"] = version_id
            meta["is_active"] = False  # staging 状态，激活前不参与在线检索
            meta["indexed_at"] = indexed_at

            parent_meta = meta.get("parent_metadata")
            if isinstance(parent_meta, dict):
                parent_meta["parent_id"] = parent_id
                parent_meta["document_id"] = document_id
                parent_meta["version_id"] = version_id

        logger.info(
            f"[Pipeline] 版本元数据写入完成 | document_id={document_id[:8]}... "
            f"version_id={version_id[:8]}... parents={len(parents)} chunks={len(chunks)}"
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
