"""Metadata 增强器 — 在切分之后、Embedding 之前增强 Document 元数据。

为每个 chunk 自动补充以下字段，提升检索质量：
- char_count / token_estimate: 长度统计
- content_type: paragraph / code / table / heading / figure
- language: zh / en / mixed
- heading_context: 从 chunk 文本推断最近的标题上下文
- source_*: 文献溯源信息（标题、作者、DOI、年份、来源文件等）
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from langchain_core.documents import Document

from agent_material.utils.logger import logger


class MetadataEnhancer:
    """在切分后的 Document 上自动补充元数据。

    所有增强均为纯文本分析，不依赖外部模型，零延迟。
    """

    # 中文 / 英文 / 数字 / 标点 Unicode 范围
    _RE_CJK = re.compile(r"[一-鿿㐀-䶿豈-﫿]")
    _RE_LATIN = re.compile(r"[a-zA-Z]")
    _RE_CODE = re.compile(
        r"(?:^\s*(?:def |class |import |from |#include |SELECT |INSERT |"
        r"curl |wget |git |npm |pip |docker |sudo |export )"
        r"|^\s*\d+\s+\S)"  # 带行号的代码/日志行
        r"|(?:^[=\-]{3,}|^[=\-]{3,}$)",  # 分隔线
        re.MULTILINE,
    )
    _RE_TABLE = re.compile(r"^\|.+\|", re.MULTILINE)
    _RE_HEADING = re.compile(r"^#{1,4}\s|^\[H[0-3]\]", re.MULTILINE)
    _RE_FIGURE = re.compile(r"^\[(?:FIGURE|TABLE_CAPTION|ALGORITHM)\]", re.MULTILINE)
    _RE_HEADING_LINE = re.compile(r"^\[H([0-3])\]\s+(.+)", re.MULTILINE)
    # 分隔线 / 纯数字数据行
    _RE_SEPARATOR = re.compile(r"[\-=\*#_]{10,}")
    _RE_NUMERIC_DUMP = re.compile(r"^[\d\s\.\,\+\-eE]{30,}$")

    # ── 文献溯源正则 ──
    _RE_DOI = re.compile(r"\b(10\.\d{4,}/[^\s]+)\b")
    _RE_YEAR = re.compile(r"\b(19|20)\d{2}\b")
    _RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
    _RE_ARXIV = re.compile(r"arXiv:\s*(\d{4}\.\d{4,})", re.IGNORECASE)
    _RE_AUTHOR_LINE = re.compile(
        r"^[A-Z][a-z]+(?:\s+[A-Z]\.)*\s+[A-Z][a-z]+", re.MULTILINE
    )

    def __init__(
        self,
        enable_language_detect: bool = True,
        enable_content_type: bool = True,
        enable_heading_context: bool = True,
        enable_source_trace: bool = True,
    ):
        self.enable_language = enable_language_detect
        self.enable_content_type = enable_content_type
        self.enable_heading = enable_heading_context
        self.enable_source = enable_source_trace

    # ------------------------------------------------------------------ #
    #  主入口
    # ------------------------------------------------------------------ #

    def enhance(self, documents: list[Document]) -> list[Document]:
        """批量增强 Document 元数据。"""
        if not documents:
            return []

        # 文献溯源：从全文提取一次摘要信息，注入所有 chunk
        source_meta: dict[str, str] = {}
        if self.enable_source:
            source_meta = self._trace_source(documents)

        enriched: list[Document] = []
        for i, doc in enumerate(documents):
            meta = dict(doc.metadata)
            text = doc.page_content

            # 溯源信息（全局共享）
            meta.update(source_meta)

            # 基础统计
            meta["char_count"] = len(text)
            meta["token_estimate"] = self._estimate_tokens(text)

            # 语言检测
            if self.enable_language:
                meta["language"] = self._detect_language(text)

            # 内容类型
            if self.enable_content_type:
                meta["content_type"] = self._detect_content_type(text, meta)

            # chunk 位置
            if "chunk_index" not in meta:
                meta["chunk_index"] = i
            meta["total_chunks"] = len(documents)

            # 标题上下文
            if self.enable_heading:
                heading = self._infer_heading_context(text, meta)
                if heading and not meta.get("heading"):
                    meta["heading_context"] = heading

            enriched.append(Document(page_content=text, metadata=meta))

        logger.info(
            f"[MetadataEnhancer] 增强完成: {len(documents)} chunks → "
            f"{sum(1 for d in enriched if d.metadata.get('content_type') == 'code')} code, "
            f"{sum(1 for d in enriched if d.metadata.get('content_type') == 'table')} table"
        )
        return enriched

    # ------------------------------------------------------------------ #
    #  文献溯源
    # ------------------------------------------------------------------ #

    @classmethod
    def _trace_source(cls, documents: list[Document]) -> dict[str, str]:
        """从 Document 列表中提取文献溯源元数据。

        一次提取，注入所有 chunk，确保同源文档的溯源信息一致。
        """
        meta: dict[str, str] = {}

        # 1. 文件来源信息（从第一个 Document 的 metadata 获取）
        first = documents[0].metadata if documents else {}
        source_file = first.get("source", "")
        # 只当 source 是有效路径时记录（过滤掉文本片段等非路径值）
        if source_file and (
            "/" in source_file or "\\" in source_file or source_file.endswith(".pdf")
        ):
            meta["source_file"] = source_file
            meta["source_filename"] = Path(source_file).name
            meta["source_ext"] = Path(source_file).suffix.lower()

        # 2. 从全文拼接文本中提取学术文献信息
        full_text = "\n".join(d.page_content for d in documents[:20])  # 前 20 个 chunk ≈ 首页
        cls._extract_bibliography(full_text, meta)

        # 3. 从 structure recovery 已解析的标题中推断文档标题
        if not meta.get("source_title"):
            for doc in documents:
                h = doc.metadata.get("heading", "")
                lv = doc.metadata.get("heading_level", 0)
                if lv == 1 and len(h) > 3 and len(h) < 120:
                    meta["source_title"] = h
                    break

        # 4. 文件哈希
        if source_file and Path(source_file).exists():
            try:
                with open(source_file, "rb") as f:
                    meta["source_hash"] = hashlib.md5(
                        f.read(65536)  # 只读前 64KB
                    ).hexdigest()
            except Exception:
                pass

        return meta

    @classmethod
    def _extract_bibliography(cls, text: str, meta: dict[str, str]) -> None:
        """从文本中提取学术文献的书目信息。"""
        # DOI
        doi_match = cls._RE_DOI.search(text)
        if doi_match:
            meta["source_doi"] = doi_match.group(1)

        # arXiv ID
        arxiv_match = cls._RE_ARXIV.search(text)
        if arxiv_match:
            meta["source_arxiv"] = arxiv_match.group(1)

        # 年份（取第一个匹配的 19xx/20xx）
        years = cls._RE_YEAR.findall(text)
        if years:
            meta["source_year"] = years[0]

        # 作者邮箱 → 推断有作者
        emails = cls._RE_EMAIL.findall(text[:2000])
        if emails:
            meta["source_has_author_info"] = "true"

        # 文档类型推断
        meta["source_type"] = cls._infer_doc_type(text)

    @staticmethod
    def _infer_doc_type(text: str) -> str:
        """推断文档类型。"""
        text_lower = text[:3000].lower()
        if any(kw in text_lower for kw in [
            "abstract", "introduction", "conclusion", "references",
            "doi", "arxiv", "fig.", "figure 1", "et al.",
        ]):
            return "academic_paper"
        if any(kw in text_lower for kw in [
            "安装", "配置", "使用说明", "用户手册", "api",
            "install", "configuration", "usage",
        ]):
            return "manual"
        return "document"

    # ------------------------------------------------------------------ #
    #  检测方法
    # ------------------------------------------------------------------ #

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """估算 token 数。

        中文约 1.5 字符/token，英文约 4 字符/token，取加权平均 ≈ 2.5。
        """
        cjk = len(MetadataEnhancer._RE_CJK.findall(text))
        latin = len(MetadataEnhancer._RE_LATIN.findall(text))
        other = len(text) - cjk - latin
        return int(cjk / 1.5 + latin / 4 + other / 3)

    @staticmethod
    def _detect_language(text: str) -> str:
        """检测文本语言。"""
        cjk = len(MetadataEnhancer._RE_CJK.findall(text))
        latin = len(MetadataEnhancer._RE_LATIN.findall(text))
        if cjk > latin * 0.8:
            return "zh"
        elif latin > cjk * 0.8:
            return "en"
        return "mixed"

    @classmethod
    def _detect_content_type(cls, text: str, meta: dict) -> str:
        """检测内容类型。

        优先级: 分隔线/噪音 > 已有标记 > table > code > heading > figure > paragraph
        """
        stripped = text.strip()

        # 分隔线优先于纯数字（含数字的分隔线如 "21 ---" 应归为 separator）
        if cls._RE_SEPARATOR.search(stripped):
            return "separator"
        if cls._RE_NUMERIC_DUMP.match(stripped):
            return "numeric_dump"

        # 从 metadata 已有标记推断
        if meta.get("has_tables") or meta.get("table_count"):
            if cls._RE_TABLE.search(stripped):
                return "table"
        if cls._RE_TABLE.search(stripped) and stripped.count("|") >= 6:
            return "table"

        if cls._RE_CODE.search(stripped):
            return "code"

        if cls._RE_HEADING.match(stripped):
            return "heading"

        if cls._RE_FIGURE.match(stripped):
            return "figure"

        return "paragraph"

    @classmethod
    def _infer_heading_context(cls, text: str, meta: dict) -> str | None:
        """从 chunk 文本中推断最近的标题上下文。

        如果 metadata 已有 heading，直接返回。
        否则从文本中提取 [H1]/[H2]/[H3] 标记或 Markdown 标题。
        """
        if meta.get("heading"):
            return meta["heading"]

        # 尝试匹配 [H1] Title 标记
        match = cls._RE_HEADING_LINE.search(text)
        if match:
            return match.group(2).strip()

        # 尝试匹配 Markdown 标题
        match = re.search(r"^#{1,4}\s+(.+)", text, re.MULTILINE)
        if match:
            return match.group(1).strip()

        return None
