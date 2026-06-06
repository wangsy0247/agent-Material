"""MinerU Markdown 数据清洗器。

清洗 MinerU 输出的 Markdown 文本，移除解析产物和出版平台噪声，
恢复标准 Markdown 数据格式，供下游切分和 Embedding 使用。

清洗步骤（按序执行）：
1. 移除 MinerU 分段标记（<!-- part ... -->）
2. 规范化图片占位符（<!-- image--> → [图片]）
3. 移除出版平台导航噪声（推荐阅读、期刊信息等）
4. 修复标题层级（跳过级别，统一格式）
5. 规范化学术排版（LaTeX 空格、上下标拼接）
6. 移除目录页噪声（点线 + 页码模式）
7. 空行规范化
8. URL/DOI 噪声移除
"""

from __future__ import annotations

import re
from langchain_core.documents import Document

from rag_mcp_server.utils.logger import logger


class MarkdownCleaner:
    """MinerU 输出 Markdown 清洗器。

    所有清洗步骤均基于纯文本正则匹配，不依赖外部模型。
    支持链式调用：``cleaner.clean(documents)`` → 清洗后的 Document 列表。
    """

    # ── MinerU 产物 ──
    RE_PART_MARKER = re.compile(r"<!--\s*part[^>]*-->", re.IGNORECASE)

    # ── 图片占位符 ──
    RE_IMAGE_PLACEHOLDER = re.compile(r"<!--\s*image\s*-->", re.IGNORECASE)

    # ── 出版平台噪声区块 ──
    # (开始标记正则, 结束标记正则) — 匹配整个区块并移除
    NOISE_BLOCKS: list[tuple[re.Pattern, re.Pattern]] = [
        # AIP Publishing
        (re.compile(r"^#*\s*Articles\s+You\s+May\s+Be\s+Interested\s+In", re.MULTILINE | re.IGNORECASE),
         re.compile(r"^#\s", re.MULTILINE)),
        (re.compile(r"^#*\s*Why\s+Publish\s+With\s+Us\??", re.MULTILINE | re.IGNORECASE),
         re.compile(r"^#\s", re.MULTILINE)),
        (re.compile(r"^#*\s*AIP\s+Advances", re.MULTILINE | re.IGNORECASE),
         re.compile(r"^#\s", re.MULTILINE)),
        # ACS Publishing
        (re.compile(r"^#*\s*Supporting\s+Information", re.MULTILINE | re.IGNORECASE),
         re.compile(r"^#\s", re.MULTILINE)),
        (re.compile(r"^#*\s*Recommended\s+Articles", re.MULTILINE | re.IGNORECASE),
         re.compile(r"^#\s", re.MULTILINE)),
    ]

    # ── 单行噪声 ──
    NOISE_LINES: list[re.Pattern] = [
        re.compile(r"^Check\s+for\s+updates$", re.IGNORECASE),
        re.compile(r"^Learn\s+More$", re.IGNORECASE),
        re.compile(r"^View\s+Online$", re.IGNORECASE),
        re.compile(r"^Download\s+PDF$", re.IGNORECASE),
        re.compile(r"^Share\s+(?:on\s+)?\w+$", re.IGNORECASE),
        re.compile(r"^Export\s+Citation$", re.IGNORECASE),
        re.compile(r"^\d+\s*DAYS\s+average\s*time\s*to\s*1st\s*decision$", re.IGNORECASE),
        re.compile(r"^OVER\s+\d+\s*MILLION\s+views", re.IGNORECASE),
        re.compile(r"^INCLUSIVE\s+scope$", re.IGNORECASE),
        re.compile(r"^Subscribe\s+to", re.IGNORECASE),
        re.compile(r"^Get\s+access$", re.IGNORECASE),
        re.compile(r"^Published\s+under\s+license", re.IGNORECASE),
        re.compile(r"^All\s+Rights\s+Reserved", re.IGNORECASE),
    ]

    # ── 目录页检测 ──
    RE_TOC_DOTLINE = re.compile(r"\.{3,}\s*\d+\s*$")
    RE_TOC_NUMBERED = re.compile(
        r"^(?:第?\s*(?:\d+\.)+\d*\s*.+?)?\s*\.{3,}\s*\d+\s*$"
    )

    # ── LaTeX 规范化 ──
    # 移除 LaTeX 中 MathJax/minerU 产生的多余空格
    RE_LATEX_SPACES = re.compile(r"(\$[^$]*?\$)")
    # minerU 经常在 \mathrm、\mathbf 等命令内部产生多余空格
    RE_MATH_SPACE = re.compile(
        r"(\\mathrm|\\mathbf|\\mathit|\\mathsf|\\mathtt|\\text|\\textbf)"
        r"\s+\{", re.IGNORECASE
    )

    # ── 空行规范化 ──
    RE_MULTI_NEWLINE = re.compile(r"\n{4,}")
    RE_TRAILING_SPACES = re.compile(r"[ \t]+$", re.MULTILINE)

    # ── 学术 DOI/URL 行 ──
    RE_PUBLISHER_HEADER = re.compile(
        r"^https?://pubs\.(?:acs|aip|rsc|wiley|springer|elsevier)\.\S*\s*$",
        re.IGNORECASE,
    )
    RE_CITE_AS = re.compile(r"^Cite\s*as:\s*.*$", re.IGNORECASE)
    RE_SUBMITTED_LINE = re.compile(
        r"^Submitted:\s*\d+.*?Published\s*Online:\s*\d+.*?$", re.IGNORECASE
    )

    def __init__(
        self,
        min_text_length: int = 100,
        normalize_latex: bool = True,
        remove_image_placeholders: bool = False,
        remove_toc: bool = True,
        normalize_headings: bool = True,
        remove_noise_blocks: bool = True,
    ):
        self.min_text_length = min_text_length
        self.normalize_latex = normalize_latex
        self.remove_image_placeholders = remove_image_placeholders
        self.remove_toc = remove_toc
        self.normalize_headings = normalize_headings
        self.remove_noise_blocks = remove_noise_blocks

    # ------------------------------------------------------------------ #
    #  主入口
    # ------------------------------------------------------------------ #

    def clean(self, documents: list[Document]) -> list[Document]:
        """批量清洗 Document 列表（MinerU 输出的 Markdown 文档）。

        Args:
            documents: MinerU 加载的 Document 列表

        Returns:
            清洗后的 Document 列表（保留原有 metadata，追加 cleaning 标记）
        """
        if not documents:
            return []

        cleaned: list[Document] = []
        for doc in documents:
            text = doc.page_content
            original_len = len(text)

            # 逐步骤清洗
            text = self._remove_part_markers(text)
            text = self._handle_image_placeholders(text)
            if self.remove_noise_blocks:
                text = self._remove_noise_blocks(text)
            text = self._remove_noise_lines(text)
            text = self._remove_publisher_headers(text)
            if self.normalize_latex:
                text = self._normalize_latex(text)
            if self.remove_toc:
                text = self._remove_toc_section(text)
            if self.normalize_headings:
                text = self._normalize_heading_hierarchy(text)
            text = self._normalize_spacing(text)
            text = self._normalize_special_chars(text)

            # 清洗后过短 → 丢弃（但保留原 Document）
            if len(text.strip()) < self.min_text_length:
                logger.warning(
                    f"[MarkdownCleaner] 清洗后文本过短 ({len(text.strip())} 字符)，"
                    f"保留原文档: {doc.metadata.get('source', 'unknown')}"
                )
                cleaned.append(doc)
                continue

            meta = dict(doc.metadata)
            meta["cleaning"] = {
                "cleaned": True,
                "original_chars": original_len,
                "cleaned_chars": len(text),
                "cleaner": "markdown_cleaner",
            }

            cleaned.append(Document(page_content=text, metadata=meta))

        logger.info(
            f"[MarkdownCleaner] 清洗完成: {len(documents)} 篇文档 → "
            f"{len(cleaned)} 篇输出"
        )
        return cleaned

    def clean_text(self, text: str) -> str:
        """清洗单段 Markdown 文本（用于外部调用）。

        Args:
            text: 待清洗的 Markdown 文本

        Returns:
            清洗后的文本
        """
        text = self._remove_part_markers(text)
        text = self._handle_image_placeholders(text)
        if self.remove_noise_blocks:
            text = self._remove_noise_blocks(text)
        text = self._remove_noise_lines(text)
        text = self._remove_publisher_headers(text)
        if self.normalize_latex:
            text = self._normalize_latex(text)
        if self.remove_toc:
            text = self._remove_toc_section(text)
        if self.normalize_headings:
            text = self._normalize_heading_hierarchy(text)
        text = self._normalize_spacing(text)
        text = self._normalize_special_chars(text)
        return text.strip()

    # ------------------------------------------------------------------ #
    #  清洗步骤
    # ------------------------------------------------------------------ #

    @staticmethod
    def _remove_part_markers(text: str) -> str:
        """移除 MinerU 分段标记。

        ``<!-- part 001 p1-20 -->`` → 移除
        ``<!-- part p1-p20 -->`` → 移除
        """
        return MarkdownCleaner.RE_PART_MARKER.sub("", text)

    def _handle_image_placeholders(self, text: str) -> str:
        """处理图片占位符。

        默认保留（因为图片位置本身包含语义信息），
        若 ``remove_image_placeholders=True`` 则替换为 ``[图片]`` 文本标记。
        """
        if self.remove_image_placeholders:
            return self.RE_IMAGE_PLACEHOLDER.sub("[图片]", text)
        # 保留但规范化格式（移除多余空格）
        return re.sub(r"<!--\s*image\s*-->", "<!-- image -->", text, flags=re.IGNORECASE)

    @classmethod
    def _remove_noise_blocks(cls, text: str) -> str:
        """移除出版平台噪声区块。

        匹配 NOISE_BLOCKS 中定义的 (开始标记, 结束标记) 对，
        移除整个区块内容。
        """
        for start_re, end_re in cls.NOISE_BLOCKS:
            text = cls._remove_block(text, start_re, end_re)
        return text

    @staticmethod
    def _remove_block(text: str, start_re: re.Pattern, end_re: re.Pattern) -> str:
        """移除由正则匹配的文本区块。

        从 start_re 首次匹配位置开始，到 end_re 首次匹配位置（不含）结束。
        """
        start_match = start_re.search(text)
        if not start_match:
            return text

        block_start = start_match.start()
        # 从区块开始位置找最近的标题行作为区块起点
        line_start = text.rfind("\n", 0, block_start)
        line_start = line_start + 1 if line_start != -1 else 0

        remaining = text[block_start:]
        end_match = end_re.search(remaining)
        if not end_match:
            # 找不到结束标记 → 移除从开始到文本末尾
            return text[:line_start].rstrip()

        block_end = block_start + end_match.start()
        return text[:line_start] + text[block_end:]

    @classmethod
    def _remove_noise_lines(cls, text: str) -> str:
        """移除单行噪声（如 "Check for updates", "Learn More" 等）。"""
        lines = text.split("\n")
        result: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                result.append(line)
                continue
            if any(pattern.match(stripped) for pattern in cls.NOISE_LINES):
                continue
            result.append(line)
        return "\n".join(result)

    @classmethod
    def _remove_publisher_headers(cls, text: str) -> str:
        """移除出版商 URL、引用行、投稿信息行。"""
        lines = text.split("\n")
        result: list[str] = []
        for line in lines:
            stripped = line.strip()
            if cls.RE_PUBLISHER_HEADER.match(stripped):
                continue
            if cls.RE_CITE_AS.match(stripped):
                continue
            if cls.RE_SUBMITTED_LINE.match(stripped):
                continue
            result.append(line)
        return "\n".join(result)

    @staticmethod
    def _normalize_latex(text: str) -> str:
        """规范 LaTeX 数学公式格式。

        - 修复 ``\\mathrm { ... }`` → ``\\mathrm{...}``（移除命令与花括号间空格）
        - 保留公式内部有效内容
        """
        def _fix_math(m: re.Match) -> str:
            formula = m.group(1)
            # 精简 \cmd { → \cmd{
            formula = MarkdownCleaner.RE_MATH_SPACE.sub(r"\1{", formula)
            return formula

        return MarkdownCleaner.RE_LATEX_SPACES.sub(_fix_math, text)

    def _remove_toc_section(self, text: str) -> str:
        """检测并移除目录区块。

        检测特征：连续 3+ 行包含点线+页码模式（``... 5`` / ``. . 12``）。
        """
        lines = text.split("\n")
        toc_scores: list[int] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                toc_scores.append(0)
                continue
            if self.RE_TOC_DOTLINE.search(stripped):
                toc_scores.append(2)
            elif self.RE_TOC_NUMBERED.match(stripped):
                toc_scores.append(1)
            else:
                toc_scores.append(0)

        # 滑动窗口：连续 5+ 行累计分数 ≥ 8 → 视为目录
        in_toc = False
        result: list[str] = []
        window_size = 5
        threshold = 6

        for i, line in enumerate(lines):
            if not in_toc:
                window = toc_scores[i:i + window_size]
                if sum(window) >= threshold:
                    in_toc = True
            if in_toc:
                if toc_scores[i] == 0:
                    # 遇到空白或非目录行 → 检查是否目录结束
                    next_window = toc_scores[i:i + 3]
                    if sum(next_window) == 0:
                        in_toc = False
                        result.append(line)
                    # 否则继续跳过（目录内空行）
                # 目录行 → 跳过
            else:
                result.append(line)

        return "\n".join(result)

    @staticmethod
    def _normalize_heading_hierarchy(text: str) -> str:
        """修复 Markdown 标题层级。

        规则：
        - 文档标题（第一个 # 标题）保持不变
        - 后续标题最多使用 ### (H3)，保持层级在 2 级以内
        - 移除空的标题行
        """
        lines = text.split("\n")
        result: list[str] = []
        first_h1 = True

        for line in lines:
            stripped = line.strip()

            # 空标题行
            heading_match = re.match(r"^(#{1,6})\s*$", stripped)
            if heading_match:
                continue

            # 普通内容行
            h_match = re.match(r"^(#{1,6})\s+(.+)", stripped)
            if not h_match:
                result.append(line)
                continue

            level = len(h_match.group(1))
            title = h_match.group(2)

            # 保持文档主标题 (# H1)
            if level == 1:
                if first_h1:
                    first_h1 = False
                else:
                    level = 2  # 后续 H1 降为 H2

            # 限制层级深度（最多 H4）
            level = min(level, 4)

            result.append(f"{'#' * level} {title}")

        return "\n".join(result)

    @staticmethod
    def _normalize_spacing(text: str) -> str:
        """规范化空行和空格。

        - 3+ 连续空行 → 2 空行
        - 行尾空格 → 移除
        - 首尾空行 → 移除
        """
        text = MarkdownCleaner.RE_MULTI_NEWLINE.sub("\n\n\n", text)
        text = MarkdownCleaner.RE_TRAILING_SPACES.sub("", text)
        return text.strip()

    @staticmethod
    def _normalize_special_chars(text: str) -> str:
        """规范化特殊字符。

        - 全角数字/字母 → 半角
        - 特殊 Unicode 空白 → 普通空格
        - 连续的不可见字符 → 移除
        """
        # 特殊 Unicode 空白 → 普通空格
        text = re.sub(r"[ -   　]", " ", text)
        # 零宽字符 → 移除
        text = re.sub(r"[​-‏⁠﻿]", "", text)
        # 控制字符（除了换行）→ 移除
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        return text

    # ------------------------------------------------------------------ #
    #  辅助方法
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_cleaning_stats(documents: list[Document]) -> dict:
        """获取清洗统计信息（供外部监控使用）。

        Returns:
            包含字符数变化、文档数等的统计字典
        """
        if not documents:
            return {"documents": 0}

        total_orig = 0
        total_cleaned = 0
        for doc in documents:
            cleaning = doc.metadata.get("cleaning", {})
            total_orig += cleaning.get("original_chars", 0)
            total_cleaned += cleaning.get("cleaned_chars", 0)

        return {
            "documents": len(documents),
            "total_original_chars": total_orig,
            "total_cleaned_chars": total_cleaned,
            "reduction_ratio": (
                (total_orig - total_cleaned) / max(total_orig, 1)
                if total_orig > 0
                else 0
            ),
        }
