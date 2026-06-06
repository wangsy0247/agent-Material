"""通用 PDF 数据清洗器。

自动识别 PDF 类型（用户手册 / 学术论文）并应用对应的清洗策略：

所有 PDF 通用：
- 孤立页码去除、CID 乱码修复、空白规范化
- 表格结构提取（pdfplumber 检测 → Markdown 格式）
- 标题层级识别（基于编号模式 + 字号信息）

用户手册特化（自动检测触发）：
- 目录页检测与去除（点线 + 页码模式）
- 代码块格式保留（shell / cmake / git 等命令行）
- FAQ 问答对结构保留
- 目录点线和尾随页码去除

学术论文特化（自动检测触发）：
- 双栏布局检测与重排（解决 pdfplumber 串行提取问题）
- 页眉页脚检测与去除（期刊名、DOI、版权声明、页码）
- 图片/表格标题抽取（Figure/Fig/Table/Algorithm 模式）
- 扫描版 PDF OCR 支持（需安装 pytesseract + pdf2image）

Usage:
    python pdf_cleaner.py --input file.pdf --output cleaned.jsonl
    python pdf_cleaner.py --input scanned.pdf --output cleaned.jsonl --ocr-lang chi_sim
    python pdf_cleaner.py -i paper.pdf -o cleaned.txt --format txt --no-ocr
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

import logging


# ============================================================================
#  数据结构
# ============================================================================


@dataclass
class PDFCleaningStats:
    """清洗统计信息。"""

    pages_input: int = 0
    pages_after_cleaning: int = 0
    page_number_lines_removed: int = 0
    cid_replacements: int = 0

    # 手册类
    toc_pages_removed: int = 0
    code_blocks_preserved: int = 0
    faq_pairs_detected: int = 0

    # 论文类
    header_footer_lines_removed: int = 0
    two_column_pages_fixed: int = 0
    tables_extracted: int = 0
    figure_captions_extracted: int = 0
    scanned_pages_ocr: int = 0
    headings_detected: int = 0


@dataclass
class ColumnLayout:
    """页面分栏布局信息。"""

    columns: int = 1
    gap_center_x: float | None = None
    gap_width: float = 0
    left_min: float = 0
    left_max: float = 0
    right_min: float = 0
    right_max: float = 0


# ============================================================================
#  通用 PDF 清洗器
# ============================================================================


class PDFCleaner:
    """通用 PDF 数据清洗器。

    自动检测 PDF 特征并应用对应的清洗策略，不区分用户手册/学术论文。
    所有清洗步骤均为自动检测触发，不需要人工指定模式。
    """

    # ================================================================
    #  配置常量
    # ================================================================

    # CID 乱码映射
    CID_REPLACEMENTS: dict[str, str] = {
        "(cid:2)": "×",
        "(cid:3)": "·",
        "(cid:4)": "•",
    }

    # 页眉/页脚区域
    HEADER_Y_RATIO = 0.08
    FOOTER_Y_RATIO = 0.08

    # 双栏检测
    COLUMN_GAP_MIN_RATIO = 0.04
    COLUMN_GAP_SEARCH_RANGE = (0.3, 0.7)
    COLUMN_GAP_HISTOGRAM_BINS = 40

    # 标题检测
    TITLE_MIN_FONT_RATIO = 1.15
    HEADING_FONT_RATIO = 1.05
    HEADING_MIN_LENGTH = 3
    HEADING_MAX_LENGTH = 120

    # 目录页检测
    TOC_DETECTION_THRESHOLD = 5

    # OCR
    OCR_MIN_CHARS_PER_PAGE = 50

    # 代码块检测模式
    CODE_PATTERNS = [
        r"^\s*\d+\s+root@[\w-]+[:~]#",
        r"^\s*\d+\s+\$\s",
        r"^\s*\d+\s+cmake\s",
        r"^\s*\d+\s+make\s",
        r"^\s*\d+\s+mpirun\s",
        r"^\s*\d+\s+git\s",
        r"^\s*\d+\s+sudo\s",
        r"^\s*\d+\s+cd\s",
        r"^\s*\d+\s+gcc\s",
        r"^\s*\d+\s+g\+\+\s",
        r"^\s*\d+\s+wget\s",
        r"^\s*\d+\s+tar\s",
        r"^\s*\d+\s+export\s",
        r"^\s*\d+\s+source\s",
        r"^\s*\d+\s+module\s",
        r"^\s*\d+\s+lsb_release",
        r"^\s*\d+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[01]\s+[01]\s+[01]\s*$",
        r"^\s*\d+\s+[A-Za-z_][A-Za-z0-9_]*\s+[\d.]+\s+\d+\s+[\d.]+\s+[\d.]+\s*$",
    ]

    # FAQ 检测
    FAQ_QUESTION_PATTERN = re.compile(r"^\([a-z0-9]\)\.\s*(?:问[：:])?", re.MULTILINE)
    FAQ_BULLET_QUESTION = re.compile(r"^\d+\.\s*问[：:]", re.MULTILINE)

    # 章节标题检测
    CHAPTER_PATTERN = re.compile(
        r"^(?:第\s*\d+\s*章\s+.+|"
        r"\d+(?:\.\d+){1,2}\s+.+)$",
        re.MULTILINE,
    )

    # 学术页眉/页脚关键词
    HEADER_FOOTER_PATTERNS = [
        re.compile(p, re.IGNORECASE) for p in [
            r"^DOI[:\s]", r"^doi[:\s]", r"\bDOI\b.*10\.\d{4,}",
            r"^Received[:\s]", r"^Accepted[:\s]", r"^Published[:\s]", r"^Revised[:\s]",
            r"©\s*\d{4}", r"Copyright\s+©",
            r"arXiv[:\s]\d{4}\.", r"^Preprint", r"^Submitted\s+to\b",
            r"^Vol\.?\s*\d+", r"^Volume\s*\d+", r"^No\.\s*\d+",
            r"^pp?\.\s*\d+[–\-]\d+", r"^Page[s]?\s*\d+\s+of\s+\d+",
            r"^ISSN[:\s]", r"^ISBN[:\s]",
            r"^\d{4}\s*[-–]\s*\d{4}\s*/\d+\s*/\d+\s*\$",
            r"\d+\s*\|?\s*Page\b", r"\bAll\s+Rights\s+Reserved\b",
            r"\bOpen\s+Access\b", r"^https?://", r"^www\.",
            r"^Proceedings\s+of\s+the\b",
            r"^\d{4}\s+(IEEE|ACM|Springer|Elsevier|Wiley)\b",
        ]
    ]

    # 图片/表格标题模式
    FIGURE_CAPTION_PATTERNS = [
        re.compile(p, re.IGNORECASE) for p in [
            r"^(?:Fig\.?|Figure)\s*\d+[\s\.:,]",
            r"^(?:Fig\.?|Figure)\s*[A-Z][\s\.:,]",
            r"^Table\s*\d+[\s\.:,]",
            r"^Algorithm\s*\d+[\s\.:,]",
            r"^Scheme\s*\d+[\s\.:,]",
        ]
    ]

    def __init__(
        self,
        # 通用开关
        enable_two_column: bool = True,
        enable_header_footer: bool = True,
        enable_table_extract: bool = True,
        enable_figure_caption: bool = True,
        enable_heading_detect: bool = True,
        enable_ocr: bool = True,
        ocr_lang: str = "eng",
        # 手册特化开关
        remove_toc_pages: bool = True,
        preserve_code_blocks: bool = True,
        preserve_faq_pairs: bool = True,
    ):
        # 通用
        self.enable_two_column = enable_two_column
        self.enable_header_footer = enable_header_footer
        self.enable_table = enable_table_extract
        self.enable_figure = enable_figure_caption
        self.enable_heading = enable_heading_detect
        self.enable_ocr = enable_ocr
        self.ocr_lang = ocr_lang
        # 手册特化
        self.remove_toc = remove_toc_pages
        self.preserve_code = preserve_code_blocks
        self.preserve_faq = preserve_faq_pairs

        self.stats = PDFCleaningStats()

    # ================================================================
    #  提取层
    # ================================================================

    @staticmethod
    def extract_pages(pdf_path: str) -> list[dict]:
        """使用 pdfplumber 提取页面文本及位置/字号信息。"""
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                words = page.extract_words(
                    keep_blank_chars=True, x_tolerance=3, y_tolerance=3,
                )
                font_sizes = [
                    c.get("size", 0) for c in page.chars if c.get("size")
                ] if page.chars else []

                pages.append({
                    "page_num": i,
                    "text": text or "",
                    "words": words,
                    "width": page.width,
                    "height": page.height,
                    "font_sizes": font_sizes,
                    "has_text_layer": bool(text and len(text.strip()) > 50),
                })
        logging.info(f"[extract] 从 {pdf_path} 提取了 {len(pages)} 页文本")
        return pages

    # ================================================================
    #  主清洗流程
    # ================================================================

    def clean(self, pages: list[dict]) -> list[dict]:
        """对提取的页面列表进行清洗。

        所有清洗步骤均为自动检测触发，不依赖文档类型。
        """
        self.stats = PDFCleaningStats()
        self.stats.pages_input = len(pages)

        # Step 0: OCR 扫描页
        pages = self._ocr_scanned_pages(pages)

        # Step 1: 预提取表格（一次打开 PDF）
        tables_by_page: dict[int, list[list[list[str | None]]]] = {}
        if self.enable_table:
            pdf_path = pages[0].get("_pdf_path", "") if pages else ""
            tables_by_page = self._extract_all_tables(pdf_path)

        # Step 2: 逐页清洗
        results: list[dict] = []
        for page in pages:
            page_num = page["page_num"]
            text = page.get("text", "")
            if not text.strip():
                continue

            tags: list[str] = []

            # 2a: 双栏重排（自动检测，单栏页无操作）
            if self.enable_two_column:
                text, fixed = self._fix_two_column_layout(page)
                if fixed:
                    tags.append("two_column_fixed")
                    self.stats.two_column_pages_fixed += 1

            # 2b: 目录页检测与去除
            if self.remove_toc:
                text, is_toc = self._remove_toc_page(text, page_num)
                if is_toc:
                    tags.append("toc_page")
                    if not text:
                        continue

            # 2c: 标题层级识别
            if self.enable_heading:
                text, n = self._annotate_headings(text, page)
                if n:
                    tags.append(f"headings:{n}")
                    self.stats.headings_detected += n

            # 2d: 图片/表格标题标记
            if self.enable_figure:
                text, n = self._mark_figure_captions(text)
                if n:
                    tags.append(f"figures:{n}")
                    self.stats.figure_captions_extracted += n

            # 2e: 代码块 / FAQ 结构保留 + 基础清洗
            text, per_page_tags = self._clean_page_text(text)
            tags.extend(per_page_tags)

            # 2f: 表格提取 & 附加
            tables_md: list[str] = []
            if self.enable_table:
                precomputed = tables_by_page.get(page_num, [])
                tables_md, text = self._extract_and_replace_tables(page, text, precomputed)
                if tables_md:
                    tags.append(f"tables:{len(tables_md)}")
                    self.stats.tables_extracted += len(tables_md)

            if tables_md:
                text += "\n\n" + "\n\n".join(tables_md)

            results.append({
                "page_num": page_num,
                "original_text": page.get("text", ""),
                "cleaned_text": text,
                "tags": tags,
            })

        # Step 3: 跨页页眉页脚去除
        if self.enable_header_footer:
            results = self._remove_headers_footers(results)

        self.stats.pages_after_cleaning = len(results)
        self._log_stats()
        return results

    # ================================================================
    #  双栏布局检测与重排
    # ================================================================

    def _fix_two_column_layout(self, page: dict) -> tuple[str, bool]:
        """检测双栏布局并重排为正确阅读顺序。单栏页面返回原文本。"""
        words = page.get("words", [])
        if not words or len(words) < 20:
            return page["text"], False

        layout = self._detect_column_layout(words, page["width"], page["height"])
        if layout.columns < 2:
            return page["text"], False

        left_words: list[dict] = []
        right_words: list[dict] = []
        spanning_words: list[dict] = []
        header_y = page["height"] * self.HEADER_Y_RATIO
        footer_y = page["height"] * (1 - self.FOOTER_Y_RATIO)

        for w in words:
            x_center = (w["x0"] + w["x1"]) / 2
            spans_gap = (
                layout.gap_center_x is not None
                and w["x0"] < layout.gap_center_x - layout.gap_width
                and w["x1"] > layout.gap_center_x + layout.gap_width
            )
            in_margin = w["top"] <= header_y or w["bottom"] >= footer_y

            if spans_gap or in_margin:
                spanning_words.append(w)
            elif layout.gap_center_x is not None and x_center < layout.gap_center_x:
                left_words.append(w)
            else:
                right_words.append(w)

        if not left_words or not right_words:
            return page["text"], False

        def _reconstruct(col_words: list[dict]) -> str:
            if not col_words:
                return ""
            col_words.sort(key=lambda w: (w["top"], w["x0"]))
            lines = self._words_to_lines(col_words)
            return "\n".join(" ".join(line) for line in lines)

        parts = [t for t in [_reconstruct(spanning_words), _reconstruct(left_words), _reconstruct(right_words)] if t]
        return "\n\n".join(parts), True

    def _detect_column_layout(
        self, words: list[dict], page_width: float, page_height: float
    ) -> ColumnLayout:
        """通过 x 坐标直方图检测分栏间隙。"""
        header_y = page_height * self.HEADER_Y_RATIO
        footer_y = page_height * (1 - self.FOOTER_Y_RATIO)

        body_x = [
            (w["x0"] + w["x1"]) / 2
            for w in words
            if header_y < w["top"] < footer_y
        ]
        if len(body_x) < 30:
            return ColumnLayout(columns=1)

        num_bins = self.COLUMN_GAP_HISTOGRAM_BINS
        bin_width = page_width / num_bins
        bins = [0] * num_bins
        for x in body_x:
            idx = min(int(x / bin_width), num_bins - 1)
            bins[idx] += 1

        max_count = max(bins) if bins else 1
        threshold = max_count * 0.15
        gap_min_bins = max(2, int(page_width * self.COLUMN_GAP_MIN_RATIO / bin_width))
        mid_start = int(num_bins * self.COLUMN_GAP_SEARCH_RANGE[0])
        mid_end = int(num_bins * self.COLUMN_GAP_SEARCH_RANGE[1])

        best_center = None
        best_width = 0
        current_start: int | None = None

        for i in range(mid_start, mid_end):
            if bins[i] < threshold:
                if current_start is None:
                    current_start = i
            elif current_start is not None:
                w = i - current_start
                if w >= gap_min_bins and w > best_width:
                    best_width = w
                    best_center = (current_start + w / 2) * bin_width
                current_start = None

        if current_start is not None:
            w = mid_end - current_start
            if w >= gap_min_bins and w > best_width:
                best_width = w
                best_center = (current_start + w / 2) * bin_width

        if best_center is None:
            return ColumnLayout(columns=1)

        return ColumnLayout(
            columns=2, gap_center_x=best_center,
            gap_width=best_width * bin_width,
            left_min=0, left_max=best_center - best_width * bin_width / 2,
            right_min=best_center + best_width * bin_width / 2, right_max=page_width,
        )

    @staticmethod
    def _words_to_lines(words: list[dict]) -> list[list[str]]:
        """将已排序的词语按 y 坐标分组为文本行。"""
        if not words:
            return []
        lines: list[list[str]] = []
        cur_line = [words[0]["text"]]
        cur_bottom = words[0]["bottom"]
        heights = sorted(w["bottom"] - w["top"] for w in words)
        threshold = (heights[len(heights) // 2] if heights else 10) * 1.5

        for w in words[1:]:
            if w["top"] - cur_bottom < threshold:
                cur_line.append(w["text"])
            else:
                lines.append(cur_line)
                cur_line = [w["text"]]
            cur_bottom = max(cur_bottom, w["bottom"])
        lines.append(cur_line)
        return lines

    # ================================================================
    #  目录页处理
    # ================================================================

    def _remove_toc_page(self, text: str, page_num: int) -> tuple[str, bool]:
        """检测并移除目录页。检测到返回 ('', True)，否则返回 (原文本, False)。"""
        lines = text.split("\n")
        if self._is_toc_page(lines):
            self.stats.toc_pages_removed += 1
            return "", True
        return text, False

    def _is_toc_page(self, lines: list[str]) -> bool:
        dotline_count = sum(
            1 for line in lines
            if re.search(r"\.{3,}", line) or re.search(r"(?:\.\s+){3,}\.", line)
        )
        return dotline_count >= self.TOC_DETECTION_THRESHOLD

    # ================================================================
    #  页眉页脚去除
    # ================================================================

    def _remove_headers_footers(self, pages: list[dict]) -> list[dict]:
        """跨页检测并去除重复的页眉页脚。"""
        if len(pages) < 2:
            return pages

        page_first: list[tuple[int, str]] = []
        page_last: list[tuple[int, str]] = []
        for idx, page in enumerate(pages):
            lines = [l.strip() for l in page.get("cleaned_text", "").split("\n") if l.strip()]
            if lines:
                page_first.append((idx, lines[0]))
                page_last.append((idx, lines[-1]))

        first_cnt = Counter(l for _, l in page_first)
        last_cnt = Counter(l for _, l in page_last)

        header_candidates = {l for l, c in first_cnt.items() if c >= 2}
        footer_candidates = {l for l, c in last_cnt.items() if c >= 2}

        header_pattern: set[str] = set()
        footer_pattern: set[str] = set()
        for _, line in page_first + page_last:
            if self._matches_hf_pattern(line):
                if any(line == l for _, l in page_first):
                    header_pattern.add(line)
                if any(line == l for _, l in page_last):
                    footer_pattern.add(line)

        all_headers = header_candidates | header_pattern
        all_footers = footer_candidates | footer_pattern

        if not all_headers and not all_footers:
            return pages

        removed = 0
        for page in pages:
            text = page.get("cleaned_text", "")
            lines = text.split("\n")
            new_lines: list[str] = []
            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    new_lines.append(line)
                    continue
                is_first = i == 0 or (i > 0 and not lines[i - 1].strip())
                is_last = i == len(lines) - 1 or (i < len(lines) - 1 and not lines[i + 1].strip())
                if (is_first and stripped in all_headers) or (is_last and stripped in all_footers):
                    removed += 1
                else:
                    new_lines.append(line)
            page["cleaned_text"] = "\n".join(new_lines).strip()

        self.stats.header_footer_lines_removed = removed
        return pages

    def _matches_hf_pattern(self, line: str) -> bool:
        line = line.strip()
        if len(line) < 5 or len(line) > 200:
            return False
        return any(p.search(line) for p in self.HEADER_FOOTER_PATTERNS)

    # ================================================================
    #  表格提取
    # ================================================================

    def _extract_and_replace_tables(
        self, page: dict, text: str,
        precomputed: list[list[list[str | None]]] | None = None,
    ) -> tuple[list[str], str]:
        """提取表格：优先 pdfplumber 检测，回退文本 | 分隔符检测。"""
        tables_md: list[str] = []

        if precomputed:
            for i, rows in enumerate(precomputed, start=1):
                md = self._rows_to_markdown(rows)
                if md:
                    tables_md.append(f"### 表格 {i}\n\n{md}")
            if tables_md:
                return tables_md, text

        # 回退：文本级 | 分隔符检测
        lines = text.split("\n")
        groups: list[list[int]] = []
        cur: list[int] = []
        in_tbl = False

        for i, line in enumerate(lines):
            s = line.strip()
            pipe = s.count("|") >= 2
            hrule = bool(re.match(r"^[\s\-+=_|]+$", s)) and len(s) > 5
            if pipe or hrule:
                if not in_tbl:
                    cur = []
                    in_tbl = True
                cur.append(i)
            elif in_tbl:
                if len(cur) >= 2:
                    groups.append(cur)
                cur = []
                in_tbl = False
        if in_tbl and len(cur) >= 2:
            groups.append(cur)

        offset = 0
        for indices in groups:
            tbl_text = [lines[i + offset] for i in indices]
            md = self._lines_to_markdown_table(tbl_text)
            if md:
                idx = len(tables_md) + 1
                tables_md.append(f"### 表格 {idx}\n\n{md}")
                first, last = indices[0] + offset, indices[-1] + offset
                lines[first : last + 1] = [f"[TABLE_{idx}]"]
                offset -= (last - first)

        return tables_md, "\n".join(lines)

    @staticmethod
    def _extract_all_tables(pdf_path: str) -> dict[int, list[list[list[str | None]]]]:
        if not pdf_path or not Path(pdf_path).exists():
            return {}
        try:
            with pdfplumber.open(pdf_path) as pdf:
                result: dict[int, list[list[list[str | None]]]] = {}
                for i, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables()
                    if tables:
                        result[i] = tables
                return result
        except Exception:
            return {}

    @staticmethod
    def _rows_to_markdown(rows: list[list[str | None]]) -> str | None:
        if not rows or len(rows) < 2:
            return None
        max_cols = max(len(r) for r in rows)
        clean: list[list[str]] = []
        for row in rows:
            cells = [str(c or "").strip().replace("\n", " ") for c in row]
            while len(cells) < max_cols:
                cells.append("")
            clean.append(cells[:max_cols])
        md = [
            "| " + " | ".join(clean[0]) + " |",
            "| " + " | ".join(["---"] * max_cols) + " |",
        ]
        for row in clean[1:]:
            md.append("| " + " | ".join(row) + " |")
        return "\n".join(md)

    @staticmethod
    def _lines_to_markdown_table(lines: list[str]) -> str | None:
        if not lines or len(lines) < 2:
            return None
        data = [l for l in lines if not re.match(r"^[\s\-+=_]+$", l.strip())]
        if len(data) < 2:
            return None
        rows: list[list[str]] = []
        for line in data:
            cells = [c.strip() for c in line.split("|")]
            while cells and not cells[0]:
                cells.pop(0)
            while cells and not cells[-1]:
                cells.pop()
            if cells:
                rows.append(cells)
        if not rows:
            return None
        max_cols = max(len(row) for row in rows)
        normalized: list[list[str]] = []
        for row in rows:
            if len(row) < max_cols:
                row = row + [""] * (max_cols - len(row))
            normalized.append(row[:max_cols])
        md = [
            "| " + " | ".join(normalized[0]) + " |",
            "| " + " | ".join(["---"] * max_cols) + " |",
        ]
        for row in normalized[1:]:
            md.append("| " + " | ".join(row) + " |")
        return "\n".join(md)

    # ================================================================
    #  图片/表格标题抽取
    # ================================================================

    @staticmethod
    def _mark_figure_captions(text: str) -> tuple[str, int]:
        """检测并标记 Figure/Table/Algorithm 标题。"""
        lines = text.split("\n")
        out: list[str] = []
        count = 0
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            is_cap = any(p.match(stripped) for p in PDFCleaner.FIGURE_CAPTION_PATTERNS)
            if is_cap:
                count += 1
                prefix = "[FIGURE]"
                if re.match(r"^Table\s*\d+", stripped, re.IGNORECASE):
                    prefix = "[TABLE_CAPTION]"
                elif re.match(r"^Algorithm\s*\d+", stripped, re.IGNORECASE):
                    prefix = "[ALGORITHM]"
                out.append(f"{prefix} {stripped}")
                i += 1
                while i < len(lines) and lines[i].strip():
                    out.append(lines[i].strip())
                    i += 1
                if i < len(lines) and not lines[i].strip():
                    out.append("")
                    i += 1
            else:
                out.append(lines[i])
                i += 1
        return "\n".join(out), count

    # ================================================================
    #  标题层级识别
    # ================================================================

    def _annotate_headings(self, text: str, page: dict) -> tuple[str, int]:
        """识别标题层级并添加 [H0]-[H3] 标记。"""
        font_sizes = page.get("font_sizes", [])
        lines = text.split("\n")
        out: list[str] = []
        count = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                out.append(line)
                continue
            level = self._classify_heading(stripped)
            if level is not None:
                out.append(f"[H{level}] {stripped}")
                count += 1
            else:
                out.append(line)

        return "\n".join(out), count

    def _classify_heading(self, text: str) -> int | None:
        """基于编号模式和文本特征判断标题层级。"""
        stripped = text.strip()
        if len(stripped) < self.HEADING_MIN_LENGTH:
            return None
        if len(stripped) > self.HEADING_MAX_LENGTH:
            return None
        # 普通句子结尾通常是标题
        if stripped.endswith((".", ",", ";", "?", "!")):
            if not re.match(r"^[A-Z]\w+\s*:", stripped):
                return None

        # 全大写 → 一级标题
        if stripped.isupper() and len(stripped.split()) >= 2:
            return 1

        # 编号模式层级判断
        if re.match(r"^\d+\.\d+\.\d+\s", stripped):
            return 3
        if re.match(r"^\d+\.\d+\s", stripped):
            return 2
        if re.match(r"^\d+[\.\s]\s*[A-Z]", stripped):
            return 1
        if re.match(r"^[IVXLCDM]+\.\s+\S", stripped):
            return 1
        if re.match(r"^[A-Z]\.\s+\S", stripped):
            return 2
        if re.match(r"^[ivxlcdm]+\.\s+\S", stripped):
            return 3

        # 中文编号
        if re.match(r"^第[一二三四五六七八九十百千\d]+\s*[章节篇部]\s", stripped):
            return 1

        # 短句 + 无常见虚词 → 可能是标题
        words = stripped.split()
        if (
            len(words) <= 12
            and words[0][0].isupper()
            and not re.search(
                r"\b(the|and|or|in|of|to|for|is|are|was|were|been|a|an)\b",
                stripped, re.IGNORECASE,
            )
        ):
            return 2

        return None

    # ================================================================
    #  单页文本清洗（含代码块 / FAQ 结构保留）
    # ================================================================

    def _clean_page_text(self, text: str) -> tuple[str, list[str]]:
        """对单页文本进行结构化清洗：代码块保留、FAQ 保留、基础清洗。"""
        tags: list[str] = []
        lines = text.split("\n")
        cleaned: list[str] = []
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if not stripped:
                i += 1
                continue

            # CID 乱码修复
            if "(cid:" in stripped:
                line, n = self._fix_cid(line)
                self.stats.cid_replacements += n

            # 去除目录点线
            line = self._remove_toc_dotline(line)

            # 孤立页码
            if self._is_page_number(stripped):
                self.stats.page_number_lines_removed += 1
                i += 1
                continue

            # 代码块（自动检测）
            if self.preserve_code and self._is_code_line(stripped):
                block, consumed = self._collect_code_block(lines, i)
                cleaned.extend(block)
                self.stats.code_blocks_preserved += 1
                i += consumed
                continue

            # FAQ 问答对（自动检测）
            if self.preserve_faq and self._is_faq_question(stripped):
                block, consumed = self._collect_faq_block(lines, i)
                cleaned.extend(block)
                self.stats.faq_pairs_detected += 1
                i += consumed
                continue

            cleaned.append(line.strip())
            i += 1

        # 规范化空行
        final = self._normalize_spacing(cleaned)
        return "\n".join(final).strip(), tags

    # ---- 基础清洗辅助 ----

    def _fix_cid(self, line: str) -> tuple[str, int]:
        count = 0
        for cid_str, repl in self.CID_REPLACEMENTS.items():
            if cid_str in line:
                line = line.replace(cid_str, repl)
                count += 1
        for m in re.finditer(r"\(cid:\d+\)", line):
            line = line.replace(m.group(), " ")
            count += 1
        return line, count

    @staticmethod
    def _remove_toc_dotline(line: str) -> str:
        line = re.sub(r"\s+\.{3,}\s*\d+\s*$", "", line)
        line = re.sub(r"\s+(?:\.\s+){2,}\d+\s*$", "", line)
        return line

    @staticmethod
    def _is_page_number(stripped: str) -> bool:
        return bool(
            re.fullmatch(r"\d{1,4}", stripped)
            or re.fullmatch(r"第\s*\d+\s*页", stripped)
            or re.fullmatch(r"Page\s*\d+", stripped, re.IGNORECASE)
            or re.fullmatch(r"[–\-]\s*\d+\s*[–\-]", stripped)
        )

    def _is_code_line(self, stripped: str) -> bool:
        return any(re.match(p, stripped) for p in self.CODE_PATTERNS)

    def _is_code_continuation(self, stripped: str) -> bool:
        return bool(
            re.fullmatch(r"\d+", stripped)
            or stripped.startswith("#")
            or re.match(r"^\s+[-\d.\s]+", stripped)
        )

    def _collect_code_block(self, lines: list[str], start: int) -> tuple[list[str], int]:
        block: list[str] = []
        i = start
        while i < len(lines):
            line = lines[i]
            s = line.strip()
            if not s:
                if i + 1 < len(lines) and self._is_code_line(lines[i + 1].strip()):
                    block.append("")
                    i += 1
                    continue
                break
            if self._is_code_line(s) or self._is_code_continuation(s):
                if "(cid:" in s:
                    line, _ = self._fix_cid(line)
                block.append(line.rstrip())
                i += 1
            else:
                break
        return block, i - start

    # ---- FAQ 处理 ----

    def _is_faq_question(self, stripped: str) -> bool:
        if re.match(r"^\([a-z]\)\.\s*.", stripped) and "?" in stripped:
            return True
        return bool(
            self.FAQ_QUESTION_PATTERN.match(stripped)
            or self.FAQ_BULLET_QUESTION.match(stripped)
        )

    def _collect_faq_block(self, lines: list[str], start: int) -> tuple[list[str], int]:
        block: list[str] = []
        i = start
        q = lines[i].strip()
        if "(cid:" in q:
            q, _ = self._fix_cid(q)
        block.append(q)
        i += 1
        empty_cnt = 0
        while i < len(lines):
            line = lines[i]
            s = line.strip()
            if not s:
                empty_cnt += 1
                if empty_cnt >= 2:
                    break
                block.append("")
                i += 1
                continue
            if self._is_faq_question(s):
                break
            if self.CHAPTER_PATTERN.match(s):
                break
            empty_cnt = 0
            if "(cid:" in s:
                line, _ = self._fix_cid(line)
                s = line.strip()
            block.append(s)
            i += 1
        return block, i - start

    # ---- 空白规范化 ----

    @staticmethod
    def _normalize_spacing(lines: list[str]) -> list[str]:
        result: list[str] = []
        prev_empty = False
        for line in lines:
            is_empty = not line.strip()
            if is_empty:
                if not prev_empty:
                    result.append("")
                prev_empty = True
            else:
                result.append(line)
                prev_empty = False
        while result and not result[0].strip():
            result.pop(0)
        while result and not result[-1].strip():
            result.pop()
        return result

    # ================================================================
    #  扫描版 PDF OCR
    # ================================================================

    def _ocr_scanned_pages(self, pages: list[dict]) -> list[dict]:
        """对无文本层的扫描页进行 OCR（需 pytesseract + pdf2image）。"""
        if not self.enable_ocr:
            return pages

        needing: list[int] = [
            i for i, p in enumerate(pages)
            if len(p.get("text", "").strip()) < self.OCR_MIN_CHARS_PER_PAGE
        ]
        if not needing:
            return pages

        try:
            import pytesseract  # noqa: F401
        except ImportError:
            logging.warning("[OCR] pytesseract 未安装，跳过 OCR。pip install pytesseract pdf2image")
            self.enable_ocr = False
            return pages
        try:
            from pdf2image import convert_from_path
        except ImportError:
            logging.warning("[OCR] pdf2image 未安装，跳过 OCR。pip install pdf2image")
            self.enable_ocr = False
            return pages

        path_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx in needing:
            p = pages[idx].get("_pdf_path", "")
            if p:
                path_to_indices[p].append(idx)

        for pdf_path, indices in path_to_indices.items():
            if not Path(pdf_path).exists():
                continue
            try:
                images = convert_from_path(pdf_path, dpi=300)
            except Exception as e:
                logging.error(f"[OCR] 转换失败 ({pdf_path}): {e}")
                continue
            for idx in indices:
                pn = pages[idx].get("page_num", 0)
                img_idx = pn - 1
                if img_idx >= len(images):
                    continue
                try:
                    ocr_text = pytesseract.image_to_string(images[img_idx], lang=self.ocr_lang)
                    pages[idx]["text"] = ocr_text
                    pages[idx]["has_text_layer"] = False
                    t = pages[idx].get("tags", [])
                    if isinstance(t, list):
                        t.append("ocr")
                    pages[idx]["tags"] = t
                    self.stats.scanned_pages_ocr += 1
                except Exception as e:
                    logging.error(f"[OCR] 第 {pn} 页失败: {e}")
        return pages

    # ================================================================
    #  统计与输出
    # ================================================================

    def _log_stats(self) -> None:
        s = self.stats
        logging.info("[PDFCleaner] 清洗统计:")
        logging.info(f"  输入: {s.pages_input} 页, 输出: {s.pages_after_cleaning} 页")
        if s.toc_pages_removed:
            logging.info(f"  目录页移除: {s.toc_pages_removed}")
        if s.two_column_pages_fixed:
            logging.info(f"  双栏重排: {s.two_column_pages_fixed} 页")
        if s.header_footer_lines_removed:
            logging.info(f"  页眉页脚去除: {s.header_footer_lines_removed} 行")
        if s.tables_extracted:
            logging.info(f"  表格提取: {s.tables_extracted} 个")
        if s.figure_captions_extracted:
            logging.info(f"  图片标题: {s.figure_captions_extracted} 个")
        if s.headings_detected:
            logging.info(f"  标题识别: {s.headings_detected} 个")
        if s.code_blocks_preserved:
            logging.info(f"  代码块保留: {s.code_blocks_preserved}")
        if s.faq_pairs_detected:
            logging.info(f"  FAQ 对: {s.faq_pairs_detected}")
        if s.scanned_pages_ocr:
            logging.info(f"  OCR 扫描: {s.scanned_pages_ocr} 页")
        if s.page_number_lines_removed:
            logging.info(f"  页码去除: {s.page_number_lines_removed}")
        if s.cid_replacements:
            logging.info(f"  CID 修复: {s.cid_replacements}")

    @staticmethod
    def save_cleaned_pages(
        pages: list[dict], output_path: str, format_: str = "jsonl",
    ) -> None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if format_ == "jsonl":
            with open(out_path, "w", encoding="utf-8") as f:
                for page in pages:
                    f.write(json.dumps({
                        "page_num": page["page_num"],
                        "tags": page.get("tags", []),
                        "text": page["cleaned_text"],
                    }, ensure_ascii=False) + "\n")
            logging.info(f"[save] JSONL → {output_path} ({len(pages)} 页)")

        elif format_ == "txt":
            with open(out_path, "w", encoding="utf-8") as f:
                for page in pages:
                    f.write(f"\n{'=' * 60}\n")
                    f.write(f"第 {page['page_num']} 页\n")
                    if page.get("tags"):
                        f.write(f"标签: {', '.join(page['tags'])}\n")
                    f.write(f"{'=' * 60}\n")
                    f.write(page["cleaned_text"] + "\n")
            logging.info(f"[save] TXT → {output_path} ({len(pages)} 页)")

        else:
            raise ValueError(f"不支持的格式: {format_}，请选择 'jsonl' 或 'txt'")


# ============================================================================
#  命令行接口
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="通用 PDF 数据清洗工具")
    parser.add_argument("--input", "-i", required=True, help="输入 PDF 文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出文件路径")
    parser.add_argument("--format", "-f", choices=["jsonl", "txt"], default="jsonl", help="输出格式")
    # 功能开关
    parser.add_argument("--no-two-column", action="store_true", help="禁用双栏检测与重排")
    parser.add_argument("--no-header-footer", action="store_true", help="禁用页眉页脚去除")
    parser.add_argument("--no-table", action="store_true", help="禁用表格提取")
    parser.add_argument("--no-figure", action="store_true", help="禁用图片标题抽取")
    parser.add_argument("--no-heading", action="store_true", help="禁用标题层级识别")
    parser.add_argument("--no-ocr", action="store_true", help="禁用 OCR")
    parser.add_argument("--keep-toc", action="store_true", help="保留目录页（默认移除）")
    parser.add_argument("--no-code-preserve", action="store_true", help="不保留代码块格式")
    parser.add_argument("--no-faq-preserve", action="store_true", help="不保留 FAQ 结构")
    parser.add_argument("--ocr-lang", default="eng", help="OCR 语言 (默认: eng)")

    args = parser.parse_args()

    raw_pages = PDFCleaner.extract_pages(args.input)
    for p in raw_pages:
        p["_pdf_path"] = args.input

    cleaner = PDFCleaner(
        enable_two_column=not args.no_two_column,
        enable_header_footer=not args.no_header_footer,
        enable_table_extract=not args.no_table,
        enable_figure_caption=not args.no_figure,
        enable_heading_detect=not args.no_heading,
        enable_ocr=not args.no_ocr,
        ocr_lang=args.ocr_lang,
        remove_toc_pages=not args.keep_toc,
        preserve_code_blocks=not args.no_code_preserve,
        preserve_faq_pairs=not args.no_faq_preserve,
    )
    cleaned = cleaner.clean(raw_pages)
    PDFCleaner.save_cleaned_pages(cleaned, args.output, format_=args.format)


if __name__ == "__main__":
    main()
