"""软件用户手册 PDF 清洗器。

针对技术文档/软件用户手册（如 ABACUS 用户手册）的特化清洗模块，
独立于现有 RAG 流程，可单独使用。

与学术论文清洗器的区别：
- 无期刊页眉/版权，但有大量目录点线
- 大量代码块/命令行（需保留缩进和格式）
- FAQ 问答对（问和答应保持连续）
- 章节层级标题（如 2.1.1、2.1.2）
- 坐标/参数数据块
- 使用 pdfplumber 提取（中文支持更好）

Usage:
    python manual_cleaner.py --input ./datas/abacus.pdf --output ./cleaned_abacus.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
import pdfplumber
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import logging
# from rag_mcp_server.utils.logger import logger

@dataclass
class CleaningStats:
    """清洗统计信息。"""

    pages_input: int = 0
    pages_after_toc_removal: int = 0
    toc_pages_removed: int = 0
    page_number_lines_removed: int = 0
    code_blocks_preserved: int = 0
    faq_pairs_detected: int = 0
    cid_replacements: int = 0
    empty_lines_collapsed: int = 0


class SoftwareManualCleaner:
    """软件用户手册 PDF 专用清洗器。

    针对技术文档的特化清洗策略：
    1. 使用 pdfplumber 提取（中文支持优于 pypdf）
    2. 检测并去除目录页
    3. 去除目录点线和尾随页码
    4. 去除孤立页码行
    5. 处理 (cid:X) 乱码
    6. 保留代码块格式（缩进、行号）
    7. 保留 FAQ 问答对结构
    8. 识别章节标题层级（为后续切分做准备）
    9. 规范化空白
    """

    # 常见 (cid:X) 映射表 — 根据实际文档扩展
    CID_REPLACEMENTS: dict[str, str] = {
        "(cid:2)": "×",  # 乘号，常见于 k 点 mesh 描述
        "(cid:3)": "·",
        "(cid:4)": "•",
    }

    # 目录页检测阈值：单页中连续点线出现的次数
    # 注意：PDF 提取后的点线可能是 ". . . . ."（带空格）或 "......."（连续）
    TOC_DETECTION_THRESHOLD = 5

    # 代码块检测模式
    CODE_PATTERNS = [
        r"^\s*\d+\s+root@[\w-]+[:~]#",  # shell 提示符
        r"^\s*\d+\s+\$\s",  # $ 命令行
        r"^\s*\d+\s+cmake\s",  # cmake 命令
        r"^\s*\d+\s+make\s",  # make 命令
        r"^\s*\d+\s+mpirun\s",  # mpirun
        r"^\s*\d+\s+git\s",  # git 命令
        r"^\s*\d+\s+sudo\s",  # sudo
        r"^\s*\d+\s+cd\s",  # cd
        r"^\s*\d+\s+gcc\s",  # gcc
        r"^\s*\d+\s+g\+\+\s",  # g++
        r"^\s*\d+\s+wget\s",  # wget
        r"^\s*\d+\s+tar\s",  # tar
        r"^\s*\d+\s+export\s",  # export
        r"^\s*\d+\s+source\s",  # source
        r"^\s*\d+\s+module\s",  # module
        r"^\s*\d+\s+lsb_release",  # lsb_release
        r"^\s*\d+\s+[-\d\.]+\s+[-\d\.]+\s+[-\d\.]+\s+[01]\s+[01]\s+[01]\s*$",  # 坐标数据行
        r"^\s*\d+\s+[A-Za-z_][A-Za-z0-9_]*\s+[\d\.]+\s+\d+\s+[\d\.]+\s+[\d\.]+\s*$",  # 性能表格
    ]

    # FAQ 检测模式
    # 模式1: "(a). 问：xxx" 或 "(a). xxx"（ABACUS 手册中的 FAQ 子项）
    FAQ_QUESTION_PATTERN = re.compile(r"^\([a-z0-9]\)\.\s*(?:问[：:])?", re.MULTILINE)
    # 模式2: "1. 问：xxx"（编号问答）
    FAQ_BULLET_QUESTION = re.compile(r"^\d+\.\s*问[：:]", re.MULTILINE)
    # 答案开头
    FAQ_ANSWER_PATTERN = re.compile(r"^答[：:]", re.MULTILINE)

    # 章节标题检测（如 "第 2 章 编译 ABACUS"、"2.1.1 介绍"、"2.1.2 检查环境"）
    CHAPTER_PATTERN = re.compile(
        r"^(?:第\s*\d+\s*章\s+.+|"
        r"\d+(?:\.\d+){1,2}\s+.+)$",
        re.MULTILINE,
    )

    def __init__(
        self,
        remove_toc_pages: bool = True,
        preserve_code_blocks: bool = True,
        preserve_faq_pairs: bool = True,
        min_line_length: int = 1,
    ):
        self.remove_toc = remove_toc_pages
        self.preserve_code = preserve_code_blocks
        self.preserve_faq = preserve_faq_pairs
        self.min_line_length = min_line_length
        self.stats = CleaningStats()

    # ------------------------------------------------------------------ #
    #  提取层
    # ------------------------------------------------------------------ #
    @staticmethod
    def extract_pages(pdf_path: str) -> list[dict]:
        """使用 pdfplumber 逐页提取文本。

        Args:
            pdf_path: PDF 文件路径

        Returns:
            每页的字典列表，格式: [{"page_num": 1, "text": "..."}, ...]
        """

        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    pages.append({"page_num": i, "text": text})

        logging.info(f"[extract] 从 {pdf_path} 提取了 {len(pages)} 页文本")
        return pages

    # ------------------------------------------------------------------ #
    #  清洗层
    # ------------------------------------------------------------------ #
    def clean(self, pages: list[dict]) -> list[dict]:
        """对提取的页面列表进行清洗。

        Args:
            pages: extract_pages() 的输出

        Returns:
            清洗后的页面列表，每页增加 "cleaned_text" 和 "tags" 字段
        """
        self.stats = CleaningStats()
        self.stats.pages_input = len(pages)

        results: list[dict] = []
        for page in pages:
            cleaned, tags = self._clean_single_page(page["text"], page["page_num"])
            if cleaned:  # 跳过空页
                results.append(
                    {
                        "page_num": page["page_num"],
                        "original_text": page["text"],
                        "cleaned_text": cleaned,
                        "tags": tags,
                    }
                )

        # 跨页去重页眉（必须在单页清洗完成后执行）
        results = self._remove_repeated_headers(results)

        self.stats.pages_after_toc_removal = len(results)
        self._log_stats()
        return results

    def _clean_single_page(self, text: str, page_num: int) -> tuple[str, list[str]]:
        """清洗单页文本，返回 (cleaned_text, tags)。"""
        tags: list[str] = []
        lines = text.split("\n")

        # Step 1: 检测目录页
        if self._is_toc_page(lines):
            tags.append("toc_page")
            self.stats.toc_pages_removed += 1
            if self.remove_toc:
                logging.debug(f"[page {page_num}] 检测到目录页，已移除")
                return "", tags
            # 如果不移除，继续清洗目录内容

        cleaned_lines: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Step 2: 跳过空行（后续统一处理）
            if not stripped:
                i += 1
                continue

            # Step 3: 处理 (cid:X) 乱码
            if "(cid:" in stripped:
                line, n = self._fix_cid_garbage(line)
                self.stats.cid_replacements += n

            # Step 4: 去除目录点线+页码（如 "5.1 介绍 ................ 180"）
            line = self._remove_toc_dotline(line)

            # Step 5: 去除孤立页码行
            if self._is_isolated_page_number(stripped):
                self.stats.page_number_lines_removed += 1
                i += 1
                continue

            # Step 6: 检测代码块
            if self.preserve_code and self._is_code_line(stripped):
                # 收集连续代码行
                code_block, consumed = self._collect_code_block(lines, i)
                cleaned_lines.extend(code_block)
                self.stats.code_blocks_preserved += 1
                i += consumed
                continue

            # Step 7: 检测 FAQ 问答对
            if self.preserve_faq and self._is_faq_question(stripped):
                faq_block, consumed = self._collect_faq_block(lines, i)
                cleaned_lines.extend(faq_block)
                self.stats.faq_pairs_detected += 1
                i += consumed
                continue

            # Step 8: 普通文本行（去除首尾空格，使用修复 CID 后的 line）
            cleaned_lines.append(line.strip())
            i += 1

        # Step 9: 合并连续空行，规范化段落间距
        final_lines = self._normalize_blank_lines(cleaned_lines)
        self.stats.empty_lines_collapsed += len(cleaned_lines) - len(final_lines)

        cleaned_text = "\n".join(final_lines).strip()
        return cleaned_text, tags

    # ------------------------------------------------------------------ #
    #  检测辅助方法
    # ------------------------------------------------------------------ #
    def _is_toc_page(self, lines: list[str]) -> bool:
        """判断是否为目录页。特征：大量连续点线。

        匹配模式：
        - "......."  连续 5+ 个点
        - ". . . . ."  带空格的点线（pdfplumber 提取后常见）
        """
        dotline_count = 0
        for line in lines:
            # 先去除已提取的文本，统计剩余的点字符密度
            # 或者直接匹配两种模式
            if re.search(r"\.{3,}", line) or re.search(r"(?:\.\s+){3,}\.", line):
                dotline_count += 1
        return dotline_count >= self.TOC_DETECTION_THRESHOLD

    def _is_isolated_page_number(self, stripped: str) -> bool:
        """判断是否为孤立的页码行。"""
        # 纯数字，或 "第 X 页"，且长度很短
        if re.fullmatch(r"\d+", stripped):
            return True
        if re.fullmatch(r"第\s*\d+\s*页", stripped):
            return True
        if re.fullmatch(r"Page\s*\d+", stripped, re.IGNORECASE):
            return True
        return False

    def _is_code_line(self, stripped: str) -> bool:
        """判断是否为代码/命令行。"""
        return any(re.match(pattern, stripped) for pattern in self.CODE_PATTERNS)

    def _is_faq_question(self, stripped: str) -> bool:
        """判断是否为 FAQ 问题行。

        匹配格式：
        - "(a). 问：xxx" 或 "(a). xxx"（带字母标号的 FAQ）
        - "1. 问：xxx"（带数字编号的 FAQ）
        """
        # 模式1: (a). 开头 + 包含问号（常见 FAQ 格式）
        if re.match(r"^\([a-z]\)\.\s*.", stripped) and "?" in stripped:
            return True
        # 模式2: 显式匹配 "问" 字开头
        return bool(
            self.FAQ_QUESTION_PATTERN.match(stripped)
            or self.FAQ_BULLET_QUESTION.match(stripped)
        )

    # ------------------------------------------------------------------ #
    #  处理辅助方法
    # ------------------------------------------------------------------ #
    def _fix_cid_garbage(self, line: str) -> tuple[str, int]:
        """修复 (cid:X) 乱码，返回 (修复后的行, 替换次数)。"""
        count = 0
        for cid_str, replacement in self.CID_REPLACEMENTS.items():
            if cid_str in line:
                line = line.replace(cid_str, replacement)
                count += 1
        # 对于未在映射表中的 (cid:数字)，替换为空格或删除
        remaining = re.findall(r"\(cid:\d+\)", line)
        for r in remaining:
            line = line.replace(r, " ")
            count += 1
        return line, count

    @staticmethod
    def _remove_toc_dotline(line: str) -> str:
        """去除目录点线和尾随页码。

        匹配模式：
        - "标题 ........... 180"
        - "标题 . . . . . . 180"
        """
        # 模式1：连续点线 + 页码
        line = re.sub(r"\s+\.{3,}\s*\d+\s*$", "", line)
        # 模式2：带空格的点线 + 页码
        line = re.sub(r"\s+(?:\.\s+){2,}\d+\s*$", "", line)
        return line

    def _collect_code_block(self, lines: list[str], start_idx: int) -> tuple[list[str], int]:
        """收集连续代码行，返回 (代码行列表, 消耗行数)。"""
        block: list[str] = []
        i = start_idx
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                # 代码块中间允许一个空行
                if i + 1 < len(lines) and self._is_code_line(lines[i + 1].strip()):
                    block.append("")
                    i += 1
                    continue
                break
            if self._is_code_line(stripped) or self._is_code_continuation(stripped):
                # 修复 CID 乱码后再保留
                if "(cid:" in stripped:
                    line, _ = self._fix_cid_garbage(line)
                block.append(line.rstrip())  # 保留行内缩进，去除尾空格
                i += 1
            else:
                break
        return block, i - start_idx

    def _is_code_continuation(self, stripped: str) -> bool:
        """判断是否为代码块的延续行（如命令输出、参数值）。"""
        # 空行、纯数字行（行号）、注释行、缩进行
        if re.fullmatch(r"\d+", stripped):
            return True
        if stripped.startswith("#"):
            return True
        if re.match(r"^\s+[-\d\.\s]+", stripped):
            return True
        return False

    def _collect_faq_block(self, lines: list[str], start_idx: int) -> tuple[list[str], int]:
        """收集 FAQ 问答对，返回 (块行列表, 消耗行数)。"""
        block: list[str] = []
        i = start_idx

        # 收集问题行
        question_line = lines[i].strip()
        if "(cid:" in question_line:
            question_line, _ = self._fix_cid_garbage(question_line)
        block.append(question_line)
        i += 1

        # 收集答案行（直到下一个问题或空两行的段落边界）
        empty_line_count = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if not stripped:
                empty_line_count += 1
                if empty_line_count >= 2:
                    break
                block.append("")
                i += 1
                continue

            # 如果遇到下一个 FAQ 问题，停止
            if self._is_faq_question(stripped):
                break

            # 如果遇到章节标题，停止
            if self.CHAPTER_PATTERN.match(stripped):
                break

            empty_line_count = 0
            # 修复 CID 乱码
            if "(cid:" in stripped:
                line, _ = self._fix_cid_garbage(line)
                stripped = line.strip()
            block.append(stripped)
            i += 1

        return block, i - start_idx

    def _remove_repeated_headers(self, pages: list[dict]) -> list[dict]:
        """移除在每页顶部重复出现的章节标题页眉。

        策略：
        1. 收集所有页面清洗后的第一行非空文本
        2. 出现 >= 2 次且匹配章节标题模式(CHAPTER_PATTERN)的行，视为页眉
        3. 保留该页眉第一次出现的页面（通常是章节起始页的真实标题）
        4. 从其余页面顶部移除该页眉

        Args:
            pages: 单页清洗后的页面列表

        Returns:
            去重页眉后的页面列表
        """
        from collections import defaultdict

        # 收集每页第一行非空文本 → 页码列表
        first_line_to_pages: dict[str, list[int]] = defaultdict(list)
        for page in pages:
            text = page.get("cleaned_text", "")
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if lines:
                first_line_to_pages[lines[0]].append(page["page_num"])

        # 识别页眉：出现 >= 2 次且匹配章节标题模式
        headers: dict[str, list[int]] = {}
        for line, page_nums in first_line_to_pages.items():
            if len(page_nums) >= 2 and self.CHAPTER_PATTERN.match(line):
                headers[line] = page_nums

        if not headers:
            return pages

        # 从各页顶部移除页眉（保留第一次出现的页面）
        for page in pages:
            text = page.get("cleaned_text", "")
            lines = text.split("\n")
            new_lines: list[str] = []
            header_removed = False

            for line in lines:
                stripped = line.strip()
                if not header_removed and stripped in headers:
                    # 该行为页眉，直接移除
                    header_removed = True
                    continue
                new_lines.append(line)
                if stripped:
                    header_removed = True

            page["cleaned_text"] = "\n".join(new_lines).strip()

        return pages

    @staticmethod
    def _normalize_blank_lines(lines: list[str]) -> list[str]:
        """规范化空行：连续空行合并为一个，段落间保留一个空行。"""
        result: list[str] = []
        prev_empty = False

        for line in lines:
            is_empty = line.strip() == ""
            if is_empty:
                if not prev_empty:
                    result.append("")
                prev_empty = True
            else:
                result.append(line)
                prev_empty = False

        # 去除首尾空行
        while result and result[0] == "":
            result.pop(0)
        while result and result[-1] == "":
            result.pop()

        return result

    # ------------------------------------------------------------------ #
    #  输出与统计
    # ------------------------------------------------------------------ #
    def _log_stats(self) -> None:
        """打印清洗统计。"""
        s = self.stats
        logging.info("[SoftwareManualCleaner] 清洗统计:")
        logging.info(f"  输入页数: {s.pages_input}")
        logging.info(f"  输出页数: {s.pages_after_toc_removal}")
        logging.info(f"  目录页移除: {s.toc_pages_removed}")
        logging.info(f"  页码行移除: {s.page_number_lines_removed}")
        logging.info(f"  CID 乱码修复: {s.cid_replacements}")
        logging.info(f"  代码块保留: {s.code_blocks_preserved}")
        logging.info(f"  FAQ 对检测: {s.faq_pairs_detected}")

    @staticmethod
    def save_cleaned_pages(
        pages: list[dict],
        output_path: str,
        format_: str = "jsonl",
    ) -> None:
        """保存清洗结果。

        Args:
            pages: clean() 的输出
            output_path: 输出文件路径
            format_: 输出格式，"jsonl" 或 "txt"
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if format_ == "jsonl":
            with open(out_path, "w", encoding="utf-8") as f:
                for page in pages:
                    record = {
                        "page_num": page["page_num"],
                        "tags": page.get("tags", []),
                        "text": page["cleaned_text"],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logging.info(f"[save] 已保存 JSONL: {output_path} ({len(pages)} 页)")

        elif format_ == "txt":
            with open(out_path, "w", encoding="utf-8") as f:
                for page in pages:
                    f.write(f"\n{'=' * 60}\n")
                    f.write(f"第 {page['page_num']} 页\n")
                    if page.get("tags"):
                        f.write(f"标签: {', '.join(page['tags'])}\n")
                    f.write(f"{'=' * 60}\n")
                    f.write(page["cleaned_text"] + "\n")
            logging.info(f"[save] 已保存 TXT: {output_path} ({len(pages)} 页)")

        else:
            raise ValueError(f"不支持的格式: {format_}，请选择 'jsonl' 或 'txt'")


# ==================================================================== #
#  命令行接口
# ==================================================================== #
def main():
    parser = argparse.ArgumentParser(description="软件用户手册 PDF 清洗工具")
    parser.add_argument("--input", "-i", required=True, help="输入 PDF 文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出文件路径")
    parser.add_argument(
        "--format",
        "-f",
        choices=["jsonl", "txt"],
        default="jsonl",
        help="输出格式 (默认: jsonl)",
    )
    parser.add_argument(
        "--keep-toc",
        action="store_true",
        help="保留目录页（默认移除）",
    )
    parser.add_argument(
        "--no-code-preserve",
        action="store_true",
        help="不保留代码块格式",
    )
    parser.add_argument(
        "--no-faq-preserve",
        action="store_true",
        help="不保留 FAQ 结构",
    )

    args = parser.parse_args()

    # 提取
    raw_pages = SoftwareManualCleaner.extract_pages(args.input)

    # 清洗
    cleaner = SoftwareManualCleaner(
        remove_toc_pages=not args.keep_toc,
        preserve_code_blocks=not args.no_code_preserve,
        preserve_faq_pairs=not args.no_faq_preserve,
    )
    cleaned_pages = cleaner.clean(raw_pages)

    # 保存
    SoftwareManualCleaner.save_cleaned_pages(
        cleaned_pages, args.output, format_=args.format
    )


if __name__ == "__main__":
    main()
