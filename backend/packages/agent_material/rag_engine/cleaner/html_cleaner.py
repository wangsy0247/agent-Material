"""HTML 数据清洗器。

独立于现有 RAG 流程，可单独使用。
适用于网页补充材料、在线数据库 HTML 报告、文档站点等。

Usage:
    python html_cleaner.py --input ./raw_pages/ --output ./cleaned_html.jsonl
"""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
import logging
from bs4 import BeautifulSoup, NavigableString, Tag

@dataclass
class HTMLCleaningStats:
    """HTML 清洗统计。"""

    documents_input: int = 0
    documents_output: int = 0
    scripts_removed: int = 0
    styles_removed: int = 0
    nav_footers_removed: int = 0
    noise_elements_removed: int = 0
    empty_docs_removed: int = 0
    short_docs_removed: int = 0


class HTMLCleaner:
    """HTML 文档专用清洗器。

    基于 BeautifulSoup 解析 HTML DOM 树，精确移除噪声元素，保留正文语义结构。
    """

    # 完全删除的标签（不保留内容）
    REMOVE_TAGS = [
        "script", "style", "noscript", "iframe", "canvas",
        "svg", "video", "audio", "embed", "object",
        "template", "portal",
    ]

    # 结构性噪音元素（完全删除）
    REMOVE_STRUCTURAL = [
        "nav", "footer", "header", "aside",
    ]

    # 常见噪音 class/id 关键词（模糊匹配）
    NOISE_KEYWORDS = [
        "ad", "ads", "advertisement", "banner",
        "sidebar", "widget", "popup", "modal", "overlay",
        "comment", "comments", "disqus", "giscus",
        "share", "social", "follow", "like",
        "breadcrumb", "pagination", "pager",
        "cookie", "consent", "gdpr", "ccpa",
        "related", "recommended", "suggested", "popular",
        "newsletter", "subscribe", "mailing", "signup",
        "download", "print", "toolbar", "actions",
        "sticky", "fixed", "float",
    ]

    # Sphinx / ReadTheDocs 特有噪音 class
    SPHINX_NOISE_CLASSES = [
        "headerlink",
        "wy-breadcrumbs",
        "rst-footer-buttons",
        "toctree-wrapper",
    ]

    def __init__(
        self,
        min_text_length: int = 50,
        remove_noise_by_class: bool = True,
        preserve_tables: bool = True,
        preserve_code_blocks: bool = True,
        remove_empty_headings: bool = True,
    ):

        self.min_text_length = min_text_length
        self.remove_noise_by_class = remove_noise_by_class
        self.preserve_tables = preserve_tables
        self.preserve_code_blocks = preserve_code_blocks
        self.remove_empty_headings = remove_empty_headings
        self.stats = HTMLCleaningStats()

    # ------------------------------------------------------------------ #
    #  主入口
    # ------------------------------------------------------------------ #
    def clean(self, documents: list[dict]) -> list[dict]:
        """批量清洗 HTML 文档。

        Args:
            documents: 列表元素为 dict，格式 {"source": "url或路径", "content": "html字符串"}

        Returns:
            清洗后的文档列表，每个元素增加 "cleaned_text" 和 "tags" 字段
        """
        self.stats = HTMLCleaningStats()
        self.stats.documents_input = len(documents)

        results: list[dict] = []
        for doc in documents:
            cleaned, tags = self._clean_single(doc["content"])
            if len(cleaned) >= self.min_text_length:
                results.append(
                    {
                        "source": doc.get("source", "unknown"),
                        "original_content": doc["content"],
                        "cleaned_text": cleaned,
                        "tags": tags,
                    }
                )
            else:
                self.stats.short_docs_removed += 1

        self.stats.documents_output = len(results)
        self._log_stats()
        return results

    def clean_html(self, html_content: str) -> str:
        """清洗单个 HTML 字符串，返回纯文本。"""
        cleaned, _ = self._clean_single(html_content)
        return cleaned

    # ------------------------------------------------------------------ #
    #  单文档清洗
    # ------------------------------------------------------------------ #
    def _clean_single(self, html_content: str) -> tuple[str, list[str]]:
        """清洗单个 HTML，返回 (纯文本, 标签列表)。"""
        tags: list[str] = []
        soup = BeautifulSoup(html_content, "html.parser")

        # Step 1: 移除 script, style, noscript 等（完全删除）
        for tag_name in self.REMOVE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()
                if tag_name == "script":
                    self.stats.scripts_removed += 1
                elif tag_name == "style":
                    self.stats.styles_removed += 1

        # Step 2: 移除导航、页脚等结构性元素
        for tag_name in self.REMOVE_STRUCTURAL:
            for tag in soup.find_all(tag_name):
                tag.decompose()
                self.stats.nav_footers_removed += 1

        # Step 3: 根据 class/id 移除噪音元素
        if self.remove_noise_by_class:
            for tag in list(soup.find_all()):
                if self._is_noise_element(tag):
                    tag.decompose()
                    self.stats.noise_elements_removed += 1

        # Step 4: 移除 Sphinx / RTD 特有噪音
        for cls in self.SPHINX_NOISE_CLASSES:
            for el in soup.find_all(class_=cls):
                el.decompose()
                self.stats.noise_elements_removed += 1

        # 移除 MathJax 配置公式（如 \renewcommand{\AA}）
        for el in soup.find_all("span", class_="math notranslate nohighlight"):
            text = el.get_text(strip=True)
            if "renewcommand" in text or "newcommand" in text:
                el.decompose()
                self.stats.noise_elements_removed += 1

        # Step 5: 提取正文
        text = self._extract_main_text(soup)

        # Step 6: 后处理
        text = self._post_process(text)

        # Step 7: 标签标记（在原始 soup 上检测，因为部分标签可能已被 decompose，
        # 但 soup 本身仍保留结构；这里用新的 soup 检查更安全）
        soup_check = BeautifulSoup(html_content, "html.parser")
        if soup_check.find("table"):
            tags.append("has_table")
        if soup_check.find("pre") or soup_check.find("code"):
            tags.append("has_code")
        if soup_check.find("img"):
            tags.append("has_image")

        return text, tags

    # ------------------------------------------------------------------ #
    #  检测与提取辅助
    # ------------------------------------------------------------------ #
    def _is_noise_element(self, tag) -> bool:
        """判断元素是否为噪音。"""
        if not isinstance(tag, Tag):
            return False
        if tag.name in ("body", "html", "main", "article"):
            return False

        try:
            classes = tag.get("class", [])
            if isinstance(classes, str):
                classes = [classes]
            class_str = " ".join(classes).lower() if classes else ""
            id_str = (tag.get("id") or "").lower()
            role_str = (tag.get("role") or "").lower()
        except (AttributeError, TypeError):
            return False

        for keyword in self.NOISE_KEYWORDS:
            if keyword in class_str or keyword in id_str or keyword in role_str:
                return True
        return False

    def _extract_main_text(self, soup: BeautifulSoup) -> str:
        """从 soup 中提取主要文本内容。"""
        # 优先尝试语义化标签
        main_selectors = [
            "main",
            "article",
            '[role="main"]',
            ".wy-nav-content",
            ".content",
            ".main-content",
            "#content",
            "#main-content",
            ".post-content",
            ".entry-content",
            ".documentation",
        ]

        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                return self._element_to_text(main_content)

        # 退而求其次：取 body
        body = soup.find("body")
        if body:
            return self._element_to_text(body)

        # 最后手段
        return soup.get_text(separator="\n", strip=True)

    def _element_to_text(self, element) -> str:
        """将 BeautifulSoup 元素递归转换为格式化的纯文本。"""
        parts: list[str] = []

        for child in element.children:
            if isinstance(child, NavigableString):
                parts.append(str(child))
            elif child.name in ("script", "style", "noscript"):
                continue
            elif child.name == "br":
                parts.append("\n")
            elif child.name in ("p", "div", "section", "article", "blockquote", "aside"):
                inner = self._element_to_text(child)
                if inner.strip():
                    parts.append("\n\n")
                    parts.append(inner)
                    parts.append("\n\n")
            elif child.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(child.name[1])
                inner = self._element_to_text(child)
                if inner.strip():
                    parts.append(f"\n\n{'#' * level} ")
                    parts.append(inner.strip())
                    parts.append("\n\n")
            elif child.name == "pre":
                inner = child.get_text()
                if inner.strip():
                    parts.append("\n\n```\n")
                    parts.append(inner.rstrip("\n"))
                    parts.append("\n```\n\n")
            elif child.name == "li":
                inner = self._element_to_text(child)
                if inner.strip():
                    parts.append("\n- ")
                    parts.append(inner.strip())
            elif child.name in ("ul", "ol"):
                inner = self._element_to_text(child)
                if inner.strip():
                    parts.append("\n")
                    parts.append(inner)
                    parts.append("\n")
            elif child.name == "table":
                if self.preserve_tables:
                    inner = self._table_to_markdown(child)
                    if inner.strip():
                        parts.append("\n\n")
                        parts.append(inner)
                        parts.append("\n\n")
                else:
                    inner = self._element_to_text(child)
                    if inner.strip():
                        parts.append("\n\n")
                        parts.append(inner)
                        parts.append("\n\n")
            elif child.name in (
                "a", "em", "strong", "b", "i", "code",
                "span", "sub", "sup", "font", "small",
            ):
                inner = self._element_to_text(child)
                parts.append(inner)
            else:
                inner = self._element_to_text(child)
                parts.append(inner)

        return "".join(parts)

    @staticmethod
    def _table_to_markdown(table: Tag) -> str:
        """将单个 <table> 转换为 Markdown 表格。"""
        rows: list[str] = []
        header_done = False

        for tr in table.find_all("tr"):
            cells = []
            for cell in tr.find_all(["td", "th"]):
                cell_text = cell.get_text(strip=True)
                cells.append(cell_text)

            if cells:
                row_str = "| " + " | ".join(cells) + " |"
                rows.append(row_str)

                if not header_done and tr.find("th"):
                    separator = "|" + "|".join([" --- " for _ in cells]) + "|"
                    rows.append(separator)
                    header_done = True

        return "\n".join(rows)

    def _post_process(self, text: str) -> str:
        """文本后处理：实体解码、空白规范化等。"""
        # HTML 实体解码
        text = html.unescape(text)

        # 兜底：移除 MathJax 宏定义（如 \renewcommand{\AA}{\text{Å}}）
        _macro_pattern = (
            re.escape(r"\(")
            + r"\s*"
            + re.escape(r"\renewcommand")
            + r"\s*"
            + re.escape(r"{")
            + r"\s*"
            + re.escape(r"\AA")
            + r"\s*"
            + re.escape(r"}")
            + r"\s*"
            + re.escape(r"{")
            + r"\s*"
            + re.escape(r"\text")
            + r"\s*"
            + re.escape(r"{")
            + r"\s*"
            + r"[ÅÅ]"
            + r"\s*"
            + re.escape(r"}")
            + r"\s*"
            + re.escape(r"}")
            + r"\s*"
            + re.escape(r"\)")
        )
        text = re.sub(_macro_pattern, "", text)

        # 规范化空行（保留代码块内部结构）
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 保护代码块中的空格不被压缩
        code_blocks: list[str] = []

        def _replace_code(m: re.Match) -> str:
            code_blocks.append(m.group(1))
            return f"\n```CODE{len(code_blocks) - 1}```\n"

        text = re.sub(r"\n```\n(.*?)\n```\n", _replace_code, text, flags=re.DOTALL)

        # 压缩正文中的连续空格
        text = re.sub(r" {2,}", " ", text)

        # 恢复代码块
        for i, code in enumerate(code_blocks):
            text = text.replace(
                f"\n```CODE{i}```\n",
                f"\n```\n{code}\n```\n",
            )

        # 去除孤立 URL 行
        text = re.sub(r"\nhttps?://\S+\n", "\n", text)

        # 宏定义移除后可能残留的首尾空行
        text = text.strip()
        text = re.sub(r"\n{2,}", "\n\n", text)

        return text.strip()

    # ------------------------------------------------------------------ #
    #  表格特化处理
    # ------------------------------------------------------------------ #
    @staticmethod
    def extract_tables(html_content: str) -> list[str]:
        """从 HTML 中提取所有表格，转换为 Markdown 格式。

        Returns:
            每个表格的 Markdown 字符串列表
        """
        soup = BeautifulSoup(html_content, "html.parser")
        tables_md: list[str] = []

        for table in soup.find_all("table"):
            rows: list[str] = []
            header_done = False

            for tr in table.find_all("tr"):
                cells = []
                for cell in tr.find_all(["td", "th"]):
                    cell_text = cell.get_text(strip=True)
                    cells.append(cell_text)

                if cells:
                    row_str = "| " + " | ".join(cells) + " |"
                    rows.append(row_str)

                    # 表头后插入分隔线
                    if not header_done and tr.find("th"):
                        separator = "|" + "|".join([" --- " for _ in cells]) + "|"
                        rows.append(separator)
                        header_done = True

            if rows:
                tables_md.append("\n".join(rows))

        return tables_md

    # ------------------------------------------------------------------ #
    #  统计与输出
    # ------------------------------------------------------------------ #
    def _log_stats(self) -> None:
        """打印清洗统计。"""
        s = self.stats
        logging.info("[HTMLCleaner] 清洗统计:")
        logging.info(f"  输入文档: {s.documents_input}")
        logging.info(f"  输出文档: {s.documents_output}")
        logging.info(f"  脚本移除: {s.scripts_removed}")
        logging.info(f"  样式移除: {s.styles_removed}")
        logging.info(f"  导航/页脚移除: {s.nav_footers_removed}")
        logging.info(f"  噪音元素移除: {s.noise_elements_removed}")
        logging.info(f"  过短过滤: {s.short_docs_removed}")

    @staticmethod
    def save_cleaned_documents(
        documents: list[dict],
        output_path: str,
        format_: str = "jsonl",
    ) -> None:
        """保存清洗后的 HTML 文档。"""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if format_ == "jsonl":
            with open(out_path, "w", encoding="utf-8") as f:
                for doc in documents:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        elif format_ == "txt":
            with open(out_path, "w", encoding="utf-8") as f:
                for doc in documents:
                    f.write(f"\n{'=' * 60}\n")
                    f.write(f"来源: {doc.get('source', 'N/A')}\n")
                    f.write(f"{'=' * 60}\n")
                    f.write(doc["cleaned_text"] + "\n")
        else:
            raise ValueError(f"不支持的格式: {format_}")

        logging.info(f"[save] 已保存 {len(documents)} 条到 {output_path}")


# ==================================================================== #
#  命令行接口
# ==================================================================== #
def main():
    parser = argparse.ArgumentParser(description="HTML 数据清洗工具")
    parser.add_argument("--input", "-i", required=True, help="输入目录或 HTML 文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出文件路径")
    parser.add_argument(
        "--format", "-f", choices=["jsonl", "txt"], default="jsonl", help="输出格式"
    )
    parser.add_argument(
        "--min-length", type=int, default=50, help="最小文本长度过滤"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    docs: list[dict] = []

    if input_path.is_file():
        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read()
        docs.append({"source": str(input_path), "content": content})
    else:
        for html_file in input_path.rglob("*.html"):
            with open(html_file, "r", encoding="utf-8") as f:
                content = f.read()
            docs.append({"source": str(html_file), "content": content})

    cleaner = HTMLCleaner(min_text_length=args.min_length)
    cleaned = cleaner.clean(docs)
    HTMLCleaner.save_cleaned_documents(cleaned, args.output, format_=args.format)


if __name__ == "__main__":
    main()
