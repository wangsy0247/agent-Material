"""语义文本分割器。

基于 RecursiveCharacterTextSplitter 做粗分，然后使用 Embedding 模型计算
相邻 chunk 的语义相似度。相似度高于阈值的相邻 chunk 会合并，以保持语义完整性。
"""
import re
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from agent_material.utils.logger import logger
from agent_material.utils.config_handler import vector_store_config

# ── embedding_model 延迟导入 ─────────────────────────────────────
_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from agent_material.rag_engine.models import embedding_model

            _embedding_model = embedding_model
        except Exception as e:
            logger.warning(f"Embedding 模型加载失败: {e}，语义切分将退化为标准切分")
            _embedding_model = None
    return _embedding_model

class SemanticTextSplitter:
    """基于语义相似度的文本分割器。

    先用 RecursiveCharacterTextSplitter 做初始切分，然后使用 Embedding 模型
    计算相邻 chunk 的语义相似度。如果相邻 chunk 的相似度超过阈值，则合并它们。
    这样可以避免在语义完整的段落中间被切断。

    Args:
        chunk_size: 初始切分的最大块大小
        chunk_overlap: 相邻块之间的重叠字符数
        separator: 切分时优先使用的分隔符列表
        embedding_model: 用于计算语义相似度的 Embedding 模型
        similarity_threshold: 合并阈值（0~1），相似度超过此值则合并
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: Optional[list[str]] = None,
        embedding_model: Optional[Embeddings] = None,
        similarity_threshold: float = 0.7,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator or [
            "\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""
        ]
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separator,
        )

    def split_text(self, text: str) -> list[str]:
        """分割文本成多个片段。

        如果配置了 embedding_model，会对相邻片段做语义相似度判断，
        相似度高于阈值的片段会合并为一个 chunk。

        Args:
            text: 要分割的原始文本

        Returns:
            分割（并可能合并）后的文本片段列表
        """
        chunks = self.splitter.split_text(text)

        if self.embedding_model and len(chunks) > 1:
            chunks = self._merge_by_similarity(chunks)

        return chunks

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """分割 Document 列表。

        先使用 RecursiveCharacterTextSplitter 做标准切分，
        然后对每个 Document 的相邻 chunk 做语义合并。

        Args:
            documents: LangChain Document 对象列表

        Returns:
            切分后的 Document 列表
        """
        split_docs = self.splitter.split_documents(documents)

        if not self.embedding_model or len(split_docs) <= 1:
            return split_docs

        # 按来源分组，对同来源的文档做语义合并
        docs_by_source: dict[str, list[Document]] = {}
        for doc in split_docs:
            source = doc.metadata.get("source", "unknown")
            docs_by_source.setdefault(source, []).append(doc)

        result: list[Document] = []
        for source, docs in docs_by_source.items():
            if len(docs) <= 1:
                result.extend(docs)
                continue

            texts = [d.page_content for d in docs]
            merged_texts = self._merge_by_similarity(texts)

            # 为合并后的文本创建新的 Document，继承原始 metadata
            for i, text in enumerate(merged_texts):
                meta = docs[min(i, len(docs) - 1)].metadata.copy()
                meta["semantic_chunk_index"] = i
                result.append(Document(page_content=text, metadata=meta))

        return result

    def _merge_by_similarity(self, chunks: list[str]) -> list[str]:
        """基于语义相似度合并相邻 chunk。

        使用渐进式合并策略：从第一个 chunk 开始，依次判断当前累积块与下一块
        的相似度。如果相似度高于阈值，则合并；否则将当前块加入结果，
        以下一块作为新的当前块。

        Args:
            chunks: 初始切分后的文本片段列表

        Returns:
            合并后的文本片段列表
        """
        # 批量获取所有 chunk 的 embedding（比逐条调用效率高）
        embeddings = self.embedding_model.embed_documents(chunks)

        merged: list[str] = [chunks[0]]
        current_embedding = embeddings[0]

        for i in range(1, len(chunks)):
            similarity = self._cosine_similarity(current_embedding, embeddings[i])

            if similarity > self.similarity_threshold:
                # 语义相似，合并到当前块
                merged[-1] = merged[-1] + "\n" + chunks[i]
                # 更新 embedding 为合并后文本的 embedding（重新计算）
                current_embedding = self.embedding_model.embed_documents([merged[-1]])[0]
            else:
                # 语义不相似，开启新块
                merged.append(chunks[i])
                current_embedding = embeddings[i]

        return merged

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """计算两个向量的余弦相似度。

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            余弦相似度，范围 [-1, 1]。对于归一化后的 embedding 通常为 [0, 1]
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
        magnitude2 = sum(b ** 2 for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)


def _splitter_cfg(name: str) -> dict:
    """读取切分器专属配置，fallback 到顶层通用值。"""
    top = {
        "chunk_size": vector_store_config.get("chunk_size", 800),
        "chunk_overlap": vector_store_config.get("chunk_overlap", 100),
    }
    section = vector_store_config.get("splitters", {}).get(name, {})
    return {**top, **section}


class BaseSplitter:
    """切分器抽象基类，统一接口 ``split_documents(docs) -> list[Document]``。"""

    def split_documents(self, documents: list[Document]) -> list[Document]:
        raise NotImplementedError


class RecursiveSplitter(BaseSplitter):
    """递归字符切分器 — 保护 LaTeX/代码块完整性，合并孤立标题。

    切分流程：
    1. 占位符保护：LaTeX 公式 ($/$$/\\(\\)/\\[\\])、代码块 (```/~~~) → 占位符
    2. 递归字符切分
    3. 占位符还原
    4. 孤立标题合并 (含标题但 < 40 字符、无句尾标点 → 向后合并)
    """

    # ── 保护区域正则 ──
    _RE_DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
    _RE_INLINE_MATH = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")
    _RE_LATEX_PAREN = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
    _RE_LATEX_BRACKET = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
    _RE_CODE_FENCE = re.compile(r"(```[^\n]*\n.+?```)", re.DOTALL)
    _RE_CODE_TILDE = re.compile(r"(~~~[^\n]*\n.+?~~~)", re.DOTALL)

    # ── 孤立标题检测 ──
    _RE_HEADING = re.compile(r"^#{1,4}\s+")
    _RE_SENTENCE_END = re.compile(r"[。！？\.!\?]$")
    _HEADING_MIN_CONTENT = 40

    def __init__(self):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=vector_store_config.get("chunk_size", 800),
            chunk_overlap=vector_store_config.get("chunk_overlap", 100),
            separators=vector_store_config.get(
                "separator", ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
            ),
            length_function=len,
        )

    # ── 公共接口 ──

    def split_documents(self, documents: list[Document]) -> list[Document]:
        result: list[Document] = []
        for doc in documents:
            chunks = self._split_protected(doc)
            result.extend(chunks)
        return result

    def split_text(self, text: str) -> list[str]:
        """分割纯文本（保护 LaTeX/代码块后切分）。"""
        text, protected = self._protect_regions(text)
        chunks = self._splitter.split_text(text)
        return [self._restore_regions(c, protected) for c in chunks]

    # ── 保护 / 还原 ──

    @classmethod
    def _protect_regions(cls, text: str) -> tuple[str, dict[str, str]]:
        """将文本中的保护区域 (LaTeX, 代码块) 替换为占位符。

        Returns:
            (替换后文本, {占位符: 原始内容})
        """
        protected: dict[str, str] = {}
        counter = 0

        def _replace(m: re.Match, prefix: str) -> str:
            nonlocal counter
            token = f"__{prefix}_{counter}__"
            protected[token] = m.group(0)
            counter += 1
            return token

        # 优先级：代码块 > 显示公式 > 括号公式 > 行内公式
        text = cls._RE_CODE_FENCE.sub(lambda m: _replace(m, "CODE"), text)
        text = cls._RE_CODE_TILDE.sub(lambda m: _replace(m, "CODE"), text)
        text = cls._RE_DISPLAY_MATH.sub(lambda m: _replace(m, "DSPMATH"), text)
        text = cls._RE_LATEX_BRACKET.sub(lambda m: _replace(m, "DSPMATH"), text)
        text = cls._RE_LATEX_PAREN.sub(lambda m: _replace(m, "INMATH"), text)
        text = cls._RE_INLINE_MATH.sub(lambda m: _replace(m, "INMATH"), text)

        return text, protected

    @staticmethod
    def _restore_regions(text: str, protected: dict[str, str]) -> str:
        """将占位符还原为原始内容。"""
        result = text
        for token, original in protected.items():
            result = result.replace(token, original)
        return result

    # ── 核心切分逻辑 ──

    def _split_protected(self, doc: Document) -> list[Document]:
        """保护 → 切分 → 还原 → 合并孤立标题。"""
        text, protected = self._protect_regions(doc.page_content)

        # 递归字符切分
        raw_chunks = self._splitter.split_text(text)

        # 还原保护区域
        restored = [self._restore_regions(c, protected) for c in raw_chunks]

        # 构建 Document 列表
        docs = [
            Document(page_content=c, metadata=dict(doc.metadata))
            for c in restored if c.strip()
        ]

        # 合并孤立标题
        return self._merge_standalone_headings(docs)

    def _merge_standalone_headings(self, chunks: list[Document]) -> list[Document]:
        """将仅含标题的短 chunk 与相邻 chunk 合并。

        规则:
        - chunk 以 # heading 开头 + 总长度 < HEADING_MIN_CONTENT + 无句尾标点
        - → 向后（优先）或向前合并
        """
        if len(chunks) <= 1:
            return chunks

        merged: list[Document] = []
        i = 0
        while i < len(chunks):
            text = chunks[i].page_content.strip()

            if self._is_standalone_heading(text):
                # 尝试向后合并（标题 + 正文）
                if i + 1 < len(chunks):
                    chunks[i + 1] = Document(
                        page_content=chunks[i].page_content + "\n" + chunks[i + 1].page_content,
                        metadata=chunks[i + 1].metadata,
                    )
                    i += 1  # 跳过当前，下一个循环处理合并后的 chunk
                elif merged:
                    # 向前合并（最后一个 chunk 是孤立标题）
                    merged[-1] = Document(
                        page_content=merged[-1].page_content + "\n" + chunks[i].page_content,
                        metadata=merged[-1].metadata,
                    )
                    i += 1
                else:
                    merged.append(chunks[i])
                    i += 1
            else:
                merged.append(chunks[i])
                i += 1

        return merged

    @classmethod
    def _is_standalone_heading(cls, text: str) -> bool:
        """判断文本是否是孤立标题块。"""
        return bool(
            cls._RE_HEADING.match(text)
            and len(text) < cls._HEADING_MIN_CONTENT
            and not cls._RE_SENTENCE_END.search(text.rstrip()[-3:])
        )


class SentenceBoundarySplitter(BaseSplitter):
    """递归切分 + 双向句边界修剪 — 同时保证句首和句尾完整。

    先用 RecursiveSplitter 切分，然后对每个 chunk：
    1. 从开头正向扫描到第一个句首字符 → 裁掉前导碎片
    2. 从末尾反向扫描到最后一个句尾标点 → 裁掉尾部碎片
    3. 裁掉的碎片不传递（依赖 chunk_overlap 保证上下文覆盖）
    4. 修剪后过短的 chunk（< min_chunk_size）与相邻 chunk 合并

    chunk 长度会有 ±200 字符波动，但句首/句尾完整率均可到 80%+。
    """

    _RE_SENTENCE_END = re.compile(r"[。！？\.!\?]")

    def __init__(self):
        cfg = _splitter_cfg("recursive")
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.get("chunk_size", 800),
            chunk_overlap=cfg.get("chunk_overlap", 100),
            separators=vector_store_config.get(
                "separator", ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
            ),
            length_function=len,
        )
        self._max_trim = 200
        self._min_chunk_size = 50  # 修剪后低于此值则合并

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks = self._splitter.split_documents(documents)
        trimmed = self._trim_both_ends(chunks)
        return self._merge_short_chunks(trimmed)

    def _trim_both_ends(self, chunks: list[Document]) -> list[Document]:
        """在句尾标点处对齐 chunk 边界，使句首完整率 = 句尾完整率。

        对每个 chunk（最后一个除外）：
        1. 反向找最后一个句尾标点 → 在此处截断 chunk N
        2. 截断的尾部前传给 chunk N+1
        3. chunk N+1 从截断点之后开始 → 自然以完整句首开头
        """
        if len(chunks) <= 1:
            return chunks

        result: list[Document] = []
        carry = ""  # 前一个 chunk 截断后传给下一个的尾部

        for i, chunk in enumerate(chunks):
            text = carry + chunk.page_content
            carry = ""

            if not text.strip():
                result.append(chunk)
                continue

            # 最后一个 chunk 不修剪
            if i == len(chunks) - 1:
                result.append(Document(page_content=text, metadata=chunk.metadata))
                break

            # 从末尾反向找最后一个句尾标点
            search_start = max(0, len(text) - self._max_trim)
            tail = text[search_start:]
            end_match = None
            for m in self._RE_SENTENCE_END.finditer(tail):
                end_match = m

            if end_match:
                cut = search_start + end_match.end()
                carry = text[cut:].lstrip()   # 尾部传给下一个 chunk
                trimmed = text[:cut]
            else:
                trimmed = text  # 找不到句尾就保留原样

            result.append(Document(page_content=trimmed, metadata=chunk.metadata))

        return result

    def _merge_short_chunks(self, chunks: list[Document]) -> list[Document]:
        """合并过短 chunk 到相邻块。"""
        if len(chunks) <= 1:
            return chunks

        merged: list[Document] = []
        i = 0
        while i < len(chunks):
            cur = chunks[i]
            cur_text = cur.page_content

            # 如果当前太短，尝试向前或向后合并
            if len(cur_text) < self._min_chunk_size:
                if merged:
                    # 合并到前一个 chunk
                    merged[-1] = Document(
                        page_content=merged[-1].page_content + "\n" + cur_text,
                        metadata=merged[-1].metadata,
                    )
                elif i + 1 < len(chunks):
                    # 向前合并
                    merged.append(Document(
                        page_content=cur_text + "\n" + chunks[i + 1].page_content,
                        metadata=chunks[i + 1].metadata,
                    ))
                    i += 1  # 跳过一个
                else:
                    merged.append(cur)
            else:
                merged.append(cur)
            i += 1

        return merged


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
            splitter = SemanticTextSplitter(
                chunk_size=vector_store_config.get("chunk_size", 800),
                chunk_overlap=vector_store_config.get("chunk_overlap", 100),
                separator=vector_store_config.get("separator", ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]),
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
    """父子文档切分器 — 用于多粒度索引。

    先用大 chunk_size 切出"父文档"（~2000 字符），再用小 chunk_size 切出"子文档"（~500 字符）。
    子文档 metadata 保留 ``parent_doc`` / ``parent_id`` 引用，供 MultiGranularityIndex 使用。
    parent_id 使用文本的 MD5 哈希（跨进程稳定）。
    """

    def __init__(self):
        import hashlib
        self._hashlib = hashlib
        cfg = _splitter_cfg("parent")
        parent_chunk_size = cfg.get("parent_chunk_size", 2000)
        child_chunk_size = cfg.get("child_chunk_size", 500)
        separators = vector_store_config.get(
            "separator", ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
        )
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=200,
            separators=separators,
            length_function=len,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=70,
            separators=separators,
            length_function=len,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        result: list[Document] = []
        for doc in documents:
            parents = self.parent_splitter.split_documents([doc])
            for parent in parents:
                children = self.child_splitter.split_documents([parent])
                parent_id = self._hashlib.md5(
                    parent.page_content.encode("utf-8")
                ).hexdigest()[:16]
                for child in children:
                    child.metadata["parent_doc"] = parent.page_content
                    child.metadata["parent_id"] = parent_id
                    result.append(child)
        logger.info(f"[Splitter] ParentDocumentSplitter 切分完成 | {len(result)} chunks")
        return result


class ParentFirstSplitter(BaseSplitter):
    """父块优先切分器 — 按标题结构先构造父块，再在父块内部切子块（OFF-005）。

    切分流程：
    1. 按 Markdown 标题（# ~ ####）把文档切成"章节"，维护每段的标题路径
    2. 章节文本 ≤ 父块尺寸 → 直接成为一个父块；
       超长章节在章节内部递归切成 ~parent_chunk_size 字符父块
       （复用 RecursiveSplitter 的占位符保护，公式/代码块不被截断）
    3. 每个父块内部切 ~child_chunk_size 字符子块（overlap 默认 70，可配置）
    4. 孤立标题与相邻内容合并，不单独成为无正文父块

    配置键（config/vector_store_config.yaml → splitters.parent）：
        parent_chunk_size: 父块尺寸（默认 2000）
        child_chunk_size: 子块尺寸（默认 500）
        child_chunk_overlap: 子块重叠（默认 70）
        parent_chunk_overlap: 超长章节内部递归切分的重叠（默认 200）
        semantic_merge_within_parent: 预留项 — 同一父块内的语义合并开关。
            语义合并且仅当此配置开启时在同一父块内做（不得跨父块），
            当前版本暂未实现，保留配置位以便后续启用。

    每个子块 metadata 写入：
        parent_id: 父块内容 MD5 前 16 位（占位，管道接入 version_context 后
                   会被 compute_parent_id 重算为确定性 ID）
        parent_doc: 父块全文（保持与 MultiGranularityIndex.insert 的契约）
        parent_metadata: {heading_path, char_start, char_end, page_*}
        heading_path: 继承父块的标题路径（list[str]）
        char_start / char_end: 相对整个输入文档拼接文本的字符偏移
        page_start / page_end: 输入 doc metadata 含 page 信息时填写；
            否则 page_unknown=True，绝不伪造页码
    """

    # Markdown 标题行（# ~ ####），MULTILINE 支持剥离多行文本中的标题行
    _RE_MD_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)

    def __init__(self):
        import hashlib
        self._hashlib = hashlib
        cfg = _splitter_cfg("parent")
        self._parent_chunk_size = cfg.get("parent_chunk_size", 2000)
        self._child_chunk_size = cfg.get("child_chunk_size", 500)
        self._child_chunk_overlap = cfg.get("child_chunk_overlap", 70)
        self._parent_chunk_overlap = cfg.get("parent_chunk_overlap", 200)
        # 预留：同一父块内的语义合并开关（暂未实现，见类 docstring）
        self._semantic_merge_within_parent = cfg.get("semantic_merge_within_parent", False)
        separators = vector_store_config.get(
            "separator", ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
        )
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._parent_chunk_size,
            chunk_overlap=self._parent_chunk_overlap,
            separators=separators,
            length_function=len,
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._child_chunk_size,
            chunk_overlap=self._child_chunk_overlap,
            separators=separators,
            length_function=len,
        )
        # 复用 RecursiveSplitter 的孤立标题合并逻辑
        self._heading_merger = RecursiveSplitter()

    # ── 公共接口 ──

    def split_documents(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []

        # 拼接整个输入文档文本，记录各 doc 的偏移区间与页码
        full_text, segments = self._concat_documents(documents)
        base_meta = dict(documents[0].metadata)
        has_page = all(isinstance(seg[2], int) for seg in segments)

        result: list[Document] = []
        for heading_path, sec_start, sec_end in self._split_sections(full_text):
            section_text = full_text[sec_start:sec_end]
            # 章节 → 一个或多个父块
            for p_text, p_start, p_end in self._split_parents(
                section_text, sec_start
            ):
                parent_id = self._hashlib.md5(
                    p_text.encode("utf-8")
                ).hexdigest()[:16]
                parent_meta = {
                    "heading_path": list(heading_path),
                    "char_start": p_start,
                    "char_end": p_end,
                    **self._page_info(p_start, p_end, segments, has_page),
                }
                # 父块内部切子块
                for c_text, c_start, c_end in self._split_children(p_text, p_start):
                    meta = {
                        **base_meta,
                        "parent_id": parent_id,
                        "parent_doc": p_text,
                        "parent_metadata": parent_meta,
                        "heading_path": list(heading_path),
                        "char_start": c_start,
                        "char_end": c_end,
                        **self._page_info(c_start, c_end, segments, has_page),
                    }
                    result.append(Document(page_content=c_text, metadata=meta))
        logger.info(
            f"[Splitter] ParentFirstSplitter 切分完成 | {len(result)} chunks, "
            f"{len({d.metadata['parent_id'] for d in result})} parents"
        )
        return result

    # ── 文档拼接与页码 ──

    @staticmethod
    def _concat_documents(
        documents: list[Document],
    ) -> tuple[str, list[tuple[int, int, object]]]:
        """将多个输入 doc 用空行拼接为单一文本。

        Returns:
            (拼接文本, [(start, end, page), ...])，page 为 doc.metadata["page"]（可缺）
        """
        parts: list[str] = []
        segments: list[tuple[int, int, object]] = []
        pos = 0
        for doc in documents:
            if parts:
                parts.append("\n\n")
                pos += 2
            parts.append(doc.page_content)
            segments.append((pos, pos + len(doc.page_content), doc.metadata.get("page")))
            pos += len(doc.page_content)
        return "".join(parts), segments

    @staticmethod
    def _page_info(
        start: int, end: int, segments: list[tuple[int, int, object]], has_page: bool
    ) -> dict:
        """根据字符区间计算页码范围；无 page 信息时显式标记 page_unknown。"""
        if not has_page:
            return {"page_unknown": True}
        pages = [
            seg[2] for seg in segments if seg[0] < end and start < seg[1]
        ]
        if not pages:
            return {"page_unknown": True}
        return {"page_start": min(pages), "page_end": max(pages)}

    # ── 章节切分（标题结构） ──

    def _split_sections(self, text: str) -> list[tuple[list[str], int, int]]:
        """按 Markdown 标题把文本切成章节。

        Returns:
            [(heading_path, char_start, char_end), ...]，区间相对整个输入文本，
            章节文本包含其起始标题行；首个标题之前的内容 heading_path 为空列表。
        """
        sections: list[tuple[list[str], int, int]] = []
        heading_stack: list[tuple[int, str]] = []  # (级别, 标题)
        current_path: list[str] = []
        current_start = 0
        pos = 0
        for line in text.splitlines(keepends=True):
            m = self._RE_MD_HEADING.match(line.rstrip("\n"))
            if m:
                if pos > current_start:
                    sections.append((current_path, current_start, pos))
                level = len(m.group(1))
                # 弹出同级及更深标题，压入新标题
                heading_stack = [(lv, t) for lv, t in heading_stack if lv < level]
                heading_stack.append((level, m.group(2).strip()))
                current_path = [t for _, t in heading_stack]
                current_start = pos
            pos += len(line)
        if pos > current_start:
            sections.append((current_path, current_start, pos))

        return self._merge_standalone_sections(sections, text)

    def _merge_standalone_sections(
        self, sections: list[tuple[list[str], int, int]], text: str
    ) -> list[tuple[list[str], int, int]]:
        """将"只有标题没有正文"的章节向后（优先）或向前合并。"""
        if len(sections) <= 1:
            return sections

        merged: list[tuple[list[str], int, int]] = []
        i = 0
        while i < len(sections):
            path, start, end = sections[i]
            sec_text = text[start:end].strip()
            # 去掉标题行后无正文，或整段是孤立标题 → 需要合并
            body = self._RE_MD_HEADING.sub("", sec_text).strip()
            if (not body or RecursiveSplitter._is_standalone_heading(sec_text)) and (
                i + 1 < len(sections) or merged
            ):
                if i + 1 < len(sections):
                    # 向后合并：章节区间连续，直接扩展当前章节到下一章节起点
                    n_path, _n_start, n_end = sections[i + 1]
                    sections[i + 1] = (n_path or path, start, n_end)
                else:
                    # 最后一个章节是孤立标题 → 向前合并
                    p_path, p_start, _p_end = merged[-1]
                    merged[-1] = (p_path, p_start, end)
                i += 1
            else:
                merged.append((path, start, end))
                i += 1
        return merged

    # ── 父块 / 子块切分（占位符保护 + 偏移定位） ──

    def _split_parents(
        self, section_text: str, base_offset: int
    ) -> list[tuple[str, int, int]]:
        """章节 → 父块列表。短章节整体成块，超长章节保护后递归切分。"""
        if len(section_text) <= self._parent_chunk_size:
            stripped = section_text.strip()
            if not stripped:
                return []
            start = section_text.find(stripped)
            return [(stripped, base_offset + start, base_offset + start + len(stripped))]

        protected_text, protected = RecursiveSplitter._protect_regions(section_text)
        raw_chunks = self._parent_splitter.split_text(protected_text)
        restored = [RecursiveSplitter._restore_regions(c, protected) for c in raw_chunks]
        docs = [
            Document(page_content=c, metadata={}) for c in restored if c.strip()
        ]
        # 父块内同样不允许孤立标题单独成块
        docs = self._heading_merger._merge_standalone_headings(docs)
        return self._locate_chunks(docs, section_text, base_offset)

    def _split_children(
        self, parent_text: str, base_offset: int
    ) -> list[tuple[str, int, int]]:
        """父块 → 子块列表（占位符保护，overlap 可配置）。"""
        protected_text, protected = RecursiveSplitter._protect_regions(parent_text)
        raw_chunks = self._child_splitter.split_text(protected_text)
        restored = [RecursiveSplitter._restore_regions(c, protected) for c in raw_chunks]
        docs = [
            Document(page_content=c, metadata={}) for c in restored if c.strip()
        ]
        docs = self._heading_merger._merge_standalone_headings(docs)
        return self._locate_chunks(docs, parent_text, base_offset)

    @staticmethod
    def _locate_chunks(
        docs: list[Document], source_text: str, base_offset: int
    ) -> list[tuple[str, int, int]]:
        """在原文中定位每个 chunk 的字符偏移。

        游标单调前进（允许 overlap）；孤立标题合并等导致文本不是原文精确
        子串时沿用游标兜底。偏移始终钳制在源文本范围内，不越界。
        """
        located: list[tuple[str, int, int]] = []
        cursor = 0
        for doc in docs:
            text = doc.page_content
            idx = source_text.find(text, cursor)
            if idx < 0:
                idx = cursor  # 兜底：不伪造精确位置，沿用游标
            start = base_offset + idx
            end = min(start + len(text), base_offset + len(source_text))
            located.append((text, start, end))
            cursor = max(cursor, idx + 1)
        return located


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


class SemanticMergeOnly(BaseSplitter):
    """仅语义合并 — 对已切分的 chunk 做相邻相似度合并，不进行文本切分。

    用于 PDF 两段式流水线: recursive → semantic_merge，
    先递归字符切分保证块大小均匀，再语义合并保持语义完整。

    优化：合并判断使用加权平均 embedding 近似，避免逐条 API 重嵌入。
    下游 PGVector.add_documents() 会为最终合并文本重新 embedding。
    """

    def __init__(self):
        cfg = _splitter_cfg("semantic_merge")
        self._similarity_threshold = cfg.get("similarity_threshold", 0.75)

    def split_documents(self, documents: list[Document]) -> list[Document]:
        embedding_model = _get_embedding_model()
        if not embedding_model or len(documents) <= 1:
            return documents

        texts = [d.page_content for d in documents]
        try:
            embeddings = embedding_model.embed_documents(texts)
        except Exception as e:
            logger.warning(f"语义合并 Embedding 失败: {e}，跳过合并")
            return documents

        # 首条初始化
        merged: list[tuple[str, dict]] = [
            (texts[0], dict(documents[0].metadata))
        ]
        current_emb = embeddings[0]
        current_len = len(texts[0])

        # 合并判断：加权平均 embedding 近似，无需 API 重嵌入
        for i in range(1, len(texts)):
            sim = SemanticTextSplitter._cosine_similarity(current_emb, embeddings[i])
            if sim > self._similarity_threshold:
                merged[-1] = (
                    merged[-1][0] + "\n" + texts[i],
                    merged[-1][1],
                )
                w1 = current_len / (current_len + len(texts[i]) + 1)
                w2 = len(texts[i]) / (current_len + len(texts[i]) + 1)
                current_emb = [w1 * a + w2 * b for a, b in zip(current_emb, embeddings[i])]
                current_len = current_len + len(texts[i]) + 1
            else:
                merged.append((texts[i], dict(documents[i].metadata)))
                current_emb = embeddings[i]
                current_len = len(texts[i])

        result = [
            Document(page_content=t, metadata=m)
            for t, m in merged
        ]
        logger.info(
            f"[Splitter] SemanticMergeOnly: {len(documents)} → {len(result)} chunks "
            f"(threshold={self._similarity_threshold})"
        )
        return result


class HybridSplitter(BaseSplitter):
    """混合切分器 — 按配置的步骤链式执行多种切分策略。

    配置示例 (vector_store_config.yaml):
        extension_pipeline:
          ".pdf":
            - recursive          # 步骤1: 递归字符切分
            - semantic_merge     # 步骤2: 语义合并相似 chunk
          ".md":
            - markdown_header    # 步骤1: 标题切分（内建递归切分）

    每个步骤的输出 metadata 会传递给下一步骤。
    """

    # 步骤名称 → 工厂函数
    STEP_FACTORY: dict[str, type[BaseSplitter]] = {
        "recursive": RecursiveSplitter,
        "sentence_boundary": SentenceBoundarySplitter,
        "semantic_merge": SemanticMergeOnly,
        "markdown_header": MarkdownHeaderSplitter,
        "html_header": HTMLHeaderSplitter,
        "parent": ParentDocumentSplitter,
    }

    def __init__(self, steps: list[str]):
        self._steps: list[BaseSplitter] = []
        for name in steps:
            factory = self.STEP_FACTORY.get(name)
            if factory is None:
                logger.warning(f"[HybridSplitter] 未知步骤: {name}，跳过")
                continue
            self._steps.append(factory())

    def split_documents(self, documents: list[Document]) -> list[Document]:
        result = documents
        for step in self._steps:
            result = step.split_documents(result)
        logger.info(
            f"[HybridSplitter] {len(documents)} → {len(result)} chunks "
            f"({[s.__class__.__name__ for s in self._steps]})"
        )
        return result