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