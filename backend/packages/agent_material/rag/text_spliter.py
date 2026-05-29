"""语义文本分割器。

基于 RecursiveCharacterTextSplitter 做粗分，然后使用 Embedding 模型计算
相邻 chunk 的语义相似度。相似度高于阈值的相邻 chunk 会合并，以保持语义完整性。
"""

from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document


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
