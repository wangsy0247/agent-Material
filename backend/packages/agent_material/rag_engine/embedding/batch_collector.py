"""Embedding 批量收集器 — 延迟提交，批量 flush。"""

from __future__ import annotations


class EmbeddingBatchCollector:
    """累积待嵌入文本，最后统一批量提交。

    用于在 Pipeline 处理过程中收集所有需要嵌入的文本，
    在最后一步一次性调用异步 Embedder 批量获取向量。

    Usage:
        collector = EmbeddingBatchCollector(async_embedder)
        collector.add("text1")
        collector.add("text2")
        # ... processing continues ...
        all_embeddings = await collector.flush()
    """

    def __init__(self, async_embedder, auto_flush_size: int = 100):
        self._embedder = async_embedder
        self._texts: list[str] = []
        self._auto_flush_size = auto_flush_size

    def add(self, text: str):
        self._texts.append(text)

    def add_batch(self, texts: list[str]):
        self._texts.extend(texts)

    async def flush(self) -> list[list[float]]:
        """提交所有累积文本，返回嵌入向量列表。"""
        texts = self._texts
        self._texts = []
        if not texts:
            return []
        return await self._embedder.aembed_documents(texts)

    @property
    def pending_count(self) -> int:
        return len(self._texts)

    def clear(self):
        self._texts.clear()
