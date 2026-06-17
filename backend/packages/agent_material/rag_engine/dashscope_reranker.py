"""DashScope Reranker — 调用阿里云 DashScope 重排序 API（qwen3-rerank）。

替代本地 HuggingFaceCrossEncoder 加载模型，消除对 sentence-transformers /
torch 的依赖，降低内存占用和初始化耗时。

接口兼容 LangChain BaseDocumentCompressor，可直接作为
ContextualCompressionRetriever 的 base_compressor 使用。
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any

import requests
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict

from agent_material.utils.logger import get_logger

logger = get_logger("dashscope_reranker")

# ── DashScope Rerank API ──────────────────────────────────────────
_DASHSCOPE_RERANK_URL = (
    "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
)


class DashScopeReranker(BaseDocumentCompressor):
    """基于 DashScope API 的重排序器。

    使用 qwen3-rerank 模型对候选文档列表精排，按 relevance_score 降序返回 top_n 条。

    Args:
        model: 模型名称，默认 qwen3-rerank（gte-rerank 已下线）
        top_n: 返回文档数量
        api_key: DashScope API Key，默认从环境变量 DASHSCOPE_API_KEY 读取
        max_retries: 失败重试次数
        timeout: 请求超时秒数
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = "qwen3-rerank"
    top_n: int = 5
    api_key: str = ""
    max_retries: int = 3
    timeout: int = 30

    def __init__(
        self,
        model: str = "qwen3-rerank",
        top_n: int = 5,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        resolved_api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        super().__init__(
            model=model,
            top_n=top_n,
            api_key=resolved_api_key,
            max_retries=max_retries,
            timeout=timeout,
        )

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> list[Document]:
        """对候选文档列表重排序，返回 top_n 条。

        Args:
            documents: 待重排序的文档列表
            query: 原始查询文本
            callbacks: LangChain 回调（未使用，保留兼容）

        Returns:
            按 relevance_score 降序排列后的 top_n 文档
        """
        if not documents:
            return []

        if len(documents) == 1:
            return list(documents)

        # 提取文档文本用于 API 请求
        texts = [doc.page_content for doc in documents]

        try:
            scores = self._call_rerank_api(query, texts)
        except Exception as e:
            logger.error(f"[DashScopeReranker] API 调用失败: {e}，回退到原始顺序")
            return list(documents[: self.top_n])

        # 构造 (doc, score) 对并按 score 降序排序
        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        result = [doc for doc, score in scored[: self.top_n]]
        logger.info(
            f"[DashScopeReranker] {len(documents)} docs → top {len(result)} | "
            f"top_score={scored[0][1]:.4f}"
        )
        return result

    def _call_rerank_api(self, query: str, documents: list[str]) -> list[float]:
        """调用 DashScope 重排序 API，返回每条文档的 relevance_score。

        Args:
            query: 查询文本
            documents: 待排序的文档文本列表（≤500 条）

        Returns:
            与 documents 等长的 relevance_score 列表（已按原始索引顺序对齐）
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(self.top_n, len(documents)),
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    _DASHSCOPE_RERANK_URL,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    results: list[dict] = data.get("results", [])
                    # 构建 index → score 映射，未返回的文档 score=0.0
                    score_map: dict[int, float] = {
                        item["index"]: item.get("relevance_score", 0.0)
                        for item in results
                    }
                    return [score_map.get(i, 0.0) for i in range(len(documents))]

                elif resp.status_code == 429:
                    last_error = RuntimeError(f"Rate limited (429): {resp.text[:300]}")
                    wait = min(2**attempt * 2, 30)
                    logger.warning(
                        f"[DashScopeReranker] 请求限流，{wait}s 后重试 "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait)

                elif resp.status_code >= 500:
                    last_error = RuntimeError(f"Server error ({resp.status_code}): {resp.text[:300]}")
                    wait = min(2**attempt, 10)
                    logger.warning(
                        f"[DashScopeReranker] 服务端错误，{wait}s 后重试 "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait)

                else:
                    raise RuntimeError(f"API {resp.status_code}: {resp.text[:300]}")

            except requests.RequestException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait = min(2**attempt, 10)
                    logger.warning(
                        f"[DashScopeReranker] 网络错误，{wait}s 后重试 "
                        f"(attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    time.sleep(wait)

        raise RuntimeError(f"重排序失败，已重试 {self.max_retries} 次: {last_error}")
