"""异步 Embedding 模型 — 异步批量 + 并发控制 + 重试。

所有 asyncio 对象在每个 aembed_documents() 调用中局部创建，
确保与 asyncio.run() 每次创建的新 event loop 兼容。
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Optional

from rag_mcp_server.utils.logger import logger


class AsyncDashScopeEmbeddings:
    """DashScope text-embedding-v4 异步封装。

    特性：
    - 批量发送（batch_size 可配，≤10）
    - 分组并发（每组 max_concurrency 个请求并发）
    - 指数退避重试（最多 3 次）
    - 同步降级接口 embed_documents() 兼容 LangChain PGVector

    注意：不存储任何 asyncio 同步原语为实例属性（Semaphore 等），
    避免 "bound to a different event loop" 错误。
    """

    API_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"

    def __init__(
        self,
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
        api_key: Optional[str] = None,
        batch_size: int = 10,
        max_concurrency: int = 4,
        rate_limit: float = 500.0,
        max_retries: int = 3,
    ):
        self.model = model
        self.dimensions = dimensions
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.batch_size = min(batch_size, 10)  # DashScope 限制 ≤10
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self._request_delay = 1.0 / rate_limit if rate_limit > 0 else 0.0

    # ── 异步入口 ──────────────────────────────────────────

    async def aembed_documents(
        self, texts: list[str], text_type: str = "document"
    ) -> list[list[float]]:
        """异步批量嵌入文本列表。

        分组并发：将批次按 max_concurrency 分组，组内并发、组间串行。
        """
        if not texts:
            return []

        total = len(texts)
        logger.info(
            f"[AsyncEmbed] 开始嵌入 {total} 条 | batch={self.batch_size} "
            f"| concurrency={self.max_concurrency} | type={text_type}"
        )

        result: list[Optional[list[float]]] = [None] * total

        # 将文本切分为批次（每批 ≤ batch_size）
        batches: list[tuple[int, list[str]]] = []
        for i in range(0, total, self.batch_size):
            batches.append((i, texts[i : i + self.batch_size]))

        # 分组并发：每组最多 max_concurrency 个批次并发执行
        for g in range(0, len(batches), self.max_concurrency):
            group = batches[g : g + self.max_concurrency]
            tasks = [
                self._embed_slot(batch, start_idx, result, text_type)
                for start_idx, batch in group
            ]
            await asyncio.gather(*tasks)

        valid = [r for r in result if r is not None]
        if len(valid) != total:
            logger.error(f"[AsyncEmbed] 结果不完整: {len(valid)}/{total}")
            raise RuntimeError(f"嵌入结果不完整: {len(valid)}/{total}")

        logger.info(f"[AsyncEmbed] 嵌入完成 | {total} 条 → {len(valid)} 个向量")
        return valid  # type: ignore[return-value]

    async def aembed_query(self, text: str) -> list[float]:
        """异步嵌入单个查询文本（使用 text_type=query）。"""
        if not text:
            return []
        embeddings = await self._embed_with_retry([text], text_type="query")
        return embeddings[0]

    # ── 内部方法 ──────────────────────────────────────────

    async def _embed_slot(
        self,
        texts: list[str],
        start_idx: int,
        result: list,
        text_type: str,
    ):
        """嵌入一批文本并写入结果槽位。"""
        try:
            embeddings = await self._embed_with_retry(texts, text_type)
            for j, emb in enumerate(embeddings):
                result[start_idx + j] = emb
        except Exception as e:
            logger.error(
                f"[AsyncEmbed] 批次失败 (start={start_idx}, size={len(texts)}): {e}"
            )
            raise

    async def _embed_with_retry(
        self, texts: list[str], text_type: str
    ) -> list[list[float]]:
        """带指数退避重试的单批次嵌入请求。"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await self._do_request(texts, text_type)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait = min(2**attempt * 2, 30)
                    logger.warning(
                        f"[AsyncEmbed] 请求失败，{wait}s 后重试 "
                        f"(attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    await asyncio.sleep(wait)

        raise RuntimeError(
            f"嵌入失败，已重试 {self.max_retries} 次: {last_error}"
        )

    async def _do_request(
        self, texts: list[str], text_type: str
    ) -> list[list[float]]:
        """执行单次 HTTP 请求（DashScope 原生 API）。"""
        import aiohttp

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": {"texts": texts},
            "parameters": {"text_type": text_type},
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.API_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    embeds = data["output"]["embeddings"]
                    embeds.sort(key=lambda x: x["text_index"])
                    return [item["embedding"] for item in embeds]
                elif resp.status == 429:
                    raise RuntimeError("Rate limited (429)")
                else:
                    body = await resp.text()
                    raise RuntimeError(f"API {resp.status}: {body[:300]}")

    # ── 同步兼容接口 ──────────────────────────────────────

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """同步嵌入（兼容 LangChain Embeddings 协议）。

        Worker 线程中无事件循环 → asyncio.run() 创建新循环。
        每次调用创建独立 event loop，无跨调用状态共享。
        """
        import concurrent.futures
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.aembed_documents(texts)).result()
        except RuntimeError:
            return asyncio.run(self.aembed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        """同步嵌入单个查询文本。"""
        return self.embed_documents([text])[0]
