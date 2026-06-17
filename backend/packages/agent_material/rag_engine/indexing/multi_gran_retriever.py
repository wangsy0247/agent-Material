"""多粒度检索器 — 子chunk 检索 → 父chunk 上下文。

两阶段检索：
1. 在 small collection 中做混合检索（向量 + BM25 + 重排序）
2. 收集唯一的 parent_id → 从 large collection 获取父chunk
3. 返回父chunk 作为最终上下文
"""

from __future__ import annotations

import time
from typing import Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from agent_material.observability.metrics import (
    retrieval_duration,
    retrieval_docs,
    postgres_query_duration,
)
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import ConfigDict

from agent_material.utils.config_handler import vector_store_config
from agent_material.utils.logger import logger

# 可选依赖
try:
    from langchain_community.retrievers import BM25Retriever

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from langchain.retrievers import EnsembleRetriever
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
except ImportError:
    try:
        from langchain_classic.retrievers import EnsembleRetriever
        from langchain_classic.retrievers.contextual_compression import (
            ContextualCompressionRetriever,
        )
    except ImportError:
        EnsembleRetriever = None
        ContextualCompressionRetriever = None

try:
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False


class MultiGranularityRetriever(BaseRetriever):
    """多粒度检索器 — 子chunk检索 + 父chunk上下文。

    Args:
        small_store: 小粒度 PGVector collection
        large_store: 大粒度 PGVector collection
        all_documents: 所有小chunk文档列表（用于 BM25 索引）
        k: 检索返回数量上限
        score_threshold: 向量相似度阈值
    """

    model_config = ConfigDict(extra="allow")

    def __init__(
        self,
        small_store,
        large_store,
        all_documents: list[Document] | None = None,
        k: int = 10,
        score_threshold: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._small_store = small_store
        self._large_store = large_store
        self.all_documents = all_documents or []
        self._k = k
        self._score_threshold = score_threshold

        # 缓存检索器链
        self._retrievers_initialized = False
        self._hybrid_retriever = None
        self._compression_retriever = None
        self._bm25_retriever = None
        self._vector_retriever = None  # HyDE 模式需要访问内部的向量检索器

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """执行多粒度检索。"""
        start = time.time()

        # 阶段 1：在小chunk 上混合检索
        child_docs = self._hybrid_search(query)

        if not child_docs:
            logger.info("[MultiRetriever] 未检索到相关文档")
            return []

        # 阶段 2：收集唯一的 parent_id
        parent_ids: list[str] = []
        seen_ids: set[str] = set()
        for doc in child_docs:
            pid = doc.metadata.get("parent_id", "")
            if pid and pid not in seen_ids:
                parent_ids.append(pid)
                seen_ids.add(pid)

        if not parent_ids:
            # 无 parent_id → 直接返回子chunk
            logger.info("[MultiRetriever] 无 parent_id，返回子chunk")
            return child_docs[: vector_store_config.get("top_n", 5)]

        # 阶段 3：从大 collection 获取父chunk
        parent_docs = self._fetch_parents(parent_ids)

        elapsed = time.time() - start
        logger.info(
            f"[MultiRetriever] 检索完成 | "
            f"子chunk={len(child_docs)} → 父chunk={len(parent_docs)} | "
            f"耗时={elapsed:.3f}s"
        )
        return parent_docs

    # ── 混合检索（小chunk）────────────────────────────────

    def _init_retrievers(self):
        """初始化检索器链。"""
        from agent_material.rag_engine.vector_store import (
            VectorSimilarityThresholdRetriever,
            TopKRetriever,
        )

        k = vector_store_config.get("k", 10)
        score_threshold = vector_store_config.get("score_threshold", 0.3)

        # 向量检索器
        vector_retriever = VectorSimilarityThresholdRetriever(
            vector_store=self._small_store,
            k=k,
            score_threshold=score_threshold,
        )
        self._vector_retriever = vector_retriever

        # BM25
        if BM25_AVAILABLE and self.all_documents:
            self._bm25_retriever = BM25Retriever.from_documents(
                self.all_documents, k=k
            )
        else:
            self._bm25_retriever = None

        # 混合检索
        if self._bm25_retriever and EnsembleRetriever:
            self._hybrid_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, self._bm25_retriever],
                weights=[
                    vector_store_config.get("vector_weight", 0.5),
                    vector_store_config.get("bm25_weight", 0.5),
                ],
            )
        else:
            self._hybrid_retriever = vector_retriever

        # 截断 + 重排序
        max_rerank = vector_store_config.get("max_rerank_docs", 10)
        truncated = TopKRetriever(
            base_retriever=self._hybrid_retriever,
            max_docs=max_rerank,
        )

        enable_reranker = vector_store_config.get("enable_reranker", True)
        if enable_reranker:
            try:
                reranker_cfg = vector_store_config.get("reranker", {})
                provider = reranker_cfg.get("provider", "dashscope")

                if provider == "dashscope":
                    from agent_material.rag_engine.dashscope_reranker import DashScopeReranker

                    compressor = DashScopeReranker(
                        model=reranker_cfg.get("model", "qwen3-rerank"),
                        top_n=reranker_cfg.get("top_n", vector_store_config.get("top_n", 5)),
                    )
                    self._compression_retriever = ContextualCompressionRetriever(
                        base_compressor=compressor,
                        base_retriever=truncated,
                    )
                    logger.info(
                        f"[MultiRetriever] DashScope 重排序器初始化成功 | "
                        f"model={reranker_cfg.get('model', 'qwen3-rerank')}"
                    )
                elif provider == "local":
                    self._compression_retriever = self._init_local_reranker(truncated)
                else:
                    logger.warning(
                        f"[MultiRetriever] 未知的重排序 provider: {provider}，跳过"
                    )
                    self._compression_retriever = None
            except Exception as e:
                logger.error(f"[MultiRetriever] 重排序器初始化失败: {e}")
                self._compression_retriever = None
        else:
            self._compression_retriever = None

        self._retrievers_initialized = True

    def _init_local_reranker(self, truncated):
        """本地 CrossEncoder 重排序器（local provider 降级路径）。"""
        if not RERANKER_AVAILABLE or not CrossEncoderReranker:
            logger.warning("[MultiRetriever] 重排序依赖未安装，跳过")
            return None

        model_path = vector_store_config.get(
            "reranker", {}
        ).get("model_path", "BAAI/bge-reranker-base")

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        cross_encoder = HuggingFaceCrossEncoder(
            model_name=model_path,
            model_kwargs={"device": device},
        )
        if hasattr(cross_encoder, "score") and not hasattr(cross_encoder, "predict"):
            object.__setattr__(cross_encoder, "predict", cross_encoder.score)

        compressor = CrossEncoderReranker(
            model=cross_encoder,
            top_n=vector_store_config.get("top_n", 5),
        )
        logger.info("[MultiRetriever] 本地 CrossEncoder 重排序器初始化成功")
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=truncated,
        )

    def _hybrid_search(self, query: str) -> list[Document]:
        """在小 collection 上执行混合检索。"""
        if not self._retrievers_initialized:
            self._init_retrievers()

        retriever = self._compression_retriever or self._hybrid_retriever
        if retriever is None:
            return []
        return retriever.invoke(query)

    def set_hyde_embedding(self, embedding: list[float] | None) -> None:
        """设置/清除 HyDE embedding 覆盖到内部的向量检索器。

        Args:
            embedding: 假设文档的 embedding 向量，None 清除覆盖
        """
        if not self._retrievers_initialized:
            self._init_retrievers()
        if self._vector_retriever is not None:
            self._vector_retriever.embedding_override = embedding
            logger.info(
                f"[MultiRetriever] HyDE embedding "
                f"{'设置' if embedding else '清除'}"
            )

    # ── 父chunk 获取 ──────────────────────────────────────

    def _fetch_parents(self, parent_ids: list[str]) -> list[Document]:
        """从大 collection 批量获取父chunk。"""
        import psycopg
        from langchain_core.documents import Document

        t0 = time.time()
        conn_str = vector_store_config.get("connection_string", "")
        psycopg_conn_str = conn_str.replace(
            "postgresql+psycopg://", "postgresql://"
        )

        # 获取 large collection name
        mg_config = vector_store_config.get("multi_granularity", {})
        base_name = vector_store_config.get("character_name", "agent")
        large_name = base_name + mg_config.get("large_collection_suffix", "_large")

        try:
            conn = psycopg.connect(psycopg_conn_str)
            cur = conn.cursor()
            # 用 ANY 批量查询
            cur.execute(
                """
                SELECT e.document, e.cmetadata
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s AND e.cmetadata->>'parent_id' = ANY(%s)
                """,
                (large_name, parent_ids),
            )
            rows = cur.fetchall()
            cur.close()
            conn.close()

            docs = [
                Document(page_content=row[0], metadata=row[1] or {})
                for row in rows
            ]

            elapsed = time.time() - t0
            retrieval_duration.labels(stage="parent_fetch").observe(elapsed)
            retrieval_docs.labels(stage="parent_fetch").observe(len(docs))
            postgres_query_duration.labels(operation="parent_fetch").observe(elapsed)
            logger.info(
                f"[MultiRetriever] 获取父chunk: {len(parent_ids)} ids → "
                f"{len(docs)} docs | {elapsed:.3f}s"
            )
            return docs
        except Exception as e:
            logger.error(f"[MultiRetriever] 获取父chunk失败: {e}")
            return []
