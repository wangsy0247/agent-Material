"""多粒度检索器 — 子chunk 检索 → 父chunk 上下文。

两阶段检索：
1. 在 small collection 中做混合检索（向量 + BM25 + 重排序）
2. 收集唯一的 parent_id → 从 large collection 获取父chunk
3. 返回父chunk 作为最终上下文
"""

from __future__ import annotations

import threading
import time

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from agent_material.observability.metrics import (
    retrieval_duration,
    retrieval_docs,
    postgres_query_duration,
)
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
        self._compressor = None  # BM25 原子刷新时复用
        self._lock = threading.Lock()  # 原子替换检索器链

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

        # 阶段 2：收集唯一的 parent_id 及其所属 document/version（消歧用）
        parent_ids: list[str] = []
        parent_scopes: list[tuple[str, str]] = []
        seen_refs: set[tuple[str, str, str]] = set()
        for doc in child_docs:
            pid = doc.metadata.get("parent_id", "")
            scope = (
                doc.metadata.get("document_id", ""),
                doc.metadata.get("version_id", ""),
            )
            ref = (pid, scope[0], scope[1])
            if pid and ref not in seen_refs:
                parent_ids.append(pid)
                parent_scopes.append(scope)
                seen_refs.add(ref)

        if not parent_ids:
            # 无 parent_id → 直接返回子chunk
            logger.info("[MultiRetriever] 无 parent_id，返回子chunk")
            return child_docs[: vector_store_config.get("top_n", 5)]

        # 阶段 3：从大 collection 获取父chunk（有序 + missing 回退，ONL-009）
        parent_docs, missing_ids = self._fetch_parents(parent_ids, parent_scopes)

        if missing_ids:
            logger.warning(
                f"[MultiRetriever] {len(missing_ids)} 个父块未命中，降级返回对应子块"
            )
            parent_by_ref = {
                (
                    d.metadata.get("parent_id", ""),
                    d.metadata.get("document_id", ""),
                    d.metadata.get("version_id", ""),
                ): d
                for d in parent_docs
            }
            child_by_ref: dict[tuple[str, str, str], Document] = {}
            for doc in child_docs:
                ref = (
                    doc.metadata.get("parent_id", ""),
                    doc.metadata.get("document_id", ""),
                    doc.metadata.get("version_id", ""),
                )
                child_by_ref.setdefault(ref, doc)

            final_docs: list[Document] = []
            for pid, scope in zip(parent_ids, parent_scopes):
                ref = (pid, scope[0], scope[1])
                if ref in parent_by_ref:
                    final_docs.append(parent_by_ref[ref])
                else:
                    child = child_by_ref.get(ref)
                    if child is not None:
                        final_docs.append(
                            Document(
                                page_content=child.page_content,
                                metadata={**child.metadata, "parent_fallback": True},
                            )
                        )
            parent_docs = final_docs

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
                    self._compressor = compressor
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
        self._compressor = compressor
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=truncated,
        )

    def refresh_documents(self, docs: list[Document]) -> bool:
        """BM25 原子刷新（OFF-007）：先在锁外构建整条新链，成功后锁内一次性替换。

        任一步构建失败记 error 并保留旧实例；BM25 不可用或文档为空时
        退化为 vector-only（hybrid=vector），同样原子替换。

        Returns:
            True=已替换/暂存新语料；False=构建失败保留旧实例
            （调用方不推进 bm25_revision，配合 RRF 前 active 二次校验兜底，
            旧语料中的退役内容不会进入最终上下文，Spec 9.3）。
        """
        if not self._retrievers_initialized:
            with self._lock:
                self.all_documents = docs
            return True
        try:
            k = vector_store_config.get("k", 10)
            # 锁外构建新实例
            if BM25_AVAILABLE and docs:
                new_bm25 = BM25Retriever.from_documents(docs, k=k)
            else:
                new_bm25 = None

            if new_bm25 and EnsembleRetriever:
                new_hybrid = EnsembleRetriever(
                    retrievers=[self._vector_retriever, new_bm25],
                    weights=[
                        vector_store_config.get("vector_weight", 0.5),
                        vector_store_config.get("bm25_weight", 0.5),
                    ],
                )
            else:
                # 降级：BM25 不可用/文档为空 → vector-only
                new_hybrid = self._vector_retriever

            max_rerank = vector_store_config.get("max_rerank_docs", 10)
            from agent_material.rag_engine.vector_store import TopKRetriever

            truncated = TopKRetriever(base_retriever=new_hybrid, max_docs=max_rerank)

            new_compression = None
            if self._compressor is not None:
                new_compression = ContextualCompressionRetriever(
                    base_compressor=self._compressor,
                    base_retriever=truncated,
                )
        except Exception as e:
            logger.error(f"[MultiRetriever] BM25 刷新构建失败，保留旧实例: {e}")
            return False

        # 全部构建成功 → 锁内一次性替换
        with self._lock:
            self.all_documents = docs
            self._bm25_retriever = new_bm25
            self._hybrid_retriever = new_hybrid
            self._compression_retriever = new_compression
        logger.info(
            f"[MultiRetriever] BM25 原子刷新完成 | 文档数={len(docs)} | "
            f"{'BM25+向量混合' if new_bm25 else 'vector-only 降级'}"
        )
        return True

    def filter_active_bm25(self, docs: list[Document]) -> list[Document]:
        """RRF 前 active 二次校验（Spec V3 BM25-005）。

        缺 chunk_id/document_id/version_id 的候选直接丢弃（INV-006）；
        其余候选用三元组精确校验，避免 legacy chunk_id 跨文档或版本碰撞时
        把已经退役的候选误判为 active。

        SQL 异常向上抛出，由调用方把当前请求 BM25 路整体降级为 vector-only。
        """
        required = ("chunk_id", "document_id", "version_id")
        candidates = [d for d in docs if all(d.metadata.get(k) for k in required)]
        dropped_no_id = len(docs) - len(candidates)
        if dropped_no_id:
            logger.warning(
                f"[MultiRetriever] BM25 候选缺版本身份字段，直接丢弃 {dropped_no_id} 条"
            )
        if not candidates:
            return []

        chunk_ids = [d.metadata["chunk_id"] for d in candidates]

        import psycopg

        conn_str = vector_store_config.get("connection_string", "")
        psycopg_conn_str = conn_str.replace(
            "postgresql+psycopg://", "postgresql://"
        )
        mg_config = vector_store_config.get("multi_granularity", {})
        base_name = vector_store_config.get("character_name", "agent")
        small_name = base_name + mg_config.get("small_collection_suffix", "_small")

        conn = psycopg.connect(psycopg_conn_str)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT DISTINCT e.cmetadata->>'chunk_id',
                                e.cmetadata->>'document_id',
                                e.cmetadata->>'version_id'
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s
                  AND e.cmetadata->>'chunk_id' = ANY(%s)
                  AND e.cmetadata->>'is_active' = 'true'
                """,
                (small_name, chunk_ids),
            )
            active_keys = {(row[0], row[1], row[2]) for row in cur.fetchall()}
            cur.close()
        finally:
            conn.close()

        kept = [
            d for d in candidates
            if (
                d.metadata["chunk_id"],
                d.metadata["document_id"],
                d.metadata["version_id"],
            ) in active_keys
        ]
        dropped = len(candidates) - len(kept)
        if dropped:
            logger.warning(
                f"[MultiRetriever] BM25 active 二次校验剔除 {dropped} 条非 active 候选"
            )
        return kept

    def _hybrid_search(self, query: str) -> list[Document]:
        """在小 collection 上执行混合检索。"""
        if not self._retrievers_initialized:
            self._init_retrievers()

        retriever = self._compression_retriever or self._hybrid_retriever
        if retriever is None:
            return []
        return retriever.invoke(query)

    def search_children(
        self,
        query: str,
        k: int | None = None,
        hyde_embedding: list[float] | None = None,
    ) -> tuple[list[Document], list[Document]]:
        """请求级子chunk召回（ONL-005/ONL-008）：返回 (向量路结果, BM25 路结果)。

        不做 rerank、不做父块回溯，供 RetrievalPipeline 逐分支调用。
        HyDE embedding 仅作为参数传入向量路，不写共享属性；
        BM25 路始终使用原始 query 文本，查询失败时降级为空列表。

        Args:
            query: 查询文本（BM25 路与无 HyDE 时的向量路使用）
            k: 各路召回数量上限，None 时用实例默认
            hyde_embedding: 请求级 HyDE 假设文档 embedding（仅向量路使用）
        """
        if not self._retrievers_initialized:
            self._init_retrievers()

        k = k or self._k

        # ── 向量路 ──
        if hyde_embedding is not None:
            # HyDE：直接用预计算 embedding 检索（active 过滤与阈值逻辑
            # 与 VectorSimilarityThresholdRetriever 保持一致）
            search_filter = None
            if vector_store_config.get("feature_flags", {}).get(
                "active_version_filter", True
            ):
                search_filter = {"is_active": True}
            docs_with_scores = self._small_store.similarity_search_with_score_by_vector(
                hyde_embedding, k=k, filter=search_filter
            )
            vector_docs: list[Document] = []
            for doc, distance in docs_with_scores:
                similarity = 1.0 - distance / 2.0  # cosine distance → similarity
                if similarity >= self._score_threshold:
                    vector_docs.append(doc)
            if not vector_docs and docs_with_scores:
                vector_docs.append(docs_with_scores[0][0])  # Top-1 兜底
        else:
            vector_docs = self._vector_retriever.search(query)

        # ── BM25 路（失败降级为空列表，由调用方记 degradation）──
        bm25_docs: list[Document] = []
        if self._bm25_retriever is not None:
            try:
                bm25_docs = self._bm25_retriever.invoke(query)
            except Exception as e:
                logger.error(f"[MultiRetriever] BM25 检索失败（降级 vector-only）: {e}")

        return vector_docs, bm25_docs

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

    def _fetch_parents(
        self,
        parent_ids: list[str],
        parent_scopes: (
            list[tuple[str, str]] | dict[str, tuple[str, str]] | None
        ) = None,
    ) -> tuple[list[Document], list[str]]:
        """从大 collection 批量获取父chunk（ONL-009：有序回溯 + active 过滤）。

        Args:
            parent_ids: 子块引用到的 parent_id（有序、去重）。
            parent_scopes: 与 parent_ids 对齐的 (document_id, version_id) 列表；
                兼容旧调用方传入 parent_id → scope 的 dict。
                存量 legacy 数据的 parent_id 只做内容哈希，可能跨文档撞库；
                提供 scope 时优先匹配同 document/version 的父块，命中不到则
                视为 missing（IDX-002：不允许引用其他 document/version 的
                parent）；子块缺少 scope 信息时退化为首条匹配（防御性兜底）。

        Returns:
            (ordered_docs, missing_ids)：
            - ordered_docs 按传入 parent_ids 的顺序重排（同一 parent 只返回一次）
            - missing_ids 为未命中的 parent_id（调用方回退子块）
            SQL 异常时返回 ([], parent_ids)，由调用方整体回退子块。
        """
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

        active_filter = vector_store_config.get("feature_flags", {}).get(
            "active_version_filter", True
        )
        # 严格 active-only（INV-006）：is_active 缺失不再视为 active
        active_clause = (
            " AND e.cmetadata->>'is_active' = 'true'" if active_filter else ""
        )

        try:
            conn = psycopg.connect(psycopg_conn_str)
            cur = conn.cursor()
            # 用 ANY 批量查询
            cur.execute(
                f"""
                SELECT e.document, e.cmetadata
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s AND e.cmetadata->>'parent_id' = ANY(%s)
                {active_clause}
                """,
                (large_name, parent_ids),
            )
            rows = cur.fetchall()
            cur.close()
            conn.close()

            # 按 (parent_id, document_id, version_id) 建立精确索引，同时保留
            # parent_id 首条匹配作兜底；再按输入顺序重排
            by_key: dict[tuple[str, str, str], Document] = {}
            by_id: dict[str, Document] = {}
            for content, meta in rows:
                meta = meta or {}
                pid = meta.get("parent_id", "")
                if not pid:
                    continue
                doc = Document(page_content=content, metadata=meta)
                by_key.setdefault(
                    (pid, meta.get("document_id", ""), meta.get("version_id", "")),
                    doc,
                )
                by_id.setdefault(pid, doc)

            ordered_docs: list[Document] = []
            missing_ids: list[str] = []
            for idx, pid in enumerate(parent_ids):
                if isinstance(parent_scopes, dict):
                    scope = parent_scopes.get(pid)
                elif parent_scopes and idx < len(parent_scopes):
                    scope = parent_scopes[idx]
                else:
                    scope = None
                if scope and scope[0]:
                    # 子块带来源信息：只接受同 document/version 的父块，
                    # 其他文档的同 parent_id 父块视为未命中（IDX-002）
                    doc = by_key.get((pid, scope[0], scope[1]))
                else:
                    doc = by_id.get(pid)
                if doc is not None:
                    ordered_docs.append(doc)
                else:
                    missing_ids.append(pid)

            elapsed = time.time() - t0
            retrieval_duration.labels(stage="parent_fetch").observe(elapsed)
            retrieval_docs.labels(stage="parent_fetch").observe(len(ordered_docs))
            postgres_query_duration.labels(operation="parent_fetch").observe(elapsed)
            logger.info(
                f"[MultiRetriever] 获取父chunk: {len(parent_ids)} ids → "
                f"{len(ordered_docs)} docs | missing={len(missing_ids)} | {elapsed:.3f}s"
            )
            return ordered_docs, missing_ids
        except Exception as e:
            # 不再静默返回 []——由调用方回退返回子块（6.3 降级表）
            logger.error(f"[MultiRetriever] 获取父chunk失败（回退子块）: {e}")
            return [], list(parent_ids)
