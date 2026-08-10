import os
import threading
import time

# 兼容 LangChain 0.3.x 和 1.x
try:
    from langchain.retrievers import EnsembleRetriever
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_classic.retrievers.contextual_compression import (
        ContextualCompressionRetriever,
    )
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import ConfigDict

from agent_material.observability.metrics import (
    retrieval_duration,
    retrieval_docs,
    postgres_query_duration,
)

from agent_material.utils.config_handler import vector_store_config
from agent_material.rag_engine.models import embedding_model
from agent_material.utils.logger import logger

from langchain_postgres.vectorstores import PGVector


class VectorSimilarityThresholdRetriever(BaseRetriever):
    """自定义 PGVector 检索器，正确处理 cosine distance 与 similarity 的转换。

    PGVector 默认使用 cosine distance，范围 [0, 2]：
        - distance = 0  → 完全相同（similarity = 1.0）
        - distance = 2  → 完全相反（similarity = 0.0）

    LangChain 的 `similarity_score_threshold` 在不同版本中对 score 的处理
    存在不一致（有时把 distance 直接当 similarity 比较），导致召回率异常。
    本检索器显式做转换：similarity = 1 - distance / 2，然后与阈值比较。

    支持 HyDE 模式：设置 embedding_override 后，跳过内部 embed_query()，
    直接使用预计算的 embedding 做向量检索。

    Args:
        vector_store: PGVector 向量库实例
        k: 召回数量上限
        score_threshold: similarity 阈值（0~1，越大越严格），默认 0.3
    """

    # Pydantic v2 兼容：允许运行时动态添加属性（如 self.vector_store）
    model_config = ConfigDict(extra="allow")

    def __init__(
        self,
        vector_store: PGVector,
        k: int = 10,
        score_threshold: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vector_store = vector_store
        self.k = k
        self.score_threshold = score_threshold
        # HyDE 模式：非 None 时使用此 embedding 替代 query embedding
        self.embedding_override: list[float] | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        # 委托给 search()；embedding_override 仅为旧路径回滚兼容保留
        return self.search(query, embedding=self.embedding_override)

    def search(
        self, query: str, embedding: list[float] | None = None
    ) -> list[Document]:
        """请求级向量检索（ONL-008）：embedding 作为参数传入，不写共享属性。

        Args:
            query: 查询文本（embedding 为 None 时用于内部 embed）
            embedding: 预计算 embedding（如 HyDE 假设文档向量），None 时按 query 检索
        """
        t0 = time.time()

        # active 版本过滤：staging/failed/retired 版本不参与在线检索
        search_filter = None
        if vector_store_config.get("feature_flags", {}).get("active_version_filter", True):
            search_filter = {"is_active": True}

        # HyDE 模式：使用预计算 embedding 做向量检索
        if embedding is not None:
            docs_with_scores = self.vector_store.similarity_search_with_score_by_vector(
                embedding, k=self.k, filter=search_filter
            )
        else:
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, k=self.k, filter=search_filter
            )

        results: list[Document] = []
        for doc, distance in docs_with_scores:
            # cosine distance → cosine similarity
            similarity = 1.0 - distance / 2.0
            if similarity >= self.score_threshold:
                results.append(doc)

        if not results and docs_with_scores:
            # 如果过滤后为空，至少保留 Top-1，避免完全无结果
            logger.warning(
                f"[PGVectorRetriever] 查询 '{query[:30]}...' 经阈值过滤后无结果，"
                f"自动回退到 Top-1（distance={docs_with_scores[0][1]:.3f}）"
            )
            results.append(docs_with_scores[0][0])

        elapsed = time.time() - t0
        retrieval_duration.labels(stage="vector_search").observe(elapsed)
        retrieval_docs.labels(stage="vector_search").observe(len(results))
        postgres_query_duration.labels(operation="vector_search").observe(elapsed)
        logger.debug(f"[PGVectorRetriever] 向量检索完成 | 返回 {len(results)} 条 | 耗时 {elapsed:.3f}s")
        return results


class TopKRetriever(BaseRetriever):
    """文档数量截断检索器包装器。

    用于在混合检索器和重排序器之间截断文档数量，
    避免过多低质量文档进入耗时的 CrossEncoder 重排序阶段。
    """
    model_config = ConfigDict(extra="allow")

    def __init__(self, base_retriever: BaseRetriever, max_docs: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.base_retriever = base_retriever
        self.max_docs = max_docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        docs = self.base_retriever.invoke(query)
        if len(docs) > self.max_docs:
            logger.info(
                f"[TopKRetriever] 截断文档数: {len(docs)} → {self.max_docs} "
                f"(减少重排序负担)"
            )
            return docs[:self.max_docs]
        return docs

# 可选依赖：BM25
try:
    from langchain_community.retrievers import BM25Retriever

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("BM25 依赖未安装（pip install rank-bm25），将不使用关键词检索")

# 可选依赖：重排序
try:
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    RERANKER_AVAILABLE = True
except ImportError:
    try:
        from langchain_classic.retrievers.document_compressors import (
            CrossEncoderReranker,
        )
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        RERANKER_AVAILABLE = True
    except ImportError:
        RERANKER_AVAILABLE = False
        logger.warning(
            "重排序依赖未安装（pip install sentence-transformers），将不使用重排序功能"
        )


class VectorStore:
    def __init__(self):
        # 加载并保存切分配置
        self._chunk_size = vector_store_config["chunk_size"]
        self._chunk_overlap = vector_store_config["chunk_overlap"]
        self._separator = vector_store_config["separator"]

        # 文档处理管道（两阶段：Loader + Splitter）
        self._pipeline = None  # 延迟初始化
        logger.info("文档处理管道已就绪（两阶段：Loader + Splitter）")

        # PGVector 延迟初始化（避免 __init__ 时触发 C 扩展 segfault）
        self._vector_store: PGVector | None = None

        # 维护内存中的文档列表，用于构建 BM25 索引
        self.all_documents: list[Document] = []

        # 检索器缓存（延迟初始化，避免模块导入时加载重排序模型）
        self._bm25_retriever = None
        self._hybrid_retriever = None
        self._compression_retriever = None
        self._retrievers_initialized = False
        self._bm25_lock = threading.Lock()  # BM25 原子刷新替换锁

        # ── 多粒度索引 ──
        mg_config = vector_store_config.get("multi_granularity", {})
        self._multi_gran_enabled = mg_config.get("enabled", False)
        self._multi_index = None
        self._multi_retriever = None

        # ── 文件追踪 & 任务队列（懒加载）──
        self._file_tracker = None
        self._task_queue = None
        self._worker = None
        # ── PG 控制面仓储（懒加载，pg_control_plane 开启时使用）──
        self._pg_repos = None
        # ── BM25 revision 生命周期（Spec V2 9）──
        self._local_bm25_revision = 0  # 本地 BM25 已构建到的 active_revision
        self._last_bm25_revision_check = 0.0  # 上次 revision 检查的 time.monotonic()

    @staticmethod
    def _pg_control_plane_enabled() -> bool:
        """pg_control_plane 开关：true=PG 控制面，false=回滚 SQLite 控制面。"""
        return bool(
            vector_store_config.get("feature_flags", {}).get("pg_control_plane", False)
        )

    def _get_pg_repos(self) -> dict:
        """PG 控制面仓储集合（conn_str 来自 vector_store_config，表结构幂等确保）。"""
        if self._pg_repos is None:
            from agent_material.rag_engine.db.pg import (
                FileStateRepository,
                KbStateRepository,
                TaskRepository,
                VersionRepository,
                ensure_schema,
            )

            conn_str = vector_store_config.get("connection_string")
            ensure_schema(conn_str)
            self._pg_repos = {
                "tasks": TaskRepository(conn_str),
                "file_states": FileStateRepository(conn_str),
                "versions": VersionRepository(conn_str),
                "kb": KbStateRepository(conn_str),
            }
        return self._pg_repos

    @property
    def vector_store(self) -> PGVector:
        """延迟初始化 PGVector，隔离 C 扩展崩溃风险。"""
        if self._vector_store is None:
            logger.info("[VectorStore] 初始化 PGVector...")
            self._vector_store = PGVector(
                embeddings=embedding_model,
                connection=vector_store_config.get("connection_string"),
                collection_name=vector_store_config["character_name"],
                use_jsonb=True,
            )
            logger.info("[VectorStore] PGVector 初始化完成")
            # PGVector 就绪后再加载已有文档
            self._load_existing_documents()
        return self._vector_store

    def _load_existing_documents(self):
        """从 PGVector 加载已有文档到内存，用于构建 BM25 索引"""
        import psycopg
        conn_str = vector_store_config.get("connection_string")
        collection_name = vector_store_config.get("character_name", "agent")
        if not conn_str:
            logger.warning("未配置 connection_string，跳过加载已有文档")
            return
        # psycopg 原生连接不支持 SQLAlchemy 的 postgresql+psycopg:// 前缀
        psycopg_conn_str = conn_str.replace("postgresql+psycopg://", "postgresql://")
        try:
            conn = psycopg.connect(psycopg_conn_str)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT e.document, e.cmetadata
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s
                """,
                (collection_name,),
            )
            rows = cur.fetchall()
            for content, meta in rows:
                self.all_documents.append(
                    Document(page_content=content, metadata=meta or {})
                )
            logger.info(f"从 PGVector 加载了 {len(self.all_documents)} 条已有文档")
            cur.close()
            conn.close()
        except Exception as e:
            logger.warning(f"加载已有文档失败: {e}")

    def _init_retrievers(self):
        """初始化混合检索器（向量 + BM25）和重排序器"""
        k = vector_store_config.get("k", 10)
        score_threshold = vector_store_config.get("score_threshold", 0.3)

        # 1. 向量检索器（带相似度阈值过滤）
        # 使用自定义检索器，显式处理 PGVector cosine distance → similarity 转换，
        # 避免 LangChain 内置 similarity_score_threshold 的行为不一致问题
        vector_retriever = VectorSimilarityThresholdRetriever(
            vector_store=self.vector_store,
            k=k,
            score_threshold=score_threshold,
        )

        # 2. BM25 关键词检索器
        if BM25_AVAILABLE and self.all_documents:
            self._bm25_retriever = BM25Retriever.from_documents(self.all_documents, k=k)
            logger.info(f"BM25 检索器初始化成功，文档数：{len(self.all_documents)}")
        else:
            self._bm25_retriever = None

        # 3. 混合检索器：RRF 融合向量 + BM25 结果
        if self._bm25_retriever:
            self._hybrid_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, self._bm25_retriever],
                weights=[
                    vector_store_config.get("vector_weight", 0.7),
                    vector_store_config.get("bm25_weight", 0.3),
                ],
            )
            logger.info("混合检索器初始化成功")
        else:
            self._hybrid_retriever = vector_retriever
            logger.info("仅使用向量检索器（BM25 未启用）")

        # 3.5 截断包装器：限制传给重排序器的最大文档数（避免 CPU 推理过载）
        max_rerank_docs = vector_store_config.get("max_rerank_docs", 8)
        truncated_retriever = TopKRetriever(
            base_retriever=self._hybrid_retriever, max_docs=max_rerank_docs
        )

        # 4. 重排序器：对截断后的结果精排
        enable_reranker = vector_store_config.get("enable_reranker", True)
        if enable_reranker:
            try:
                reranker_cfg = vector_store_config.get("reranker", {})
                provider = reranker_cfg.get("provider", "dashscope")

                if provider == "dashscope":
                    from agent_material.rag_engine.dashscope_reranker import DashScopeReranker

                    compressor = DashScopeReranker(
                        model=reranker_cfg.get("model", "qwen3-rerank"),
                        top_n=reranker_cfg.get("top_n", 5),
                    )
                    self._compression_retriever = ContextualCompressionRetriever(
                        base_compressor=compressor, base_retriever=truncated_retriever
                    )
                    logger.info(
                        f"DashScope 重排序器初始化成功 | model={reranker_cfg.get('model', 'qwen3-rerank')} | "
                        f"top_n={reranker_cfg.get('top_n', 5)}"
                    )
                elif provider == "local":
                    self._compression_retriever = self._init_local_reranker(truncated_retriever)
                else:
                    logger.warning(f"未知的重排序 provider: {provider}，跳过重排序")
                    self._compression_retriever = None
            except Exception as e:
                logger.error(f"重排序器初始化失败: {e}")
                self._compression_retriever = None
        else:
            logger.info("重排序器已禁用（enable_reranker=false），使用混合检索结果直接返回")
            self._compression_retriever = None

        # 5. 向量索引（HNSW / IVFFlat）：加速 PGVector 向量检索
        self._ensure_vector_index()

    def _ensure_vector_index(self) -> None:
        """确保 PGVector embedding 列上的向量索引已创建（幂等安全）。

        首次调用时创建索引，后续调用为 no-op（IF NOT EXISTS）。
        失败仅记录警告，不阻断检索链路。
        """
        index_config = vector_store_config.get("vector_index")
        if not index_config:
            return

        try:
            from agent_material.rag_engine.vector_index import (
                build_psycopg_conn_str,
                create_vector_index,
            )

            conn_str = vector_store_config.get("connection_string", "")
            if not conn_str:
                logger.warning("[VectorStore] 未配置 connection_string，跳过向量索引创建")
                return

            psycopg_conn_str = build_psycopg_conn_str(conn_str)
            create_vector_index(psycopg_conn_str, index_config)
        except Exception as e:
            logger.warning(f"[VectorStore] 向量索引创建异常，检索回退到顺序扫描: {e}")

    def _init_local_reranker(self, truncated_retriever):
        """本地 CrossEncoder 重排序器（local provider 降级路径）。

        需要安装 sentence-transformers 和 torch。
        """
        if not RERANKER_AVAILABLE:
            logger.warning("重排序依赖未安装（pip install sentence-transformers），跳过重排序")
            return None

        model_path = vector_store_config.get(
            "reranker", {}
        ).get("model_path", "BAAI/bge-reranker-base")

        is_local = os.path.exists(model_path) and os.path.isdir(model_path)
        if is_local:
            logger.info(f"检测到本地重排序模型: {model_path}")
        else:
            logger.info(f"将从 HuggingFace Hub 加载重排序模型: {model_path}")

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        logger.info(f"重排序模型使用设备: {device}")

        cross_encoder = HuggingFaceCrossEncoder(
            model_name=model_path,
            model_kwargs={"device": device},
        )
        if hasattr(cross_encoder, "score") and not hasattr(cross_encoder, "predict"):
            object.__setattr__(cross_encoder, "predict", cross_encoder.score)
        compressor = CrossEncoderReranker(
            model=cross_encoder, top_n=vector_store_config.get("top_n", 3)
        )
        logger.info("本地 CrossEncoder 重排序器初始化成功")
        return ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=truncated_retriever
        )

    def get_retriever(self):
        """返回检索器（首次调用时初始化，避免模块导入时触发 segfault）。

        当多粒度索引启用时，返回 MultiGranularityRetriever。
        每次调用先做节流版 BM25 revision 检测（Spec 9.2，pg 模式多进程兜底）。
        """
        self.maybe_refresh_bm25()

        if self._multi_gran_enabled:
            if self._multi_retriever is None:
                self._init_multi_retriever()
            return self._multi_retriever

        if not self._retrievers_initialized:
            self._init_retrievers()
            self._retrievers_initialized = True
        return self._compression_retriever or self._hybrid_retriever

    def set_hyde_embedding(self, embedding: list[float] | None) -> None:
        """设置/清除 HyDE embedding 覆盖，使向量检索使用假设文档的 embedding。

        在多粒度模式下传播到 MultiGranularityRetriever 内部的向量检索器；
        标准模式下传播到 EnsembleRetriever 内部的 VectorSimilarityThresholdRetriever。

        Args:
            embedding: 假设文档的 embedding 向量，传入 None 清除覆盖恢复默认行为
        """
        if self._multi_gran_enabled and self._multi_retriever is not None:
            self._multi_retriever.set_hyde_embedding(embedding)
        elif self._hybrid_retriever is not None:
            # 标准模式：找到 EnsembleRetriever 中的 VectorSimilarityThresholdRetriever
            if hasattr(self._hybrid_retriever, "retrievers"):
                for r in self._hybrid_retriever.retrievers:
                    if isinstance(r, VectorSimilarityThresholdRetriever):
                        r.embedding_override = embedding
                        logger.info(
                            f"[VectorStore] HyDE embedding "
                            f"{'设置' if embedding else '清除'}（标准模式）"
                        )
                        return
            logger.warning("[VectorStore] 未找到向量检索器，无法设置 HyDE embedding")

    def _init_multi_retriever(self):
        """初始化多粒度检索器。

        冷启动（Spec V2 9.1）：无条件只加载 active 子块构建 BM25
        （存量数据已 backfill is_active；staging 数据本就不该进 BM25），
        并记录当前 active_revision 为本地 bm25_revision。
        """
        from agent_material.rag_engine.indexing.multi_gran_retriever import MultiGranularityRetriever

        mi = self._get_multi_index()
        mi.small_store  # 触发 PGVector 初始化
        mi.large_store  # 触发 PGVector 初始化

        # 加载小chunk文档用于 BM25（仅 active，与 active_version_filter 开关无关）
        small_docs = self._load_documents_from_collection(mi.small_name, active_only=True)

        self._multi_retriever = MultiGranularityRetriever(
            small_store=mi.small_store,
            large_store=mi.large_store,
            all_documents=small_docs,
        )
        self.all_documents = small_docs  # 保持兼容
        self._init_bm25_revision()  # 冷启动记录本地 bm25_revision
        logger.info("[VectorStore] 多粒度检索器初始化完成")

    def _get_multi_index(self):
        if self._multi_index is None:
            from agent_material.rag_engine.indexing.multi_granularity import MultiGranularityIndex
            self._multi_index = MultiGranularityIndex(enabled=self._multi_gran_enabled)
        return self._multi_index

    def _load_documents_from_collection(
        self, collection_name: str, active_only: bool = False
    ) -> list[Document]:
        """从指定 PGVector collection 加载所有文档到内存。

        active_only=True 时只加载 active 版本（is_active 缺失视为 active，
        兼容存量数据）。
        """
        import psycopg
        conn_str = vector_store_config.get("connection_string")
        if not conn_str:
            return []
        psycopg_conn_str = conn_str.replace("postgresql+psycopg://", "postgresql://")
        docs = []
        try:
            conn = psycopg.connect(psycopg_conn_str)
            cur = conn.cursor()
            sql = """SELECT e.document, e.cmetadata
                     FROM langchain_pg_embedding e
                     JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                     WHERE c.name = %s"""
            if active_only:
                sql += (
                    " AND (e.cmetadata->>'is_active' = 'true'"
                    " OR e.cmetadata->>'is_active' IS NULL)"
                )
            cur.execute(sql, (collection_name,))
            for content, meta in cur.fetchall():
                docs.append(Document(page_content=content, metadata=meta or {}))
            cur.close()
            conn.close()
            logger.info(f"从 {collection_name} 加载了 {len(docs)} 条文档")
        except Exception as e:
            logger.warning(f"加载 {collection_name} 文档失败: {e}")
        return docs

    # ------------------------------------------------------------------ #
    #  BM25 revision 生命周期（Spec V2 9）
    # ------------------------------------------------------------------ #
    def _kb_id(self) -> str:
        """知识库 ID（与 worker/扫描入队侧一致：character_name）。"""
        return vector_store_config.get("character_name", "agent")

    def _init_bm25_revision(self) -> None:
        """冷启动（Spec 9.1-3）：把当前 active_revision 记为本地 bm25_revision。"""
        if not self._pg_control_plane_enabled():
            return
        try:
            kb_repo = self._get_pg_repos()["kb"]
            kb_id = self._kb_id()
            state = kb_repo.get(kb_id)
            self._local_bm25_revision = state["active_revision"]
            kb_repo.update_bm25_revision(kb_id, self._local_bm25_revision)
            logger.info(
                f"[VectorStore] 冷启动 BM25 revision 记录 | kb={kb_id} | "
                f"revision={self._local_bm25_revision}"
            )
        except Exception as e:
            logger.warning(f"[VectorStore] 冷启动记录 bm25_revision 失败（后续检测兜底）: {e}")

    def maybe_refresh_bm25(self) -> None:
        """BM25 revision 检测刷新（Spec 9.2）：多进程兜底的热刷新路径。

        每次检索请求路径调用（get_retriever 顶部），节流
        `bm25_revision_check_interval` 秒（默认 2s）查一次 PG；
        active_revision 领先本地 bm25_revision 时重建 BM25。
        进程内 on_activated 回调仍是快路径，本机制覆盖多查询进程场景。
        所有异常只记日志，绝不影响检索链路。
        """
        if not self._pg_control_plane_enabled():
            return
        interval = float(
            vector_store_config.get("bm25_revision_check_interval", 2.0)
        )
        now = time.monotonic()
        if now - self._last_bm25_revision_check < interval:
            return
        self._last_bm25_revision_check = now
        try:
            kb_repo = self._get_pg_repos()["kb"]
            kb_id = self._kb_id()
            # 单次 get 同时完成 needs_bm25_refresh 判断与目标 revision 读取
            state = kb_repo.get(kb_id)
            if state["active_revision"] <= self._local_bm25_revision:
                return
            logger.info(
                f"[VectorStore] 检测到 active_revision 领先 "
                f"(remote={state['active_revision']} > local={self._local_bm25_revision})，"
                f"重建 BM25"
            )
            if not self.refresh_bm25():
                # 刷新失败：不推进本地 revision，下个节流窗口自动重试（Spec 9.3）
                logger.warning(
                    "[VectorStore] BM25 revision 刷新失败（degradation: 保留旧实例，"
                    "旧语料由 RRF 前 active 二次校验兜底）"
                )
        except Exception as e:
            logger.warning(f"[VectorStore] BM25 revision 检查失败（下次检索重试）: {e}")

    def _capture_active_revision(self):
        """刷新前读取 (kb_repo, kb_id, target_revision)；pg 关闭或失败返回 None。"""
        if not self._pg_control_plane_enabled():
            return None
        try:
            kb_repo = self._get_pg_repos()["kb"]
            kb_id = self._kb_id()
            target = kb_repo.get(kb_id)["active_revision"]
            return kb_repo, kb_id, target
        except Exception as e:
            logger.warning(f"[VectorStore] 读取 active_revision 失败: {e}")
            return None

    def _commit_bm25_revision(self, captured) -> None:
        """BM25 替换成功后推进本地/远端 bm25_revision 到刷新前捕获的值。

        用刷新前的 revision（而非刷新后重读）避免把构建窗口内的新激活
        误标为已构建——新激活会被下次检测再次捕获。
        """
        if captured is None:
            return
        kb_repo, kb_id, target = captured
        self._local_bm25_revision = target
        try:
            kb_repo.update_bm25_revision(kb_id, target)
        except Exception as e:
            logger.warning(f"[VectorStore] 回写 bm25_revision 失败: {e}")

    def refresh_bm25(self) -> bool:
        """BM25 原子刷新（OFF-007）：重载 active 文档，先构建新实例再一次性替换。

        - 多粒度模式：重载 small collection（始终只加载 active 子块，Spec 9.1），
          检索器已初始化则调 refresh_documents 原子替换，否则只更新 all_documents。
        - 非多粒度模式：重载基础 collection，锁内替换 BM25+Ensemble+压缩链。
        - 构建失败保留旧实例，不影响向量检索（6.3 降级表）。

        Returns:
            True=语料/检索器已更新到最新（可推进 bm25_revision）；
            False=构建失败保留旧实例（revision 不推进，等待重试）。
        """
        # 多粒度模式始终只加载 active（Spec 9.1：staging/retired 不进 BM25）；
        # 非多粒度旧链路保持 active_version_filter 开关语义（回滚兼容）。
        flags = vector_store_config.get("feature_flags", {})
        active_only = True if self._multi_gran_enabled else flags.get(
            "active_version_filter", True
        )
        # 刷新前捕获目标 revision（pg 模式），成功替换后推进
        captured = self._capture_active_revision()

        if self._multi_gran_enabled:
            mi = self._get_multi_index()
            docs = self._load_documents_from_collection(
                mi.small_name, active_only=active_only
            )
            if self._multi_retriever is not None:
                ok = self._multi_retriever.refresh_documents(docs)
            else:
                self.all_documents = docs  # 待首次检索时初始化
                ok = True
            if ok:
                self._commit_bm25_revision(captured)
            return ok

        docs = self._load_documents_from_collection(
            vector_store_config.get("character_name", "agent"), active_only=active_only
        )
        if not self._retrievers_initialized:
            self.all_documents = docs  # 待首次检索时初始化
            self._commit_bm25_revision(captured)
            return True
        try:
            k = vector_store_config.get("k", 10)
            new_bm25 = (
                BM25Retriever.from_documents(docs, k=k)
                if BM25_AVAILABLE and docs
                else None
            )
            vector_retriever = VectorSimilarityThresholdRetriever(
                vector_store=self.vector_store,
                k=k,
                score_threshold=vector_store_config.get("score_threshold", 0.3),
            )
            if new_bm25:
                new_hybrid = EnsembleRetriever(
                    retrievers=[vector_retriever, new_bm25],
                    weights=[
                        vector_store_config.get("vector_weight", 0.7),
                        vector_store_config.get("bm25_weight", 0.3),
                    ],
                )
            else:
                new_hybrid = vector_retriever
            truncated = TopKRetriever(
                base_retriever=new_hybrid,
                max_docs=vector_store_config.get("max_rerank_docs", 8),
            )
            new_compression = None
            if self._compression_retriever is not None:
                new_compression = ContextualCompressionRetriever(
                    base_compressor=self._compression_retriever.base_compressor,
                    base_retriever=truncated,
                )
        except Exception as e:
            logger.error(f"[VectorStore] BM25 刷新构建失败，保留旧实例: {e}")
            return False
        with self._bm25_lock:
            self.all_documents = docs
            self._bm25_retriever = new_bm25
            self._hybrid_retriever = new_hybrid
            self._compression_retriever = new_compression
        self._commit_bm25_revision(captured)
        logger.info(f"[VectorStore] BM25 原子刷新完成 | 文档数={len(docs)}")
        return True

    def _process_single_file(self, file_path: str) -> list[Document]:
        """处理单个文件：两阶段管道（Loader → Dispatcher → Splitter）。

        Returns:
            切分后的 Document 列表（空列表表示处理失败或结果为空）
        """
        if self._pipeline is None:
            from agent_material.rag_engine.document_pipeline import ProcessingPipeline
            self._pipeline = ProcessingPipeline()

        return self._pipeline.process(file_path)

    def load_document(self) -> list[int]:
        """扫描目录并增量加载文档（扫描 → 入队 → Worker 异步处理）。

        pg_control_plane 开启（Spec V2 6.1）：PG 控制面扫描，只创建任务，
        不提前更新 content_hash / 标记 deleted / 改 active_version_id；
        关闭：旧 SQLite 控制面（回滚路径，行为不变）。

        Returns:
            成功入队的 task_id 列表（PG 模式 dedup 命中返回 None 不计入）
        """
        if self._pg_control_plane_enabled():
            repos = self._get_pg_repos()
            from agent_material.rag_engine.db.worker import pg_scan_and_enqueue

            task_ids = pg_scan_and_enqueue(
                directory=vector_store_config["document_directory"],
                allow_types=vector_store_config["allow_type"],
                task_repo=repos["tasks"],
                file_state_repo=repos["file_states"],
                kb_id=vector_store_config.get("character_name", "agent"),
            )
            self._ensure_worker_started()
            logger.info(f"[VectorStore] PG 扫描入队完成 | 共入队 {len(task_ids)}")
            return task_ids

        from agent_material.rag_engine.db.task_queue import TaskType

        tracker = self._get_file_tracker()
        queue = self._get_task_queue()

        versioned = vector_store_config.get("feature_flags", {}).get(
            "versioned_ingest", True
        )
        result = tracker.scan_directory(
            directory=vector_store_config["document_directory"],
            allow_types=vector_store_config["allow_type"],
            rapid=True,
            defer_state_update=versioned,
        )
        file_info = result.get("file_info", {})

        # 入队处理任务
        task_ids: list[int] = []
        for file_path in result["new"] + result["modified"]:
            payload = {"file_path": file_path}
            if versioned:
                # md5/mtime/size 随任务传递，激活成功后才落库（OFF-002）
                payload.update(file_info.get(file_path, {}))
            task_ids.append(queue.enqueue(TaskType.INGEST_FILE, payload))

        for file_path in result["deleted"]:
            task_ids.append(
                queue.enqueue(
                    TaskType.DELETE_FILE,
                    {"source": file_path, "file_path": file_path},
                )
            )

        # 启动 Worker（首次调用时）
        self._ensure_worker_started()

        logger.info(
            f"[VectorStore] 扫描入队完成 | 新增 {len(result['new'])} | "
            f"修改 {len(result['modified'])} | 未变 {len(result['unchanged'])} | "
            f"已删 {len(result['deleted'])} | 共入队 {len(task_ids)}"
        )
        return task_ids

    def _get_file_tracker(self):
        if self._file_tracker is None:
            from agent_material.rag_engine.db.file_tracker import FileTracker
            from agent_material.utils.config_handler import queue_config

            db_path = queue_config.get("queue", {}).get("db_path", "rag_queue.db")
            self._file_tracker = FileTracker(db_path=db_path)
        return self._file_tracker

    def _get_task_queue(self):
        if self._task_queue is None:
            from agent_material.rag_engine.db.task_queue import TaskQueue
            from agent_material.utils.config_handler import queue_config

            db_path = queue_config.get("queue", {}).get("db_path", "rag_queue.db")
            self._task_queue = TaskQueue(db_path=db_path)
        return self._task_queue

    def _ensure_worker_started(self):
        if self._worker is not None and self._worker.is_running():
            return
        from agent_material.rag_engine.db.worker import BackgroundWorker
        from agent_material.utils.config_handler import queue_config

        queue_cfg = queue_config.get("queue", {})
        if self._pg_control_plane_enabled():
            # PG 控制面（Spec V2）：Worker 使用 PG 仓储，不再读写 SQLite
            repos = self._get_pg_repos()
            self._worker = BackgroundWorker(
                task_repo=repos["tasks"],
                version_repo=repos["versions"],
                file_state_repo=repos["file_states"],
                kb_repo=repos["kb"],
                poll_interval=queue_cfg.get("poll_interval", 2.0),
                max_retries=queue_cfg.get("max_retries", 3),
                lease_seconds=queue_cfg.get("lease_seconds", 300),
                on_activated=self.refresh_bm25,
            )
        else:
            from agent_material.rag_engine.db.version_registry import DocumentVersionRegistry

            db_path = queue_cfg.get("db_path", "rag_queue.db")
            self._worker = BackgroundWorker(
                task_queue=self._get_task_queue(),
                file_tracker=self._get_file_tracker(),
                poll_interval=queue_cfg.get("poll_interval", 2.0),
                max_retries=queue_cfg.get("max_retries", 3),
                on_activated=self.refresh_bm25,
                version_registry=DocumentVersionRegistry(db_path=db_path),
            )
        self._worker.start()

    def enqueue_ingest_text(self, text: str, source: str) -> int | None:
        """PG 模式：文本封装 ingest_text 任务入队（Spec 11 统一写入入口）。

        dedup_key 由内容 hash + 规范化 source 派生，同内容同来源不重复入队。
        Returns:
            task_id；已有相同 pending/processing 任务时返回 None。
        """
        from agent_material.rag_engine import versioning

        repos = self._get_pg_repos()
        content_hash = versioning.compute_content_hash(text.encode("utf-8"))
        kb_id = vector_store_config.get("character_name", "agent")
        document_id = versioning.compute_document_id(source, kb_id=kb_id)
        task_id = repos["tasks"].enqueue(
            "ingest_text",
            {"text": text, "source": source},
            dedup_key=f"ingest_text:{document_id}:{content_hash}",
        )
        self._ensure_worker_started()
        return task_id

    def get_task_status(self, task_id: int) -> dict | None:
        """查询任务状态：pg 模式查 TaskRepository，sqlite 模式查 TaskQueue。"""
        if self._pg_control_plane_enabled():
            row = self._get_pg_repos()["tasks"].get(int(task_id))
            if row is None:
                return None
            return {
                k: (v.isoformat() if hasattr(v, "isoformat") else v)
                for k, v in row.items()
            }
        return self._get_task_queue().get_status(int(task_id))

    def get_kb_state(self) -> dict | None:
        """知识库 revision 状态（Spec 15 可运维性：active_revision/bm25_revision 可观测）。

        pg_control_plane 关闭（SQLite 回滚路径）时返回 None。
        """
        if not self._pg_control_plane_enabled():
            return None
        kb_id = self._kb_id()
        row = self._get_pg_repos()["kb"].get(kb_id)
        return {
            "kb_id": kb_id,
            "active_revision": row["active_revision"],
            "bm25_revision": row["bm25_revision"],
            "local_bm25_revision": self._local_bm25_revision,
            "bm25_lagging": row["active_revision"] > self._local_bm25_revision,
            "updated_at": row["updated_at"].isoformat()
            if hasattr(row.get("updated_at"), "isoformat")
            else row.get("updated_at"),
        }

    def run_consistency_check(self) -> dict:
        """在线数据一致性检测（Spec 15 可运维性），基于 langchain_pg_embedding.cmetadata。

        检测项（均应为 0）：
        ① 同一 document_id 存在多个 active version 的文档数；
        ② active child 找不到 active parent 的子块数（仅多粒度模式）。
        """
        import psycopg

        conn_str = vector_store_config.get("connection_string", "")
        if not conn_str:
            raise RuntimeError("未配置 connection_string，无法执行一致性检测")
        psycopg_conn_str = conn_str.replace("postgresql+psycopg://", "postgresql://")

        base = vector_store_config.get("character_name", "agent")
        mg = vector_store_config.get("multi_granularity", {})
        if self._multi_gran_enabled:
            small = base + mg.get("small_collection_suffix", "_small")
            large = base + mg.get("large_collection_suffix", "_large")
            names = [small, large]
        else:
            small = large = base
            names = [base]

        conn = psycopg.connect(psycopg_conn_str)
        try:
            cur = conn.cursor()
            # ① 同一 document_id 多个 active version
            cur.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT e.cmetadata->>'document_id' AS document_id
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = ANY(%s)
                      AND e.cmetadata->>'is_active' = 'true'
                      AND e.cmetadata->>'document_id' IS NOT NULL
                    GROUP BY e.cmetadata->>'document_id'
                    HAVING COUNT(DISTINCT e.cmetadata->>'version_id') > 1
                ) s
                """,
                (names,),
            )
            multi_active = cur.fetchone()[0]

            # ② active child 找不到 active parent（仅多粒度有父子分层）
            orphan = 0
            if self._multi_gran_enabled:
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s
                      AND e.cmetadata->>'is_active' = 'true'
                      AND e.cmetadata->>'parent_id' IS NOT NULL
                      AND NOT EXISTS (
                          SELECT 1
                          FROM langchain_pg_embedding p
                          JOIN langchain_pg_collection pc
                            ON p.collection_id = pc.uuid
                          WHERE pc.name = %s
                            AND p.cmetadata->>'parent_id'
                                = e.cmetadata->>'parent_id'
                            AND p.cmetadata->>'is_active' = 'true'
                      )
                    """,
                    (small, large),
                )
                orphan = cur.fetchone()[0]
            cur.close()
        finally:
            conn.close()

        return {
            "collections": names,
            "multiple_active_version_documents": multi_active,
            "orphan_active_children": orphan,
            "ok": multi_active == 0 and orphan == 0,
        }

    def add_documents(self, documents: list[Document]):
        """手动添加文档（供外部调用），添加后自动重建检索器。

        优先使用 extension_pipeline 混合切分，否则回退 extension_splitter_map。
        """
        if not documents:
            return

        from agent_material.rag_engine.text_spliter import RecursiveSplitter, HybridSplitter

        ext_pipeline: dict[str, list[str]] = vector_store_config.get("extension_pipeline", {})

        # 按扩展名分组
        docs_by_ext: dict[str, list[Document]] = {}
        for doc in documents:
            source = doc.metadata.get("source", "")
            ext = os.path.splitext(source)[1].lower() or ".unknown"
            docs_by_ext.setdefault(ext, []).append(doc)

        all_splits: list[Document] = []
        for ext, docs in docs_by_ext.items():
            steps = ext_pipeline.get(ext, ["recursive"])
            try:
                splitter = HybridSplitter(steps)
                splits = splitter.split_documents(docs)
                all_splits.extend(splits)
                logger.info(f"手动添加文档: {ext} → {steps} | 产出 {len(splits)} chunks")
            except Exception as e:
                logger.error(f"手动添加文档切分失败 ({ext}): {e}")
                fallback = RecursiveSplitter()
                splits = fallback.split_documents(docs)
                all_splits.extend(splits)

        if not all_splits:
            return

        # metadata 增强（补充语言/类型/长度等统计信息）
        try:
            from agent_material.rag_engine.metadata_enhancer import MetadataEnhancer
            enhancer = MetadataEnhancer()
            all_splits = enhancer.enhance(all_splits)
        except Exception as e:
            logger.warning(f"Metadata 增强失败，跳过: {e}")

        self.vector_store.add_documents(all_splits)
        self.all_documents.extend(all_splits)
        self._retrievers_initialized = False
        logger.info(
            f"手动添加了 {len(all_splits)} 条文档，检索器将在下次查询时自动重建"
        )

    def delete_documents_by_source(self, source: str):
        """根据来源路径删除文档（支持多粒度索引）。"""
        if self._multi_gran_enabled:
            self._get_multi_index().delete_by_source(source)
        else:
            import psycopg
            conn_str = vector_store_config.get("connection_string")
            collection_name = vector_store_config.get("character_name", "agent")
            if not conn_str:
                logger.error("未配置 connection_string，无法删除文档")
                return
            psycopg_conn_str = conn_str.replace("postgresql+psycopg://", "postgresql://")
            try:
                conn = psycopg.connect(psycopg_conn_str)
                cur = conn.cursor()
                cur.execute(
                    """
                    DELETE FROM langchain_pg_embedding
                    WHERE uuid IN (
                        SELECT e.uuid
                        FROM langchain_pg_embedding e
                        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                        WHERE c.name = %s AND e.cmetadata->>'source' = %s
                    )
                    """,
                    (collection_name, source),
                )
                deleted = cur.rowcount
                conn.commit()
                cur.close()
                conn.close()
                logger.info(f"已删除来源为 {source} 的 {deleted} 条文档")
            except Exception as e:
                logger.error(f"删除文档失败: {e}")

        # 同步清理内存和追踪
        self.all_documents = [
            d for d in self.all_documents if d.metadata.get("source") != source
        ]
        self._retrievers_initialized = False
        self._remove_md5_hex_for_source(source)
        # 删除后同步刷新 BM25（OFF-007）
        try:
            self.refresh_bm25()
        except Exception as e:
            logger.error(f"[VectorStore] 删除后 BM25 刷新失败（保留旧实例）: {e}")

    # ------------------------------------------------------------------ #
    #  MD5 辅助（已委托给 FileTracker）
    # ------------------------------------------------------------------ #
    def _remove_md5_hex_for_source(self, source: str) -> bool:
        """移除 source 的 MD5 记录（委托 FileTracker）。"""
        self._get_file_tracker().remove_source(source)
        return True

    def get_document_count(self) -> int:
        """获取当前向量库中的文档总数。

        支持多粒度索引（返回两个 collection 的总和）。
        """
        if self._multi_gran_enabled:
            counts = self._get_multi_index().get_document_count()
            return counts.get("total", 0)

        import psycopg
        conn_str = vector_store_config.get("connection_string")
        collection_name = vector_store_config.get("character_name", "agent")
        if not conn_str:
            return len(self.all_documents)
        psycopg_conn_str = conn_str.replace("postgresql+psycopg://", "postgresql://")
        try:
            conn = psycopg.connect(psycopg_conn_str)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT COUNT(*) FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s
                """,
                (collection_name,),
            )
            count = cur.fetchone()[0]
            cur.close()
            conn.close()
            return count
        except Exception:
            return len(self.all_documents)

    def get_all_sources(self) -> list[dict]:
        """获取向量库中所有唯一的文献来源列表。

        支持多粒度索引。
        """
        if self._multi_gran_enabled:
            return self._get_multi_index().get_all_sources()

        import psycopg
        conn_str = vector_store_config.get("connection_string")
        collection_name = vector_store_config.get("character_name", "agent")
        if not conn_str:
            logger.error("未配置 connection_string，无法获取文献列表")
            return []
        psycopg_conn_str = conn_str.replace("postgresql+psycopg://", "postgresql://")
        try:
            conn = psycopg.connect(psycopg_conn_str)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT e.cmetadata->>'source' AS source, COUNT(*) AS chunk_count
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s AND e.cmetadata->>'source' IS NOT NULL
                GROUP BY e.cmetadata->>'source'
                ORDER BY source
                """,
                (collection_name,),
            )
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return [{"source": row[0], "chunk_count": row[1]} for row in rows]
        except Exception as e:
            logger.error(f"获取文献列表失败: {e}")
            return []


if __name__ == "__main__":
    vector_store = VectorStore()
    print(f"当前文档数: {vector_store.get_document_count()}")
    vector_store.load_document()
    print(f"加载后文档数: {vector_store.get_document_count()}")
