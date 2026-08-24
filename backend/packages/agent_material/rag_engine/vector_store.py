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
from agent_material.rag_engine.indexing_service import (
    DirectWriteForbiddenError,
    get_indexing_service,
)
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
    from agent_material.rag_engine.bm25_retriever import (
        MaterialBM25Retriever as BM25Retriever,
        build_material_bm25_retriever,
    )

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


class BM25LoadError(RuntimeError):
    """BM25 语料加载失败（BM25-003）：连接/SQL/collection/metadata 契约错误。

    加载失败必须显式抛出，不得退化为"合法空列表"——合法空知识库只能由
    成功 SQL + 0 行结果确认。
    """


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
        self._bm25_rebuild_lock = threading.Lock()  # 整次构建串行，防旧快照后写覆盖

        # ── 多粒度索引 ──
        mg_config = vector_store_config.get("multi_granularity", {})
        self._multi_gran_enabled = mg_config.get("enabled", False)
        self._multi_index = None
        self._multi_retriever = None

        # ── PostgreSQL 后台索引 Worker（懒加载）──
        self._worker = None
        # ── PG 控制面仓储（懒加载）──
        self._pg_repos = None
        # ── 统一索引服务（ENT-001，懒加载，与 REST/Agent 工具同一单例）──
        self._indexing_service = None
        # ── BM25 revision 生命周期（Spec V2 9）──
        self._local_bm25_revision = 0  # 本地 BM25 已构建到的 active_revision
        self._last_bm25_revision_check = 0.0  # 上次 revision 检查的 time.monotonic()

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

    def _get_indexing_service(self):
        """统一索引服务（ENT-001）：与 REST/Agent 工具共用进程内单例。

        注入 _ensure_worker_started 作为入队后的 Worker 唤醒钩子。
        """
        if self._indexing_service is None:
            self._indexing_service = get_indexing_service(
                ensure_worker=self._ensure_worker_started
            )
        return self._indexing_service

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
            active_only = vector_store_config.get("feature_flags", {}).get(
                "active_version_filter", True
            )
            active_clause = (
                " AND e.cmetadata->>'is_active' = 'true'" if active_only else ""
            )
            cur.execute(
                f"""
                SELECT e.document, e.cmetadata
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s
                {active_clause}
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
            self._bm25_retriever = build_material_bm25_retriever(
                self.all_documents,
                k=k,
                config=vector_store_config,
                retriever_cls=BM25Retriever,
            )
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
        """初始化多粒度检索器（冷启动）。

        冷启动与热刷新共用同一个 rebuild_to_revision 状态机（BM25-001）：
        先捕获 target_revision → 加载显式 active 子块 → 构建 → 原子替换
        → local_bm25_revision = target_revision，消除"先加载后读 revision"
        的竞争窗口（INV-007）。

        加载/构建失败时先挂一个空语料的 vector-only 兜底实例（当前请求
        走向量路），revision 不推进，由 maybe_refresh_bm25 节流重试。
        """
        if self.refresh_bm25():
            logger.info("[VectorStore] 多粒度检索器初始化完成")
            return
        if self._multi_retriever is None:
            # 无旧实例可保留：建立空语料兜底实例，保证检索链路可用（vector-only）
            from agent_material.rag_engine.indexing.multi_gran_retriever import MultiGranularityRetriever

            mi = self._get_multi_index()
            mi.small_store  # 触发 PGVector 初始化
            mi.large_store  # 触发 PGVector 初始化
            self._multi_retriever = MultiGranularityRetriever(
                small_store=mi.small_store,
                large_store=mi.large_store,
                all_documents=[],
            )
            self.all_documents = []
            logger.warning(
                "[VectorStore] 冷启动 BM25 构建失败，暂以 vector-only 兜底实例服务，"
                "等待 revision 检测重试"
            )

    def _get_multi_index(self):
        if self._multi_index is None:
            from agent_material.rag_engine.indexing.multi_granularity import MultiGranularityIndex
            self._multi_index = MultiGranularityIndex(enabled=self._multi_gran_enabled)
        return self._multi_index

    def _load_documents_from_collection(
        self,
        collection_name: str,
        active_only: bool = False,
        require_parent_id: bool = False,
    ) -> list[Document]:
        """从指定 PGVector collection 加载所有文档到内存。

        BM25-002（INV-006）：active_only=True 时只加载显式 is_active=true
        且 document_id/version_id/chunk_id 齐全的行；require_parent_id=True
        （多粒度子块）时还要求 parent_id 存在。缺失 is_active 不再视为 active。

        BM25-003：连接失败、SQL 失败、collection 不存在、metadata 契约
        校验失败均抛 BM25LoadError，绝不返回"合法空列表"。合法空知识库
        只能由成功 SQL + 0 行结果确认。
        """
        import psycopg
        conn_str = vector_store_config.get("connection_string")
        if not conn_str:
            raise BM25LoadError("未配置 connection_string，无法加载 BM25 语料")
        psycopg_conn_str = conn_str.replace("postgresql+psycopg://", "postgresql://")
        try:
            conn = psycopg.connect(psycopg_conn_str)
        except Exception as e:
            raise BM25LoadError(f"连接 PostgreSQL 失败: {e}") from e
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT 1 FROM langchain_pg_collection WHERE name = %s",
                (collection_name,),
            )
            if cur.fetchone() is None:
                raise BM25LoadError(f"collection 不存在: {collection_name}")
            sql = """SELECT e.document, e.cmetadata
                     FROM langchain_pg_embedding e
                     JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                     WHERE c.name = %s"""
            if active_only:
                sql += " AND e.cmetadata->>'is_active' = 'true'"
            cur.execute(sql, (collection_name,))
            rows = cur.fetchall()
            cur.close()
        except BM25LoadError:
            raise
        except Exception as e:
            raise BM25LoadError(
                f"加载 {collection_name} 文档 SQL 执行失败: {e}"
            ) from e
        finally:
            conn.close()

        docs = []
        required = ("document_id", "version_id", "chunk_id")
        for content, meta in rows:
            meta = meta or {}
            if active_only:
                # metadata 契约校验（INV-006）：active 行必须键齐全，
                # 多粒度子块还必须带 parent_id；缺任一即加载失败
                missing = [k for k in required if not meta.get(k)]
                if require_parent_id and not meta.get("parent_id"):
                    missing.append("parent_id")
                if missing:
                    raise BM25LoadError(
                        f"{collection_name} 存在 metadata 契约不全的 active 行 "
                        f"(缺失 {missing}，chunk_id={meta.get('chunk_id')!r})，"
                        f"请先完成存量迁移回填"
                    )
            docs.append(Document(page_content=content, metadata=meta))
        logger.info(f"从 {collection_name} 加载了 {len(docs)} 条文档")
        return docs

    # ------------------------------------------------------------------ #
    #  BM25 revision 生命周期（Spec V3 11）
    # ------------------------------------------------------------------ #
    def _kb_id(self) -> str:
        """知识库 ID（与 worker/扫描入队侧一致：character_name）。"""
        return vector_store_config.get("character_name", "agent")

    def maybe_refresh_bm25(self) -> None:
        """BM25 revision 检测刷新（Spec 9.2）：多进程兜底的热刷新路径。

        每次检索请求路径调用（get_retriever 顶部），节流
        `bm25_revision_check_interval` 秒（默认 2s）查一次 PG；
        active_revision 领先本地 bm25_revision 时重建 BM25。
        进程内 on_activated 回调仍是快路径，本机制覆盖多查询进程场景。
        所有异常只记日志，绝不影响检索链路。
        """
        interval = float(
            vector_store_config.get("bm25_revision_check_interval", 2.0)
        )
        now = time.monotonic()
        # 0.0 是“尚未检查/显式要求立即检查”的哨兵值。不能直接参与
        # monotonic 差值判断，否则系统启动不足 interval 秒时首次检查会被误节流。
        if (
            self._last_bm25_revision_check > 0.0
            and now - self._last_bm25_revision_check < interval
        ):
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
        """刷新前读取 (kb_repo, kb_id, target_revision)；失败返回 None。"""
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
        self._local_bm25_revision = max(self._local_bm25_revision, target)
        try:
            kb_repo.update_bm25_revision(kb_id, target)
        except Exception as e:
            logger.warning(f"[VectorStore] 回写 bm25_revision 失败: {e}")

    def refresh_bm25(self) -> bool:
        """串行执行整次 BM25 构建，避免较旧快照在并发刷新中最后覆盖。"""
        lock = getattr(self, "_bm25_rebuild_lock", None)
        if lock is None:  # 兼容少量通过 __new__ 构造的轻量测试对象
            lock = self._bm25_rebuild_lock = threading.Lock()
        with lock:
            return self._refresh_bm25_locked()

    def _refresh_bm25_locked(self) -> bool:
        """唯一 BM25 构建状态机（BM25-001 / INV-007）：冷启动、热刷新、
        手工 rebuild 共用同一 rebuild_to_revision 语义流程：

            捕获 target_revision → 加载显式 active 子块 → 锁外构建新 BM25
            → 锁内原子替换 → local_bm25_revision = target_revision

        - 先捕获 revision：构建窗口内的新激活不会被误标为已构建，
          最坏情况是下一次检测再刷新（BM25-001）。
        - 加载失败（连接/SQL/collection/metadata 契约）显式失败：
          保留旧实例、revision 不推进、下节流窗口重试（BM25-003）。
        - 合法空知识库（成功 SQL + 0 行）正常构建 vector-only 实例并
          推进 revision。

        Returns:
            True=语料/检索器已更新到 target_revision（revision 已推进）；
            False=加载或构建失败保留旧实例（revision 不推进，等待重试）。
        """
        # 多粒度模式始终只加载 active（BM25-002：staging/retired 不进 BM25）；
        # 非多粒度旧链路保持 active_version_filter 开关语义（回滚兼容）。
        flags = vector_store_config.get("feature_flags", {})
        active_only = True if self._multi_gran_enabled else flags.get(
            "active_version_filter", True
        )
        # 刷新前捕获目标 revision（pg 模式），成功替换后推进
        captured = self._capture_active_revision()
        kb_id = captured[1] if captured else self._kb_id()
        target = captured[2] if captured else None
        logger.info(
            "bm25_refresh_started",
            extra={"kb_id": kb_id, "target_revision": target},
        )

        # ── 加载（失败显式抛错 → 本次刷新失败，不得当作空语料）──
        try:
            if self._multi_gran_enabled:
                mi = self._get_multi_index()
                docs = self._load_documents_from_collection(
                    mi.small_name, active_only=active_only, require_parent_id=True
                )
            else:
                docs = self._load_documents_from_collection(
                    vector_store_config.get("character_name", "agent"),
                    active_only=active_only,
                )
        except Exception as e:
            logger.warning(
                "bm25_refresh_failed",
                extra={"kb_id": kb_id, "stage": "load", "error": str(e)},
            )
            return False

        if self._multi_gran_enabled:
            if self._multi_retriever is not None:
                ok = self._multi_retriever.refresh_documents(docs)
            else:
                # 冷启动：锁外构建新检索器实例，成功后锁内原子挂上
                from agent_material.rag_engine.indexing.multi_gran_retriever import (
                    MultiGranularityRetriever,
                )

                try:
                    new_retriever = MultiGranularityRetriever(
                        small_store=mi.small_store,
                        large_store=mi.large_store,
                        all_documents=docs,
                    )
                except Exception as e:
                    logger.warning(
                        "bm25_refresh_failed",
                        extra={"kb_id": kb_id, "stage": "build", "error": str(e)},
                    )
                    return False
                with self._bm25_lock:
                    self._multi_retriever = new_retriever
                    self.all_documents = docs
                ok = True
            if ok:
                self._commit_bm25_revision(captured)
                logger.info(
                    "bm25_refresh_succeeded",
                    extra={
                        "kb_id": kb_id,
                        "local_bm25_revision": self._local_bm25_revision,
                        "doc_count": len(docs),
                    },
                )
            else:
                logger.warning(
                    "bm25_refresh_failed",
                    extra={"kb_id": kb_id, "stage": "build", "error": "refresh_documents 返回 False"},
                )
            return ok

        # ── 非多粒度旧链路（回滚兼容）──
        if not self._retrievers_initialized:
            self.all_documents = docs  # 待首次检索时初始化
            self._commit_bm25_revision(captured)
            return True
        try:
            k = vector_store_config.get("k", 10)
            new_bm25 = (
                build_material_bm25_retriever(
                    docs,
                    k=k,
                    config=vector_store_config,
                    retriever_cls=BM25Retriever,
                )
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
            logger.warning(
                "bm25_refresh_failed",
                extra={"kb_id": kb_id, "stage": "build", "error": str(e)},
            )
            return False
        with self._bm25_lock:
            self.all_documents = docs
            self._bm25_retriever = new_bm25
            self._hybrid_retriever = new_hybrid
            self._compression_retriever = new_compression
        self._commit_bm25_revision(captured)
        logger.info(
            "bm25_refresh_succeeded",
            extra={
                "kb_id": kb_id,
                "local_bm25_revision": self._local_bm25_revision,
                "doc_count": len(docs),
            },
        )
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
        """使用 PostgreSQL 控制面扫描目录并创建异步索引任务。

        扫描阶段不提前更新 content_hash、deleted 或 active_version_id；
        文件状态只在版本激活事务中更新。

        Returns:
            成功入队的 task_id 列表（PG 模式 dedup 命中返回 None 不计入）
        """
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

    def _ensure_worker_started(self):
        if self._worker is not None and self._worker.is_running():
            return
        from agent_material.rag_engine.db.worker import BackgroundWorker
        from agent_material.utils.config_handler import queue_config

        queue_cfg = queue_config.get("queue", {})
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
        self._worker.start()

    def start_worker(self) -> None:
        """启动 PG 后台 Worker，并恢复 pending/租约过期任务。"""
        self._ensure_worker_started()

    def stop_worker(self, timeout: float = 10.0) -> None:
        """停止后台 Worker（WRK-001 shutdown）：停止领取新任务，等待当前任务
        到安全点退出，并记录未完成任务状态。"""
        worker = self._worker
        if worker is None:
            return
        worker.stop()
        worker.join(timeout=timeout)
        try:
            summary = self.get_task_summary()
            logger.info(f"[VectorStore] Worker 已停止，任务队列终态: {summary}")
        except Exception as e:
            logger.warning(f"[VectorStore] Worker 停止后读取任务摘要失败: {e}")

    def enqueue_ingest_text(self, text: str, source: str) -> int | None:
        """文本封装 ingest_text 任务入队（统一写入入口，ENT-001）。

        source 先经统一 canonicalizer 规范化（ENT-002），dedup_key 由
        内容 hash + canonical source 派生，同内容同来源不重复入队。
        Returns:
            task_id；已有相同 pending/processing 任务时返回 None。
        """
        return self._get_indexing_service().enqueue_text(text, source)

    def enqueue_ingest_file(self, file_path: str) -> int | None:
        """单个已落盘文件封装 ingest_file 任务入队（统一写入入口，ENT-001）。

        与目录扫描同一 canonical/dedup 契约；已追踪且内容 hash 未变的文件
        幂等跳过。Returns: task_id；幂等命中或重复任务时返回 None。
        """
        return self._get_indexing_service().enqueue_file(file_path)

    def enqueue_delete_by_source(self, source: str) -> dict:
        """按来源创建 delete_document 任务（ENT-003）。

        source 不存在也允许创建（Worker 侧幂等 no-op completed）。
        Returns:
            {"source", "canonical_source", "task_id", "status"}，
            status=accepted / deduplicated（后者 task_id 为 None）。
        """
        return self._get_indexing_service().enqueue_delete(source)

    def enqueue_batch_delete(self, sources: list[str]) -> list[dict]:
        """批量创建 delete_document 任务，每个 source 独立明细（ENT-003）。"""
        return self._get_indexing_service().enqueue_batch_delete(sources)

    def get_task_status(self, task_id: int) -> dict | None:
        """查询 PostgreSQL 控制面的任务状态。"""
        return self._get_indexing_service().get_task(task_id)

    def get_task_summary(self) -> dict:
        """返回 PostgreSQL 控制面的任务状态汇总。"""
        return self._get_indexing_service().get_task_summary()

    def retry_failed_task(self, task_id: int) -> dict:
        """显式重试 failed 索引任务。"""
        return self._get_indexing_service().retry_failed_task(task_id)

    def get_active_artifact(self, artifact_id: str) -> dict | None:
        """读取 active 版本的图表元数据，供受认证的资源接口使用。"""
        return self._get_pg_repos()["versions"].get_artifact(
            artifact_id, active_only=True
        )

    def get_kb_state(self) -> dict | None:
        """知识库 revision 状态（active_revision/bm25_revision 可观测）。"""
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
        ② active child 找不到同 document/version 的唯一 active parent；
        ③ 版本化检索必需 metadata 缺失的向量行数。
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

            # ② active child 找不到同 document/version 的唯一 active parent
            orphan = 0
            if self._multi_gran_enabled:
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    LEFT JOIN (
                        SELECT p.cmetadata->>'parent_id' AS parent_id,
                               p.cmetadata->>'document_id' AS document_id,
                               p.cmetadata->>'version_id' AS version_id
                        FROM langchain_pg_embedding p
                        JOIN langchain_pg_collection pc ON p.collection_id = pc.uuid
                        WHERE pc.name = %s
                          AND p.cmetadata->>'is_active' = 'true'
                        GROUP BY 1, 2, 3
                        HAVING COUNT(*) = 1
                    ) valid_parent
                      ON valid_parent.parent_id = e.cmetadata->>'parent_id'
                     AND valid_parent.document_id = e.cmetadata->>'document_id'
                     AND valid_parent.version_id = e.cmetadata->>'version_id'
                    WHERE c.name = %s
                      AND e.cmetadata->>'is_active' = 'true'
                      AND e.cmetadata->>'parent_id' IS NOT NULL
                      AND valid_parent.parent_id IS NULL
                    """,
                    (large, small),
                )
                orphan = cur.fetchone()[0]

            # ③ 所有参与版本化检索的行都必须具备完整身份与显式 active 标记。
            # chunk_id 是子块身份，父块只要求 parent_id；不能把合法父块误报为缺口。
            cur.execute(
                """
                SELECT COUNT(*)
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = ANY(%s)
                  AND (
                    NULLIF(e.cmetadata->>'source', '') IS NULL OR
                    NULLIF(e.cmetadata->>'document_id', '') IS NULL OR
                    NULLIF(e.cmetadata->>'version_id', '') IS NULL OR
                    NULLIF(e.cmetadata->>'parent_id', '') IS NULL OR
                    (
                      c.name = %s
                      AND COALESCE(e.cmetadata->>'chunk_type', 'child') <> 'parent'
                      AND NULLIF(e.cmetadata->>'chunk_id', '') IS NULL
                    ) OR
                    NOT (e.cmetadata ? 'is_active')
                  )
                """,
                (names, small),
            )
            metadata_gaps = cur.fetchone()[0]
            cur.close()
        finally:
            conn.close()

        return {
            "collections": names,
            "multiple_active_version_documents": multi_active,
            "orphan_active_children": orphan,
            "metadata_contract_gaps": metadata_gaps,
            "ok": multi_active == 0 and orphan == 0 and metadata_gaps == 0,
        }

    def add_documents(self, documents: list[Document]):
        """拒绝绕过版本控制的直接写入。

        手动文本必须通过 enqueue_ingest_text / RagIndexingService.enqueue_text，
        以满足父子元数据、staging 校验和原子激活约束。
        """
        raise DirectWriteForbiddenError(
            "禁止 VectorStore.add_documents 直写 PGVector（ENT-005/INV-002），"
            "请改用 enqueue_ingest_text 走统一索引任务"
        )

    def delete_documents_by_source(self, source: str):
        """拒绝绕过版本控制的直接物理删除。"""
        raise DirectWriteForbiddenError(
            "禁止 VectorStore.delete_documents_by_source 直删 PGVector（ENT-005/INV-002），"
            "请改用 enqueue_delete_by_source 创建删除任务"
        )

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
        active_only = vector_store_config.get("feature_flags", {}).get(
            "active_version_filter", True
        )
        active_clause = " AND e.cmetadata->>'is_active' = 'true'" if active_only else ""
        try:
            conn = psycopg.connect(psycopg_conn_str)
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT COUNT(*) FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s
                {active_clause}
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
        active_only = vector_store_config.get("feature_flags", {}).get(
            "active_version_filter", True
        )
        active_clause = " AND e.cmetadata->>'is_active' = 'true'" if active_only else ""
        try:
            conn = psycopg.connect(psycopg_conn_str)
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT e.cmetadata->>'source' AS source, COUNT(*) AS chunk_count
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s AND e.cmetadata->>'source' IS NOT NULL
                {active_clause}
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


# ── 进程内统一单例（WRK-001：REST / RagService / Agent Tool 共享同一实例）──
_shared_store: VectorStore | None = None
_shared_lock = threading.Lock()


def get_vector_store() -> VectorStore:
    """返回进程内唯一的 VectorStore 实例（懒加载，线程安全）。

    单例保证只有一个后台 Worker、一份 BM25 状态，避免多实例状态观测错位。
    """
    global _shared_store
    with _shared_lock:
        if _shared_store is None:
            _shared_store = VectorStore()
    return _shared_store


def peek_vector_store() -> VectorStore | None:
    """返回已创建的单例（未创建返回 None）；shutdown 等场景避免触发构造。"""
    return _shared_store


if __name__ == "__main__":
    vector_store = VectorStore()
    print(f"当前文档数: {vector_store.get_document_count()}")
    vector_store.load_document()
    print(f"加载后文档数: {vector_store.get_document_count()}")
