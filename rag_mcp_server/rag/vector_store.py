import os
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

from rag_mcp_server.observability.metrics import (
    retrieval_duration,
    retrieval_docs,
    postgres_query_duration,
)

from rag_mcp_server.config import vector_store_config
from rag_mcp_server.models import embedding_model
from rag_mcp_server.utils.logger import logger

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
        t0 = time.time()

        # HyDE 模式：使用预计算 embedding 做向量检索
        if self.embedding_override is not None:
            docs_with_scores = self.vector_store.similarity_search_with_score_by_vector(
                self.embedding_override, k=self.k
            )
        else:
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=self.k)

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

        # ── 多粒度索引 ──
        mg_config = vector_store_config.get("multi_granularity", {})
        self._multi_gran_enabled = mg_config.get("enabled", False)
        self._multi_index = None
        self._multi_retriever = None

        # ── 文件追踪 & 任务队列（懒加载）──
        self._file_tracker = None
        self._task_queue = None
        self._worker = None

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
                    from rag_mcp_server.rag.dashscope_reranker import DashScopeReranker

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
            from rag_mcp_server.rag.vector_index import (
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
        """
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
        """初始化多粒度检索器。"""
        from rag_mcp_server.indexing.multi_granularity import MultiGranularityIndex
        from rag_mcp_server.indexing.multi_gran_retriever import MultiGranularityRetriever

        mi = self._get_multi_index()
        mi.small_store  # 触发 PGVector 初始化
        mi.large_store  # 触发 PGVector 初始化

        # 加载小chunk文档用于 BM25
        small_docs = self._load_documents_from_collection(mi.small_name)

        self._multi_retriever = MultiGranularityRetriever(
            small_store=mi.small_store,
            large_store=mi.large_store,
            all_documents=small_docs,
        )
        self.all_documents = small_docs  # 保持兼容
        logger.info("[VectorStore] 多粒度检索器初始化完成")

    def _get_multi_index(self):
        if self._multi_index is None:
            from rag_mcp_server.indexing.multi_granularity import MultiGranularityIndex
            self._multi_index = MultiGranularityIndex(enabled=self._multi_gran_enabled)
        return self._multi_index

    def _load_documents_from_collection(self, collection_name: str) -> list[Document]:
        """从指定 PGVector collection 加载所有文档到内存。"""
        import psycopg
        conn_str = vector_store_config.get("connection_string")
        if not conn_str:
            return []
        psycopg_conn_str = conn_str.replace("postgresql+psycopg://", "postgresql://")
        docs = []
        try:
            conn = psycopg.connect(psycopg_conn_str)
            cur = conn.cursor()
            cur.execute(
                """SELECT e.document, e.cmetadata
                   FROM langchain_pg_embedding e
                   JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                   WHERE c.name = %s""",
                (collection_name,),
            )
            for content, meta in cur.fetchall():
                docs.append(Document(page_content=content, metadata=meta or {}))
            cur.close()
            conn.close()
            logger.info(f"从 {collection_name} 加载了 {len(docs)} 条文档")
        except Exception as e:
            logger.warning(f"加载 {collection_name} 文档失败: {e}")
        return docs

    def _process_single_file(self, file_path: str) -> list[Document]:
        """处理单个文件：两阶段管道（Loader → Dispatcher → Splitter）。

        Returns:
            切分后的 Document 列表（空列表表示处理失败或结果为空）
        """
        if self._pipeline is None:
            from rag_mcp_server.rag.document_pipeline import ProcessingPipeline
            self._pipeline = ProcessingPipeline()

        return self._pipeline.process(file_path)

    def load_document(self):
        """扫描目录并增量加载文档（扫描 → 入队 → Worker 异步处理）。

        流程：
        1. FileTracker 扫描目录，识别新增/修改/删除文件
        2. 新增/修改文件入队 ingest_file
        3. 已删除文件入队 delete_file
        4. BackgroundWorker 异步消费任务，处理 Pipeline → 写入向量库
        5. 非阻塞返回（任务在后台执行）
        """
        from rag_mcp_server.db.task_queue import TaskType

        tracker = self._get_file_tracker()
        queue = self._get_task_queue()

        result = tracker.scan_directory(
            directory=vector_store_config["document_directory"],
            allow_types=vector_store_config["allow_type"],
            rapid=True,
        )

        # 入队处理任务
        enqueued = 0
        for file_path in result["new"] + result["modified"]:
            queue.enqueue(TaskType.INGEST_FILE, {"file_path": file_path})
            enqueued += 1

        for file_path in result["deleted"]:
            queue.enqueue(
                TaskType.DELETE_FILE,
                {"source": file_path, "file_path": file_path},
            )
            enqueued += 1

        # 启动 Worker（首次调用时）
        self._ensure_worker_started()

        logger.info(
            f"[VectorStore] 扫描入队完成 | 新增 {len(result['new'])} | "
            f"修改 {len(result['modified'])} | 未变 {len(result['unchanged'])} | "
            f"已删 {len(result['deleted'])} | 共入队 {enqueued}"
        )
        return

    def _get_file_tracker(self):
        if self._file_tracker is None:
            from rag_mcp_server.db.file_tracker import FileTracker
            from rag_mcp_server.config import queue_config

            db_path = queue_config.get("queue", {}).get("db_path", "rag_queue.db")
            self._file_tracker = FileTracker(db_path=db_path)
        return self._file_tracker

    def _get_task_queue(self):
        if self._task_queue is None:
            from rag_mcp_server.db.task_queue import TaskQueue
            from rag_mcp_server.config import queue_config

            db_path = queue_config.get("queue", {}).get("db_path", "rag_queue.db")
            self._task_queue = TaskQueue(db_path=db_path)
        return self._task_queue

    def _ensure_worker_started(self):
        if self._worker is not None and self._worker.is_running():
            return
        from rag_mcp_server.db.worker import BackgroundWorker

        self._worker = BackgroundWorker(
            task_queue=self._get_task_queue(),
            file_tracker=self._get_file_tracker(),
        )
        self._worker.start()

    def add_documents(self, documents: list[Document]):
        """手动添加文档（供外部调用），添加后自动重建检索器。

        优先使用 extension_pipeline 混合切分，否则回退 extension_splitter_map。
        """
        if not documents:
            return

        from rag_mcp_server.rag.text_spliter import RecursiveSplitter, HybridSplitter

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
            from rag_mcp_server.rag.metadata_enhancer import MetadataEnhancer
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
