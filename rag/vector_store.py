import os
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

from utils.config_handler import chroma_config
from model.model_factory import embedding_model
from utils.path_tools import get_abs_path
from utils.logger import logger
from utils.file_handler import (
    txt_loader,
    md_loader,
    pdf_loader,
    docx_loader,
    pptx_loader,
    filelist_with_allow_type,
    get_file_md5_hex,
)


class ChromaSimilarityThresholdRetriever(BaseRetriever):
    """自定义 Chroma 检索器，正确处理 cosine distance 与 similarity 的转换。

    Chroma 默认使用 cosine distance，范围 [0, 2]：
        - distance = 0  → 完全相同（similarity = 1.0）
        - distance = 2  → 完全相反（similarity = 0.0）

    LangChain 的 `similarity_score_threshold` 在不同版本中对 Chroma score 的处理
    存在不一致（有时把 distance 直接当 similarity 比较），导致召回率异常。
    本检索器显式做转换：similarity = 1 - distance / 2，然后与阈值比较。

    Args:
        vector_store: Chroma 向量库实例
        k: 召回数量上限
        score_threshold: similarity 阈值（0~1，越大越严格），默认 0.3
    """

    # Pydantic v2 兼容：允许运行时动态添加属性（如 self.vector_store）
    model_config = ConfigDict(extra="allow")

    def __init__(
        self,
        vector_store: Chroma,
        k: int = 10,
        score_threshold: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vector_store = vector_store
        self.k = k
        self.score_threshold = score_threshold

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        # 获取原始 distance（Chroma 的 similarity_search_with_score 返回 distance）
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
                f"[ChromaRetriever] 查询 '{query[:30]}...' 经阈值过滤后无结果，"
                f"自动回退到 Top-1（distance={docs_with_scores[0][1]:.3f}）"
            )
            results.append(docs_with_scores[0][0])

        return results

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
        # 根据配置选择切分策略
        splitter_type = chroma_config.get("splitter_type", "recursive")
        if splitter_type == "semantic":
            from rag.text_spliter import SemanticTextSplitter

            self.splitter = SemanticTextSplitter(
                chunk_size=chroma_config["chunk_size"],
                chunk_overlap=chroma_config["chunk_overlap"],
                separator=chroma_config["separator"],
                embedding_model=embedding_model,
                similarity_threshold=chroma_config.get("similarity_threshold", 0.75),
            )
            logger.info("使用语义切分器 (SemanticTextSplitter)")
        else:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chroma_config["chunk_size"],
                chunk_overlap=chroma_config["chunk_overlap"],
                separators=chroma_config["separator"],
                length_function=len,
            )
            logger.info("使用标准切分器 (RecursiveCharacterTextSplitter)")

        self.vector_store = Chroma(
            collection_name=chroma_config["character_name"],
            embedding_function=embedding_model,
            persist_directory=chroma_config["persist_directory"],
        )

        # 维护内存中的文档列表，用于构建 BM25 索引
        self.all_documents: list[Document] = []
        self._load_existing_documents()

        # 检索器缓存（延迟初始化，避免模块导入时加载重排序模型）
        self._bm25_retriever = None
        self._hybrid_retriever = None
        self._compression_retriever = None
        self._retrievers_initialized = False

    def _load_existing_documents(self):
        """从 Chroma 加载已有文档到内存，用于构建 BM25 索引"""
        try:
            data = self.vector_store.get()
            if data and data.get("documents"):
                docs = data["documents"]
                metadatas = data.get("metadatas") or [{}] * len(docs)
                for i, content in enumerate(docs):
                    self.all_documents.append(
                        Document(page_content=content, metadata=metadatas[i])
                    )
                logger.info(f"从 Chroma 加载了 {len(self.all_documents)} 条已有文档")
        except Exception as e:
            logger.warning(f"加载已有文档失败: {e}")

    def _init_retrievers(self):
        """初始化混合检索器（向量 + BM25）和重排序器"""
        k = chroma_config.get("k", 10)
        score_threshold = chroma_config.get("score_threshold", 0.3)

        # 1. 向量检索器（带相似度阈值过滤）
        # 使用自定义检索器，显式处理 Chroma cosine distance → similarity 转换，
        # 避免 LangChain 内置 similarity_score_threshold 的行为不一致问题
        vector_retriever = ChromaSimilarityThresholdRetriever(
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
                    chroma_config.get("vector_weight", 0.7),
                    chroma_config.get("bm25_weight", 0.3),
                ],
            )
            logger.info("混合检索器初始化成功")
        else:
            self._hybrid_retriever = vector_retriever
            logger.info("仅使用向量检索器（BM25 未启用）")

        # 4. CrossEncoder 重排序器：对混合检索结果精排
        if RERANKER_AVAILABLE:
            try:
                model_path = chroma_config.get(
                    "reranker_model", "BAAI/bge-reranker-base"
                )

                # 判断是本地路径还是 HuggingFace Hub 模型名
                is_local = os.path.exists(model_path) and os.path.isdir(model_path)
                if is_local:
                    logger.info(f"检测到本地重排序模型: {model_path}")
                else:
                    logger.info(f"将从 HuggingFace Hub 加载重排序模型: {model_path}")

                # 自动选择 device：GPU 优先，否则 CPU
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"
                logger.info(f"重排序模型使用设备: {device}")

                cross_encoder = HuggingFaceCrossEncoder(
                    model_name=model_path, model_kwargs={"device": device}
                )

                # 兼容性处理：旧版 LangChain CrossEncoderReranker 可能调用 predict()
                # 新版 HuggingFaceCrossEncoder 已改为 score()，这里做向后兼容
                # 注意：HuggingFaceCrossEncoder 继承自 Pydantic v2 BaseModel，
                # 直接赋值会被拒绝，必须使用 object.__setattr__ 绕过校验
                if hasattr(cross_encoder, "score") and not hasattr(cross_encoder, "predict"):
                    object.__setattr__(cross_encoder, "predict", cross_encoder.score)
                compressor = CrossEncoderReranker(
                    model=cross_encoder, top_n=chroma_config.get("top_n", 3)
                )
                self._compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor, base_retriever=self._hybrid_retriever
                )
                logger.info("重排序器初始化成功")
            except Exception as e:
                logger.error(f"重排序器初始化失败: {e}")
                self._compression_retriever = None
        else:
            self._compression_retriever = None

    def get_retriever(self):
        """返回检索器（首次调用时初始化，避免模块导入时触发 segfault）"""
        if not self._retrievers_initialized:
            self._init_retrievers()
            self._retrievers_initialized = True
        return self._compression_retriever or self._hybrid_retriever

    def load_document(self):
        """加载文档到向量数据库，加载完成后自动重建检索器"""

        def check_md5_hex(md5_for_check: str) -> bool:
            if not os.path.exists(chroma_config["md5_hex_store"]):
                open(chroma_config["md5_hex_store"], "w", encoding="utf-8").close()

            with open(chroma_config["md5_hex_store"], "r", encoding="utf-8") as f:
                for line in f.readlines():
                    if line.strip() == md5_for_check:
                        return True
                return False

        def save_md5_hex(md5_hex: str):
            with open(chroma_config["md5_hex_store"], "a", encoding="utf-8") as f:
                f.write(md5_hex + "\n")

        def get_file_documents(file_path: str):
            if file_path.endswith(".pdf"):
                return pdf_loader(file_path)
            elif file_path.endswith(".txt"):
                return txt_loader(file_path)
            elif file_path.endswith(".md"):
                return md_loader(file_path)
            elif file_path.endswith(".docx"):
                return docx_loader(file_path)
            elif file_path.endswith(".pptx"):
                return pptx_loader(file_path)
            else:
                raise ValueError(
                    f"Unsupported file type: {os.path.splitext(file_path)[1]}"
                )

        files: list[str] = filelist_with_allow_type(
            path=chroma_config["document_directory"],
            allow_type=chroma_config["allow_type"],
        )

        new_docs_loaded = False

        for file in files:
            md5_hex = get_file_md5_hex(file)
            if not md5_hex:
                continue

            if check_md5_hex(md5_hex):
                logger.info(f"文件{file}已加载过，跳过")
                continue

            try:
                documents = get_file_documents(file)
                if not documents:
                    logger.error(f"文件{file}为空")
                    continue

                spliter_documents = self.splitter.split_documents(documents)
                if not spliter_documents:
                    logger.error(f"文件{file}分割后为空")
                    continue

                self.vector_store.add_documents(spliter_documents)
                self.all_documents.extend(spliter_documents)
                new_docs_loaded = True

                logger.info(f"文件{file}加载成功")
                save_md5_hex(md5_hex)
            except Exception as e:
                logger.error(f"文件{file}加载失败: {str(e)}")

        # 如果有新文档加载，重置初始化标记，下次 get_retriever() 时重建
        if new_docs_loaded:
            self._retrievers_initialized = False
            logger.info("有新文档加载，检索器将在下次查询时自动重建")

    def add_documents(self, documents: list[Document]):
        """手动添加文档（供外部调用），添加后自动重建检索器"""
        if not documents:
            return

        split_docs = self.splitter.split_documents(documents)
        if not split_docs:
            return

        self.vector_store.add_documents(split_docs)
        self.all_documents.extend(split_docs)
        self._retrievers_initialized = False
        logger.info(f"手动添加了 {len(split_docs)} 条文档，检索器将在下次查询时自动重建")

    def delete_documents_by_source(self, source: str):
        """根据来源路径删除文档（支持增量更新）"""
        try:
            # 通过 Chroma 底层 API 按 metadata 过滤删除
            self.vector_store._collection.delete(where={"source": source})
            # 同步从内存列表中移除
            self.all_documents = [
                d for d in self.all_documents if d.metadata.get("source") != source
            ]
            self._retrievers_initialized = False
            logger.info(f"已删除来源为 {source} 的文档，检索器将在下次查询时自动重建")
        except Exception as e:
            logger.error(f"删除文档失败: {e}")

    def get_document_count(self) -> int:
        """获取当前向量库中的文档总数"""
        return len(self.all_documents)


if __name__ == "__main__":
    vector_store = VectorStore()
    print(f"当前文档数: {vector_store.get_document_count()}")
    vector_store.load_document()
    print(f"加载后文档数: {vector_store.get_document_count()}")
