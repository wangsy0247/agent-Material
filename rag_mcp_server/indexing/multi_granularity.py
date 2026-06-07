"""多粒度索引管理器 — 两层 PGVector collection 管理。

- agent_small: 子chunk（~500 字符），用于精确向量检索
- agent_large: 父chunk（~2000 字符），用于提供丰富上下文

检索流程：在小collection中匹配 → 通过 parent_id 映射到父chunk → 返回父chunk
"""

from langchain_core.documents import Document

from rag_mcp_server.config import vector_store_config
from rag_mcp_server.models import embedding_model
from rag_mcp_server.utils.logger import logger


from langchain_postgres.vectorstores import PGVector


class MultiGranularityIndex:
    """管理两个 PGVector collection：子chunk（检索用）+ 父chunk（上下文用）。

    子chunk 包含 metadata.parent_id，指向父chunk 的唯一标识。
    父chunk 去重存储（相同 parent_id 只存一条）。
    """

    def __init__(
        self,
        enabled: bool = True,
    ):
        self._enabled = enabled
        self._base_name = vector_store_config.get("character_name", "agent")
        mg_config = vector_store_config.get("multi_granularity", {})

        self._small_suffix = mg_config.get("small_collection_suffix", "_small")
        self._large_suffix = mg_config.get("large_collection_suffix", "_large")

        self._small_name = self._base_name + self._small_suffix
        self._large_name = self._base_name + self._large_suffix

        self._conn_str = vector_store_config.get("connection_string", "")

        # 延迟初始化
        self._small_store: PGVector | None = None
        self._large_store: PGVector | None = None

        if self._enabled:
            logger.info(
                f"[MultiIndex] 多粒度索引已启用 | "
                f"small={self._small_name}, large={self._large_name}"
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def small_name(self) -> str:
        return self._small_name

    @property
    def large_name(self) -> str:
        return self._large_name

    @property
    def base_name(self) -> str:
        return self._base_name

    # ── 向量库懒初始化 ────────────────────────────────────

    @property
    def small_store(self) -> PGVector:
        if self._small_store is None:
            self._small_store = PGVector(
                embeddings=embedding_model,
                connection=self._conn_str,
                collection_name=self._small_name,
                use_jsonb=True,
            )
            logger.info(f"[MultiIndex] small collection 初始化完成: {self._small_name}")
        return self._small_store

    @property
    def large_store(self) -> PGVector:
        if self._large_store is None:
            self._large_store = PGVector(
                embeddings=embedding_model,
                connection=self._conn_str,
                collection_name=self._large_name,
                use_jsonb=True,
            )
            logger.info(f"[MultiIndex] large collection 初始化完成: {self._large_name}")
        return self._large_store

    # ── 写入 ──────────────────────────────────────────────

    def insert(
        self,
        child_chunks: list[Document],
        source: str = "",
    ) -> list[str]:
        """写入子chunk + 父chunk 到两个 collection。

        优化：所有文本（父+子）合并为一次批量 embedding 调用，再分别写入两个 collection。

        Args:
            child_chunks: 带有 parent_id/parent_doc 元数据的子chunk
            source: 来源文件路径

        Returns:
            写入的父chunk parent_id 列表
        """
        if not child_chunks:
            return []

        if not self._enabled:
            store = self._get_base_store()
            store.add_documents(child_chunks)
            logger.info(
                f"[MultiIndex] 单粒度模式: 写入 {len(child_chunks)} chunks → {self._base_name}"
            )
            return []

        # 提取父chunk（去重）
        parent_map: dict[str, str] = {}
        for doc in child_chunks:
            pid = doc.metadata.get("parent_id", "")
            ptext = doc.metadata.get("parent_doc", "")
            if pid and ptext and pid not in parent_map:
                parent_map[pid] = ptext

        # 构建父chunk Document 列表
        parent_docs: list[Document] = []
        base_meta = child_chunks[0].metadata.copy()
        base_meta.pop("parent_doc", None)
        base_meta.pop("chunk_index", None)
        base_meta.pop("total_chunks", None)
        base_meta["chunk_type"] = "parent"
        base_meta["source"] = source or base_meta.get("source", "unknown")
        for pid, ptext in parent_map.items():
            meta = base_meta.copy()
            meta["parent_id"] = pid
            parent_docs.append(Document(page_content=ptext, metadata=meta))

        # 构建子chunk Document 列表（剥离 parent_doc）
        slim_children: list[Document] = []
        for doc in child_chunks:
            meta = doc.metadata.copy()
            meta.pop("parent_doc", None)
            meta["chunk_type"] = "child"
            meta["source"] = source or meta.get("source", "unknown")
            slim_children.append(Document(page_content=doc.page_content, metadata=meta))

        # ── 批量 embedding：父chunk + 子chunk 合并为一次 API 调用 ──
        all_docs = parent_docs + slim_children
        all_texts = [d.page_content for d in all_docs]
        n_parents = len(parent_docs)

        embeddings = embedding_model.embed_documents(all_texts)

        parent_embeddings = embeddings[:n_parents]
        child_embeddings = embeddings[n_parents:]

        # 使用 add_embeddings 避免二次 API 调用
        if parent_docs:
            self.large_store.add_embeddings(
                texts=[d.page_content for d in parent_docs],
                embeddings=parent_embeddings,
                metadatas=[d.metadata for d in parent_docs],
            )
            logger.info(
                f"[MultiIndex] 批量写入 {len(parent_docs)} 父chunks → {self._large_name}"
            )

        if slim_children:
            self.small_store.add_embeddings(
                texts=[d.page_content for d in slim_children],
                embeddings=child_embeddings,
                metadatas=[d.metadata for d in slim_children],
            )
            logger.info(
                f"[MultiIndex] 批量写入 {len(slim_children)} 子chunks → {self._small_name}"
            )

        logger.info(
            f"[MultiIndex] 批量入库完成 | 父={len(parent_docs)} 子={len(slim_children)} | "
            f"共 {len(all_texts)} 条文本 1 次 embedding 调用"
        )
        return list(parent_map.keys())

    def _get_base_store(self) -> PGVector:
        """获取基础 collection（未启用多粒度时使用）。"""
        return PGVector(
            embeddings=embedding_model,
            connection=self._conn_str,
            collection_name=self._base_name,
            use_jsonb=True,
        )

    # ── 删除 ──────────────────────────────────────────────

    def delete_by_source(self, source: str):
        """从两个 collection 中按 source 删除文档。"""
        import psycopg

        psycopg_conn_str = self._conn_str.replace(
            "postgresql+psycopg://", "postgresql://"
        )

        collections = (
            [self._small_name, self._large_name]
            if self._enabled
            else [self._base_name]
        )

        try:
            conn = psycopg.connect(psycopg_conn_str)
            cur = conn.cursor()
            total_deleted = 0

            for col_name in collections:
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
                    (col_name, source),
                )
                total_deleted += cur.rowcount

            conn.commit()
            cur.close()
            conn.close()

            logger.info(
                f"[MultiIndex] 已删除 source={source} 的 {total_deleted} 条记录"
            )
        except Exception as e:
            logger.error(f"[MultiIndex] 删除失败: {e}")

    # ── 查询统计 ──────────────────────────────────────────

    def get_document_count(self) -> dict:
        """获取各 collection 的文档数。"""
        import psycopg

        psycopg_conn_str = self._conn_str.replace(
            "postgresql+psycopg://", "postgresql://"
        )

        collections = (
            [self._small_name, self._large_name]
            if self._enabled
            else [self._base_name]
        )

        result: dict[str, int] = {}
        try:
            conn = psycopg.connect(psycopg_conn_str)
            cur = conn.cursor()
            for col_name in collections:
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s
                    """,
                    (col_name,),
                )
                result[col_name] = cur.fetchone()[0]
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"[MultiIndex] 查询统计失败: {e}")
            for col_name in collections:
                result[col_name] = -1

        result["total"] = sum(result.values())
        return result

    def get_all_sources(self) -> list[dict]:
        """获取所有 collection 中唯一的文献来源列表。"""
        import psycopg

        psycopg_conn_str = self._conn_str.replace(
            "postgresql+psycopg://", "postgresql://"
        )

        collections = (
            [self._small_name, self._large_name]
            if self._enabled
            else [self._base_name]
        )

        source_map: dict[str, int] = {}
        try:
            conn = psycopg.connect(psycopg_conn_str)
            cur = conn.cursor()
            for col_name in collections:
                cur.execute(
                    """
                    SELECT e.cmetadata->>'source' AS source, COUNT(*) AS chunk_count
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s AND e.cmetadata->>'source' IS NOT NULL
                    GROUP BY e.cmetadata->>'source'
                    """,
                    (col_name,),
                )
                for source, count in cur.fetchall():
                    source_map[source] = source_map.get(source, 0) + count
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"[MultiIndex] 获取来源失败: {e}")
            return []

        return sorted(
            [{"source": s, "chunk_count": c} for s, c in source_map.items()],
            key=lambda x: x["source"],
        )
