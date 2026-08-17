"""多粒度索引管理器 — 两层 PGVector collection 管理。

- agent_small: 子chunk（~500 字符），用于精确向量检索
- agent_large: 父chunk（~2000 字符），用于提供丰富上下文

检索流程：在小collection中匹配 → 通过 parent_id 映射到父chunk → 返回父chunk
"""

import uuid

from langchain_core.documents import Document

from agent_material.utils.config_handler import vector_store_config
from agent_material.rag_engine.models import embedding_model
from agent_material.utils.logger import logger


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
            self._ensure_vector_index()
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
            self._ensure_vector_index()
        return self._large_store

    def _ensure_vector_index(self) -> None:
        """确保 PGVector embedding 列上的向量索引已创建（幂等安全）。"""
        index_config = vector_store_config.get("vector_index")
        if not index_config:
            return

        try:
            from agent_material.rag_engine.vector_index import (
                build_psycopg_conn_str,
                create_vector_index,
            )

            if not self._conn_str:
                return

            psycopg_conn_str = build_psycopg_conn_str(self._conn_str)
            create_vector_index(psycopg_conn_str, index_config)
        except Exception as e:
            logger.warning(
                f"[MultiIndex] 向量索引创建异常，检索回退到顺序扫描: {e}"
            )

    # ── 写入 ──────────────────────────────────────────────

    def insert(
        self,
        child_chunks: list[Document],
        source: str = "",
        versioned: bool = False,
    ) -> list[str]:
        """写入子chunk + 父chunk 到两个 collection。

        优化：所有文本（父+子）合并为一次批量 embedding 调用，再分别写入两个 collection。

        Args:
            child_chunks: 带有 parent_id/parent_doc 元数据的子chunk
            source: 来源文件路径
            versioned: 版本化写入（OFF-003）：
                - 行 uuid 由 parent_id/chunk_id 确定性生成（uuid5），重复写入幂等
                - 父chunk metadata 使用子块携带的 parent_metadata
                - 缺少 chunk_id/parent_id 的子块跳过并记 warning

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

        if versioned:
            # 剥离缺少确定性 ID 的子块（OFF-003 要求确定性写入）
            valid_chunks: list[Document] = []
            for doc in child_chunks:
                if doc.metadata.get("chunk_id") and doc.metadata.get("parent_id"):
                    valid_chunks.append(doc)
                else:
                    logger.warning(
                        f"[MultiIndex] 跳过缺少 chunk_id/parent_id 的子块 | "
                        f"source={doc.metadata.get('source', source)}"
                    )
            if not valid_chunks:
                return []
            return self._insert_versioned(valid_chunks, source)

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

    def _insert_versioned(
        self,
        child_chunks: list[Document],
        source: str = "",
    ) -> list[str]:
        """版本化写入（OFF-003）：确定性行 uuid + 父子 metadata 契约。

        - 父chunk 行 uuid = uuid5(NAMESPACE_URL, "agent-material:parent:{parent_id}")
        - 子chunk 行 uuid = uuid5(NAMESPACE_URL, "agent-material:chunk:{chunk_id}")
        - 父chunk metadata 取自子块的 parent_metadata，补 chunk_type/source/parent_id
        - 子chunk 剥离 parent_doc/parent_metadata，补 chunk_type/source
        """
        # 提取父chunk（去重），metadata 取首个子块携带的 parent_metadata
        parent_map: dict[str, tuple[str, dict]] = {}
        for doc in child_chunks:
            pid = doc.metadata["parent_id"]
            ptext = doc.metadata.get("parent_doc", "")
            if pid not in parent_map:
                parent_map[pid] = (
                    ptext,
                    dict(doc.metadata.get("parent_metadata") or {}),
                )

        parent_docs: list[Document] = []
        for pid, (ptext, pmeta) in parent_map.items():
            meta = pmeta
            meta["chunk_type"] = "parent"
            meta["source"] = source or meta.get("source", "unknown")
            meta["parent_id"] = pid
            parent_docs.append(Document(page_content=ptext, metadata=meta))

        slim_children: list[Document] = []
        for doc in child_chunks:
            meta = doc.metadata.copy()
            meta.pop("parent_doc", None)
            meta.pop("parent_metadata", None)
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

        if parent_docs:
            self.large_store.add_embeddings(
                texts=[d.page_content for d in parent_docs],
                embeddings=parent_embeddings,
                metadatas=[d.metadata for d in parent_docs],
                ids=[
                    str(uuid.uuid5(uuid.NAMESPACE_URL, f"agent-material:parent:{pid}"))
                    for pid in parent_map
                ],
            )
            logger.info(
                f"[MultiIndex] 版本化写入 {len(parent_docs)} 父chunks → {self._large_name}"
            )

        if slim_children:
            self.small_store.add_embeddings(
                texts=[d.page_content for d in slim_children],
                embeddings=child_embeddings,
                metadatas=[d.metadata for d in slim_children],
                ids=[
                    str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"agent-material:chunk:{d.metadata['chunk_id']}",
                        )
                    )
                    for d in slim_children
                ],
            )
            logger.info(
                f"[MultiIndex] 版本化写入 {len(slim_children)} 子chunks → {self._small_name}"
            )

        logger.info(
            f"[MultiIndex] 版本化入库完成 | 父={len(parent_docs)} 子={len(slim_children)} | "
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

    # ── 版本化操作（OFF-002 / OFF-003）────────────────────

    def _psycopg_conn_str(self) -> str:
        return self._conn_str.replace("postgresql+psycopg://", "postgresql://")

    def delete_by_version(self, document_id: str, version_id: str):
        """按 document_id + version_id 从两个 collection 删除（staging 清理/重试幂等）。"""
        import psycopg

        collections = (
            [self._small_name, self._large_name]
            if self._enabled
            else [self._base_name]
        )

        try:
            conn = psycopg.connect(self._psycopg_conn_str())
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
                        WHERE c.name = %s
                          AND e.cmetadata->>'document_id' = %s
                          AND e.cmetadata->>'version_id' = %s
                    )
                    """,
                    (col_name, document_id, version_id),
                )
                total_deleted += cur.rowcount

            conn.commit()
            cur.close()
            conn.close()

            logger.info(
                f"[MultiIndex] 已删除 document_id={document_id} version_id={version_id} "
                f"的 {total_deleted} 条记录"
            )
        except Exception as e:
            # Spec V2 6.5：staging 清理失败必须终止任务 —— 记录后抛给调用方
            logger.error(f"[MultiIndex] 按版本删除失败: {e}")
            raise

    def validate_version(self, document_id: str, version_id: str) -> tuple[bool, dict]:
        """SQL 校验指定版本的写入完整性（OFF-003 最低校验条件）。

        校验项：父块数>0、子块数>0、子块 parent_id 无缺失、每个 parent_id
        在 large collection 存在且唯一、空文本 chunk 数为 0、chunk_id 无重复。

        Returns:
            (是否通过, 统计 dict)
        """
        import psycopg

        stats: dict = {
            "parent_count": 0,
            "child_count": 0,
            "missing_parent_id": 0,
            "missing_parent": 0,
            "duplicate_parent": 0,
            "empty_text": 0,
            "duplicate_chunk_id": 0,
        }

        conn = psycopg.connect(self._psycopg_conn_str())
        cur = conn.cursor()
        try:
            # 父块数 / 空文本父块
            cur.execute(
                """
                SELECT COUNT(*),
                       COUNT(*) FILTER (WHERE e.document IS NULL OR btrim(e.document) = '')
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s
                  AND e.cmetadata->>'document_id' = %s
                  AND e.cmetadata->>'version_id' = %s
                """,
                (self._large_name, document_id, version_id),
            )
            row = cur.fetchone()
            stats["parent_count"] = row[0]
            stats["empty_text"] += row[1]

            # 子块数 / parent_id 缺失 / 空文本子块 / chunk_id 重复
            cur.execute(
                """
                SELECT COUNT(*),
                       COUNT(*) FILTER (WHERE e.cmetadata->>'parent_id' IS NULL),
                       COUNT(*) FILTER (WHERE e.document IS NULL OR btrim(e.document) = '')
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s
                  AND e.cmetadata->>'document_id' = %s
                  AND e.cmetadata->>'version_id' = %s
                """,
                (self._small_name, document_id, version_id),
            )
            row = cur.fetchone()
            stats["child_count"] = row[0]
            stats["missing_parent_id"] = row[1]
            stats["empty_text"] += row[2]

            cur.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT e.cmetadata->>'chunk_id' AS chunk_id
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s
                      AND e.cmetadata->>'document_id' = %s
                      AND e.cmetadata->>'version_id' = %s
                    GROUP BY e.cmetadata->>'chunk_id'
                    HAVING COUNT(*) > 1
                ) dup
                """,
                (self._small_name, document_id, version_id),
            )
            stats["duplicate_chunk_id"] = cur.fetchone()[0]

            # parent_id 在 large collection 的存在性与唯一性
            cur.execute(
                """
                SELECT COUNT(*),
                       COUNT(*) FILTER (WHERE cnt = 0),
                       COUNT(*) FILTER (WHERE cnt > 1)
                FROM (
                    SELECT DISTINCT e.cmetadata->>'parent_id' AS pid
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s
                      AND e.cmetadata->>'document_id' = %s
                      AND e.cmetadata->>'version_id' = %s
                      AND e.cmetadata->>'parent_id' IS NOT NULL
                ) p
                LEFT JOIN LATERAL (
                    SELECT COUNT(*) AS cnt
                    FROM langchain_pg_embedding pe
                    JOIN langchain_pg_collection pc ON pe.collection_id = pc.uuid
                    WHERE pc.name = %s
                      AND pe.cmetadata->>'parent_id' = p.pid
                      AND pe.cmetadata->>'document_id' = %s
                      AND pe.cmetadata->>'version_id' = %s
                ) pcnt ON TRUE
                """,
                (
                    self._small_name,
                    document_id,
                    version_id,
                    self._large_name,
                    document_id,
                    version_id,
                ),
            )
            row = cur.fetchone()
            stats["missing_parent"] = row[1]
            stats["duplicate_parent"] = row[2]
        finally:
            cur.close()
            conn.close()

        ok = (
            stats["parent_count"] > 0
            and stats["child_count"] > 0
            and stats["missing_parent_id"] == 0
            and stats["missing_parent"] == 0
            and stats["duplicate_parent"] == 0
            and stats["empty_text"] == 0
            and stats["duplicate_chunk_id"] == 0
        )
        return ok, stats

    def set_active_version(
        self,
        document_id: str,
        new_version_id: str,
        old_version_id: str | None,
    ):
        """单个 psycopg 事务内完成 PG 侧激活翻牌（OFF-002）。

        对 small + large 两个 collection：
        - old_version_id 非空 → 旧版本 is_active=false
        - 新版本 is_active=true
        任一失败整体回滚并抛出（旧版本继续在线服务）。
        """
        import psycopg

        collections = (
            [self._small_name, self._large_name]
            if self._enabled
            else [self._base_name]
        )

        conn = psycopg.connect(self._psycopg_conn_str())
        cur = conn.cursor()
        try:
            for col_name in collections:
                if old_version_id:
                    cur.execute(
                        """
                        UPDATE langchain_pg_embedding e
                        SET cmetadata = e.cmetadata || '{"is_active": false}'::jsonb
                        FROM langchain_pg_collection c
                        WHERE e.collection_id = c.uuid
                          AND c.name = %s
                          AND e.cmetadata->>'document_id' = %s
                          AND e.cmetadata->>'version_id' = %s
                        """,
                        (col_name, document_id, old_version_id),
                    )
                cur.execute(
                    """
                    UPDATE langchain_pg_embedding e
                    SET cmetadata = e.cmetadata || '{"is_active": true}'::jsonb
                    FROM langchain_pg_collection c
                    WHERE e.collection_id = c.uuid
                      AND c.name = %s
                      AND e.cmetadata->>'document_id' = %s
                      AND e.cmetadata->>'version_id' = %s
                    """,
                    (col_name, document_id, new_version_id),
                )
            conn.commit()
            logger.info(
                f"[MultiIndex] PG 激活翻牌完成 | document_id={document_id} "
                f"new={new_version_id} old={old_version_id}"
            )
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

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
        active_only = vector_store_config.get("feature_flags", {}).get(
            "active_version_filter", True
        )
        active_clause = " AND e.cmetadata->>'is_active' = 'true'" if active_only else ""
        try:
            conn = psycopg.connect(psycopg_conn_str)
            cur = conn.cursor()
            for col_name in collections:
                cur.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s
                    {active_clause}
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
        active_only = vector_store_config.get("feature_flags", {}).get(
            "active_version_filter", True
        )
        active_clause = " AND e.cmetadata->>'is_active' = 'true'" if active_only else ""
        try:
            conn = psycopg.connect(psycopg_conn_str)
            cur = conn.cursor()
            for col_name in collections:
                cur.execute(
                    f"""
                    SELECT e.cmetadata->>'source' AS source, COUNT(*) AS chunk_count
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s AND e.cmetadata->>'source' IS NOT NULL
                    {active_clause}
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
