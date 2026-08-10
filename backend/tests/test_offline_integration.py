"""离线集成测试（Spec 10.2）— 真实 PGVector + 真实 DashScope Embedding。

与 test_versioned_ingest.py（全 fake 单测）不同，本模块使用：
- 真实 PGVector（config/vector_store_config.yaml 的 connection_string）
- 真实 DashScope Embedding API（text-embedding-v4）

隔离策略：
- monkeypatch vector_store_config["character_name"] = "agent_test"，
  全部数据只写入 agent_test_small / agent_test_large 两个独立 collection，
  绝不触碰生产 collection（agent_small / agent_large）；
- 队列 / 注册表 / FileTracker 使用 tmp_path 下的临时 sqlite；
- Worker 不启动线程，直接同步调用 _handle_ingest；
- 模块级 fixture 结束时用 SQL 清空并删除 agent_test_* collection 行。

无 DASHSCOPE_API_KEY 或 PG 不可达时，整个模块 skip。
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from dotenv import load_dotenv

load_dotenv()

from agent_material.utils.config_handler import vector_store_config  # noqa: E402

# ── 依赖探测（import 期执行，决定是否整体 skip）─────────────

_HAS_DASHSCOPE_KEY = bool(os.environ.get("DASHSCOPE_API_KEY"))


def _pg_reachable() -> bool:
    """试探 PGVector 连接（connection_string 转 psycopg 形式）。"""
    try:
        import psycopg

        conn_str = vector_store_config.get("connection_string", "").replace(
            "postgresql+psycopg://", "postgresql://"
        )
        if not conn_str:
            return False
        conn = psycopg.connect(conn_str, connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


_PG_OK = _pg_reachable()

pytestmark = [
    pytest.mark.skipif(
        not _HAS_DASHSCOPE_KEY, reason="缺少 DASHSCOPE_API_KEY，跳过离线集成测试"
    ),
    pytest.mark.skipif(not _PG_OK, reason="PGVector 不可达，跳过离线集成测试"),
]

from agent_material.rag_engine import versioning  # noqa: E402
from agent_material.rag_engine.db.file_tracker import FileTracker  # noqa: E402
from agent_material.rag_engine.db.task_queue import TaskQueue  # noqa: E402
from agent_material.rag_engine.db.version_registry import (  # noqa: E402
    DocumentVersionRegistry,
)
from agent_material.rag_engine.db.worker import BackgroundWorker  # noqa: E402
from agent_material.rag_engine.indexing.multi_gran_retriever import (  # noqa: E402
    BM25_AVAILABLE,
    MultiGranularityRetriever,
)
from langchain_core.documents import Document  # noqa: E402
from langchain_core.retrievers import BaseRetriever  # noqa: E402
from pydantic import ConfigDict  # noqa: E402

# ── 常量 ──────────────────────────────────────────────────

TEST_BASE_NAME = "agent_test"
TEST_SMALL = f"{TEST_BASE_NAME}_small"
TEST_LARGE = f"{TEST_BASE_NAME}_large"

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "docs"
SIC_DOC = FIXTURE_DIR / "sic_thermal.md"          # 术语 A: ZetaTherm-QX917
DFT_DOC = FIXTURE_DIR / "dft_functional.md"       # 术语: PseudoCore-KW304
PVK_DOC = FIXTURE_DIR / "perovskite_defect.md"    # 术语: DefectPair-LM552

TERM_A = "ZetaTherm-QX917"
TERM_B = "ZetaTherm-BB203"


def _psycopg_conn_str() -> str:
    return vector_store_config["connection_string"].replace(
        "postgresql+psycopg://", "postgresql://"
    )


# ── SQL 辅助 ──────────────────────────────────────────────


def _cleanup_test_collections():
    """清空 agent_test_* 两个 collection 的全部 embedding 行及 collection 行。"""
    import psycopg

    conn = psycopg.connect(_psycopg_conn_str())
    cur = conn.cursor()
    cur.execute(
        """
        DELETE FROM langchain_pg_embedding
        WHERE collection_id IN (
            SELECT uuid FROM langchain_pg_collection WHERE name = ANY(%s)
        )
        """,
        ([TEST_SMALL, TEST_LARGE],),
    )
    cur.execute(
        "DELETE FROM langchain_pg_collection WHERE name = ANY(%s)",
        ([TEST_SMALL, TEST_LARGE],),
    )
    conn.commit()
    cur.close()
    conn.close()


def _doc_row_counts(document_id: str) -> dict:
    """两个测试 collection 中指定 document_id 的行数。"""
    import psycopg

    counts = {}
    conn = psycopg.connect(_psycopg_conn_str())
    cur = conn.cursor()
    for col in (TEST_SMALL, TEST_LARGE):
        cur.execute(
            """
            SELECT COUNT(*)
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = %s AND e.cmetadata->>'document_id' = %s
            """,
            (col, document_id),
        )
        counts[col] = cur.fetchone()[0]
    cur.close()
    conn.close()
    return counts


def _version_active_flags(document_id: str, version_id: str) -> dict:
    """指定版本在两个 collection 中的 is_active 分布：{col: {flag: count}}。"""
    import psycopg

    result = {}
    conn = psycopg.connect(_psycopg_conn_str())
    cur = conn.cursor()
    for col in (TEST_SMALL, TEST_LARGE):
        cur.execute(
            """
            SELECT e.cmetadata->>'is_active', COUNT(*)
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = %s
              AND e.cmetadata->>'document_id' = %s
              AND e.cmetadata->>'version_id' = %s
            GROUP BY e.cmetadata->>'is_active'
            """,
            (col, document_id, version_id),
        )
        result[col] = {row[0]: row[1] for row in cur.fetchall()}
    cur.close()
    conn.close()
    return result


def _source_row_counts(source: str) -> dict:
    import psycopg

    counts = {}
    conn = psycopg.connect(_psycopg_conn_str())
    cur = conn.cursor()
    for col in (TEST_SMALL, TEST_LARGE):
        cur.execute(
            """
            SELECT COUNT(*)
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = %s AND e.cmetadata->>'source' = %s
            """,
            (col, source),
        )
        counts[col] = cur.fetchone()[0]
    cur.close()
    conn.close()
    return counts


def _load_active_docs(collection_name: str) -> list[Document]:
    """从测试 collection 加载 active 文档（is_active 缺失视为 active）。

    与 vector_store._load_documents_from_collection(active_only=True) 同语义。
    """
    import psycopg

    docs = []
    conn = psycopg.connect(_psycopg_conn_str())
    cur = conn.cursor()
    cur.execute(
        """
        SELECT e.document, e.cmetadata
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s
          AND (e.cmetadata->>'is_active' = 'true'
               OR e.cmetadata->>'is_active' IS NULL)
        """,
        (collection_name,),
    )
    for content, meta in cur.fetchall():
        docs.append(Document(page_content=content, metadata=meta or {}))
    cur.close()
    conn.close()
    return docs


def _doc_version_ids(file_path: Path) -> tuple[str, str]:
    """按 worker 同款逻辑计算 (document_id, version_id)。"""
    data = file_path.read_bytes()
    content_hash = versioning.compute_content_hash(data)
    document_id = versioning.compute_document_id(str(file_path), kb_id=TEST_BASE_NAME)
    return document_id, versioning.compute_version_id(content_hash)


# ── 模块级 fixture ────────────────────────────────────────


class _CountingPipeline:
    """包装真实 ProcessingPipeline，统计 process 调用次数（幂等断言用）。"""

    def __init__(self, real):
        self._real = real
        self.calls = 0

    def process(self, *args, **kwargs):
        self.calls += 1
        return self._real.process(*args, **kwargs)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """模块级集成环境：隔离 collection + 临时 sqlite + 同步 Worker。"""
    mp = pytest.MonkeyPatch()
    mp.setitem(vector_store_config, "character_name", TEST_BASE_NAME)
    flags = vector_store_config.setdefault("feature_flags", {})
    for flag in (
        "versioned_ingest",
        "parent_first_split",
        "active_version_filter",
        "atomic_bm25_refresh",
    ):
        mp.setitem(flags, flag, True)

    db_path = str(tmp_path_factory.mktemp("offline_it") / "queue.db")
    queue = TaskQueue(db_path=db_path)
    tracker = FileTracker(db_path=db_path)
    registry = DocumentVersionRegistry(db_path=db_path)

    bm25_refresh_calls: list[int] = []
    worker = BackgroundWorker(
        task_queue=queue,
        file_tracker=tracker,
        version_registry=registry,
        on_activated=lambda: bm25_refresh_calls.append(1),
    )
    # 预创建真实多粒度索引 + 计数包装 pipeline（不开线程，同步调用）
    multi_index = worker._get_multi_index()
    pipeline = _CountingPipeline(worker._get_pipeline())
    worker._pipeline = pipeline

    # 防残留：开始前先清一次（上次异常退出的残留）
    _cleanup_test_collections()

    yield SimpleNamespace(
        worker=worker,
        queue=queue,
        tracker=tracker,
        registry=registry,
        multi_index=multi_index,
        pipeline=pipeline,
        bm25_refresh_calls=bm25_refresh_calls,
    )

    # 彻底清理：删除 agent_test_* 全部数据与 collection 行
    _cleanup_test_collections()
    mp.undo()


def _ingest(env, file_path: Path, task_id: int = 1):
    """同步执行版本化入库（不经队列、不开线程）。"""
    env.worker._handle_ingest(
        SimpleNamespace(id=task_id, payload={"file_path": str(file_path)})
    )


def _active_search(store, query: str, k: int = 3, source: str | None = None):
    """只检索 active 版本的向量检索（与在线链路同一 filter 语义）。

    source 非空时追加 source 过滤，把断言限定在指定文档的 chunk 上，
    避免与同 collection 中其他测试文档的唯一术语互相干扰。
    """
    search_filter = {"is_active": True}
    if source is not None:
        search_filter["source"] = source
    return store.similarity_search_with_score(query, k=k, filter=search_filter)


# ── 用例 1：新增文档入库 ──────────────────────────────────


def test_new_document_ingest(env):
    document_id, version_id = _doc_version_ids(SIC_DOC)

    _ingest(env, SIC_DOC)

    # registry 状态 active
    row = env.registry.get(document_id, version_id)
    assert row is not None
    assert row["status"] == "active"
    assert env.registry.get_active(document_id)["version_id"] == version_id

    # validate_version 通过
    ok, stats = env.multi_index.validate_version(document_id, version_id)
    assert ok, f"validate_version 未通过: {stats}"

    # parent/child 数与 registry 记录一致
    counts = _doc_row_counts(document_id)
    assert row["parent_count"] == stats["parent_count"] == counts[TEST_LARGE] > 0
    assert row["child_count"] == stats["child_count"] == counts[TEST_SMALL] > 0

    # active 向量检索能命中唯一术语
    results = _active_search(env.multi_index.small_store, "ZetaTherm-QX917 是什么", k=3)
    assert results, "向量检索无任何结果"
    assert any(TERM_A in doc.page_content for doc, _ in results), (
        "active 检索未命中术语 ZetaTherm-QX917: "
        f"{[d.page_content[:50] for d, _ in results]}"
    )


# ── 用例 2：重复入库幂等 ──────────────────────────────────


def test_repeated_ingest_idempotent(env):
    document_id, version_id = _doc_version_ids(DFT_DOC)

    _ingest(env, DFT_DOC, task_id=2)
    counts_after_first = _doc_row_counts(document_id)
    assert counts_after_first[TEST_SMALL] > 0
    pipeline_calls_after_first = env.pipeline.calls

    # 连续再 ingest 两次：相同内容已 active → 幂等跳过
    _ingest(env, DFT_DOC, task_id=3)
    _ingest(env, DFT_DOC, task_id=4)

    # pipeline 未被重复调用（第二次起走幂等跳过分支）
    assert env.pipeline.calls == pipeline_calls_after_first

    # registry 只有一个版本且为 active
    versions = env.registry.list_versions(document_id)
    assert len(versions) == 1
    assert versions[0]["version_id"] == version_id
    assert versions[0]["status"] == "active"

    # small/large collection 行数三次后不变（确定性 uuid5 写入幂等）
    assert _doc_row_counts(document_id) == counts_after_first


# ── 用例 3：修改文档 → 版本切换 ───────────────────────────


def test_modified_document_version_switch(env, tmp_path):
    # document_id 由 source 路径生成 —— 必须修改同一 tmp 副本路径的内容
    doc = tmp_path / "sic_copy.md"
    doc.write_text(SIC_DOC.read_text(encoding="utf-8"), encoding="utf-8")

    # v1：含术语 A，ingest 并激活
    _ingest(env, doc, task_id=5)
    document_id, version_a = _doc_version_ids(doc)
    assert env.registry.get_active(document_id)["version_id"] == version_a

    # 激活后在线检索 A 命中（限定本测试文档的 source，避免撞上用例 1 中
    # 仍 active 的 sic_thermal.md——它同样含术语 A）
    source = str(doc)
    results = _active_search(
        env.multi_index.small_store, "ZetaTherm-QX917 是什么", k=5, source=source
    )
    assert any(TERM_A in d.page_content for d, _ in results)

    # v2：术语 A → 术语 B，同一 source 路径再 ingest
    content = doc.read_text(encoding="utf-8")
    assert TERM_A in content
    doc.write_text(content.replace(TERM_A, TERM_B), encoding="utf-8")
    _ingest(env, doc, task_id=6)
    _, version_b = _doc_version_ids(doc)
    assert version_b != version_a

    # registry：新版本 active、旧版本 retired
    assert env.registry.get_active(document_id)["version_id"] == version_b
    assert env.registry.get(document_id, version_a)["status"] == "retired"
    assert env.registry.get(document_id, version_b)["status"] == "active"

    # active 检索：B 命中、A 不命中（旧版本已翻 false，不参与在线检索）
    results_b = _active_search(
        env.multi_index.small_store, "ZetaTherm-BB203 是什么", k=5, source=source
    )
    assert any(TERM_B in d.page_content for d, _ in results_b)
    results_a = _active_search(
        env.multi_index.small_store, "ZetaTherm-QX917 是什么", k=5, source=source
    )
    assert not any(TERM_A in d.page_content for d, _ in results_a)

    # SQL 抽查：两个 collection 中旧版本的行 is_active 已全部翻为 false
    flags_a = _version_active_flags(document_id, version_a)
    for col in (TEST_SMALL, TEST_LARGE):
        col_flags = flags_a[col]
        assert col_flags, f"{col} 中未找到旧版本行"
        assert set(col_flags) == {"false"}, f"{col} 旧版本 is_active 未全部翻 false: {col_flags}"

    # 新版本行全部 is_active=true
    flags_b = _version_active_flags(document_id, version_b)
    for col in (TEST_SMALL, TEST_LARGE):
        assert set(flags_b[col]) == {"true"}, f"{col} 新版本 is_active 异常: {flags_b[col]}"


# ── 用例 4：删除文档 ──────────────────────────────────────


def test_delete_document(env, tmp_path):
    doc = tmp_path / "pvk_copy.md"
    doc.write_text(PVK_DOC.read_text(encoding="utf-8"), encoding="utf-8")
    source = str(doc)

    _ingest(env, doc, task_id=7)
    counts = _source_row_counts(source)
    assert counts[TEST_SMALL] > 0 and counts[TEST_LARGE] > 0

    # 删除：两个 collection 中该 source 行清空
    env.multi_index.delete_by_source(source)
    assert _source_row_counts(source) == {TEST_SMALL: 0, TEST_LARGE: 0}

    # 检索层面：active 检索不再命中该文档唯一术语
    results = _active_search(env.multi_index.small_store, "DefectPair-LM552 是什么", k=5)
    assert not any("DefectPair-LM552" in d.page_content for d, _ in results)

    # 注意：当前实现的 delete_by_source 只清理向量库，不清理 registry 中的
    # document_versions 记录（本用例不断言 registry 清理，仅断言向量库与检索层面）。
    # 若后续实现补充 registry 清理，可在此追加断言。

    # 重复删除幂等不报错
    env.multi_index.delete_by_source(source)
    assert _source_row_counts(source) == {TEST_SMALL: 0, TEST_LARGE: 0}


# ── 用例 5：BM25 原子刷新（无需重启检索新术语）────────────


class _FakeVectorRetriever(BaseRetriever):
    """BM25 刷新测试不需要真实向量链路，仅占位。"""

    model_config = ConfigDict(extra="allow")

    def _get_relevant_documents(self, query, *, run_manager=None):
        return []


@pytest.mark.skipif(not BM25_AVAILABLE, reason="BM25Retriever 不可用")
def test_bm25_refresh_without_restart(env, tmp_path):
    term_old = "Bm25Scope-AA100"
    term_new = "Bm25Scope-ZZ200"

    # v1：含 term_old，ingest 激活
    doc = tmp_path / "bm25_doc.md"
    content = SIC_DOC.read_text(encoding="utf-8").replace(TERM_A, term_old)
    doc.write_text(content, encoding="utf-8")
    _ingest(env, doc, task_id=8)

    # 构造检索器（small_store 指向测试 collection），初始刷新为当前 active 文档集
    retriever = MultiGranularityRetriever(
        small_store=env.multi_index.small_store,
        large_store=env.multi_index.large_store,
        all_documents=[],
    )
    retriever._vector_retriever = _FakeVectorRetriever()
    retriever._retrievers_initialized = True
    retriever.refresh_documents(_load_active_docs(TEST_SMALL))

    # 旧文档集检索不到新术语
    assert retriever._bm25_retriever is not None
    hits = retriever._bm25_retriever.invoke(term_new)
    assert not any(term_new in d.page_content for d in hits)

    # v2：term_old → term_new，同一 source 路径再 ingest（新版本激活）
    doc.write_text(
        doc.read_text(encoding="utf-8").replace(term_old, term_new), encoding="utf-8"
    )
    _ingest(env, doc, task_id=9)

    # 不重建对象，仅 refresh_documents（重新从 collection 加载 active 文档）
    retriever.refresh_documents(_load_active_docs(TEST_SMALL))

    # BM25 无需重启即可检索到新术语
    hits_new = retriever._bm25_retriever.invoke(term_new)
    assert hits_new, "BM25 刷新后无任何结果"
    assert term_new in hits_new[0].page_content, (
        f"BM25 刷新后 top-1 未命中 {term_new}: {hits_new[0].page_content[:80]}"
    )
    # 旧术语已随 active 文档集替换而退出 BM25 语料
    hits_old = retriever._bm25_retriever.invoke(term_old)
    assert not any(term_old in d.page_content for d in hits_old)
