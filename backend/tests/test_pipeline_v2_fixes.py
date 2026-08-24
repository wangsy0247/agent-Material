"""V2 在线修正的单元测试（Spec V2 10.4 / 10.5）。

- resolve_parents：父块原位替换，保持全局相关性顺序；
- build_context：首块超预算截断保留、context_truncated 标记。
"""

from langchain_core.documents import Document

from agent_material.rag_engine.request_context import RequestContext
from agent_material.rag_engine.retrieval_pipeline import RetrievalPipeline


def _child(content: str, pid: str = "") -> Document:
    meta = {"chunk_id": f"cid-{content}"}
    if pid:
        meta["parent_id"] = pid
    return Document(page_content=content, metadata=meta)


def _parent(pid: str) -> Document:
    return Document(
        page_content=f"parent-content-{pid}",
        metadata={"parent_id": pid, "source": "test.pdf"},
    )


class _FakeRetriever:
    """按 parent_id 字典返回父块；missing 中的 id 查不到。"""

    def __init__(self, parents: dict[str, Document], missing: list[str] | None = None):
        self._parents = parents
        self._missing = missing or []

    def _fetch_parents(self, parent_ids: list[str], parent_scopes=None):
        found = [self._parents[p] for p in parent_ids if p in self._parents]
        missing = [p for p in parent_ids if p not in self._parents]
        return found, missing


def _pipeline(retriever, budget: int = 8000) -> RetrievalPipeline:
    return RetrievalPipeline(
        retriever_getter=lambda: retriever,
        config={"context_budget_chars": budget},
    )


class TestResolveParentsInPlace:
    def test_no_parent_child_stays_in_original_position(self):
        # 无 parent_id 的子块排第 1，父块不得插到它前面（V2 10.4）
        children = [
            _child("no-pid-first"),
            _child("c1", "p1"),
            _child("c2", "p2"),
        ]
        ctx = RequestContext(original_query="q")
        ctx.reranked_children = children
        retriever = _FakeRetriever({"p1": _parent("p1"), "p2": _parent("p2")})

        _pipeline(retriever).resolve_parents(ctx)

        contents = [d.page_content for d in ctx.resolved_parents]
        assert contents == ["no-pid-first", "parent-content-p1", "parent-content-p2"]

    def test_duplicate_parent_keeps_first_occurrence_position(self):
        children = [
            _child("c1", "p1"),
            _child("c2", "p2"),
            _child("c1-dup", "p1"),  # 重复父块：跳过，不占位
        ]
        ctx = RequestContext(original_query="q")
        ctx.reranked_children = children
        retriever = _FakeRetriever({"p1": _parent("p1"), "p2": _parent("p2")})

        _pipeline(retriever).resolve_parents(ctx)

        contents = [d.page_content for d in ctx.resolved_parents]
        assert contents == ["parent-content-p1", "parent-content-p2"]

    def test_missing_parent_fallback_in_place(self):
        children = [
            _child("c1", "p1"),
            _child("c-missing", "p-missing"),
            _child("c2", "p2"),
        ]
        ctx = RequestContext(original_query="q")
        ctx.reranked_children = children
        retriever = _FakeRetriever({"p1": _parent("p1"), "p2": _parent("p2")})

        _pipeline(retriever).resolve_parents(ctx)

        contents = [d.page_content for d in ctx.resolved_parents]
        assert contents == ["parent-content-p1", "c-missing", "parent-content-p2"]
        fallback = ctx.resolved_parents[1]
        assert fallback.metadata["parent_fallback"] is True
        # 降级事件已记录
        assert any(e["component"] == "parent_fetch" for e in ctx.degradation_events)


class TestBuildContextBudget:
    def _ctx_with_parents(self, docs: list[Document]) -> RequestContext:
        ctx = RequestContext(original_query="q")
        ctx.reranked_children = docs
        ctx.resolved_parents = docs
        return ctx

    def test_first_block_over_budget_is_truncated_not_empty(self):
        big = Document(
            page_content="x" * 5000,
            metadata={"source": "big.pdf", "parent_id": "p1"},
        )
        ctx = self._ctx_with_parents([big])

        _pipeline(_FakeRetriever({}), budget=200).build_context(ctx)

        assert ctx.context_frozen
        assert 0 < len(ctx.final_context) <= 200
        assert ctx.context_truncated is True
        # 来源仍可追溯
        assert len(ctx.sources) == 1
        assert ctx.sources[0]["source"] == "big.pdf"

    def test_tail_blocks_dropped_marks_truncated(self):
        docs = [
            Document(page_content="a" * 100, metadata={"source": f"f{i}.pdf"})
            for i in range(5)
        ]
        ctx = self._ctx_with_parents(docs)

        _pipeline(_FakeRetriever({}), budget=250).build_context(ctx)

        assert ctx.context_truncated is True
        assert len(ctx.context_docs) < len(docs)
        assert len(ctx.final_context) <= 250

    def test_no_truncation_flag_false(self):
        docs = [Document(page_content="短内容", metadata={"source": "a.pdf"})]
        ctx = self._ctx_with_parents(docs)

        _pipeline(_FakeRetriever({}), budget=8000).build_context(ctx)

        assert ctx.context_truncated is False
        assert "短内容" in ctx.final_context

    def test_summary_contains_truncation_flag(self):
        ctx = RequestContext(original_query="q")
        ctx.context_truncated = True
        assert ctx.summary()["context_truncated"] is True
