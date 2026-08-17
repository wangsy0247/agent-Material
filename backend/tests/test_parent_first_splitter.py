"""ParentFirstSplitter 父块优先切分与版本溯源元数据单元测试（OFF-005/OFF-006）。

核心断言：
- 父块按 Markdown 标题结构生成；超长章节切成多个 ~2000 字符父块
- 子块 ~500 字符，且每个子块属于且只属于一个 parent_id（不跨父块）
- 不存在"只有标题没有正文"的父块
- 公式/代码块占位符保护：公式不被从中间截断
- 相同输入两次切分产出相同结构（确定性）
- 无 page 信息时 page_unknown=True，不伪造页码
- version_context 写入确定性 parent_id / chunk_id 与版本溯源元数据
- feature_flags.parent_first_split=false 时管道回滚到旧步骤链
"""

import pytest
from langchain_core.documents import Document

from agent_material.rag_engine.text_spliter import (
    HybridSplitter,
    ParentFirstSplitter,
)
from agent_material.rag_engine.versioning import (
    compute_chunk_id,
    compute_parent_id,
)
from agent_material.utils.config_handler import vector_store_config

# 与 config/vector_store_config.yaml 的 splitters.parent 配置保持一致
PARENT_SIZE = vector_store_config["splitters"]["parent"]["parent_chunk_size"]  # 2000
CHILD_SIZE = vector_store_config["splitters"]["parent"]["child_chunk_size"]  # 500
# 占位符还原 + 孤立标题合并带来的长度容差
PARENT_TOL = 400
CHILD_TOL = 250


def _long_body(n_chars: int) -> str:
    """生成指定长度的无公式长正文（句子重复，便于稳定切分）。"""
    sentence = "深度势能模型通过神经网络拟合原子间相互作用势能面，兼顾精度与效率。"
    return (sentence * (n_chars // len(sentence) + 1))[:n_chars]


@pytest.fixture()
def sample_md() -> str:
    """带多级标题、超长章节、公式块与孤立标题的 Markdown。"""
    return (
        "# ParentFirst 测试文档\n\n"
        "前置摘要段落，没有任何标题归属，用于验证空标题路径章节。\n\n"
        "## 1 Introduction\n\n"
        + _long_body(300)
        + "\n\n### 1.1 Background\n\n"
        + _long_body(200)
        + "\n\n## 2 Methods\n\n"
        + _long_body(4500)  # 超长章节 → 多个父块
        + "\n\n## 3 Empty Section\n"  # 孤立标题（无正文，紧跟下一标题）
        + "## 4 Results\n\n"
        "包含公式的段落，能量表达式如下：\n\n"
        "$$E = mc^2 + \\frac{1}{2}mv^2$$\n\n"
        + _long_body(900)  # 使公式章节需要多个子块
        + "\n\n### 4.1 Discussion\n\n"
        "短小节正文。\n"
    )


@pytest.fixture()
def chunks(sample_md: str) -> list[Document]:
    splitter = ParentFirstSplitter()
    return splitter.split_documents(
        [Document(page_content=sample_md, metadata={"source": "test.md"})]
    )


def _parents(chunks: list[Document]) -> dict[str, list[Document]]:
    """按 parent_id 分组（保持出现顺序）。"""
    groups: dict[str, list[Document]] = {}
    for c in chunks:
        groups.setdefault(c.metadata["parent_id"], []).append(c)
    return groups


class TestParentStructure:
    """父块结构：标题路径 / 超长章节 / 尺寸约束。"""

    def test_parents_follow_heading_structure(self, chunks):
        parents = _parents(chunks)
        # 章节数 ≥ 5（前言 / Intro / Background / Methods / Results(合并3) / Discussion）
        assert len(parents) >= 5
        paths = {
            tuple(g[0].metadata["parent_metadata"]["heading_path"])
            for g in parents.values()
        }
        assert ("ParentFirst 测试文档", "1 Introduction") in paths
        assert ("ParentFirst 测试文档", "1 Introduction", "1.1 Background") in paths
        assert ("ParentFirst 测试文档", "2 Methods") in paths
        # 子块 heading_path 继承父块
        for c in chunks:
            assert c.metadata["heading_path"] == c.metadata["parent_metadata"]["heading_path"]

    def test_long_section_split_into_multiple_parents(self, chunks):
        methods_parents = [
            g[0].metadata["parent_doc"]
            for g in _parents(chunks).values()
            if g[0].metadata["parent_metadata"]["heading_path"]
            == ["ParentFirst 测试文档", "2 Methods"]
        ]
        assert len(methods_parents) >= 2  # 4500 字 > 2000 → 至少 2 个父块
        for p in methods_parents:
            assert len(p) <= PARENT_SIZE + PARENT_TOL

    def test_parent_size_within_limit(self, chunks):
        for g in _parents(chunks).values():
            assert len(g[0].metadata["parent_doc"]) <= PARENT_SIZE + PARENT_TOL

    def test_no_heading_only_parent(self, chunks):
        """不存在只有标题没有正文的父块。"""
        for g in _parents(chunks).values():
            parent_doc = g[0].metadata["parent_doc"]
            body = ParentFirstSplitter._RE_MD_HEADING.sub("", parent_doc).strip()
            assert body, f"父块只有标题没有正文: {parent_doc!r}"

    def test_standalone_heading_merged(self, chunks):
        """孤立标题 "## 3 Empty Section" 被合并，不单独成父块。"""
        for g in _parents(chunks).values():
            path = g[0].metadata["parent_metadata"]["heading_path"]
            assert "3 Empty Section" not in path


class TestChildChunks:
    """子块：尺寸 / 归属唯一性 / 偏移。"""

    def test_child_size_within_limit(self, chunks):
        for c in chunks:
            assert len(c.page_content) <= CHILD_SIZE + CHILD_TOL

    def test_child_belongs_to_exactly_one_parent(self, chunks):
        """每个子块只属于一个 parent_id，且偏移范围落在父块范围内。"""
        for c in chunks:
            meta = c.metadata
            assert meta["parent_id"]
            assert meta["parent_doc"]
            pmeta = meta["parent_metadata"]
            # 子块偏移在父块偏移范围内（含边界）
            assert pmeta["char_start"] <= meta["char_start"]
            assert meta["char_end"] <= pmeta["char_end"]
            assert meta["char_start"] < meta["char_end"]

    def test_parent_doc_consistent_per_parent_id(self, chunks):
        """同一 parent_id 的所有子块携带相同的 parent_doc。"""
        for pid, group in _parents(chunks).items():
            docs = {c.metadata["parent_doc"] for c in group}
            assert len(docs) == 1, f"parent_id={pid} 对应多个 parent_doc"

    def test_formula_not_truncated(self, chunks):
        """公式块完整 — 任何含 $$ 的子块 $$ 成对出现，且完整公式原样保留。"""
        formula = "$$E = mc^2 + \\frac{1}{2}mv^2$$"
        assert any(formula in c.page_content for c in chunks)
        for c in chunks:
            assert c.page_content.count("$$") % 2 == 0, f"公式被截断: {c.page_content!r}"
        # 父块同样完整
        for g in _parents(chunks).values():
            parent_doc = g[0].metadata["parent_doc"]
            assert parent_doc.count("$$") % 2 == 0

    def test_char_offsets_reference_full_document(self, sample_md, chunks):
        """char_start/char_end 相对整个输入文档拼接文本。"""
        for c in chunks:
            assert 0 <= c.metadata["char_start"] < c.metadata["char_end"] <= len(sample_md)


class TestDeterminism:
    """确定性：相同输入两次切分产出相同结构。"""

    def test_same_input_same_output(self, sample_md):
        splitter = ParentFirstSplitter()

        def _signature(docs):
            return [
                (
                    d.page_content,
                    d.metadata["parent_id"],
                    d.metadata["char_start"],
                    d.metadata["char_end"],
                    tuple(d.metadata["heading_path"]),
                )
                for d in docs
            ]

        run1 = splitter.split_documents([Document(page_content=sample_md, metadata={})])
        run2 = splitter.split_documents([Document(page_content=sample_md, metadata={})])
        assert _signature(run1) == _signature(run2)


class TestPageInfo:
    """页码：有 page 信息时填 page_start/page_end，否则 page_unknown=True。"""

    def test_page_unknown_when_no_page_metadata(self, chunks):
        for c in chunks:
            assert c.metadata.get("page_unknown") is True
            assert "page_start" not in c.metadata
            assert "page_end" not in c.metadata
            assert c.metadata["parent_metadata"].get("page_unknown") is True

    def test_page_range_from_metadata(self, sample_md):
        mid = len(sample_md) // 2
        docs = [
            Document(page_content=sample_md[:mid], metadata={"page": 1}),
            Document(page_content=sample_md[mid:], metadata={"page": 2}),
        ]
        result = ParentFirstSplitter().split_documents(docs)
        assert result
        for c in result:
            assert "page_unknown" not in c.metadata
            assert isinstance(c.metadata["page_start"], int)
            assert isinstance(c.metadata["page_end"], int)
            assert c.metadata["page_start"] <= c.metadata["page_end"]
            assert 1 <= c.metadata["page_start"] <= 2
            assert 1 <= c.metadata["page_end"] <= 2


class TestVersionContext:
    """version_context → 确定性 parent_id / chunk_id 与版本溯源元数据。"""

    @staticmethod
    def _apply(chunks, document_id="doc-1", version_id="ver-1"):
        from agent_material.rag_engine.document_pipeline import ProcessingPipeline

        # 深拷贝一份，避免原地修改影响其他断言
        copied = [
            Document(
                page_content=c.page_content,
                metadata={
                    **c.metadata,
                    "parent_metadata": dict(c.metadata["parent_metadata"]),
                },
            )
            for c in chunks
        ]
        return ProcessingPipeline._apply_version_context(
            copied, {"document_id": document_id, "version_id": version_id}
        )

    def test_version_metadata_written(self, chunks):
        result = self._apply(chunks)
        for c in result:
            meta = c.metadata
            assert meta["document_id"] == "doc-1"
            assert meta["version_id"] == "ver-1"
            assert meta["is_active"] is False
            assert meta["indexed_at"]
            assert meta["parent_id"]
            assert meta["chunk_id"]
            # parent_metadata 同步写入版本信息
            pmeta = meta["parent_metadata"]
            assert pmeta["parent_id"] == meta["parent_id"]
            assert pmeta["document_id"] == "doc-1"
            assert pmeta["version_id"] == "ver-1"

    def test_deterministic_across_reruns(self, chunks):
        run1 = [(c.metadata["parent_id"], c.metadata["chunk_id"]) for c in self._apply(chunks)]
        run2 = [(c.metadata["parent_id"], c.metadata["chunk_id"]) for c in self._apply(chunks)]
        assert run1 == run2

    def test_ids_match_pure_functions(self, chunks):
        """parent_id / chunk_id 与 versioning 纯函数计算结果一致。"""
        result = self._apply(chunks)
        parents = _parents(result)
        for position, (pid, group) in enumerate(parents.items()):
            parent_doc = group[0].metadata["parent_doc"]
            assert pid == compute_parent_id("doc-1", "ver-1", position, parent_doc)
            for child_pos, c in enumerate(group):
                assert c.metadata["chunk_id"] == compute_chunk_id(
                    "doc-1", "ver-1", pid, child_pos, c.page_content
                )

    def test_version_change_changes_ids(self, chunks):
        """version_id 变化 → parent_id / chunk_id 全变，document_id 不变。"""
        v1 = self._apply(chunks, version_id="ver-1")
        v2 = self._apply(chunks, version_id="ver-2")
        p1 = {c.metadata["parent_id"] for c in v1}
        p2 = {c.metadata["parent_id"] for c in v2}
        c1 = {c.metadata["chunk_id"] for c in v1}
        c2 = {c.metadata["chunk_id"] for c in v2}
        assert p1.isdisjoint(p2)
        assert c1.isdisjoint(c2)
        assert all(c.metadata["document_id"] == "doc-1" for c in v2)

    def test_content_tweak_changes_only_affected_parent(self, chunks):
        """内容微调（同 version_id）→ 受影响父块的 parent_id 变化，其余不变。"""
        tweaked = [
            Document(
                page_content=c.page_content,
                metadata={
                    **c.metadata,
                    "parent_metadata": dict(c.metadata["parent_metadata"]),
                },
            )
            for c in chunks
        ]
        # 微调最后一个父块的 parent_doc（等价于父块内容变化）
        last_pid = tweaked[-1].metadata["parent_id"]
        for c in tweaked:
            if c.metadata["parent_id"] == last_pid:
                c.metadata["parent_doc"] += "（修订）"

        base = self._apply(chunks)
        mod = self._apply(tweaked)
        base_map = {c.metadata["parent_id"]: c.metadata["parent_doc"] for c in base}
        for c in mod:
            old_doc = c.metadata["parent_doc"]
            if old_doc.endswith("（修订）"):
                # 内容变化的父块 → 新 parent_id 不在原集合中
                assert c.metadata["parent_id"] not in base_map
            else:
                assert c.metadata["parent_id"] in base_map


class TestFeatureFlagRollback:
    """feature_flags.parent_first_split 控制切分路径（回滚验证）。"""

    def test_flag_on_uses_parent_first(self, monkeypatch):
        from agent_material.rag_engine.document_pipeline import ProcessingPipeline

        monkeypatch.setitem(
            vector_store_config, "feature_flags", {"parent_first_split": True}
        )
        pipeline = ProcessingPipeline()
        for ext in (".pdf", ".md", ".html"):
            splitter = pipeline._get_splitter(f"doc{ext}")
            assert isinstance(splitter, ParentFirstSplitter), ext

    def test_flag_off_rolls_back_to_pipeline_chain(self, monkeypatch):
        from agent_material.rag_engine.document_pipeline import ProcessingPipeline

        monkeypatch.setitem(
            vector_store_config, "feature_flags", {"parent_first_split": False}
        )
        pipeline = ProcessingPipeline()
        splitter = pipeline._get_splitter("doc.pdf")
        assert isinstance(splitter, HybridSplitter)

    def test_flag_missing_rolls_back(self, monkeypatch):
        from agent_material.rag_engine.document_pipeline import ProcessingPipeline

        monkeypatch.delitem(vector_store_config, "feature_flags", raising=False)
        pipeline = ProcessingPipeline()
        assert isinstance(pipeline._get_splitter("doc.pdf"), HybridSplitter)

    def test_txt_unaffected_by_flag(self, monkeypatch):
        from agent_material.rag_engine.document_pipeline import ProcessingPipeline

        monkeypatch.setitem(
            vector_store_config, "feature_flags", {"parent_first_split": True}
        )
        pipeline = ProcessingPipeline()
        assert isinstance(pipeline._get_splitter("doc.txt"), HybridSplitter)
