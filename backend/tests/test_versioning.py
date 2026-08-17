"""确定性 ID 模块（versioning）单元测试（OFF-001）。

核心断言：
- 相同输入永远生成相同 ID（纯函数，无随机、无时间）
- 内容变化 → version_id / parent_id / chunk_id 变化，document_id 不变
- 不同 source 即使内容相同 → document_id 不同
"""

from agent_material.rag_engine.versioning import (
    compute_chunk_id,
    compute_content_hash,
    compute_document_id,
    compute_parent_id,
    compute_version_id,
)


class TestContentHash:
    def test_deterministic(self):
        data = b"hello materials"
        assert compute_content_hash(data) == compute_content_hash(data)

    def test_full_sha256_hex(self):
        # 完整内容 hash 是 64 位 hex
        h = compute_content_hash(b"abc")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_content_change_changes_hash(self):
        assert compute_content_hash(b"a") != compute_content_hash(b"b")


class TestDocumentId:
    def test_deterministic(self):
        assert compute_document_id("datas/a.pdf") == compute_document_id("datas/a.pdf")

    def test_content_irrelevant(self):
        # document_id 只依赖 source，与内容无关
        assert compute_document_id("datas/a.pdf") == compute_document_id("datas/a.pdf", kb_id="agent")

    def test_different_source_different_id(self):
        # 不同 source 即使内容相同，document_id 也必须不同
        assert compute_document_id("datas/a.pdf") != compute_document_id("datas/b.pdf")

    def test_different_kb_different_id(self):
        assert compute_document_id("datas/a.pdf", kb_id="agent") != compute_document_id(
            "datas/a.pdf", kb_id="other"
        )

    def test_path_normalized(self):
        # 路径写法差异经 normpath 规范化后 ID 一致
        assert compute_document_id("datas//a.pdf") == compute_document_id("datas/a.pdf")
        assert compute_document_id("datas/sub/../a.pdf") == compute_document_id("datas/a.pdf")

    def test_id_length(self):
        assert len(compute_document_id("datas/a.pdf")) == 32


class TestVersionId:
    def test_deterministic(self):
        h = compute_content_hash(b"content v1")
        assert compute_version_id(h) == compute_version_id(h)

    def test_content_change_changes_version_id(self):
        v1 = compute_version_id(compute_content_hash(b"content v1"))
        v2 = compute_version_id(compute_content_hash(b"content v2"))
        assert v1 != v2

    def test_id_length(self):
        assert len(compute_version_id(compute_content_hash(b"x"))) == 32


class TestParentAndChunkId:
    def _ids(self, content: bytes, source: str = "datas/a.pdf"):
        doc_id = compute_document_id(source)
        version_id = compute_version_id(compute_content_hash(content))
        parent_id = compute_parent_id(doc_id, version_id, 0, "parent text")
        chunk_id = compute_chunk_id(doc_id, version_id, parent_id, 0, "child text")
        return doc_id, version_id, parent_id, chunk_id

    def test_deterministic(self):
        assert self._ids(b"v1") == self._ids(b"v1")

    def test_content_change_propagates(self):
        doc1, ver1, par1, chk1 = self._ids(b"content v1")
        doc2, ver2, par2, chk2 = self._ids(b"content v2")
        # 内容变化：document_id 不变，其余全变
        assert doc1 == doc2
        assert ver1 != ver2
        assert par1 != par2
        assert chk1 != chk2

    def test_position_change_changes_id(self):
        doc_id = compute_document_id("datas/a.pdf")
        version_id = compute_version_id(compute_content_hash(b"v1"))
        assert compute_parent_id(doc_id, version_id, 0, "p") != compute_parent_id(
            doc_id, version_id, 1, "p"
        )
        parent_id = compute_parent_id(doc_id, version_id, 0, "p")
        assert compute_chunk_id(doc_id, version_id, parent_id, 0, "c") != compute_chunk_id(
            doc_id, version_id, parent_id, 1, "c"
        )

    def test_chunk_content_change_changes_id(self):
        doc_id = compute_document_id("datas/a.pdf")
        version_id = compute_version_id(compute_content_hash(b"v1"))
        parent_id = compute_parent_id(doc_id, version_id, 0, "p")
        assert compute_chunk_id(doc_id, version_id, parent_id, 0, "c1") != compute_chunk_id(
            doc_id, version_id, parent_id, 0, "c2"
        )

    def test_chunk_chain_includes_parent(self):
        # 同一内容挂到不同 parent 下，chunk_id 必须不同
        doc_id = compute_document_id("datas/a.pdf")
        version_id = compute_version_id(compute_content_hash(b"v1"))
        p1 = compute_parent_id(doc_id, version_id, 0, "p1")
        p2 = compute_parent_id(doc_id, version_id, 1, "p2")
        assert compute_chunk_id(doc_id, version_id, p1, 0, "c") != compute_chunk_id(
            doc_id, version_id, p2, 0, "c"
        )

    def test_different_version_chunks_not_shared(self):
        # 不同版本的 chunk 不得共用 chunk_id（文档与位置、内容都相同）
        doc_id = compute_document_id("datas/a.pdf")
        ids = []
        for content in (b"v1", b"v2"):
            ver = compute_version_id(compute_content_hash(content))
            par = compute_parent_id(doc_id, ver, 0, "same parent")
            ids.append(compute_chunk_id(doc_id, ver, par, 0, "same child"))
        assert ids[0] != ids[1]
