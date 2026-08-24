from types import SimpleNamespace

from langchain_core.documents import Document

from agent_material.rag_engine import mineru_loader
from agent_material.rag_engine.db.pg import schema
from agent_material.rag_engine.document_pipeline import ProcessingPipeline
from agent_material.rag_engine.fusion import doc_fusion_key, weighted_rrf
from agent_material.rag_engine.indexing import structured_artifacts as artifacts


def _enable_structured_index(monkeypatch, tmp_path):
    config = {
        "enabled": True,
        "artifact_root": str(tmp_path),
        "include_figures": True,
        "include_tables": True,
        "require_figure_asset": True,
        "require_semantic_figure_signal": True,
        "require_table_content": True,
        "fail_on_incomplete_artifact": True,
        "nearby_context_chars": 200,
        "nearby_context_blocks": 3,
        "max_table_chars_per_record": 12000,
    }
    monkeypatch.setitem(artifacts.vector_store_config, "structured_index", config)


def test_mineru_content_list_builds_figure_and_table_entities(monkeypatch, tmp_path):
    _enable_structured_index(monkeypatch, tmp_path)
    image = SimpleNamespace(
        name="figure-1.png", path="images/figure-1.png", data=b"png-image"
    )
    result = SimpleNamespace(
        content_list=[
            {"type": "text", "text": "实验结果如下。", "page_idx": 0},
            {
                "type": "image",
                "img_path": "images/figure-1.png",
                "image_caption": ["图1 模型准确率变化"],
                "bbox": [10, 20, 300, 220],
                "page_idx": 0,
            },
            {
                "type": "table",
                "table_caption": ["表1 消融实验"],
                "table_body": "<table><tr><th>方法</th><th>准确率</th></tr>"
                "<tr><td>Hybrid</td><td>0.91</td></tr></table>",
                "page_idx": 1,
            },
        ],
        images=[image],
    )

    docs = artifacts.build_artifact_documents(
        result,
        source="paper.pdf",
        version_context={"document_id": "doc1", "version_id": "v1"},
    )

    assert [doc.metadata["content_type"] for doc in docs] == ["figure", "table"]
    assert docs[0].metadata["page_start"] == 1
    assert docs[1].metadata["page_start"] == 2
    assert docs[0].metadata["asset_uri"].startswith("rag-asset://")
    assert str(tmp_path) not in docs[0].metadata["asset_uri"]
    assert artifacts.ArtifactStore(str(tmp_path)).resolve(
        docs[0].metadata["asset_uri"]
    ).read_bytes() == b"png-image"
    assert "Hybrid" in docs[1].page_content
    assert artifacts.validate_artifact_documents(docs) == [
        "图表缺少 artifact_id",
        "图表缺少 artifact_id",
    ]


def test_pipeline_assigns_deterministic_artifact_identity(monkeypatch, tmp_path):
    _enable_structured_index(monkeypatch, tmp_path)
    raw = Document(
        page_content="[图表]\n图1 检索召回率",
        metadata={
            "content_type": "figure",
            "artifact_ordinal": 0,
            "artifact_content_hash": "abc",
            "asset_uri": "rag-asset://doc/v/image.png",
            "source": "paper.pdf",
        },
    )
    prepared = ProcessingPipeline._prepare_artifact_chunks([raw])
    first = ProcessingPipeline._apply_version_context(
        prepared, {"document_id": "doc", "version_id": "v1"}
    )[0]
    second = ProcessingPipeline._apply_version_context(
        ProcessingPipeline._prepare_artifact_chunks([raw]),
        {"document_id": "doc", "version_id": "v1"},
    )[0]

    assert first.metadata["artifact_id"] == second.metadata["artifact_id"]
    assert first.metadata["parent_metadata"]["artifact_id"] == first.metadata["artifact_id"]
    assert first.metadata["parent_doc"] == raw.page_content
    assert artifacts.validate_artifact_documents([first]) == []


def test_artifact_fusion_deduplicates_multiple_representations():
    left = Document(
        page_content="图表标题表示",
        metadata={"artifact_id": "a1", "chunk_id": "c1"},
    )
    right = Document(
        page_content="图表附近文本表示",
        metadata={"artifact_id": "a1", "chunk_id": "c2"},
    )

    assert doc_fusion_key(left) == "artifact:a1"
    assert len(weighted_rrf([([left], 1.0), ([right], 1.0)])) == 1


def test_long_table_keeps_full_database_content_but_caps_embedding_text(
    monkeypatch, tmp_path
):
    _enable_structured_index(monkeypatch, tmp_path)
    monkeypatch.setitem(
        artifacts.vector_store_config["structured_index"],
        "max_table_chars_per_record",
        256,
    )
    full_table = "<table>" + "<tr><td>Hybrid 0.91</td></tr>" * 100 + "</table>"
    result = SimpleNamespace(
        content_list=[{"type": "table", "table_body": full_table, "page_idx": 0}],
        images=[],
    )

    doc = artifacts.build_artifact_documents(
        result,
        source="paper.pdf",
        version_context={"document_id": "doc", "version_id": "v1"},
    )[0]

    assert len(doc.page_content) <= 256
    assert doc.metadata["searchable_text_truncated"] is True
    assert doc.metadata["structured_content"] == full_table


def test_plain_image_noise_is_filtered_but_mineru_chart_is_indexed(
    monkeypatch, tmp_path
):
    _enable_structured_index(monkeypatch, tmp_path)
    logo = SimpleNamespace(name="logo.jpg", path="images/logo.jpg", data=b"logo")
    chart = SimpleNamespace(name="chart.jpg", path="images/chart.jpg", data=b"chart")
    result = SimpleNamespace(
        content_list=[
            {"type": "image", "img_path": "images/logo.jpg", "page_idx": 0},
            {"type": "text", "text": "RESEARCH ARTICLE", "page_idx": 0},
            {
                "type": "text",
                "text": "Figure 3(a) shows thermal conductivity at 300 K.",
                "page_idx": 1,
            },
            {
                "type": "chart",
                "chart_caption": [],
                "img_path": "images/chart.jpg",
                "page_idx": 1,
            },
        ],
        images=[logo, chart],
    )

    docs = artifacts.build_artifact_documents(
        result,
        source="paper.pdf",
        version_context={"document_id": "doc", "version_id": "v1"},
    )

    assert len(docs) == 1
    assert docs[0].metadata["mineru_content_type"] == "chart"
    assert docs[0].metadata["caption"].startswith("Figure 3(a)")
    assert docs[0].metadata["asset_uri"].startswith("rag-asset://")


def test_schema_contains_versioned_artifact_control_plane():
    ddl = "\n".join(schema._ddl_statements("test_"))

    assert "CREATE TABLE IF NOT EXISTS test_rag_artifacts" in ddl
    assert "artifact_count" in ddl
    assert "visual_description JSONB" in ddl
    assert "visual_status" in ddl
    assert "rag-asset" not in ddl


def test_extract_single_prefers_structured_mineru_result(monkeypatch, tmp_path):
    _enable_structured_index(monkeypatch, tmp_path)
    profile = mineru_loader.PdfProfile(1, 1, 0, 0, "native", False)
    image = SimpleNamespace(name="f.png", path="images/f.png", data=b"image")
    result = SimpleNamespace(
        state="done",
        error=None,
        markdown="# 实验\n\n这是包含足够有效正文字符的MinerU结构化解析结果。",
        content_list=[
            {
                "type": "image",
                "img_path": "images/f.png",
                "image_caption": ["图1 实验结果"],
                "page_idx": 0,
            }
        ],
        images=[image],
    )

    class Client:
        def extract(self, *_args, **_kwargs):
            return result

        def flash_extract(self, *_args, **_kwargs):
            raise AssertionError("结构化索引开启时不应调用 flash_extract")

    docs = mineru_loader._extract_single(
        "paper.pdf",
        Client(),
        1,
        1,
        profile=profile,
        version_context={"document_id": "doc", "version_id": "v1"},
    )

    assert len(docs) == 2
    assert docs[1].metadata["content_type"] == "figure"
    assert docs[1].metadata["asset_uri"].startswith("rag-asset://doc/v1/")
