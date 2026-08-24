from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from agent_material.rag_engine.indexing.structured_artifacts import ArtifactStore
from agent_material.rag_engine.indexing.visual_vector_index import (
    DashScopeMultimodalEmbeddings,
    VisualVectorIndex,
    build_visual_index_documents,
)


def _figure() -> Document:
    return Document(
        page_content="[图表]\nThermal conductivity versus temperature",
        metadata={
            "content_type": "figure",
            "artifact_id": "fig-3",
            "document_id": "doc-1",
            "version_id": "v-1",
            "caption": "Thermal conductivity along three directions",
            "visual_summary": "Three curves decrease with temperature.",
            "visual_asset_uri": "rag-asset://doc-1/v-1/main.png",
            "visual_asset_hash": "main-hash",
            "page_start": 4,
            "bbox": [10, 20, 900, 700],
            "panel_assets": [
                {
                    "label": "b",
                    "asset_uri": "rag-asset://doc-1/v-1/panel-b.png",
                    "asset_hash": "panel-hash",
                    "bbox_pixels": [500, 0, 1000, 800],
                }
            ],
        },
    )


def test_build_visual_documents_deduplicates_child_chunks_and_keeps_panel_identity():
    first = _figure()
    duplicate = Document(page_content="duplicate child", metadata=dict(first.metadata))

    docs = build_visual_index_documents([first, duplicate])

    assert [doc.metadata["visual_entity_id"] for doc in docs] == [
        "fig-3:main",
        "fig-3:panel:b",
    ]
    assert docs[1].metadata["panel_id"] == "b"
    assert docs[1].metadata["evidence_bbox_coordinate_space"] == "image_pixels"
    assert all("D:\\" not in doc.metadata["visual_asset_uri"] for doc in docs)


class _FakeEmbeddings:
    model = "fake-visual"
    dimension = 3

    def embed_image_uris(self, uris):
        assert uris == [
            "rag-asset://doc-1/v-1/main.png",
            "rag-asset://doc-1/v-1/panel-b.png",
        ]
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    def embed_query(self, query):
        assert query == "图3(b)的热导率趋势"
        return [0.0, 1.0, 0.0]


class _FakeStore:
    def __init__(self):
        self.added = None

    def add_embeddings(self, **kwargs):
        self.added = kwargs

    def similarity_search_with_score_by_vector(self, vector, *, k, filter):
        assert vector == [0.0, 1.0, 0.0]
        assert filter == {"is_active": True}
        panel = build_visual_index_documents([_figure()])[1]
        panel.metadata["is_active"] = True
        return [(panel, 0.1)]


def test_visual_vector_index_inserts_main_and_panel_with_deterministic_metadata():
    store = _FakeStore()
    index = VisualVectorIndex(
        {"collection_suffix": "_visual", "score_threshold": 0.25},
        embeddings=_FakeEmbeddings(),
        store=store,
    )

    assert index.insert_versioned([_figure()]) == 2
    assert store.added is not None
    assert store.added["embeddings"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert store.added["metadatas"][0]["visual_embedding_model"] == "fake-visual"
    assert store.added["metadatas"][1]["panel_id"] == "b"
    assert len(set(store.added["ids"])) == 2


def test_visual_search_uses_active_filter_and_returns_channel_score():
    index = VisualVectorIndex(
        {"collection_suffix": "_visual", "score_threshold": 0.25},
        embeddings=_FakeEmbeddings(),
        store=_FakeStore(),
    )

    docs = index.search("图3(b)的热导率趋势", k=3)

    assert len(docs) == 1
    assert docs[0].metadata["retrieval_channel"] == "visual"
    assert docs[0].metadata["visual_similarity"] == pytest.approx(0.95)
    assert docs[0].metadata["panel_id"] == "b"


def test_dashscope_image_payload_uses_data_uri_not_host_path(tmp_path, monkeypatch):
    store = ArtifactStore(str(tmp_path))
    uri, _ = store.save_bytes(b"png-bytes", "figure.png", "doc", "v1")
    embedder = DashScopeMultimodalEmbeddings(
        {"dimension": 3, "max_retries": 0, "max_image_bytes": 1024}
    )
    embedder._store = store
    captured = {}

    def fake_request(contents):
        captured["contents"] = contents
        return [[1.0, 0.0, 0.0]]

    monkeypatch.setattr(embedder, "_request", fake_request)

    assert embedder.embed_image_uris([uri]) == [[1.0, 0.0, 0.0]]
    image = captured["contents"][0]["image"]
    assert image.startswith("data:image/png;base64,")
    assert str(Path(tmp_path)) not in image
