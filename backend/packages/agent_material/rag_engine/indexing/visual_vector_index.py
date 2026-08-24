"""P2 cross-modal figure and panel vector index.

The public index stores only ``rag-asset://`` logical URIs. Local image bytes are
resolved immediately before an authorised embedding request and are sent as data
URIs; host paths never enter PostgreSQL metadata or API payload logs.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import time
import uuid
from typing import Any, Iterable

import httpx
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from agent_material.rag_engine.indexing.structured_artifacts import ArtifactStore
from agent_material.utils.config_handler import vector_store_config
from agent_material.utils.logger import get_logger


logger = get_logger("visual_vector_index")


def visual_embedding_config() -> dict[str, Any]:
    structured = vector_store_config.get("structured_index", {}) or {}
    return dict(structured.get("multimodal_embedding", {}) or {})


def visual_embedding_enabled() -> bool:
    value = visual_embedding_config().get("enabled", False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def visual_collection_name(config: dict[str, Any] | None = None) -> str:
    cfg = config or visual_embedding_config()
    base = str(vector_store_config.get("character_name", "agent"))
    return base + str(cfg.get("collection_suffix", "_visual"))


def _merged_metadata(doc: Document) -> dict[str, Any]:
    parent = dict(doc.metadata.get("parent_metadata") or {})
    parent.update(doc.metadata or {})
    return parent


def build_visual_index_documents(documents: Iterable[Document]) -> list[Document]:
    """Build one visual-vector entity for each figure and explicit panel.

    Repeated child chunks belonging to the same artifact are collapsed by a stable
    ``visual_entity_id``. Tables and figures without a logical image URI are skipped.
    """

    entities: dict[str, Document] = {}
    for source_doc in documents:
        meta = _merged_metadata(source_doc)
        if meta.get("content_type") != "figure":
            continue
        artifact_id = str(meta.get("artifact_id") or "")
        asset_uri = str(meta.get("visual_asset_uri") or meta.get("asset_uri") or "")
        if not artifact_id or not asset_uri.startswith(ArtifactStore.SCHEME):
            continue

        common = {
            "source": meta.get("source"),
            "source_filename": meta.get("source_filename"),
            "document_id": meta.get("document_id"),
            "version_id": meta.get("version_id"),
            "content_type": "figure",
            "artifact_id": artifact_id,
            "caption": meta.get("caption") or "",
            "page_start": meta.get("page_start"),
            "page_end": meta.get("page_end"),
            "page_unknown": bool(meta.get("page_unknown", False)),
            "bbox": meta.get("bbox"),
            "bbox_coordinate_space": meta.get("bbox_coordinate_space"),
            "visual_status": meta.get("visual_status", "not_requested"),
            "visual_summary": meta.get("visual_summary") or "",
            "readability_status": meta.get("readability_status", "unknown"),
            "legend_mapping_status": meta.get(
                "legend_mapping_status", "not_applicable"
            ),
            "numeric_fact_status": meta.get("numeric_fact_status", "not_applicable"),
            "parent_backtrack_exempt": True,
            "chunk_type": "visual",
            "is_active": False,
        }
        text = "\n".join(
            part
            for part in (
                "[图像向量]",
                str(common["caption"]),
                str(common["visual_summary"]),
            )
            if part
        ).strip()

        main_id = f"{artifact_id}:main"
        entities.setdefault(
            main_id,
            Document(
                page_content=text,
                metadata={
                    **common,
                    "visual_entity_id": main_id,
                    "panel_id": None,
                    "visual_asset_uri": asset_uri,
                    "visual_asset_hash": meta.get("visual_asset_hash")
                    or meta.get("asset_hash"),
                    "panel_assets": meta.get("panel_assets") or [],
                    "evidence_bbox": meta.get("bbox"),
                },
            ),
        )

        for index, panel in enumerate(meta.get("panel_assets") or []):
            if not isinstance(panel, dict):
                continue
            panel_uri = str(panel.get("asset_uri") or "")
            if not panel_uri.startswith(ArtifactStore.SCHEME):
                continue
            label = str(panel.get("label") or index + 1).strip().lower()
            panel_id = f"{artifact_id}:panel:{label}"
            panel_text = f"{text}\n[panel {label}]".strip()
            entities.setdefault(
                panel_id,
                Document(
                    page_content=panel_text,
                    metadata={
                        **common,
                        "visual_entity_id": panel_id,
                        "panel_id": label,
                        "visual_asset_uri": panel_uri,
                        "visual_asset_hash": panel.get("asset_hash"),
                        "panel_assets": [],
                        "evidence_bbox": panel.get("bbox_pixels"),
                        "evidence_bbox_coordinate_space": "image_pixels",
                    },
                ),
            )
    return list(entities.values())


class DashScopeMultimodalEmbeddings(Embeddings):
    """DashScope independent text/image embeddings in one semantic space."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or visual_embedding_config())
        self.model = str(self.config.get("model", "qwen3-vl-embedding"))
        self.dimension = int(self.config.get("dimension", 1024))
        self.endpoint = str(
            self.config.get(
                "endpoint",
                "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
                "multimodal-embedding/multimodal-embedding",
            )
        )
        self.api_key_env = str(self.config.get("api_key_env", "DASHSCOPE_API_KEY"))
        self.timeout_seconds = float(self.config.get("timeout_seconds", 45))
        self.max_retries = max(0, int(self.config.get("max_retries", 2)))
        self.batch_size = max(1, int(self.config.get("batch_size", 8)))
        self.max_image_bytes = int(self.config.get("max_image_bytes", 10 * 1024 * 1024))
        self._store = ArtifactStore()

    def _api_key(self) -> str:
        value = os.getenv(self.api_key_env, "").strip()
        if not value:
            raise RuntimeError(f"缺少多模态向量 API Key 环境变量: {self.api_key_env}")
        return value

    def _request(self, contents: list[dict[str, str]]) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": {"contents": contents},
            "parameters": {"dimension": self.dimension},
        }
        error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(
                        self.endpoint,
                        headers={
                            "Authorization": f"Bearer {self._api_key()}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                response.raise_for_status()
                body = response.json()
                rows = list((body.get("output") or {}).get("embeddings") or [])
                rows.sort(key=lambda item: int(item.get("index", 0)))
                vectors = [list(item.get("embedding") or []) for item in rows]
                if len(vectors) != len(contents) or any(
                    len(vector) != self.dimension for vector in vectors
                ):
                    raise RuntimeError(
                        "多模态向量响应数量或维度不符合配置契约"
                    )
                return vectors
            except Exception as exc:  # network/provider failures share one retry policy
                error = exc
                if attempt < self.max_retries:
                    time.sleep(min(0.25 * (2**attempt), 1.0))
        raise RuntimeError("多模态向量调用失败") from error

    def _batched(self, contents: list[dict[str, str]]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(contents), self.batch_size):
            vectors.extend(self._request(contents[start : start + self.batch_size]))
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._batched([{"text": text} for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._request([{"text": text}])[0]

    def _image_data_uri(self, asset_uri: str) -> str:
        path = self._store.resolve(asset_uri)
        data = path.read_bytes()
        if not data:
            raise ValueError("图片资源为空")
        if len(data) > self.max_image_bytes:
            raise ValueError("图片资源超过多模态向量大小限制")
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def embed_image_uris(self, asset_uris: list[str]) -> list[list[float]]:
        return self._batched(
            [{"image": self._image_data_uri(asset_uri)} for asset_uri in asset_uris]
        )


class VisualVectorIndex:
    """Version-aware PGVector collection for figure and panel image vectors."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        embeddings: DashScopeMultimodalEmbeddings | Any | None = None,
        store: Any | None = None,
    ):
        self.config = dict(config or visual_embedding_config())
        self.collection_name = visual_collection_name(self.config)
        self.embeddings = embeddings or DashScopeMultimodalEmbeddings(self.config)
        self._store = store

    @property
    def store(self):
        if self._store is None:
            from langchain_postgres.vectorstores import PGVector

            self._store = PGVector(
                embeddings=self.embeddings,
                connection=vector_store_config.get("connection_string", ""),
                collection_name=self.collection_name,
                use_jsonb=True,
            )
        return self._store

    def insert_versioned(self, documents: Iterable[Document]) -> int:
        visual_docs = build_visual_index_documents(documents)
        if not visual_docs:
            return 0
        vectors = self.embeddings.embed_image_uris(
            [str(doc.metadata["visual_asset_uri"]) for doc in visual_docs]
        )
        for doc in visual_docs:
            doc.metadata["visual_embedding_model"] = self.embeddings.model
            doc.metadata["visual_embedding_dimension"] = self.embeddings.dimension
        self.store.add_embeddings(
            texts=[doc.page_content for doc in visual_docs],
            embeddings=vectors,
            metadatas=[doc.metadata for doc in visual_docs],
            ids=[
                str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"agent-material:visual:{doc.metadata['visual_entity_id']}",
                    )
                )
                for doc in visual_docs
            ],
        )
        return len(visual_docs)

    def search(self, query: str, *, k: int | None = None) -> list[Document]:
        if not query.strip():
            return []
        limit = int(k or self.config.get("k", 8))
        threshold = float(self.config.get("score_threshold", 0.25))
        query_vector = self.embeddings.embed_query(query)
        search_filter = None
        if vector_store_config.get("feature_flags", {}).get(
            "active_version_filter", True
        ):
            search_filter = {"is_active": True}
        rows = self.store.similarity_search_with_score_by_vector(
            query_vector, k=limit, filter=search_filter
        )
        results: list[Document] = []
        for doc, distance in rows:
            similarity = 1.0 - float(distance) / 2.0
            if similarity < threshold:
                continue
            metadata = dict(doc.metadata or {})
            metadata["visual_similarity"] = similarity
            metadata["retrieval_channel"] = "visual"
            results.append(Document(page_content=doc.page_content, metadata=metadata))
        return results
