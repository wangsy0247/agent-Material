"""P1 图表视觉描述契约与离线增强辅助函数。

视觉模型只负责把图片转换成可校验的结构化描述。图片本身不直接进入
PGVector；结构化描述会追加到 P0 图表对象的检索文本中，继续复用现有
Embedding、BM25、RRF 和版本化索引链路。
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Literal, Protocol

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field

from agent_material.utils.config_handler import vector_store_config
from agent_material.utils.logger import logger


VISUAL_SCHEMA_VERSION = "visual-chart-v1"
_JSON_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE
)


class VisualAxis(BaseModel):
    """图表坐标轴语义。"""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    unit: str = ""


class VisualDescription(BaseModel):
    """视觉模型输出的稳定 JSON 契约。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["visual-chart-v1"] = VISUAL_SCHEMA_VERSION
    chart_type: str = "unknown"
    title: str = ""
    x_axis: VisualAxis = Field(default_factory=VisualAxis)
    y_axis: VisualAxis = Field(default_factory=VisualAxis)
    legend: list[str] = Field(default_factory=list)
    trends: list[str] = Field(default_factory=list)
    key_values: list[str] = Field(default_factory=list)
    visual_summary: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


def visual_enrichment_config() -> dict[str, Any]:
    structured = vector_store_config.get("structured_index", {}) or {}
    return dict(structured.get("visual_enrichment", {}) or {})


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def visual_enrichment_enabled() -> bool:
    return _as_bool(visual_enrichment_config().get("enabled"), False)


def parse_visual_description(payload: str | dict[str, Any]) -> VisualDescription:
    """解析并严格校验视觉模型输出，拒绝未声明字段。"""
    if isinstance(payload, dict):
        return VisualDescription.model_validate(payload)
    text = payload.strip()
    fenced = _JSON_FENCE_RE.match(text)
    if fenced:
        text = fenced.group("body").strip()
    return VisualDescription.model_validate_json(text)


def visual_description_text(
    description: VisualDescription, gate: dict[str, Any] | None = None
) -> str:
    """生成确定性的中文检索表示。"""
    gate = gate or {}
    trends = gate.get("safe_trends", description.trends)
    key_values = gate.get("verified_key_values", description.key_values)
    summary = gate.get("safe_visual_summary", description.visual_summary)
    parts = [
        f"图表类型：{description.chart_type}",
        f"图表标题：{description.title}" if description.title else "",
        (
            f"横轴：{description.x_axis.name} {description.x_axis.unit}".strip()
            if description.x_axis.name or description.x_axis.unit
            else ""
        ),
        (
            f"纵轴：{description.y_axis.name} {description.y_axis.unit}".strip()
            if description.y_axis.name or description.y_axis.unit
            else ""
        ),
        f"图例：{'；'.join(description.legend)}" if description.legend else "",
        f"趋势：{'；'.join(trends)}" if trends else "",
        f"已核验关键读数：{'；'.join(key_values)}" if key_values else "",
        f"视觉摘要：{summary}" if summary else "",
    ]
    return "\n".join(part for part in parts if part)


def apply_visual_description(
    document: Document,
    description: VisualDescription,
    *,
    model: str,
    prompt_version: str = VISUAL_SCHEMA_VERSION,
) -> Document:
    """把视觉描述幂等合并进图表对象及其检索文本。"""
    metadata = document.metadata.copy()
    base_text = str(metadata.get("pre_visual_searchable_text") or document.page_content)
    from agent_material.rag_engine.indexing.visual_fact_gate import (
        evaluate_visual_fact_gate,
    )

    gate = evaluate_visual_fact_gate(description, metadata)
    visual_text = visual_description_text(description, gate)
    metadata.update(
        {
            "pre_visual_searchable_text": base_text,
            "visual_description": description.model_dump(mode="json"),
            "visual_summary": description.visual_summary,
            "visual_model": model,
            "visual_prompt_version": prompt_version,
            "visual_status": "completed",
            "visual_confidence": description.confidence,
            "readability_status": gate["readability_status"],
            "legend_mapping_status": gate["legend_mapping_status"],
            "numeric_fact_status": gate["numeric_fact_status"],
            "verified_key_values": gate["verified_key_values"],
            "fact_gate_reasons": gate["fact_gate_reasons"],
        }
    )
    enriched_text = (
        f"{base_text}\n[视觉解析]\n{visual_text}" if visual_text else base_text
    )
    return Document(page_content=enriched_text, metadata=metadata)


def visual_description_json(description: VisualDescription) -> str:
    return json.dumps(
        description.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class VisualEnrichmentError(RuntimeError):
    """视觉增强无法按配置完成。"""


class VisualDescriptionClient(Protocol):
    model: str

    def describe(self, image_path: Path, context: dict[str, Any]) -> VisualDescription:
        ...


def _visual_prompt(context: dict[str, Any]) -> str:
    schema = json.dumps(
        VisualDescription.model_json_schema(), ensure_ascii=False, separators=(",", ":")
    )
    caption = str(context.get("caption") or "")[:800]
    nearby = str(context.get("nearby_context") or "")[:1200]
    return (
        "分析这张材料科学论文图表，只报告图中直接可见或可可靠读取的信息。"
        "无法确认的数值不要猜测，写入 warnings。必须只返回一个符合给定 JSON Schema "
        f"的 JSON 对象，不要使用 Markdown。\nJSON Schema: {schema}\n"
        f"论文图注：{caption}\n邻近正文：{nearby}"
    )


class DashScopeVisualDescriptionClient:
    """通过 DashScope OpenAI 兼容接口生成离线图表描述。"""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or visual_enrichment_config()
        self.model = str(cfg.get("model") or "qwen3-vl-plus")
        self.prompt_version = str(cfg.get("prompt_version") or VISUAL_SCHEMA_VERSION)
        self.max_image_bytes = int(cfg.get("max_image_bytes", 10 * 1024 * 1024))
        api_key_env = str(cfg.get("api_key_env") or "DASHSCOPE_API_KEY")
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise VisualEnrichmentError(f"视觉增强缺少环境变量: {api_key_env}")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise VisualEnrichmentError("缺少 openai 客户端依赖") from exc
        self._client = OpenAI(
            api_key=api_key,
            base_url=str(cfg.get("base_url") or "").rstrip("/"),
            timeout=float(cfg.get("timeout_seconds", 45)),
            max_retries=int(cfg.get("max_retries", 2)),
        )

    def describe(self, image_path: Path, context: dict[str, Any]) -> VisualDescription:
        size = image_path.stat().st_size
        if size <= 0:
            raise VisualEnrichmentError("图表资源为空")
        if size > self.max_image_bytes:
            raise VisualEnrichmentError("图表资源超过视觉模型大小限制")
        mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        if mime_type not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
            raise VisualEnrichmentError(f"视觉模型不支持的图片类型: {mime_type}")
        data_url = (
            f"data:{mime_type};base64,"
            f"{base64.b64encode(image_path.read_bytes()).decode('ascii')}"
        )
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "你是严谨的科研图表解析器，禁止补造图中不存在的信息。",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": _visual_prompt(context)},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": False},
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise VisualEnrichmentError("视觉模型返回空内容")
        return parse_visual_description(content)


class VisualDescriptionCache:
    """按图片内容、模型和 prompt 版本缓存已校验的视觉描述。"""

    def __init__(self, root: Path):
        self.root = root.resolve() / ".visual_cache"

    @staticmethod
    def _key(asset_hash: str, model: str, prompt_version: str) -> str:
        raw = f"{asset_hash}\0{model}\0{prompt_version}".encode()
        return hashlib.sha256(raw).hexdigest()

    def load(
        self, asset_hash: str, model: str, prompt_version: str
    ) -> VisualDescription | None:
        target = self.root / f"{self._key(asset_hash, model, prompt_version)}.json"
        if not target.is_file():
            return None
        try:
            return parse_visual_description(target.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("visual_enrichment_cache_invalid", extra={"cache_key": target.stem})
            return None

    def save(
        self,
        asset_hash: str,
        model: str,
        prompt_version: str,
        description: VisualDescription,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{self._key(asset_hash, model, prompt_version)}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(visual_description_json(description), encoding="utf-8")
        os.replace(temporary, target)


def _failed_document(
    document: Document, *, model: str, prompt_version: str, error: Exception
) -> Document:
    metadata = document.metadata.copy()
    metadata.update(
        {
            "visual_status": "failed",
            "visual_model": model,
            "visual_prompt_version": prompt_version,
            "visual_error_code": type(error).__name__,
        }
    )
    return Document(page_content=document.page_content, metadata=metadata)


def enrich_artifact_documents(
    documents: list[Document],
    *,
    client: VisualDescriptionClient | None = None,
) -> list[Document]:
    """离线增强 figure 文档；失败策略由配置决定。"""
    if not visual_enrichment_enabled() or not documents:
        return documents

    from agent_material.rag_engine.indexing.structured_artifacts import ArtifactStore

    cfg = visual_enrichment_config()
    failure_policy = str(cfg.get("failure_policy") or "fallback_text")
    max_images = max(0, int(cfg.get("max_images_per_document", 32)))
    prompt_version = str(cfg.get("prompt_version") or VISUAL_SCHEMA_VERSION)
    store = ArtifactStore()
    visual_client = client
    model = str(getattr(client, "model", "") or cfg.get("model") or "qwen3-vl-plus")
    cache = VisualDescriptionCache(store.root)
    enriched: list[Document] = []
    attempted = 0

    for document in documents:
        metadata = document.metadata
        if metadata.get("content_type") != "figure" or not metadata.get("asset_uri"):
            enriched.append(document)
            continue
        if attempted >= max_images:
            skipped = document.metadata.copy()
            skipped["visual_status"] = "skipped_limit"
            enriched.append(Document(page_content=document.page_content, metadata=skipped))
            continue
        attempted += 1
        try:
            asset_hash = str(
                metadata.get("visual_asset_hash") or metadata.get("asset_hash") or ""
            )
            if not asset_hash:
                raise VisualEnrichmentError("图表缺少 asset_hash")
            description = cache.load(asset_hash, model, prompt_version)
            if description is None:
                if visual_client is None:
                    visual_client = DashScopeVisualDescriptionClient(cfg)
                    model = visual_client.model
                image_path = store.resolve(
                    str(metadata.get("visual_asset_uri") or metadata["asset_uri"])
                )
                description = visual_client.describe(image_path, metadata)
                cache.save(asset_hash, model, prompt_version, description)
            result = apply_visual_description(
                document,
                description,
                model=model,
                prompt_version=prompt_version,
            )
            result.metadata["visual_content_hash"] = hashlib.sha256(
                visual_description_json(description).encode()
            ).hexdigest()
            enriched.append(result)
        except Exception as exc:
            logger.warning(
                "visual_enrichment_failed",
                extra={
                    "artifact_ordinal": metadata.get("artifact_ordinal"),
                    "error_type": type(exc).__name__,
                },
            )
            if failure_policy == "fail_version":
                raise VisualEnrichmentError("视觉增强失败，关闭当前索引版本") from exc
            enriched.append(
                _failed_document(
                    document,
                    model=model,
                    prompt_version=prompt_version,
                    error=exc,
                )
            )
    return enriched
