"""MinerU 图表结构化索引辅助模块。

P0 只实现对象级检索：把 MinerU ``content_list`` 中的 figure/table 转换为
独立 LangChain Document，并把图片保存到内容寻址的本地资源目录。向量索引仍然
使用文本表示；图片向量与视觉重排不在本模块职责内。
"""

from __future__ import annotations

import hashlib
import html
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from langchain_core.documents import Document

from agent_material.utils.config_handler import vector_store_config
from agent_material.utils.path_tools import get_abs_path


ARTIFACT_CONTENT_TYPES = frozenset({"figure", "table"})
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_FIGURE_REF_RE = re.compile(
    r"(?:\bfig(?:ure)?\.?\s*\d+[a-z]?\b|图\s*\d+)", re.IGNORECASE
)
_PANEL_LABEL_RE = re.compile(r"^\(?[a-z]\)?$", re.IGNORECASE)


def structured_index_config() -> dict:
    return dict(vector_store_config.get("structured_index", {}) or {})


def structured_index_enabled() -> bool:
    return bool(structured_index_config().get("enabled", False))


def is_artifact_document(doc: Document) -> bool:
    return str(doc.metadata.get("content_type", "")).lower() in ARTIFACT_CONTENT_TYPES


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "\n".join(part for item in value if (part := _as_text(item)))
    return str(value).strip()


def _plain_text(value: Any) -> str:
    text = html.unescape(_TAG_RE.sub(" ", _as_text(value)))
    return _SPACE_RE.sub(" ", text).strip()


def _item_page(item: dict, page_offset: int) -> tuple[int | None, bool]:
    raw = item.get("page_idx", item.get("page_index", item.get("page")))
    try:
        # MinerU content_list 的 page_idx 为 0-based。
        return page_offset + int(raw) + 1, False
    except (TypeError, ValueError):
        return None, True


def _item_type(item: dict) -> str | None:
    raw = str(item.get("type", item.get("content_type", ""))).lower()
    if raw in {"image", "figure", "chart"}:
        return "figure"
    if raw == "table":
        return "table"
    return None


def _caption(item: dict, artifact_type: str) -> str:
    keys = (
        ("table_caption", "caption", "title")
        if artifact_type == "table"
        else (
            "chart_caption",
            "image_caption",
            "figure_caption",
            "caption",
            "title",
        )
    )
    return next((_as_text(item.get(key)) for key in keys if _as_text(item.get(key))), "")


def _footnote(item: dict, artifact_type: str) -> str:
    keys = (
        ("table_footnote", "footnote")
        if artifact_type == "table"
        else (
            "chart_footnote",
            "image_footnote",
            "figure_footnote",
            "footnote",
        )
    )
    return next((_as_text(item.get(key)) for key in keys if _as_text(item.get(key))), "")


def _structured_content(item: dict, artifact_type: str) -> str:
    if artifact_type == "table":
        keys = ("table_body", "html", "markdown", "text", "content")
    else:
        keys = ("text", "ocr_text", "content")
    return next((_as_text(item.get(key)) for key in keys if _as_text(item.get(key))), "")


def _image_path(item: dict) -> str:
    for key in ("img_path", "image_path", "path", "src"):
        value = _as_text(item.get(key))
        if value:
            return value.replace("\\", "/")
    return ""


def _nearby_context(
    items: list[dict],
    index: int,
    page_offset: int,
    limit: int,
    max_blocks: int,
) -> str:
    if limit <= 0:
        return ""
    page, _ = _item_page(items[index], page_offset)
    candidates: list[tuple[int, int, str]] = []
    for direction in (-1, 1):
        pos = index + direction
        while 0 <= pos < len(items):
            candidate = items[pos]
            candidate_page, _ = _item_page(candidate, page_offset)
            if page is not None and candidate_page != page:
                break
            raw_type = str(candidate.get("type", "")).lower()
            if raw_type in {"text", "title", "paragraph"}:
                text = _plain_text(candidate.get("text", candidate.get("content")))
                if text:
                    candidates.append((abs(pos - index), pos, text))
            pos += direction
    # 图表引用文本优先，其次按距离排序。MinerU 经常把 caption 留空，但会在
    # 相邻正文中保留 “Fig. 3(a) shows ...”，该信号比最近的普通段落更重要。
    candidates.sort(
        key=lambda value: (
            0 if _FIGURE_REF_RE.search(value[2]) else 1,
            value[0],
            value[1],
        )
    )
    return "\n".join(text for _, _, text in candidates[:max_blocks])[:limit]


def _caption_from_context(context: str) -> str:
    for line in context.splitlines():
        match = _FIGURE_REF_RE.search(line)
        if match:
            # 保留引用所在句的核心部分，避免整段正文复制进 caption。
            start = max(line.rfind(".", 0, match.start()) + 1, 0)
            ends = [pos for pos in (line.find(".", match.end()), line.find("。", match.end())) if pos >= 0]
            end = min(ends) + 1 if ends else min(len(line), start + 240)
            return line[start:end].strip()[:240]
    return ""


class ArtifactStore:
    """内容寻址的本地图片存储；索引中只保存 ``rag-asset://`` 逻辑 URI。"""

    SCHEME = "rag-asset://"

    def __init__(self, root: str | None = None):
        configured = root or structured_index_config().get("artifact_root", ".rag_assets")
        self.root = Path(configured if os.path.isabs(configured) else get_abs_path(configured))

    @staticmethod
    def _safe_suffix(name: str) -> str:
        suffix = PurePosixPath(name).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"}:
            return suffix
        return ".bin"

    def save_image(self, image: Any, document_id: str, version_id: str) -> tuple[str, str]:
        data = bytes(getattr(image, "data", b""))
        if not data:
            raise ValueError("MinerU 图片资源为空")
        return self.save_bytes(
            data,
            str(getattr(image, "name", "")),
            document_id,
            version_id,
        )

    def save_bytes(
        self,
        data: bytes,
        name: str,
        document_id: str,
        version_id: str,
    ) -> tuple[str, str]:
        """保存 MinerU 原图或预处理派生图，并返回逻辑 URI 与内容哈希。"""
        if not data:
            raise ValueError("图表资源为空")
        digest = hashlib.sha256(data).hexdigest()
        filename = f"{digest[:24]}{self._safe_suffix(name)}"
        relative = PurePosixPath(document_id) / version_id / filename
        target = self.root.joinpath(*relative.parts).resolve()
        root = self.root.resolve()
        if root != target and root not in target.parents:
            raise ValueError("非法图表资源路径")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
        return f"{self.SCHEME}{relative.as_posix()}", digest

    def resolve(self, asset_uri: str) -> Path:
        if not asset_uri.startswith(self.SCHEME):
            raise ValueError("不支持的图表资源 URI")
        relative = PurePosixPath(asset_uri[len(self.SCHEME):])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("非法图表资源 URI")
        target = self.root.joinpath(*relative.parts).resolve()
        root = self.root.resolve()
        if root != target and root not in target.parents:
            raise ValueError("图表资源 URI 越界")
        return target


def _image_lookup(images: Iterable[Any]) -> dict[str, Any]:
    lookup: dict[str, Any] = {}
    for image in images:
        for raw in (getattr(image, "path", ""), getattr(image, "name", "")):
            key = str(raw).replace("\\", "/")
            if key:
                lookup[key] = image
                lookup[PurePosixPath(key).name] = image
    return lookup


def build_artifact_documents(
    result: Any,
    *,
    source: str,
    version_context: dict | None,
    page_offset: int = 0,
    ordinal_offset: int = 0,
) -> list[Document]:
    """把 MinerU ExtractResult 转换为一图/一表一个 Document。"""
    items = list(getattr(result, "content_list", None) or [])
    if not items:
        return []

    cfg = structured_index_config()
    include_figures = bool(cfg.get("include_figures", True))
    include_tables = bool(cfg.get("include_tables", True))
    context_limit = int(cfg.get("nearby_context_chars", 500))
    context_blocks = max(1, int(cfg.get("nearby_context_blocks", 3)))
    require_figure_signal = bool(cfg.get("require_semantic_figure_signal", True))
    max_table_chars = max(256, int(cfg.get("max_table_chars_per_record", 12000)))
    image_by_path = _image_lookup(getattr(result, "images", None) or [])
    store = ArtifactStore()
    document_id = str((version_context or {}).get("document_id", "unversioned"))
    version_id = str((version_context or {}).get("version_id", "unversioned"))

    artifacts: list[Document] = []
    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        raw_type = str(item.get("type", item.get("content_type", ""))).lower()
        artifact_type = _item_type(item)
        if artifact_type is None:
            continue
        if artifact_type == "figure" and not include_figures:
            continue
        if artifact_type == "table" and not include_tables:
            continue

        ordinal = ordinal_offset + len(artifacts)
        page, page_unknown = _item_page(item, page_offset)
        caption = _caption(item, artifact_type)
        footnote = _footnote(item, artifact_type)
        structured = _structured_content(item, artifact_type)
        nearby = _nearby_context(
            items,
            item_index,
            page_offset,
            context_limit,
            context_blocks,
        )
        if artifact_type == "figure":
            derived_caption = _caption_from_context(nearby)
            if not caption or _PANEL_LABEL_RE.fullmatch(caption):
                caption = " ".join(
                    part for part in (derived_caption, caption) if part
                ).strip()
            has_semantic_signal = bool(
                raw_type in {"chart", "figure"}
                or caption
                or footnote
                or structured
                or _FIGURE_REF_RE.search(nearby)
            )
            if require_figure_signal and not has_semantic_signal:
                continue
        image_path = _image_path(item)
        image = image_by_path.get(image_path) or image_by_path.get(PurePosixPath(image_path).name)
        asset_uri = None
        asset_hash = None
        if image is not None and version_context:
            asset_uri, asset_hash = store.save_image(image, document_id, version_id)

        label = "表格" if artifact_type == "table" else "图表"
        search_parts = [f"[{label}]", caption, _plain_text(structured), footnote, nearby]
        searchable_text = "\n".join(part for part in search_parts if part).strip()
        if len(searchable_text) <= len(f"[{label}]"):
            searchable_text = f"[{label}] 文档第{page or '未知'}页，第{ordinal + 1}个{label}"
        searchable_text_truncated = False
        if artifact_type == "table" and len(searchable_text) > max_table_chars:
            # 数据库仍保存完整 structured_content；仅限制送入 embedding 的文本，
            # 防止超长 HTML 表格超过模型输入上限并拖垮整批离线索引。
            searchable_text = searchable_text[:max_table_chars].rstrip()
            searchable_text_truncated = True

        raw_hash = hashlib.sha256(
            (artifact_type + "\0" + caption + "\0" + structured + "\0" + (asset_hash or ""))
            .encode("utf-8")
        ).hexdigest()
        metadata = {
            "source": source,
            "file_type": "pdf",
            "loader": "mineru",
            "content_type": artifact_type,
            "mineru_content_type": raw_type,
            "artifact_ordinal": ordinal,
            "caption": caption,
            "nearby_context": nearby,
            "structured_content": structured,
            "asset_uri": asset_uri,
            "asset_hash": asset_hash,
            "artifact_content_hash": raw_hash,
            "document_id": document_id,
            "version_id": version_id,
            "searchable_text_truncated": searchable_text_truncated,
            "page_start": page,
            "page_end": page,
            "page_unknown": page_unknown,
            "bbox": item.get("bbox"),
            # MinerU content_list 的 bbox 使用 0..1000 页面归一化坐标。
            "bbox_coordinate_space": "normalised_1000",
            "mineru_image_path": image_path or None,
            "parent_backtrack_exempt": False,
        }
        artifacts.append(Document(page_content=searchable_text, metadata=metadata))
    return artifacts


def artifact_rows_from_documents(documents: list[Document]) -> list[dict]:
    """从已写入版本元数据的 artifact 文档生成数据库行，并按 artifact_id 去重。"""
    rows: dict[str, dict] = {}
    for doc in documents:
        if not is_artifact_document(doc):
            continue
        meta = doc.metadata
        artifact_id = str(meta.get("artifact_id", ""))
        if not artifact_id:
            continue
        rows.setdefault(
            artifact_id,
            {
                "artifact_id": artifact_id,
                "document_id": meta.get("document_id"),
                "version_id": meta.get("version_id"),
                "artifact_type": meta.get("content_type"),
                "ordinal": int(meta.get("artifact_ordinal", 0)),
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "bbox": meta.get("bbox"),
                "caption": meta.get("caption") or "",
                "nearby_context": meta.get("nearby_context") or "",
                "searchable_text": doc.page_content,
                "structured_content": meta.get("structured_content") or "",
                "asset_uri": meta.get("asset_uri"),
                "content_hash": meta.get("artifact_content_hash") or "",
                "parser": meta.get("loader", "mineru"),
                "visual_description": meta.get("visual_description"),
                "visual_summary": meta.get("visual_summary") or "",
                "visual_model": meta.get("visual_model"),
                "visual_status": meta.get("visual_status") or "not_requested",
                "visual_confidence": meta.get("visual_confidence"),
                "visual_prompt_version": meta.get("visual_prompt_version"),
                "visual_content_hash": meta.get("visual_content_hash"),
                "visual_error_code": meta.get("visual_error_code"),
                "visual_asset_uri": meta.get("visual_asset_uri"),
                "panel_assets": meta.get("panel_assets") or [],
                "readability_status": meta.get("readability_status", "unknown"),
                "legend_mapping_status": meta.get(
                    "legend_mapping_status", "not_applicable"
                ),
                "numeric_fact_status": meta.get(
                    "numeric_fact_status", "not_applicable"
                ),
                "verified_key_values": meta.get("verified_key_values") or [],
            },
        )
    return list(rows.values())


def validate_artifact_documents(documents: list[Document]) -> list[str]:
    """P0 图表完整性门禁；存在明确artifact时失败关闭，不激活残缺版本。"""
    cfg = structured_index_config()
    if not bool(cfg.get("enabled", False)) or not bool(
        cfg.get("fail_on_incomplete_artifact", True)
    ):
        return []
    require_figure_asset = bool(cfg.get("require_figure_asset", True))
    require_table_content = bool(cfg.get("require_table_content", True))
    problems: list[str] = []
    seen: set[str] = set()
    for doc in documents:
        if not is_artifact_document(doc):
            continue
        meta = doc.metadata
        artifact_id = str(meta.get("artifact_id", ""))
        if not artifact_id:
            problems.append("图表缺少 artifact_id")
            continue
        if artifact_id in seen:
            problems.append(f"图表 artifact_id 重复: {artifact_id}")
        seen.add(artifact_id)
        kind = meta.get("content_type")
        if kind == "figure" and require_figure_asset and not meta.get("asset_uri"):
            problems.append(f"图表资源缺失: {artifact_id}")
        if kind == "table" and require_table_content and not meta.get("structured_content"):
            problems.append(f"表格结构化内容缺失: {artifact_id}")
        if not doc.page_content.strip():
            problems.append(f"图表检索文本为空: {artifact_id}")
    return problems
