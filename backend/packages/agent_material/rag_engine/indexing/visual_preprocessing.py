"""P1 图表高分辨率重绘与确定性 panel 裁剪。

预处理只生成内容寻址的派生资源，不覆盖 MinerU 原图。索引与在线回答始终保存
``rag-asset://`` URI；PDF 本机路径仅在离线重绘进程内部使用。
"""

from __future__ import annotations

import io
from pathlib import Path, PurePosixPath
from typing import Any

from langchain_core.documents import Document

from agent_material.rag_engine.indexing.structured_artifacts import ArtifactStore
from agent_material.utils.config_handler import vector_store_config
from agent_material.utils.logger import logger


def visual_preprocessing_config() -> dict[str, Any]:
    structured = vector_store_config.get("structured_index", {}) or {}
    visual = structured.get("visual_enrichment", {}) or {}
    return dict(visual.get("image_preprocessing", {}) or {})


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def visual_preprocessing_enabled() -> bool:
    structured = vector_store_config.get("structured_index", {}) or {}
    visual = structured.get("visual_enrichment", {}) or {}
    cfg = visual_preprocessing_config()
    return _as_bool(visual.get("enabled"), False) and _as_bool(cfg.get("enabled"), True)


def _asset_identity(metadata: dict[str, Any]) -> tuple[str, str]:
    document_id = str(metadata.get("document_id") or "")
    version_id = str(metadata.get("version_id") or "")
    if document_id and version_id:
        return document_id, version_id
    asset_uri = str(metadata.get("asset_uri") or "")
    if asset_uri.startswith(ArtifactStore.SCHEME):
        parts = PurePosixPath(asset_uri[len(ArtifactStore.SCHEME) :]).parts
        if len(parts) >= 3:
            return parts[0], parts[1]
    raise ValueError("图表缺少版本身份")


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("缺少 Pillow，无法检查图表清晰度") from exc
    with Image.open(path) as image:
        return image.size


def _normalise_pdf_bbox(
    raw_bbox: Any,
    page_width: float,
    page_height: float,
    coordinate_space: str,
) -> tuple[tuple[float, float, float, float] | None, str]:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None, "full_page"
    try:
        x0, y0, x1, y1 = (float(value) for value in raw_bbox)
    except (TypeError, ValueError):
        return None, "full_page"
    if x1 <= x0 or y1 <= y0 or min(x0, y0) < 0:
        return None, "full_page"
    if coordinate_space == "pdf_points":
        if x1 > page_width * 1.05 or y1 > page_height * 1.05:
            return None, "full_page"
        return (x0, y0, min(x1, page_width), min(y1, page_height)), "pdf_points"
    if coordinate_space in {"normalised_1000", "normalized_1000"}:
        if x1 > 1000 or y1 > 1000:
            return None, "full_page"
        return (
            x0 * page_width / 1000,
            y0 * page_height / 1000,
            x1 * page_width / 1000,
            y1 * page_height / 1000,
        ), "normalised_1000"
    if x1 <= page_width * 1.05 and y1 <= page_height * 1.05:
        return (x0, y0, min(x1, page_width), min(y1, page_height)), "pdf_points"
    if x1 <= 1000 and y1 <= 1000:
        return (
            x0 * page_width / 1000,
            y0 * page_height / 1000,
            x1 * page_width / 1000,
            y1 * page_height / 1000,
        ), "normalised_1000"
    return None, "full_page"


def _render_pdf_region(
    pdf_path: Path,
    page_number: int,
    raw_bbox: Any,
    *,
    dpi: int,
    padding_points: float,
    bbox_coordinate_space: str,
) -> tuple[bytes, dict[str, Any]]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("缺少 PyMuPDF，无法从原 PDF 高分辨率重绘") from exc
    with fitz.open(pdf_path) as pdf:
        if page_number < 1 or page_number > len(pdf):
            raise ValueError("图表页码越界")
        page = pdf[page_number - 1]
        coords, bbox_mode = _normalise_pdf_bbox(
            raw_bbox,
            float(page.rect.width),
            float(page.rect.height),
            bbox_coordinate_space,
        )
        clip = fitz.Rect(coords) if coords else page.rect
        if coords and padding_points > 0:
            clip = fitz.Rect(
                max(page.rect.x0, clip.x0 - padding_points),
                max(page.rect.y0, clip.y0 - padding_points),
                min(page.rect.x1, clip.x1 + padding_points),
                min(page.rect.y1, clip.y1 + padding_points),
            )
        if clip.is_empty or clip.width < 8 or clip.height < 8:
            raise ValueError("图表 bbox 无有效面积")
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0),
            clip=clip,
            alpha=False,
        )
        return pixmap.tobytes("png"), {
            "render_dpi": dpi,
            "render_bbox_mode": bbox_mode,
            "render_clip_points": [
                round(clip.x0, 2),
                round(clip.y0, 2),
                round(clip.x1, 2),
                round(clip.y1, 2),
            ],
            "width_px": pixmap.width,
            "height_px": pixmap.height,
        }


def _runs(values: list[bool]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            result.append((start, index))
            start = None
    return result


def _projection_segments(
    image: Any,
    *,
    axis: str,
    whitespace_ratio: float,
    min_gap_ratio: float,
    min_panel_ratio: float,
) -> list[tuple[int, int, int, int]]:
    """使用长空白沟槽做确定性分栏；不依赖 OCR 或额外视觉模型。"""
    from PIL import ImageOps

    gray = ImageOps.grayscale(image)
    width, height = gray.size
    pixels = gray.load()
    if axis == "vertical":
        length, cross = width, height
        ink_ratio = [
            sum(1 for y in range(height) if pixels[x, y] < 240) / max(height, 1)
            for x in range(width)
        ]
    else:
        length, cross = height, width
        ink_ratio = [
            sum(1 for x in range(width) if pixels[x, y] < 240) / max(width, 1)
            for y in range(height)
        ]
    min_gap = max(8, int(length * min_gap_ratio))
    gaps = [
        (start, end)
        for start, end in _runs([ratio <= whitespace_ratio for ratio in ink_ratio])
        if end - start >= min_gap and start > length * 0.08 and end < length * 0.92
    ]
    if not gaps:
        return []
    cuts = [0, *[(start + end) // 2 for start, end in gaps], length]
    boxes: list[tuple[int, int, int, int]] = []
    min_panel = max(64, int(length * min_panel_ratio))
    for start, end in zip(cuts, cuts[1:]):
        if end - start < min_panel:
            continue
        box = (start, 0, end, cross) if axis == "vertical" else (0, start, cross, end)
        boxes.append(box)
    return boxes if 2 <= len(boxes) <= 4 else []


def detect_panel_boxes(image_path: Path, config: dict[str, Any]) -> list[tuple[int, int, int, int]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("缺少 Pillow，无法进行 panel 裁剪") from exc
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        kwargs = {
            "whitespace_ratio": float(config.get("whitespace_ratio", 0.008)),
            "min_gap_ratio": float(config.get("min_gap_ratio", 0.018)),
            "min_panel_ratio": float(config.get("min_panel_ratio", 0.2)),
        }
        axes = (
            ("vertical", "horizontal")
            if image.width >= image.height
            else (
                "horizontal",
                "vertical",
            )
        )
        for axis in axes:
            boxes = _projection_segments(image, axis=axis, **kwargs)
            if boxes:
                return boxes
    return []


def _save_panel_assets(
    store: ArtifactStore,
    image_path: Path,
    document_id: str,
    version_id: str,
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    if not _as_bool(config.get("panel_crops_enabled"), True):
        return []
    manual_panels = metadata.get("panel_bboxes") or []
    labels: list[str] = []
    boxes: list[tuple[int, int, int, int]] = []
    for index, panel in enumerate(manual_panels):
        if not isinstance(panel, dict):
            continue
        raw_box = panel.get("bbox_pixels")
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
            continue
        try:
            box = tuple(int(value) for value in raw_box)
        except (TypeError, ValueError):
            continue
        boxes.append(box)
        labels.append(str(panel.get("label") or chr(ord("a") + index)).lower())
    if not boxes:
        if not _as_bool(config.get("auto_panel_detection"), False):
            return []
        boxes = detect_panel_boxes(image_path, config)
        labels = [chr(ord("a") + index) for index in range(len(boxes))]
    if not boxes:
        return []
    from PIL import Image

    assets: list[dict[str, Any]] = []
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        for index, box in enumerate(boxes):
            crop = image.crop(box)
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG", optimize=True)
            uri, digest = store.save_bytes(
                buffer.getvalue(),
                f"panel-{index}.png",
                document_id,
                version_id,
            )
            assets.append(
                {
                    "label": labels[index],
                    "asset_uri": uri,
                    "asset_hash": digest,
                    "bbox_pixels": list(box),
                    "width_px": crop.width,
                    "height_px": crop.height,
                }
            )
    return assets


def preprocess_figure_document(document: Document) -> Document:
    """必要时从原 PDF 重绘，并为复合图生成可查询选择的 panel 派生图。"""
    if document.metadata.get("content_type") != "figure" or not document.metadata.get("asset_uri"):
        return document
    cfg = visual_preprocessing_config()
    if not visual_preprocessing_enabled():
        return document
    metadata = document.metadata.copy()
    store = ArtifactStore()
    original_path = store.resolve(str(metadata["asset_uri"]))
    if not original_path.is_file():
        metadata["visual_preprocess_status"] = "failed"
        metadata["visual_preprocess_error_code"] = "AssetMissing"
        return Document(page_content=document.page_content, metadata=metadata)
    try:
        width, height = _image_size(original_path)
        min_width = int(cfg.get("min_width_px", 1200))
        min_height = int(cfg.get("min_height_px", 800))
        target_uri = str(metadata["asset_uri"])
        target_hash = str(metadata.get("asset_hash") or "")
        target_path = original_path
        render_meta: dict[str, Any] = {
            "width_px": width,
            "height_px": height,
            "render_dpi": None,
            "render_bbox_mode": "mineru_asset",
        }
        status = "source_sufficient"
        if width < min_width or height < min_height:
            source = Path(str(metadata.get("source") or ""))
            page_number = int(metadata.get("page_start") or 0)
            if source.is_file() and source.suffix.lower() == ".pdf" and page_number > 0:
                data, render_meta = _render_pdf_region(
                    source,
                    page_number,
                    metadata.get("bbox"),
                    dpi=max(144, int(cfg.get("render_dpi", 300))),
                    padding_points=max(0.0, float(cfg.get("padding_points", 6))),
                    bbox_coordinate_space=str(
                        metadata.get("bbox_coordinate_space")
                        or cfg.get("bbox_coordinate_space")
                        or "normalised_1000"
                    ),
                )
                if (
                    int(render_meta["width_px"]) < min_width
                    or int(render_meta["height_px"]) < min_height
                ):
                    current_dpi = int(render_meta["render_dpi"])
                    scale = max(
                        min_width / max(int(render_meta["width_px"]), 1),
                        min_height / max(int(render_meta["height_px"]), 1),
                    )
                    max_dpi = max(current_dpi, int(cfg.get("max_render_dpi", 450)))
                    target_dpi = min(
                        max_dpi,
                        max(current_dpi, int(current_dpi * scale + 0.999)),
                    )
                    if target_dpi > current_dpi:
                        data, render_meta = _render_pdf_region(
                            source,
                            page_number,
                            metadata.get("bbox"),
                            dpi=target_dpi,
                            padding_points=max(0.0, float(cfg.get("padding_points", 6))),
                            bbox_coordinate_space=str(
                                metadata.get("bbox_coordinate_space")
                                or cfg.get("bbox_coordinate_space")
                                or "normalised_1000"
                            ),
                        )
                document_id, version_id = _asset_identity(metadata)
                target_uri, target_hash = store.save_bytes(
                    data, "pdf-render.png", document_id, version_id
                )
                target_path = store.resolve(target_uri)
                status = "pdf_rendered"
            else:
                status = "fallback_source"
        document_id, version_id = _asset_identity(metadata)
        panels = _save_panel_assets(
            store,
            target_path,
            document_id,
            version_id,
            cfg,
            metadata,
        )
        metadata.update(
            {
                "visual_original_asset_uri": metadata["asset_uri"],
                "visual_asset_uri": target_uri,
                "visual_asset_hash": target_hash,
                "visual_preprocess_status": status,
                "visual_width_px": int(render_meta["width_px"]),
                "visual_height_px": int(render_meta["height_px"]),
                "visual_render_dpi": render_meta.get("render_dpi"),
                "visual_render_bbox_mode": render_meta.get("render_bbox_mode"),
                "panel_assets": panels,
            }
        )
    except Exception as exc:
        logger.warning(
            "visual_preprocessing_failed",
            extra={"error_type": type(exc).__name__},
        )
        metadata.update(
            {
                "visual_asset_uri": metadata.get("asset_uri"),
                "visual_asset_hash": metadata.get("asset_hash"),
                "visual_preprocess_status": "failed",
                "visual_preprocess_error_code": type(exc).__name__,
                "panel_assets": [],
            }
        )
    return Document(page_content=document.page_content, metadata=metadata)


def preprocess_artifact_documents(documents: list[Document]) -> list[Document]:
    return [preprocess_figure_document(document) for document in documents]


def select_visual_asset_uri(metadata: dict[str, Any], query: str = "") -> str:
    """按查询中的 ``(a)``/``panel b`` 选择裁剪图，否则使用高清主图。"""
    import re

    matched_uri = str(metadata.get("matched_visual_asset_uri") or "")
    if matched_uri:
        return matched_uri
    match = re.search(
        r"(?:panel\s*|子图\s*|图\s*\d*\s*)[（(]?\s*([a-d])\s*[)）]?",
        query,
        flags=re.IGNORECASE,
    )
    if match:
        label = match.group(1).lower()
        for panel in metadata.get("panel_assets") or []:
            if str(panel.get("label") or "").lower() == label:
                return str(panel.get("asset_uri") or "")
    return str(metadata.get("visual_asset_uri") or metadata.get("asset_uri") or "")
