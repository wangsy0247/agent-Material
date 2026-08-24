"""检索结果融合 — weighted RRF（单查询与跨查询）。

设计约束（对应 Spec ONL-005 / ONL-006 / ONL-010）：
- RRF 分数只依赖排名与权重：score += weight / (rrf_k + rank)，
  不直接混加不同量纲的原始分数（向量距离 / BM25 分）。
- 同一 chunk 在多个分支中只保留一条记录，并累加排名贡献。
- 分数相同时使用确定性次序：原始 query 分支内的排名、再按 chunk_id。
- 去重键优先使用稳定 chunk_id，缺失时回退到内容的 sha256
  （禁止 Python 进程随机 hash 作为主去重键）。
"""

from __future__ import annotations

import hashlib

from langchain_core.documents import Document

DEFAULT_RRF_K = 60


def doc_fusion_key(doc: Document) -> str:
    """返回稳定去重键：图表按 artifact_id，其余按 chunk_id。"""
    artifact_id = doc.metadata.get("artifact_id", "")
    if artifact_id:
        return f"artifact:{artifact_id}"
    chunk_id = doc.metadata.get("chunk_id", "")
    if chunk_id:
        return str(chunk_id)
    digest = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
    return f"content:{digest[:32]}"


def weighted_rrf(
    rankings: list[tuple[list[Document], float]],
    rrf_k: int = DEFAULT_RRF_K,
) -> list[Document]:
    """对多路有序结果做 weighted RRF 融合。

    Args:
        rankings: [(有序文档列表, 分支权重), ...]。第一个分支视为原始 query
                  分支，其排名用于分数相同情况下的确定性决胜。
        rrf_k: RRF 平滑常数。

    Returns:
        按融合分数降序排列的去重文档列表；每条文档的 metadata 写入
        ``rrf_score`` 字段。
    """
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    original_rank: dict[str, int] = {}

    for branch_index, (branch_docs, weight) in enumerate(rankings):
        seen_in_branch: set[str] = set()
        for rank, doc in enumerate(branch_docs, start=1):
            key = doc_fusion_key(doc)
            if key not in docs:
                docs[key] = doc
            else:
                _merge_retrieval_metadata(docs[key], doc)
            # 同一通道中的 main/panel 或重复 chunk 只贡献一次，避免单路刷分。
            if key in seen_in_branch:
                continue
            seen_in_branch.add(key)
            contribution = weight / (rrf_k + rank)
            scores[key] = scores.get(key, 0.0) + contribution
            channel = str(doc.metadata.get("retrieval_channel") or "")
            if channel:
                channels = list(docs[key].metadata.get("retrieval_channels") or [])
                if channel not in channels:
                    channels.append(channel)
                docs[key].metadata["retrieval_channels"] = channels
                contributions = dict(
                    docs[key].metadata.get("rrf_channel_contributions") or {}
                )
                contributions[channel] = contributions.get(channel, 0.0) + contribution
                docs[key].metadata["rrf_channel_contributions"] = contributions
            if branch_index == 0 and key not in original_rank:
                original_rank[key] = rank

    # 确定性排序：分数降序 → 原始 query 分支排名升序 → chunk_id 字典序
    ordered_keys = sorted(
        scores,
        key=lambda k: (-scores[k], original_rank.get(k, 1 << 30), k),
    )

    fused: list[Document] = []
    for key in ordered_keys:
        doc = docs[key]
        doc.metadata["rrf_score"] = scores[key]
        fused.append(doc)
    return fused


def _merge_retrieval_metadata(target: Document, incoming: Document) -> None:
    """Merge cross-modal evidence without replacing the canonical text document."""

    channel = str(incoming.metadata.get("retrieval_channel") or "")
    channels = list(target.metadata.get("retrieval_channels") or [])
    if channel and channel not in channels:
        channels.append(channel)
    if channels:
        target.metadata["retrieval_channels"] = channels

    if channel != "visual":
        return
    panel_id = incoming.metadata.get("panel_id")
    if panel_id and not target.metadata.get("matched_panel_id"):
        target.metadata["matched_panel_id"] = panel_id
    target.metadata.setdefault(
        "matched_visual_entity_id", incoming.metadata.get("visual_entity_id")
    )
    target.metadata.setdefault(
        "matched_visual_asset_uri", incoming.metadata.get("visual_asset_uri")
    )
    target.metadata.setdefault(
        "matched_evidence_bbox", incoming.metadata.get("evidence_bbox")
    )
    target.metadata.setdefault(
        "matched_evidence_bbox_coordinate_space",
        incoming.metadata.get("evidence_bbox_coordinate_space")
        or incoming.metadata.get("bbox_coordinate_space"),
    )
    if incoming.metadata.get("visual_similarity") is not None:
        current = target.metadata.get("visual_similarity")
        incoming_score = incoming.metadata["visual_similarity"]
        if current is None or incoming_score > current:
            target.metadata["visual_similarity"] = incoming_score
