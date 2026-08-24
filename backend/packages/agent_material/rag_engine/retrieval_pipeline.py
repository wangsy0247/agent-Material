"""请求级检索管道（ONL-004 ~ ONL-011）。

由 RagService 持有，依赖注入：retriever_getter / rewriter / reranker / config。
所有请求状态写入 RequestContext，禁止写共享对象属性（ONL-001/ONL-008）。

流水线（每步 record_duration，降级走 add_degradation）：

    plan_queries      查询计划：original(1.0) → step_back(0.8) → expand(0.7)，≤ max_queries
    retrieve_branches 逐分支 Vector + BM25 → 单查询 weighted RRF（分支阶段不调 Reranker）
    fuse              跨查询 weighted RRF，保留 fusion_top_n 候选
    global_rerank     原始 query 全局 Rerank（每请求恰好一次），失败回退 RRF 排名
    resolve_parents   有序父块回溯，missing 回退子块（parent_fallback=True）
    build_context     字符预算内拼接上下文 + 同源 sources，freeze_context 冻结
"""

from __future__ import annotations

import os
import inspect
import time
from typing import Any, Callable

from langchain_core.documents import Document

from agent_material.rag_engine.evidence_locator import build_evidence_locator
from agent_material.rag_engine.fusion import doc_fusion_key, weighted_rrf
from agent_material.rag_engine.request_context import RequestContext
from agent_material.utils.logger import get_logger


class RetrievalPipeline:
    """请求级检索管道。所有中间产物写入 RequestContext。"""

    def __init__(
        self,
        retriever_getter: Callable[[], Any],
        rewriter: Any = None,
        reranker: Any = None,
        config: dict | None = None,
        logger: Any = None,
    ):
        """
        Args:
            retriever_getter: 返回 MultiGranularityRetriever 的可调用（每次取最新，支持热更新）
            rewriter: QueryRewriter 实例（改写关闭时可为 None）
            reranker: 全局 Reranker（None 时从 retriever._compressor 解析）
            config: vector_store_config 字典
            logger: 日志器
        """
        self._retriever_getter = retriever_getter
        self._rewriter = rewriter
        self._reranker = reranker
        self._config = config or {}
        self._logger = logger or get_logger("retrieval_pipeline")

    # ── 配置读取 ─────────────────────────────────────────────

    def _rewrite_cfg(self) -> dict:
        return self._config.get("query_rewriting", {})

    def _flags(self) -> dict:
        return self._config.get("feature_flags", {})

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _visual_cfg(self) -> dict:
        structured = self._config.get("structured_index", {}) or {}
        return dict(structured.get("multimodal_embedding", {}) or {})

    @staticmethod
    def _tag_channel(docs: list[Document], channel: str) -> list[Document]:
        tagged: list[Document] = []
        for doc in docs:
            metadata = dict(doc.metadata or {})
            metadata["retrieval_channel"] = channel
            metadata["retrieval_channels"] = [channel]
            tagged.append(Document(page_content=doc.page_content, metadata=metadata))
        return tagged

    # ── 1. 查询计划（ONL-004 / ONL-008）────────────────────────

    def plan_queries(
        self,
        ctx: RequestContext,
        enable_rewrite: bool = False,
        strategy: str = "",
    ) -> None:
        """生成检索查询计划并（可选）生成请求级 HyDE embedding。

        顺序固定：original(1.0) → step_back(0.8) → expand(0.7)，
        唯一查询总数 ≤ max_queries，原始 query 始终在列。
        任一改写失败 → 保留原始 query + degradation，不中断。
        """
        t0 = time.time()
        if not ctx.retrieval_plan:
            self.plan_retrieval_route(ctx)
        cfg = self._rewrite_cfg()
        weights_cfg = cfg.get("weights", {})
        w_original = float(weights_cfg.get("original", 1.0))
        w_step_back = float(weights_cfg.get("step_back", 0.8))
        w_expand = float(weights_cfg.get("expand", 0.7))
        max_queries = int(cfg.get("max_queries", 3))

        original = ctx.original_query
        queries: list[dict] = [{"query": original, "weight": w_original, "type": "original"}]
        seen = {original}

        if enable_rewrite and self._rewriter is not None:
            strat = strategy or cfg.get("default_strategy", "")

            # Step-back（顺序第 2）
            if "step_back" in strat and len(queries) < max_queries:
                try:
                    _, sb = self._rewriter.step_back(original)
                    if sb and sb.strip() and sb not in seen:
                        seen.add(sb)
                        queries.append({"query": sb, "weight": w_step_back, "type": "step_back"})
                except Exception as e:
                    ctx.add_degradation("rewrite:step_back", str(e), "original_query")
                    self._logger.warning(f"[Pipeline] step_back 改写失败（保留原始 query）: {e}")

            # Expand（顺序第 3）
            if "expand" in strat and len(queries) < max_queries:
                try:
                    variants = self._rewriter.expand(original)
                    for v in variants:
                        if len(queries) >= max_queries:
                            break
                        if v and v.strip() and v not in seen:
                            seen.add(v)
                            queries.append({"query": v, "weight": w_expand, "type": "expand"})
                except Exception as e:
                    ctx.add_degradation("rewrite:expand", str(e), "original_query")
                    self._logger.warning(f"[Pipeline] expand 改写失败（保留原始 query）: {e}")

        ctx.generated_queries = queries

        # HyDE：仅影响当前请求的向量检索（embedding 存 ctx，作为参数传递）
        hyde_requested = cfg.get("enable_hyde", False) or "hyde" in (strategy or "")
        if hyde_requested and self._flags().get("request_scoped_hyde", False):
            ctx.hyde_embedding = self._build_hyde_embedding(ctx, original)

        ctx.record_duration("plan_queries", t0)

    def plan_retrieval_route(self, ctx: RequestContext) -> None:
        """为本次请求选择文本/图片通道，不调用额外模型。"""
        from agent_material.rag_engine.retrieval_routing import plan_retrieval_route

        visual_cfg = self._visual_cfg()
        visual_available = self._as_bool(visual_cfg.get("enabled", False)) and bool(
            self._flags().get("enable_visual_search", True)
        )
        plan = plan_retrieval_route(
            ctx.original_query,
            visual_available=visual_available,
            question_type=getattr(ctx.route_result, "question_type", None),
        )
        ctx.retrieval_plan = plan.to_dict()

    def _build_hyde_embedding(self, ctx: RequestContext, query: str):
        """生成 HyDE 假设文档 embedding；失败回退 None（原始 query embedding）。"""
        try:
            if self._rewriter is None:
                raise RuntimeError("QueryRewriter 未注入")
            hyde_doc = self._rewriter.hyde(query)
            if not hyde_doc:
                raise RuntimeError("HyDE 假设文档为空")

            from agent_material.rag_engine.models import embedding_model

            if embedding_model is None:
                raise RuntimeError("Embedding 模型未初始化")
            return embedding_model.embed_query(hyde_doc)
        except Exception as e:
            ctx.add_degradation("hyde", str(e), "original_query_embedding")
            self._logger.warning(f"[Pipeline] HyDE 失败（回退原始 query embedding）: {e}")
            return None

    # ── 2. 分支召回（ONL-005）─────────────────────────────────

    def retrieve_branches(self, ctx: RequestContext) -> None:
        """按计划顺序逐查询执行 Vector + BM25 → 单查询 weighted RRF。

        分支阶段不调 Reranker。BM25 不可用 → vector-only + degradation；
        向量检索抛错 → BM25-only 降级，都不可用则抛明确异常。
        """
        t0 = time.time()
        retriever = self._retriever_getter()
        k = int(self._config.get("k", 10))
        vector_weight = float(self._config.get("vector_weight", 0.5))
        bm25_weight = float(self._config.get("bm25_weight", 0.5))
        visual_cfg = self._visual_cfg()
        visual_weight = float(visual_cfg.get("weight", 0.4))
        flags = self._flags()
        route_plan = ctx.retrieval_plan or {}
        enable_vector = bool(flags.get("enable_vector_search", True)) and bool(
            route_plan.get("enable_text_vector", True)
        )
        enable_bm25 = bool(flags.get("enable_bm25_search", True)) and bool(
            route_plan.get("enable_bm25", True)
        )
        enable_visual = (
            self._as_bool(visual_cfg.get("enabled", False))
            and bool(flags.get("enable_visual_search", True))
            and bool(route_plan.get("enable_visual", False))
        )
        if not enable_vector and not enable_bm25 and not enable_visual:
            raise ValueError("文本向量、BM25 与图片向量召回不能同时关闭")

        branches: list[list[Document]] = []
        for entry in ctx.generated_queries:
            query = entry["query"]
            # HyDE embedding 仅 original 分支的向量路使用（BM25 始终用原始文本）
            hyde = ctx.hyde_embedding if entry.get("type") == "original" else None

            if hasattr(retriever, "search_children"):
                try:
                    search = retriever.search_children
                    parameters = inspect.signature(search).parameters
                    if "enable_vector" in parameters:
                        vector_docs, bm25_docs = search(
                            query,
                            k=k,
                            hyde_embedding=hyde,
                            enable_vector=enable_vector,
                            enable_bm25=enable_bm25,
                        )
                    else:
                        # 旧自定义 retriever 的兼容路径；正式实现会在源头跳过分支。
                        vector_docs, bm25_docs = search(query, k=k, hyde_embedding=hyde)
                        vector_docs = vector_docs if enable_vector else []
                        bm25_docs = bm25_docs if enable_bm25 else []
                except Exception as e:
                    # 向量检索失败 → 尝试 BM25-only 降级（6.3 降级表）
                    bm25_only = self._bm25_fallback(retriever, query) if enable_bm25 else None
                    if bm25_only is None:
                        if not enable_visual:
                            raise RuntimeError(
                                f"向量检索失败且 BM25 不可用，无法完成召回: {e}"
                            ) from e
                        ctx.add_degradation("text_retrieval", str(e), "visual_only")
                        vector_docs, bm25_docs = [], []
                    else:
                        ctx.add_degradation("vector_search", str(e), "bm25_only")
                        vector_docs, bm25_docs = [], bm25_only
            else:
                # 非多粒度检索器（旧链路兼容）：整体 invoke 作为向量路
                vector_docs = retriever.invoke(query)
                bm25_docs = []

            visual_docs: list[Document] = []
            visual_scope = str(visual_cfg.get("query_scope", "original_only"))
            should_search_visual = enable_visual and (
                visual_scope == "all_queries" or entry.get("type") == "original"
            )
            if should_search_visual:
                visual_search = getattr(retriever, "search_visual", None)
                if visual_search is None:
                    self._degrade_once(
                        ctx,
                        "visual_search",
                        "当前检索器不支持图片向量召回",
                        "text_hybrid",
                    )
                else:
                    try:
                        visual_docs = list(visual_search(query, k=k))
                        panel_id = route_plan.get("panel_id")
                        if panel_id:
                            panel_matches = [
                                doc
                                for doc in visual_docs
                                if str(doc.metadata.get("panel_id") or "").lower()
                                == str(panel_id).lower()
                            ]
                            if panel_matches:
                                visual_docs = panel_matches
                    except Exception as e:
                        self._degrade_once(ctx, "visual_search", str(e), "text_hybrid")
                        self._logger.warning(
                            f"[Pipeline] 图片向量召回失败（回退文本混合检索）: {e}"
                        )

            if enable_bm25 and not bm25_docs and not self._bm25_available(retriever):
                self._degrade_once(ctx, "bm25", "BM25 检索器不可用或查询失败", "vector_only")

            # RRF 前 active 二次校验（Spec V2 9.3/10.2）：
            # BM25 语料可能滞后（刷新失败/多进程未同步），批量校验剔除
            # 已退役/已删除候选；校验 SQL 失败 → 当前请求 BM25 路整体弃用。
            if bm25_docs:
                checker = getattr(retriever, "filter_active_bm25", None)
                if checker is not None:
                    try:
                        bm25_docs = checker(bm25_docs)
                    except Exception as e:
                        self._logger.error(
                            f"[Pipeline] BM25 active 二次校验失败（降级 vector-only）: {e}"
                        )
                        self._degrade_once(ctx, "bm25_active_check", str(e), "vector_only")
                        bm25_docs = []

            rankings: list[tuple[list[Document], float]] = []
            if enable_vector and vector_docs:
                rankings.append((self._tag_channel(vector_docs, "text_vector"), vector_weight))
            if enable_bm25 and bm25_docs:
                rankings.append((self._tag_channel(bm25_docs, "bm25"), bm25_weight))
            if should_search_visual and visual_docs:
                rankings.append((self._tag_channel(visual_docs, "visual"), visual_weight))
            branch = weighted_rrf(rankings)
            # 候选携带稳定 chunk_id（doc_fusion_key 兜底在融合内完成）
            branches.append(branch)

        ctx.retrieved_children = branches
        ctx.record_duration("retrieve_branches", t0)

    @staticmethod
    def _bm25_available(retriever: Any) -> bool:
        return getattr(retriever, "_bm25_retriever", None) is not None

    @staticmethod
    def _bm25_fallback(retriever: Any, query: str) -> list[Document] | None:
        bm25 = getattr(retriever, "_bm25_retriever", None)
        if bm25 is None:
            return None
        try:
            return bm25.invoke(query)
        except Exception:
            return None

    @staticmethod
    def _degrade_once(ctx: RequestContext, component: str, reason: str, fallback: str):
        if not any(e["component"] == component for e in ctx.degradation_events):
            ctx.add_degradation(component, reason, fallback)

    # ── 3. 跨查询融合（ONL-006 / ONL-010）──────────────────────

    def fuse(self, ctx: RequestContext) -> None:
        """跨查询 weighted RRF 融合（第一分支=原始 query），保留 fusion_top_n。"""
        t0 = time.time()
        top_n = int(self._rewrite_cfg().get("fusion_top_n", 20))

        if self._flags().get("cross_query_rrf", False):
            rankings = [
                (branch, entry.get("weight", 1.0))
                for branch, entry in zip(ctx.retrieved_children, ctx.generated_queries)
            ]
            fused = weighted_rrf(rankings)
        else:
            # 开关关闭：按分支顺序拼接 + 稳定 chunk_id 去重（禁止随机 hash）
            seen: set[str] = set()
            fused = []
            for branch in ctx.retrieved_children:
                for doc in branch:
                    key = doc_fusion_key(doc)
                    if key not in seen:
                        seen.add(key)
                        fused.append(doc)

        ctx.fused_children = fused[:top_n]
        ctx.record_duration("fuse", t0)

    # ── 4. 全局 Rerank（ONL-007）──────────────────────────────

    def global_rerank(self, ctx: RequestContext) -> None:
        """每请求恰好一次 Reranker 调用（query=original_query）。

        成功按 relevance 排序（rerank_score 已由 reranker 回写 metadata）；
        失败/超时/限流 → 回退跨查询 RRF 排名 + degradation。
        """
        t0 = time.time()
        candidates = ctx.fused_children
        top_n = int(self._config.get("reranker", {}).get("top_n", 3))
        reranker = self._resolve_reranker()

        if self._flags().get("global_rerank", False) and reranker is not None and candidates:
            try:
                ranked = reranker.compress_documents(candidates, ctx.original_query)
                ranked = list(ranked)
                # reranker 内部 API 失败时会静默回退原顺序（无 rerank_score 回写）
                if ranked and "rerank_score" not in ranked[0].metadata:
                    ctx.add_degradation(
                        "reranker", "API 调用失败，未返回 relevance_score", "rrf_order"
                    )
                ctx.reranked_children = ranked[:top_n]
            except Exception as e:
                ctx.add_degradation("reranker", str(e), "rrf_order")
                self._logger.warning(f"[Pipeline] 全局 Rerank 失败（回退 RRF 排名）: {e}")
                ctx.reranked_children = candidates[:top_n]
        else:
            ctx.reranked_children = candidates[:top_n]

        ctx.record_duration("global_rerank", t0)

    def _resolve_reranker(self):
        if self._reranker is not None:
            return self._reranker
        try:
            retriever = self._retriever_getter()
            return getattr(retriever, "_compressor", None)
        except Exception:
            return None

    # ── 5. 有序父块回溯（ONL-009）──────────────────────────────

    def resolve_parents(self, ctx: RequestContext) -> None:
        """按 reranked 子块顺序**原位**回溯父块（V2 Spec 10.4）。

        逐个遍历 reranked children：
        - 有 parent_id 且父块存在：在原位置替换为父块；
        - 父块缺失：在原位置保留子块并标记 parent_fallback；
        - 无 parent_id：在原位置保留子块；
        - 相同 parent_id 只保留第一次出现。

        父块 SQL 已过滤非 active 版本（_fetch_parents 内部保证）。
        """
        t0 = time.time()
        children = ctx.reranked_children
        if not self._flags().get("parent_backtrack", True):
            ctx.resolved_parents = list(children)
            ctx.record_duration("resolve_parents", t0)
            return
        retriever = self._retriever_getter()

        # parent_id 不是全局唯一键；以 (parent_id, document_id, version_id)
        # 作为引用身份，避免 legacy 数据在不同文档/版本间串读。
        parent_ids: list[str] = []
        parent_scopes: list[tuple[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for doc in children:
            pid = doc.metadata.get("parent_id", "")
            scope = (
                doc.metadata.get("document_id", ""),
                doc.metadata.get("version_id", ""),
            )
            ref = (pid, scope[0], scope[1])
            if pid and ref not in seen:
                seen.add(ref)
                parent_ids.append(pid)
                parent_scopes.append(scope)

        if not parent_ids:
            ctx.resolved_parents = list(children)
            ctx.record_duration("resolve_parents", t0)
            return

        fetch = getattr(retriever, "_fetch_parents", None)
        if fetch is None:
            ordered_docs, missing_ids = [], list(parent_ids)
        else:
            ordered_docs, missing_ids = fetch(parent_ids, parent_scopes)

        parent_by_ref = {
            (
                d.metadata.get("parent_id", ""),
                d.metadata.get("document_id", ""),
                d.metadata.get("version_id", ""),
            ): d
            for d in ordered_docs
        }
        parent_by_id = {d.metadata.get("parent_id", ""): d for d in ordered_docs}

        if missing_ids:
            ctx.add_degradation(
                "parent_fetch",
                f"{len(missing_ids)} 个父块未命中: {missing_ids[:5]}",
                "child_fallback",
            )

        # 原位替换：严格按 reranked children 顺序生成结果
        final_docs: list[Document] = []
        emitted_refs: set[tuple[str, str, str]] = set()
        for child in children:
            pid = child.metadata.get("parent_id", "")
            if not pid:
                # 无 parent_id：原位保留子块
                final_docs.append(child)
                continue
            ref = (
                pid,
                child.metadata.get("document_id", ""),
                child.metadata.get("version_id", ""),
            )
            if ref in emitted_refs:
                # 同一父块只保留第一次出现
                continue
            emitted_refs.add(ref)
            if ref[1]:
                parent = parent_by_ref.get(ref)
            else:
                parent = parent_by_id.get(pid)
            if parent is not None:
                # 父块来自数据库，通常不携带本次请求的 RRF/Rerank 分数。
                # 将子块的请求级排序信息复制到返回副本，避免来源 score 变为 0，
                # 同时保留命中的具体 child 身份用于排障。
                parent_metadata = dict(parent.metadata)
                for score_key in ("rerank_score", "rrf_score", "score"):
                    if child.metadata.get(score_key) is not None:
                        parent_metadata[score_key] = child.metadata[score_key]
                for evidence_key in (
                    "retrieval_channels",
                    "rrf_channel_contributions",
                    "visual_similarity",
                    "matched_panel_id",
                    "matched_visual_entity_id",
                    "matched_visual_asset_uri",
                    "matched_evidence_bbox",
                    "matched_evidence_bbox_coordinate_space",
                ):
                    if child.metadata.get(evidence_key) is not None:
                        parent_metadata[evidence_key] = child.metadata[evidence_key]
                parent_metadata["matched_chunk_id"] = child.metadata.get("chunk_id")
                final_docs.append(
                    Document(
                        page_content=parent.page_content,
                        metadata=parent_metadata,
                    )
                )
            else:
                final_docs.append(
                    Document(
                        page_content=child.page_content,
                        metadata={**child.metadata, "parent_fallback": True},
                    )
                )

        ctx.resolved_parents = final_docs
        ctx.record_duration("resolve_parents", t0)

    # ── 6. 上下文构建与冻结（ONL-002 / ONL-011）────────────────

    def build_context(self, ctx: RequestContext) -> None:
        """按父块相关性顺序拼接上下文，受字符预算限制，并构造同源 sources。

        预算超出时按顺序丢弃尾部块。最后 freeze_context 冻结（不可变）。
        """
        t0 = time.time()
        budget = int(self._config.get("context_budget_chars", 8000))

        # parent_id → 子块最佳分数（rerank_score 优先，其次 rrf_score）
        pid_score: dict[str, float] = {}
        for child in ctx.reranked_children:
            pid = child.metadata.get("parent_id", "")
            score = child.metadata.get("rerank_score", child.metadata.get("rrf_score"))
            if pid and score is not None and pid not in pid_score:
                pid_score[pid] = score

        context_parts: list[str] = []
        docs_used: list[Document] = []
        sources: list[dict] = []
        total_chars = 0
        truncated = False

        for doc in ctx.resolved_parents:
            m = doc.metadata
            source = m.get("source", "未知来源")
            source_filename = m.get("source_filename") or (
                os.path.basename(source) if isinstance(source, str) else "未知来源"
            )
            page_unknown = bool(m.get("page_unknown", False))
            page_start = m.get("page_start")
            page_end = m.get("page_end")

            page_line = ""
            if not page_unknown and page_start is not None:
                if page_end is not None and page_end != page_start:
                    page_line = f"页码：{page_start}-{page_end}\n"
                else:
                    page_line = f"页码：{page_start}\n"

            index = len(sources) + 1
            content_type = m.get("content_type", "text")
            evidence_label = {
                "figure": "图表证据",
                "table": "表格证据",
            }.get(content_type, "参考资料")
            block = (
                f"[{evidence_label}{index}] 来源：{source_filename}\n"
                f"{page_line}{doc.page_content}\n\n"
            )
            if total_chars + len(block) > budget:
                if not docs_used:
                    # 首块即超预算：保留预算范围内的截断内容，
                    # 不允许检索结果非空但最终上下文为空（V2 Spec 10.5）
                    block = block[:budget]
                    self._logger.warning(
                        f"[Pipeline] 首个上下文块（{len(block)} 字符）超过预算 {budget}，"
                        f"按预算截断保留"
                    )
                    context_parts.append(block)
                    docs_used.append(doc)
                    truncated = True
                else:
                    self._logger.info(
                        f"[Pipeline] 上下文预算截断: {total_chars} + {len(block)} > {budget}，"
                        f"丢弃尾部 {len(ctx.resolved_parents) - len(docs_used)} 块"
                    )
                    truncated = True
                break

            context_parts.append(block)
            docs_used.append(doc)
            total_chars += len(block)

            pid = m.get("parent_id", "")
            score = m.get("rerank_score", m.get("rrf_score", pid_score.get(pid, 0.0)))
            content = doc.page_content
            source_record = {
                "index": index,
                "source": source,
                "source_filename": source_filename,
                "page_start": page_start,
                "page_end": page_end,
                "page_unknown": page_unknown,
                "heading_path": m.get("heading_path"),
                "parent_id": pid or None,
                "chunk_id": m.get("chunk_id"),
                "content_type": content_type,
                "artifact_id": m.get("artifact_id"),
                "caption": m.get("caption"),
                "asset_uri": m.get("asset_uri"),
                "visual_asset_uri": m.get("visual_asset_uri"),
                "panel_assets": m.get("panel_assets") or [],
                "bbox": m.get("bbox"),
                "bbox_coordinate_space": m.get("bbox_coordinate_space"),
                "retrieval_channels": m.get("retrieval_channels") or [],
                "visual_similarity": m.get("visual_similarity"),
                "matched_panel_id": m.get("matched_panel_id"),
                "matched_visual_entity_id": m.get("matched_visual_entity_id"),
                "matched_visual_asset_uri": m.get("matched_visual_asset_uri"),
                "matched_evidence_bbox": m.get("matched_evidence_bbox"),
                "matched_evidence_bbox_coordinate_space": m.get(
                    "matched_evidence_bbox_coordinate_space"
                ),
                "visual_status": m.get("visual_status", "not_requested"),
                "visual_summary": m.get("visual_summary"),
                "visual_model": m.get("visual_model"),
                "visual_confidence": m.get("visual_confidence"),
                "visual_prompt_version": m.get("visual_prompt_version"),
                "readability_status": m.get("readability_status", "unknown"),
                "legend_mapping_status": m.get("legend_mapping_status", "not_applicable"),
                "numeric_fact_status": m.get("numeric_fact_status", "not_applicable"),
                "document_id": m.get("document_id"),
                "version_id": m.get("version_id"),
                "score": score,
                "snippet": content[:300] + "..." if len(content) > 300 else content,
            }
            source_record["evidence_locator"] = build_evidence_locator(m)
            sources.append(source_record)

        # 首块被截断时补记 sources（正文截断但来源仍需可追溯）
        if truncated and not sources and docs_used:
            doc = docs_used[0]
            m = doc.metadata
            source = m.get("source", "未知来源")
            content = doc.page_content
            source_record = {
                "index": 1,
                "source": source,
                "source_filename": m.get("source_filename")
                or (os.path.basename(source) if isinstance(source, str) else "未知来源"),
                "page_start": m.get("page_start"),
                "page_end": m.get("page_end"),
                "page_unknown": bool(m.get("page_unknown", False)),
                "heading_path": m.get("heading_path"),
                "parent_id": m.get("parent_id") or None,
                "chunk_id": m.get("chunk_id"),
                "content_type": m.get("content_type", "text"),
                "artifact_id": m.get("artifact_id"),
                "caption": m.get("caption"),
                "asset_uri": m.get("asset_uri"),
                "visual_asset_uri": m.get("visual_asset_uri"),
                "panel_assets": m.get("panel_assets") or [],
                "bbox": m.get("bbox"),
                "bbox_coordinate_space": m.get("bbox_coordinate_space"),
                "retrieval_channels": m.get("retrieval_channels") or [],
                "visual_similarity": m.get("visual_similarity"),
                "matched_panel_id": m.get("matched_panel_id"),
                "matched_visual_entity_id": m.get("matched_visual_entity_id"),
                "matched_visual_asset_uri": m.get("matched_visual_asset_uri"),
                "matched_evidence_bbox": m.get("matched_evidence_bbox"),
                "matched_evidence_bbox_coordinate_space": m.get(
                    "matched_evidence_bbox_coordinate_space"
                ),
                "visual_status": m.get("visual_status", "not_requested"),
                "visual_summary": m.get("visual_summary"),
                "visual_model": m.get("visual_model"),
                "visual_confidence": m.get("visual_confidence"),
                "visual_prompt_version": m.get("visual_prompt_version"),
                "readability_status": m.get("readability_status", "unknown"),
                "legend_mapping_status": m.get("legend_mapping_status", "not_applicable"),
                "numeric_fact_status": m.get("numeric_fact_status", "not_applicable"),
                "document_id": m.get("document_id"),
                "version_id": m.get("version_id"),
                "score": m.get(
                    "rerank_score",
                    m.get("rrf_score", pid_score.get(m.get("parent_id", ""), 0.0)),
                ),
                "snippet": content[:300] + "..." if len(content) > 300 else content,
            }
            source_record["evidence_locator"] = build_evidence_locator(m)
            sources.append(source_record)

        ctx.context_truncated = truncated
        ctx.freeze_context("".join(context_parts), docs_used, sources)
        ctx.record_duration("build_context", t0)
