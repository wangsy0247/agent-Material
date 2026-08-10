import os
import time
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agent_material.rag_engine.vector_store import VectorStore
from agent_material.rag_engine.request_context import RequestContext
from agent_material.utils.prompt_loader import get_prompts
from agent_material.rag_engine.models import chat_model
from agent_material.utils.config_handler import vector_store_config, prompt_config
from agent_material.utils.path_tools import get_abs_path
from agent_material.observability.structured_logger import get_logger
from agent_material.observability.metrics import (
    retrieval_duration,
    retrieval_docs,
    llm_call_total,
    llm_call_duration,
    llm_token_total,
)

#: 无关问题固定拒答文案
IRRELEVANT_REPLY = (
    "当前问题与材料科学知识库领域无关，暂时无法回答。"
    "请提出材料科学、计算材料学或相关领域的问题。"
)
#: 无检索上下文时的固定回复
NO_CONTEXT_REPLY = "根据现有资料无法回答该问题。"


def _get_run_config(
    run_name: str,
    query: str = "",
    tags: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建 LangChain run config（支持 LangFuse Callback 注入）。"""
    metadata: dict[str, Any] = {"query": query}
    if extra_metadata:
        metadata.update(extra_metadata)
    config: dict[str, Any] = {"run_name": run_name, "metadata": metadata}
    if tags:
        config["tags"] = tags

    # 注入 LangFuse Callback Handler（若已初始化）
    try:
        from agent_material.utils.langfuse_setup import get_langfuse_handler
        handler = get_langfuse_handler()
        if handler:
            callbacks = config.setdefault("callbacks", [])
            if isinstance(callbacks, list):
                callbacks.append(handler)
    except ImportError:
        pass
    return config


class RagService:
    """RAG 服务层 — 检索 + LLM 生成 + 查询路由 + 查询改写。

    内置路由功能（自动）：闲聊直答、无关拦截、问题类型标注。
    可选改写功能（opt-in）：扩展、HyDE、退一步、分解。
    """

    def __init__(self):
        self.vector_store = VectorStore()
        prompt_path = prompt_config.get(
            "rag_prompt_path", "backend/packages/agent_material/prompts/rag/rag_prompt.txt"
        )
        self.prompt_text = get_prompts(get_abs_path(prompt_path)) or ""
        self.prompts_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self._chain = None  # 延迟初始化（避免无 Chat API key 时阻断纯检索）
        self._logger = get_logger("rag_service")

        # ── 可选组件（延迟初始化） ──
        self._router = None
        self._rewriter = None
        self._chat_prompt = None
        self._pipeline = None  # 请求级检索管道（single_retrieval_sse 路径）

        # ── 配置 ──
        self._routing_config = vector_store_config.get("query_routing", {})
        self._rewrite_config = vector_store_config.get("query_rewriting", {})
        self._default_strategy = self._rewrite_config.get("default_strategy", "step_back+expand")
        self._enable_hyde = self._rewrite_config.get("enable_hyde", False)

    # ── 延迟初始化 ────────────────────────────────────────────

    def _get_chain(self):
        if self._chain is None:
            if self.model is None:
                raise RuntimeError("Chat 模型未初始化，请检查环境变量")
            self._chain = self.prompts_template | self.model | StrOutputParser()
        return self._chain

    def _get_router(self):
        if self._router is None:
            from agent_material.rag_engine.query_router import QueryRouter

            self._router = QueryRouter()
        return self._router

    def _get_rewriter(self):
        if self._rewriter is None:
            from agent_material.rag_engine.query_rewriter import QueryRewriter

            self._rewriter = QueryRewriter()
        return self._rewriter

    def _get_chat_prompt(self) -> PromptTemplate:
        if self._chat_prompt is None:
            path = prompt_config.get(
                "chat_prompt_path", "backend/packages/agent_material/prompts/rag/chat_prompt.txt"
            )
            text = get_prompts(get_abs_path(path)) or ""
            self._chat_prompt = PromptTemplate.from_template(text)
        return self._chat_prompt

    def _get_pipeline(self):
        """请求级检索管道（延迟初始化，依赖注入，无共享可变状态）。"""
        if self._pipeline is None:
            from agent_material.rag_engine.retrieval_pipeline import RetrievalPipeline

            self._pipeline = RetrievalPipeline(
                retriever_getter=lambda: self.retriever,
                rewriter=self._get_rewriter(),
                config=vector_store_config,
                logger=self._logger,
            )
        return self._pipeline

    @staticmethod
    def _single_retrieval_enabled() -> bool:
        """single_retrieval_sse 开关：关闭时走旧路径（回滚兼容）。"""
        return vector_store_config.get("feature_flags", {}).get(
            "single_retrieval_sse", False
        )

    # ── 检索器 ────────────────────────────────────────────────

    @property
    def retriever(self):
        """每次获取最新的检索器（支持热更新）。"""
        return self.vector_store.get_retriever()

    def retriever_docx(self, query: str):
        """执行检索，返回相关文档列表"""
        start = time.time()
        docs = self.retriever.invoke(query)
        elapsed = time.time() - start
        retrieval_duration.labels(stage="full").observe(elapsed)
        retrieval_docs.labels(stage="full").observe(len(docs))
        self._logger.info(
            "retriever_done",
            extra={"docs_count": len(docs), "duration_ms": elapsed * 1000},
        )
        return docs

    # ── 直接回答（闲聊） ──────────────────────────────────────

    def _direct_answer(self, query: str) -> str:
        """对闲聊/寒暄类查询直接 LLM 回答（无 RAG 检索）。"""
        chain = self._get_chat_prompt() | self.model | StrOutputParser()
        return chain.invoke(
            {"input": query},
            config=_get_run_config(
                run_name="direct_answer",
                query=query,
                tags=["chat", "no_rag"],
            ),
        )

    # ── 文档融合工具 ──────────────────────────────────────────

    @staticmethod
    def _merge_and_dedup_docs(doc_lists: list[list[Document]]) -> list[Document]:
        """合并多路检索结果，按 page_content 去重（取前200字符为键）。"""
        seen: set[int] = set()
        merged: list[Document] = []
        for docs in doc_lists:
            for doc in docs:
                key = hash(doc.page_content[:200])
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
        return merged

    # ── 改写策略检索 ──────────────────────────────────────────

    def _resolve_strategy(self, strategy: str) -> str:
        """解析改写策略：空字符串使用配置默认，enable_hyde 自动追加。

        Args:
            strategy: 原始策略字符串（空=使用默认, none=跳过, 其他=直接使用）

        Returns:
            解析后的策略字符串
        """
        # 显式指定了有效策略 → 直接使用
        if strategy and strategy != "none":
            resolved = strategy
        else:
            resolved = self._default_strategy
            self._logger.info(f"[Strategy] using default: {resolved}")

        # HyDE 开关：自动在策略前追加 hyde+
        if self._enable_hyde and "hyde" not in resolved:
            resolved = f"hyde+{resolved}"
            self._logger.info(f"[Strategy] HyDE enabled → {resolved}")

        return resolved

    def _retrieve_with_strategy(
        self, query: str, strategy: str
    ) -> list[Document]:
        """根据改写策略执行检索，支持组合策略（用 + 连接多个策略）。

        单策略示例: "expand", "hyde", "step_back", "decompose"
        组合策略示例: "step_back+expand", "hyde+expand", "decompose+expand"
        特殊值: "" 或 "auto" → 使用 _resolve_strategy() 解析

        组合时策略按顺序流水线执行：前一个策略的输出查询列表
        作为下一个策略的输入，最终所有查询合并去重后多路检索。
        """
        # 空字符串 → 解析为实际策略
        if not strategy:
            strategy = self._resolve_strategy(strategy)
        if strategy == "none":
            return self.retriever_docx(query)

        strategies = [s.strip() for s in strategy.split("+") if s.strip()]

        if not strategies:
            return self.retriever_docx(query)

        if len(strategies) == 1:
            return self._retrieve_single(query, strategies[0])
        else:
            return self._retrieve_combined(query, strategies)

    def _retrieve_single(self, query: str, strategy: str) -> list[Document]:
        """单策略检索（快速路径）。"""
        if strategy == "expand":
            return self._retrieve_with_expansion(query)
        elif strategy == "hyde":
            return self._retrieve_with_hyde(query)
        elif strategy == "step_back":
            return self._retrieve_with_stepback(query)
        elif strategy == "decompose":
            return self._retrieve_with_decomposition(query)
        else:
            self._logger.warning(f"未知改写策略: {strategy}，回退到原始查询")
            return self.retriever_docx(query)

    def _retrieve_combined(
        self, query: str, strategies: list[str]
    ) -> list[Document]:
        """流水线组合多种改写策略，最终多路检索融合。

        流水线模型：
        1. 阶段 1：初始化查询列表 = [原始查询]
        2. 阶段 2：按顺序执行各策略，每个策略消费并扩展查询列表：
           - step_back: 对每个查询生成抽象版本 → 查询列表翻倍
           - decompose: 对每个查询拆分子问题 → 查询列表扩展
           - expand: 对每个查询生成同义变体 → 查询列表 × N
           - hyde: 只对第一个查询生成假设文档 → 保存 embedding
        3. 阶段 3：查询列表去重
        4. 阶段 4：多路检索（如 hyde 设置了 embedding，向量路使用假设文档 embedding）
        5. 阶段 5：结果合并去重 → 截断返回
        """
        cfg = self._rewrite_config
        merge_top_k = cfg.get("expansion", {}).get("merge_top_k", 10)
        final_top_k = cfg.get("expansion", {}).get("final_top_k", 5)
        rewriter = self._get_rewriter()

        # 阶段 1：初始查询列表
        queries: list[str] = [query]
        hyde_embedding: list[float] | None = None

        # 阶段 2：流水线执行各策略
        for strategy in strategies:
            valid = self._validate_strategy(strategy)
            if not valid:
                self._logger.warning(f"[Combined] 跳过未知策略: {strategy}")
                continue

            if strategy == "expand":
                queries = self._pipeline_expand(queries, rewriter)

            elif strategy == "step_back":
                queries = self._pipeline_stepback(queries, rewriter)

            elif strategy == "decompose":
                queries = self._pipeline_decompose(queries, rewriter)

            elif strategy == "hyde":
                # HyDE 只对第一个查询（原始查询）生成假设文档
                hyde_embedding = self._pipeline_hyde(queries[0], rewriter)

        # 阶段 3：查询去重
        seen: set[str] = set()
        unique_queries: list[str] = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)

        self._logger.info(
            f"[Combined] pipeline={'+'.join(strategies)} | "
            f"{len(unique_queries)} unique queries"
        )

        # 阶段 4：多路检索
        if hyde_embedding:
            self.vector_store.set_hyde_embedding(hyde_embedding)

        try:
            all_results: list[list[Document]] = []
            for q in unique_queries:
                docs = self.retriever_docx(q)
                if docs:
                    all_results.append(docs[:merge_top_k])
        finally:
            if hyde_embedding:
                self.vector_store.set_hyde_embedding(None)

        # 阶段 5：合并去重 → 截断
        merged = self._merge_and_dedup_docs(all_results)
        self._logger.info(
            f"[Combined] {len(unique_queries)} queries → "
            f"{len(merged)} unique docs → top {final_top_k}"
        )
        return merged[:final_top_k]

    # ── 流水线各阶段 ──────────────────────────────────────────

    @staticmethod
    def _validate_strategy(strategy: str) -> bool:
        """校验策略名称是否合法。"""
        return strategy in ("expand", "hyde", "step_back", "decompose")

    @staticmethod
    def _pipeline_expand(
        queries: list[str], rewriter
    ) -> list[str]:
        """流水线阶段：查询扩展 — 对每个查询生成同义变体。"""
        expanded: list[str] = []
        for q in queries:
            variants = rewriter.expand(q)
            expanded.extend(variants)
        return expanded

    @staticmethod
    def _pipeline_stepback(
        queries: list[str], rewriter
    ) -> list[str]:
        """流水线阶段：退一步 — 对每个查询生成抽象版本。"""
        result: list[str] = []
        for q in queries:
            orig, sb = rewriter.step_back(q)
            result.append(orig)
            if sb and sb != orig:
                result.append(sb)
        return result

    @staticmethod
    def _pipeline_decompose(
        queries: list[str], rewriter
    ) -> list[str]:
        """流水线阶段：分解 — 对每个查询拆分子问题。"""
        decomposed: list[str] = []
        for q in queries:
            subs = rewriter.decompose(q)
            decomposed.extend(subs)
        return decomposed

    def _pipeline_hyde(
        self, query: str, rewriter
    ) -> list[float] | None:
        """流水线阶段：HyDE — 为第一个查询生成假设文档 embedding。"""
        try:
            hyde_doc = rewriter.hyde(query)
            if not hyde_doc:
                self._logger.warning("[Combined] HyDE 假设文档为空")
                return None

            from agent_material.rag_engine.models import embedding_model

            if embedding_model is None:
                self._logger.error("[Combined] Embedding 模型未初始化")
                return None

            return embedding_model.embed_query(hyde_doc)
        except Exception as e:
            self._logger.error(f"[Combined] HyDE 阶段失败: {e}")
            return None

    def _retrieve_with_expansion(self, query: str) -> list[Document]:
        """查询扩展检索：生成变体 → 多路检索 → 融合。"""
        cfg = self._rewrite_config.get("expansion", {})
        merge_top_k = cfg.get("merge_top_k", 10)
        final_top_k = cfg.get("final_top_k", 5)

        rewriter = self._get_rewriter()
        variants = rewriter.expand(query)

        all_results: list[list[Document]] = []
        for vq in variants:
            docs = self.retriever_docx(vq)
            if docs:
                all_results.append(docs[:merge_top_k])

        merged = self._merge_and_dedup_docs(all_results)
        self._logger.info(
            f"[Expand] {len(variants)} variants → {len(merged)} unique docs"
        )
        return merged[:final_top_k]

    def _retrieve_with_hyde(self, query: str) -> list[Document]:
        """HyDE 检索：假设文档 embedding 替代 query embedding 走完整检索链。

        流程：
        1. LLM 生成假设文档
        2. 对假设文档做 embedding
        3. 将 embedding 注入向量检索器的 embedding_override
        4. 走完整检索链（向量用 HyDE embedding + BM25 用原始 query + RRF + CrossEncoder + 父chunk 映射）
        5. 清除 embedding_override
        """
        rewriter = self._get_rewriter()
        hyde_doc = rewriter.hyde(query)

        if not hyde_doc:
            self._logger.warning("[HyDE] 假设文档为空，回退到原始查询")
            return self.retriever_docx(query)

        # 获取假设文档的 embedding
        from agent_material.rag_engine.models import embedding_model

        if embedding_model is None:
            self._logger.error("[HyDE] Embedding 模型未初始化")
            return self.retriever_docx(query)

        hyde_vector = embedding_model.embed_query(hyde_doc)

        # 将 HyDE embedding 注入向量检索器，走完整检索链
        # BM25 仍使用原始 query text，CrossEncoder 也使用原始 query
        self.vector_store.set_hyde_embedding(hyde_vector)
        try:
            docs = self.retriever_docx(query)
        finally:
            self.vector_store.set_hyde_embedding(None)  # 清除覆盖

        self._logger.info(
            f"[HyDE] hyde_doc={len(hyde_doc)} chars → {len(docs)} final docs"
        )
        return docs

    def _retrieve_with_stepback(self, query: str) -> list[Document]:
        """退一步检索：抽象查询 + 原始查询 → 双路检索 → 融合。"""
        cfg = self._rewrite_config.get("step_back", {})
        merge_top_k = cfg.get("merge_top_k", 10)
        final_top_k = cfg.get("final_top_k", 5)

        rewriter = self._get_rewriter()
        orig, stepback = rewriter.step_back(query)

        all_results: list[list[Document]] = []
        for q in (orig, stepback):
            docs = self.retriever_docx(q)
            if docs:
                all_results.append(docs[:merge_top_k])

        merged = self._merge_and_dedup_docs(all_results)
        self._logger.info(
            f"[StepBack] 2 queries → {len(merged)} unique docs"
        )
        return merged[:final_top_k]

    def _retrieve_with_decomposition(self, query: str) -> list[Document]:
        """分解检索：拆子问题 → 逐路检索 → 融合。"""
        cfg = self._rewrite_config.get("decomposition", {})
        merge_top_k = cfg.get("merge_top_k", 10)
        final_top_k = cfg.get("final_top_k", 5)

        rewriter = self._get_rewriter()
        sub_queries = rewriter.decompose(query)

        all_results: list[list[Document]] = []
        for sq in sub_queries:
            docs = self.retriever_docx(sq)
            if docs:
                all_results.append(docs[:merge_top_k])

        merged = self._merge_and_dedup_docs(all_results)
        self._logger.info(
            f"[Decompose] {len(sub_queries)} sub-queries → {len(merged)} unique docs"
        )
        return merged[:final_top_k]

    # ── 请求级 prepare / stream_answer（ONL-001/002/003）───────

    def prepare(
        self,
        query: str,
        enable_rewrite: bool = False,
        rewrite_strategy: str = "",
    ) -> RequestContext:
        """请求级检索准备：路由 → （新路径）检索管道全流程 或（旧路径）现有逻辑。

        全程只执行一次检索，结果冻结进 RequestContext（final_context/sources）。
        chat/irrelevant 高置信时标记 route_decision 后直接返回（不检索）。
        """
        ctx = RequestContext(original_query=query)
        routing_enabled = self._routing_config.get("enabled", True)
        model_name = chat_model.model_name if chat_model else "unknown"

        # ── 阶段 1：查询路由（失败 → 正常 RAG + degradation）──
        ctx.route_decision = "rag"
        if routing_enabled:
            t0 = time.time()
            try:
                classification = self._get_router().classify(query)
                ctx.record_duration("route_classify", t0)
                ctx.route_result = classification
                retrieval_duration.labels(stage="route_classify").observe(time.time() - t0)
                llm_call_total.labels(model=model_name, purpose="classify").inc()

                threshold = self._routing_config.get("confidence_threshold", 0.6)
                if (
                    self._routing_config.get("irrelevant_detection", True)
                    and classification.is_irrelevant
                    and classification.confidence >= threshold
                ):
                    ctx.route_decision = "irrelevant"
                    return ctx
                if (
                    self._routing_config.get("chat_detection", True)
                    and classification.is_chat
                    and classification.confidence >= threshold
                ):
                    ctx.route_decision = "chat"
                    return ctx
                # question_type（factual/summary/comparison）仅记录，不改变策略（ONL-003）
            except Exception as e:
                ctx.add_degradation("router", str(e), "normal_rag")
                self._logger.warning(f"查询路由失败: {e}，回退到正常 RAG 流程")

        # ── 阶段 2：检索 ──
        t_retrieve = time.time()
        if self._single_retrieval_enabled():
            # 新路径：请求级检索管道（plan→branches→fuse→rerank→parents→freeze）
            ctx.rewrite_strategy = (
                self._resolve_strategy(rewrite_strategy) if enable_rewrite else ""
            )
            pipeline = self._get_pipeline()
            pipeline.plan_queries(
                ctx, enable_rewrite=enable_rewrite, strategy=ctx.rewrite_strategy
            )
            pipeline.retrieve_branches(ctx)
            pipeline.fuse(ctx)
            pipeline.global_rerank(ctx)
            pipeline.resolve_parents(ctx)
            pipeline.build_context(ctx)
        else:
            # 旧路径（回滚兼容）：现有 _retrieve_with_strategy / retriever_docx
            if enable_rewrite:
                strategy = self._resolve_strategy(rewrite_strategy)
                ctx.rewrite_strategy = strategy
                context_docx = self._retrieve_with_strategy(query, strategy)
            else:
                context_docx = self.retriever_docx(query)
            self._freeze_legacy_context(ctx, context_docx)
        ctx.record_duration("retrieve_total", t_retrieve)
        retrieval_duration.labels(stage="total").observe(time.time() - t_retrieve)
        return ctx

    def _freeze_legacy_context(self, ctx: RequestContext, context_docx: list[Document]):
        """旧路径结果冻结：保持原有上下文拼接格式与 sources 结构。"""
        context = ""
        sources: list[dict] = []
        for i, doc in enumerate(context_docx, start=1):
            source = doc.metadata.get("source", "未知来源")
            source_name = (
                os.path.basename(source) if isinstance(source, str) else "未知来源"
            )
            context += f"[参考资料{i}] 来源：{source_name}\n{doc.page_content}\n\n"
            content = doc.page_content
            sources.append(
                {
                    "index": i,
                    "source": source,
                    "source_filename": source_name,
                    "page_start": doc.metadata.get("page_start"),
                    "page_end": doc.metadata.get("page_end"),
                    "page_unknown": bool(doc.metadata.get("page_unknown", False)),
                    "heading_path": doc.metadata.get("heading_path"),
                    "parent_id": doc.metadata.get("parent_id") or None,
                    "score": doc.metadata.get(
                        "rerank_score", doc.metadata.get("score", 0.0)
                    ),
                    "snippet": content[:300] + "..." if len(content) > 300 else content,
                }
            )
        ctx.freeze_context(context, context_docx, sources)

    def stream_answer(self, ctx: RequestContext):
        """基于冻结上下文流式生成（不再检索）。

        chat → chat prompt 直答流；irrelevant → 固定拒答文案；
        rag → ctx.final_context 拼 prompt 流式生成（无上下文 → 无法回答文案）。
        """
        model_name = chat_model.model_name if chat_model else "unknown"

        if ctx.route_decision == "irrelevant":
            yield IRRELEVANT_REPLY
            return

        if ctx.route_decision == "chat":
            chain = self._get_chat_prompt() | self.model | StrOutputParser()
            with llm_call_duration.labels(model=model_name, purpose="direct_answer").time():
                llm_call_total.labels(model=model_name, purpose="direct_answer").inc()
                for chunk in chain.stream(
                    {"input": ctx.original_query},
                    config=_get_run_config(
                        run_name="direct_answer",
                        query=ctx.original_query,
                        tags=["chat", "no_rag"],
                    ),
                ):
                    yield chunk
            return

        # rag 路径
        if not ctx.final_context:
            yield NO_CONTEXT_REPLY
            return

        t_gen = time.time()
        llm_call_total.labels(model=model_name, purpose="generate").inc()
        output_chars = 0
        for chunk in self._get_chain().stream(
            {"input": ctx.original_query, "context": ctx.final_context},
            config=_get_run_config(
                run_name="rag_generate_stream",
                query=ctx.original_query,
                tags=["rag", "stream"],
                extra_metadata={"context_doc_count": len(ctx.context_docs)},
            ),
        ):
            output_chars += len(chunk)
            yield chunk
        llm_call_duration.labels(model=model_name, purpose="generate").observe(
            time.time() - t_gen
        )
        llm_token_total.labels(model=model_name, type="input").inc(
            len(ctx.final_context) // 2
        )
        llm_token_total.labels(model=model_name, type="output").inc(output_chars // 2)
        llm_token_total.labels(model=model_name, type="total").inc(
            len(ctx.final_context) // 2 + output_chars // 2
        )

    # ── RAG 生成 ──────────────────────────────────────────────

    def get_summary(
        self,
        query: str,
        enable_rewrite: bool = False,
        rewrite_strategy: str = "",
    ) -> str:
        """基于检索结果生成回答（自动路由 + 可选改写）。

        Args:
            query: 用户问题
            enable_rewrite: 是否启用查询改写（额外 LLM 调用，更高延迟/成本）
            rewrite_strategy: 改写策略。空字符串使用 config.default_strategy，
                             或显式指定策略名/组合策略

        Returns:
            生成的回答文本
        """
        if self._single_retrieval_enabled():
            ctx = self.prepare(query, enable_rewrite, rewrite_strategy)
            self._logger.info("request_summary", extra=ctx.summary())
            return "".join(self.stream_answer(ctx))

        routing_enabled = self._routing_config.get("enabled", True)
        model_name = chat_model.model_name if chat_model else "unknown"

        # ── 阶段 1：查询路由（自动）──
        if routing_enabled:
            t0 = time.time()
            try:
                classification = self._get_router().classify(query)
                retrieval_duration.labels(stage="route_classify").observe(time.time() - t0)
                llm_call_total.labels(model=model_name, purpose="classify").inc()

                # 无关问题拦截
                if (
                    self._routing_config.get("irrelevant_detection", True)
                    and classification.is_irrelevant
                    and classification.confidence
                    >= self._routing_config.get("confidence_threshold", 0.6)
                ):
                    return IRRELEVANT_REPLY

                # 闲聊直接回答
                if (
                    self._routing_config.get("chat_detection", True)
                    and classification.is_chat
                    and classification.confidence
                    >= self._routing_config.get("confidence_threshold", 0.6)
                ):
                    with llm_call_duration.labels(model=model_name, purpose="direct_answer").time():
                        llm_call_total.labels(model=model_name, purpose="direct_answer").inc()
                        return self._direct_answer(query)
            except Exception as e:
                self._logger.warning(f"查询路由失败: {e}，回退到正常 RAG 流程")

        # ── 阶段 2：检索（可选改写）──
        t_retrieve = time.time()
        if enable_rewrite:
            strategy = self._resolve_strategy(rewrite_strategy)
            context_docx = self._retrieve_with_strategy(query, strategy)
        else:
            context_docx = self.retriever_docx(query)
        retrieval_duration.labels(stage="total").observe(time.time() - t_retrieve)

        # ── 阶段 3：生成 ──
        if not context_docx:
            return "根据现有资料无法回答该问题。"

        context = ""
        for i, doc in enumerate(context_docx, start=1):
            source = doc.metadata.get("source", "未知来源")
            source_name = (
                os.path.basename(source) if isinstance(source, str) else "未知来源"
            )
            context += f"[参考资料{i}] 来源：{source_name}\n{doc.page_content}\n\n"

        t_gen = time.time()
        llm_call_total.labels(model=model_name, purpose="generate").inc()
        result = self._get_chain().invoke(
            {"input": query, "context": context},
            config=_get_run_config(
                run_name="rag_generate",
                query=query,
                tags=["rag", "generate"],
                extra_metadata={"context_doc_count": len(context_docx)},
            ),
        )
        llm_call_duration.labels(model=model_name, purpose="generate").observe(
            time.time() - t_gen
        )
        # 估算 token 消耗（非精确，实际 token 数需从 LLM response 获取）
        token_est = len(context) // 2 + len(result) // 2
        llm_token_total.labels(model=model_name, type="input").inc(len(context) // 2)
        llm_token_total.labels(model=model_name, type="output").inc(len(result) // 2)
        llm_token_total.labels(model=model_name, type="total").inc(token_est)
        return result

    def get_summary_stream(
        self,
        query: str,
        enable_rewrite: bool = False,
        rewrite_strategy: str = "",
    ):
        """流式生成 RAG 回答（自动路由 + 可选改写）。"""
        if self._single_retrieval_enabled():
            ctx = self.prepare(query, enable_rewrite, rewrite_strategy)
            self._logger.info("request_summary", extra=ctx.summary())
            yield from self.stream_answer(ctx)
            return

        routing_enabled = self._routing_config.get("enabled", True)

        # ── 阶段 1：查询路由（自动）──
        if routing_enabled:
            try:
                classification = self._get_router().classify(query)

                if (
                    self._routing_config.get("irrelevant_detection", True)
                    and classification.is_irrelevant
                    and classification.confidence
                    >= self._routing_config.get("confidence_threshold", 0.6)
                ):
                    yield IRRELEVANT_REPLY
                    return

                if (
                    self._routing_config.get("chat_detection", True)
                    and classification.is_chat
                    and classification.confidence
                    >= self._routing_config.get("confidence_threshold", 0.6)
                ):
                    yield self._direct_answer(query)
                    return
            except Exception as e:
                self._logger.warning(f"查询路由失败: {e}，回退到正常 RAG 流程")

        # ── 阶段 2：检索（可选改写）──
        if enable_rewrite:
            strategy = self._resolve_strategy(rewrite_strategy)
            context_docx = self._retrieve_with_strategy(query, strategy)
        else:
            context_docx = self.retriever_docx(query)

        # ── 阶段 3：流式生成 ──
        if not context_docx:
            yield "根据现有资料无法回答该问题。"
            return

        context = ""
        for i, doc in enumerate(context_docx, start=1):
            source = doc.metadata.get("source", "未知来源")
            source_name = (
                os.path.basename(source) if isinstance(source, str) else "未知来源"
            )
            context += f"[参考资料{i}] 来源：{source_name}\n{doc.page_content}\n\n"

        for chunk in self._get_chain().stream(
            {"input": query, "context": context},
            config=_get_run_config(
                run_name="rag_generate_stream",
                query=query,
                tags=["rag", "stream"],
                extra_metadata={"context_doc_count": len(context_docx)},
            ),
        ):
            yield chunk


if __name__ == "__main__":
    rag_service = RagService()
    query = "新能源有哪些领域的发展前景比较好"
    print(f"问题: {query}")
    print(f"文档总数: {rag_service.vector_store.get_document_count()}")
    print(f"\n回答:\n{rag_service.get_summary(query)}")
