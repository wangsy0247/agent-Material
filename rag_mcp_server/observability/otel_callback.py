"""Layer 2b: LangChain Callback → OpenTelemetry Span 桥接。

将 LangChain 组件的生命周期回调（on_llm_start / on_chain_start
/ on_retriever_start / on_tool_start）转换为 OTel Span。

使用方式：
  from rag_mcp_server.observability.otel_callback import OpenTelemetryCallback
  callback = OpenTelemetryCallback()
  chain.invoke(input, config={"callbacks": [callback]})

注意：与 LangSmith callback 并行运行，互不冲突。
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish


class OpenTelemetryCallback(BaseCallbackHandler):
    """将 LangChain 组件回调桥接到 OpenTelemetry Span。

    覆盖的节点：
    - on_llm_start/end        → Span "llm.call"           (LLM 调用)
    - on_chain_start/end      → Span "chain.execute"      (Chain 执行)
    - on_retriever_start/end  → Span "retriever.search"   (检索器调用)
    - on_tool_start/end       → Span "tool.execute"       (工具调用)
    - on_agent_action/on_agent_finish → Span "agent.action" / "agent.finish"
    """

    def __init__(self, tracer=None):
        super().__init__()
        if tracer is None:
            from opentelemetry import trace
            tracer = trace.get_tracer("langchain")
        self._tracer = tracer
        self._spans: dict[UUID, Any] = {}  # run_id → OTel Span
        self._start_times: dict[UUID, float] = {}  # run_id → start time

    # ── LLM ───────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        span = self._tracer.start_span(
            "llm.call",
            attributes={
                "llm.model": serialized.get("kwargs", {}).get("model_name")
                or serialized.get("name", "unknown"),
                "llm.prompts.count": len(prompts),
                "llm.prompts.total_chars": sum(len(p) for p in prompts),
                "langchain.run_id": str(run_id),
            },
        )
        if tags:
            span.set_attribute("langchain.tags", ",".join(tags))
        self._spans[run_id] = span
        self._start_times[run_id] = time.time()

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        span = self._spans.pop(run_id, None)
        start = self._start_times.pop(run_id, None)
        if span is None:
            return
        if start:
            span.set_attribute("duration_ms", (time.time() - start) * 1000)

        # 提取 token 用量（兼容 OpenAI / Anthropic 格式）
        try:
            if hasattr(response, "llm_output") and response.llm_output:
                usage = response.llm_output.get("token_usage", {})
                span.set_attribute("llm.token.input", usage.get("prompt_tokens", 0))
                span.set_attribute("llm.token.output", usage.get("completion_tokens", 0))
                span.set_attribute("llm.token.total", usage.get("total_tokens", 0))
        except Exception:
            pass

        # 提取 finish_reason
        try:
            generations = getattr(response, "generations", [[]])
            if generations and generations[0]:
                gen = generations[0][0]
                if hasattr(gen, "generation_info"):
                    finish = gen.generation_info.get("finish_reason")
                    if finish:
                        span.set_attribute("llm.finish_reason", str(finish))
        except Exception:
            pass

        span.end()

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        span = self._spans.pop(run_id, None)
        self._start_times.pop(run_id, None)
        if span:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(error)[:500])
            span.record_exception(error)
            span.end()

    # ── Retriever ─────────────────────────────────────────────

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        span = self._tracer.start_span(
            "retriever.search",
            attributes={
                "retriever.name": serialized.get("name", "unknown"),
                "retriever.query": query[:200],
                "langchain.run_id": str(run_id),
            },
        )
        if tags:
            span.set_attribute("langchain.tags", ",".join(tags))
        self._spans[run_id] = span
        self._start_times[run_id] = time.time()

    def on_retriever_end(
        self,
        documents: list[Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        span = self._spans.pop(run_id, None)
        start = self._start_times.pop(run_id, None)
        if span is None:
            return
        if start:
            span.set_attribute("duration_ms", (time.time() - start) * 1000)
        span.set_attribute("retriever.docs_count", len(documents))
        span.end()

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        span = self._spans.pop(run_id, None)
        self._start_times.pop(run_id, None)
        if span:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(error)[:500])
            span.end()

    # ── Chain ─────────────────────────────────────────────────

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        chain_name = serialized.get("name", serialized.get("id", "unknown"))
        span = self._tracer.start_span(
            f"chain.{chain_name}",
            attributes={
                "chain.name": chain_name,
                "langchain.run_id": str(run_id),
            },
        )
        if tags:
            span.set_attribute("langchain.tags", ",".join(tags))
        self._spans[run_id] = span
        self._start_times[run_id] = time.time()

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        span = self._spans.pop(run_id, None)
        start = self._start_times.pop(run_id, None)
        if span is None:
            return
        if start:
            span.set_attribute("duration_ms", (time.time() - start) * 1000)
        span.end()

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        span = self._spans.pop(run_id, None)
        self._start_times.pop(run_id, None)
        if span:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(error)[:500])
            span.end()

    # ── Tool ──────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown")
        span = self._tracer.start_span(
            f"tool.{tool_name}",
            attributes={
                "tool.name": tool_name,
                "langchain.run_id": str(run_id),
            },
        )
        if tags:
            span.set_attribute("langchain.tags", ",".join(tags))
        self._spans[run_id] = span
        self._start_times[run_id] = time.time()

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        span = self._spans.pop(run_id, None)
        start = self._start_times.pop(run_id, None)
        if span is None:
            return
        if start:
            span.set_attribute("duration_ms", (time.time() - start) * 1000)
        output_str = str(output) if output is not None else ""
        span.set_attribute("tool.output_length", len(output_str))
        span.end()

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        span = self._spans.pop(run_id, None)
        self._start_times.pop(run_id, None)
        if span:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(error)[:500])
            span.end()

    # ── Agent ─────────────────────────────────────────────────

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        span = self._tracer.start_span(
            "agent.action",
            attributes={
                "agent.tool": action.tool,
                "agent.tool_input": str(action.tool_input)[:200],
                "langchain.run_id": str(run_id),
            },
        )
        self._spans[run_id] = span
        self._start_times[run_id] = time.time()

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        span = self._spans.pop(run_id, None)
        start = self._start_times.pop(run_id, None)
        if span:
            if start:
                span.set_attribute("duration_ms", (time.time() - start) * 1000)
            span.end()
