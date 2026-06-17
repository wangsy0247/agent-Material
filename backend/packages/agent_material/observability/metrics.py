"""Layer 3: Prometheus 指标 — Agent 系统级 (gateway + agent + mcp + memory)。"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry(auto_describe=True)

# ══════════════════════════════════════════════════════════════════
# 1. HTTP / Gateway 指标
# ══════════════════════════════════════════════════════════════════

http_request_total = Counter(
    "gateway_http_request_total", "HTTP 请求总数",
    ["method", "endpoint", "status_code"], registry=REGISTRY,
)
http_request_duration = Histogram(
    "gateway_http_request_duration_seconds", "HTTP 请求延迟",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY,
)
http_request_in_flight = Gauge(
    "gateway_http_request_in_flight", "当前处理中的 HTTP 请求数",
    ["method"], registry=REGISTRY,
)

# ══════════════════════════════════════════════════════════════════
# 2. Agent 推理指标
# ══════════════════════════════════════════════════════════════════

agent_invocation_total = Counter(
    "agent_invocation_total", "Agent 调用总数",
    ["mode", "status"], registry=REGISTRY,
)
# mode: stream / invoke / resume
# status: success / hittl_interrupt / error / max_iterations

agent_iteration_count = Histogram(
    "agent_iteration_count", "Agent 推理轮次分布",
    buckets=[1, 2, 3, 5, 7, 10, 15],
    registry=REGISTRY,
)
agent_duration = Histogram(
    "agent_duration_seconds", "Agent 总体执行时间",
    ["mode"], buckets=[0.5, 1, 2, 5, 10, 30, 60, 120, 300],
    registry=REGISTRY,
)

# ══════════════════════════════════════════════════════════════════
# 3. LangGraph 节点指标
# ══════════════════════════════════════════════════════════════════

graph_node_duration = Histogram(
    "agent_graph_node_duration_seconds", "LangGraph 各节点执行延迟",
    ["node"],
    buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    registry=REGISTRY,
)
# node: memory / summarization / agent / human_approval / tools / finalize

graph_node_iterations = Counter(
    "agent_graph_node_iterations_total", "LangGraph 节点执行次数",
    ["node"], registry=REGISTRY,
)

# ══════════════════════════════════════════════════════════════════
# 4. LLM 调用指标 (Agent 侧)
# ══════════════════════════════════════════════════════════════════

llm_call_total = Counter(
    "agent_llm_call_total", "LLM 调用总数",
    ["model", "purpose"], registry=REGISTRY,
)
# purpose: agent_reasoning / summarization / memory_extract / reflection

llm_call_duration = Histogram(
    "agent_llm_call_duration_seconds", "LLM 调用延迟",
    ["model", "purpose"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120],
    registry=REGISTRY,
)
llm_token_total = Counter(
    "agent_llm_token_total", "LLM Token 消耗",
    ["model", "type"], registry=REGISTRY,
)
llm_call_error_total = Counter(
    "agent_llm_call_error_total", "LLM 调用错误",
    ["model", "error_type"], registry=REGISTRY,
)

# ══════════════════════════════════════════════════════════════════
# 5. MCP 工具调用指标
# ══════════════════════════════════════════════════════════════════

mcp_tool_call_total = Counter(
    "mcp_tool_call_total", "MCP 工具调用总数",
    ["server", "tool", "status"], registry=REGISTRY,
)
mcp_tool_call_duration = Histogram(
    "mcp_tool_call_duration_seconds", "MCP 工具调用延迟",
    ["server", "tool"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60],
    registry=REGISTRY,
)
mcp_session_pool_size = Gauge(
    "mcp_session_pool_size", "MCP 会话池大小",
    ["server"], registry=REGISTRY,
)
mcp_connection_errors_total = Counter(
    "mcp_connection_errors_total", "MCP 连接错误次数",
    ["server"], registry=REGISTRY,
)

# ══════════════════════════════════════════════════════════════════
# 6. 记忆系统指标
# ══════════════════════════════════════════════════════════════════

memory_operation_duration = Histogram(
    "memory_operation_duration_seconds", "记忆操作延迟",
    ["operation"],
    buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5],
    registry=REGISTRY,
)
# operation: recall / insert / extract / build_context / compress / session_load

memory_session_count = Gauge(
    "memory_session_count", "活跃会话数",
    registry=REGISTRY,
)
memory_vector_count = Gauge(
    "memory_vector_count", "长期记忆向量数",
    registry=REGISTRY,
)
memory_store_size = Gauge(
    "memory_store_size_bytes", "SQLite 会话存储文件大小",
    registry=REGISTRY,
)

# ══════════════════════════════════════════════════════════════════
# 7. RAG 检索指标 (从 rag_mcp_server 合并)
# ══════════════════════════════════════════════════════════════════

retrieval_duration = Histogram(
    "rag_retrieval_duration_seconds", "检索各阶段延迟",
    ["stage"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
    registry=REGISTRY,
)
retrieval_docs = Histogram(
    "rag_retrieval_docs_count", "各阶段返回的文档数",
    ["stage"],
    buckets=[1, 3, 5, 10, 20, 50, 100],
    registry=REGISTRY,
)
retrieval_empty_total = Counter(
    "rag_retrieval_empty_total", "零结果检索次数",
    ["method"], registry=REGISTRY,
)
postgres_query_duration = Histogram(
    "rag_postgres_query_duration_seconds", "PostgreSQL 查询延迟",
    ["operation"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    registry=REGISTRY,
)

# RAG 请求指标 (原 rag_mcp_server server 层)
request_total = Counter(
    "rag_request_total", "RAG 请求总数",
    ["method", "status"], registry=REGISTRY,
)
request_duration = Histogram(
    "rag_request_duration_seconds", "RAG 请求总延迟",
    ["method"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY,
)
request_in_flight = Gauge(
    "rag_request_in_flight", "当前正在处理的 RAG 请求数",
    ["method"], registry=REGISTRY,
)

# RAG 任务队列指标
queue_depth = Gauge(
    "rag_queue_depth", "任务队列深度",
    ["status"], registry=REGISTRY,
)
queue_task_duration = Histogram(
    "rag_queue_task_duration_seconds", "单个后台任务处理耗时",
    ["type"],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0],
    registry=REGISTRY,
)
queue_worker_idle = Gauge(
    "rag_queue_worker_idle_seconds", "Worker 空闲时间",
    registry=REGISTRY,
)

# ══════════════════════════════════════════════════════════════════
# 9. 流式输出指标 (Agent SSE 流式)
# ══════════════════════════════════════════════════════════════════

stream_first_token_latency = Histogram(
    "agent_stream_first_token_latency_seconds", "首 token 延迟",
    ["mode"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10],
    registry=REGISTRY,
)
# mode: agent_chat / agent_resume

stream_total_chunks = Histogram(
    "agent_stream_total_chunks", "流式总 chunk 数",
    ["mode"], buckets=[1, 10, 50, 100, 250, 500, 1000],
    registry=REGISTRY,
)
stream_duration = Histogram(
    "agent_stream_duration_seconds", "流式总耗时",
    ["mode"], buckets=[1, 5, 10, 30, 60, 120, 300],
    registry=REGISTRY,
)

# ══════════════════════════════════════════════════════════════════
# 10. 辅助工具
# ══════════════════════════════════════════════════════════════════


class StreamMetrics:
    """Agent SSE 流式输出度量收集器。"""

    def __init__(self, mode: str = "agent_chat"):
        self.mode = mode
        self._start_time: float | None = None
        self._first_token_time: float | None = None
        self._last_chunk_time: float | None = None
        self._chunk_count = 0

    @property
    def first_token_latency(self) -> float:
        if self._start_time and self._first_token_time:
            return self._first_token_time - self._start_time
        return 0.0

    async def awrap(self, stream):
        """包装异步生成器，收集首 token 延迟、chunk 数、总耗时等指标。"""
        self._start_time = time.time()
        async for chunk in stream:
            now = time.time()
            if self._first_token_time is None:
                self._first_token_time = now
            self._chunk_count += 1
            self._last_chunk_time = now
            yield chunk
        # 上报
        if self._first_token_time:
            stream_first_token_latency.labels(mode=self.mode).observe(self.first_token_latency)
            stream_total_chunks.labels(mode=self.mode).observe(self._chunk_count)
        if self._start_time and self._last_chunk_time:
            stream_duration.labels(mode=self.mode).observe(self._last_chunk_time - self._start_time)


@contextmanager
def track_stage(stage: str, doc_count: int = 0):
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        # 复用 retrieval 类的 stage 概念，也可扩展
        from agent_material.observability.metrics import REGISTRY
        for m in REGISTRY.collect():
            for s in m.samples:
                if s.name == "agent_stage_duration_seconds" and s.labels.get("stage") == stage:
                    return


def get_metrics() -> bytes:
    return generate_latest(REGISTRY)
