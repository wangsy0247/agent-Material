"""Layer 3: Prometheus 指标暴露 + Grafana Dashboard 定义。

指标分类：
  RED (Rate/Error/Duration)      — 请求级
  检索阶段                        — vector / bm25 / rerank / parent_fetch
  LLM 专项                       — token / 延迟 / 流式输出
  外部 API                       — DashScope embedding / rerank
  存储                           — PostgreSQL
  任务队列                       — pending / processing / failed
  流式输出                       — 首 token 延迟、token 生成速率、chunk 间隔

使用方式：
  from rag_mcp_server.observability.metrics import (
      request_duration, retrieval_duration, llm_token_total, ...
  )
  with request_duration.labels(method="rag_search").time():
      ...
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# ── 全局 Registry ────────────────────────────────────────────────
REGISTRY = CollectorRegistry(auto_describe=True)

# ══════════════════════════════════════════════════════════════════
# 1. 请求级 RED 指标
# ══════════════════════════════════════════════════════════════════

request_total = Counter(
    "rag_request_total",
    "RAG MCP 请求总数",
    ["method", "status"],
    registry=REGISTRY,
)

request_duration = Histogram(
    "rag_request_duration_seconds",
    "RAG 请求总延迟（含所有子阶段）",
    ["method"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY,
)

request_in_flight = Gauge(
    "rag_request_in_flight",
    "当前正在处理的请求数",
    ["method"],
    registry=REGISTRY,
)

# ══════════════════════════════════════════════════════════════════
# 2. 检索阶段指标
# ══════════════════════════════════════════════════════════════════

retrieval_duration = Histogram(
    "rag_retrieval_duration_seconds",
    "检索各阶段延迟",
    ["stage"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
    registry=REGISTRY,
)
# stage 标签值:
#   vector_search  — PGVector 向量检索
#   bm25_search    — BM25 关键词检索
#   rrf_fusion     — RRF 融合
#   topk_truncate  — TopK 截断
#   rerank         — 重排序 (DashScope API / 本地 CrossEncoder)
#   parent_fetch   — 多粒度父 chunk 查询

retrieval_docs = Histogram(
    "rag_retrieval_docs_count",
    "各阶段返回的文档数",
    ["stage"],
    buckets=[1, 3, 5, 10, 20, 50, 100],
    registry=REGISTRY,
)

retrieval_empty_total = Counter(
    "rag_retrieval_empty_total",
    "零结果检索次数",
    ["method"],
    registry=REGISTRY,
)

# ══════════════════════════════════════════════════════════════════
# 3. LLM 专项指标
# ══════════════════════════════════════════════════════════════════

llm_call_total = Counter(
    "rag_llm_call_total",
    "LLM 调用总数",
    ["model", "purpose"],
    registry=REGISTRY,
)
# purpose: rewrite / generate / classify / direct_answer / hyde

llm_call_duration = Histogram(
    "rag_llm_call_duration_seconds",
    "LLM 调用延迟",
    ["model", "purpose"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
    registry=REGISTRY,
)

llm_token_total = Counter(
    "rag_llm_token_total",
    "LLM Token 消耗",
    ["model", "type"],
    registry=REGISTRY,
)
# type: input / output / total

llm_call_error_total = Counter(
    "rag_llm_call_error_total",
    "LLM 调用错误次数",
    ["model", "error_type"],
    registry=REGISTRY,
)
# error_type: timeout / rate_limit / api_error / unknown

# ── 流式输出专项 ─────────────────────────────────────────────────

llm_stream_first_token_latency = Histogram(
    "rag_llm_stream_first_token_latency_seconds",
    "首 token 延迟（从请求到第一个 chunk）",
    ["model", "purpose"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
    registry=REGISTRY,
)

llm_stream_chunk_count = Histogram(
    "rag_llm_stream_chunk_count",
    "流式响应总 chunk 数",
    ["model", "purpose"],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500],
    registry=REGISTRY,
)

llm_stream_tokens_per_second = Histogram(
    "rag_llm_stream_tokens_per_second",
    "流式输出 Token 生成速率",
    ["model", "purpose"],
    buckets=[5, 10, 20, 30, 50, 80, 100, 150],
    registry=REGISTRY,
)

llm_stream_inter_chunk_latency = Histogram(
    "rag_llm_stream_inter_chunk_latency_seconds",
    "流式输出相邻 chunk 间隔（过大表示网络/模型卡顿）",
    ["model"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=REGISTRY,
)

# ══════════════════════════════════════════════════════════════════
# 4. 外部 API 指标
# ══════════════════════════════════════════════════════════════════

dashscope_api_total = Counter(
    "dashscope_api_total",
    "DashScope API 调用总数",
    ["api", "status"],
    registry=REGISTRY,
)
# api: embedding / rerank
# status: success / rate_limited / error

dashscope_api_duration = Histogram(
    "dashscope_api_duration_seconds",
    "DashScope API 调用延迟",
    ["api"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    registry=REGISTRY,
)

dashscope_api_retries = Counter(
    "dashscope_api_retries_total",
    "DashScope API 重试次数",
    ["api"],
    registry=REGISTRY,
)

# ══════════════════════════════════════════════════════════════════
# 5. 存储指标
# ══════════════════════════════════════════════════════════════════

postgres_query_duration = Histogram(
    "rag_postgres_query_duration_seconds",
    "PostgreSQL 查询延迟",
    ["operation"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    registry=REGISTRY,
)
# operation: vector_search / parent_fetch / checkpointer_read / checkpointer_write

postgres_connection_pool = Gauge(
    "rag_postgres_connection_pool_size",
    "PostgreSQL 连接池状态（需 psycopg pool 集成）",
    ["state"],
    registry=REGISTRY,
)
# state: active / idle / overflow

# ══════════════════════════════════════════════════════════════════
# 6. 任务队列指标
# ══════════════════════════════════════════════════════════════════

queue_depth = Gauge(
    "rag_queue_depth",
    "任务队列深度",
    ["status"],
    registry=REGISTRY,
)
# status: pending / processing / failed / completed

queue_task_duration = Histogram(
    "rag_queue_task_duration_seconds",
    "单个后台任务处理耗时",
    ["type"],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0],
    registry=REGISTRY,
)
# type: ingest / delete / reindex

queue_worker_idle = Gauge(
    "rag_queue_worker_idle_seconds",
    "Worker 空闲时间（距离上次处理任务的秒数）",
    registry=REGISTRY,
)

# ══════════════════════════════════════════════════════════════════
# 7. 流式输出度量工具
# ══════════════════════════════════════════════════════════════════


class StreamMetrics:
    """流式输出度量收集器。

    使用方式:
      sm = StreamMetrics(model="qwen3.6-plus", purpose="rag_generate")
      for chunk, metrics in sm.wrap(stream_iterator):
          yield chunk
      # 读取最终指标: sm.first_token_latency, sm.total_chunks, sm.tokens_per_second
    """

    def __init__(self, model: str = "unknown", purpose: str = "generate"):
        self.model = model
        self.purpose = purpose
        self._start_time: float | None = None
        self._first_token_time: float | None = None
        self._last_chunk_time: float | None = None
        self._chunk_count = 0
        self._total_chars = 0

    @property
    def first_token_latency(self) -> float:
        if self._start_time and self._first_token_time:
            return self._first_token_time - self._start_time
        return 0.0

    @property
    def total_chunks(self) -> int:
        return self._chunk_count

    @property
    def tokens_per_second(self) -> float:
        if self._first_token_time and self._last_chunk_time:
            elapsed = self._last_chunk_time - self._first_token_time
            if elapsed > 0:
                return self._chunk_count / elapsed
        return 0.0

    @property
    def avg_inter_chunk_latency(self) -> float:
        if self._chunk_count > 1 and self._first_token_time and self._last_chunk_time:
            return (self._last_chunk_time - self._first_token_time) / (self._chunk_count - 1)
        return 0.0

    def wrap(self, stream) -> Generator[Any, None, None]:
        """包装流式迭代器，自动收集度量并上报 Prometheus。"""
        self._start_time = time.time()
        prev_time = self._start_time

        for chunk in stream:
            now = time.time()
            if self._first_token_time is None:
                self._first_token_time = now
            else:
                inter = now - prev_time
                if inter > 0.001:
                    llm_stream_inter_chunk_latency.labels(model=self.model).observe(inter)

            self._chunk_count += 1
            if isinstance(chunk, str):
                self._total_chars += len(chunk)
            prev_time = now
            self._last_chunk_time = now
            yield chunk

        # 上报最终指标
        if self._first_token_time:
            llm_stream_first_token_latency.labels(
                model=self.model, purpose=self.purpose
            ).observe(self.first_token_latency)

            llm_stream_chunk_count.labels(
                model=self.model, purpose=self.purpose
            ).observe(self._chunk_count)

            tps = self.tokens_per_second
            if tps > 0:
                llm_stream_tokens_per_second.labels(
                    model=self.model, purpose=self.purpose
                ).observe(tps)


# ══════════════════════════════════════════════════════════════════
# 8. 辅助工具 + /metrics 端点
# ══════════════════════════════════════════════════════════════════


@contextmanager
def track_stage(stage: str, doc_count: int = 0):
    """上下文管理器：自动记录检索阶段延迟 + 返回文档数。

    with track_stage("vector_search", doc_count=len(docs)):
        docs = vector_store.similarity_search(query)
    """
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        retrieval_duration.labels(stage=stage).observe(elapsed)
        if doc_count > 0:
            retrieval_docs.labels(stage=stage).observe(doc_count)


def track_llm_call(model: str, purpose: str, status: str = "success"):
    """记录 LLM 调用（Count + Duration 装饰器/上下文管理器）。"""
    return _LLMTracker(model, purpose, status)


class _LLMTracker:
    """LLM 调用追踪上下文管理器。"""

    def __init__(self, model: str, purpose: str, status: str = "success"):
        self.model = model
        self.purpose = purpose
        self.status = status
        self._start = 0.0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self._start
        if exc_type:
            llm_call_total.labels(model=self.model, purpose=self.purpose).inc()
            error_type = _classify_error(exc_val)
            llm_call_error_total.labels(model=self.model, error_type=error_type).inc()
        else:
            llm_call_total.labels(model=self.model, purpose=self.purpose).inc()
            llm_call_duration.labels(model=self.model, purpose=self.purpose).observe(elapsed)
        return False  # 不吞异常

    def record_tokens(self, input_tokens: int = 0, output_tokens: int = 0):
        if input_tokens:
            llm_token_total.labels(model=self.model, type="input").inc(input_tokens)
        if output_tokens:
            llm_token_total.labels(model=self.model, type="output").inc(output_tokens)
        if input_tokens or output_tokens:
            llm_token_total.labels(model=self.model, type="total").inc(input_tokens + output_tokens)

    def time(self):
        """装饰器用法: @track_llm_call(...).time()"""
        def decorator(fn):
            def wrapper(*args, **kwargs):
                with self:
                    return fn(*args, **kwargs)
            return wrapper
        return decorator


def _classify_error(exc) -> str:
    msg = str(exc).lower()
    if "timeout" in msg:
        return "timeout"
    if "rate" in msg or "429" in msg or "limit" in msg:
        return "rate_limit"
    if "4" in msg[:3] or "5" in msg[:3]:
        return "api_error"
    return "unknown"


def get_metrics() -> bytes:
    """生成 Prometheus 格式的指标文本。"""
    return generate_latest(REGISTRY)


# ══════════════════════════════════════════════════════════════════
# 9. Grafana Dashboard JSON（预置仪表板）
# ══════════════════════════════════════════════════════════════════

GRAFANA_DASHBOARD = {
    "title": "RAG MCP Server - 可观测性仪表板",
    "tags": ["rag", "mcp", "llm"],
    "refresh": "10s",
    "panels": [
        # ── Row 1: 请求概览 ──
        {
            "title": "请求 QPS",
            "targets": [{"expr": "rate(rag_request_total[1m])"}],
            "type": "graph",
        },
        {
            "title": "请求延迟 P50/P95/P99",
            "targets": [
                {"expr": "histogram_quantile(0.50, rate(rag_request_duration_seconds_bucket[5m]))", "legendFormat": "P50"},
                {"expr": "histogram_quantile(0.95, rate(rag_request_duration_seconds_bucket[5m]))", "legendFormat": "P95"},
                {"expr": "histogram_quantile(0.99, rate(rag_request_duration_seconds_bucket[5m]))", "legendFormat": "P99"},
            ],
            "type": "graph",
        },
        {
            "title": "请求错误率",
            "targets": [
                {"expr": "sum(rate(rag_request_total{status='error'}[1m])) / sum(rate(rag_request_total[1m]))"}
            ],
            "type": "stat",
        },
        {
            "title": "当前并发请求",
            "targets": [{"expr": "rag_request_in_flight"}],
            "type": "stat",
        },
        # ── Row 2: 检索链路分解 ──
        {
            "title": "检索各阶段延迟分布",
            "targets": [
                {"expr": "rate(rag_retrieval_duration_seconds_sum[5m]) / rate(rag_retrieval_duration_seconds_count[5m])",
                 "legendFormat": "{{stage}}"},
            ],
            "type": "bargauge",
        },
        {
            "title": "检索各阶段返回文档数",
            "targets": [
                {"expr": "rate(rag_retrieval_docs_count_sum[5m]) / rate(rag_retrieval_docs_count_count[5m])",
                 "legendFormat": "{{stage}}"},
            ],
            "type": "bargauge",
        },
        {
            "title": "零结果检索次数",
            "targets": [{"expr": "rate(rag_retrieval_empty_total[5m])"}],
            "type": "graph",
        },
        # ── Row 3: LLM 调用 ──
        {
            "title": "LLM 调用 QPS (by purpose)",
            "targets": [{"expr": "rate(rag_llm_call_total[1m])"}],
            "type": "graph",
        },
        {
            "title": "LLM 延迟 (by purpose)",
            "targets": [
                {"expr": "histogram_quantile(0.95, rate(rag_llm_call_duration_seconds_bucket[5m]))",
                 "legendFormat": "{{purpose}}"},
            ],
            "type": "graph",
        },
        {
            "title": "LLM Token 消耗速率",
            "targets": [
                {"expr": "rate(rag_llm_token_total{type='input'}[5m])", "legendFormat": "input/s"},
                {"expr": "rate(rag_llm_token_total{type='output'}[5m])", "legendFormat": "output/s"},
            ],
            "type": "graph",
        },
        {
            "title": "LLM 调用错误率",
            "targets": [
                {"expr": "sum(rate(rag_llm_call_error_total[1m])) / sum(rate(rag_llm_call_total[1m]))"}
            ],
            "type": "stat",
        },
        # ── Row 4: 流式输出 ──
        {
            "title": "流式首 Token 延迟 P50/P95",
            "targets": [
                {"expr": "histogram_quantile(0.50, rate(rag_llm_stream_first_token_latency_seconds_bucket[5m]))",
                 "legendFormat": "P50 {{purpose}}"},
                {"expr": "histogram_quantile(0.95, rate(rag_llm_stream_first_token_latency_seconds_bucket[5m]))",
                 "legendFormat": "P95 {{purpose}}"},
            ],
            "type": "graph",
        },
        {
            "title": "流式 Token 生成速率",
            "targets": [
                {"expr": "rate(rag_llm_stream_tokens_per_second_sum[5m]) / rate(rag_llm_stream_tokens_per_second_count[5m])"}
            ],
            "type": "graph",
        },
        {
            "title": "流式 Chunk 间隔 P95",
            "targets": [
                {"expr": "histogram_quantile(0.95, rate(rag_llm_stream_inter_chunk_latency_seconds_bucket[5m]))"}
            ],
            "type": "graph",
        },
        # ── Row 5: 外部 API ──
        {
            "title": "DashScope API 延迟 (embedding vs rerank)",
            "targets": [
                {"expr": "histogram_quantile(0.95, rate(dashscope_api_duration_seconds_bucket[5m]))",
                 "legendFormat": "{{api}}"},
            ],
            "type": "graph",
        },
        {
            "title": "DashScope API 限流次数",
            "targets": [{"expr": "rate(dashscope_api_total{status='rate_limited'}[5m])"}],
            "type": "graph",
        },
        {
            "title": "DashScope API 重试次数",
            "targets": [{"expr": "rate(dashscope_api_retries_total[5m])"}],
            "type": "graph",
        },
        # ── Row 6: 任务队列 ──
        {
            "title": "任务队列深度",
            "targets": [{"expr": "rag_queue_depth"}],
            "type": "bargauge",
        },
        {
            "title": "任务处理耗时 (by type)",
            "targets": [
                {"expr": "histogram_quantile(0.95, rate(rag_queue_task_duration_seconds_bucket[5m]))",
                 "legendFormat": "{{type}}"},
            ],
            "type": "graph",
        },
    ],
}
