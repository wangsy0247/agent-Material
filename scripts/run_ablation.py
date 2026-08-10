#!/usr/bin/env python3
"""离线检索消融运行器（Spec 10.5 / 阶段 3）。

对固定评测集依次运行 6 种检索配置，只做检索级指标，不调用 LLM 生成答案：

1. dense_only            Dense only（bm25_weight=0，RRF 中 BM25 路权重置零）
2. bm25_only             BM25 only（vector_weight=0，RRF 中向量路权重置零）
3. dense_bm25_rrf        Dense + BM25 RRF（cross_query_rrf 开、global_rerank 关、
                         父块回溯关）
4. hybrid_global_rerank  Hybrid + Global Rerank（父块回溯关）
5. hybrid_rerank_parent  Hybrid + Rerank + Parent-child（父块回溯开）
6. full_rewrite          完整链路 + Step-back/Expand（enable_rewrite=True）

实现要点：
- 全程复用同一个 RagService 实例；每种配置通过原地修改模块级
  ``vector_store_config`` dict 切换开关（管道每请求现读该 dict，不改 yaml 文件）。
- 父块回溯开关没有对应配置项，通过对 RetrievalPipeline 实例方法做运行时
  monkeypatch（实例属性遮蔽类方法）实现，同样不触碰源码文件。
- 单路消融（dense_only / bm25_only）：RetrievalPipeline.retrieve_branches
  无论权重是否为零都会同时执行向量与 BM25 两路召回，权重只影响 RRF 排序，
  不影响候选池——这导致两种配置 top10 候选集合完全相同（2026-08 探针实证，
  3 道题 fused top10 集合一致、仅顺序不同），Recall@10 在并集池上饱和。
  为使"单路"语义真实生效，ProfileSession 在对应权重为 0 时对
  MultiGranularityRetriever.search_children 做实例级包装，直接从候选池中
  剔除该路结果（不改 backend 源码）。
- 为保证 6 种配置的检索行为可比，运行期间统一关闭 query_routing（路由是检索
  上游的独立阶段，且每次 classify 都是一次 LLM 调用）。"不可回答/无关"题的
  线上拦截由路由负责，这里改为统计检索噪声（见下）。
- 健壮性：每完成一个配置即增量覆盖写入结果 JSON（中断不丢已完成配置）；
  日志写入 logs/ablation_YYYYMMDD_HHMM.log；支持 --only/--skip 断点续跑，
  已有结果文件中未重跑的配置会保留并入最终输出。

指标（无需 gold chunk id，用语义命中近似）：
- 从 ground_truth 提取关键词：长度 ≥2 的英文/数字/化学式 token（如 nscf、
  out_dos、DOS1_smearing.dat、Si）+ 中文连续段落的全部 2-gram。
- 排名列表 = reranked_children（全局 Rerank 或 RRF 前 top_n）+ fused_children
  中未出现过的候选（按 RRF 顺序补足），截取 top10 计算指标。reranker.top_n=3
  时 reranked_children 只有 3 条，因此用 fused 余量补足 top5/top10 评估。
- Recall@5 / Recall@10：ground_truth 关键词在 top5/top10 拼接文本中的命中率，
  按题平均（unanswerable 题不参与）。
- MRR：top10 内首个包含任一关键词的块排名倒数，按题平均（同上）。
- nDCG@10：块增益 = min(命中关键词数, 3) / 3，理想排序为增益降序。
- 延迟：每题 prepare() 墙钟耗时与 ctx.stage_durations 的 p50/p95。
- unanswerable 题单独统计"检索噪声"：问题关键词（从 question 提取）在 top10
  文本中的命中率，理想情况下应显著低于可回答题的 Recall。

用法：
    export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
    python scripts/run_ablation.py [--limit N] [--configs name1,name2]
                                   [--only name1,name2] [--skip name3]

结果增量写入 evaluation_data/ablation_results.json，日志写入
logs/ablation_YYYYMMDD_HHMM.log，并在 stdout 打印对比表。
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import re
import statistics
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "packages"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from agent_material.rag_engine.rag_service import RagService  # noqa: E402
from agent_material.utils.config_handler import vector_store_config  # noqa: E402

DEFAULT_DATASET = ROOT / "evaluation_data" / "fixed_eval_dataset.json"
DEFAULT_OUTPUT = ROOT / "evaluation_data" / "ablation_results.json"
LOG_DIR = ROOT / "logs"

log = logging.getLogger("ablation")


def setup_logging() -> Path:
    """日志同时写控制台与 logs/ablation_YYYYMMDD_HHMM.log。"""
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"ablation_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
    log.setLevel(logging.INFO)
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log_path

# ── 消融配置定义 ──────────────────────────────────────────────
# 每个 profile 描述对 vector_store_config 的原地修改 + 父块回溯/改写开关。
PROFILES: list[dict] = [
    {
        "name": "dense_only",
        "description": "Dense only（RRF 中 BM25 路权重置 0）",
        "overrides": {
            "vector_weight": 0.5,
            "bm25_weight": 0.0,
            "feature_flags": {"cross_query_rrf": True, "global_rerank": False},
        },
        "parent_backtrack": False,
        "enable_rewrite": False,
    },
    {
        "name": "bm25_only",
        "description": "BM25 only（RRF 中向量路权重置 0）",
        "overrides": {
            "vector_weight": 0.0,
            "bm25_weight": 0.5,
            "feature_flags": {"cross_query_rrf": True, "global_rerank": False},
        },
        "parent_backtrack": False,
        "enable_rewrite": False,
    },
    {
        "name": "dense_bm25_rrf",
        "description": "Dense + BM25 RRF（无全局 Rerank、无父块回溯）",
        "overrides": {
            "vector_weight": 0.5,
            "bm25_weight": 0.5,
            "feature_flags": {"cross_query_rrf": True, "global_rerank": False},
        },
        "parent_backtrack": False,
        "enable_rewrite": False,
    },
    {
        "name": "hybrid_global_rerank",
        "description": "Hybrid + Global Rerank（无父块回溯）",
        "overrides": {
            "vector_weight": 0.5,
            "bm25_weight": 0.5,
            "feature_flags": {"cross_query_rrf": True, "global_rerank": True},
        },
        "parent_backtrack": False,
        "enable_rewrite": False,
    },
    {
        "name": "hybrid_rerank_parent",
        "description": "Hybrid + Rerank + Parent-child（父块回溯开）",
        "overrides": {
            "vector_weight": 0.5,
            "bm25_weight": 0.5,
            "feature_flags": {"cross_query_rrf": True, "global_rerank": True},
        },
        "parent_backtrack": True,
        "enable_rewrite": False,
    },
    {
        "name": "full_rewrite",
        "description": "完整链路 + Step-back/Expand 改写",
        "overrides": {
            "vector_weight": 0.5,
            "bm25_weight": 0.5,
            "feature_flags": {"cross_query_rrf": True, "global_rerank": True},
        },
        "parent_backtrack": True,
        "enable_rewrite": True,
        "rewrite_strategy": "step_back+expand",
    },
]

# ── 关键词提取（ground_truth 语义命中近似）────────────────────

# 英文/数字/化学式 token：字母开头，可含数字、下划线、点、连字符（≥2 字符），
# 或纯数字（含小数/科学计数）。
_TOKEN_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9_.\-]*|\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
)
_CJK_RUN_RE = re.compile(r"[一-鿿]+")

# 无信息中文 2-gram 过滤（出现在套话中，不代表内容命中）
_STOP_BIGRAMS = {
    "根据", "现有", "资料", "无法", "回答", "问题", "需要", "可以",
    "进行", "设置", "使用", "计算", "文件", "参数", "输出", "输入",
}


def extract_keywords(text: str) -> list[str]:
    """从文本提取关键词集合：英文/数字 token + 中文 2-gram。

    简单实现，只用于检索命中的近似评估，不做精确语义匹配。
    """
    keywords: set[str] = set()
    for tok in _TOKEN_RE.findall(text):
        tok = tok.strip("._-")
        if len(tok) >= 2 or tok.replace(".", "").isdigit():
            keywords.add(tok.lower())
    for run in _CJK_RUN_RE.findall(text):
        if len(run) == 2:
            keywords.add(run)
        else:
            for i in range(len(run) - 1):
                gram = run[i : i + 2]
                if gram not in _STOP_BIGRAMS:
                    keywords.add(gram)
    return sorted(keywords)


# ── 检索级指标 ────────────────────────────────────────────────


def _doc_key(doc) -> str:
    chunk_id = doc.metadata.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    return f"content:{hash(doc.page_content[:200])}"


def build_ranked_texts(ctx, limit: int = 10) -> list[str]:
    """构造评估用排名列表：reranked 优先，fused 余量补足到 limit 条。"""
    ranked = list(ctx.reranked_children)
    seen = {_doc_key(d) for d in ranked}
    for doc in ctx.fused_children:
        if len(ranked) >= limit:
            break
        key = _doc_key(doc)
        if key not in seen:
            seen.add(key)
            ranked.append(doc)
    return [d.page_content for d in ranked[:limit]]


def compute_metrics(keywords: list[str], texts: list[str]):
    """对单题计算 Recall@5/10、RR、nDCG@10（基于关键词命中）。"""
    if not keywords or not texts:
        return {"recall_at_5": 0.0, "recall_at_10": 0.0, "rr": 0.0, "ndcg_at_10": 0.0}
    lowered = [t.lower() for t in texts]
    hits = [sum(1 for kw in keywords if kw in t) for t in lowered]

    found5 = {kw for kw in keywords if any(kw in t for t in lowered[:5])}
    found10 = {kw for kw in keywords if any(kw in t for t in lowered[:10])}
    total = len(keywords)
    recall5 = len(found5) / total
    recall10 = len(found10) / total

    rr = 0.0
    for rank, h in enumerate(hits, start=1):
        if h > 0:
            rr = 1.0 / rank
            break

    gains = [min(h, 3) / 3.0 for h in hits]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "recall_at_5": round(recall5, 4),
        "recall_at_10": round(recall10, 4),
        "rr": round(rr, 4),
        "ndcg_at_10": round(ndcg, 4),
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1)
    return round(ordered[max(idx, 0)], 2)


# ── 配置切换 ──────────────────────────────────────────────────


def _resolve_parents_noop(self, ctx):
    """父块回溯关闭时的替代实现：直接使用 reranked 子块。"""
    t0 = time.time()
    ctx.resolved_parents = list(ctx.reranked_children)
    ctx.record_duration("resolve_parents", t0)


def _make_search_children_wrapper(original, drop_vector: bool, drop_bm25: bool):
    """单路消融：从候选池中剔除权重为 0 的一路召回结果。

    retrieve_branches 无论权重是否为零都会同时执行两路召回（权重只影响
    RRF 排序），导致 dense_only / bm25_only 的候选池完全相同。这里在
    search_children 层直接丢弃对应分支，使单路语义真实生效。
    """

    # 注意：该函数写入实例 __dict__（非描述符），调用时不会自动绑定 self，
    # 因此签名不含 self；original 是已绑定方法。
    def wrapper(query, k=None, hyde_embedding=None):
        vector_docs, bm25_docs = original(query, k=k, hyde_embedding=hyde_embedding)
        if drop_vector:
            vector_docs = []
        if drop_bm25:
            bm25_docs = []
        return vector_docs, bm25_docs

    return wrapper


class ProfileSession:
    """应用/恢复一个消融 profile（原地改 config dict + monkeypatch 父块回溯
    + 单路消融时包装 search_children 剔除零权重分支）。"""

    def __init__(self, service: RagService, profile: dict):
        self._service = service
        self._profile = profile
        self._backup: dict = {}
        self._retriever = None

    def __enter__(self):
        cfg = vector_store_config
        overrides = self._profile["overrides"]
        for key, value in overrides.items():
            if isinstance(value, dict):
                self._backup[key] = copy.deepcopy(cfg.get(key, {}))
                cfg.setdefault(key, {}).update(value)
            else:
                self._backup[key] = cfg.get(key)
                cfg[key] = value

        # 父块回溯 monkeypatch（实例属性遮蔽类方法，不修改源码）
        pipeline = self._service._get_pipeline()
        if not self._profile["parent_backtrack"]:
            pipeline.resolve_parents = types.MethodType(
                _resolve_parents_noop, pipeline
            )
        else:
            pipeline.__dict__.pop("resolve_parents", None)

        # 单路消融：权重为 0 的分支直接从候选池剔除（实例级包装，可恢复）
        vector_weight = float(cfg.get("vector_weight", 0.5))
        bm25_weight = float(cfg.get("bm25_weight", 0.5))
        drop_vector = vector_weight == 0.0
        drop_bm25 = bm25_weight == 0.0
        if drop_vector or drop_bm25:
            retriever = self._service.retriever
            if hasattr(retriever, "search_children"):
                self._retriever = retriever
                # MultiGranularityRetriever 是 pydantic 模型：用 object.__setattr__
                # 写入实例 __dict__ 以可靠遮蔽类方法（普通赋值会进 __pydantic_extra__）
                object.__setattr__(
                    retriever,
                    "search_children",
                    _make_search_children_wrapper(
                        retriever.search_children, drop_vector, drop_bm25
                    ),
                )
                log.info(
                    "单路消融: search_children 已包装 "
                    f"(drop_vector={drop_vector}, drop_bm25={drop_bm25})"
                )
            else:
                log.warning("检索器无 search_children，单路剔除未生效")
        return self

    def __exit__(self, *exc):
        cfg = vector_store_config
        for key, value in self._backup.items():
            cfg[key] = value
        pipeline = self._service._pipeline
        if pipeline is not None:
            pipeline.__dict__.pop("resolve_parents", None)
        if self._retriever is not None:
            self._retriever.__dict__.pop("search_children", None)
            self._retriever = None
        return False


def config_snapshot(profile: dict) -> dict:
    """记录 profile 生效后与本实验相关的配置快照。"""
    cfg = vector_store_config
    flags = cfg.get("feature_flags", {})
    reranker = cfg.get("reranker", {})
    return {
        "description": profile["description"],
        "k": cfg.get("k"),
        "score_threshold": cfg.get("score_threshold"),
        "vector_weight": cfg.get("vector_weight"),
        "bm25_weight": cfg.get("bm25_weight"),
        "reranker_top_n": reranker.get("top_n"),
        "reranker_model": reranker.get("model"),
        "cross_query_rrf": flags.get("cross_query_rrf"),
        "global_rerank": flags.get("global_rerank"),
        "active_version_filter": flags.get("active_version_filter"),
        "parent_backtrack": profile["parent_backtrack"],
        "enable_rewrite": profile["enable_rewrite"],
        "rewrite_strategy": profile.get("rewrite_strategy", ""),
        "query_routing_enabled": cfg.get("query_routing", {}).get("enabled"),
    }


# ── 主流程 ────────────────────────────────────────────────────


def run_profile(service: RagService, profile: dict, dataset: list[dict]) -> dict:
    """对单个配置跑完整评测集，返回汇总指标与逐题明细。"""
    details: list[dict] = []
    failures: list[dict] = []
    name = profile["name"]

    with ProfileSession(service, profile):
        snapshot = config_snapshot(profile)
        for idx, item in enumerate(dataset):
            question = item["question"]
            qtype = item.get("question_type", "unknown")
            wall0 = time.time()
            try:
                ctx = service.prepare(
                    question,
                    enable_rewrite=profile["enable_rewrite"],
                    rewrite_strategy=profile.get("rewrite_strategy", ""),
                )
            except Exception as e:  # 单题失败不阻断整个配置
                failures.append(
                    {"index": idx, "question": question, "error": str(e)[:300]}
                )
                log.info(f"  [{name}] Q{idx} 失败: {e}")
                continue
            wall_ms = round((time.time() - wall0) * 1000, 2)

            texts = build_ranked_texts(ctx, limit=10)
            if qtype == "unanswerable":
                # 不可回答题：统计检索噪声（问题关键词在 top10 中的命中率），
                # 不参与 Recall/MRR 汇总；线上拦截由 query_routing 负责。
                q_keywords = extract_keywords(question)
                noise_hits = sum(
                    1 for kw in q_keywords if any(kw in t.lower() for t in texts)
                )
                metrics = {
                    "noise_hit_rate_at_10": (
                        round(noise_hits / len(q_keywords), 4) if q_keywords else 0.0
                    ),
                    "retrieved_blocks": len(texts),
                }
            else:
                keywords = extract_keywords(item.get("ground_truth", ""))
                metrics = compute_metrics(keywords, texts)
                metrics["keyword_count"] = len(keywords)

            details.append(
                {
                    "index": idx,
                    "question": question,
                    "question_type": qtype,
                    "metrics": metrics,
                    "latency_ms": wall_ms,
                    "stage_durations": dict(ctx.stage_durations),
                    "degradation_events": len(ctx.degradation_events),
                    "degradation_detail": ctx.degradation_events,
                    "reranked_top10": [
                        d.page_content for d in ctx.reranked_children[:10]
                    ],
                    "resolved_parents_top10": [
                        d.page_content[:2000] for d in ctx.resolved_parents[:10]
                    ],
                }
            )
            if (idx + 1) % 10 == 0:
                log.info(f"  [{name}] 进度 {idx + 1}/{len(dataset)}")

    answerable = [d for d in details if d["question_type"] != "unanswerable"]
    unanswerable = [d for d in details if d["question_type"] == "unanswerable"]
    latencies = [d["latency_ms"] for d in details]
    retrieve_ms = [
        d["stage_durations"].get("retrieve_total", 0.0) for d in details
    ]

    def _avg(key: str) -> float:
        vals = [d["metrics"].get(key, 0.0) for d in answerable]
        return round(statistics.fmean(vals), 4) if vals else 0.0

    summary = {
        "questions_total": len(dataset),
        "questions_succeeded": len(details),
        "questions_failed": len(failures),
        "answerable_count": len(answerable),
        "recall_at_5": _avg("recall_at_5"),
        "recall_at_10": _avg("recall_at_10"),
        "mrr": _avg("rr"),
        "ndcg_at_10": _avg("ndcg_at_10"),
        "unanswerable_noise_hit_rate_at_10": (
            round(
                statistics.fmean(
                    d["metrics"]["noise_hit_rate_at_10"] for d in unanswerable
                ),
                4,
            )
            if unanswerable
            else None
        ),
        "unanswerable_empty_retrieval_rate": (
            round(
                sum(1 for d in unanswerable if d["metrics"]["retrieved_blocks"] == 0)
                / len(unanswerable),
                4,
            )
            if unanswerable
            else None
        ),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "retrieve_total_p50_ms": _percentile(retrieve_ms, 50),
        "retrieve_total_p95_ms": _percentile(retrieve_ms, 95),
        "degradation_events_total": sum(d["degradation_events"] for d in details),
    }
    return {
        "config_snapshot": snapshot,
        "summary": summary,
        "failures": failures,
        "per_question": details,
    }


def print_comparison(results: dict) -> None:
    """在 stdout/日志打印 6 配置对比表。"""
    lines = [
        "",
        "=" * 92,
        "消融结果对比（可回答题汇总；噪声列为 unanswerable 题的问题关键词命中率）",
        "=" * 92,
        (
            f"{'配置':<24}{'R@5':>8}{'R@10':>8}{'MRR':>8}{'nDCG@10':>9}"
            f"{'p50(ms)':>10}{'p95(ms)':>10}{'噪声@10':>9}{'降级':>6}"
        ),
        "-" * 92,
    ]
    for name, res in results["configurations"].items():
        s = res["summary"]
        noise = s["unanswerable_noise_hit_rate_at_10"]
        noise_str = f"{noise:.4f}" if noise is not None else "-"
        lines.append(
            f"{name:<24}{s['recall_at_5']:>8.4f}{s['recall_at_10']:>8.4f}"
            f"{s['mrr']:>8.4f}{s['ndcg_at_10']:>9.4f}"
            f"{s['latency_p50_ms']:>10.0f}{s['latency_p95_ms']:>10.0f}"
            f"{noise_str:>9}{s['degradation_events_total']:>6}"
        )
    lines.append("=" * 92)
    for line in lines:
        log.info(line)


def save_results(results: dict, out_path: Path, partial: bool) -> None:
    """增量保存：每完成一个配置即覆盖写入，中断不丢已完成部分。"""
    results["partial"] = partial
    results["timestamp"] = datetime.now(timezone.utc).isoformat()
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _parse_names(raw: str) -> set[str]:
    return {s.strip() for s in raw.split(",") if s.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description="检索消融运行器（Spec 10.5）")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟用）")
    parser.add_argument(
        "--configs",
        default="",
        help="只跑指定配置（逗号分隔，等同于 --only）",
    )
    parser.add_argument(
        "--only",
        default="",
        help="只跑指定配置（逗号分隔，如 dense_only,bm25_only）",
    )
    parser.add_argument(
        "--skip",
        default="",
        help="跳过指定配置（逗号分隔），其余照跑；已有结果文件中未重跑的配置保留",
    )
    args = parser.parse_args()

    log_path = setup_logging()
    log.info(f"日志文件: {log_path}")

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.limit > 0:
        dataset = dataset[: args.limit]

    selected = _parse_names(args.only) or _parse_names(args.configs)
    skipped = _parse_names(args.skip)
    profiles = [
        p
        for p in PROFILES
        if (not selected or p["name"] in selected) and p["name"] not in skipped
    ]
    if not profiles:
        log.error("没有匹配的消融配置")
        return 1
    unknown = (selected | skipped) - {p["name"] for p in PROFILES}
    if unknown:
        log.warning(f"未知的配置名（忽略）: {sorted(unknown)}")

    # 运行期间统一关闭 query_routing：路由是检索上游的独立阶段，且每次
    # classify 是一次 LLM 调用（60+ 题 × 6 配置下延迟/成本不可接受）。
    # RagService 缓存的是同一嵌套 dict 的引用，原地修改即生效。
    routing_cfg = vector_store_config.setdefault("query_routing", {})
    routing_backup = copy.deepcopy(routing_cfg)
    routing_cfg["enabled"] = False

    # 运行期间统一关闭 active_version_filter：agent_small 全部 18571 行的
    # metadata 均无 is_active 键，PGVector 等值过滤 {"is_active": True}
    # 要求键存在且为布尔 true → 向量路恒返回 0 条（Top-1 兜底也拿不到），
    # 线上混合检索实际退化为纯 BM25（2026-08 探针实证，见结果 notes）。
    # 关闭该开关后 filter=None，与 _fetch_parents SQL 的"缺失视为 active"
    # 语义一致，dense 路才真正参与消融对比。这是配置级回滚开关，不改源码。
    flags_cfg = vector_store_config.setdefault("feature_flags", {})
    avf_backup = flags_cfg.get("active_version_filter", True)
    flags_cfg["active_version_filter"] = False

    log.info(f"加载评测集: {args.dataset}（{len(dataset)} 题）")
    log.info(f"消融配置: {[p['name'] for p in profiles]}")
    t0 = time.time()
    service = RagService()
    service._get_pipeline()  # 提前触发管道/检索器初始化
    log.info(f"RagService 初始化完成（{time.time() - t0:.1f}s）")

    out_path = Path(args.output)

    # 断点续跑：已有结果文件中未重跑的配置保留并入最终输出
    existing: dict = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8")).get(
                "configurations", {}
            )
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"已有结果文件不可读（忽略）: {e}")
    reused = [n for n in existing if n not in {p["name"] for p in profiles}]
    if reused:
        log.info(f"复用已有结果中的配置: {reused}")

    results: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "num_questions": len(dataset),
        "notes": [
            "检索级指标（关键词命中近似），未调用 LLM 生成答案，未做 RAGAS。",
            "运行期间统一关闭 query_routing；unanswerable 题只统计检索噪声。",
            "排名列表 = reranked_children + fused_children 余量补足 top10。",
            "父块回溯开关通过 RetrievalPipeline 实例方法 monkeypatch 实现，"
            "其余开关通过原地修改 vector_store_config dict 实现，均未改源码。",
            "单路消融（dense_only/bm25_only）：管道无论权重是否为零都会同时执行"
            "两路召回，权重只影响 RRF 排序；为使单路语义真实生效，对"
            " search_children 做实例级包装直接剔除零权重分支的候选（否则两种"
            "配置 top10 候选集合完全相同，Recall 在并集池上饱和——2026-08 探针"
            "实证）。旧版结果中六配置 R@10 全同即此口径问题所致。",
            "重要发现：agent_small 全部文档 metadata 均无 is_active 键，线上"
            " active_version_filter=true 时向量路等值过滤恒命中 0 条，混合检索"
            "实际长期退化为纯 BM25。本实验统一关闭 active_version_filter"
            "（filter=None，与 _fetch_parents SQL 的缺失即 active 语义一致），"
            "dense 路才真实参与对比；该问题属 backend 缺陷，本次未改源码，"
            "建议后续修复元数据写入或过滤语义。",
            "指标口径局限：Recall/MRR 基于 ground_truth 关键词命中近似，非"
            " gold chunk 精确判定，绝对值偏高，只适合配置间横向对比。",
        ],
        "configurations": {},
    }
    try:
        for profile in profiles:
            log.info(f">>> 运行配置 {profile['name']}: {profile['description']}")
            t_cfg = time.time()
            results["configurations"][profile["name"]] = run_profile(
                service, profile, dataset
            )
            s = results["configurations"][profile["name"]]["summary"]
            log.info(
                f"<<< {profile['name']} 完成（{time.time() - t_cfg:.1f}s）| "
                f"R@5={s['recall_at_5']} R@10={s['recall_at_10']} MRR={s['mrr']} "
                f"失败={s['questions_failed']}"
            )
            save_results(results, out_path, partial=True)  # 每配置增量保存
    finally:
        # 恢复路由/版本过滤配置（进程退出前保持现场干净）
        routing_cfg.clear()
        routing_cfg.update(routing_backup)
        flags_cfg["active_version_filter"] = avf_backup

    # 并入未重跑的已有配置（断点续跑场景），按 PROFILES 定义顺序排列
    for name in reused:
        results["configurations"][name] = existing[name]
    results["configurations"] = {
        p["name"]: results["configurations"][p["name"]]
        for p in PROFILES
        if p["name"] in results["configurations"]
    }

    print_comparison(results)
    save_results(results, out_path, partial=False)
    log.info(f"结果已写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
