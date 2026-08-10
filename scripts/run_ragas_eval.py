#!/usr/bin/env python3
"""RAGAS 端到端质量评测（Spec V2 阶段 5）。

对固定评测集（evaluation_data/fixed_eval_dataset.json，91 条）逐条执行在线
管道 ``RagService.prepare(query)`` 检索 + ``RagService.stream_answer(ctx)``
流式生成（拼接全部 token），收集 question/answer/contexts/ground_truth，
然后用 ragas 评估四个指标：

- faithfulness        回答是否忠实于检索上下文（无幻觉）
- context_recall      检索内容覆盖参考答案的程度
- context_precision   检索内容与问题的相关度
- answer_correctness  回答相比参考答案的语义正确性

口径说明：
- 路由保持线上默认开启（prepare 默认 enable_rewrite=False，与线上一致）。
  unanswerable 题大概率被路由拦截（irrelevant → 固定拒答文案、无检索上下
  文），其 ground_truth 为"无法回答"类说明文字，与 RAGAS 四指标语义不同
  （context_recall 衡量的是"检索覆盖参考答案"，对拒答题无意义）。因此
  **四指标均值只统计 answerable 题**；unanswerable 题单独评分并归入
  ``unanswerable`` 节点，不混入均值。路由拦截率另行列出。
- contexts 取 ctx.context_docs（冻结后实际进入 prompt 的块），为空时回退
  resolved_parents → reranked_children。
- 评估 LLM 为 DashScope qwen3.6-plus（temperature=0），embedding 为
  DashScope text-embedding-v4，与旧基线（ragas_eval_scores.json）一致。

健壮性：
- 每题完成即增量覆盖写 evaluation_data/ragas_v2_filled.json（防中断）；
  重跑时已有 answer 的条目自动跳过（断点续跑）。
- 评分阶段每完成一条即增量写 evaluation_data/ragas_v2_scores.json。

用法：
    export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
    python3 scripts/run_ragas_eval.py                     # 全流程
    python3 scripts/run_ragas_eval.py --limit 3           # 冒烟（前 3 题）
    python3 scripts/run_ragas_eval.py --skip-fill         # 只评分
    python3 scripts/run_ragas_eval.py --only-fill         # 只填充不评分
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "packages"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

# ── 修复 ragas 0.4.x → langchain_community 的 vertexai 导入问题 ──
# （与 scripts/fill_ragas_dataset.py 相同的 workaround，须在 import ragas 前执行）
import langchain_community.chat_models  # noqa: E402

if not hasattr(langchain_community.chat_models, "vertexai"):
    _mock = MagicMock()
    _mock.ChatVertexAI = MagicMock()
    langchain_community.chat_models.vertexai = _mock
    sys.modules["langchain_community.chat_models.vertexai"] = _mock

from agent_material.rag_engine.rag_service import RagService  # noqa: E402
from agent_material.utils.logger import logger  # noqa: E402

EVAL_DIR = ROOT / "evaluation_data"
DEFAULT_DATASET = EVAL_DIR / "fixed_eval_dataset.json"
DEFAULT_FILLED = EVAL_DIR / "ragas_v2_filled.json"
DEFAULT_SCORES = EVAL_DIR / "ragas_v2_scores.json"

METRIC_NAMES = [
    "faithfulness",
    "context_recall",
    "context_precision",
    "answer_correctness",
]


# ══════════════════════════════════════════════════════════════════════════════
#  阶段 1：检索 + 生成填充
# ══════════════════════════════════════════════════════════════════════════════


def _contexts_from_ctx(ctx) -> list[str]:
    """取评估用上下文块文本：冻结 context_docs 优先，回退父块/重排子块。"""
    docs = ctx.context_docs or ctx.resolved_parents or ctx.reranked_children
    return [d.page_content for d in docs]


def fill_dataset(
    dataset: list[dict],
    filled_path: Path,
    limit: int = 0,
) -> list[dict]:
    """逐条 prepare + stream_answer，增量落盘，支持断点续跑。"""
    records: list[dict] = []
    if filled_path.exists():
        try:
            records = json.loads(filled_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"已有填充文件不可读，重新填充: {e}")
            records = []

    items = dataset[:limit] if limit > 0 else dataset
    # 对齐长度：已有记录按 index 复用，新条目补空壳
    while len(records) < len(items):
        records.append({})

    todo = [
        i
        for i in range(len(items))
        if not records[i].get("answer") and not records[i].get("error")
    ]
    if not todo:
        logger.info("所有条目均已填充，跳过检索/生成阶段")
        return records

    logger.info(f"初始化 RagService（待填充 {len(todo)}/{len(items)} 条）...")
    t0 = time.time()
    service = RagService()
    service._get_pipeline()
    logger.info(f"RagService 就绪（{time.time() - t0:.1f}s）")

    for done_cnt, idx in enumerate(todo, start=1):
        item = items[idx]
        question = item["question"]
        qtype = item.get("question_type", "unknown")
        logger.info(f"[Fill {done_cnt}/{len(todo)}] idx={idx} {question[:60]}")
        wall0 = time.time()
        try:
            ctx = service.prepare(question)  # enable_rewrite=False（线上默认）
            answer = "".join(service.stream_answer(ctx))
            records[idx] = {
                "index": idx,
                "question": question,
                "ground_truth": item.get("ground_truth", ""),
                "question_type": qtype,
                "route_decision": ctx.route_decision,
                "contexts": _contexts_from_ctx(ctx),
                "answer": answer,
                "latency_ms": round((time.time() - wall0) * 1000, 2),
                "degradation_events": len(ctx.degradation_events),
            }
            logger.info(
                f"  → route={ctx.route_decision} contexts="
                f"{len(records[idx]['contexts'])} answer={len(answer)} 字符"
            )
        except Exception as e:  # 单题失败不阻断
            logger.error(f"  → 失败: {e}")
            records[idx] = {
                "index": idx,
                "question": question,
                "ground_truth": item.get("ground_truth", ""),
                "question_type": qtype,
                "contexts": [],
                "answer": "",
                "error": str(e)[:300],
            }
        # 每题完成即增量落盘
        filled_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    logger.info(f"填充完成: {filled_path}")
    return records


# ══════════════════════════════════════════════════════════════════════════════
#  阶段 2：RAGAS 评估
# ══════════════════════════════════════════════════════════════════════════════


def _create_eval_llm():
    """创建 RAGAS 评估用的 LLM（DashScope 千问，与旧基线一致）。"""
    import os

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model="qwen3.6-plus",
        temperature=0.0,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
    )


def _create_eval_embeddings():
    """DashScope text-embedding-v4（AnswerCorrectness 的相似度计算依赖）。

    DashScope OpenAI 兼容模式不支持 /embeddings，用项目自带
    AsyncDashScopeEmbeddings（原生 API）+ LangchainEmbeddingsWrapper。
    """
    from langchain_core.embeddings import Embeddings
    from ragas.embeddings.base import LangchainEmbeddingsWrapper

    from agent_material.rag_engine.embedding.async_embeddings import (
        AsyncDashScopeEmbeddings,
    )

    class _DashScopeEmbeddingsAdapter(Embeddings):
        def __init__(self, model_name: str = "text-embedding-v4", dimensions: int = 1024):
            self._delegate = AsyncDashScopeEmbeddings(
                model=model_name,
                dimensions=dimensions,
                max_concurrency=1,  # 串行发送，避免撞 DashScope 限流
            )

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._delegate.embed_documents(texts)

        def embed_query(self, text: str) -> list[float]:
            return self._delegate.embed_query(text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return await self._delegate.aembed_documents(texts)

        async def aembed_query(self, text: str) -> list[float]:
            return await self._delegate.aembed_query(text)

    return LangchainEmbeddingsWrapper(_DashScopeEmbeddingsAdapter())


def _build_metrics() -> dict:
    from ragas.metrics import (
        AnswerCorrectness,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )
    from ragas.metrics._answer_correctness import AnswerSimilarity

    eval_llm = _create_eval_llm()
    eval_embeddings = _create_eval_embeddings()
    return {
        "faithfulness": Faithfulness(llm=eval_llm),
        "context_recall": ContextRecall(llm=eval_llm),
        "context_precision": ContextPrecision(llm=eval_llm),
        "answer_correctness": AnswerCorrectness(
            llm=eval_llm,
            answer_similarity=AnswerSimilarity(embeddings=eval_embeddings),
        ),
    }


async def _score_one(sample, metrics: dict) -> dict:
    """单条样本并发执行 4 个指标评分。"""

    async def _one(name: str, metric):
        try:
            return name, round(await metric.single_turn_ascore(sample), 4)
        except Exception as e:
            logger.warning(f"{name} 评分失败: {e}")
            return name, None

    return dict(await asyncio.gather(*(_one(n, m) for n, m in metrics.items())))


async def _run_evaluation(
    items: list[dict],
    metrics: dict,
    concurrency: int,
    scores_path: Path,
    meta: dict,
    prior_details: list[dict] | None = None,
) -> list[dict]:
    """异步并发评分（信号量限流），每条完成即增量写 scores 文件。

    prior_details：断点续跑时此前已评分的明细，增量保存与最终合并都包含它。
    """
    from ragas.dataset_schema import SingleTurnSample

    prior = list(prior_details or [])
    sem = asyncio.Semaphore(concurrency)
    total = len(items)
    details: list[dict | None] = [None] * total
    t_start = time.time()
    done = 0

    async def _process(pos: int, r: dict) -> None:
        nonlocal done
        sample = SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        )
        async with sem:
            scores = await _score_one(sample, metrics)
        details[pos] = {
            "index": r["index"],
            "question": r["question"],
            "question_type": r["question_type"],
            "route_decision": r.get("route_decision", ""),
            **scores,
        }
        done += 1
        logger.info(f"[Eval] {done}/{total} ({(time.time() - t_start):.0f}s)")
        _save_scores(scores_path, meta, prior + [d for d in details if d], partial=True)

    await asyncio.gather(*(_process(i, r) for i, r in enumerate(items)))
    return prior + [d for d in details if d]


def _summarize(details: list[dict]) -> dict:
    """各指标均值/中位数/极值（忽略 None）。"""
    summary: dict[str, dict] = {}
    for name in METRIC_NAMES:
        vals = sorted(d[name] for d in details if d.get(name) is not None)
        if vals:
            n = len(vals)
            summary[name] = {
                "mean": round(sum(vals) / n, 4),
                "median": round(vals[n // 2], 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "count": n,
            }
        else:
            summary[name] = {"mean": None, "count": 0}
    return summary


def _save_scores(
    scores_path: Path, meta: dict, details: list[dict], partial: bool
) -> None:
    """组装并写 scores JSON（增量/最终共用）。

    四指标均值只统计 answerable 题；unanswerable 题单独汇总（其
    ground_truth 为"无法回答"说明文，与 RAGAS 指标语义不同，见文件头注释）。
    """
    answerable = [d for d in details if d["question_type"] != "unanswerable"]
    unanswerable = [d for d in details if d["question_type"] == "unanswerable"]
    payload = {
        **meta,
        "partial": partial,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "四指标均值（summary）仅统计 answerable 题；unanswerable 题的"
            " ground_truth 为'无法回答'类说明，与 RAGAS 指标语义不同，单独"
            "归入 unanswerable 节点，不混入均值。",
            "评估 LLM=qwen3.6-plus(t=0)，embedding=text-embedding-v4，"
            "与旧基线 ragas_eval_scores.json（40 题）同模型，但评测集扩为"
            " 91 题固定集，绝对值与旧基线可比性受限（题集不同）。",
            "contexts 取 ctx.context_docs（实际进入 prompt 的冻结块）。",
        ],
        "scored_count": len(details),
        "answerable_count": len(answerable),
        "summary": _summarize(answerable),
        "unanswerable": {
            "count": len(unanswerable),
            "summary": _summarize(unanswerable),
            "details": unanswerable,
        },
        "details": answerable,
    }
    scores_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_evaluation(
    records: list[dict],
    scores_path: Path,
    concurrency: int,
) -> dict:
    """对填充结果执行 RAGAS 评分并落盘。"""
    valid = [r for r in records if r.get("answer") and not r.get("error")]
    if not valid:
        logger.error("没有有效条目可供评估")
        return {"error": "无有效条目"}

    # 断点续跑：已有 scores 文件中已评分的 index 跳过（防中断重评）
    prior_details: list[dict] = []
    if scores_path.exists():
        try:
            prev = json.loads(scores_path.read_text(encoding="utf-8"))
            prior_details = prev.get("details", []) + prev.get(
                "unanswerable", {}
            ).get("details", [])
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"已有 scores 文件不可读（重新评分）: {e}")
    scored_idx = {d["index"] for d in prior_details}
    if scored_idx:
        valid = [r for r in valid if r["index"] not in scored_idx]
        logger.info(f"断点续跑: 已评分 {len(scored_idx)} 条，剩余 {len(valid)} 条")
    if not valid and prior_details:
        logger.info("全部条目均已评分，直接汇总")
        _save_scores(scores_path, {}, prior_details, partial=False)
        return json.loads(scores_path.read_text(encoding="utf-8"))

    logger.info(f"RAGAS 评估: {len(valid)} 条待评（并发={concurrency}）")

    metrics = _build_metrics()
    meta = {
        "dataset": str(DEFAULT_DATASET),
        "filled": str(DEFAULT_FILLED),
        "concurrency": concurrency,
    }
    t0 = time.time()
    details = asyncio.run(
        _run_evaluation(valid, metrics, concurrency, scores_path, meta, prior_details)
    )
    meta["evaluation_time_s"] = round(time.time() - t0, 1)
    _save_scores(scores_path, meta, details, partial=False)
    logger.info(f"评估结果已保存: {scores_path}")
    return json.loads(scores_path.read_text(encoding="utf-8"))


def _print_report(result: dict) -> None:
    print("\n" + "=" * 64)
    print("RAGAS V2 评估结果（answerable 题均值；unanswerable 单独统计）")
    print("=" * 64)
    print(f"{'指标':<22}{'均值':>8}{'中位数':>8}{'最低':>8}{'最高':>8}{'n':>5}")
    print("-" * 64)
    for name in METRIC_NAMES:
        s = result["summary"].get(name, {})
        if s.get("mean") is not None:
            print(
                f"{name:<22}{s['mean']:>8.4f}{s['median']:>8.4f}"
                f"{s['min']:>8.4f}{s['max']:>8.4f}{s['count']:>5}"
            )
        else:
            print(f"{name:<22}{'N/A':>8}")
    u = result.get("unanswerable", {})
    print("-" * 64)
    print(f"unanswerable 题: {u.get('count', 0)} 条（均值见 scores 文件，不计入上表）")
    print("=" * 64)


# ══════════════════════════════════════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS 端到端质量评测（Spec V2 阶段 5）")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--filled", type=Path, default=DEFAULT_FILLED)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 题（冒烟用）")
    parser.add_argument("--skip-fill", action="store_true", help="跳过填充，仅评分")
    parser.add_argument("--only-fill", action="store_true", help="只填充，不评分")
    parser.add_argument("--eval-concurrency", type=int, default=5)
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    logger.info(f"加载评测集: {args.dataset}（{len(dataset)} 题）")

    if args.skip_fill:
        records = json.loads(args.filled.read_text(encoding="utf-8"))
    else:
        records = fill_dataset(dataset, args.filled, limit=args.limit)

    if args.only_fill:
        return 0

    result = run_evaluation(records, args.scores, args.eval_concurrency)
    if "error" not in result:
        _print_report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
