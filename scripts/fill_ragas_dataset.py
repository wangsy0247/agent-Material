"""用 RAG 检索 + LLM 生成填充 RAGAS 评测集，并运行 RAGAS 评估。

流程：
1. 读取评测集 (question + ground_truth)
2. 逐条 RAG 检索 chunk → 填入 contexts，LLM 生成回答 → 填入 answer
3. 增量保存，支持断点续跑
4. 用 RAGAS 指标并发评估检索+生成质量

输入: evaluation_data/ragas_eval_dataset.json
输出:
  - evaluation_data/ragas_eval_dataset_filled.json  (填充后的数据集)
  - evaluation_data/ragas_eval_scores.json           (各指标得分)

用法:
    python scripts/fill_ragas_dataset.py
    python scripts/fill_ragas_dataset.py --skip-fill              # 跳过填充，仅评估
    python scripts/fill_ragas_dataset.py --max-eval 10            # 只评估前10条
    python scripts/fill_ragas_dataset.py --eval-concurrency 8     # 并发数
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

# ── 修复 ragas → langchain_community 的导入问题 ──
import langchain_community.chat_models  # noqa: E402
if not hasattr(langchain_community.chat_models, "vertexai"):
    _mock = MagicMock()
    _mock.ChatVertexAI = MagicMock()
    langchain_community.chat_models.vertexai = _mock
    sys.modules["langchain_community.chat_models.vertexai"] = _mock

# 直接导入 agent_material（backend/packages 已在 sys.path 中）

from agent_material.rag_engine.rag_service import RagService  # noqa: E402
from agent_material.utils.logger import logger  # noqa: E402

# ── 默认路径（可被命令行参数覆盖） ──
EVAL_DIR = PROJECT_ROOT / "evaluation_data"
DEFAULT_INPUT_PATH = EVAL_DIR / "ragas_eval_dataset.json"
DEFAULT_OUTPUT_PATH = EVAL_DIR / "ragas_eval_dataset_filled.json"
DEFAULT_SCORES_PATH = EVAL_DIR / "ragas_eval_scores.json"


# ════════════════════════════════════════════════════════════════════════════════
#  进度管理
# ════════════════════════════════════════════════════════════════════════════════


def load_dataset(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_progress(progress_path: Path) -> int:
    if progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as f:
            return json.load(f).get("done", 0)
    return 0


def save_progress(progress_path: Path, done: int) -> None:
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump({"done": done, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f)


def save_incremental(records: list[dict], done: int, output_path: Path) -> None:
    output = []
    for i, r in enumerate(records):
        if i < done and r.get("contexts") and r.get("answer"):
            output.append({
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "contexts": r["contexts"],
                "answer": r["answer"],
            })
        else:
            output.append({
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "contexts": [],
                "answer": "",
            })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


# ════════════════════════════════════════════════════════════════════════════════
#  RAG 检索 + 生成填充
# ════════════════════════════════════════════════════════════════════════════════


def fill_dataset(
    records: list[dict],
    output_path: Path,
    progress_path: Path,
) -> list[dict]:
    """对每条 question 执行 RAG 检索和生成，填入 contexts 和 answer。"""
    total = len(records)
    start_idx = load_progress(progress_path)

    if start_idx > 0:
        logger.info(f"断点续跑: 从第 {start_idx + 1}/{total} 条开始")
        if output_path.exists():
            saved = load_dataset(output_path)
            for i in range(min(start_idx, len(saved))):
                if saved[i].get("contexts") and saved[i].get("answer"):
                    records[i]["contexts"] = saved[i]["contexts"]
                    records[i]["answer"] = saved[i]["answer"]

    logger.info("初始化 RagService ...")
    rag = RagService()
    logger.info(
        f"RagService 就绪 | 文档数: {rag.vector_store.get_document_count()} | "
        f"模型: {rag.model.model_name if rag.model else 'unknown'}"
    )

    for idx in range(start_idx, total):
        question = records[idx]["question"]
        logger.info(f"[{idx + 1}/{total}] 处理: {question[:80]}...")

        try:
            docs = rag.retriever_docx(question)
            contexts = [doc.page_content for doc in docs]
            logger.info(f"  → 检索到 {len(contexts)} 个 chunk")

            answer = rag.get_summary(question)
            logger.info(f"  → 生成回答 {len(answer)} 字符")

            records[idx]["contexts"] = contexts
            records[idx]["answer"] = answer

        except Exception as e:
            logger.error(f"  → 失败: {e}")
            records[idx]["contexts"] = [f"[检索/生成失败: {e}]"]
            records[idx]["answer"] = ""

        save_incremental(records, idx + 1, output_path)
        save_progress(progress_path, idx + 1)

    if progress_path.exists():
        progress_path.unlink()

    logger.info(f"填充完成，输出: {output_path}")
    return records


# ════════════════════════════════════════════════════════════════════════════════
#  RAGAS 评估（异步并发版）
# ════════════════════════════════════════════════════════════════════════════════


def _create_eval_llm():
    """创建 RAGAS 评估用的 LLM（DashScope 千问）。"""
    import os
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model="qwen3.6-plus",
        temperature=0.0,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
    )


def _create_eval_embeddings():
    """创建 RAGAS 评估用的 Embedding 模型（DashScope text-embedding-v4）。

    AnswerCorrectness 指标依赖 AnswerSimilarity，后者需要 embedding 计算
    生成答案和参考答案之间的语义相似度。

    DashScope 的 OpenAI 兼容模式不支持 /embeddings 端点，因此使用项目
    自带的 AsyncDashScopeEmbeddings（直接调用 DashScope 原生 API）。
    通过一个 LangChain Embeddings 适配器让 ragas 接受。
    """
    from langchain_core.embeddings import Embeddings
    from agent_material.rag_engine.embedding.async_embeddings import AsyncDashScopeEmbeddings
    from ragas.embeddings.base import LangchainEmbeddingsWrapper

    # LangChain Embeddings 适配器 — AsyncDashScopeEmbeddings 实现了全套接口
    # 但不继承 langchain_core.embeddings.Embeddings，所以无法通过 isinstance 校验。
    # 这里做个显式继承适配。
    #
    # 注意：ragas 的 AnswerSimilarity 对 reference 和 response 各调一次
    # embed_text()（单条），所以日志中会看到大量「嵌入 1 条」的记录。
    # 这是 ragas 内部行为决定的，不影响功能正确性。
    class _DashScopeEmbeddingsAdapter(Embeddings):
        def __init__(self, model_name: str = "text-embedding-v4", dimensions: int = 1024):
            self._delegate = AsyncDashScopeEmbeddings(
                model=model_name,
                dimensions=dimensions,
                max_concurrency=1,  # 串行发送，避免并发请求撞 DashScope 限流
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


async def _score_one_sample(
    sample,
    metrics: dict,
    idx: int,
    question_preview: str,
) -> dict:
    """对单条样本并发执行全部 4 个指标的评分。

    每个指标的 single_turn_ascore 是独立异步 LLM 调用，
    并发执行可将 4 次调用的时间压缩为最慢那次的时间。
    """
    async def _score_metric(name: str, metric) -> tuple[str, float | None]:
        try:
            score = await metric.single_turn_ascore(sample)
            return name, round(score, 4)
        except Exception as e:
            logger.warning(f"[{idx}] {name} 评分失败: {e}")
            return name, None

    tasks = [_score_metric(name, m) for name, m in metrics.items()]
    results = await asyncio.gather(*tasks)

    item_result = {"question": question_preview, "index": idx}
    for name, score in results:
        item_result[name] = score

    return item_result


async def _run_async_evaluation(
    valid_items: list[dict],
    metrics: dict,
    concurrency: int,
) -> tuple[dict[str, list[float]], list[dict]]:
    """异步并发执行 RAGAS 评估。

    并发模型：
    - 单条样本内：4 个指标并发评分（1 次 gather）
    - 样本间：用 Semaphore 控制并发数，防止打爆 API 速率限制

    Args:
        valid_items: 有效条目列表
        metrics: {"name": metric_instance} 字典
        concurrency: 样本间最大并发数

    Returns:
        (scores_dict, details_list)
    """
    from ragas.dataset_schema import SingleTurnSample

    sem = asyncio.Semaphore(concurrency)
    total = len(valid_items)
    t_start = time.time()

    async def _process_one(idx: int, r: dict) -> tuple[int, dict]:
        sample = SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        )
        preview = r["question"][:60] + "..."

        async with sem:
            item_result = await _score_one_sample(sample, metrics, idx, preview)

        # 逐条打印进度
        elapsed = time.time() - t_start
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        logger.info(
            f"[Eval] {idx + 1}/{total} ({rate:.1f} 条/s) | {preview}"
        )
        return idx, item_result

    # 所有样本并发（信号量控制实际并发度）
    all_tasks = [_process_one(i, r) for i, r in enumerate(valid_items)]
    results = await asyncio.gather(*all_tasks)

    # 按原始顺序重建得分和详情
    results.sort(key=lambda x: x[0])
    ordered = [r for _, r in results]

    scores: dict[str, list[float]] = {name: [] for name in metrics}
    for item in ordered:
        for name in metrics:
            if item.get(name) is not None:
                scores[name].append(item[name])

    return scores, ordered


def run_ragas_evaluation(
    records: list[dict],
    scores_path: Path,
    max_eval: int = 0,
    concurrency: int = 5,
) -> dict:
    """用 RAGAS 指标评估填充后的数据集（异步并发版）。

    指标说明（均为 0~1，越高越好）：
    - faithfulness:      回答是否忠实于检索到的上下文（无幻觉）
    - context_recall:    检索内容覆盖参考答案的程度
    - context_precision: 检索内容与问题的相关度
    - answer_correctness:回答相比参考答案的语义正确性
    """
    from ragas.metrics import (
        Faithfulness,
        ContextRecall,
        ContextPrecision,
        AnswerCorrectness,
    )

    # 过滤有效条目
    valid = [
        r for r in records
        if r.get("contexts") and r.get("answer") and any(
            "[检索/生成失败" not in c for c in r["contexts"]
        )
    ]
    if not valid:
        logger.warning("没有有效条目可供评估")
        return {"error": "无有效条目", "valid_count": 0, "total": len(records)}

    # 限制评估数量
    if max_eval and max_eval < len(valid):
        valid = valid[:max_eval]
        logger.info(f"RAGAS 评估: 限制前 {len([r for r in records if r.get('contexts') and r.get('answer')])} 条 → 实际评估 {len(valid)} 条")

    eval_count = len(valid)
    total_llm_calls = eval_count * 4  # 4 个指标
    logger.info(
        f"RAGAS 评估: {eval_count} 条有效 "
        f"(预计 {total_llm_calls} 次 LLM 调用, 并发={concurrency})"
    )

    eval_llm = _create_eval_llm()
    eval_embeddings = _create_eval_embeddings()

    # AnswerCorrectness 需要 AnswerSimilarity（embedding-based 相似度计算）
    from ragas.metrics._answer_correctness import AnswerSimilarity
    answer_similarity = AnswerSimilarity(embeddings=eval_embeddings)

    metrics = {
        "faithfulness": Faithfulness(llm=eval_llm),
        "context_recall": ContextRecall(llm=eval_llm),
        "context_precision": ContextPrecision(llm=eval_llm),
        "answer_correctness": AnswerCorrectness(
            llm=eval_llm,
            answer_similarity=answer_similarity,
        ),
    }

    # 异步并发评估
    t_start = time.time()
    scores, details = asyncio.run(
        _run_async_evaluation(valid, metrics, concurrency)
    )
    elapsed = time.time() - t_start
    logger.info(f"RAGAS 评估完成 | 耗时 {elapsed:.1f}s | {eval_count / elapsed:.1f} 条/s")

    # 汇总统计
    summary: dict[str, dict] = {}
    for name, vals in scores.items():
        if vals:
            sorted_vals = sorted(vals)
            n = len(sorted_vals)
            summary[name] = {
                "mean": round(sum(vals) / n, 4),
                "median": round(sorted_vals[n // 2], 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "count": n,
            }
        else:
            summary[name] = {"mean": None, "count": 0}

    result = {
        "summary": summary,
        "valid_count": eval_count,
        "total": len(records),
        "evaluation_time_s": round(elapsed, 1),
        "concurrency": concurrency,
        "details": details,
    }

    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"评估结果已保存: {scores_path}")
    return result


def _print_eval_report(result: dict, scores_path: Path, output_path: Path) -> None:
    """打印评估报告摘要。"""
    print("\n" + "=" * 60)
    print("RAGAS 评估结果")
    print("=" * 60)
    print(f"有效条目: {result['valid_count']} / {result['total']}")
    if result.get("evaluation_time_s"):
        print(f"耗时: {result['evaluation_time_s']}s (并发: {result.get('concurrency', 'N/A')})")

    print("\n" + "-" * 40)
    print(f"{'指标':<22} {'均值':>8} {'中位数':>8} {'最低':>8} {'最高':>8}")
    print("-" * 40)
    for name in ["faithfulness", "context_recall", "context_precision", "answer_correctness"]:
        s = result["summary"].get(name, {})
        if s.get("mean") is not None:
            print(
                f"{name:<22} {s['mean']:>8.4f} {s['median']:>8.4f} "
                f"{s['min']:>8.4f} {s['max']:>8.4f}"
            )
        else:
            print(f"{name:<22} {'N/A':>8}")
    print("-" * 40)
    print(f"\n详细结果: {scores_path}")
    print(f"填充数据: {output_path}")


# ════════════════════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════════════════════


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="用 RAG 检索 + LLM 生成填充 RAGAS 评测集，并运行评估",
    )
    parser.add_argument(
        "--input", "-i", type=Path, default=DEFAULT_INPUT_PATH,
        help=f"输入数据集路径 (默认: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=DEFAULT_OUTPUT_PATH,
        help=f"填充后的输出路径 (默认: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--scores", "-s", type=Path, default=DEFAULT_SCORES_PATH,
        help=f"评估得分输出路径 (默认: {DEFAULT_SCORES_PATH})",
    )
    parser.add_argument(
        "--skip-fill", action="store_true",
        help="跳过检索/生成填充阶段，仅运行评估",
    )
    parser.add_argument(
        "--max-eval", type=int, default=0,
        help="最多评估 N 条（0=全部），用于快速测试",
    )
    parser.add_argument(
        "--eval-concurrency", type=int, default=5,
        help="RAGAS 评估并发数（默认 5，即同时进行 N 条样本的评分）",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    output_path: Path = args.output
    scores_path: Path = args.scores
    progress_path = output_path.parent / f".ragas_fill_progress_{output_path.stem}.json"

    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        sys.exit(1)

    records = load_dataset(input_path)
    total = len(records)
    logger.info(f"加载评测集: {total} 条")
    logger.info(f"输入: {input_path}")
    logger.info(f"输出: {output_path}")
    logger.info(f"评分: {scores_path}")

    # ── 阶段 1: 检索 + 生成填充 ──
    if args.skip_fill:
        if not output_path.exists():
            logger.error(f"--skip-fill 模式下输出文件不存在: {output_path}")
            sys.exit(1)
        logger.info("跳过检索/生成阶段，从已有文件加载")
        records = load_dataset(output_path)
    else:
        start_idx = load_progress(progress_path)
        all_filled = (
            start_idx == 0
            and output_path.exists()
            and all(
                r.get("contexts") and r.get("answer")
                for r in load_dataset(output_path)
            )
        )
        if all_filled:
            logger.info("数据集已全部填充，跳过检索/生成阶段")
            records = load_dataset(output_path)
        else:
            records = fill_dataset(records, output_path, progress_path)

    # ── 阶段 2: RAGAS 评估 ──
    result = run_ragas_evaluation(
        records, scores_path,
        max_eval=args.max_eval,
        concurrency=args.eval_concurrency,
    )
    _print_eval_report(result, scores_path, output_path)


if __name__ == "__main__":
    main()
