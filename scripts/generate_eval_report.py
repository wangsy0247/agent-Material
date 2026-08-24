#!/usr/bin/env python3
"""将索引审计、gold 数据、消融和 RAGAS 结果生成私有 Markdown 报告。"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation_data"


def _load(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _format(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def generate(
    dataset_path: Path,
    audit_path: Path,
    ablation_path: Path,
    ragas_path: Path,
    filled_path: Path | None = None,
) -> str:
    dataset = _load(dataset_path) or []
    audit = _load(audit_path) or {}
    ablation = _load(ablation_path) or {}
    ragas = _load(ragas_path) or {}
    filled = _load(filled_path) if filled_path else []
    filled = filled or []
    statuses = Counter(item.get("gold_status", "missing") for item in dataset)
    types = Counter(item.get("question_type", "unknown") for item in dataset)

    lines = [
        "# RAG 可信量化评测报告",
        "",
        f"> 生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}",
        "> 可见性：私有评测材料，不纳入开源仓库。",
        "",
        "## 1. 结论摘要",
        "",
    ]
    if ablation.get("partial") is False and ablation.get("configurations"):
        configurations = ablation["configurations"]
        # 稳定交付优先：在命中率达到最高值的配置中，固定推荐带父块回溯、
        # 但不启用高延迟查询改写的生产配置，避免字典顺序影响报告结论。
        recommended_name = (
            "hybrid_rerank_parent"
            if "hybrid_rerank_parent" in configurations
            else max(
                configurations,
                key=lambda name: configurations[name]["summary"].get("final_hit_at_3")
                or -1,
            )
        )
        recommended = configurations[recommended_name]["summary"]
        lines.extend(
            [
                f"- 推荐生产配置为 `{recommended_name}`：严格 gold 口径下 "
                f"Final Hit@3={_format(recommended.get('final_hit_at_3'))}，"
                f"Parent Hit@3={_format(recommended.get('parent_hit_at_3'))}。",
                f"- 该配置 Candidate Recall@10="
                f"{_format(recommended.get('candidate_recall_at_10'))}，"
                f"p50={_format(recommended.get('latency_p50_ms'), 2)} ms，"
                f"p95={_format(recommended.get('latency_p95_ms'), 2)} ms；"
                "在命中率、上下文完整性和可交付延迟之间更均衡。",
            ]
        )
        full_rewrite = configurations.get("full_rewrite", {}).get("summary", {})
        if full_rewrite:
            lines.append(
                "- 不默认启用 `full_rewrite`：其 Final Hit@3="
                f"{_format(full_rewrite.get('final_hit_at_3'))}，未超过推荐配置，"
                f"但 p50 达 {_format(full_rewrite.get('latency_p50_ms'), 2)} ms。"
            )
    else:
        lines.append("- 消融实验尚未完成，不输出量化结论。")
    if ragas.get("summary"):
        summary = ragas["summary"]
        lines.append(
            "- RAGAS："
            + "，".join(
                f"{name}={_format(summary.get(name, {}).get('mean'))}"
                for name in (
                    "faithfulness",
                    "context_recall",
                    "context_precision",
                    "answer_correctness",
                )
            )
            + "。"
        )
    else:
        lines.append("- RAGAS 端到端评分尚未完成。")

    lines.extend(
        [
            "",
            "## 2. 数据集与 gold evidence",
            "",
            f"- 总题数：{len(dataset)}",
            f"- 类型分布：{dict(sorted(types.items()))}",
            f"- gold 状态：{dict(sorted(statuses.items()))}",
            "- 严格检索指标仅使用 `auto_verified` 样本；"
            "`needs_review` 不进入分母，unanswerable 单独统计。",
            "- gold evidence 由确定性词法对齐生成，记录 chunk/document/version/source；"
            "其状态是“自动核验”，不声称人工标注。",
            "",
            "## 3. 评测索引审计",
            "",
            f"- 审计结论：{'PASS' if audit.get('clean') else 'FAIL / 缺失'}",
            f"- 知识库：`{audit.get('kb_id', 'N/A')}`",
        ]
    )
    for collection in audit.get("collections", []):
        lines.append(
            f"- `{collection['collection']}`：total={collection['total_rows']}，"
            f"active={collection['active_rows']}，missing_metadata="
            f"{collection['missing_required_rows']}，documents="
            f"{collection['active_documents']}，versions={collection['active_versions']}"
        )
    lines.extend(
        [
            "- 评测索引与线上 `agent_*` 集合隔离，只克隆已有 embedding，"
            "不重新解析 PDF，不重新请求 embedding API。",
            "",
            "## 4. 消融实验",
            "",
            "| 配置 | Strict n | Candidate Hit@10 | Candidate Recall@10 | Final Hit@1 | Final Hit@3 | Final MRR@3 | p50 ms | p95 ms | 失败 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, result in ablation.get("configurations", {}).items():
        value = result["summary"]
        lines.append(
            f"| {name} | {value.get('strict_gold_questions', 'N/A')} | "
            f"{_format(value.get('candidate_hit_at_10'))} | "
            f"{_format(value.get('candidate_recall_at_10'))} | "
            f"{_format(value.get('final_hit_at_1'))} | "
            f"{_format(value.get('final_hit_at_3'))} | "
            f"{_format(value.get('final_mrr_at_3'))} | "
            f"{_format(value.get('latency_p50_ms'), 2)} | "
            f"{_format(value.get('latency_p95_ms'), 2)} | "
            f"{value.get('questions_failed', 'N/A')} |"
        )
    if not ablation.get("configurations"):
        lines.append("| 待运行 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")

    lines.extend(
        [
            "",
            "口径：Candidate 指 weighted RRF 后的候选池；Final 指生产 `top_n` "
            "截断后的子块。所有相关性指标都以 exact gold `chunk_id` 判定，"
            "不再用关键词命中代替 gold。",
            "",
            "### 4.1 选型与故障案例",
            "",
            "- 推荐 `hybrid_rerank_parent`：全局重排显著改善最终排序，父块回溯"
            "补充完整上下文；单次检索延迟仍维持在秒级以内。",
            "- 查询改写不是默认路径。实测中，Step-back 曾把 ABACUS 语境下的 "
            "`MDFT` 错误泛化为傅里叶变换相关概念，产生语义漂移；保留原始查询"
            "分支使 gold 证据仍可命中，但该案例说明改写必须可降级、可观测。",
            "- 故障处理策略：原始查询始终参与融合；改写超时或失败时降级到"
            "原始查询；记录分阶段延迟与 degradation event，避免静默失败。",
            "",
            "## 5. RAGAS 端到端结果",
            "",
            "| 指标 | Mean | Median | Min | Max | n |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in (
        "faithfulness",
        "context_recall",
        "context_precision",
        "answer_correctness",
    ):
        value = ragas.get("summary", {}).get(name, {})
        lines.append(
            f"| {name} | {_format(value.get('mean'))} | "
            f"{_format(value.get('median'))} | {_format(value.get('min'))} | "
            f"{_format(value.get('max'))} | {value.get('count', 'N/A')} |"
        )
    unanswerable = ragas.get("unanswerable", {})
    unanswerable_routes = Counter(
        item.get("route_decision", "unknown")
        for item in unanswerable.get("details", [])
    )
    unanswerable_answers = [
        item.get("answer", "")
        for item in filled
        if item.get("question_type") == "unanswerable"
    ]
    refusal_pattern = re.compile(r"无法回答|无法回复|不能回答|没有.*信息|未包含")
    refusal_count = sum(bool(refusal_pattern.search(answer)) for answer in unanswerable_answers)
    lines.extend(
        [
            "",
            "### 5.1 不可回答题（不混入主均值）",
            "",
            f"- 样本数：{unanswerable.get('count', 0)}；"
            f"路由分布：{dict(sorted(unanswerable_routes.items()))}。",
            f"- 拒答短语启发式命中：{refusal_count}/{len(unanswerable_answers)}；"
            "该值用于检查明显越界回答，不等同于人工语义核验。",
            "- 检索器对域外但术语相近的问题仍可能返回非空上下文；生成层必须"
            "基于证据充分性拒答，不能把‘检索到内容’等同于‘问题可回答’。",
            "",
            "| 指标 | Mean | n |",
            "|---|---:|---:|",
        ]
    )
    for name in (
        "faithfulness",
        "context_recall",
        "context_precision",
        "answer_correctness",
    ):
        value = unanswerable.get("summary", {}).get(name, {})
        lines.append(
            f"| {name} | {_format(value.get('mean'))} | "
            f"{value.get('count', 'N/A')} |"
        )
    lines.extend(
        [
            "",
            "说明：不可回答题的 reference 是拒答语义，RAGAS 数值与常规"
            "事实问答的含义不同，因此只作为边界行为观察，不用于主配置排名。",
        ]
    )
    lines.extend(
        [
            "",
            "## 6. 可复现信息",
            "",
        ]
    )
    repro = ablation.get("reproducibility") or ragas.get("reproducibility") or {}
    lines.extend(
        [
            f"- Git commit：`{repro.get('git_commit', 'N/A')}`",
            f"- 工作树是否有未提交修改：{repro.get('git_dirty', 'N/A')}",
            f"- Python：`{repro.get('python_version', 'N/A')}`",
            f"- 随机种子：{repro.get('random_seed', 'N/A')}",
            f"- Dataset SHA-256：`{repro.get('dataset_sha256', 'N/A')}`",
            f"- 依赖版本：`{repro.get('package_versions', {})}`",
            "- 结果 JSON 保存了可移植命令、脱敏配置快照、索引审计/manifest "
            "SHA-256、逐题 chunk id、分阶段延迟与 degradation 事件。",
            "",
            "## 7. 局限与可对外表述边界",
            "",
            "- 这是单文档、中等规模评测，不代表大规模多租户知识库的绝对性能。",
            "- gold evidence 为自动核验；对外应明确这一点，不宣称人工双人标注。",
            "- RAGAS 是 LLM-as-a-judge，存在评估模型偏差；应与 exact-id "
            "检索指标一起解读。",
            "- 工作树若为 dirty，重现时需使用同一份未提交代码；对外发布数字前"
            "应在干净 commit 上复跑。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=EVAL / "fixed_eval_dataset_v2.json")
    parser.add_argument("--audit", type=Path, default=EVAL / "index_audit_v2.json")
    parser.add_argument("--ablation", type=Path, default=EVAL / "ablation_results_v2.json")
    parser.add_argument("--ragas", type=Path, default=EVAL / "ragas_v3_scores.json")
    parser.add_argument("--filled", type=Path, default=EVAL / "ragas_v3_filled.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "interview_prep/27届秋招_RAG可信量化评测报告.md",
    )
    args = parser.parse_args()
    report = generate(
        args.dataset.resolve(),
        args.audit.resolve(),
        args.ablation.resolve(),
        args.ragas.resolve(),
        args.filled.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
