"""Chunk 切分后评估与监控模块。

功能：
1. 加载清洗后的文档并进行切分
2. 多维度抽样（随机/分层/长度异常/边界问题）
3. 自动标记可疑 chunk（噪音残留、语义断裂等）
4. 生成人工评估模板
5. 评估结果记录与监控指标追踪

Usage:
    python chunk_evaluator.py \
        --input ./workspace/abacus_cleaned.jsonl \
        --output ./workspace/eval_report.json \
        --sample-size 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from rag_mcp_server.utils.logger import logger

try:
    from rag_mcp_server.rag.text_spliter import SemanticTextSplitter
    from rag_mcp_server.models import embedding_model

    SEMANTIC_SPLITTER_AVAILABLE = True
except ImportError:
    SEMANTIC_SPLITTER_AVAILABLE = False
    logger.warning("SemanticTextSplitter 不可用，将使用简单切分")


# ==================================================================== #
#  数据模型
# ==================================================================== #
@dataclass
class ChunkMetrics:
    """单个 chunk 的度量指标。"""

    chunk_id: str
    source: str
    page_num: Optional[int] = None
    char_length: int = 0
    word_count: int = 0
    line_count: int = 0
    has_code: bool = False
    has_table: bool = False
    has_url: bool = False
    has_cid_residual: bool = False
    has_header_residual: bool = False
    boundary_issue: bool = False
    avg_line_length: float = 0.0
    chinese_ratio: float = 0.0


@dataclass
class EvaluationRecord:
    """单条人工评估记录。"""

    chunk_id: str
    source: str
    content_preview: str
    evaluated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    evaluator: str = ""

    # 五维度评分 (1-5)
    semantic_completeness: int = 0
    information_density: int = 0
    context_independence: int = 0
    readability: int = 0
    boundary_quality: int = 0

    # 问题标注
    issues: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def overall_score(self) -> float:
        if all(v > 0 for v in [
            self.semantic_completeness,
            self.information_density,
            self.context_independence,
            self.readability,
            self.boundary_quality,
        ]):
            return sum([
                self.semantic_completeness,
                self.information_density,
                self.context_independence,
                self.readability,
                self.boundary_quality,
            ]) / 5.0
        return 0.0


# ==================================================================== #
#  Chunk 切分器包装
# ==================================================================== #
class ChunkSplitter:
    """文档切分包装器，支持 SemanticTextSplitter 和回退方案。"""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        use_semantic: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic = use_semantic and SEMANTIC_SPLITTER_AVAILABLE

        if self.use_semantic:
            self.splitter = SemanticTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_model=embedding_model,
                similarity_threshold=0.75,
            )
            logger.info("使用 SemanticTextSplitter 进行切分")
        else:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""],
                length_function=len,
            )
            logger.info("使用 RecursiveCharacterTextSplitter 进行切分")

    def split(self, pages: list[dict]) -> list[Document]:
        """将清洗后的页面切分为 chunk。

        Args:
            pages: 清洗后的页面列表，每个元素含 "cleaned_text" 和 "source" 等

        Returns:
            LangChain Document 列表
        """
        documents = []
        for page in pages:
            # 兼容不同来源的字段名
            content = page.get("cleaned_text") or page.get("text") or ""
            doc = Document(
                page_content=content,
                metadata={
                    "source": page.get("source", "unknown"),
                    "page_num": page.get("page_num"),
                    "tags": page.get("tags", []),
                },
            )
            documents.append(doc)

        return self.splitter.split_documents(documents)


# ==================================================================== #
#  抽样器
# ==================================================================== #
class ChunkSampler:
    """Chunk 多维度抽样器。"""

    @staticmethod
    def random_sample(chunks: list[Document], n: int = 50) -> list[Document]:
        """随机抽样。"""
        k = min(n, len(chunks))
        return random.sample(chunks, k)

    @staticmethod
    def stratified_by_source(chunks: list[Document], n_per_source: int = 3) -> list[Document]:
        """按来源文件分层抽样。"""
        from collections import defaultdict

        by_source = defaultdict(list)
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            by_source[source].append(chunk)

        samples = []
        for source, source_chunks in by_source.items():
            k = min(n_per_source, len(source_chunks))
            samples.extend(random.sample(source_chunks, k))
        return samples

    @staticmethod
    def length_anomaly(chunks: list[Document], low_pct: float = 5, high_pct: float = 95) -> list[Document]:
        """按长度异常抽样（过长/过短）。"""
        lengths = [len(c.page_content) for c in chunks]
        sorted_lengths = sorted(lengths)
        low_threshold = sorted_lengths[int(len(lengths) * low_pct / 100)]
        high_threshold = sorted_lengths[int(len(lengths) * high_pct / 100)]

        return [
            c for c in chunks
            if len(c.page_content) < low_threshold or len(c.page_content) > high_threshold
        ]

    @staticmethod
    def boundary_issues(chunks: list[Document]) -> list[Document]:
        """抽样边界可能有问题的 chunk。

        检测：以不完整词开头或结尾。
        """
        suspicious_endings = [
            " the", " a", " an", " in", " of", " for", " with", " by", " to", " and", " or",
        ]
        suspicious_startings = ["tion", "ing", "ed", "ly", "ment", "ness", "able", "ible"]

        suspicious = []
        for chunk in chunks:
            text = chunk.page_content.strip()
            if not text:
                continue

            words = text.split()
            first_word = words[0].lower() if words else ""
            last_word = words[-1].lower() if words else ""

            # 检查开头
            if any(first_word.endswith(s) for s in suspicious_startings):
                suspicious.append(chunk)
                continue

            # 检查结尾
            if any(last_word == s.strip() for s in suspicious_endings):
                suspicious.append(chunk)
                continue

            # 检查结尾标点
            if text and text[-1] not in ".。！?？;；:":
                # 非完整句子结尾
                if len(text) < 100:  # 短的非完整句子更可疑
                    suspicious.append(chunk)
                    continue

        return suspicious


# ==================================================================== #
#  自动质检器
# ==================================================================== #
class ChunkAutoFlagger:
    """自动标记可疑 chunk。"""

    @staticmethod
    def flag(chunks: list[Document]) -> dict[str, list[dict]]:
        """返回分类的可疑 chunk 列表。

        Returns:
            {"too_short": [...], "too_long": [...], ...}
        """
        flags: dict[str, list[dict]] = {
            "too_short": [],
            "too_long": [],
            "likely_header_residual": [],
            "likely_copyright_residual": [],
            "formula_garbage": [],
            "repetitive": [],
            "boundary_issues": [],
            "low_chinese_ratio": [],
            "url_dominant": [],
        }

        content_hashes: dict[str, int] = {}

        for i, chunk in enumerate(chunks):
            text = chunk.page_content
            meta = chunk.metadata
            record = {
                "index": i,
                "source": meta.get("source", "N/A"),
                "page_num": meta.get("page_num"),
                "preview": text[:120],
            }

            # 过短
            if len(text) < 30:
                flags["too_short"].append(record)

            # 过长
            if len(text) > 3000:
                flags["too_long"].append(record)

            # 页眉残留
            if re.search(r"J\.\s*Phys|Nature\s*\d|Science\s*\d|目录\s*$", text, re.I):
                flags["likely_header_residual"].append(record)

            # 版权残留
            if re.search(r"Copyright|All rights reserved|版权声明", text, re.I):
                flags["likely_copyright_residual"].append(record)

            # CID 残留
            if "(cid:" in text:
                flags["formula_garbage"].append(record)

            # 重复检测（前100字符哈希）
            h = hashlib.md5(text[:100].encode()).hexdigest()
            if h in content_hashes:
                flags["repetitive"].append({
                    **record,
                    "duplicate_of": content_hashes[h],
                })
            else:
                content_hashes[h] = i

            # 中文比例过低
            chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
            if len(text) > 100 and chinese_chars / len(text) < 0.05:
                flags["low_chinese_ratio"].append(record)

            # URL 过多
            urls = re.findall(r"https?://\S+", text)
            if len(urls) > 3 and sum(len(u) for u in urls) / len(text) > 0.3:
                flags["url_dominant"].append(record)

        return flags


# ==================================================================== #
#  评估报告生成器
# ==================================================================== #
class EvaluationReporter:
    """人工评估报告管理。"""

    @staticmethod
    def generate_evaluation_template(chunk: Document, chunk_id: str) -> str:
        """为单个 chunk 生成人工评估模板。"""
        text = chunk.page_content
        meta = chunk.metadata

        template = f"""
{'=' * 60}
Chunk 人工评估记录
{'=' * 60}

评估人: ____________
评估日期: {datetime.now().strftime('%Y-%m-%d')}

--- Chunk 信息 ---
Chunk ID: {chunk_id}
来源: {meta.get('source', 'N/A')}
页码: {meta.get('page_num', 'N/A')}
字符长度: {len(text)}
词数: {len(text.split())}
行数: {len(text.split(chr(10)))}

--- 内容预览 ---
{text[:500]}{'...' if len(text) > 500 else ''}

--- 评分 (1-5 分) ---
[ ] 语义完整性: _____  (5=完整语义单元, 1=句子被切断)
[ ] 信息密度: _____    (5=每句都有价值, 1=大量空白/噪音)
[ ] 上下文独立性: _____ (5=脱离原文能理解, 1=大量指代)
[ ] 可读性: _____      (5=格式清晰无乱码, 1=格式混乱)
[ ] 边界质量: _____    (5=在段落边界切开, 1=在词中间切开)

综合评分: _____

--- 问题标注 ---
[ ] 句子被切断
[ ] 页眉/版权残留
[ ] 参考文献混入
[ ] 公式/代码乱码
[ ] 孤立图表说明
[ ] URL 被截断
[ ] 其他: ____________

--- 备注 ---
_________________________________
"""
        return template.strip()

    @staticmethod
    def save_report(
        report_path: str,
        chunks: list[Document],
        samples: list[Document],
        auto_flags: dict,
        human_scores: list[EvaluationRecord],
        split_config: dict,
    ) -> None:
        """保存完整评估报告。"""
        avg_scores = {}
        if human_scores:
            for key in ["semantic_completeness", "information_density", "context_independence", "readability", "boundary_quality"]:
                values = [getattr(s, key) for s in human_scores if getattr(s, key) > 0]
                avg_scores[key] = round(sum(values) / len(values), 2) if values else 0

        report = {
            "generated_at": datetime.now().isoformat(),
            "total_chunks": len(chunks),
            "sample_size": len(samples),
            "split_config": split_config,
            "auto_flags_summary": {k: len(v) for k, v in auto_flags.items()},
            "average_scores": avg_scores,
            "human_scores": [asdict(s) for s in human_scores],
        }

        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"[EvaluationReporter] 报告已保存: {report_path}")


# ==================================================================== #
#  监控指标记录器
# ==================================================================== #
class ChunkMonitor:
    """轻量级 Chunk 监控器。"""

    def __init__(self, log_dir: str = "logs/chunk_monitoring"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_ingestion(self, metrics: dict) -> None:
        """记录入库事件。"""
        record = {
            "event": "ingestion",
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }
        self._append("ingestion.jsonl", record)

    def log_splitting(self, split_metrics: dict) -> None:
        """记录切分事件。"""
        record = {
            "event": "splitting",
            "timestamp": datetime.now().isoformat(),
            **split_metrics,
        }
        self._append("splitting.jsonl", record)

    def log_evaluation(self, eval_result: dict) -> None:
        """记录评估事件。"""
        record = {
            "event": "evaluation",
            "timestamp": datetime.now().isoformat(),
            **eval_result,
        }
        self._append("evaluation.jsonl", record)

    def _append(self, filename: str, record: dict) -> None:
        filepath = self.log_dir / filename
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def generate_daily_summary(self, date_str: Optional[str] = None) -> dict:
        """生成日报摘要。"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        summaries: dict[str, list[dict]] = {"ingestion": [], "splitting": [], "evaluation": []}

        for event_type in summaries:
            log_file = self.log_dir / f"{event_type}.jsonl"
            if not log_file.exists():
                continue
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if record.get("timestamp", "").startswith(date_str):
                            summaries[event_type].append(record)
                    except json.JSONDecodeError:
                        continue

        return {
            "date": date_str,
            "ingestion_count": len(summaries["ingestion"]),
            "splitting_count": len(summaries["splitting"]),
            "evaluation_count": len(summaries["evaluation"]),
            "details": summaries,
        }


# ==================================================================== #
#  主评估流程
# ==================================================================== #
class ChunkEvaluationPipeline:
    """Chunk 评估完整 Pipeline。"""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        sample_size: int = 100,
        output_dir: str = "workspace/evaluation",
    ):
        self.splitter = ChunkSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.sampler = ChunkSampler()
        self.flagger = ChunkAutoFlagger()
        self.reporter = EvaluationReporter()
        self.monitor = ChunkMonitor(log_dir=f"{output_dir}/monitoring")
        self.sample_size = sample_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        pages: list[dict],
        split_config: Optional[dict] = None,
    ) -> dict:
        """执行完整评估流程。

        Args:
            pages: 清洗后的页面列表
            split_config: 切分配置信息（用于报告记录）

        Returns:
            评估结果字典
        """
        logger.info("[ChunkEvaluationPipeline] 开始评估流程")

        # 1. 切分
        logger.info("Step 1: 文档切分...")
        chunks = self.splitter.split(pages)
        logger.info(f"  生成 {len(chunks)} 个 chunk")

        self.monitor.log_splitting({
            "total_chunks": len(chunks),
            "chunk_size": self.splitter.chunk_size,
            "chunk_overlap": self.splitter.chunk_overlap,
            "use_semantic": self.splitter.use_semantic,
        })

        # 2. 自动标记
        logger.info("Step 2: 自动质检...")
        auto_flags = self.flagger.flag(chunks)
        total_flags = sum(len(v) for v in auto_flags.values())
        logger.info(f"  自动标记问题: {total_flags} 个")

        # 3. 多维度抽样
        logger.info("Step 3: 多维度抽样...")
        samples: set[int] = set()

        # 3.1 随机抽样
        random_samples = self.sampler.random_sample(chunks, n=self.sample_size // 2)
        samples.update(id(c) for c in random_samples)

        # 3.2 分层抽样
        stratified_samples = self.sampler.stratified_by_source(chunks, n_per_source=2)
        samples.update(id(c) for c in stratified_samples)

        # 3.3 异常抽样
        anomaly_samples = self.sampler.length_anomaly(chunks)
        samples.update(id(c) for c in anomaly_samples[:20])

        # 3.4 边界问题抽样
        boundary_samples = self.sampler.boundary_issues(chunks)
        samples.update(id(c) for c in boundary_samples[:20])

        # 去重并限制数量
        unique_samples = [c for c in chunks if id(c) in samples]
        if len(unique_samples) > self.sample_size:
            unique_samples = random.sample(unique_samples, self.sample_size)

        logger.info(f"  抽样数量: {len(unique_samples)}")

        # 4. 生成评估模板
        logger.info("Step 4: 生成人工评估模板...")
        template_path = self.output_dir / "evaluation_templates.txt"
        with open(template_path, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(unique_samples):
                chunk_id = f"chunk_{i:04d}"
                template = self.reporter.generate_evaluation_template(chunk, chunk_id)
                f.write(template + "\n\n")

        logger.info(f"  评估模板已保存: {template_path}")

        # 5. 保存自动标记报告
        flag_report_path = self.output_dir / "auto_flag_report.json"
        with open(flag_report_path, "w", encoding="utf-8") as f:
            json.dump(auto_flags, f, ensure_ascii=False, indent=2)

        logger.info(f"  自动标记报告已保存: {flag_report_path}")

        # 6. 保存评估框架（供后续人工打分后填充）
        score_template_path = self.output_dir / "score_template.json"
        empty_scores = [
            {
                "chunk_id": f"chunk_{i:04d}",
                "source": c.metadata.get("source", "N/A"),
                "page_num": c.metadata.get("page_num"),
                "content_preview": c.page_content[:200],
                "semantic_completeness": 0,
                "information_density": 0,
                "context_independence": 0,
                "readability": 0,
                "boundary_quality": 0,
                "issues": [],
                "notes": "",
            }
            for i, c in enumerate(unique_samples)
        ]
        with open(score_template_path, "w", encoding="utf-8") as f:
            json.dump(empty_scores, f, ensure_ascii=False, indent=2)

        logger.info(f"  评分模板已保存: {score_template_path}")

        # 7. 保存 chunk 监控基线
        self.monitor.log_ingestion({
            "total_chunks": len(chunks),
            "avg_chunk_length": sum(len(c.page_content) for c in chunks) / max(len(chunks), 1),
            "median_chunk_length": sorted(len(c.page_content) for c in chunks)[len(chunks) // 2] if chunks else 0,
            "min_chunk_length": min((len(c.page_content) for c in chunks), default=0),
            "max_chunk_length": max((len(c.page_content) for c in chunks), default=0),
            "auto_flag_count": total_flags,
        })

        logger.info("[ChunkEvaluationPipeline] 评估流程完成")

        return {
            "total_chunks": len(chunks),
            "sample_size": len(unique_samples),
            "auto_flag_count": total_flags,
            "output_dir": str(self.output_dir),
            "template_path": str(template_path),
            "flag_report_path": str(flag_report_path),
            "score_template_path": str(score_template_path),
        }


# ==================================================================== #
#  命令行接口
# ==================================================================== #
def main():
    parser = argparse.ArgumentParser(description="Chunk 切分后评估与监控工具")
    parser.add_argument("--input", "-i", required=True, help="清洗后的 JSONL 文件路径")
    parser.add_argument("--output", "-o", default="workspace/evaluation", help="输出目录")
    parser.add_argument("--chunk-size", type=int, default=800, help="切分块大小")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="切分重叠大小")
    parser.add_argument("--sample-size", type=int, default=100, help="抽样数量")
    parser.add_argument("--no-semantic", action="store_true", help="不使用语义切分")

    args = parser.parse_args()

    # 加载清洗后的页面
    pages = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            pages.append(json.loads(line))

    # 运行评估
    pipeline = ChunkEvaluationPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        sample_size=args.sample_size,
        output_dir=args.output,
    )

    # 强制控制是否使用语义切分
    if args.no_semantic:
        pipeline.splitter.use_semantic = False

    result = pipeline.run(pages)

    print("\n===== 评估完成 =====")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
