"""Chunk 切分后评估与监控模块。

功能：
1. 直接对接 ProcessingPipeline 输出（list[Document]），也支持 JSONL 输入
2. 自动评估指标（语义连贯性、长度均匀度、边界质量、信息密度、冗余度）
3. 混合切分策略对比（多组配置并行评估，生成对比报告）

Usage:
    # 方式 1：直接评估 Pipeline 输出
    python -c "
    from agent_material.rag_engine.document_pipeline import ProcessingPipeline
    from agent_material.rag_engine.utils.chunk_evaluator import ChunkEvaluator
    pipeline = ProcessingPipeline()
    docs = pipeline.process('paper.pdf')
    evaluator = ChunkEvaluator()
    report = evaluator.evaluate(docs)
    evaluator.print_report(report)
    "

    # 方式 2：对比多组切分策略
    python chunk_evaluator.py --compare --input paper.pdf

    # 方式 3：传统 JSONL 输入
    python chunk_evaluator.py --input cleaned.jsonl --output ./eval/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from langchain_core.documents import Document

from agent_material.utils.logger import logger


# ==================================================================== #
#  数据模型
# ==================================================================== #


@dataclass
class AutoMetrics:
    """单组切分策略的自动评估指标。"""

    strategy_name: str = ""
    total_chunks: int = 0
    total_chars: int = 0

    # 长度统计
    avg_length: float = 0.0
    median_length: int = 0
    min_length: int = 0
    max_length: int = 0
    length_std: float = 0.0        # 标准差，越低越均匀
    length_uniformity: float = 0.0  # 0~1，越高越均匀

    # 语义连贯性（相邻 chunk 相似度统计）
    avg_adjacent_similarity: float = 0.0
    min_adjacent_similarity: float = 0.0
    cross_section_boundaries: int = 0  # 跨章节边界的 chunk 数

    # 边界质量
    sentence_start_ratio: float = 0.0   # 以完整句子开头的比例
    sentence_end_ratio: float = 0.0     # 以完整句子结尾的比例
    boundary_score: float = 0.0         # 综合边界质量

    # 信息密度
    avg_info_density: float = 0.0       # 有效信息占比

    # 冗余度
    duplicate_ratio: float = 0.0        # 重复 chunk 比例
    near_duplicate_pairs: int = 0       # 近似重复对

    # 噪音残留
    cid_residual_count: int = 0
    header_residual_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CompareReport:
    """多组策略对比报告。"""

    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    input_summary: dict = field(default_factory=dict)
    strategies: list[AutoMetrics] = field(default_factory=list)
    ranking: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "input_summary": self.input_summary,
            "strategies": [s.to_dict() for s in self.strategies],
            "ranking": self.ranking,
        }


@dataclass
class EvaluationRecord:
    """单条人工评估记录。"""

    chunk_id: str
    source: str
    content_preview: str
    evaluated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    semantic_completeness: int = 0
    information_density: int = 0
    context_independence: int = 0
    readability: int = 0
    boundary_quality: int = 0
    issues: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def overall_score(self) -> float:
        scores = [
            self.semantic_completeness, self.information_density,
            self.context_independence, self.readability, self.boundary_quality,
        ]
        if all(v > 0 for v in scores):
            return sum(scores) / 5.0
        return 0.0


# ==================================================================== #
#  Chunk 抽样器
# ==================================================================== #


class ChunkSampler:
    """多维度抽样器。"""

    @staticmethod
    def random_sample(chunks: list[Document], n: int = 50) -> list[Document]:
        k = min(n, len(chunks))
        return random.sample(chunks, k) if chunks else []

    @staticmethod
    def stratified_by_source(chunks: list[Document], n_per_source: int = 3) -> list[Document]:
        by_source = defaultdict(list)
        for c in chunks:
            by_source[c.metadata.get("source", "unknown")].append(c)
        samples = []
        for source_chunks in by_source.values():
            k = min(n_per_source, len(source_chunks))
            samples.extend(random.sample(source_chunks, k))
        return samples

    @staticmethod
    def length_anomaly(chunks: list[Document], low_pct: float = 5, high_pct: float = 95) -> list[Document]:
        lengths = sorted(len(c.page_content) for c in chunks)
        if not lengths:
            return []
        low = lengths[int(len(lengths) * low_pct / 100)]
        high = lengths[int(len(lengths) * high_pct / 100)]
        return [c for c in chunks if len(c.page_content) < low or len(c.page_content) > high]

    @staticmethod
    def boundary_issues(chunks: list[Document]) -> list[Document]:
        suspicious_endings = [" the", " a", " an", " in", " of", " for", " with", " by", " to"]
        suspicious_startings = ["tion", "ing", "ed", "ly", "ment", "ness"]
        suspicious = []
        for c in chunks:
            text = c.page_content.strip()
            if not text:
                continue
            words = text.split()
            if not words:
                continue
            if any(words[0].lower().endswith(s) for s in suspicious_startings):
                suspicious.append(c)
            elif any(words[-1].lower() == s.strip() for s in suspicious_endings):
                suspicious.append(c)
            elif text[-1] not in ".。！?？;；:\"" and len(text) < 100:
                suspicious.append(c)
        return suspicious


# ==================================================================== #
#  Chunk 自动质检器（噪音 / 残留检测）
# ==================================================================== #


class ChunkAutoFlagger:
    """自动标记可疑 chunk。"""

    @staticmethod
    def flag(chunks: list[Document]) -> dict[str, list[dict]]:
        flags: dict[str, list[dict]] = {
            "too_short": [], "too_long": [],
            "header_residual": [], "copyright_residual": [],
            "cid_garbage": [], "repetitive": [],
            "url_dominant": [], "low_chinese_ratio": [],
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
            if len(text) < 30:
                flags["too_short"].append(record)
            if len(text) > 3000:
                flags["too_long"].append(record)
            if re.search(r"J\.\s*Phys|Nature\s*\d|Science\s*\d|目录\s*$", text, re.I):
                flags["header_residual"].append(record)
            if re.search(r"Copyright|All rights reserved|版权声明", text, re.I):
                flags["copyright_residual"].append(record)
            if "(cid:" in text:
                flags["cid_garbage"].append(record)
            h = hashlib.md5(text[:100].encode()).hexdigest()
            if h in content_hashes:
                flags["repetitive"].append({**record, "duplicate_of": content_hashes[h]})
            else:
                content_hashes[h] = i
            chinese = sum(1 for c in text if "一" <= c <= "鿿")
            if len(text) > 100 and chinese / len(text) < 0.05:
                flags["low_chinese_ratio"].append(record)
            urls = re.findall(r"https?://\S+", text)
            if len(urls) > 3 and sum(len(u) for u in urls) / len(text) > 0.3:
                flags["url_dominant"].append(record)

        return flags


# ==================================================================== #
#  Chunk 监控器
# ==================================================================== #


class ChunkMonitor:
    """轻量级 Chunk 监控器，记录入库 / 切分 / 评估事件。"""

    def __init__(self, log_dir: str = "logs/chunk_monitoring"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, metrics: dict) -> None:
        record = {"event": event_type, "timestamp": datetime.now().isoformat(), **metrics}
        filepath = self.log_dir / f"{event_type}.jsonl"
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def generate_daily_summary(self, date_str: str | None = None) -> dict:
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
                        r = json.loads(line)
                        if r.get("timestamp", "").startswith(date_str):
                            summaries[event_type].append(r)
                    except json.JSONDecodeError:
                        continue
        return {
            "date": date_str,
            **{f"{k}_count": len(v) for k, v in summaries.items()},
            "details": summaries,
        }


# ==================================================================== #
#  ChunkEvaluator — 核心评估器（修复版）
# ==================================================================== #


class ChunkEvaluator:
    """Chunk 自动评估器 — 支持 Pipeline 输出 + 自动指标 + 策略对比。"""

    # 句尾标点（chunk 以这些结尾 → 句尾完整）
    _RE_SENTENCE_END = re.compile(r"[。！？…》」』】）\"\”\.!\?:]\s*$")
    # 句首"非完整"模式（chunk 以这些开头 → 句首不完整）
    _RE_NOT_SENTENCE_START = re.compile(
        r"^[ 　]*("
        r"[a-z]"                               # 小写英文 → 续文
        r"|[,，;；:：、。！？…\)】」》]"          # 标点开头 → 碎片
        r"|\d+[\.\,\)、\s]*[a-z]"               # 数字 + 小写 → 编号续文
        r"|[的了吗着呢过地与得]"                  # 中文虚词开头 → 续文
        r"|https?://"                           # URL 开头 → 碎片
        r")",
    )
    # 英文词尾碎片（单独 regex，需 IGNORECASE）
    _RE_EN_FRAGMENT = re.compile(
        r"^[ 　]*(?:tion|ing|ment|ness|able|ful|ize|ly|ed|er|est)\b",
        re.IGNORECASE,
    )

    def _is_sentence_start(self, text: str) -> bool:
        """检查文本是否以完整句首开头。"""
        if not text:
            return False
        return (
            not self._RE_NOT_SENTENCE_START.match(text)
            and not self._RE_EN_FRAGMENT.match(text)
        )

    def _is_sentence_end(self, text: str) -> bool:
        """检查文本是否以完整句尾结尾。"""
        return bool(self._RE_SENTENCE_END.search(text.rstrip()))

    # 停止词（高密度则信息量低）
    _RE_STOP = re.compile(
        r"\b(the|a|an|is|are|was|were|been|be|in|on|at|to|for|of|with|by|from|"
        r"and|or|but|not|this|that|these|those|it|its|they|them|we|you|he|she|"
        r"的|了|在|是|有|和|也|就|都|而|及|与|着|或|一个|没有|我们|他们|它们|这个|那个|"
        r"可以|可能|能够|应该|需要|已经|还是|这些|那些|如果|虽然|但是|因为|所以|因此|然而|而且)\b"
    )

    def __init__(self, output_dir: str = "workspace/evaluation"):
        self.monitor = ChunkMonitor(log_dir=f"{output_dir}/monitoring")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  修复 1：直接接受 list[Document]（ProcessingPipeline 输出）
    # ------------------------------------------------------------------ #

    @staticmethod
    def from_pipeline(chunks: list[Document]) -> list[Document]:
        """直接接受 ProcessingPipeline.process() 输出的 list[Document]。
        无需 JSONL 中间文件。
        """
        return chunks

    @staticmethod
    def from_jsonl(path: str) -> list[Document]:
        """从 JSONL 文件加载。"""
        docs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                docs.append(Document(
                    page_content=record.get("text") or record.get("cleaned_text", ""),
                    metadata={
                        "source": record.get("source", path),
                        "page_num": record.get("page_num"),
                        "tags": record.get("tags", []),
                    },
                ))
        return docs

    # ------------------------------------------------------------------ #
    #  修复 2：自动评估指标
    # ------------------------------------------------------------------ #

    def compute_metrics(
        self,
        chunks: list[Document],
        strategy_name: str = "default",
    ) -> AutoMetrics:
        """计算一组 chunk 的全部自动评估指标。"""
        if not chunks:
            return AutoMetrics(strategy_name=strategy_name)

        m = AutoMetrics(strategy_name=strategy_name)
        m.total_chunks = len(chunks)

        lengths = [len(c.page_content) for c in chunks]
        m.total_chars = sum(lengths)
        m.min_length = min(lengths)
        m.max_length = max(lengths)
        m.avg_length = m.total_chars / len(chunks)
        sorted_lens = sorted(lengths)
        m.median_length = sorted_lens[len(sorted_lens) // 2]

        # 长度均匀度
        if len(chunks) > 1:
            mean = m.avg_length
            variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
            m.length_std = math.sqrt(variance)
            # 归一化：std/mean 越小越均匀，截断为 0~1
            cv = m.length_std / mean if mean > 0 else 1
            m.length_uniformity = max(0.0, 1.0 - min(cv, 1.0))

        # 边界质量
        m.sentence_start_ratio = sum(
            1 for c in chunks if self._is_sentence_start(c.page_content.lstrip())
        ) / len(chunks)
        m.sentence_end_ratio = sum(
            1 for c in chunks if self._is_sentence_end(c.page_content)
        ) / len(chunks)
        m.boundary_score = (m.sentence_start_ratio + m.sentence_end_ratio) / 2

        # 信息密度
        densities = []
        for c in chunks:
            text = c.page_content
            if not text:
                continue
            # 停止词占比越低，信息密度越高
            stop_chars = sum(len(m.group()) for m in self._RE_STOP.finditer(text))
            densities.append(max(0.0, 1.0 - stop_chars / len(text)))
        m.avg_info_density = sum(densities) / len(densities) if densities else 0

        # 跨章节边界检测
        for c in chunks:
            path = c.metadata.get("section_path", "")
            content = c.page_content
            # chunk 内容包含了不同于其 metadata 的标题 → 跨了边界
            heading_markers = re.findall(r"\[H[0-3]\]\s+(.+)", content)
            if heading_markers and path:
                # 如果内容中的标题不在当前 section_path 中，可能是跨边界
                for hm in heading_markers:
                    if hm.strip() not in path:
                        m.cross_section_boundaries += 1
                        break

        # 冗余检测
        text_hashes: dict[str, int] = {}
        for i, c in enumerate(chunks):
            h = hashlib.md5(c.page_content[:200].encode()).hexdigest()
            if h in text_hashes:
                m.near_duplicate_pairs += 1
            else:
                text_hashes[h] = i
        m.duplicate_ratio = m.near_duplicate_pairs / len(chunks) if chunks else 0

        # 语义连贯性（相邻 chunk 相似度）
        m.avg_adjacent_similarity, m.min_adjacent_similarity = \
            self._compute_adjacent_similarity(chunks)

        # 噪音残留
        for c in chunks:
            if "(cid:" in c.page_content:
                m.cid_residual_count += 1
            if re.search(r"Copyright|All rights reserved|版权声明|第\s*\d+\s*页\s*$",
                         c.page_content):
                m.header_residual_count += 1

        # 日志 + 监控
        self.monitor.log_event("evaluation", {
            "strategy": strategy_name,
            "total_chunks": m.total_chunks,
            "avg_length": round(m.avg_length, 1),
            "boundary_score": round(m.boundary_score, 3),
            "length_uniformity": round(m.length_uniformity, 3),
            "duplicate_ratio": round(m.duplicate_ratio, 4),
            "cid_residual": m.cid_residual_count,
        })

        return m

    @staticmethod
    def _compute_adjacent_similarity(chunks: list[Document]) -> tuple[float, float]:
        """计算相邻 chunk 的文本相似度（基于 token 重叠，不依赖 embedding）。

        使用 Jaccard 相似度：两个相邻 chunk 共享的 bigram 比例。
        """
        if len(chunks) < 2:
            return 0.0, 0.0

        similarities: list[float] = []
        for i in range(len(chunks) - 1):
            a_tokens = set(_bigrams(chunks[i].page_content))
            b_tokens = set(_bigrams(chunks[i + 1].page_content))
            union = len(a_tokens | b_tokens)
            if union > 0:
                sim = len(a_tokens & b_tokens) / union
                similarities.append(sim)

        if not similarities:
            return 0.0, 0.0
        return sum(similarities) / len(similarities), min(similarities)

    # ------------------------------------------------------------------ #
    #  修复 3：多策略对比
    # ------------------------------------------------------------------ #

    def compare(
        self,
        file_path: str,
        strategies: dict[str, list[str]] | None = None,
    ) -> CompareReport:
        """使用多组切分策略处理同一个文件，生成对比报告。

        Args:
            file_path: 待评估的 PDF/MD/HTML 文件路径
            strategies: 策略名 → 步骤列表，默认对比三组:
                {"仅递归": ["recursive"],
                 "递归+语义": ["recursive", "semantic_merge"]}

        Returns:
            CompareReport 包含各组策略的完整指标和排名
        """
        if strategies is None:
            strategies = {
                "仅递归": ["recursive"],
                "递归+语义": ["recursive", "semantic_merge"],
            }

        from agent_material.rag_engine.text_spliter import HybridSplitter
        from agent_material.rag_engine.document_pipeline import DocumentLoader

        logger.info(f"[Compare] 加载文档: {file_path}")
        docs = DocumentLoader.load(file_path)
        if not docs:
            raise ValueError(f"文件加载为空: {file_path}")

        report = CompareReport(input_summary={
            "file": file_path,
            "doc_count": len(docs),
            "total_chars": sum(len(d.page_content) for d in docs),
        })

        for name, steps in strategies.items():
            logger.info(f"[Compare] 策略「{name}」: {steps}")
            splitter = HybridSplitter(steps)
            chunks = splitter.split_documents(docs)
            metrics = self.compute_metrics(chunks, strategy_name=name)
            report.strategies.append(metrics)

        # 综合排名（边界质量 * 长度均匀度，越高越好）
        scored = [
            (m.boundary_score * 0.4 + m.length_uniformity * 0.3
             + (1 - m.duplicate_ratio) * 0.2 + m.avg_info_density * 0.1, m)
            for m in report.strategies
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        for rank, (_score, m) in enumerate(scored, start=1):
            report.ranking[m.strategy_name] = rank

        # 保存报告
        report_path = self.output_dir / f"compare_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"[Compare] 报告已保存: {report_path}")

        return report

    # ------------------------------------------------------------------ #
    #  评估主入口
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        chunks: list[Document],
        strategy_name: str = "default",
        sample_size: int = 50,
        output_dir: str | None = None,
    ) -> dict:
        """对一组 chunk 执行完整评估：自动指标 + 抽样 + 质检。

        Args:
            chunks: ProcessingPipeline 输出的 list[Document]
            strategy_name: 策略名称（用于报告）
            sample_size: 人工评估抽样数
            output_dir: 输出目录（默认 self.output_dir）

        Returns:
            评估结果 dict
        """
        out = Path(output_dir) if output_dir else self.output_dir
        out.mkdir(parents=True, exist_ok=True)

        # 1. 自动指标
        metrics = self.compute_metrics(chunks, strategy_name)

        # 2. 自动质检
        flags = ChunkAutoFlagger.flag(chunks)

        # 3. 抽样
        sampler = ChunkSampler()
        sample_ids: set[int] = set()
        for s in sampler.random_sample(chunks, sample_size // 2):
            sample_ids.add(id(s))
        for s in sampler.stratified_by_source(chunks, n_per_source=3):
            sample_ids.add(id(s))
        for s in sampler.length_anomaly(chunks)[:10]:
            sample_ids.add(id(s))
        for s in sampler.boundary_issues(chunks)[:10]:
            sample_ids.add(id(s))
        samples = [c for c in chunks if id(c) in sample_ids]

        # 4. 预填评分模板（基于自动检测结果）
        score_template = []
        for i, c in enumerate(samples):
            text = c.page_content.strip()
            meta = c.metadata

            # ── 自动预填 issues ──
            auto_issues: list[str] = []
            ct = meta.get("content_type", "paragraph")

            # 分隔线 / 数字 dump → 直接标记为噪音
            if ct in ("separator", "numeric_dump"):
                auto_issues.append(f"[自动] 无意义内容 ({ct})")
            if len(text) < 30:
                auto_issues.append("[自动] 过短 (<30字符)")
            if len(text) > 3000:
                auto_issues.append("[自动] 过长 (>3000字符)")
            if "(cid:" in text:
                auto_issues.append("[自动] CID 乱码残留")
            if re.search(r"Copyright|All rights reserved|版权声明", text, re.I):
                auto_issues.append("[自动] 版权残留")

            # ── 自动预填评分 ──
            # 噪音类型直接给最低分
            if ct in ("separator", "numeric_dump"):
                boundary_quality = 1
                semantic_completeness = 1
                readability = 1
                information_density = 1
            else:
                # boundary_quality: 句首 + 句尾完整性
                starts_clean = self._is_sentence_start(text)
                ends_clean = self._is_sentence_end(text)
                if starts_clean and ends_clean:
                    boundary_quality = 5
                elif starts_clean or ends_clean:
                    boundary_quality = 3
                else:
                    boundary_quality = 1
                    auto_issues.append("[自动] 句首句尾均非完整边界")

                # semantic_completeness: 长度 + 边界综合
                if len(text) >= 200 and ends_clean:
                    semantic_completeness = 5
                elif len(text) >= 100 and ends_clean:
                    semantic_completeness = 4
                elif len(text) >= 50:
                    semantic_completeness = 3
                else:
                    semantic_completeness = 1
                    auto_issues.append("[自动] 过短，可能语义不完整")

                # readability: 基于 content_type 和噪音
                if ct == "code":
                    readability = 3
                elif ct == "table":
                    readability = 4
                elif "(cid:" in text:
                    readability = 1
                else:
                    readability = 5

                # information_density: 停止词占比映射
                stop_chars = sum(len(m.group()) for m in self._RE_STOP.finditer(text))
                density = max(0.0, 1.0 - stop_chars / max(len(text), 1))
                if density > 0.9:
                    information_density = 5
                elif density > 0.8:
                    information_density = 4
                elif density > 0.6:
                    information_density = 3
                else:
                    information_density = 1
                    auto_issues.append("[自动] 信息密度低")

            # context_independence: 保留人工判断
            context_independence = 0

            score_template.append({
                "chunk_id": f"chunk_{i:04d}",
                "source": meta.get("source", "N/A"),
                "content_preview": text[:200],
                "char_length": len(text),
                "content_type": ct,
                "language": meta.get("language", ""),
                "semantic_completeness": semantic_completeness,
                "information_density": information_density,
                "context_independence": context_independence,
                "readability": readability,
                "boundary_quality": boundary_quality,
                "issues": auto_issues,
                "notes": "",
            })
        template_path = out / "score_template.json"
        with open(template_path, "w", encoding="utf-8") as f:
            json.dump(score_template, f, ensure_ascii=False, indent=2)

        # 5. 输出
        return {
            "metrics": metrics.to_dict(),
            "auto_flags": {k: len(v) for k, v in flags.items()},
            "sample_count": len(samples),
            "template_path": str(template_path),
            "total_chunks": len(chunks),
        }

    @staticmethod
    def print_report(result: dict) -> None:
        """打印评估结果。"""
        m = result["metrics"]
        print(f"\n{'='*50}")
        print(f"  切分评估报告 — {m.get('strategy_name', 'N/A')}")
        print(f"{'='*50}")
        print(f"  Chunk 总数:     {m['total_chunks']}")
        print(f"  平均长度:       {m['avg_length']:.0f} 字符")
        print(f"  长度标准差:     {m['length_std']:.0f}")
        print(f"  长度均匀度:     {m['length_uniformity']:.3f}")
        print(f"  边界质量:       {m['boundary_score']:.3f}")
        print(f"  句首完整率:     {m['sentence_start_ratio']:.1%}")
        print(f"  句尾完整率:     {m['sentence_end_ratio']:.1%}")
        print(f"  信息密度:       {m['avg_info_density']:.3f}")
        print(f"  重复率:         {m['duplicate_ratio']:.3f}")
        print(f"  相邻相似度(均值): {m['avg_adjacent_similarity']:.3f}")
        print(f"  跨章节边界:     {m['cross_section_boundaries']}")
        print(f"  CID 残留:       {m['cid_residual_count']}")
        print(f"  页眉残留:       {m['header_residual_count']}")
        print(f"{'='*50}")
        print(f"  自动标记问题:   {result['auto_flags']}")
        print(f"  抽样数:         {result['sample_count']}")
        print(f"  评分模板:       {result['template_path']}")

    @staticmethod
    def print_compare(report: CompareReport) -> None:
        """打印策略对比报告。"""
        print(f"\n{'='*60}")
        print(f"  切分策略对比报告")
        print(f"{'='*60}")
        print(f"  {'策略':<16} {'chunks':>6} {'边界':>6} {'均匀':>6} {'信息':>6} {'排名':>4}")
        print(f"  {'─'*16} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*4}")
        for m in report.strategies:
            print(
                f"  {m.strategy_name:<16} {m.total_chunks:>6} "
                f"{m.boundary_score:>6.3f} {m.length_uniformity:>6.3f} "
                f"{m.avg_info_density:>6.3f} {report.ranking.get(m.strategy_name, '?'):>4}"
            )


# ==================================================================== #
#  工具函数
# ==================================================================== #


def _bigrams(text: str) -> set[str]:
    """提取文本的字符级 bigram 集合。"""
    cleaned = re.sub(r"\s+", " ", text)
    return {cleaned[i:i+2] for i in range(len(cleaned) - 1)}


# ==================================================================== #
#  命令行接口
# ==================================================================== #


def main():
    parser = argparse.ArgumentParser(description="Chunk 切分评估工具")
    parser.add_argument("--input", "-i", help="输入文件（PDF/JSONL/路径）")
    parser.add_argument("--output", "-o", default="workspace/evaluation", help="输出目录")
    parser.add_argument("--sample-size", type=int, default=100, help="抽样数量")
    parser.add_argument(
        "--compare", action="store_true",
        help="对比模式：用多组切分策略处理同一文件并生成对比报告",
    )
    args = parser.parse_args()

    if args.compare and args.input:
        # 修复 3：多策略对比
        evaluator = ChunkEvaluator(output_dir=args.output)
        report = evaluator.compare(args.input)
        ChunkEvaluator.print_compare(report)

    elif args.input:
        # 修复 1：直接接受 JSONL 或文件路径
        if args.input.endswith(".jsonl"):
            chunks = ChunkEvaluator.from_jsonl(args.input)
        else:
            # PDF / MD 等 → 走完整 Pipeline
            from agent_material.rag_engine.document_pipeline import ProcessingPipeline
            pipeline = ProcessingPipeline()
            chunks = pipeline.process(args.input)

        evaluator = ChunkEvaluator(output_dir=args.output)
        result = evaluator.evaluate(chunks, sample_size=args.sample_size)
        ChunkEvaluator.print_report(result)
        logger.info(f"评估完成 | 产出 {len(chunks)} chunks")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
