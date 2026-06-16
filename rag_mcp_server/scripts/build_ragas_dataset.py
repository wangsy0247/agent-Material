"""使用 MinerU + LLM 构建 RAGAS 评估数据集。

流程：
1. 对 abacus.pdf 随机抽取 20 个页码区间，用 MinerU flash_extract 解析，每段生成 1 个问答对。
2. 从当前目录随机抽取 10 个其他 PDF，每个解析并生成 2 个问答对。
3. 合并为 40 条 RAGAS 格式数据，保存为 JSON / CSV。

RAGAS 字段：
- question: 问题
- contexts: 参考上下文列表（来自 PDF 片段）
- ground_truth: 标准答案
- source: PDF 文件名
- page_range: 页码区间
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pypdf
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# 把 rag_mcp_server 包的父目录加入 sys.path，以便导入 rag_mcp_server 模块
RAG_MCP_SERVER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAG_MCP_SERVER_ROOT.parent))

from rag_mcp_server.utils.logger import logger  # noqa: E402

# ════════════════════════════════════════════════════════════════════════════════
#  配置
# ════════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PDF_ROOT = PROJECT_ROOT / "data"
ABACUS_PDF = PDF_ROOT / "abacus.pdf"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FLASH_MAX_PAGES = 20
FLASH_MAX_SIZE_MB = 10

LLM_MODEL = os.environ.get("RAGAS_LLM_MODEL", "qwen3.6-plus")
LLM_BASE_URL = os.environ.get(
    "RAGAS_LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
LLM_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 4096

# 调用 MinerU 的间隔（秒），避免触发速率限制
MINERU_CALL_INTERVAL = 1.5


# ════════════════════════════════════════════════════════════════════════════════
#  LLM 初始化
# ════════════════════════════════════════════════════════════════════════════════


def get_llm() -> ChatOpenAI:
    """初始化用于生成问答对的 LLM。"""
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )


# ════════════════════════════════════════════════════════════════════════════════
#  PDF 工具
# ════════════════════════════════════════════════════════════════════════════════


def get_pdf_page_count(pdf_path: Path) -> int:
    """获取 PDF 总页数。"""
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        count = len(reader.pages)
        reader.close()
        return count
    except Exception as e:
        logger.error(f"[PDF] 无法读取页数 {pdf_path}: {e}")
        return 0


def list_pdf_files(root: Path, exclude: set[str] | None = None) -> list[Path]:
    """列出目录下所有 PDF 文件（递归一层）。"""
    exclude = exclude or set()
    pdfs = [
        p for p in root.glob("*.pdf")
        if p.is_file() and p.name not in exclude
    ]
    # 也检查一级子目录（兼容当前 data/ 下的结构）
    pdfs.extend([
        p for p in root.rglob("*.pdf")
        if p.is_file() and p.name not in exclude and p not in pdfs
    ])
    return sorted(set(pdfs))


# ════════════════════════════════════════════════════════════════════════════════
#  PDF 物理切分（绕过 MinerU Flash 10MB 文件大小限制）
# ════════════════════════════════════════════════════════════════════════════════


def split_pdf_pages_to_temp(
    pdf_path: Path,
    start_page: int,
    end_page: int,
) -> Path | None:
    """将 PDF 的指定页码范围切分为一个临时小文件。

    MinerU Flash API 要求上传文件 ≤10MB，且 page_range 参数并不会减少
    实际上传的文件大小。因此需要先在本地把目标页切出来再上传。
    """
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        writer = pypdf.PdfWriter()
        for i in range(start_page - 1, min(end_page, len(reader.pages))):
            writer.add_page(reader.pages[i])

        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".pdf", prefix=f"mineru_{pdf_path.stem}_p{start_page}-{end_page}_"
        )
        os.close(tmp_fd)
        tmp_file = Path(tmp_path)

        with open(tmp_file, "wb") as f:
            writer.write(f)

        reader.close()

        mb = tmp_file.stat().st_size / (1024 * 1024)
        logger.debug(f"[PDF] 切分临时文件: {tmp_file.name} ({mb:.2f} MB)")
        return tmp_file

    except Exception as e:
        logger.error(f"[PDF] 切分 {pdf_path.name} p{start_page}-{end_page} 失败: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════════
#  MinerU 文本提取
# ════════════════════════════════════════════════════════════════════════════════


def extract_pdf_pages(
    pdf_path: Path | str,
    start_page: int,
    end_page: int,
    total_pages: int,
) -> str | None:
    """使用 MinerU flash_extract 解析指定页码范围的 PDF。

    MinerU Flash API 有两个硬性限制：
    - 上传文件大小 ≤ 10MB
    - 上传文件总页数 ≤ 20 页（若原文件 >20 页，必须传 page_range）

    因此分三种情况处理：
    1. 原文件 ≤ 10MB 且总页数 ≤ 20：直接上传原文件。
    2. 原文件 ≤ 10MB 但总页数 > 20：上传原文件并附带 page_range。
    3. 原文件 > 10MB：本地切分出目标页为临时文件，上传临时文件。

    Args:
        pdf_path: PDF 文件路径
        start_page: 起始页（1-indexed）
        end_page: 结束页（1-indexed，包含）
        total_pages: PDF 总页数

    Returns:
        解析得到的 Markdown 文本；失败返回 None
    """
    from mineru import MinerU

    pdf_path = Path(pdf_path)
    page_range = f"{start_page}-{end_page}"
    logger.info(f"[MinerU] 解析 {pdf_path.name} | 页码 {page_range}")

    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    use_page_range = total_pages > FLASH_MAX_PAGES
    needs_split = file_size_mb > FLASH_MAX_SIZE_MB

    tmp_file: Path | None = None
    try:
        if needs_split:
            # 文件过大：本地切分后上传小文件，不需要 page_range
            tmp_file = split_pdf_pages_to_temp(pdf_path, start_page, end_page)
            if tmp_file is None:
                return None
            target_path = tmp_file
            use_page_range = False
        else:
            target_path = pdf_path

        client = MinerU()
        call_kwargs: dict[str, Any] = {
            "language": "ch",
            "timeout": 300,
        }
        if use_page_range:
            call_kwargs["page_range"] = page_range

        result = client.flash_extract(
            str(target_path),
            **call_kwargs,
        )
        client.close()

        if result.state != "done":
            logger.error(
                f"[MinerU] {pdf_path.name} p{page_range} 解析失败: "
                f"state={result.state}, error={result.error}"
            )
            return None

        if not result.markdown:
            logger.warning(f"[MinerU] {pdf_path.name} p{page_range} 返回空文本")
            return None

        logger.info(
            f"[MinerU] {pdf_path.name} p{page_range} 完成 | "
            f"{len(result.markdown)} 字符"
        )
        return result.markdown

    except Exception as e:
        logger.error(f"[MinerU] {pdf_path.name} p{page_range} 异常: {e}")
        return None

    finally:
        if tmp_file and tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass
        # 简单限速
        time.sleep(MINERU_CALL_INTERVAL)


def extract_pdf_safe(
    pdf_path: Path | str,
    start_page: int | None = None,
    end_page: int | None = None,
) -> tuple[str | None, str]:
    """安全解析 PDF 片段，自动处理页码越界和文件过大。

    Returns:
        (markdown_text, page_range_str)
    """
    pdf_path = Path(pdf_path)
    total_pages = get_pdf_page_count(pdf_path)
    if total_pages == 0:
        return None, "unknown"

    if start_page is None or end_page is None:
        start_page = 1
        end_page = min(total_pages, FLASH_MAX_PAGES)

    # 限制在有效范围内
    start_page = max(1, start_page)
    end_page = min(total_pages, end_page)
    if start_page > end_page:
        start_page = end_page

    page_range_str = f"{start_page}-{end_page}"
    text = extract_pdf_pages(pdf_path, start_page, end_page, total_pages)
    return text, page_range_str


# ════════════════════════════════════════════════════════════════════════════════
#  问答对生成
# ════════════════════════════════════════════════════════════════════════════════


QA_GENERATION_PROMPT = """你是一名严谨的文档理解专家。请根据下面从学术论文中提取的 Markdown 片段，生成 {n_questions} 个高质量问答对，用于评估 RAG 系统的检索和回答能力。

━━━ 核心要求（必须全部满足）━━━

1. 【答案完整性】每个答案必须是 1-3 句完整的中文句子，包含足够的上下文信息，使答案可以脱离问题独立理解。
   ❌ 错误示例："5.924 Bohr"、"cuda"、"123 Ω μm"
   ✅ 正确示例："根据第一性原理计算，钼（Mo）体心立方结构的平衡晶格常数为 5.924 Bohr，该结果与实验值吻合良好。"

2. 【问题自包含】问题必须包含足够的上下文，使其可以脱离原文独立理解。
   ❌ 错误示例："根据表3，3C-SiC的体积模量是多少？"
   ✅ 正确示例："在机器学习势函数 DPA-Semi 的研究中，作者通过第一性原理计算得到的 3C-SiC 体积模量是多少？与实验值相比精度如何？"

3. 【中文提问与回答】所有问题和答案都使用中文。如果原文是英文，请翻译为中文，但保留专业术语（如 DFT、LCAO、van der Waals 等）。

4. 【问题类型多样化】优先生成以下类型的问题（避免纯数值查询）：
   - 概念解释型："XXX 方法的核心思想是什么？它在材料科学中有哪些应用？"
   - 方法对比型："方法 A 和方法 B 在处理 XXX 问题时各有什么优缺点？"
   - 因果分析型："为什么在 XXX 体系中需要使用 YYY 方法来描述 ZZZ 效应？"
   - 结论推断型："根据文中的实验结果，可以得出哪些关于 XXX 性能的关键结论？"

5. 【输出格式】仅输出合法的 JSON 数组，不要包含 Markdown 代码块标记或任何其他说明文字。

输出格式：
[
  {{
    "question": "用中文提出的完整问题",
    "answer": "1-3句完整的中文答案，包含上下文信息"
  }}
]

文本片段：
{context}
"""


def parse_json_response(text: str) -> list[dict[str, str]]:
    """从 LLM 响应中提取 JSON 数组。"""
    text = text.strip()
    # 去掉可能的 markdown 代码块
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 尝试从文本中截取第一个 [ ... ] 数组
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start : end + 1])
        else:
            raise

    if isinstance(data, dict) and "questions" in data:
        data = data["questions"]
    if not isinstance(data, list):
        raise ValueError(f"LLM 返回不是数组: {type(data)}")
    return data


MIN_ANSWER_LENGTH = 20
MIN_QUESTION_LENGTH = 15


def _qa_quality_ok(q: str, a: str) -> bool:
    """检查单个问答对的质量是否达标。"""
    if len(a) < MIN_ANSWER_LENGTH:
        return False
    if len(q) < MIN_QUESTION_LENGTH:
        return False
    # 拒绝只有数字+单位的答案
    import re
    if re.fullmatch(r"[\d.,\s~±%ΩμmnpA-Za-z]+\s*\w*", a):
        return False
    return True


def generate_qa_pairs(
    llm: ChatOpenAI,
    context: str,
    n_questions: int = 1,
    max_retries: int = 2,
) -> list[dict[str, str]]:
    """根据上下文生成问答对，自动过滤低质量结果并重试。"""
    prompt = QA_GENERATION_PROMPT.format(
        n_questions=n_questions,
        context=context[:16000],
    )

    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke(prompt)
            qa_list = parse_json_response(response.content)
            # 统一字段名 + 质量过滤
            normalized = []
            for item in qa_list:
                q = item.get("question") or item.get("q")
                a = item.get("answer") or item.get("a") or item.get("ground_truth")
                if q and a:
                    q, a = str(q).strip(), str(a).strip()
                    if _qa_quality_ok(q, a):
                        normalized.append({"question": q, "answer": a})
                    else:
                        logger.warning(
                            f"[LLM] 质量不达标 | q={len(q)}chars a={len(a)}chars | "
                            f"q={q[:60]}... | a={a[:60]}..."
                        )
            if normalized:
                return normalized
            # 质量过滤后为空，重试
            logger.warning(f"[LLM] 质量过滤后无有效结果 (尝试 {attempt + 1})")
        except Exception as e:
            logger.warning(f"[LLM] 生成问答对失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
        if attempt < max_retries:
            time.sleep(2 ** attempt)

    logger.error(f"[LLM] 最终失败，max_retries={max_retries}")
    return []


# ════════════════════════════════════════════════════════════════════════════════
#  数据集构建
# ════════════════════════════════════════════════════════════════════════════════


def random_page_range(
    total_pages: int,
    max_pages: int = FLASH_MAX_PAGES,
    target_window: int = 4,
) -> tuple[int, int]:
    """随机生成一个页码区间，用于 PDF 片段解析。

    窗口固定 3-5 页，确保每个上下文片段聚焦、精确，LLM 能生成高质量问答对。
    对于页数很少的 PDF，返回全部页面。
    """
    if total_pages <= target_window:
        return 1, total_pages

    window = min(target_window, total_pages)
    start = random.randint(1, total_pages - window + 1)
    end = start + window - 1
    return start, end


def generate_non_overlapping_ranges(
    total_pages: int,
    n_ranges: int,
    window: int = 4,
) -> list[tuple[int, int]]:
    """生成 n_ranges 个互不重叠的页码区间，确保均匀覆盖全书。

    将 PDF 均分为 n_ranges 个段，每段内随机取一个 window 大小的窗口。
    这样既保证了分散覆盖全书，又避免了区间重叠导致的上下文重复。
    """
    if total_pages <= window * n_ranges:
        # 总页数不够分，退化为允许重叠的窗口=3页
        window = max(2, total_pages // (n_ranges * 2))
        if window < 2:
            window = 2

    segment_size = total_pages // n_ranges
    ranges: list[tuple[int, int]] = []

    for i in range(n_ranges):
        seg_start = i * segment_size + 1
        seg_end = (i + 1) * segment_size if i < n_ranges - 1 else total_pages
        # 确保窗口不超出段边界
        max_start = max(seg_start, seg_end - window + 1)
        if seg_start > max_start:
            start = seg_start
        else:
            start = random.randint(seg_start, max_start)
        end = min(start + window - 1, total_pages)
        ranges.append((start, end))

    random.shuffle(ranges)
    return ranges


def build_abacus_qa(
    llm: ChatOpenAI,
    n_pairs: int = 20,
) -> list[dict[str, Any]]:
    """从 abacus.pdf 生成 n_pairs 个问答对，均匀覆盖全书。"""
    if not ABACUS_PDF.exists():
        raise FileNotFoundError(f"未找到 {ABACUS_PDF}")

    total_pages = get_pdf_page_count(ABACUS_PDF)
    logger.info(f"[Dataset] abacus.pdf 共 {total_pages} 页，目标生成 {n_pairs} 个问答对")

    # 使用分层采样，确保 20 个区间均匀覆盖全书、互不重叠
    page_ranges = generate_non_overlapping_ranges(total_pages, n_ranges=n_pairs, window=5)
    logger.info(f"[Dataset] abacus.pdf 生成了 {len(page_ranges)} 个互不重叠的页码区间")

    records: list[dict[str, Any]] = []
    for start, end in page_ranges:
        context, page_range = extract_pdf_safe(ABACUS_PDF, start, end)
        if not context:
            logger.warning(f"[Dataset] abacus.pdf p{page_range} 解析为空，跳过")
            continue

        qa_list = generate_qa_pairs(llm, context, n_questions=1)
        for qa in qa_list:
            records.append({
                "question": qa["question"],
                "contexts": [context],
                "ground_truth": qa["answer"],
                "source": ABACUS_PDF.name,
                "page_range": page_range,
            })

        if len(records) % 5 == 0:
            logger.info(f"[Dataset] abacus.pdf 进度: {len(records)}/{n_pairs}")

    logger.info(f"[Dataset] abacus.pdf 成功生成 {len(records)} 个问答对")
    return records


def build_random_pdfs_qa(
    llm: ChatOpenAI,
    n_files: int = 10,
    pairs_per_file: int = 2,
) -> list[dict[str, Any]]:
    """随机抽取 n_files 个 PDF，每个生成 pairs_per_file 个问答对，均匀覆盖。"""
    all_pdfs = list_pdf_files(PDF_ROOT, exclude={ABACUS_PDF.name})
    if len(all_pdfs) < n_files:
        logger.warning(
            f"[Dataset] PDF 数量不足 {n_files}，实际 {len(all_pdfs)}"
        )
        n_files = len(all_pdfs)

    sampled = random.sample(all_pdfs, n_files)
    logger.info(f"[Dataset] 随机抽取 {len(sampled)} 个 PDF: {[p.name for p in sampled]}")

    records: list[dict[str, Any]] = []
    for pdf_path in sampled:
        total_pages = get_pdf_page_count(pdf_path)
        if total_pages == 0:
            logger.warning(f"[Dataset] 无法读取 {pdf_path.name}，跳过")
            continue

        # 使用分层采样确保 2 个区间均匀覆盖该 PDF
        page_ranges = generate_non_overlapping_ranges(
            total_pages, n_ranges=pairs_per_file, window=5
        )

        file_records: list[dict[str, Any]] = []
        for start, end in page_ranges:
            context, page_range = extract_pdf_safe(pdf_path, start, end)
            if not context:
                continue

            qa_list = generate_qa_pairs(llm, context, n_questions=1)
            for qa in qa_list:
                if len(file_records) >= pairs_per_file:
                    break
                file_records.append({
                    "question": qa["question"],
                    "contexts": [context],
                    "ground_truth": qa["answer"],
                    "source": pdf_path.name,
                    "page_range": page_range,
                })

            if len(file_records) >= pairs_per_file:
                break

        records.extend(file_records)
        logger.info(
            f"[Dataset] {pdf_path.name} 生成 {len(file_records)}/{pairs_per_file} 个问答对"
        )

    logger.info(f"[Dataset] 随机 PDF 共成功生成 {len(records)} 个问答对")
    return records


# ════════════════════════════════════════════════════════════════════════════════
#  输出
# ════════════════════════════════════════════════════════════════════════════════


def save_dataset(records: list[dict[str, Any]], output_dir: Path) -> dict[str, Path]:
    """保存数据集为 JSON 和 CSV（自动去重）。"""
    # 去重：基于 (question, ground_truth, source) 去除完全重复的条目
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for r in records:
        key = (r["question"], r["ground_truth"], r["source"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
        else:
            logger.warning(f"[Dataset] 去重移除: {r['question'][:60]}...")

    if len(deduped) < len(records):
        logger.info(f"[Dataset] 去重: {len(records)} → {len(deduped)} 条")

    # RAGAS 标准字段
    ragas_records = []
    for r in deduped:
        ragas_records.append({
            "question": r["question"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
            "source": r["source"],
            "page_range": r["page_range"],
        })

    json_path = output_dir / "ragas_evaluation_dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ragas_records, f, ensure_ascii=False, indent=2)

    csv_path = output_dir / "ragas_evaluation_dataset.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "contexts", "ground_truth", "source", "page_range"])
        writer.writeheader()
        for r in ragas_records:
            row = dict(r)
            row["contexts"] = "\n---\n".join(row["contexts"])
            writer.writerow(row)

    logger.info(f"[Dataset] 已保存 JSON: {json_path}")
    logger.info(f"[Dataset] 已保存 CSV: {csv_path}")
    return {"json": json_path, "csv": csv_path}


# ════════════════════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════════════════════


def main() -> None:
    random.seed(42)

    if not LLM_API_KEY:
        logger.error("请先设置环境变量 DASHSCOPE_API_KEY")
        return

    llm = get_llm()
    logger.info("[Main] 开始构建 RAGAS 评估数据集")

    # 1) abacus.pdf: 20 个问答对
    abacus_records = build_abacus_qa(llm, n_pairs=20)

    # 2) 随机 10 个 PDF: 每个 2 个问答对
    random_records = build_random_pdfs_qa(llm, n_files=10, pairs_per_file=2)

    # 合并
    all_records = abacus_records + random_records
    logger.info(f"[Main] 数据集总条数: {len(all_records)}")

    # 保存
    paths = save_dataset(all_records, OUTPUT_DIR)

    # 打印摘要
    print("\n" + "=" * 60)
    print(f"RAGAS 评估数据集构建完成")
    print(f"总条数: {len(all_records)}")
    print(f"  - abacus.pdf: {len(abacus_records)} 条")
    print(f"  - 随机 10 个 PDF: {len(random_records)} 条")
    print(f"JSON: {paths['json']}")
    print(f"CSV:  {paths['csv']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
