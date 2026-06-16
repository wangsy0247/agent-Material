"""使用 MinerU 文档加载器 + RAGAS TestsetGenerator 自动生成评测集。

流程：
1. MinerU flash_extract 解析 PDF → Markdown 文本
2. RecursiveCharacterTextSplitter 切分 Markdown → LangChain Document 列表
3. RAGAS TestsetGenerator 基于 Document 列表生成问答对
4. 输出 RAGAS 标准格式评测集

输入：data/ 目录下的 PDF 文件（可通过命令行指定）
输出：evaluation_data/ragas_generated_testset.json (question + ground_truth + contexts + metadata)

用法：
    python rag_mcp_server/scripts/generate_ragas_testset.py                        # 处理 data/ 下所有 PDF
    python rag_mcp_server/scripts/generate_ragas_testset.py data/abacus.pdf        # 处理指定文件
    python rag_mcp_server/scripts/generate_ragas_testset.py -n 50                  # 生成 50 条问答
    python rag_mcp_server/scripts/generate_ragas_testset.py --pages-per-doc 5      # 每文档抽样 5 页

依赖：
    pip install ragas mineru langchain-text-splitters langchain-openai datasets
"""

from __future__ import annotations

import argparse
import json
import random
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

# 路径设置
SCRIPT_DIR = Path(__file__).resolve().parent
RAG_MCP_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = RAG_MCP_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402
from langchain_core.documents import Document as LCDocument  # noqa: E402
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402

from rag_mcp_server.utils.logger import logger  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MinerU 限制
FLASH_MAX_PAGES = 20
FLASH_MAX_SIZE_MB = 10
MINERU_CALL_INTERVAL = 1.5


# ════════════════════════════════════════════════════════════════════════════════
#  MinerU 文档加载器
# ════════════════════════════════════════════════════════════════════════════════


class MinerUDocumentLoader:
    """基于 MinerU flash_extract 的文档加载器。

    将 PDF 解析为 Markdown，切分为 LangChain Document 列表。
    自动处理：文件过大本地切分、页码越界、速率限制。
    """

    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        pages_per_doc: int = 10,
    ):
        """
        Args:
            chunk_size: 文本切分块大小（字符数）
            chunk_overlap: 块之间重叠字符数
            pages_per_doc: 每个 PDF 抽样解析的最大页数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pages_per_doc = pages_per_doc
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    # ── PDF 工具 ──

    @staticmethod
    def get_pdf_page_count(pdf_path: Path) -> int:
        try:
            import pypdf
            reader = pypdf.PdfReader(str(pdf_path))
            count = len(reader.pages)
            reader.close()
            return count
        except Exception as e:
            logger.error(f"[PDF] 无法读取页数 {pdf_path}: {e}")
            return 0

    @staticmethod
    def split_pdf_pages_to_temp(pdf_path: Path, start_page: int, end_page: int) -> Path | None:
        """将 PDF 指定页码范围切分为临时小文件（用于绕过 MinerU 文件大小限制）。"""
        import tempfile
        import pypdf

        try:
            reader = pypdf.PdfReader(str(pdf_path))
            writer = pypdf.PdfWriter()
            for i in range(start_page - 1, min(end_page, len(reader.pages))):
                writer.add_page(reader.pages[i])

            tmp_fd, tmp_path = tempfile.mkstemp(
                suffix=".pdf", prefix=f"mineru_{pdf_path.stem}_p{start_page}-{end_page}_"
            )
            import os
            os.close(tmp_fd)
            tmp_file = Path(tmp_path)

            with open(tmp_file, "wb") as f:
                writer.write(f)
            reader.close()
            return tmp_file
        except Exception as e:
            logger.error(f"[PDF] 切分 {pdf_path.name} p{start_page}-{end_page} 失败: {e}")
            return None

    # ── MinerU 文本提取 ──

    def _extract_pages(self, pdf_path: Path, start_page: int, end_page: int) -> str | None:
        """使用 MinerU flash_extract 解析指定页码范围。"""
        from mineru import MinerU

        total_pages = self.get_pdf_page_count(pdf_path)
        if total_pages == 0:
            return None

        page_range = f"{start_page}-{end_page}"
        logger.info(f"[MinerU] 解析 {pdf_path.name} | 页码 {page_range}")

        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        use_page_range = total_pages > FLASH_MAX_PAGES
        needs_split = file_size_mb > FLASH_MAX_SIZE_MB

        tmp_file: Path | None = None
        try:
            if needs_split:
                tmp_file = self.split_pdf_pages_to_temp(pdf_path, start_page, end_page)
                if tmp_file is None:
                    return None
                target_path = tmp_file
                use_page_range = False
            else:
                target_path = pdf_path

            client = MinerU()
            call_kwargs: dict = {"language": "ch", "timeout": 300}
            if use_page_range:
                call_kwargs["page_range"] = page_range

            result = client.flash_extract(str(target_path), **call_kwargs)
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
                f"[MinerU] {pdf_path.name} p{page_range} 完成 | {len(result.markdown)} 字符"
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
            time.sleep(MINERU_CALL_INTERVAL)

    # ── 公开接口 ──

    def load(self, pdf_path: Path | str) -> list[LCDocument]:
        """加载单个 PDF 为 Document 列表。

        Args:
            pdf_path: PDF 文件路径

        Returns:
            LangChain Document 列表
        """
        pdf_path = Path(pdf_path)
        total_pages = self.get_pdf_page_count(pdf_path)
        if total_pages == 0:
            return []

        pages = min(self.pages_per_doc, total_pages)

        # 从 PDF 中部随机采样一段（避免总是读前几页的目录/摘要）
        max_start = total_pages - pages + 1
        if max_start > 1:
            start_page = random.randint(1, max_start)
        else:
            start_page = 1
        end_page = start_page + pages - 1

        markdown = self._extract_pages(pdf_path, start_page, end_page)
        if not markdown:
            return []

        documents = self._splitter.create_documents(
            texts=[markdown],
            metadatas=[{
                "source": pdf_path.name,
                "page_range": f"{start_page}-{end_page}",
                "loader": "mineru",
            }],
        )

        # 过滤过短的 chunk（页眉页脚等噪声）
        documents = [d for d in documents if len(d.page_content.strip()) >= 50]

        logger.info(
            f"[Loader] {pdf_path.name} p{start_page}-{end_page} → "
            f"{len(documents)} chunks ({len(markdown)} chars raw)"
        )
        return documents

    def load_directory(
        self,
        data_dir: Path | str,
        file_pattern: str = "*.pdf",
        max_files: int | None = None,
        exclude: set[str] | None = None,
    ) -> list[LCDocument]:
        """加载目录下所有 PDF，返回 Document 列表。

        Args:
            data_dir: 数据目录
            file_pattern: 文件匹配模式
            max_files: 最多处理的文件数（None = 不限）
            exclude: 排除的文件名集合
        """
        data_dir = Path(data_dir)
        exclude = exclude or set()

        pdf_paths = [
            p for p in sorted(data_dir.rglob(file_pattern))
            if p.is_file() and p.name not in exclude
        ]

        if max_files:
            pdf_paths = pdf_paths[:max_files]

        logger.info(f"[Loader] 发现 {len(pdf_paths)} 个 PDF，开始加载 ...")

        all_docs: list[LCDocument] = []
        for i, pdf_path in enumerate(pdf_paths):
            logger.info(f"[Loader] [{i + 1}/{len(pdf_paths)}] {pdf_path.name}")
            try:
                docs = self.load(pdf_path)
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"[Loader] {pdf_path.name} 加载失败: {e}")

        logger.info(f"[Loader] 共加载 {len(all_docs)} 个 Document chunk")
        return all_docs


# ════════════════════════════════════════════════════════════════════════════════
#  RAGAS TestsetGenerator 封装
# ════════════════════════════════════════════════════════════════════════════════


class TestsetBuilder:
    """封装 RAGAS TestsetGenerator，使用 DashScope 千问 + Embedding 生成评测集。"""

    def __init__(
        self,
        llm_model: str = "qwen3.5-flash",
        embedding_model: str = "text-embedding-v4",
        llm_temperature: float = 0.3,
    ):
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.llm_temperature = llm_temperature

    def _create_ragas_llm(self):
        """创建 RAGAS 兼容的 LLM（DashScope 千问）。"""
        raw_llm = ChatOpenAI(
            model=self.llm_model,
            temperature=self.llm_temperature,
        )
        from ragas.llms.base import LangchainLLMWrapper
        return LangchainLLMWrapper(raw_llm)

    def _create_ragas_embeddings(self):
        """创建 RAGAS 兼容的 Embedding 模型（DashScope text-embedding-v4）。"""
        raw_emb = OpenAIEmbeddings(
            model=self.embedding_model,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=__import__("os").environ.get("DASHSCOPE_API_KEY", ""),
            dimensions=1024,
        )
        from ragas.embeddings.base import LangchainEmbeddingsWrapper
        return LangchainEmbeddingsWrapper(raw_emb)

    def _build_query_distribution(self):
        """构建查询类型分布（单跳 + 多跳 混合）。"""
        from ragas.testset.synthesizers import (
            SingleHopSpecificQuerySynthesizer,
            MultiHopAbstractQuerySynthesizer,
            MultiHopSpecificQuerySynthesizer,
        )
        from ragas.testset.synthesizers.base import BaseSynthesizer

        llm = self._create_ragas_llm()

        return [
            (SingleHopSpecificQuerySynthesizer(llm=llm), 0.5),
            (MultiHopSpecificQuerySynthesizer(llm=llm), 0.25),
            (MultiHopAbstractQuerySynthesizer(llm=llm), 0.25),
        ]

    def generate(
        self,
        documents: list[LCDocument],
        testset_size: int = 40,
        language: str = "chinese",
    ) -> dict:
        """基于 Document 列表生成 RAGAS 评测集。

        Args:
            documents: LangChain Document 列表
            testset_size: 目标生成问答对数量
            language: 问答语言 (chinese / english)

        Returns:
            {"question": ..., "ground_truth": ..., "contexts": [...], "metadata": {...}}
            的列表（dict 格式）
        """
        from ragas.testset import TestsetGenerator

        if len(documents) == 0:
            logger.error("文档列表为空，无法生成评测集")
            return {"records": [], "error": "empty documents"}

        # 限制问答数不超过 chunk 数
        actual_size = min(testset_size, len(documents))
        if actual_size < testset_size:
            logger.warning(f"文档数 ({len(documents)}) 不足，问答数从 {testset_size} 调整为 {actual_size}")

        llm = self._create_ragas_llm()
        embeddings = self._create_ragas_embeddings()
        query_distribution = self._build_query_distribution()

        generator = TestsetGenerator(
            llm=llm,
            embedding_model=embeddings,
            llm_context=f"You are an expert in materials science. Generate all questions and answers in {language}. Questions should be detailed, self-contained, and test deep understanding of computational materials science concepts like DFT, ABACUS software, electronic structure calculations, etc.",
        )

        logger.info(
            f"[RAGAS] 开始生成 {actual_size} 条问答 "
            f"(chunks: {len(documents)}, model: {self.llm_model})"
        )
        t_start = time.time()

        testset = generator.generate_with_langchain_docs(
            documents=documents,
            testset_size=actual_size,
            query_distribution=query_distribution,
        )

        elapsed = time.time() - t_start
        logger.info(f"[RAGAS] 生成完成 | {len(testset)} 条问答 | 耗时 {elapsed:.1f}s")

        # 转换为标准 RAGAS 评测格式
        records = []
        for item in testset:
            q = item.question if hasattr(item, "question") else item.get("question", "")
            a = item.answer if hasattr(item, "answer") else item.get("answer", "")
            ctx = item.contexts if hasattr(item, "contexts") else item.get("contexts", [])

            # contexts 统一为字符串列表
            if isinstance(ctx, str):
                ctx = [ctx]

            records.append({
                "question": str(q),
                "ground_truth": str(a),
                "contexts": [str(c) for c in ctx],
            })

        return {"records": records, "count": len(records), "generation_time_s": round(elapsed, 1)}


# ════════════════════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 MinerU 文档加载器 + RAGAS TestsetGenerator 生成评测集",
    )
    parser.add_argument(
        "inputs", nargs="*",
        help="PDF 文件或目录路径（默认: data/）",
    )
    parser.add_argument(
        "-n", "--num-questions", type=int, default=40,
        help="生成问答对数量（默认 40）",
    )
    parser.add_argument(
        "--pages-per-doc", type=int, default=10,
        help="每个 PDF 抽样解析页数（默认 10）",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=2000,
        help="文本切分块大小（默认 2000 字符）",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=200,
        help="块间重叠字符数（默认 200）",
    )
    parser.add_argument(
        "-o", "--output",
        default=str(OUTPUT_DIR / "ragas_generated_testset.json"),
        help="输出 JSON 路径",
    )
    parser.add_argument(
        "--model", default="qwen3.5-flash",
        help="生成问答的 LLM 模型（默认 qwen3.5-flash）",
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="最多处理的 PDF 文件数",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认 42）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # ── 阶段 1: MinerU 加载文档 ──
    loader = MinerUDocumentLoader(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        pages_per_doc=args.pages_per_doc,
    )

    if args.inputs:
        # 用户指定了输入文件/目录
        all_docs: list[LCDocument] = []
        for inp in args.inputs:
            p = Path(inp)
            if p.is_file():
                all_docs.extend(loader.load(p))
            elif p.is_dir():
                all_docs.extend(loader.load_directory(p, max_files=args.max_files))
            else:
                logger.warning(f"路径不存在: {p}")
    else:
        # 默认：加载 data/ 下所有 PDF
        all_docs = loader.load_directory(DATA_DIR, max_files=args.max_files)

    if not all_docs:
        logger.error("未加载到任何文档，退出")
        sys.exit(1)

    logger.info(f"文档加载完成: {len(all_docs)} chunks")

    # ── 阶段 2: RAGAS TestsetGenerator 生成问答 ──
    builder = TestsetBuilder(llm_model=args.model)
    result = builder.generate(
        documents=all_docs,
        testset_size=args.num_questions,
        language="chinese",
    )

    records = result.get("records", [])
    if not records:
        logger.error("生成问答对为空，退出")
        sys.exit(1)

    # ── 阶段 3: 保存 ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "generator": "ragas_testset_generator",
            "loader": "mineru",
            "model": args.model,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "pages_per_doc": args.pages_per_doc,
            "generation_time_s": result.get("generation_time_s"),
            "document_count": len(all_docs),
            "question_count": len(records),
        },
        "data": records,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"RAGAS 评测集生成完成")
    print(f"{'=' * 60}")
    print(f"  PDF 文件数: {len(set(d.metadata.get('source', '') for d in all_docs))}")
    print(f"  Document 块数: {len(all_docs)}")
    print(f"  生成问答数: {len(records)}")
    print(f"  耗时: {result.get('generation_time_s', 'N/A')}s")
    print(f"  输出: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
