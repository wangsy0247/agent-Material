"""查询改写器 — 多策略查询重写以提升检索召回率。

策略：
- expand:      查询扩展，生成同义/近义变体 → 多路检索 → 融合
- hyde:        假设文档生成，用假想文档的 embedding 做向量检索
- step_back:   退一步提问，抽象查询 + 原始查询 → 双路检索 → 融合
- decompose:   查询分解，多跳问题拆解为子问题 → 逐路检索 → 综合

所有线上策略通过 RagService.prepare()/search() 集成到统一检索管线。
"""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from agent_material.utils.config_handler import (
    agent_config,
    prompt_config,
    vector_store_config,
)
from agent_material.utils.prompt_loader import get_prompts
from agent_material.utils.path_tools import get_abs_path
from agent_material.utils.logger import get_logger


class QueryRewriter:
    """查询改写器 — 多种重写策略提升检索召回率。

    每种策略通过 LLM 生成改写后的查询或辅助文本，然后用改写结果
    执行多路检索并融合结果。
    """

    def __init__(self, model=None):
        self._config = vector_store_config.get("query_rewriting", {})
        self._model = model or self._create_model()
        self._logger = get_logger("query_rewriter")
        # 延迟加载所有 prompt
        self._prompts: dict[str, PromptTemplate] = {}

    def _create_model(self):
        """创建查询改写专用模型，默认关闭思考与流式输出。"""
        model_cfg = self._config.get("model", {})
        model_name = model_cfg.get("name") or agent_config.get(
            "chat_model_name", "qwen3.5-flash"
        )
        return ChatOpenAI(
            model_name=model_name,
            max_tokens=int(model_cfg.get("max_tokens", 1024)),
            streaming=False,
            extra_body={
                "enable_thinking": bool(model_cfg.get("enable_thinking", False))
            },
        )

    def _load_prompt(self, config_key: str, default_path: str) -> PromptTemplate:
        """从配置文件加载 PromptTemplate（带缓存）。"""
        if config_key not in self._prompts:
            path = prompt_config.get(config_key, default_path)
            text = get_prompts(get_abs_path(path)) or ""
            self._prompts[config_key] = PromptTemplate.from_template(text)
        return self._prompts[config_key]

    def _invoke_llm(self, prompt_template: PromptTemplate, **kwargs) -> str:
        """调用 LLM 并返回纯文本输出。"""
        chain = prompt_template | self._model | StrOutputParser()
        return chain.invoke(kwargs)

    # ── Query Expansion ──────────────────────────────────────

    def expand(self, query: str, num_queries: int | None = None) -> list[str]:
        """查询扩展：生成多个同义/近义查询变体。

        Args:
            query: 原始查询
            num_queries: 生成的查询数量（含原始查询），默认从配置读取

        Returns:
            查询变体列表，第一个元素始终是原始查询
        """
        cfg = self._config.get("expansion", {})
        if num_queries is None:
            num_queries = cfg.get("num_queries", 3)

        try:
            prompt = self._load_prompt(
                "query_expansion_prompt_path",
                "backend/packages/agent_material/prompts/rag/query_expansion.txt",
            )
            raw = self._invoke_llm(prompt, query=query, num=num_queries - 1)

            # 解析输出：每行一个变体，去重，过滤过短文本
            variants: list[str] = []
            seen: set[str] = set()
            for line in raw.strip().split("\n"):
                v = line.strip()
                if v and len(v) > 5 and v not in seen:
                    seen.add(v)
                    variants.append(v)

            result = [query] + variants[: num_queries - 1]
            self._logger.info(
                f"[Expand] '{query[:40]}...' → {len(result)} variants"
            )
            return result
        except Exception as e:
            self._logger.error(f"[Expand] 失败: {e}，回退到原始查询")
            return [query]

    # ── HyDE ─────────────────────────────────────────────────

    def hyde(self, query: str, doc_length: int | None = None) -> str:
        """HyDE：生成假设文档文本。

        注意：此方法仅返回假设文档文本。调用方需要使用此文本的 embedding
        进行向量检索，同时用原始查询进行 BM25 检索。

        Args:
            query: 原始查询
            doc_length: 目标文档字数，默认从配置读取

        Returns:
            假设文档文本
        """
        cfg = self._config.get("hyde", {})
        if doc_length is None:
            doc_length = cfg.get("doc_length", 150)

        try:
            prompt = self._load_prompt(
                "hyde_generation_prompt_path",
                "backend/packages/agent_material/prompts/rag/hyde_generation.txt",
            )
            hyde_doc = self._invoke_llm(
                prompt, query=query, length=doc_length
            ).strip()

            self._logger.info(
                f"[HyDE] query='{query[:40]}...' | doc_length={len(hyde_doc)} chars"
            )
            return hyde_doc
        except Exception as e:
            self._logger.error(f"[HyDE] 失败: {e}")
            return ""

    # ── Step-Back ────────────────────────────────────────────

    def step_back(self, query: str) -> tuple[str, str]:
        """退一步提问：生成抽象/通用版本。

        Returns:
            (原始查询, 退一步查询)
        """
        try:
            prompt = self._load_prompt(
                "stepback_generation_prompt_path",
                "backend/packages/agent_material/prompts/rag/stepback_generation.txt",
            )
            stepback_query = self._invoke_llm(prompt, query=query).strip()

            self._logger.info(
                f"[StepBack] '{query[:40]}...' → '{stepback_query[:40]}...'"
            )
            return (query, stepback_query)
        except Exception as e:
            self._logger.error(f"[StepBack] 失败: {e}")
            return (query, query)

    # ── Query Decomposition ──────────────────────────────────

    def decompose(self, query: str, max_sub_queries: int | None = None) -> list[str]:
        """查询分解：将多跳问题分解为子问题。

        Args:
            query: 原始复杂查询
            max_sub_queries: 最大子问题数量，默认从配置读取

        Returns:
            子问题列表（为空时回退到原始查询）
        """
        cfg = self._config.get("decomposition", {})
        if max_sub_queries is None:
            max_sub_queries = cfg.get("max_sub_queries", 5)

        try:
            prompt = self._load_prompt(
                "query_decomposition_prompt_path",
                "backend/packages/agent_material/prompts/rag/query_decomposition.txt",
            )
            raw = self._invoke_llm(
                prompt, query=query, max_sub_queries=max_sub_queries
            )

            sub_queries: list[str] = []
            for line in raw.strip().split("\n"):
                sq = line.strip()
                if sq and len(sq) > 5:
                    sub_queries.append(sq)

            result = sub_queries[:max_sub_queries]
            self._logger.info(
                f"[Decompose] '{query[:40]}...' → {len(result)} sub-queries"
            )
            return result if result else [query]
        except Exception as e:
            self._logger.error(f"[Decompose] 失败: {e}")
            return [query]
