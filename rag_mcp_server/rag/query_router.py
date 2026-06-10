"""查询路由与分类 — 闲聊识别、无关问题拦截、问题类型标注。

使用 ChatOpenAI.with_structured_output() + Pydantic 模型实现可靠的查询分类，
遵循项目 lazy-init 模式，不阻塞模块导入。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag_mcp_server.models import chat_model
from rag_mcp_server.config import vector_store_config, prompt_config
from rag_mcp_server.utils.prompt_loader import get_prompts
from rag_mcp_server.utils.path import get_abs_path
from rag_mcp_server.utils.logger import get_logger


class QueryClassification(BaseModel):
    """查询分类的结构化输出模型。

    Attributes:
        is_chat: 是否是闲聊/寒暄（无需检索知识库）
        is_irrelevant: 是否与知识库领域完全无关
        question_type: 问题类型标签
        confidence: 分类置信度 (0.0~1.0)
        reasoning: 分类理由摘要
    """

    is_chat: bool = Field(description='是否是闲聊问候或寒暄（如"你好"、"谢谢"等），无需检索知识库')
    is_irrelevant: bool = Field(
        description="是否与材料科学、计算材料学、LAMMPS、Abacus软件、量子化学等知识库领域完全无关"
    )
    question_type: str = Field(
        description="问题类型: factual=事实性查询, summary=总结性查询, comparison=对比性查询, unknown=不确定"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="分类置信度，0.0~1.0")
    reasoning: str = Field(description="分类理由的简短摘要")


class QueryRouter:
    """查询路由器 — 对用户查询进行分类和路由决策。

    功能：
    1. 闲聊检测 → 直接 LLM 回答（跳过 RAG，降低延迟）
    2. 无关问题拦截 → 返回拒绝消息
    3. 问题类型标注 → 下游可选择不同生成策略

    使用 structured output 确保分类结果可靠可解析。
    """

    def __init__(self):
        self._model = chat_model
        self._chain = None  # 延迟初始化
        prompt_path = prompt_config.get(
            "query_classification_prompt_path",
            "rag_mcp_server/prompts/query_classification.txt",
        )
        self._prompt_text = get_prompts(get_abs_path(prompt_path)) or ""
        self._prompt_template = PromptTemplate.from_template(self._prompt_text)
        self._config = vector_store_config.get("query_routing", {})
        self._confidence_threshold = self._config.get("confidence_threshold", 0.7)
        self._logger = get_logger("query_router")

    def _get_chain(self):
        """延迟初始化结构化输出链。

        with_structured_output() 内部将 streaming 设为 False，与现有 chat_model
        的 streaming=True 配置不冲突。
        """
        if self._chain is None:
            if self._model is None:
                raise RuntimeError("Chat 模型未初始化，请检查环境变量")
            structured = self._model.with_structured_output(QueryClassification)
            self._chain = self._prompt_template | structured
        return self._chain

    def classify(self, query: str) -> QueryClassification:
        """对用户查询进行分类。

        Args:
            query: 用户原始查询

        Returns:
            QueryClassification 对象

        Raises:
            ValueError: 模型返回无法解析的结果时
        """
        chain = self._get_chain()
        result = chain.invoke({"query": query})
        self._logger.info(
            f"[QueryRouter] query='{query[:50]}...' | "
            f"is_chat={result.is_chat} | is_irrelevant={result.is_irrelevant} | "
            f"type={result.question_type} | confidence={result.confidence:.2f}"
        )
        return result
