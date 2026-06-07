import os
from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag_mcp_server.rag.vector_store import VectorStore
from rag_mcp_server.utils.prompt_loader import get_prompts
from rag_mcp_server.models import chat_model
from rag_mcp_server.config import prompt_config
from rag_mcp_server.utils.path import get_abs_path


def _get_run_config(
    run_name: str,
    query: str = "",
    tags: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建 LangChain run config（简化版，不依赖 LangSmith）。"""
    metadata: dict[str, Any] = {"query": query}
    if extra_metadata:
        metadata.update(extra_metadata)
    config: dict[str, Any] = {"run_name": run_name, "metadata": metadata}
    if tags:
        config["tags"] = tags
    return config


class RagService:
    def __init__(self):
        self.vector_store = VectorStore()
        prompt_path = prompt_config.get("rag_prompt_path", "rag_mcp_server/prompts/rag_prompt.txt")
        self.prompt_text = get_prompts(get_abs_path(prompt_path)) or ""
        self.prompts_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self._chain = None  # 延迟初始化（避免无 Chat API key 时阻断纯检索）

    def _get_chain(self):
        if self._chain is None:
            if self.model is None:
                raise RuntimeError("Chat 模型未初始化，请检查 ANTHROPIC_API_KEY 环境变量")
            self._chain = self.prompts_template | self.model | StrOutputParser()
        return self._chain

    @property
    def retriever(self):
        """每次获取最新的检索器（支持热更新）。"""
        return self.vector_store.get_retriever()

    def retriever_docx(self, query: str):
        """执行检索，返回相关文档列表"""
        return self.retriever.invoke(query)

    def get_summary(self, query: str):
        """基于检索结果生成回答"""
        context_docx = self.retriever_docx(query)

        if not context_docx:
            return "根据现有资料无法回答该问题。"

        context = ""
        for i, doc in enumerate(context_docx, start=1):
            source = doc.metadata.get("source", "未知来源")
            source_name = (
                os.path.basename(source) if isinstance(source, str) else "未知来源"
            )
            context += f"[参考资料{i}] 来源：{source_name}\n{doc.page_content}\n\n"

        return self._get_chain().invoke(
            {"input": query, "context": context},
            config=_get_run_config(
                run_name="rag_generate",
                query=query,
                tags=["rag", "generate"],
                extra_metadata={"context_doc_count": len(context_docx)},
            ),
        )

    def get_summary_stream(self, query: str):
        """流式生成 RAG 回答"""
        context_docx = self.retriever_docx(query)

        if not context_docx:
            yield "根据现有资料无法回答该问题。"
            return

        context = ""
        for i, doc in enumerate(context_docx, start=1):
            source = doc.metadata.get("source", "未知来源")
            source_name = (
                os.path.basename(source) if isinstance(source, str) else "未知来源"
            )
            context += f"[参考资料{i}] 来源：{source_name}\n{doc.page_content}\n\n"

        for chunk in self._get_chain().stream(
            {"input": query, "context": context},
            config=_get_run_config(
                run_name="rag_generate_stream",
                query=query,
                tags=["rag", "stream"],
                extra_metadata={"context_doc_count": len(context_docx)},
            ),
        ):
            yield chunk


if __name__ == "__main__":
    rag_service = RagService()
    query = "新能源有哪些领域的发展前景比较好"
    print(f"问题: {query}")
    print(f"文档总数: {rag_service.vector_store.get_document_count()}")
    print(f"\n回答:\n{rag_service.get_summary(query)}")
