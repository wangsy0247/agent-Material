import os
from rag.vector_store import VectorStore
from utils.prompt_loader import get_prompts
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from model.model_factory import chat_model
from utils.config_handler import prompt_config
from utils.path_tools import get_abs_path
from utils.langsmith_setup import get_run_config


class RagService:
    def __init__(self):
        self.vector_store = VectorStore()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = get_prompts(get_abs_path(prompt_config["rag_prompt_path"]))
        self.prompts_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        return self.prompts_template | chat_model | StrOutputParser()

    def retriever_docx(self, query: str):
        """执行检索，返回相关文档列表"""
        return self.retriever.invoke(query)

    def get_summary(self, query: str):
        """基于检索结果生成回答"""
        context_docx = self.retriever_docx(query)

        # 相似度阈值过滤后可能返回空结果
        if not context_docx:
            return "根据现有资料无法回答该问题。"

        # 精简上下文拼接，只保留来源和正文，节省 token
        context = ""
        for i, doc in enumerate(context_docx, start=1):
            source = doc.metadata.get("source", "未知来源")
            # 只保留文件名，不保留完整路径
            source_name = (
                os.path.basename(source) if isinstance(source, str) else "未知来源"
            )
            context += f"[参考资料{i}] 来源：{source_name}\n{doc.page_content}\n\n"

        return self.chain.invoke(
            {"input": query, "context": context},
            config=get_run_config(
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

        for chunk in self.chain.stream(
            {"input": query, "context": context},
            config=get_run_config(
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
