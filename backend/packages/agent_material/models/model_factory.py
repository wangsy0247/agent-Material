import os

from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import DashScopeEmbeddings
from agent_material.utils.config_handler import agent_config, rag_config
from agent_material.utils.langsmith_setup import setup_langsmith

# 程序启动时尝试初始化 LangSmith（静默失败不影响业务）
setup_langsmith()


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[BaseChatModel | Embeddings]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[BaseChatModel | Embeddings]:
        return ChatAnthropic(
            model=agent_config["chat_model_name"],
            max_tokens=agent_config["max_tokens"],
            streaming=True,
        )


class EmbeddingModelFactory(BaseModelFactory):
    def generator(self) -> Optional[BaseChatModel | Embeddings]:
        return DashScopeEmbeddings(model=rag_config["embedding_model_name"])


chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingModelFactory().generator()

if __name__ == "__main__":
    print(chat_model)
    print(embedding_model)
