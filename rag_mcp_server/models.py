"""独立模型初始化 — 不依赖主项目的 model_factory。"""

import os
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings

from rag_mcp_server.config import agent_config, vector_store_config
from rag_mcp_server.utils.logger import logger


# ---- 通义千问 (Qwen) 默认方案 ----
def _create_chat_model() -> Optional[BaseChatModel]:
    """初始化 Chat 模型（通义千问 / Qwen）。"""
    try:
        model_name = agent_config.get("chat_model_name", "qwen3.5-flash")
        max_tokens = agent_config.get("max_tokens", 8192)
        return ChatOpenAI(
            model_name=model_name,
            max_tokens=max_tokens,
            streaming=True,
        )
    except Exception as e:
        logger.error(f"[Models] Chat 模型初始化失败: {e}")
        return None


def _create_embedding_model() -> Optional[Embeddings]:
    """初始化 Embedding 模型（DashScope）。"""
    try:
        model_name = vector_store_config.get("embedding_model_name", "text-embedding-v4")
        return DashScopeEmbeddings(model=model_name, dimensions=1024)
    except Exception as e:
        logger.error(f"[Models] Embedding 模型初始化失败: {e}")
        return None


chat_model = _create_chat_model()
embedding_model = _create_embedding_model()
