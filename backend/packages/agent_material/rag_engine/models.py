"""独立模型初始化 — 不依赖主项目的 model_factory。

Embedding 模型统一使用 AsyncDashScopeEmbeddings（异步批量 + 速率限制 + 重试）。
通过同步兼容接口 embed_documents() / embed_query() 与 LangChain PGVector 集成。
"""

import os
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from agent_material.utils.config_handler import agent_config, vector_store_config
from agent_material.utils.logger import logger


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


def _create_embedding_model():
    """初始化 Embedding 模型 — 使用异步批量封装。

    返回 AsyncDashScopeEmbeddings，其 embed_documents()/embed_query() 同步兼容接口
    内部使用 asyncio.run() 调度异步批量 HTTP 请求，自动分批 + 并发 + 速率限制 + 重试。
    """
    try:
        from agent_material.rag_engine.embedding.async_embeddings import AsyncDashScopeEmbeddings

        async_cfg = vector_store_config.get("async_embedding", {})
        model_name = vector_store_config.get("embedding_model_name", "text-embedding-v4")
        return AsyncDashScopeEmbeddings(
            model=model_name,
            dimensions=1024,
            batch_size=async_cfg.get("batch_size", 10),
            max_concurrency=async_cfg.get("max_concurrency", 4),
            rate_limit=async_cfg.get("rate_limit", 10.0),
            max_retries=async_cfg.get("retry_on_failure", 3),
        )
    except Exception as e:
        logger.error(f"[Models] Embedding 模型初始化失败: {e}")
        return None


chat_model = _create_chat_model()
embedding_model = _create_embedding_model()


def get_async_embedding_model():
    """获取异步 Embedding 模型（与 embedding_model 同一实例）。"""
    return embedding_model
