"""LangFuse 可观测性配置模块。

默认使用云端 LangFuse（https://cloud.langfuse.com），
提供 LangChain Callback Handler 自动追踪 LLM / Chain / Agent 调用。

使用方式:
  1. 在 .env 中设置（云端项目管理页生成密钥对）:
     LANGFUSE_PUBLIC_KEY="pk-lf-..."
     LANGFUSE_SECRET_KEY="sk-lf-..."
     LANGFUSE_HOST="https://cloud.langfuse.com"   # 自部署时改为自建地址
  2. 程序启动时调用 setup_langfuse()
  3. 在 chain.invoke(config={"callbacks": [langfuse_handler]}) 中传入

若未设置 LANGFUSE_SECRET_KEY，追踪静默跳过，不影响业务。

兼容: langfuse SDK >= 3.0 (CallbackHandler 不再接受 secret_key, 需先初始化 Langfuse client)
"""

from __future__ import annotations

import os
from typing import Any

from agent_material.utils.logger import logger

_handler: Any = None
_client: Any = None
_enabled: bool = False


def setup_langfuse(
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
    enabled: bool = True,
) -> bool:
    """初始化 LangFuse 客户端 + LangChain Callback Handler。

    SDK v3+: CallbackHandler 只接受 public_key，不直接接受 secret_key/host。
    需先创建 Langfuse() client（注册全局单例），
    CallbackHandler 通过 public_key 查找已注册的 client。

    Args:
        public_key: LangFuse Public Key，默认从 LANGFUSE_PUBLIC_KEY 读取
        secret_key: LangFuse Secret Key，默认从 LANGFUSE_SECRET_KEY 读取
        host: LangFuse 服务地址，默认 https://cloud.langfuse.com
        enabled: 是否启用追踪

    Returns:
        是否成功启用
    """
    global _handler, _client, _enabled

    env_enabled = os.environ.get("LANGFUSE_ENABLED", "true").strip().lower()
    if not enabled or env_enabled in {"0", "false", "no", "off"}:
        _handler = None
        _client = None
        _enabled = False
        logger.info("[LangFuse] env LANGFUSE_ENABLED=false，跳过初始化")
        return False

    try:
        pk = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        sk = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
        if not sk:
            logger.info("[LangFuse] LANGFUSE_SECRET_KEY 未设置，追踪未启用")
            return False

        host = host or os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        os.environ.setdefault("LANGFUSE_HOST", host)

        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        # Step 1: 创建 Langfuse client（注册全局单例，负责数据上报）
        _client = Langfuse(
            public_key=pk,
            secret_key=sk,
            host=host,
        )
        # Step 2: 创建 CallbackHandler（通过 public_key 查找已注册的 client）
        _handler = CallbackHandler(public_key=pk)
        _enabled = True
        logger.info(f"[LangFuse] 追踪已启用，host={host}")
        return True

    except ImportError:
        logger.warning("[LangFuse] pip install langfuse 未安装，追踪跳过")
        return False
    except Exception as e:
        logger.warning(f"[LangFuse] 初始化失败: {e}，追踪跳过")
        return False


def get_langfuse_handler() -> Any | None:
    """获取 LangFuse Callback Handler（用于传入 chain.invoke(config={callbacks: ...})）。"""
    return _handler if _enabled else None


def is_tracing_enabled() -> bool:
    return _enabled


def get_run_config(
    run_name: str = "agent_react",
    query: str = "",
    tags: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    session_id: str = "",
    user_id: str = "default",
) -> dict[str, Any]:
    """构建 LangChain run config，传入 LangFuse Callback + metadata。

    Args:
        run_name: 本次运行的名称
        query: 用户查询，写入 metadata 便于筛选
        tags: 标签列表
        extra_metadata: 额外自定义元数据
        session_id: LangFuse session ID（用于按会话分组）
        user_id: 用户 ID

    Returns:
        LangChain 的 config 字典，可直接传给 invoke / stream / ainvoke
    """
    metadata: dict[str, Any] = {"query": query, "langfuse_session_id": session_id, "langfuse_user_id": user_id}
    if extra_metadata:
        metadata.update(extra_metadata)

    config: dict[str, Any] = {
        "run_name": run_name,
        "metadata": metadata,
    }
    if tags:
        config["tags"] = tags

    handler = get_langfuse_handler()
    if handler:
        callbacks = config.setdefault("callbacks", [])
        if isinstance(callbacks, list):
            callbacks.append(handler)

    return config


def log_feedback(
    trace_id: str,
    key: str = "user_feedback",
    score: float = 0.0,
    comment: str = "",
) -> None:
    """向 LangFuse 发送用户反馈。

    Args:
        trace_id: LangFuse Trace ID
        key: 反馈维度（如 "helpfulness", "accuracy"）
        score: 0.0 ~ 1.0
        comment: 评论
    """
    if not _enabled:
        return
    try:
        _client.score(
            trace_id=trace_id,
            name=key,
            value=score,
            comment=comment,
        )
        logger.info(f"[LangFuse] 反馈已记录: {key}={score}")
    except Exception as e:
        logger.warning(f"[LangFuse] 反馈记录失败: {e}")


def flush() -> None:
    """等待 LangFuse 缓冲区刷新（优雅关闭时调用）。"""
    if _enabled and _client:
        try:
            _client.flush()
            logger.info("[LangFuse] 缓冲区已刷新")
        except Exception:
            pass
