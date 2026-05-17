"""LangSmith 可观测性配置模块。

使用方式：
1. 在 .env 文件或环境变量中设置：
   export LANGCHAIN_API_KEY="ls-your-api-key"
   export LANGCHAIN_PROJECT="agent-material-rag"
2. 在程序入口调用 setup_langsmith()
3. 所有 LangChain 组件调用会自动上报到 LangSmith

如果未设置 LANGCHAIN_API_KEY，LangSmith 追踪会静默跳过，不影响业务运行。
"""

import os
from typing import Any

from utils.logger import logger


def setup_langsmith() -> bool:
    """检查并配置 LangSmith 环境变量，返回是否启用追踪。"""
    api_key = os.environ.get("LANGCHAIN_API_KEY")

    if not api_key:
        logger.info("[LangSmith] LANGCHAIN_API_KEY 未设置，追踪未启用")
        return False

    # 确保追踪开关已打开
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    # 默认项目名称
    os.environ.setdefault("LANGCHAIN_PROJECT", "agent-material-rag")
    # 可选：设置 LangSmith 端点（私有化部署时修改）
    # os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    project = os.environ.get("LANGCHAIN_PROJECT", "unknown")
    logger.info(f"[LangSmith] 追踪已启用，项目: {project}")
    return True


def is_tracing_enabled() -> bool:
    """检查 LangSmith 追踪是否已启用。"""
    return (
        os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
        and bool(os.environ.get("LANGCHAIN_API_KEY"))
    )


def get_run_config(
    run_name: str,
    query: str = "",
    tags: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建 LangChain run config，用于 chain / agent / retriever 的调用。

    Args:
        run_name: 本次运行的名称（如 "rag_generate", "agent_react"）
        query: 用户查询，会写入 metadata 便于在 LangSmith 中筛选
        tags: 标签列表（如 ["rag", "v1"]）
        extra_metadata: 额外的自定义元数据

    Returns:
        LangChain 的 config 字典，可直接传给 invoke / stream / ainvoke
    """
    metadata: dict[str, Any] = {"query": query}
    if extra_metadata:
        metadata.update(extra_metadata)

    config: dict[str, Any] = {
        "run_name": run_name,
        "metadata": metadata,
    }
    if tags:
        config["tags"] = tags

    return config


def log_feedback(run_id: str, key: str, score: float, comment: str = "") -> None:
    """向 LangSmith 发送人工反馈分数（用于后续评估和标注）。

    Args:
        run_id: LangSmith Trace 的 run ID（从回调中获取）
        key: 反馈维度（如 "faithfulness", "helpfulness"）
        score: 分数（通常 0~1）
        comment: 可选的文字评论
    """
    try:
        from langsmith import Client

        client = Client()
        client.create_feedback(
            run_id=run_id,
            key=key,
            score=score,
            comment=comment,
        )
        logger.info(f"[LangSmith] 反馈已记录: {key}={score}")
    except Exception as e:
        logger.warning(f"[LangSmith] 反馈记录失败: {e}")


if __name__ == "__main__":
    enabled = setup_langsmith()
    print(f"Tracing enabled: {enabled}")
    print(f"Config example: {get_run_config('test_run', query='测试查询')}")
