"""独立配置加载 — 不依赖主项目的 agent_material 包。"""

import os
import yaml
from rag_mcp_server.utils.path import get_abs_path
from rag_mcp_server.utils.logger import logger


def _load_yaml(path: str) -> dict:
    abs_path = get_abs_path(path)
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"加载配置失败 {path}: {e}")
        return {}


# ---- 核心配置 ----
vector_store_config: dict = _load_yaml("config/vector_store_config.yaml")
agent_config: dict = _load_yaml("config/agent_config.yaml")
prompt_config: dict = _load_yaml("config/prompts_config.yaml")
queue_config: dict = _load_yaml("config/queue_config.yaml")

# ---- 环境变量覆盖 / 校验 ----
if not os.environ.get("DASHSCOPE_API_KEY"):
    logger.warning("[Config] DASHSCOPE_API_KEY 未设置，Embedding 模型初始化将失败")

if not os.environ.get("ANTHROPIC_API_KEY"):
    logger.warning("[Config] ANTHROPIC_API_KEY 未设置，Chat 模型初始化将失败")
