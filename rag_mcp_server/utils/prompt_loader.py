"""提示词加载器 — 独立于主项目。"""

from rag_mcp_server.utils.path import get_abs_path
from rag_mcp_server.utils.logger import logger


def get_prompts(prompt_path: str) -> str | None:
    try:
        path = get_abs_path(prompt_path)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"加载提示词文件失败 {prompt_path}: {e}")
        return None
