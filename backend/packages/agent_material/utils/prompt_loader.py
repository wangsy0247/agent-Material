from agent_material.utils.path_tools import get_abs_path
from agent_material.utils.logger import logger


def get_prompts(prompt_path: str):
    try:
        path = get_abs_path(prompt_path)
        return open(path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"加载{prompt_path}文件失败: {e}")
        return None
