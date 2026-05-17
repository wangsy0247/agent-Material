import os
from utils.path_tools import get_abs_path
from utils.logger import logger
from utils.config_handler import prompt_config


def get_prompts(prompt_path: str):
    try:
        path = get_abs_path(prompt_path)
        return open(path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"加载{prompt_path}文件失败: {e}")
        return None
