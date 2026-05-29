"""路径工具 — 独立于主项目。"""

import os


def get_project_path() -> str:
    """
    获取项目根目录的绝对路径。

    从 rag_mcp_server 包位置向上查找，直到找到同时包含 backend/ 和 config/ 目录的位置。
    """
    # 当前文件在 rag_mcp_server/utils/path.py
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    while current_dir != os.path.dirname(current_dir):
        if os.path.isdir(os.path.join(current_dir, "backend")) and os.path.isdir(
            os.path.join(current_dir, "config")
        ):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return current_dir


def get_abs_path(relative_path: str) -> str:
    """根据相对路径获取绝对路径（相对于项目根目录）。"""
    return os.path.normpath(os.path.join(get_project_path(), relative_path))
