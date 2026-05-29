import os


def get_project_path() -> str:
    """
    获取项目根目录的绝对路径

    从当前文件向上查找，直到找到同时包含 backend/ 和 config/ 目录的位置
    （这是项目根目录的标志性特征）
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 当前文件在 backend/packages/agent_material/utils/path_tools.py
    while current_dir != os.path.dirname(current_dir):
        # 项目根目录标志：同时存在 backend/ 和 config/
        if os.path.isdir(os.path.join(current_dir, "backend")) and os.path.isdir(
            os.path.join(current_dir, "config")
        ):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return current_dir


def get_abs_path(relativa_path: str) -> str:
    """
    根据相对路径获取绝对路径（相对于项目根目录）
    """
    return os.path.normpath(os.path.join(get_project_path(), relativa_path))
