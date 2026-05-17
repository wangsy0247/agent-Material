import os


def get_project_path() -> str:
    """
    获取项目根目录的绝对路径
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)


def get_abs_path(relativa_path: str) -> str:
    """
    根据相对路径获取绝对路径
    """
    # 确保路径格式正确，处理不同操作系统的路径分隔符
    return os.path.normpath(os.path.join(get_project_path(), relativa_path))
