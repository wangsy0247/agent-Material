import logging
from utils.path_tools import get_abs_path
import os
from datetime import datetime


LOG_ROOT = get_abs_path("logs")

os.makedirs(LOG_ROOT, exist_ok=True)

DEFAULT_LOG_FOMATETER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s %(filename)s:%(lineno)d - %(message)s"
)


def get_logger(
    name: str = "default",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_file_name=None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 日志格式
    console_formatter = logging.Formatter(
        "%(name)s - %(levelname)s %(filename)s:%(lineno)d - %(message)s"
    )
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    if logger.handlers:
        return logger

    # 控制台日志
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if not log_file_name:
        log_file_name = os.path.join(
            LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        )

    # 文件日志
    file_handler = logging.FileHandler(f"{LOG_ROOT}/{name}.log")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


logger = get_logger()

if __name__ == "__main__":
    logger.info("hello world")
    logger.debug("hello debug")
    logger.error("hello error")
    logger.warning("hello warning")
