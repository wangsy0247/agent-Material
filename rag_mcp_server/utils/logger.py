"""日志工具 — 独立于主项目。"""

import logging
import os
from datetime import datetime

LOG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
os.makedirs(LOG_ROOT, exist_ok=True)

_console_formatter = logging.Formatter(
    "%(name)s - %(levelname)s %(filename)s:%(lineno)d - %(message)s"
)
_file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)


def get_logger(name: str = "default") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_console_formatter)
    logger.addHandler(console_handler)

    log_file = os.path.join(LOG_ROOT, f"{name}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_file_formatter)
    logger.addHandler(file_handler)

    return logger


logger = get_logger()
