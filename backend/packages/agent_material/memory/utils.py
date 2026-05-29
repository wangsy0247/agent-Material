"""
记忆系统工具函数
"""

import time
import re
from datetime import datetime


def estimate_tokens(text: str) -> int:
    """估算文本的 token 数量（轻量级，无需 tiktoken）
    
    中文按字符数 × 0.6，英文按单词数 × 1.3，取较大值
    """
    if not text:
        return 0
    
    # 中文字符数
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    # 英文单词数（粗略）
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    
    # 中文 token 估算：每字约 0.6 token（GPT类分词器）
    # 英文 token 估算：每词约 1.3 token
    return int(max(chinese_chars * 0.6 + english_words * 1.3, len(text) * 0.25))


def estimate_messages_tokens(messages: list[dict]) -> int:
    """估算消息列表的总 token 数"""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        total += 4  # 每条消息的格式开销（role, metadata等）
    return total


def format_timestamp(ts: float | None = None) -> str:
    """格式化时间戳为可读字符串"""
    if ts is None:
        ts = time.time()
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def format_relative_time(ts: float) -> str:
    """返回相对时间描述（如"2小时前"）"""
    delta = time.time() - ts
    if delta < 60:
        return "刚刚"
    elif delta < 3600:
        return f"{int(delta / 60)} 分钟前"
    elif delta < 86400:
        return f"{int(delta / 3600)} 小时前"
    elif delta < 2592000:
        return f"{int(delta / 86400)} 天前"
    else:
        return format_timestamp(ts)


def sanitize_filename(name: str) -> str:
    """将字符串转为安全的文件名"""
    return re.sub(r'[^\w\-_.]', '_', name)[:64]


def truncate_text(text: str, max_chars: int, suffix: str = "...") -> str:
    """截断文本到指定字符数"""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix
