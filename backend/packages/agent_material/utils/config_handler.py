import os
import re
from typing import Any

import yaml
from dotenv import load_dotenv

from agent_material.utils.path_tools import get_abs_path

load_dotenv()

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """递归展开 YAML 中的 ``${NAME:-default}`` 环境变量表达式。"""
    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            configured = os.environ.get(name)
            if configured not in (None, ""):
                return configured
            return default or ""

        return _ENV_PATTERN.sub(_replace, value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return _expand_env(yaml.safe_load(file) or {})


def get_rag_config():
    rag_config_path = get_abs_path("config/rag_config.yaml")
    return _load_yaml(rag_config_path)


def get_agent_config():
    agent_config_path = get_abs_path("config/agent_config.yaml")
    return _load_yaml(agent_config_path)


def get_vector_store_config():
    vector_store_config_path = get_abs_path("config/vector_store_config.yaml")
    return _load_yaml(vector_store_config_path)


def get_prompt_config():
    prompt_config_path = get_abs_path("config/prompt_config.yaml")
    return _load_yaml(prompt_config_path)


def get_sc_config():
    sc_config_path = get_abs_path("config/sc_config.yaml")
    return _load_yaml(sc_config_path)


def get_memory_config():
    memory_config_path = get_abs_path("config/memory_config.yaml")
    return _load_yaml(memory_config_path)


def get_queue_config():
    queue_config_path = get_abs_path("config/queue_config.yaml")
    return _load_yaml(queue_config_path)


rag_config = get_rag_config()
agent_config = get_agent_config()
vector_store_config = get_vector_store_config()
prompt_config = get_prompt_config()
sc_config = get_sc_config()
memory_config = get_memory_config()
queue_config = get_queue_config()

if __name__ == "__main__":
    print(rag_config)
    print(agent_config)
    print(prompt_config)
    print(sc_config)
