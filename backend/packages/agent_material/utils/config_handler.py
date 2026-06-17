import os
import yaml
from agent_material.utils.path_tools import get_abs_path


def get_rag_config():
    rag_config_path = get_abs_path("config/rag_config.yaml")
    with open(rag_config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_agent_config():
    agent_config_path = get_abs_path("config/agent_config.yaml")
    with open(agent_config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_vector_store_config():
    vector_store_config_path = get_abs_path("config/vector_store_config.yaml")
    with open(vector_store_config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# def get_chroma_config():
#     """Backward compatibility alias for get_vector_store_config."""
#     return get_vector_store_config()


def get_prompt_config():
    prompt_config_path = get_abs_path("config/prompt_config.yaml")
    with open(prompt_config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_sc_config():
    sc_config_path = get_abs_path("config/sc_config.yaml")
    with open(sc_config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# def get_mp_config():
#     mp_config_path = get_abs_path("config/mp_config.yaml")
#     with open(mp_config_path, "r", encoding="utf-8") as f:
#         return yaml.safe_load(f)


def get_memory_config():
    memory_config_path = get_abs_path("config/memory_config.yaml")
    with open(memory_config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_queue_config():
    queue_config_path = get_abs_path("config/queue_config.yaml")
    with open(queue_config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


rag_config = get_rag_config()
agent_config = get_agent_config()
vector_store_config = get_vector_store_config()
# chroma_config = vector_store_config  # backward compatibility alias
prompt_config = get_prompt_config()
sc_config = get_sc_config()
# mp_config = get_mp_config()
memory_config = get_memory_config()
queue_config = get_queue_config()

if __name__ == "__main__":
    print(rag_config)
    print(agent_config)
    print(chroma_config)
    print(prompt_config)
    print(sc_config)
