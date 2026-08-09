"""诊断 Segmentation Fault 的脚本"""
import faulthandler
faulthandler.enable()

import sys
from pathlib import Path

# 项目根目录
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
# backend/packages 用于导入 agent_material
sys.path.insert(0, str(project_root / "backend" / "packages"))

print("=== 启用 faulthandler，准备定位 segfault ===")
print()

# 逐步导入，精确定位
modules = [
    ("os, sys", "import os; import sys"),
    ("agent_material.utils.langsmith_setup", "from agent_material.utils import langsmith_setup"),
    ("agent_material.models.model_factory", "from agent_material.models import model_factory"),
    ("agent_material.memory.manager", "from agent_material.memory import manager"),
    ("agent_material.agents.tools.agent_tools", "from agent_material.agents.tools import agent_tools"),
    ("agent_material.agents.React_agent", "from agent_material.agents import React_agent"),
    ("backend.app.gateway.routers.agent", "from backend.app.gateway.routers import agent"),
    ("backend.app.gateway.routers.rag", "from backend.app.gateway.routers import rag"),
    ("backend.app.gateway.routers.system", "from backend.app.gateway.routers import system"),
    ("backend.app.gateway.routers.memory", "from backend.app.gateway.routers import memory"),
    ("backend.app.gateway.app", "from backend.app.gateway import app"),
]

for name, code in modules:
    print(f"Importing: {name} ...", end=" ", flush=True)
    try:
        exec(code)
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
