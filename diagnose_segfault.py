"""诊断 Segmentation Fault 的脚本"""
import faulthandler
faulthandler.enable()

import sys
sys.path.insert(0, '.')

print("=== 启用 faulthandler，准备定位 segfault ===")
print()

# 逐步导入，精确定位
modules = [
    ("os, sys", "import os; import sys"),
    ("utils.langsmith_setup", "from utils import langsmith_setup"),
    ("model.model_factory", "from model import model_factory"),
    ("memory.manager", "from memory import manager"),
    ("agent.tools.agent_tools", "from agent.tools import agent_tools"),
    ("agent.React_agent", "from agent import React_agent"),
    ("api.routes.agent", "from api.routes import agent"),
    ("api.routes.rag", "from api.routes import rag"),
    ("api.routes.system", "from api.routes import system"),
    ("api.routes.memory", "from api.routes import memory"),
    ("api_server", "import api_server"),
]

for name, code in modules:
    print(f"Importing: {name} ...", end=" ", flush=True)
    try:
        exec(code)
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

print()
print("=== 所有模块导入成功 ===")
print("=== 尝试启动 uvicorn（无 reload） ===")

import uvicorn
import api_server
import os

config = uvicorn.Config(
    api_server.app,
    host=os.environ.get("AGENT_API_HOST", "0.0.0.0"),
    port=int(os.environ.get("AGENT_API_PORT", "8000")),
    reload=False,
    log_level="info",
)
server = uvicorn.Server(config)

import asyncio
async def run():
    try:
        await asyncio.wait_for(server.serve(), timeout=8.0)
    except asyncio.TimeoutError:
        print("\n=== uvicorn 启动成功（8秒后自动退出）===")

asyncio.run(run())
print("=== 诊断脚本正常结束 ===")
