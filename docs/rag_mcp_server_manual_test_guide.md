# RAG MCP Server 手动测试教程

## 前置条件

```bash
# 确认环境变量已设置
echo $DASHSCOPE_API_KEY
echo $ANTHROPIC_API_KEY

# 确认 pgvector Docker 在运行
docker ps | grep pgvector

# 确认 pgvector Python 包可用
python -c "from pgvector.sqlalchemy import Vector; print('OK')"
```

---

## 方法一：stdio 模式手动交互（推荐，最简单）

stdio 模式下，Server 通过标准输入输出与你通信，使用 [JSON-RPC](https://www.jsonrpc.org/) 格式。

### 步骤 1：启动 Server

```bash
cd /mnt/d/Langchain_study/Agent智能体/Agent\ Material
python -m rag_mcp_server --transport stdio
```

> 你会看到 Server 启动日志输出到 stderr，stdin/stdout 用于 MCP 通信。

### 步骤 2：发送 MCP 初始化请求

在**另一个终端**中，向正在运行的 Server 进程发送 JSON-RPC 消息。

由于 stdio 模式需要精确控制换行，我们用 Python 脚本代替手动输入：

```python
import json
import subprocess

proc = subprocess.Popen(
    ["python", "-m", "rag_mcp_server", "--transport", "stdio"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    cwd="/mnt/d/Langchain_study/Agent智能体/Agent Material"
)

def send(req: dict):
    msg = json.dumps(req) + "\n"
    proc.stdin.write(msg)
    proc.stdin.flush()

send({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "1.0"}
    }
})

response = json.loads(proc.stdout.readline())
print("初始化响应:", json.dumps(response, indent=2, ensure_ascii=False))

# 发送 initialized 通知
send({
    "jsonrpc": "2.0",
    "method": "notifications/initialized"
})
```

保存为 `test_stdio.py` 后运行：

```bash
python test_stdio.py
```

### 步骤 3：列出工具

在上面的脚本中继续添加：

```python
send({
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list"
})

response = json.loads(proc.stdout.readline())
print("工具列表:", json.dumps(response, indent=2, ensure_ascii=False))
```

### 步骤 4：调用工具

#### 4.1 rag_get_stats — 查看向量库统计

```python
send({
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
        "name": "rag_get_stats",
        "arguments": {}
    }
})

response = json.loads(proc.stdout.readline())
print("统计信息:", json.dumps(response, indent=2, ensure_ascii=False))
```

#### 4.2 rag_search — 检索文档

```python
send({
    "jsonrpc": "2.0",
    "id": 4,
    "method": "tools/call",
    "params": {
        "name": "rag_search",
        "arguments": {"query": "锂离子电池正极材料"}
    }
})

response = json.loads(proc.stdout.readline())
print("检索结果:", json.dumps(response, indent=2, ensure_ascii=False))
```

#### 4.3 rag_query — RAG 问答

```python
send({
    "jsonrpc": "2.0",
    "id": 5,
    "method": "tools/call",
    "params": {
        "name": "rag_query",
        "arguments": {"query": "固态电解质的优缺点"}
    }
})

response = json.loads(proc.stdout.readline())
print("问答结果:", json.dumps(response, indent=2, ensure_ascii=False))
```

#### 4.4 rag_load_documents — 加载文档

```python
send({
    "jsonrpc": "2.0",
    "id": 6,
    "method": "tools/call",
    "params": {
        "name": "rag_load_documents",
        "arguments": {}
    }
})

response = json.loads(proc.stdout.readline())
print("加载结果:", json.dumps(response, indent=2, ensure_ascii=False))
```

#### 4.5 rag_add_documents — 手动添加文档

```python
send({
    "jsonrpc": "2.0",
    "id": 7,
    "method": "tools/call",
    "params": {
        "name": "rag_add_documents",
        "arguments": {
            "documents": ["硅基负极材料具有高理论容量。", "固态电解质比液态更安全。"],
            "sources": ["manual_note_1", "manual_note_2"]
        }
    }
})

response = json.loads(proc.stdout.readline())
print("添加结果:", json.dumps(response, indent=2, ensure_ascii=False))
```

#### 4.6 rag_delete_by_source — 删除文档

```python
send({
    "jsonrpc": "2.0",
    "id": 8,
    "method": "tools/call",
    "params": {
        "name": "rag_delete_by_source",
        "arguments": {"source": "manual_note_1"}
    }
})

response = json.loads(proc.stdout.readline())
print("删除结果:", json.dumps(response, indent=2, ensure_ascii=False))
```

---

## 方法二：使用 MCP Inspector（官方 GUI 工具）

MCP Inspector 是一个官方提供的交互式 Web UI，可以可视化地测试 MCP Server。

### 安装

```bash
npx @modelcontextprotocol/inspector@latest
```

> 需要 Node.js 环境。如果没有，使用下面的 Python 方法。

### 启动 Inspector 并连接 RAG Server

```bash
cd /mnt/d/Langchain_study/Agent智能体/Agent\ Material
npx @modelcontextprotocol/inspector@latest \
  --command "python" \
  --args "-m,rag_mcp_server,--transport,stdio"
```

浏览器会自动打开 `http://localhost:5173`，你可以：
- 点击 **List Tools** 查看所有工具
- 选择工具后填写参数，点击 **Call Tool** 调用
- 查看 JSON 请求和响应

---

## 方法三：SSE 模式 + HTTP 客户端（curl / Postman）

SSE 模式下 Server 暴露 HTTP 端点，可以用任何 HTTP 客户端测试。

### 步骤 1：启动 SSE Server

```bash
cd /mnt/d/Langchain_study/Agent智能体/Agent\ Material
python -m rag_mcp_server --transport sse --port 8003
```

### 步骤 2：使用 curl 测试

#### 初始化会话

```bash
curl -N http://127.0.0.1:8003/sse
```

> 这会建立 SSE 连接，持续接收事件流。按 Ctrl+C 退出。

实际测试需要用 MCP SDK 或编写客户端脚本。推荐使用方法四的 Python 客户端。

---

## 方法四：Python 客户端脚本（最方便）

### 完整测试脚本

创建 `manual_test_rag.py`：

```python
#!/usr/bin/env python3
"""RAG MCP Server 手动测试脚本 — 使用官方 MCP Python SDK"""

import asyncio
import json
import os
import sys

PROJECT_ROOT = "/mnt/d/Langchain_study/Agent智能体/Agent Material"
os.chdir(PROJECT_ROOT)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_all_tools():
    params = StdioServerParameters(
        command="python",
        args=["-m", "rag_mcp_server"],
        env=os.environ.copy(),  # 继承当前环境变量（含 API Key）
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. 初始化
            print("=" * 50)
            print("[1/7] 初始化会话...")
            await session.initialize()
            print("✅ 初始化成功\n")

            # 2. 列出工具
            print("[2/7] 列出工具...")
            tools = await session.list_tools()
            for t in tools.tools:
                print(f"  • {t.name}: {t.description[:50]}...")
            print()

            # 3. rag_get_stats
            print("[3/7] 调用 rag_get_stats...")
            result = await session.call_tool("rag_get_stats", arguments={})
            text = result.content[0].text
            try:
                stats = json.loads(text)
                print(f"  文档数: {stats['document_count']}")
                print(f"  Collection: {stats['collection_name']}")
                print(f"  连接: {stats['connection_string_masked']}")
            except Exception:
                print(f"  返回: {text[:200]}")
            print()

            # 4. rag_search
            print("[4/7] 调用 rag_search (query='锂离子电池')...")
            result = await session.call_tool(
                "rag_search", arguments={"query": "锂离子电池"}
            )
            text = result.content[0].text
            try:
                data = json.loads(text)
                docs = data.get("results", [])
                print(f"  检索到 {len(docs)} 条文档")
                for i, doc in enumerate(docs[:3], 1):
                    snippet = doc.get("page_content", "")[:60].replace("\n", " ")
                    print(f"    #{i} {snippet}...")
            except Exception:
                print(f"  返回: {text[:200]}")
            print()

            # 5. rag_query
            print("[5/7] 调用 rag_query (query='固态电解质的优缺点')...")
            result = await session.call_tool(
                "rag_query", arguments={"query": "固态电解质的优缺点"}
            )
            text = result.content[0].text
            print(f"  回答: {text[:200]}...")
            print()

            # 6. rag_add_documents
            print("[6/7] 调用 rag_add_documents...")
            result = await session.call_tool(
                "rag_add_documents",
                arguments={
                    "documents": ["测试文档内容 1", "测试文档内容 2"],
                    "sources": ["test_1", "test_2"]
                }
            )
            text = result.content[0].text
            print(f"  结果: {text}")
            print()

            # 7. rag_delete_by_source
            print("[7/7] 调用 rag_delete_by_source...")
            result = await session.call_tool(
                "rag_delete_by_source", arguments={"source": "test_1"}
            )
            text = result.content[0].text
            print(f"  结果: {text}")
            print()

            print("=" * 50)
            print("所有工具测试完成！")


if __name__ == "__main__":
    asyncio.run(test_all_tools())
```

运行：

```bash
cd /mnt/d/Langchain_study/Agent智能体/Agent\ Material
python manual_test_rag.py
```

---

## 方法五：逐个工具独立测试

如果你想单独测试某个工具，可以写最小化脚本：

### 只测试 rag_get_stats

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(
        command="python", args=["-m", "rag_mcp_server"],
        env={"PYTHONPATH": "/mnt/d/Langchain_study/Agent智能体/Agent Material/backend/packages"}
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("rag_get_stats", {})
            print(result.content[0].text)

asyncio.run(main())
```

### 只测试 rag_load_documents

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command="python", args=["-m", "rag_mcp_server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("开始加载文档...")
            result = await session.call_tool("rag_load_documents", {})
            print(result.content[0].text)

asyncio.run(main())
```

---

## 常见问题排查

### Q1: 报错 `ModuleNotFoundError: No module named 'pgvector'`

```bash
# 从已安装的 conda 环境复制
conda_env="/home/wsy/anaconda3/envs/langchain/lib/python3.12/site-packages"
target="/home/wsy/anaconda3/lib/python3.13/site-packages"
cp -r "$conda_env/pgvector" "$target/"
cp -r "$conda_env/pgvector-0.4.2.dist-info" "$target/"
```

### Q2: 报错 `DASHSCOPE_API_KEY 未设置`

```bash
export DASHSCOPE_API_KEY=你的密钥
export ANTHROPIC_API_KEY=你的密钥
```

### Q3: rag_search / rag_query 返回 0 条结果

因为数据库为空。先运行 `rag_load_documents` 加载文档：

```bash
# 如果想重新加载所有文档，先删除 md5 记录
rm /mnt/d/Langchain_study/Agent智能体/Agent\ Material/md5.text
python manual_test_rag.py
```

### Q4: PDF 加载失败 (`pypdf` not found)

```bash
pip install pypdf
```

### Q5: PostgreSQL 连接失败

```bash
# 检查 Docker 容器
docker ps | grep pgvector

# 检查端口连通性
python -c "import psycopg; psycopg.connect('postgresql://postgres:postgres@127.0.0.1:5432/postgres'); print('OK')"
```

---

## 预期输出示例

当数据库有数据时，`rag_search` 的典型输出：

```json
{
  "results": [
    {
      "page_content": "磷酸铁锂（LiFePO4）是一种橄榄石结构的锂离子电池正极材料...",
      "metadata": {"source": "docs/cathode_LiFePO4.md", "category": "电池材料"}
    },
    {
      "page_content": "三元材料（NCM）是指镍钴锰酸锂...",
      "metadata": {"source": "docs/cathode_NCM.md", "category": "电池材料"}
    }
  ],
  "count": 2
}
```

当数据库为空时，`rag_query` 的输出：

```
根据现有资料无法回答该问题。
```
