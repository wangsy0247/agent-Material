#!/usr/bin/env python3
"""
RAG MCP Server 独立测试脚本

支持两种测试方式：
    1. stdio 传输（子进程）——默认，最常用
    2. SSE 传输（HTTP 流式）——需要先启动 SSE server

运行方式：
    cd /mnt/d/Langchain_study/Agent智能体/Agent Material
    python scripts/test_rag_mcp_server.py

环境要求：
    - mcp>=1.0.0
    - pgvector Python 包（pip install pgvector）
    - PostgreSQL + pgvector 扩展（Docker 容器已启动）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any

# 确保项目根目录和 backend/packages 在路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGES_PATH = os.path.join(PROJECT_ROOT, "backend", "packages")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if PACKAGES_PATH not in sys.path:
    sys.path.insert(0, PACKAGES_PATH)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ------------------------------------------------------------------
# 测试辅助
# ------------------------------------------------------------------


def _print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def _pretty_json(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


# ------------------------------------------------------------------
# stdio 模式测试
# ------------------------------------------------------------------


async def test_stdio():
    """通过 stdio 子进程测试 RAG MCP Server。"""
    _print_section("stdio 模式测试")

    env = os.environ.copy()
    env["PYTHONPATH"] = PACKAGES_PATH
    params = StdioServerParameters(
        command="python",
        args=["-m", "rag_mcp_server"],
        env=env,
    )

    print(f"[启动] {params.command} {' '.join(params.args)}")
    print(f"[工作目录] {PROJECT_ROOT}")

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. 初始化
            print("\n[1/6] 初始化会话...")
            await session.initialize()
            print("  ✅ 初始化成功")

            # 2. 列出工具
            print("\n[2/6] 列出可用工具...")
            tools = await session.list_tools()
            print(f"  发现 {len(tools.tools)} 个工具:")
            for t in tools.tools:
                print(f"    - {t.name}: {t.description[:60]}...")

            # 3. 测试 rag_get_stats（不依赖数据库有数据，只需能连接）
            print("\n[3/6] 调用 rag_get_stats（获取向量库统计）...")
            t0 = time.time()
            try:
                result = await session.call_tool("rag_get_stats", arguments={})
                elapsed = time.time() - t0
                content = result.content[0].text if result.content else ""
                print(f"  ✅ 耗时 {elapsed:.2f}s")
                try:
                    stats = json.loads(content)
                    print(f"  文档数: {stats.get('document_count', 'N/A')}")
                    print(f"  Collection: {stats.get('collection_name', 'N/A')}")
                except json.JSONDecodeError:
                    print(f"  返回内容: {content[:200]}")
            except Exception as e:
                print(f"  ❌ 失败: {e}")

            # 4. 测试 rag_search（需要向量库有数据才能返回结果）
            print("\n[4/6] 调用 rag_search（检索：锂离子电池正极材料）...")
            t0 = time.time()
            try:
                result = await session.call_tool(
                    "rag_search",
                    arguments={"query": "锂离子电池正极材料"},
                )
                elapsed = time.time() - t0
                content = result.content[0].text if result.content else ""
                print(f"  ✅ 耗时 {elapsed:.2f}s")
                try:
                    data = json.loads(content)
                    results = data.get("results", [])
                    print(f"  检索到 {len(results)} 条文档")
                    for i, doc in enumerate(results[:3], 1):
                        snippet = doc.get("page_content", "")[:80].replace("\n", " ")
                        source = doc.get("metadata", {}).get("source", "未知")
                        print(f"    #{i} [{source}] {snippet}...")
                except json.JSONDecodeError:
                    print(f"  返回内容: {content[:300]}")
            except Exception as e:
                print(f"  ❌ 失败: {e}")

            # 5. 测试 rag_query（需要向量库有数据 + LLM 可用）
            print("\n[5/6] 调用 rag_query（RAG 问答：固态电解质的优缺点）...")
            print("  ⚠️  此调用会触发 LLM，可能需要 10~30 秒")
            t0 = time.time()
            try:
                result = await session.call_tool(
                    "rag_query",
                    arguments={"query": "固态电解质的优缺点"},
                )
                elapsed = time.time() - t0
                content = result.content[0].text if result.content else ""
                print(f"  ✅ 耗时 {elapsed:.2f}s")
                print(f"  回答预览:\n  {content[:300]}...")
            except Exception as e:
                print(f"  ❌ 失败: {e}")

            # 6. 测试 rag_load_documents（扫描 data/ 目录）
            print("\n[6/6] 调用 rag_load_documents（扫描 data/ 目录）...")
            t0 = time.time()
            try:
                result = await session.call_tool("rag_load_documents", arguments={})
                elapsed = time.time() - t0
                content = result.content[0].text if result.content else ""
                print(f"  ✅ 耗时 {elapsed:.2f}s")
                print(f"  结果: {content}")
            except Exception as e:
                print(f"  ❌ 失败: {e}")

    print("\n[stdio 测试结束]")


# ------------------------------------------------------------------
# SSE 模式测试
# ------------------------------------------------------------------


async def test_sse(host: str = "127.0.0.1", port: int = 8003):
    """通过 SSE HTTP 流式测试 RAG MCP Server。"""
    from mcp.client.sse import sse_client

    _print_section("SSE 模式测试")
    url = f"http://{host}:{port}/sse"
    print(f"[连接] {url}")

    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            print("\n[1/3] 初始化会话...")
            await session.initialize()
            print("  ✅ 初始化成功")

            print("\n[2/3] 列出可用工具...")
            tools = await session.list_tools()
            print(f"  发现 {len(tools.tools)} 个工具:")
            for t in tools.tools:
                print(f"    - {t.name}")

            print("\n[3/3] 调用 rag_get_stats...")
            try:
                result = await session.call_tool("rag_get_stats", arguments={})
                content = result.content[0].text if result.content else ""
                print(f"  ✅ 返回:\n  {content[:500]}")
            except Exception as e:
                print(f"  ❌ 失败: {e}")

    print("\n[SSE 测试结束]")


# ------------------------------------------------------------------
# 快速冒烟测试（不启动子进程，直接 import 检查）
# ------------------------------------------------------------------


def smoke_test():
    """快速冒烟测试：验证 server 模块能正确导入，工具注册无误。"""
    _print_section("冒烟测试（无子进程）")

    print("[1/3] 导入 rag_mcp_server.server...")
    from rag_mcp_server.server import create_server
    print("  ✅ 导入成功")

    print("\n[2/3] 创建 FastMCP Server...")
    server = create_server()
    print(f"  ✅ Server 名称: {server.name}")

    print("\n[3/3] 列出工具（同步调用）...")
    import asyncio
    tools = asyncio.run(server.list_tools())
    expected = {
        "rag_search",
        "rag_query",
        "rag_load_documents",
        "rag_add_documents",
        "rag_delete_by_source",
        "rag_get_stats",
    }
    actual = {t.name for t in tools}
    missing = expected - actual
    extra = actual - expected

    print(f"  期望工具: {sorted(expected)}")
    print(f"  实际工具: {sorted(actual)}")
    if missing:
        print(f"  ❌ 缺失工具: {missing}")
    elif extra:
        print(f"  ⚠️  额外工具: {extra}")
    else:
        print("  ✅ 工具列表完全匹配")

    print("\n[冒烟测试结束]")
    return len(missing) == 0


# ------------------------------------------------------------------
# CLI 入口
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="RAG MCP Server 独立测试")
    parser.add_argument(
        "--mode",
        choices=["smoke", "stdio", "sse"],
        default="stdio",
        help="测试模式: smoke(冒烟)/stdio(子进程)/sse(HTTP)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="SSE 模式主机")
    parser.add_argument("--port", type=int, default=8003, help="SSE 模式端口")
    args = parser.parse_args()

    if args.mode == "smoke":
        ok = smoke_test()
        sys.exit(0 if ok else 1)
    elif args.mode == "stdio":
        asyncio.run(test_stdio())
    elif args.mode == "sse":
        asyncio.run(test_sse(host=args.host, port=args.port))


if __name__ == "__main__":
    main()
