• FastAPI + SSE 流式服务详解
  ══════════════════════════

  一、什么是 SSE？
  ────────────────

  SSE (Server-Sent Events) 是一种服务器向客户端单向推送实时数据的技术。基于 HTTP 协议，客户端发起一次请求，服务
  器保持连接打开，持续发送数据流。

  ┌──────────────┐                    ┌──────────────┐
  │   客户端      │ ─── GET /stream ──▶│   服务器      │
  │  (浏览器/APP) │◀──── data: 你好    │  (FastAPI)   │
  │              │◀──── data: 我是    │              │
  │              │◀──── data: Kimi   │              │
  │              │◀──── data: [DONE] │              │
  └──────────────┘                    └──────────────┘
        ↑ 单连接、服务器单向推送、自动重连

  SSE 数据格式
  ────────────
  Content-Type: text/event-stream

  data: {"chunk": "你好"}

  data: {"chunk": "我是"}

  data: {"chunk": "Kimi"}

  data: [DONE]

  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

  二、为什么大模型应用必须用 SSE？
  ────────────────────────────────

  大模型生成文本是逐字/逐句的，如果等全部生成完再返回，用户体验极差：

  ┌───────────┬──────────────────────────────────┬────────────────────┐
  │ 方式      │ 延迟体验                         │ 适用场景           │
  ├───────────┼──────────────────────────────────┼────────────────────┤
  │ 普通 HTTP │ 等待 10~30 秒才看到结果          │ 简单查询           │
  │ SSE 流式  │ 几百毫秒内看到第一个字，实时输出 │ 大模型对话 ⭐      │
  │ WebSocket │ 实时双向，但实现复杂             │ 多人协作、实时编辑 │
  └───────────┴──────────────────────────────────┴────────────────────┘

  你的项目现状：Agent 已经支持 stream_mode="values"，但没有服务层封装，前端无法消费这个流。

  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

  三、FastAPI 实现 SSE 的核心代码
  ───────────────────────────────

  3.1 基础 SSE 响应
  ─────────────────

  from fastapi import FastAPI
  from fastapi.responses import StreamingResponse
  import asyncio

  app = FastAPI()

  async def fake_llm_stream():
      """模拟大模型流式生成"""
      words = ["你好", "，", "我", "是", "Kimi", "。", "有", "什么", "可以", "帮", "你", "？"]
      for word in words:
          await asyncio.sleep(0.2)  # 模拟生成延迟
          yield f"data: {word}\n\n"  # SSE 格式：必须以 \n\n 结尾
      yield "data: [DONE]\n\n"       # 结束标记

  @app.get("/stream")
  async def stream():
      return StreamingResponse(
          fake_llm_stream(),
          media_type="text/event-stream",  # 关键：声明 SSE
      )

  3.2 结合你的项目：RAG + Agent SSE 服务
  ──────────────────────────────────────

  这是基于你现有代码的完整实现：

  # api_server.py
  from fastapi import FastAPI, HTTPException
  from fastapi.responses import StreamingResponse
  from fastapi.middleware.cors import CORSMiddleware
  from pydantic import BaseModel
  import asyncio
  import json
  import sys
  from pathlib import Path

  # 添加项目根目录到路径
  sys.path.append(str(Path(__file__).parent))

  from agent.React_agent import agent_executor  # 你的 Agent
  from rag.rag_service import RAGService        # 你的 RAG 服务

  app = FastAPI(title="材料科学 LLM Agent API")

  # 跨域配置（前端调用需要）
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )

  # ============ 数据模型 ============
  class ChatRequest(BaseModel):
      question: str
      session_id: str = "default"  # 用于区分不同对话
      stream: bool = True          # 是否流式返回

  class RAGRequest(BaseModel):
      question: str
      top_k: int = 5

  # ============ SSE 工具函数 ============
  def format_sse(data: dict) -> str:
      """将数据格式化为 SSE 格式"""
      return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

  # ============ Agent 流式接口 ============
  @app.post("/agent/chat")
  async def agent_chat(request: ChatRequest):
      """
      Agent 对话接口，支持 SSE 流式输出

      返回事件类型：
      - type: "thought"    → Agent 思考过程
      - type: "tool_call"  → 调用工具（如查询Materials Project）
      - type: "tool_result"→ 工具返回结果
      - type: "answer"     → 最终答案片段
      - type: "done"       → 流结束
      """
      if not request.stream:
          # 非流式：直接返回完整结果
          result = await agent_executor.ainvoke({"input": request.question})
          return {"answer": result["output"]}

      async def event_stream():
          # 流式执行 Agent
          async for event in agent_executor.astream_events(
              {"input": request.question},
              version="v2",
          ):
              kind = event["event"]

              if kind == "on_chain_start":
                  yield format_sse({
                      "type": "status",
                      "content": "开始思考..."
                  })

              elif kind == "on_chat_model_stream":
                  # 大模型实时输出的 token
                  token = event["data"]["chunk"].content
                  if token:
                      yield format_sse({
                          "type": "answer",
                          "content": token
                      })

              elif kind == "on_tool_start":
                  # 开始调用工具
                  yield format_sse({
                      "type": "tool_call",
                      "tool": event["name"],
                      "input": event["data"].get("input", {})
                  })

              elif kind == "on_tool_end":
                  # 工具调用结束
                  yield format_sse({
                      "type": "tool_result",
                      "tool": event["name"],
                      "output": str(event["data"].get("output", ""))[:500]  # 截断避免过大
                  })

              elif kind == "on_chain_end":
                  yield format_sse({
                      "type": "status",
                      "content": "思考完成"
                  })

          # 流结束标记
          yield format_sse({"type": "done"})

      return StreamingResponse(
          event_stream(),
          media_type="text/event-stream",
      )

  # ============ RAG 流式接口 ============
  @app.post("/rag/query")
  async def rag_query(request: RAGRequest):
      """
      RAG 知识库查询，支持 SSE 流式返回检索过程和生成结果
      """
      rag_service = RAGService()  # 你的 RAG 服务实例

      async def event_stream():
          # 1. 发送检索状态
          yield format_sse({
              "type": "status",
              "content": "正在检索相关文献..."
          })

          # 2. 执行检索（你的混合检索管线）
          docs = await rag_service.aretrieve(request.question, top_k=request.top_k)

          # 3. 发送检索到的文献
          sources = []
          for i, doc in enumerate(docs):
              sources.append({
                  "index": i + 1,
                  "source": doc.metadata.get("source", "未知"),
                  "score": doc.metadata.get("score", 0),
                  "snippet": doc.page_content[:200] + "..."
              })

          yield format_sse({
              "type": "sources",
              "content": sources
          })

          # 4. 流式生成答案
          yield format_sse({
              "type": "status",
              "content": "正在生成答案..."
          })

          async for token in rag_service.astream_answer(request.question, docs):
              yield format_sse({
                  "type": "answer",
                  "content": token
              })

          yield format_sse({"type": "done"})

      return StreamingResponse(
          event_stream(),
          media_type="text/event-stream",
      )

  # ============ 健康检查 ============
  @app.get("/health")
  async def health():
      return {"status": "ok", "service": "material-agent"}

  if __name__ == "__main__":
      import uvicorn
      uvicorn.run(app, host="0.0.0.0", port=8000)

  3.3 前端消费 SSE（JavaScript）
  ──────────────────────────────

  // 调用 Agent 流式接口
  async function chatWithAgent(question) {
      const response = await fetch('http://localhost:8000/agent/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, stream: true })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          // chunk 格式: "data: {"type":"answer","content":"你好"}\n\n"

          const lines = chunk.split('\n\n');
          for (const line of lines) {
              if (line.startsWith('data: ')) {
                  const data = JSON.parse(line.slice(6));

                  switch(data.type) {
                      case 'thought':
                          showThinking(data.content);      // 显示思考过程
                          break;
                      case 'tool_call':
                          showToolCall(data.tool);          // 显示"正在查询Materials Project..."
                          break;
                      case 'answer':
                          appendAnswer(data.content);       // 逐字显示答案
                          break;
                      case 'done':
                          hideLoading();                    // 结束
                          break;
                  }
              }
          }
      }
  }

  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

  四、SSE vs WebSocket 对比
  ─────────────────────────

  ┌────────────┬───────────────────────────┬──────────────────────────┐
  │ 特性       │ SSE                       │ WebSocket                │
  ├────────────┼───────────────────────────┼──────────────────────────┤
  │ 协议       │ HTTP/1.1 或 HTTP/2        │ TCP 全双工               │
  │ 方向       │ 服务器 → 客户端 单向      │ 双向实时通信             │
  │ 复杂度     │ 简单（纯 HTTP）           │ 较复杂（需要握手、心跳） │
  │ 自动重连   │ ✅ 浏览器原生支持         │ ❌ 需手动实现            │
  │ 二进制数据 │ ❌ 只支持文本             │ ✅ 支持                  │
  │ 大模型场景 │ ✅ 首选（单向推送 token） │ 多用户协作时考虑         │
  │ 浏览器兼容 │ 现代浏览器均支持          │ 现代浏览器均支持         │
  └────────────┴───────────────────────────┴──────────────────────────┘

  结论：大模型流式输出是服务器单向推送文本，SSE 完全够用且更简单。

  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

  五、部署相关配置
  ────────────────

  5.1 Dockerfile
  ──────────────

  FROM python:3.11-slim

  WORKDIR /app

  # 安装依赖
  COPY pyproject.toml .
  RUN pip install fastapi uvicorn sse-starlette -i https://pypi.tuna.tsinghua.edu.cn/simple

  # 复制代码
  COPY . .

  EXPOSE 8000

  CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]

  5.2 Nginx 反向代理（SSE 关键配置）
  ──────────────────────────────────

  location / {
      proxy_pass http://localhost:8000;

      # SSE 关键：关闭缓冲，立即转发
      proxy_buffering off;
      proxy_cache off;

      # 保持长连接
      proxy_read_timeout 3600s;
      proxy_send_timeout 3600s;

      # 支持 EventSource
      proxy_set_header Connection '';
      proxy_http_version 1.1;
      chunked_transfer_encoding off;
  }

  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

  六、进阶：完整事件流设计
  ────────────────────────

  生产环境中，建议定义标准化的事件类型：

  class SSEEvent:
      """SSE 事件类型规范"""

      # 状态事件
      START = "start"           # 开始处理
      THINKING = "thinking"     # Agent 思考中
      RETRIEVING = "retrieving" # RAG 检索中

      # 内容事件
      TOKEN = "token"           # 大模型 token
      TOOL_CALL = "tool_call"   # 工具调用
      TOOL_RESULT = "tool_result"  # 工具结果
      SOURCE = "source"         # 引用来源

      # 结束事件
      DONE = "done"             # 正常完成
      ERROR = "error"           # 发生错误
      CANCELLED = "cancelled"   # 用户取消

  示例事件流：

  data: {"type": "start", "timestamp": 1715731200}

  data: {"type": "thinking", "content": "用户问的是石墨烯的能带结构，我需要..."}

  data: {"type": "tool_call", "tool": "mp_query_tool", "input": {"material": "graphene"}}

  data: {"type": "tool_result", "tool": "mp_query_tool", "output": "Found 3 structures..."}

  data: {"type": "token", "content": "石墨", "index": 0}
  data: {"type": "token", "content": "烯", "index": 1}
  data: {"type": "token", "content": "是", "index": 2}

  data: {"type": "source", "content": {"title": "...", "page": 3, "url": "..."}}

  data: {"type": "done"}

  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

  七、总结
  ────────

  ┌────────────────┬────────────────────────────────────────────────────┐
  │ 要点           │ 说明                                               │
  ├────────────────┼────────────────────────────────────────────────────┤
  │ SSE 是什么     │ HTTP 单向服务器推送，文本流                        │
  │ 为什么用大模型 │ 逐 token 输出，提升用户体验                        │
  │ FastAPI 怎么写 │ StreamingResponse + media_type="text/event-stream" │
  │ 你的项目怎么用 │ 把 agent_executor.astream_events() 封装成 SSE      │
  │ 部署注意       │ Nginx 关 proxy_buffering，Docker 暴露 8000 端口    │
  └────────────────┴────────────────────────────────────────────────────┘