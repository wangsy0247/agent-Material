调用mcp工具锁死问题

  第一层：先说现象
  
  ▎ "我们有个 Agent 系统，调用 MCP 工具时会卡死。有意思的是——不是每次都卡，单独调用 MCP 工具是好的，但一旦 Agent 同时调了 RAG 检索，就挂了。"

  停顿，看面试官反应。通常他会问："为什么？"

  ---
  第二层：排查过程
  
  ▎ "我排查了两轮。第一轮怀疑是 MCP 管道并发写入冲突，给 stdio 加了锁——没用。
  ▎
  ▎ 第二轮我仔细看了日志的时序，发现挂掉的时候，后台正好在加载 reranker 模型。这是个同步阻塞操作，持续 3 秒。但这和执行 MCP 调用有什么关系？"

  这里展示你没有停留在表面。

  ---
  第三层：关键发现
  
  ▎ "我用 py-spy 看了一下线程栈，发现整个进程只有一个线程在被使用——但 asyncio.run() 创建了一个嵌套的事件循环。
  ▎
  ▎ MCP 的响应是通过 stdout reader 这个后台 Task 来接收的。这个 Task 注册在外层事件循环里。但 asyncio.run() 把线程完全交给了内层循环。外层循环里的 Task——包括
  ▎  reader——拿不到任何 CPU 时间。
  ▎
  ▎ 相当于：快递员已经把包裹送到了楼下，但门卫睡着了，没人去开门。"

  ---
  第四层：根因总结
  
  ▎ "根因是三层叠加：
  ▎
  ▎ 1. asyncio.run() 嵌套 — sync 函数里跑 async，创建了隔离的事件循环
  ▎ 2. 长时间同步阻塞 — reranker 加载 3 秒不释放 CPU
  ▎ 3. Agent 并行工具调用 — 让上面两个问题同时触发
  ▎
  ▎ 单独任何一个都不会挂，三个凑齐了就是必现死锁。"

  ---
  第五层：修复与反思
  
  ▎ "修复方向有两个。工程上最直接的：把所有 asyncio.run() 去掉，函数全部改成 async def，保证全程在同一条事件循环里。
  ▎
  ▎ 架构上的改进：用 LangGraph 的 StateGraph 替代 create_agent。这不是为了修 bug——是因为显式状态机让我能看到每一步的状态转换，并行工具调用变成了串行 
  ▎ ToolNode，排查这类问题的时候不需要猜框架的黑盒行为。"

  ---
  面试官可能会追问
  
  Q: 为什么不一开始就用 async？

  ▎ "历史原因——FastAPI 的 StreamingResponse 要求返回一个 sync generator，所以 excutor_stream 当时写成了 sync。改 LangGraph 的时候一起重构掉了。"

  Q: MCP stdio 管道和事件循环有什么关系？

  ▎ "MCP stdio 协议底层是子进程的 stdin/stdout 管道。Python asyncio 通过 create_subprocess_exec 把管道注册成 file descriptor，由事件循环的 selector 
  ▎ 来监听可读事件。reader task 本质上是 await stream.readline()——没有事件循环调度它，它就只是一行不会执行的代码。"