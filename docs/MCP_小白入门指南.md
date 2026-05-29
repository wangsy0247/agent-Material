# MCP（Model Context Protocol）小白入门指南

> **目标读者**：刚接触 MCP、想了解其原理并在 LangChain/LangGraph 中使用它的开发者。
>
> **学习目标**：理解 MCP 是什么、为什么需要它、它的三层架构如何工作，以及如何在实际项目中落地。

---

## 目录

- [MCP（Model Context Protocol）小白入门指南](#mcpmodel-context-protocol小白入门指南)
  - [目录](#目录)
  - [一、基础知识](#一基础知识)
    - [1.1 一句话定义](#11-一句话定义)
    - [1.2 核心架构：客户端-服务器](#12-核心架构客户端-服务器)
    - [1.3 通信方式](#13-通信方式)
    - [1.4 MCP Server 能暴露什么？三大原语](#14-mcp-server-能暴露什么三大原语)
  - [二、为什么要使用 MCP？](#二为什么要使用-mcp)
    - [2.1 没有 MCP 之前的世界](#21-没有-mcp-之前的世界)
    - [2.2 MCP 带来的改变](#22-mcp-带来的改变)
    - [2.3 一句话总结](#23-一句话总结)
  - [三、MCP 三层架构详解](#三mcp-三层架构详解)
    - [3.1 客户端（Client）](#31-客户端client)
    - [3.2 传输层（Transport）](#32-传输层transport)
    - [3.3 服务端（Server）](#33-服务端server)
    - [3.4 三层协作全景图](#34-三层协作全景图)
  - [四、在 LangChain / LangGraph 中使用 MCP](#四在-langchain--langgraph-中使用-mcp)
    - [4.1 安装依赖](#41-安装依赖)
    - [4.2 准备一个 MCP Server](#42-准备一个-mcp-server)
    - [4.3 在 LangChain 中使用（基础版）](#43-在-langchain-中使用基础版)
    - [4.4 在 LangGraph 中使用（推荐生产环境）](#44-在-langgraph-中使用推荐生产环境)
    - [4.5 连接远程 MCP Server（SSE 模式）](#45-连接远程-mcp-serversse-模式)
  - [五、学习路线建议](#五学习路线建议)
  - [六、推荐资源](#六推荐资源)
  - [附录：一句话记忆法](#附录一句话记忆法)

---

## 一、基础知识

### 1.1 一句话定义

**MCP（Model Context Protocol，模型上下文协议）** 是 Anthropic 推出的一个**开放标准协议**，它定义了 AI 应用（客户端）如何连接外部工具和数据源（服务器）的方式。

你可以把它类比成 **"AI 世界的 USB-C 接口"**：

- **以前**：每款耳机要配不同的充电线（每个工具要写不同的对接代码）
- **现在**：一个 Type-C 口，什么设备都能接（MCP 统一了对接方式）

### 1.2 核心架构：客户端-服务器

```
┌─────────────────────────────────────────┐
│              MCP Host                   │  ← AI 应用载体（如 Claude Desktop、你的 Python 程序）
│  ┌──────────────┐  ┌──────────────┐    │
│  │  MCP Client  │  │  MCP Client  │    │  ← 负责连接 Server 的客户端
│  └──────┬───────┘  └──────┬───────┘    │
│         │                  │            │
└─────────┼──────────────────┼────────────┘
          │ stdio / SSE     │ stdio / SSE
┌─────────┴──────────────────┴────────────┐
│  ┌──────────────────────────────────┐   │
│  │         MCP Server 1             │   │  ← 提供"查天气"功能
│  │    (e.g., Weather Service)       │   │
│  └──────────────────────────────────┘   │
│  ┌──────────────────────────────────┐   │
│  │         MCP Server 2             │   │  ← 提供"查数据库"功能
│  │    (e.g., PostgreSQL Server)     │   │
│  └──────────────────────────────────┘   │
│  ┌──────────────────────────────────┐   │
│  │         MCP Server 3             │   │  ← 提供"文件系统操作"功能
│  │    (e.g., Filesystem Server)     │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

**三个关键角色：**

| 角色 | 说明 | 例子 |
|------|------|------|
| **Host** | 运行 AI 的应用程序 | Claude Desktop、IDE、你的 LangChain 程序 |
| **Client** | Host 内负责维护 Server 连接的组件 | `mcp.ClientSession` |
| **Server** | 暴露具体能力的服务 | 天气查询、数据库查询、GitHub API 等 |

### 1.3 通信方式

| 方式 | 说明 | 适用场景 |
|------|------|----------|
| **stdio** | 标准输入输出，本地子进程通信 | 本地工具（文件系统、本地数据库） |
| **SSE** | Server-Sent Events，基于 HTTP 的流式通信 | 远程云服务 |
| **Streamable HTTP** | 基于 HTTP 的流式传输（SSE 的演进版） | 现代远程服务，更灵活高效 |

### 1.4 MCP Server 能暴露什么？三大原语

| 原语 | 英文名 | 通俗解释 | 类比 |
|------|--------|----------|------|
| **工具** | Tools | 让 AI 能"执行动作"（查询、修改、调用 API） | 函数 / Function Call |
| **资源** | Resources | 让 AI 能"读取上下文"（文件内容、数据库记录） | 只读数据源 |
| **提示** | Prompts | 预定义的交互模板，帮助 AI 更好地完成任务 | Prompt 模板 |

> 对初学者来说，**Tools（工具）** 是最常用、最重要的，它基本等同于你熟悉的 Function Calling。

---

## 二、为什么要使用 MCP？

### 2.1 没有 MCP 之前的世界

假设你的 AI 应用需要连接 5 个外部能力：查天气、查股票、操作数据库、读本地文件、发邮件。

**传统方式（没有 MCP）：**

```python
# 方式一：每个工具都要手写对接代码
weather_result = call_weather_api(city)      # 自己封装 HTTP
stock_result = call_stock_api(code)          # 另一套认证方式
db_result = execute_sql(query)               # 自己写数据库连接
file_content = read_local_file(path)         # 本地 IO 代码
email_result = send_email_via_smtp(...)      # SMTP 协议

# 方式二：LangChain 的各种工具包
from langchain_community.tools import WeatherTool, SQLTool...
# 每个工具初始化参数都不一样，学习成本高
```

**痛点：**

- N 个工具，N 种对接方式
- 工具更新了你得跟着改代码
- 换个项目，代码很难复用
- 要把工具给别的 AI 用（比如从 Claude 换到 GPT），又得重写一遍

### 2.2 MCP 带来的改变

| 维度 | 传统方式 | MCP 方式 |
|------|----------|----------|
| **接口标准** | 百花齐放，各写各的 | 统一协议，一套代码对接所有 |
| **工具发现** | 硬编码有哪些工具 | AI 自动发现 Server 有哪些工具 |
| **复用性** | 换个项目/模型要重写 | 一次开发，到处使用 |
| **生态** | 孤岛化 | 社区共享，现成的 Server 越来越多 |
| **安全性** | 代码侵入性强 | Server 独立进程，权限可控 |

### 2.3 一句话总结

> **MCP 让你不再需要为每个工具写定制对接代码，像插 U 盘一样，插上就能让 AI 获得新能力。**

---

## 三、MCP 三层架构详解

如果把 MCP 比作一台机器，它可以拆成三层：**客户端（外交官）、传输层（交通系统）、服务端（工匠）**。

---

### 3.1 客户端（Client）

#### 定位与职责

客户端是 **"住在 Host 里的外交官"**。

```
┌─────────────────────────────────────────┐
│              MCP Host                   │  ← AI 应用（你的程序/Claude Desktop）
│         ┌──────────────┐                │
│         │  MCP Client  │ ←──────────────┤  外交官：负责对外联络
│         └──────┬───────┘                │
│                │ stdio / SSE            │
└────────────────┼────────────────────────┘
                 │
         ┌───────┴───────┐
         │   MCP Server  │
         └───────────────┘
```

**Host** 是你的 AI 应用本体，**Client** 是 Host 派出去跟 Server 打交道的人。一个 Host 可以同时养多个 Client，每个 Client 连接一个不同的 Server。

#### 核心职责

| 职责 | 通俗解释 | 对应代码/概念 |
|------|----------|---------------|
| **建立连接** | 找到 Server 并搭上线 | `stdio_client()` 或 `sse_client()` |
| **初始化会话** | 互相打招呼，确认协议版本 | `session.initialize()` |
| **发现能力** | 问 Server："你有什么工具？" | `session.list_tools()` |
| **调用工具** | 让 Server 执行具体任务 | `session.call_tool()` |
| **读取资源** | 从 Server 获取数据 | `session.read_resource()` |
| **订阅通知** | 告诉 Server："有变化通知我" | `session.subscribe()` |
| **管理生命周期** | 保持连接、处理断开、清理资源 | `async with` 上下文管理 |

#### 代码示例

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 定义怎么找到 Server（类似名片）
server_params = StdioServerParameters(
    command="python",      # 通过什么命令启动
    args=["my_server.py"], # 参数
    env=None,              # 环境变量
)

# 客户端的生命周期
async with stdio_client(server_params) as (read, write):
    # read/write 是两根"管道"，负责收/发数据
    
    async with ClientSession(read, write) as session:
        # session 是"会话"，封装了所有外交礼仪
        
        await session.initialize()  # 握手：确认双方都能正常交流
        
        # 发现 Server 有哪些工具
        tools_result = await session.list_tools()
        print(tools_result.tools)  # [Tool(name='read_file', ...), Tool(name='search', ...)]
        
        # 调用工具
        result = await session.call_tool(
            name="read_file",                  # 工具名
            arguments={"path": "/tmp/a.txt"}   # 参数
        )
```

#### 关键特性

**自动发现（Introspection）**

客户端不需要提前知道 Server 有什么工具，连接后通过 `list_tools()` 动态获取。这意味着：

- 同一个 Client 代码可以连接任意 Server
- Server 更新了工具，Client 自动感知

**异步设计**

所有操作都是 `async/await`，因为一个 Host 可能同时连接多个 Server，同步会卡住。

---

### 3.2 传输层（Transport）

#### 定位与职责

传输层是 **"连接两地的交通系统"**。客户端和服务端说的是同一种语言（MCP 协议），但语言要靠交通系统传递。

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Client    │◄────►│  Transport  │◄────►│   Server    │
│  (外交官)   │      │ (交通系统)   │      │ (能力提供者) │
└─────────────┘      └─────────────┘      └─────────────┘
       │                                         ▲
       │         方式1：stdio（本地地铁）         │
       │         方式2：SSE（远程高铁）            │
       │         方式3：Streamable HTTP（磁悬浮）  │
       └─────────────────────────────────────────┘
```

传输层屏蔽了"怎么传"的细节，让上层只关心"传什么"。

#### 三种交通方式对比

| 维度 | **stdio** | **SSE** | **Streamable HTTP** |
|------|-----------|---------|---------------------|
| **比喻** | 本地地铁 | 远程高铁 | 磁悬浮（更快更灵活） |
| **通信对象** | 本地子进程 | 远程网络服务 | 远程网络服务 |
| **协议基础** | 操作系统管道 | HTTP + SSE 单向流 | HTTP（可流式可非流式） |
| **端点数量** | 无（直接管道） | 2 个（/sse + POST） | 1 个（统一端点） |
| **流式支持** | 天然双向流 | 仅 Server→Client | 双向可选，按需流式 |
| **状态要求** | 有状态长连接 | 有状态长连接 | 支持无状态 |
| **适用场景** | 本地工具 | 传统远程服务 | 现代云原生/Serverless |
| **复杂度** | 低 | 中 | 低（比 SSE 更简单） |

#### stdio 详解：本地进程通信

```python
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="python",
    args=["weather_server.py"],
)

async with stdio_client(server_params) as (read, write):
    # 底层发生了什么？
    # 1. OS 启动 python weather_server.py 子进程
    # 2. Client 的 write → 连接到 Server 进程的 stdin
    # 3. Server 的 stdout → 连接到 Client 的 read
    # 4. 双方通过 JSON-RPC 消息对话
    ...
```

**stdio 的数据流向：**

```
Client 进程                    Server 进程（子进程）
┌──────────┐   JSON-RPC      ┌──────────┐
│  write   │ ──────────────► │  stdin   │
│  (发送)  │                 │ (接收)   │
├──────────┤                 ├──────────┤
│   read   │ ◄────────────── │  stdout  │
│  (接收)  │    JSON-RPC     │  (发送)  │
└──────────┘                 └──────────┘
```

**特点：**

- Client 是 Server 的"爸爸"：由 Client 启动，Server 生命周期受 Client 控制
- Server 的 stderr 通常用于打日志，stdout 专用于 MCP 通信
- 最安全，因为不暴露网络端口

#### SSE 详解：网络流式通信

```python
from mcp.client.sse import sse_client

async with sse_client("http://localhost:3001/sse") as (read, write):
    # 底层发生了什么？
    # 1. Client 发 HTTP GET /sse 建立 SSE 长连接（Server → Client 的通道）
    # 2. Server 通过 SSE 推送消息给 Client
    # 3. Client 通过 HTTP POST 发送消息给 Server（Client → Server 的通道）
    ...
```

**SSE 的数据流向：**

```
Client                           Server (http://localhost:3001)
┌──────────┐                    
│ HTTP GET │ ──► /sse ─────────►┌──────────┐
│ (开通道) │   建立 SSE 流      │          │
└──────────┘                    │          │
┌──────────┐                    │  Server  │
│ read(SSE)│ ◄──────────────── │ 推送消息 │
│(接收数据)│   Server-Sent     │          │
└──────────┘   Events          │          │
┌──────────┐                    │          │
│HTTP POST │ ──► /message ────►│ 接收请求 │
│(发送数据)│                    └──────────┘
└──────────┘
```

**SSE 的痛点：**

- 需要维护两个端点（`/sse` 和 `/message`）
- 必须保持长连接，对 Serverless/无状态架构不友好
- 一旦 SSE 连接断开，会话就断了，需要重连

#### Streamable HTTP 详解：新一代远程传输

Streamable HTTP 是 MCP 规范在 **2025-03-26** 版本引入的传输方式，目的是**取代 SSE，解决 SSE 的痛点**。

##### 核心设计思想

Streamable HTTP 把 SSE 的"双端点+强制长连接"简化为 **"单端点+按需流式"**：

```
Client                                          Server
┌──────────┐                                    ┌──────────┐
│ POST /mcp│ ──► {jsonrpc: "2.0", id: 1, ...} ─►│          │
│          │                                    │          │
│          │ ◄── 200 OK {result: ...}           │          │
│          │     （普通响应，非流式）              │          │
└──────────┘                                    │          │
                                                │          │
┌──────────┐                                    │          │
│ POST /mcp│ ──► {jsonrpc: "2.0", id: 2, ...} ─►│ 统一端点  │
│Accept:   │     + Header: Accept: text/        │  /mcp    │
│event-stre│       event-stream                  │          │
│am         │                                    │          │
│          │ ◄── 200 OK + SSE Stream            │          │
│ read(SSE)│     data: {result: ...}            │          │
│(流式接收)│     data: {result: ...}            │          │
└──────────┘                                    └──────────┘
```

##### 关键机制

**1. 统一端点**

所有 MCP 消息都发到同一个 URL（比如 `/mcp`），不再需要 `/sse` 和 `/message` 两个端点：

```http
POST /mcp HTTP/1.1
Content-Type: application/json
Accept: application/json, text/event-stream

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": {"city": "北京"}
  }
}
```

**2. 按需流式（Content Negotiation）**

Client 通过 `Accept` 头告诉 Server：

| Client 的 Accept 头 | Server 的行为 |
|---------------------|---------------|
| `application/json` | 直接返回完整 JSON（非流式，最简单） |
| `text/event-stream` | 返回 SSE 流（支持逐步推送） |
| 两者都接受 | Server 自行决定 |

**3. 支持无状态（Stateless）**

这是 Streamable HTTP 相比 SSE 最大的进步：

| 特性 | SSE | Streamable HTTP |
|------|-----|-----------------|
| 会话状态 | 必须维护长连接状态 | 可放在 Cookie/Header 中 |
| 负载均衡 | 必须粘滞（Sticky Session） | 可任意分发到不同实例 |
| Serverless | 不友好 | 完美支持 |
| 断线恢复 | 需要重连重建会话 | 携带 Session ID 即可恢复 |

```http
# 无状态模式下，会话信息放在 Header 中
POST /mcp HTTP/1.1
Mcp-Session-Id: abc-123-xyz
Content-Type: application/json

{...}
```

**4. 会话恢复（Session Resumption）**

如果连接断了，Client 可以携带之前的 `Mcp-Session-Id` 恢复会话，不用重新 `initialize`：

```python
# Python 伪代码展示恢复流程
headers = {
    "Mcp-Session-Id": "之前的会话ID",  # 携带旧会话 ID
    "Accept": "application/json"
}
response = requests.post("http://server/mcp", json=request, headers=headers)
# Server 认出会话 ID，直接恢复之前的上下文
```

##### 什么时候用 Streamable HTTP？

| 场景 | 推荐传输方式 |
|------|-------------|
| 本地脚本/文件系统工具 | **stdio** |
| 传统远程服务、简单场景 | **SSE** |
| 云原生/Serverless 架构 | **Streamable HTTP** |
| 高并发、需要负载均衡 | **Streamable HTTP** |
| 微服务架构、无状态服务 | **Streamable HTTP** |

#### 传输层之上的消息格式：JSON-RPC 2.0

无论 stdio、SSE 还是 Streamable HTTP，传的都是 **JSON-RPC 2.0** 消息：

```json
// Client 请求（调用工具）
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "read_file",
    "arguments": {"path": "/tmp/test.txt"}
  }
}

// Server 响应（返回结果）
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [{"type": "text", "text": "文件内容..."}]
  }
}

// Server 通知（无需回复）
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {"uri": "file:///tmp/test.txt"}
}
```

**三种消息类型：**

- **Request（请求）**：有 `id`，必须收到 Response
- **Response（响应）**：对应某个 `id` 的结果
- **Notification（通知）**：没 `id`，发了就发了，不用回

#### 安全认证：OAuth 2.1 机制

当你把 MCP Server 部署到公网时，**任何人都能连上来调用你的工具**，这显然很危险。MCP 规范定义了基于 **OAuth 2.1** 的安全认证体系。

##### 为什么传输层需要 OAuth？

想象你的 MCP Server 暴露了"发送邮件"、"操作数据库"这样的敏感工具：

```
没认证：                    有 OAuth 认证：
任何人 ──► MCP Server        合法用户 ──► [OAuth 检查] ──► MCP Server
都能调用！                   非法用户 ──► ❌ 拒绝访问
```

##### MCP OAuth 的核心角色

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│ MCP Client  │────►│  MCP Server      │     │   User      │
│  (你的应用)  │     │  (能力提供者)     │◄────│  (最终用户)  │
└──────┬──────┘     └────────┬─────────┘     └──────┬──────┘
       │                     │                       │
       │                     ▼                       │
       │            ┌──────────────────┐            │
       │            │ Authorization    │            │
       └───────────►│ Server           │◄───────────┘
          拿 Token   │ (身份认证中心)    │   登录授权
                    └──────────────────┘
```

| 角色 | 说明 | 类比 |
|------|------|------|
| **MCP Client** | 请求使用工具的应用 | 你的 AI 程序 |
| **MCP Server** | 提供工具的服务 | 邮件服务、数据库服务 |
| **Authorization Server** | 发 Token 的认证中心 | 公司的 SSO 系统、Auth0、Keycloak |
| **User** | 最终授权的人 | 你自己 |

##### 标准 OAuth 流程（Authorization Code + PKCE）

这是 MCP 推荐的标准流程，适合有用户界面的场景：

```
Step 1: Client 发现认证端点
─────────────────────────────────────────────────────────
Client ──GET /.well-known/oauth-authorization-server──► Server
Server ◄────────────返回认证配置（授权URL、Token URL）────── Client


Step 2: 生成 PKCE 参数（防攻击）
─────────────────────────────────────────────────────────
Client 生成：
  code_verifier  = "随机字符串"        （本地保存）
  code_challenge = SHA256(code_verifier)  （发送给 Server）


Step 3: 引导用户授权（浏览器跳转）
─────────────────────────────────────────────────────────
Client 打开浏览器，访问：
https://auth-server.com/authorize?
  response_type=code&
  client_id=your-app-id&
  redirect_uri=http://localhost:8080/callback&
  code_challenge=xxx&
  scope=mcp:tool:read mcp:resource:read
                ▲
                └── MCP 定义的权限范围

User 在浏览器中登录并点击"同意授权"


Step 4: 拿到授权码
─────────────────────────────────────────────────────────
Auth Server 重定向到：
http://localhost:8080/callback?code=授权码

Client 从 URL 中提取 code


Step 5: 换取 Access Token
─────────────────────────────────────────────────────────
Client ──POST /token──► Auth Server
Body: {
  grant_type: "authorization_code",
  code: "授权码",
  code_verifier: "之前生成的随机字符串",  ← PKCE 校验
  client_id: "your-app-id"
}

Auth Server ◄────返回──── Client
{
  access_token: "eyJhbGciOiJSUzI1NiIs...",
  token_type: "Bearer",
  expires_in: 3600,
  refresh_token: "用来续期的Token"
}


Step 6: 带 Token 调用 MCP
─────────────────────────────────────────────────────────
POST /mcp HTTP/1.1
Authorization: Bearer eyJhbGciOiJSUzI1NiIs...
Content-Type: application/json

{"jsonrpc": "2.0", "method": "tools/call", ...}
```

##### MCP 定义的权限范围（Scopes）

OAuth 的 `scope` 参数控制 Client 能做什么，MCP 规范推荐的标准 scope：

| Scope | 含义 | 示例 |
|-------|------|------|
| `mcp:tool:read` | 查看工具列表 | `list_tools` |
| `mcp:tool:call` | 调用工具 | `call_tool` |
| `mcp:resource:read` | 读取资源 | `read_resource` |
| `mcp:resource:subscribe` | 订阅资源变更 | `subscribe` |
| `mcp:prompt:read` | 读取提示模板 | `list_prompts`, `get_prompt` |
| `mcp:sampling` | 让 Server 请求 AI 生成内容 | 高级功能 |

Server 可以根据 scope 决定返回哪些工具——**没权限的工具对 Client 完全不可见**。

##### 简化流程：Client Credentials（机器对机器）

如果你的 MCP Client 不是代表用户，而是后台服务本身：

```
Client ──POST /token──► Auth Server
Body: {
  grant_type: "client_credentials",
  client_id: "服务ID",
  client_secret: "服务密钥"
}

Auth Server ◄────返回 Access Token──── Client

Client 之后每次请求都带上这个 Token
```

适合场景：
- 企业内部服务互调
- 后台定时任务
- 不需要用户介入的自动化流程

##### 在代码中集成 OAuth

```python
import httpx
from mcp.client.sse import sse_client

# Step 1: 获取 Access Token（Client Credentials 方式）
async def get_access_token():
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://auth.example.com/token",
            data={
                "grant_type": "client_credentials",
                "client_id": "my-mcp-client",
                "client_secret": "super-secret-key",
                "scope": "mcp:tool:call mcp:resource:read"
            }
        )
        return resp.json()["access_token"]

# Step 2: 连接 MCP Server 时带上 Token
token = await get_access_token()

headers = {
    "Authorization": f"Bearer {token}"
}

async with sse_client("https://mcp.example.com/sse", headers=headers) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        ...
```

##### OAuth vs 其他认证方式对比

| 认证方式 | 适用场景 | 安全性 | 复杂度 |
|----------|----------|--------|--------|
| **无认证** | 本地 stdio、内网信任环境 | ⚠️ 低 | 简单 |
| **API Key** | 简单远程服务 | 🟡 中 | 简单 |
| **OAuth 2.1** | 生产环境、多用户、第三方接入 | 🟢 高 | 中等 |
| **mTLS** | 企业内网、服务网格 | 🟢 高 | 复杂 |

**建议：**

- 本地开发/stdio 模式：**不需要认证**（进程隔离足够安全）
- 内网远程服务：**API Key 或 mTLS**
- 公网生产服务：**OAuth 2.1**（标准、安全、可控）

---

### 3.3 服务端（Server）

#### 定位与职责

服务端是 **"实际干活的工匠"**。它手里有真本事（查天气、读文件、查数据库），通过 MCP 协议把这些本事标准化地"挂出去"，等 Client 来调用。

```
┌─────────────────────────────────────────────────────┐
│                    MCP Server                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Tools     │  │  Resources  │  │   Prompts   │ │
│  │  (工具箱)   │  │  (资料库)   │  │  (模板库)   │ │
│  │             │  │             │  │             │ │
│  │ • 查天气    │  │ • 文件内容  │  │ • 代码审查  │ │
│  │ • 读文件    │  │ • 数据库表  │  │ • 文本分析  │ │
│  │ • 搜网页    │  │ • API 数据  │  │ • 翻译模板  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
│                                                      │
│  ┌───────────────────────────────────────────────┐  │
│  │              Request Handler                   │  │
│  │         (接收请求 → 分发到对应能力)             │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                        ▲
                        │ stdio / SSE / Streamable HTTP
                        │
                ┌───────┴───────┐
                │  MCP Client   │
                └───────────────┘
```

#### 三大原语的具体实现

##### Tools（工具）— 最能干的

工具是**可执行的函数**，让 AI 能够"动手做事"。

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WeatherServer")

@mcp.tool()
async def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息。
    
    Args:
        city: 城市名称，如"北京"、"上海"
    """
    # 这里写真实逻辑
    return f"{city}今天晴天，25°C"

# MCP 会自动生成工具的 JSON Schema：
# {
#   "name": "get_weather",
#   "description": "获取指定城市的天气信息",
#   "inputSchema": {
#     "type": "object",
#     "properties": {
#       "city": {"type": "string", "description": "城市名称..."}
#     },
#     "required": ["city"]
#   }
# }
```

**工具的特点：**

- 必须有函数名 + 参数描述（AI 靠这个决定要不要调用）
- 通过 `inputSchema` 自动暴露给 Client
- 可以是同步或异步函数
- 返回标准格式（text/image/resource）

##### Resources（资源）— 只读的资料库

资源是**可读的数据源**，让 AI 能够"获取上下文"。

```python
@mcp.resource("weather://{city}/current")
async def get_current_weather(city: str) -> str:
    """
    天气数据资源，通过 URI 访问。
    URI 格式: weather://北京/current
    """
    return "晴天，25°C，湿度 60%"

@mcp.resource("config://app")
async def get_app_config() -> str:
    """应用配置资源"""
    return '{"theme": "dark", "lang": "zh"}'
```

**资源的特点：**

- 用 **URI** 标识（类似网址），如 `file:///home/user/doc.txt`
- 只读（如果要修改，应该做成 Tool）
- 支持参数化 URI（`weather://{city}`）
- Client 可以订阅资源变更（Server 主动推送更新）

##### Prompts（提示词模板）— 预设的交互套路

Prompts 是**预定义的提示词模板**，帮助 AI 更好地完成特定任务。

```python
@mcp.prompt()
def review_code(code: str, language: str = "python") -> str:
    """
    代码审查提示词模板。
    
    Args:
        code: 要审查的代码
        language: 编程语言
    """
    return f"""请作为资深 {language} 工程师，审查以下代码：

```{language}
{code}
```

请关注：
1. 潜在 Bug
2. 性能问题
3. 代码风格
4. 安全漏洞

请给出具体修改建议。"""

# 调用方式
# Client 可以请求这个 Prompt，然后把返回的字符串塞给 AI
```

**Prompts 的特点：**

- 不是直接给 AI 看的，而是 Client 请求后，由 Host 决定怎么用
- 可以带参数，动态生成
- 适合团队共享标准化的 Prompt 模板

#### 服务端的生命周期

```
┌─────────────┐
│   启动      │ ◄── 运行 npx/python 命令，Server 进程启动
└──────┬──────┘
       ▼
┌─────────────┐
│   初始化    │ ◄── Server 注册自己有哪些 Tools/Resources/Prompts
└──────┬──────┘
       ▼
┌─────────────┐
│  等待连接   │ ◄── stdio: 等 Client 连接 stdin/stdout
│             │     SSE:  启动 HTTP 服务，等 Client 连 /sse
│             │     Streamable HTTP: 等 POST 请求
└──────┬──────┘
       ▼
┌─────────────┐
│  能力协商   │ ◄── Client 发 initialize，双方交换协议版本和能力列表
└──────┬──────┘
       ▼
┌─────────────┐
│  处理请求   │ ◄── 循环：接收 Request → 执行 → 返回 Response
│  (工作期)   │
└──────┬──────┘
       ▼
┌─────────────┐
│  断开/终止  │ ◄── Client 断开、进程退出、错误发生
└─────────────┘
```

#### 完整的服务端示例

```python
# server.py
from mcp.server.fastmcp import FastMCP

# 1. 创建 Server 实例（取个名字）
mcp = FastMCP("MyDataServer", dependencies=["requests"])

# 2. 注册工具
@mcp.tool()
async def search_user(user_id: str) -> dict:
    """根据用户 ID 查询用户信息"""
    # 模拟数据库查询
    users = {
        "001": {"name": "张三", "age": 25},
        "002": {"name": "李四", "age": 30},
    }
    return users.get(user_id, {"error": "用户不存在"})

# 3. 注册资源
@mcp.resource("users://list")
async def list_users() -> str:
    """用户列表资源"""
    return "001-张三, 002-李四"

# 4. 注册 Prompt
@mcp.prompt()
def analyze_user_data(data: str) -> str:
    """分析用户数据的提示词模板"""
    return f"请分析以下用户数据并给出洞察：\n\n{data}"

# 5. 启动（stdio 模式）
if __name__ == "__main__":
    mcp.run(transport='stdio')
    
# 也可以：
# mcp.run(transport='sse', port=3001)           # SSE 模式
# mcp.run(transport='streamable-http', port=3001)  # Streamable HTTP 模式
```

**运行方式：**

```bash
# stdio 模式（由 Client 启动）
python server.py

# SSE 模式（独立运行，Client 通过网络连接）
python server.py --transport sse --port 3001

# Streamable HTTP 模式（推荐用于云原生部署）
python server.py --transport streamable-http --port 3001
```

---

### 3.4 三层协作全景图

```
┌──────────────────────────────────────────────────────────────┐
│                         MCP Host                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                    MCP Client                           │  │
│  │  1. 启动 Server: stdio_client(server_params)           │  │
│  │  2. 建立会话: ClientSession(read, write)               │  │
│  │  3. 握手: session.initialize()                         │  │
│  │  4. 发现工具: session.list_tools() ◄────────────────┐  │  │
│  │  5. 调用工具: session.call_tool(name, args) ────────┼──┼──┐
│  └───────────────────────────────────────────────────────┼──┘  │
└────────────────────────────────────────────────────────┼──────┘
                    │ Transport Layer                      │
                    │ (JSON-RPC over stdio/SSE/HTTP)       │
                    ▼                                      │
┌────────────────────────────────────────────────────────┼──────┐
│                      MCP Server                         │      │
│  ┌──────────────────────────────────────────────────┐  │      │
│  │              Request Handler                      │  │      │
│  │  接收 initialize ──► 返回能力列表 ───────────────┼──┘      │
│  │  接收 list_tools ──► 返回工具清单 ───────────────┼─────────┘
│  │  接收 call_tool  ──► 执行 search_user()          │
│  │                      返回结果                      │
│  └──────────────────────────────────────────────────┘
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  │   Tools     │  │  Resources  │  │   Prompts   │
│  │ search_user │  │ users://    │  │ analyze_... │
│  └─────────────┘  └─────────────┘  └─────────────┘
└──────────────────────────────────────────────────────────────┘
```

**调用链总结：**

1. **Client** 决定连接哪个 Server，通过 **Transport** 建立通道
2. 双方通过 **JSON-RPC** 握手，确认"语言互通"
3. Client 问 Server："你有什么本事？"（list_tools）
4. Server 回答："我会查用户、读列表..."（返回 Tool 定义）
5. 当 AI 需要查用户时，Client 发请求："用 search_user 查 001"
6. **Transport** 把请求送到 Server
7. **Server** 执行真实逻辑，把结果经 **Transport** 送回 **Client**
8. Client 把结果交给 AI，AI 继续下一步思考

---

## 四、在 LangChain / LangGraph 中使用 MCP

MCP 是协议，LangChain 官方提供了适配器 `langchain-mcp-adapters` 来桥接。

### 4.1 安装依赖

```bash
pip install langchain-mcp-adapters langchain langchain-openai
```

### 4.2 准备一个 MCP Server

这里以官方的文件系统 Server 为例（只读访问本地文件）：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/home/user/workspace"
      ]
    }
  }
}
```

> 你也可以自己用 Python 写 Server（见第三节服务端部分）。

### 4.3 在 LangChain 中使用（基础版）

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor

# 1. 定义 Server 启动参数
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/home/user/workspace"],
    env=None,
)

async def main():
    # 2. 建立 stdio 连接
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化
            await session.initialize()
            
            # 3. 动态加载 MCP Server 提供的所有工具
            tools = await load_mcp_tools(session)
            print(f"发现工具: {[tool.name for tool in tools]}")
            # 输出类似: ['read_file', 'list_directory', ...]
            
            # 4. 绑定到 LangChain 模型
            model = ChatOpenAI(model="gpt-4o")
            
            # 5. 创建 Agent
            agent = create_react_agent(model, tools)
            agent_executor = AgentExecutor(agent=agent, tools=tools)
            
            # 6. 运行
            result = await agent_executor.ainvoke({
                "input": "请列出 /home/user/workspace 目录下的所有文件，并总结内容"
            })
            print(result["output"])

if __name__ == "__main__":
    asyncio.run(main())
```

**核心逻辑拆解：**

1. `stdio_client` 启动并连接 MCP Server 进程
2. `ClientSession` 建立会话
3. `load_mcp_tools(session)` ⭐ **自动发现 Server 暴露的所有工具**，不用你手动写每个工具
4. 剩下的就是标准 LangChain Agent 流程

### 4.4 在 LangGraph 中使用（推荐生产环境）

LangGraph 比传统 Agent 更适合复杂任务流：

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent  # LangGraph 版本

server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/home/user/workspace"],
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 加载 MCP 工具
            tools = await load_mcp_tools(session)
            
            # LangGraph 的 create_react_agent 返回的是一个 StateGraph
            model = ChatOpenAI(model="gpt-4o")
            graph = create_react_agent(model, tools)
            
            # 运行（LangGraph 的 invoke 接口）
            result = await graph.ainvoke({
                "messages": [
                    {"role": "user", "content": "读取 README.md 并提取关键配置信息"}
                ]
            })
            
            # 输出消息历史
            for msg in result["messages"]:
                print(f"{msg.type}: {msg.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 4.5 连接远程 MCP Server（SSE 模式）

如果你的 Server 跑在远程（比如 `http://localhost:3001/sse`）：

```python
from mcp.client.sse import sse_client

async def main():
    async with sse_client("http://localhost:3001/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            # ... 后续一样
```

### 4.6 连接带 OAuth 认证的远程 Server

```python
import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client

# 第一步：获取 Access Token
async def get_access_token():
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://auth.example.com/token",
            data={
                "grant_type": "client_credentials",
                "client_id": "my-langchain-app",
                "client_secret": "xxx",
                "scope": "mcp:tool:call mcp:resource:read"
            }
        )
        return resp.json()["access_token"]

# 第二步：带 Token 连接 MCP
async def main():
    token = await get_access_token()
    
    headers = {"Authorization": f"Bearer {token}"}
    
    async with sse_client("https://mcp.example.com/sse", headers=headers) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            
            model = ChatOpenAI(model="gpt-4o")
            graph = create_react_agent(model, tools)
            
            result = await graph.ainvoke({
                "messages": [{"role": "user", "content": "查询用户 001 的信息"}]
            })
            print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 五、学习路线建议

| 阶段 | 目标 | 行动 |
|------|------|------|
| **第 1 步** | 建立直观认知 | 安装 Claude Desktop，配置几个官方 MCP Server（文件系统、GitHub），亲身体验 AI 调用工具 |
| **第 2 步** | 理解协议本身 | 读 [MCP 官方文档](https://modelcontextprotocol.io/introduction)，重点看 Architecture 和 Tools 部分 |
| **第 3 步** | 动手写 Python Client | 用 `langchain-mcp-adapters` 跑通上面的示例代码 |
| **第 4 步** | 尝试写自己的 Server | 用 Python SDK（`mcp.server.fastmcp`）写一个简单 Server（比如查询本地 SQLite 数据库） |
| **第 5 步** | 掌握传输层差异 | 对比 stdio / SSE / Streamable HTTP 三种方式，理解各自的适用场景 |
| **第 6 步** | 学习安全认证 | 为你的远程 Server 添加 OAuth 认证，理解 scope 和 token 机制 |
| **第 7 步** | 整合到项目 | 在 LangGraph 工作流中引入 MCP 工具，处理真实业务场景 |

---

## 六、推荐资源

| 资源 | 链接 | 说明 |
|------|------|------|
| **官方文档** | https://modelcontextprotocol.io/introduction | 最权威的学习资料 |
| **官方 Python SDK** | https://github.com/modelcontextprotocol/python-sdk | 写 Server/Client 的必备库 |
| **LangChain 适配器** | https://github.com/langchain-ai/langchain-mcp-adapters | 在 LangChain 生态中使用 MCP |
| **官方 Server 集市** | https://github.com/modelcontextprotocol/servers | 官方维护的现成 Server，开箱即用 |
| **MCP 规范（含 OAuth）** | https://spec.modelcontextprotocol.io/ | 协议规范原文，含安全认证章节 |

---

## 附录：一句话记忆法

| 层次 | 比喻 | 核心记忆点 |
|------|------|------------|
| **客户端** | 外交官 | "代表 Host 出去谈生意，发现能力、发起调用" |
| **传输层** | 交通系统 | "stdio 是本地地铁，SSE 是远程高铁，Streamable HTTP 是磁悬浮；OAuth 是安检闸机" |
| **服务端** | 工匠 | "手里有真工具(Tools)、真资料(Resources)、真套路(Prompts)" |

---

> **最后用一个比喻帮你记住整体：**
>
> 如果把 AI 比作一个人的大脑，MCP 就是大脑的**神经系统标准接口**。以前大脑想控制手脚，每种手脚都要接不同的线；现在有了 MCP（统一神经接口），只要是符合标准的手脚（Server），插上就能用。而 LangChain/LangGraph 就是帮你更快地把这套神经系统组装成完整机器人的工具箱。
