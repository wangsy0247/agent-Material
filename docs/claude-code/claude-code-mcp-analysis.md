# Claude-Code MCP 实现深度分析报告

> 分析对象：`Claude-Code-main` 源码中的 MCP（Model Context Protocol）模块  
> 分析日期：2026-05-20  
> 报告位置：`knowledge/claude-code-mcp-analysis.md`

---

## 1. 总体架构概览

Claude-Code 的 MCP 实现是一个**双向架构**：
- **作为 MCP Client**：连接外部 MCP Server，将其 tools/resources/prompts 暴露给 Agent
- **作为 MCP Server**：通过 `--mcp` 参数启动，将 Claude-Code 的原生工具（Bash、Read、Write 等）通过 stdio MCP 暴露给外部客户端（如 IDE 插件）

整个 MCP 子系统分为以下几个层次：

```
┌─────────────────────────────────────────────────────────────┐
│  UI 层 (React Components)                                    │
│  MCPSettings / MCPReconnect / MCPToolListView / Elicitation  │
├─────────────────────────────────────────────────────────────┤
│  命令层 (Commands)                                           │
│  /mcp 命令 — 管理、重连、启用/禁用服务器                     │
├─────────────────────────────────────────────────────────────┤
│  连接管理层 (Hooks & Services)                               │
│  useManageMCPConnections / MCPConnectionManager              │
├─────────────────────────────────────────────────────────────┤
│  客户端核心 (MCP Client Core)                                │
│  client.ts — 连接、发现、调用、结果转换                       │
├─────────────────────────────────────────────────────────────┤
│  传输层 (Transports)                                         │
│  stdio / SSE / HTTP / WS / claudeai-proxy / SDK / in-process │
├─────────────────────────────────────────────────────────────┤
│  认证层 (Auth)                                               │
│  OAuth 2.0 / OIDC / XAA (Cross-App Access) / Claude.ai Proxy│
├─────────────────────────────────────────────────────────────┤
│  配置层 (Config)                                             │
│  多作用域配置 + 企业策略控制 + 去重合并                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 核心模块分析

### 2.1 MCP Client 核心 (`src/services/mcp/client.ts`)

这是整个 MCP 子系统的**心脏**，承担了以下职责：

#### 2.1.1 连接工厂 (`connectToServer`)
- **多路传输支持**：根据配置类型自动创建对应的 Transport：
  - `stdio` → `StdioClientTransport`（本地 CLI 工具，如 `npx`、`uvx`）
  - `sse` → `SSEClientTransport`（远程 SSE 服务器，带自定义 fetch 注入 auth header）
  - `http` → `StreamableHTTPClientTransport`（流式 HTTP，MCP 2025-11-05 规范）
  - `ws` / `ws-ide` → 自定义 `WebSocketTransport`
  - `sse-ide` → IDE 专用 SSE（无 auth）
  - `claudeai-proxy` → Claude.ai 云端代理（Bearer Token + 401 自动重试）
  - `sdk` → `SdkControlClientTransport`（CLI ↔ SDK 进程间通信）
  - `in-process` → `createLinkedTransportPair()`（内存内通信，避免子进程开销）

- **连接缓存与失效**：使用 `lodash.memoize` 按配置哈希缓存连接；当 `onclose` 触发时自动清除缓存，下次调用时透明重连
- **进程清理策略**（stdio 专用）：`SIGINT` → 100ms 等待 → `SIGTERM` → 400ms 等待 → `SIGKILL`，确保 Docker 容器也能被终止

#### 2.1.2 能力发现
- **`fetchToolsForClient`**：调用 `tools/list`，将每个 MCP Tool 转换为 Claude-Code 内部 `Tool` 格式
  - 工具名前缀化：`mcp__<serverName>__<toolName>`，防止与原生工具冲突
  - 描述截断：最多 `2048` 字符，防止 OpenAPI 生成的巨型描述撑爆上下文窗口
  - 注解映射：将 MCP 的 `isReadOnly`/`isDestructive`/`isOpenWorld`/`isConcurrencySafe` 映射到内部权限模型
- **`fetchResourcesForClient`**：调用 `resources/list`，返回带 `server` 字段的资源列表
- **`fetchCommandsForClient`**：调用 `prompts/list`，将 MCP Prompt 转换为斜杠命令

#### 2.1.3 工具调用 (`callMCPTool`)
- 超时默认约 **27.8 小时**，支持通过 `AbortSignal` 取消
- **会话过期检测**：捕获 HTTP 404 + JSON-RPC `-32001`（Session not found）和 `-32000`（Connection closed），自动清除缓存并抛 `McpSessionExpiredError`，支持调用方重试一次
- **URL Elicitation 重试**：MCP Server 可以在工具执行中请求用户打开 URL（`-32042` 错误），Claude-Code 会挂起工具调用、弹出 UI 对话框或结构化 I/O，用户完成后自动重试
- **大结果卸载**：超过阈值的结果不直接放入上下文窗口，而是持久化到磁盘，Agent 收到"请读取文件"的指令

#### 2.1.4 结果转换 (`transformMCPResult`)
支持 MCP 规范定义的全部内容类型：
- `text` → 直接文本块
- `image` → 图片块（自动 resize/compress）
- `audio` → 音频块
- `resource` / `resource_link` → 资源引用块
- `blob` → 二进制大对象持久化

---

### 2.2 连接状态管理 (`useManageMCPConnections`)

这是一个 React Hook，是连接管理层的中枢：

#### 2.2.1 并发连接策略
- **分区连接**：将服务器分为 **local**（stdio/sdk，低并发，默认 3）和 **remote**（sse/http/ws，高并发，默认 20）
- **动态槽释放**：使用 `p-map` 动态释放连接槽，而非固定批次边界
- **指数退避重连**：SSE 等远程传输断开后自动重连，`MAX_RECONNECT_ATTEMPTS=5`，`INITIAL_BACKOFF_MS=1000`，`MAX_BACKOFF_MS=30000`

#### 2.2.2 批量状态更新
为避免并行连接时的 React 重渲染风暴，所有状态更新在 **16ms 窗口内**批量 flush 到 `AppState`

#### 2.2.3 动态更新监听
如果 MCP Server 声明了 `listChanged` 能力，会自动注册通知处理器：
- `ToolListChangedNotificationSchema`
- `ResourceListChangedNotificationSchema`
- `PromptListChangedNotificationSchema`
能力变更时实时刷新 Agent 的工具列表

---

### 2.3 配置系统 (`src/services/mcp/config.ts`)

#### 2.3.1 多作用域配置合并
配置来源按优先级从高到低：
```
enterprise > local > project（含父目录遍历）> user > plugin > claudeai > dynamic
```
- **Project 作用域**：读取 `.mcp.json`，支持从当前目录向上遍历到根目录（类似 `.git` 的查找逻辑）
- **Plugin 去重**：插件提供的 MCP Server 与手动配置通过签名（command 数组或 URL）去重
- **Claude.ai 配置**：从 Claude.ai API 获取组织级 MCP 服务器列表（需 `user:mcp_servers` scope）

#### 2.3.2 企业策略控制
- **Allowlist/Denylist**：可按名称、命令数组、URL 通配符阻止/允许服务器
- **Windows npx 守卫**：Windows 上如果使用 `npx` 但没有 `cmd /c` 包装会发出警告
- **原子写入**：`.mcp.json` 通过 temp file + `datasync` + `rename` 保证写入原子性

#### 2.3.3 环境变量展开
支持在 command、args、env、URL、headers 中展开 `$VAR` / `${VAR}`

---

### 2.4 认证系统 (`src/services/mcp/auth.ts` + `xaa.ts`)

#### 2.4.1 标准 OAuth 2.0 / OIDC
`ClaudeAuthProvider` 实现了 MCP SDK 的 `OAuthClientProvider` 接口：
- **自动发现**：RFC 9728 → RFC 8414 发现授权服务器元数据（带 path-aware fallback，修复 Azure AD/Okta realm 问题）
- **Token 存储**：存于系统安全存储（macOS Keychain / Windows Credential Manager / Linux Secret Service），Key 为配置 SHA-256 哈希
- **主动刷新**：Token 在过期前 **5 分钟**自动刷新
- **跨进程刷新去重**：使用文件锁（lockfile）防止多进程同时刷新
- **Step-up Auth**：检测到 403 + `insufficient_scope` 时缓存 scope，下次 OAuth 流程自动申请更大权限

#### 2.4.2 XAA (Cross-App Access / SEP-990)
企业 SSO 场景，**一次浏览器登录，授权所有 XAA 配置的 MCP Server**：
1. 用户通过标准 OIDC `authorization_code` + PKCE 登录企业 IdP
2. `id_token` 缓存到安全存储
3. 对每个 XAA MCP Server：
   - RFC 9728 PRM 发现受保护资源
   - RFC 8414 AS 元数据发现
   - RFC 8693 Token Exchange：`id_token` → ID-JAG（Identity Assertion Authorization Grant）
   - RFC 7523 JWT Bearer Grant：ID-JAG → `access_token`
4. 后续刷新通过 `xaaRefresh()` 静默完成，**无需浏览器交互**

#### 2.4.3 Claude.ai Proxy
- 使用用户的 Claude.ai OAuth Token
- 自定义 fetch wrapper，401 时强制刷新 Token 并重试一次
- 组织管理员在 Claude.ai Web 端配置 MCP Server，终端用户零配置使用

#### 2.4.4 Auth 失败缓存
远程服务器返回 401 时，会写入 **15 分钟 TTL** 的缓存文件（`mcp-needs-auth-cache.json`），避免反复探测需要重新授权的服务器。当服务器处于 `needs-auth` 状态时，会自动注入一个 `McpAuthTool` 到 Agent 工具列表，用户可以通过调用该工具触发重新认证。

---

### 2.5 权限系统 (`src/services/mcp/channelPermissions.ts` + `interactiveHandler.ts`)

Claude-Code 的 MCP 工具权限不是简单的 allow/deny，而是一个**多通道竞速**模型：

```
用户本地确认 ←─┬─→ Bridge (CCR/claude.ai Web) 远程确认
               │
               └─→ Channel Relay (Telegram/iMessage/Discord) 手机确认
               │
               └─→ Hooks (自动化脚本预批准)
               │
               └─→ Bash Classifier (AI 自动分类批准)
```

- **本地权限对话框**：标准的 TUI 对话框，支持 allow/deny/always/rules
- **Bridge 权限**：当用户通过 claude.ai Web 端连接时，权限请求会同步到 Web 端，先响应者获胜
- **Channel Relay**（实验性）：通过外部渠道（Telegram/iMessage/Discord）发送权限请求，用户在手机上回复 `yes abc12` 即可批准。关键安全设计：
  - CC **不直接解析**原始文本，只接受渠道服务器发出的结构化事件
  - 信任边界是 Allowlist（`tengu_harbor_ledger`），而非终端本身
  - 请求 ID 使用 5 字母短码（FNV-1a 哈希），过滤冒犯性字符串
- **Tool-scoped Rules**：MCP 工具使用 `mcp__server__tool` 全名进行权限匹配，避免内置工具的 deny 规则误伤同名 MCP 工具

---

### 2.6 传输层详解

| 传输类型 | 场景 | 关键特性 |
|---------|------|---------|
| **stdio** | 本地 CLI 工具 | stderr 捕获用于调试；SIGINT→SIGTERM→SIGKILL 三级清理 |
| **sse** | 远程 SSE 服务器 | 自定义 `eventSourceInit.fetch` 注入 auth header 和 proxy；SSE GET 排除在 60s 超时外 |
| **http** | 流式 HTTP (2025-11-05 规范) | `StreamableHTTPClientTransport` + `wrapFetchWithTimeout`；`Accept: application/json, text/event-stream` 规范化 |
| **ws / ws-ide** | WebSocket 服务器 | 使用 `ws` 库（Node）或 Bun 原生 WebSocket；IDE 版带独立 auth token |
| **sse-ide** | IDE SSE | 无 auth，专供 VS Code 等 IDE 扩展 |
| **claudeai-proxy** | Claude.ai 云端连接器 | Bearer Token + 401 强制刷新重试 |
| **sdk** | SDK 内嵌 MCP Server | `SdkControlClientTransport` 通过 stdout 路由 JSON-RPC 消息 |
| **in-process** | Chrome / Computer Use 内置 | `createLinkedTransportPair()` 内存内通信，避免每个子进程 325MB 内存开销 |

**Proxy 支持**：`HTTP_PROXY` / `HTTPS_PROXY` 环境变量透传到 HTTP/SSE/WebSocket 传输。

---

### 2.7 参数诱导 (`elicitationHandler.ts`)

MCP Server 可以声明 `sampling/elicitation` 能力，在工具调用过程中**向用户请求额外参数**：
- **Form 模式**：渲染基于 JSON Schema 的表单对话框
- **URL 模式**：要求用户打开外部 URL 完成操作（如 OAuth 授权），完成后自动通知
- **Hook 支持**：支持 `executeElicitationHooks` 自动化预填或拒绝

---

### 2.8 Claude-Code 作为 MCP Server (`src/entrypoints/mcp.ts`)

当启动参数包含 `--mcp` 时，Claude-Code 变身为 MCP Server：
- 通过 `StdioServerTransport` 在 stdio 上暴露 MCP 协议
- 注册 `ListToolsRequestSchema` 和 `CallToolRequestSchema` 处理器
- 工具描述动态生成：调用每个工具的 `tool.prompt()` 生成描述
- 输入 Schema：Zod → JSON Schema 转换（`zodToJsonSchema`）
- 输出 Schema：仅当根为 `type: "object"` 时包含（跳过 `anyOf`/`oneOf` 根兼容问题）
- 工具调用走标准 `tool.call()` 路径，但使用最小化的 `ToolUseContext`（非交互式、无 Agent、空消息历史）
- 使用 LRU 文件状态缓存防止内存无限增长

---

## 3. 关键设计模式

### 3.1 Memoization + 主动失效
`connectToServer` 被 memoize 缓存，但 `onclose` 事件会主动清除 memo key 和所有 fetch 缓存（tools/resources/commands/skills），使得调用方无需关心重连逻辑。

### 3.2 懒加载（Lazy Loading）
特性门控的模块（如 `claudeInChromeToolRendering`、`computerUseWrapper`）使用 `require()` 在运行时按需加载，避免启动时加载重型依赖。

### 3.3 两阶段配置加载
- **Phase 1（快速）**：本地文件配置立即加载，不阻塞启动
- **Phase 2（网络）**：Claude.ai 连接器配置异步获取，完成后合并更新

### 3.4 状态机模型
每个 MCP Server 的状态流转：
```
pending → connected → (failed | needs-auth | disabled)
         ↑_____________│
           (自动重连)
```

### 3.5 错误遥测安全
大量使用 `TelemetrySafeError_I_VERIFIED_THIS_IS_NOT_CODE_OR_FILEPATHS` 命名约定，确保包含用户数据的错误消息在写入分析日志前被脱敏。

---

## 4. 独特特性总结

| 特性 | 说明 |
|------|------|
| **IDE 集成传输** | `sse-ide` / `ws-ide` 专门供 VS Code 等 IDE 扩展使用，带独立认证和遥测 |
| **内存内内置 Server** | Chrome MCP 和 Computer Use MCP 通过 `createLinkedTransportPair()` 在进程内运行，节省约 **650MB** 内存 |
| **Channel 通知** | 实验性的 `notifications/claude/channel` 能力，允许 MCP Server 向对话队列推送消息，实现双向通信 |
| **URL Elicitation 自动重试** | 工具执行中请求打开 URL，用户完成后自动恢复工具调用，无需手动重试 |
| **大结果卸载** | 超大 MCP 结果自动落盘，避免上下文窗口爆炸 |
| **插件签名去重** | 插件和手动配置指向同一命令/URL 时静默去重 |
| **Stdio 三级信号清理** | SIGINT → SIGTERM → SIGKILL 的渐进式终止，专门解决 Docker 容器清理问题 |
| **SDK MCP 模式** | CLI 作为 MCP Client，SDK 进程内运行 MCP Server，通过控制通道而非网络通信 |
| **XAA 企业单点登录** | 一次 IdP 登录 → 缓存 `id_token` → 静默授权所有 XAA 服务器，无 per-server  consent 屏幕 |
| **Bash 分类器集成** | 对 Bash 工具运行异步 AI 分类器，可在用户响应前自动批准安全命令 |

---

## 5. 与我们项目的对比与借鉴

| 维度 | Claude-Code | 我们的项目 (agent-material) |
|------|------------|---------------------------|
| **传输支持** | stdio / SSE / HTTP / WS / IDE / in-process / SDK / claudeai-proxy | stdio / SSE |
| **认证** | OAuth 2.0 + OIDC + XAA + Claude.ai Proxy | 无（配置中仅预留字段） |
| **配置管理** | 7 层作用域 + 企业策略 + 自动去重 | 单层 YAML + 环境变量 |
| **权限模型** | 多通道竞速（本地 + Bridge + Channel + Hooks + Classifier） | HITL 中断 + 前端审批 |
| **动态更新** | 支持 `listChanged` 实时刷新 | 不支持 |
| **结果处理** | 大结果自动落盘、图片压缩、二进制持久化 | 直接返回文本 |
| **作为 Server** | 支持 `--mcp` 模式暴露原生工具 | 支持 FastAPI 嵌入 + stdio / SSE |
| **重连策略** | 指数退避 + 自动重连 + 会话过期检测 | 每次调用新建临时连接 |
| **参数诱导** | 支持表单和 URL 两种 elicitation 模式 | 不支持 |

### 5.1 可直接借鉴的要点

1. **工具名前缀化**：`mcp__<server>__<tool>` 的命名方式可有效防止命名冲突，建议在我们的 `mcp_tool_manager.py` 中引入
2. **连接缓存**：我们的项目"每次调用新建临时连接"虽然简单但开销大，可参考 Claude-Code 的 memoize + onclose 失效模式
3. **描述截断**：2048 字符的描述上限可防止 OpenAPI 文档撑爆上下文
4. **大结果卸载**：RAG 或 MP 下载返回大结果时，可考虑自动写入文件而非直接塞入上下文
5. **两阶段配置加载**：本地配置立即加载，网络配置（如云端 MCP 注册表）异步合并
6. **Auth 失败缓存**：15 分钟 TTL 的 `needs-auth` 缓存，避免反复尝试已失效的认证
7. **进程清理策略**：stdio 模式下 `SIGINT → SIGTERM → SIGKILL` 的三级清理逻辑

### 5.2 暂不适用但值得关注的特性

1. **XAA 企业认证**：当前项目面向科研个人用户，暂无企业 SSO 需求
2. **Channel Relay**：需要外部消息渠道基础设施
3. **IDE 专用传输**：需要与 VS Code 等 IDE 深度集成
4. **SDK 控制传输**：项目架构不同（Python FastAPI vs Node CLI + SDK）

---

## 6. 关键文件索引

| 文件路径 | 职责 |
|---------|------|
| `src/services/mcp/client.ts` | MCP Client 核心：连接、发现、调用、结果转换 |
| `src/services/mcp/useManageMCPConnections.ts` | React Hook：连接生命周期、并发控制、状态同步 |
| `src/services/mcp/config.ts` | 配置加载、合并、验证、策略控制 |
| `src/services/mcp/types.ts` | Zod Schema + TypeScript 类型定义 |
| `src/services/mcp/auth.ts` | OAuth 2.0 / OIDC / Token 管理 / Step-up |
| `src/services/mcp/xaa.ts` | XAA：RFC 8693 Token Exchange + RFC 7523 JWT Bearer |
| `src/services/mcp/xaaIdpLogin.ts` | XAA IdP 登录：OIDC `id_token` 获取与缓存 |
| `src/services/mcp/InProcessTransport.ts` | 内存内进程间传输 |
| `src/services/mcp/SdkControlTransport.ts` | CLI ↔ SDK 控制通道传输 |
| `src/services/mcp/claudeai.ts` | Claude.ai 云端 MCP 配置获取 |
| `src/services/mcp/elicitationHandler.ts` | 参数诱导（Form / URL 模式） |
| `src/services/mcp/channelPermissions.ts` | 外部渠道权限中继 |
| `src/services/mcp/normalization.ts` | 名称规范化 |
| `src/services/mcp/mcpStringUtils.ts` | 工具名解析与构建 |
| `src/entrypoints/mcp.ts` | Claude-Code 作为 MCP Server 的入口 |
| `src/tools/MCPTool/MCPTool.ts` | MCP Tool 基础原型定义 |
| `src/commands/mcp/index.ts` | `/mcp` CLI 命令入口 |
| `src/components/mcp/*.tsx` | MCP 相关的 React UI 组件 |

---

*报告完成。如需针对某个具体子模块进行更深入的分析，请告知。*
