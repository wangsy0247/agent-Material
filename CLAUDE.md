# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Materials Science LLM Agent (材料科学领域 LLM Agent) — a FastAPI + LangGraph + PGVector system integrating RAG literature search, MCP tool calling (Materials Project database), multi-turn memory, and streaming conversation. Frontend is a build-free vanilla JS SPA. All code, comments, docs, and UI are in Chinese.

## Commands

```bash
# Initialization (install deps, create .env, validate configs, create runtime dirs)
make setup

# Start API server (reload is disabled by default)
make run

# Manual start
bash scripts/start_server.sh          # dev with hot reload
bash scripts/start_server.sh prod     # production

# Start PostgreSQL only (for checkpointer/vector store)
docker-compose up -d

# Run all tests
python -m pytest backend/tests

# Run a single test file
python -m pytest backend/tests/test_sse_protocol.py

# WSL/Anaconda-specific libffi troubleshooting is documented in docs/deployment.md.
```

## Architecture

```
frontend/                        Vanilla JS SPA (chat + RAG management pages)
backend/
  app/gateway/                   FastAPI gateway — routers, SSE protocol, stream buffer, auth
  packages/agent_material/       Core Python package (the real logic lives here)
    agents/                      LangGraph ReAct agent (StateGraph) + tools
    rag_engine/                  RAG pipeline — document loading, chunking, vector retrieval, rerank
    memory/                      Three-layer memory (short-term / SQLite / long-term file-based)
    mcp/                         MCP client, session pooling, tool caching
    models/                      LLM model factory
    observability/               Structured logging + OTel tracing + Prometheus metrics
    utils/                       Config loader, path tools, prompt loader
  tests/                         pytest tests
mp_material_server/              Materials Project MCP server (stdio transport)
config/                          YAML configs (agent, auth, memory, prompt, queue, rag, vector_store)
scripts/                         Shell scripts (start server, WSL network fix, RAG dataset tools)
```

### Request Flow

1. FastAPI gateway (`backend/app/gateway/app.py`) registers routers under `/api/v1/` and serves `frontend/` as static files
2. `routers/agent.py` handles `/agent/chat` — creates SSE stream, wires memory, manages HITL approval
3. `ReactAgent` (`agents/React_agent.py`) takes user query, delegates to the compiled LangGraph `StateGraph`
4. The graph (`agents/graph/agent_graph.py`) runs nodes: memory → optional summarization → agent (LLM) → human_approval → tools → loop back to agent → finalize
5. Tools come from two sources: local RAG tools (process-internal) + MCP tools (subprocess via stdio)
6. Memory (`memory/manager.py`) injects system prompt + long-term context at graph entry; messages persist to SQLite via `SessionStore`

### SSE Protocol (6-channel event stream)

The SSE stream uses `id/event/data` triple-line framing. Event types are defined in `backend/app/gateway/schemas.py` as `SSEEvent.type`:

| Event Type | Purpose |
|---|---|
| `token` | LLM-generated token (with monotonic `index` for out-of-order detection) |
| `status` | Phase transitions (memory injected, phase:generating) |
| `tool_call` | Tool invocation requested |
| `tool_result` | Tool result (truncated to 500 chars in stream, with `truncated` flag) |
| `thinking` | Model reasoning content |
| `approval_required` | HITL — sensitive tool needs human confirmation |
| `done` | Normal completion (terminal; mutually exclusive with error/cancelled) |
| `error` | Runtime error (terminal) |
| `cancelled` | Client disconnect/abort (terminal) |

Heartbeat comment frames (`: ping`) are sent every 15s during silent periods — they do not carry event IDs and are not stored in the stream buffer.

### Stream Buffer & Reconnection (断流续传)

`backend/app/gateway/stream_buffer.py` maintains an in-process ring buffer (keyed by `(user_id, session_id)`, max 2000 events per run, 30min TTL). The `/agent/chat/resume` endpoint replays events after a given `Last-Event-ID`, then follows live events until a terminal frame. **Caveat**: this is single-process in-memory — only works with single-worker dev mode; multi-worker prod needs Redis.

### LangGraph State Machine

The graph (`agents/graph/agent_graph.py`) defines `AgentState` with fields: `messages`, `memory_context`, `tool_calls`, `iteration_count`, `session_id`, `pending_approval`, `query`, `final_answer`, `user_id`.

Node sequence: `memory` → `should_summarize` (conditional) → `agent` → `should_continue` (conditional: tools → `human_approval` → `tools` → `agent`; end/max_iter → `finalize`).

- HITL: `human_approval_node` checks `_is_sensitive_tool()` (configurable via `config/agent_config.yaml` `sensitive_keywords`), calls LangGraph `interrupt()` to suspend
- Summarization: triggered when non-system messages exceed `message_threshold` (default 20), delegates to `ContextCompressor` for a 4-stage compression
- Max 10 iterations; exceeding triggers forced termination in `finalize_node`
- Checkpointer: PostgreSQL via `langgraph-checkpoint-postgres` (configurable to SQLite or in-memory for dev)

### Memory System

Three layers coordinated by `memory/manager.py`:

1. **Short-term**: in-memory per-session message list (`ShortTermMemory`)
2. **Medium-term**: SQLite persistence via `SessionStore` (stored in `logs/` dir, WAL mode)
3. **Long-term**: file-based markdown memory (`USER.md` / `MEMORY.md`) in per-user directories under `memory_data/`, with optional vector retrieval

Session lifecycle: `MemoryManager.create_session()` → `load_session()` → messages flow → `end_session()` triggers automatic memory extraction. All operations are keyed by `(user_id, session_id)` for multi-tenant isolation.

### RAG Pipeline

`rag_engine/` contains the full pipeline: document loading (`file_loader.py` for txt/md/html/docx/pptx; `mineru_loader.py` for PDF via MinerU API → Markdown), text splitting (`text_spliter.py` with recursive/semantic/parent/markdown-header strategies), multi-granularity vector indexing (small + large chunk collections in PGVector), hybrid retrieval (`vector_store.py` with PGVector vector search + BM25 keyword search fused via RRF), DashScope reranking (`dashscope_reranker.py`, with local CrossEncoder fallback), query rewriting (`query_rewriter.py` — expand/hyde/step_back/decompose, pipelined with `+`), and query routing (`query_router.py` — chat detection, irrelevant-question interception, classification). `RagService` (`rag_service.py`) is the main query entry point; `RagIndexingService` and the PostgreSQL repositories are the only RAG write/control plane. PostgreSQL stores indexing tasks, document versions, file state, and knowledge-base revisions.

### MCP Tool System

MCP servers are configured in `extensions_config.json`. `mcp/client.py` uses `langchain-mcp-adapters` to create a `MultiServerMCPClient`. Tools are cached by `mcp/cache.py` with config-file mtime-based invalidation. The `mp_material_server/` directory implements a Materials Project API wrapper as an MCP stdio server.

### Config System

`config/` holds 8 YAML files. `utils/config_handler.py` loads and merges them at import time into module-level singletons, supporting `${ENV_VAR:-default}` variable substitution. Key settings:
- `agent_config.yaml`: model name (`qwen3.6-plus`), checkpointer type (`postgres`/`sqlite`/`memory`), summarization thresholds, sensitive tool keywords for HITL
- `memory_config.yaml`: context window budget, long-term memory toggle
- `vector_store_config.yaml`: embedding model, collection names, HNSW index parameters, BM25 parameters

### Infrastructure & Database Ports

- **pgvector**: Docker Compose service on loopback host port **5433** (container 5432). Development credentials are declared in `.env.example`; non-local deployments must override them.
- **API server**: default port 8000 (`AGENT_API_PORT`)
- **Auth DB**: SQLite at `logs/auth_users.db`; no default administrator is created. Use `scripts/create_admin.py`.
- **Memory DB**: SQLite at `logs/memory_sessions.db` (WAL mode, stores session metadata + messages)
- **RAG control plane**: PostgreSQL tables for tasks, document versions, file states, and revisions

### Multi-Tenancy & Heavy-Import Avoidance

`ReactAgent` and `MemoryManager` are shared singletons (created lazily in `auth/dependencies.py`). User isolation is achieved purely by threading `user_id` through every method call — sessions, memory pools, and long-term files are all keyed by `(user_id, session_id)`.

Heavy modules (PGVector, PyTorch, sentence-transformers, MCP clients) are **lazily imported inside functions** rather than at module top level. This avoids WSL + Anaconda segfaults during uvicorn's multiprocess fork (hot reload spawns a second process). If you add imports of psycopg, torch, or ONNX-related packages, keep them inside function bodies or guard with the same pattern used in `routers/rag.py` and `dependencies.py`.

### Frontend

No bundler/build step. Two HTML pages (`index.html` for chat, `rag.html` for RAG management). JS modules in `static/js/`:
- `common.js` — shared utilities (API calls, auth token)
- `stream_client.js` — SSE stream parser with reconnection/Last-Event-ID support
- `app.js` — chat UI logic
- `rag.js` — RAG management UI
- `toast.js` — toast notification component

Vendor libraries (marked, highlight.js, DOMPurify) are vendored in `static/vendor/`.

## Observability

Three-layer system in `observability/`:
1. Structured JSON logging with request/trace context injection
2. OpenTelemetry tracing (auto-instruments FastAPI, manual spans for graph nodes and MCP calls)
3. Prometheus metrics at `/metrics` (request counters, latency histograms, agent iteration counts)

LangSmith and Langfuse integrations are available via environment variables (see `.env.example`).

## Additional Context

- The `docs/` directory is gitignored (personal notes removed from VCS in commit 77073d8) but contains locally retained design documents covering the SSE protocol, MCP setup, RAG architecture, memory system, and observability analysis. These are valuable context when working on those subsystems.
- The `mp_material_server/` is a standalone Python package with its own `requirements.txt` — zero dependencies on the main project. It exposes two tools: `mp_query` (material search) and `mp_download` (POSCAR + properties download; annotated as destructive, which triggers HITL approval).
- The codebase uses Ruff for linting (local `.ruff_cache/`). There is no pre-commit hook or formatter configuration committed.

## Key Environment Variables

See `.env.example` for the full list. Critical ones:
- `DASHSCOPE_API_KEY` — LLM API key (required)
- `MP_API_KEY` — Materials Project API key
- `POSTGRES_CHECKPOINTER_URL` — PostgreSQL connection string for LangGraph checkpointer
- `AGENT_API_RELOAD` — enable/disable uvicorn hot reload (default: true)
