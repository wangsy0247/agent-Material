# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Materials science LLM Agent system — FastAPI backend + ReAct agent + RAG knowledge base + three-layer memory. Package name: `agent-material`.

## Commands

```bash
pip install -e .                           # Install dependencies
pytest -v                                  # Run all tests
pytest tests/test_rag_retrieval.py -v -s   # RAG eval (slow — loads models)
bash scripts/start_server.sh dev           # Dev server with hot reload
uvicorn api_server:app --reload            # Alternative dev server
```

## Architecture

**Startup flow**: `api_server.py` → lifespan initializes MCP tools → FastAPI serves routes at `/api/v1/*` → frontend at `/`.

**Request flow** (agent chat): `api/routes/agent.py` → `api/dependencies.py` (lazy singleton via `@lru_cache`) → `agent/React_agent.py` → `model/model_factory.py` (ChatAnthropic via DeepSeek) + tools.

**Config system**: All non-sensitive config in `config/*.yaml`, loaded by `utils/config_handler.py` at import time. Sensitive values (API keys) in `.env`. Never hardcode paths or parameters — use config_handler.

**LLM**: `ChatAnthropic` pointed at DeepSeek (`ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic`). Embeddings via `DashScopeEmbeddings`.

**RAG pipeline** (4-stage in `rag/vector_store.py`):
1. Vector retrieval (ChromaDB, custom `ChromaSimilarityThresholdRetriever` — explicitly converts cosine distance → similarity)
2. BM25 keyword retrieval (in-memory index over loaded documents)
3. Hybrid fusion (RRF, `EnsembleRetriever`)
4. CrossEncoder rerank (`HuggingFaceCrossEncoder` + `ContextualCompressionRetriever`, model at `config/chroma_config.yaml` `reranker_model`)

**Memory system** (3 layers, coordinated by `memory/manager.py`):
- L1: `short_term.py` — in-memory context window, token-budget-aware compression
- L2: `session_store.py` — SQLite (WAL mode) at `logs/memory_sessions.db`
- L3: `long_term.py` — file-based (`workspace/memory/USER.md`, `MEMORY.md`) + ChromaDB vector memory

**MCP**: `mcp_servers/client.py` connects to stdio/SSE MCP servers configured in `config/mcp_config.yaml`. Tools are converted to LangChain `StructuredTool` by `agent/tools/mcp_tool_manager.py`.

**Tools** (available to agent): Native tools in `agent/tools/agent_tools.py` (`@tool` decorator) + MCP tools auto-discovered at startup. Agent middleware (`agent/tools/middleware.py`) monitors tool calls and logs before model invocation.

## Key conventions

- **Chinese**: All docstrings, comments, and log messages use Chinese.
- **Type hints required**: Use Python 3.10+ syntax (`str | None`, `list[Document]`).
- **Lazy loading**: Heavy objects (VectorStore, reranker, Agent, RagService) use module-level globals + first-call initialization pattern. See `api/dependencies.py`.
- **Paths**: Always use `utils/path_tools.py::get_abs_path()` for relative paths to support different launch methods (python, uvicorn, pytest).
- **Logging**: Use `utils/logger.py::get_logger(name)`, never `print`.
- **Pydantic V2**: Use `model_dump()`, `ConfigDict(extra="allow")`, not V1 API.

## Testing

`tests/conftest.py` injects `MagicMock` for missing dependencies so tests import cleanly even without all packages installed. RAG tests (`test_rag_retrieval.py`) evaluate Recall/Precision/MRR/NDCG across 13 docs and 8 queries, with TF-IDF fallback if DashScope is unavailable.

## Gotchas

- ChromaDB `similarity_search_with_score` returns **cosine distance** (0–2), not similarity. The custom `ChromaSimilarityThresholdRetriever` converts: `similarity = 1 - distance/2`.
- Reranker model path in `config/chroma_config.yaml` — if the local path doesn't exist, reranking is silently skipped.
- MCP tool connections are created and torn down per-invocation (not persistent), to avoid stale connection complexity.
- `CrossEncoderReranker` version mismatch: this project patches `HuggingFaceCrossEncoder` to add `.predict()` if only `.score()` exists.

For detailed file listings, API docs, and extension guides, see `AGENTS.md`.
