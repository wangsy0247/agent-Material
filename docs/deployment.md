# Deployment

## Supported release profile

The initial open-source profile is one FastAPI process plus one PostgreSQL 16
instance with pgvector. It is suitable for a trusted local network, development,
evaluation, and portfolio demonstrations.

## Local setup

```bash
docker compose up -d
make setup
uv run python scripts/create_admin.py your_admin
make run
```

`docker-compose.yaml` starts PostgreSQL/pgvector only. The API runs on the host.
The database port binds to `127.0.0.1` and uses explicit development credentials;
replace them for every non-local deployment.

## Production prerequisites

Before any Internet-facing deployment, add:

- TLS and an authenticated reverse proxy;
- request-body, login, upload, query, and task rate limits;
- database backups and a tested restore procedure;
- centralized redacted logs and alerting;
- network egress rules for LLM, OCR, telemetry, and MCP destinations;
- secret-manager injection and rotation;
- dependency and application security scanning;
- an independent review of tenant and document authorization requirements.

Do not enable `AGENT_ALLOW_RUNTIME_MCP_CONFIG` on an exposed deployment. An MCP
stdio command can execute with the API process permissions, and an HTTP MCP URL
can make outbound requests.

## Multi-worker limitation

Do not run multiple Uvicorn workers in `v0.1`. SSE replay state and other caches
are process-local. Multiple workers can route a resume request to a process that
does not own the original event buffer. Use a shared Redis/event-store design
before scaling horizontally.

## WSL/Conda libffi

Some Anaconda environments on WSL load a `libffi` version incompatible with
`psycopg`. Confirm the actual linker error first. As an environment-specific
workaround, run the affected command with:

```bash
LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7 uv run pytest backend/tests
```

Do not place this setting in shared startup scripts: the path is
distribution-specific and may prevent startup on other systems.
