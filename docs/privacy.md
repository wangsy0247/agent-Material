# Data flow and privacy

Agent Material processes user questions, conversation history, uploaded files,
retrieved passages, generated answers, tool parameters, and operational logs.
Deployers are responsible for obtaining permission to process this data.

## Local data

Depending on configuration, the service stores:

- authentication and session records in SQLite under `logs/`;
- source PDFs under `data/`;
- extracted figure assets under `.rag_assets/`;
- document versions, indexing tasks, metadata, and vectors in PostgreSQL;
- downloaded Materials Project files under `downloads/` or `workspace/`;
- process-local SSE replay events and BM25 snapshots in memory.

The entire `interview_prep/`, `evaluation_data/`, and `data/` directories are
private local content and are excluded from the public repository. Extracted
`.rag_assets/` may reproduce figures from those documents and is therefore also
private and excluded. Runtime paths such as `logs/`, `workspace/`, and
`downloads/` are excluded as well.
Backups and logs must be protected separately; `.gitignore` is not an
access-control mechanism.

## External data transfers

When enabled, data may leave the deployment:

| Provider or integration | Data that may be sent | Purpose |
| --- | --- | --- |
| DashScope | prompts, questions, retrieved passages, embedding text; extracted figure/Panel images only when a visual feature is explicitly enabled | chat, text and multimodal embeddings, reranking, opt-in visual description and multimodal answers |
| MinerU service/SDK backend | uploaded PDF pages or document bytes | PDF parsing and OCR |
| Materials Project | formula/material query parameters | material search and download |
| Langfuse / OpenTelemetry exporter | prompts, outputs, tool calls, traces, metadata | observability |

Review each provider's terms, region, retention, and training settings before
enabling it. Do not process confidential or regulated documents unless the
deployment and providers have been approved for that use.

## Defaults

- External tracing is disabled in `.env.example`.
- Offline visual enrichment, image/Panel embeddings, and online multimodal answers
  are disabled in `.env.example`.
- Langfuse and OTLP exporters have no credentials by default.
- Runtime MCP configuration is disabled by default.
- The API binds to loopback and CORS allows only localhost by default.

Unset credentials do not guarantee that every custom extension is local. Audit
`extensions_config.json`, model configuration, and any added MCP server before
deployment.

## Secrets

Use environment variables or a secret manager. Never commit `.env`, API keys,
JWT secrets, database passwords, OAuth tokens, request captures, or production
configuration. Materials Project keys belong to the individual account and must
not be shared in a public demo.

## Retention and deletion

Knowledge-base deletion is asynchronous and should be verified through task
status and PostgreSQL consistency checks. Removing a source from the active
index does not automatically remove external-provider records, logs, backups,
or previously downloaded files. Define retention and deletion procedures for
each storage and provider before accepting user data.
