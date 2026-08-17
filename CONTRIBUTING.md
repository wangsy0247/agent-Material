# Contributing

Thank you for helping improve Agent Material.

## Development workflow

1. Create a focused branch from the current default branch.
2. Copy `.env.example` to `.env` and set a local `AUTH_SECRET_KEY` of at least
   32 random characters. Never commit `.env`.
3. Start PostgreSQL/pgvector with `docker compose up -d`.
4. Run `make setup` in an isolated Python environment. It installs the PDF and
   Materials Project feature sets from `uv.lock`; the local CrossEncoder remains
   optional via `uv sync --extra local-reranker`.
5. Run `uv run pytest backend/tests` before opening a pull request.
6. Add or update tests for behavior changes and update user-facing docs.

Create the first administrator interactively with:

```bash
uv run python scripts/create_admin.py <username> --email <optional-email>
```

## Pull requests

Keep changes small and explain their purpose, risks, and validation. A pull
request must not contain credentials, anything from `interview_prep/`,
`evaluation_data/`, or `data/`, generated databases, model files, downloaded
Materials Project data, or local vector indexes.

Changes to authentication, MCP commands, file upload, retrieval authorization,
or external data transfer require explicit security tests. Dependency updates
must preserve third-party notices and pass a license audit.

Before a release, run `uv run pip-audit` and review every direct and vendored
dependency against `THIRD_PARTY_NOTICES.md`. Automated license output is an aid,
not a substitute for reviewing upstream license text.

## Reporting security issues

Follow `SECURITY.md`; do not open a public issue for an unpatched vulnerability.

## License

By submitting a contribution, you agree that it is licensed under Apache-2.0.
