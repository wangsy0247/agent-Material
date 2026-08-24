# Changelog

All notable changes will be documented in this file. The project follows
semantic versioning once the first public release is tagged.

## [Unreleased]

### Added

- Open-source governance, security, licensing, and third-party notice files.
- Administrator-only knowledge-base mutation and MCP configuration access.
- Streaming, size-limited, signature-checked, atomic file upload.

### Changed

- JWT secrets are now required at startup and no default administrator is
  created automatically.
- CORS defaults to localhost origins and runtime MCP updates are disabled by
  default.
- Personal interview material, evaluation data/results, and knowledge-base
  documents are now explicitly local-only and excluded as whole directories.

### Removed

- The application `Dockerfile` and image-build context configuration. Local
  PostgreSQL/pgvector setup remains available through `docker-compose.yaml`.
