# Security policy

## Supported versions

Security fixes are provided for the latest tagged release and the default
branch. Earlier development snapshots are not supported.

## Reporting a vulnerability

Do not publish exploit details, credentials, or private documents in an issue.
Use GitHub private vulnerability reporting for this repository:

<https://github.com/wangsy0247/agent-Material/security/advisories/new>

If private reporting is unavailable, contact the maintainer through the
repository owner profile and disclose only that you have a security report.
The maintainer will provide a private channel. Expect acknowledgement within
seven days; remediation timelines depend on severity and reproducibility.

## Deployment boundary

The initial open-source release targets a trusted, single-tenant local
deployment. Knowledge-base mutation and MCP configuration are administrator
operations. Runtime MCP configuration is disabled by default. Do not expose
the service directly to the public Internet without TLS, a reverse proxy,
rate limits, access logging, backups, and an independent security review.

Never include uploaded documents, `.env` files, API keys, access tokens,
database dumps, vector indexes, or production logs in a vulnerability report.
