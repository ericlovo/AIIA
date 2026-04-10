# Security Policy

## Supported versions

Only the latest minor release of AIIA receives security fixes. Older
versions do not get backports.

| Version | Supported |
|---------|-----------|
| 0.4.x   | Yes       |
| < 0.4   | No        |

## Reporting a vulnerability

**Please do not open public GitHub issues for security bugs.** Tenant
isolation, memory sync, and MCP tool execution are all security-sensitive
paths and we would rather fix issues quietly than have them exploited in
the wild.

Preferred reporting paths, in order:

1. **GitHub Security Advisories** — open a private advisory at
   https://github.com/ericlovo/AIIA/security/advisories/new. This is the
   fastest way to reach the maintainers and keeps the discussion scoped to
   people who need to see it.
2. **Email** — `security@aplora.ai` for anything that can't be reported
   through GitHub.

When you report, please include:

- A short description of the vulnerability.
- The version or commit SHA you reproduced it against.
- Reproduction steps, PoC code, or a minimal failing test if you have one.
- Your assessment of the impact (information disclosure, RCE, DoS, auth
  bypass, etc.).

## What to expect

- **Acknowledgment within 48 hours** of your report.
- **Triage + initial assessment within 5 business days**, including whether
  we can reproduce the issue and an initial severity rating.
- **Fix timeline** depending on severity: critical issues are patched as
  quickly as we can cut a release; medium/low issues land in the next
  scheduled minor release.
- **Public disclosure** is coordinated with the reporter. We'll credit you
  in the advisory and the CHANGELOG unless you prefer to stay anonymous.

## Scope

In-scope for this policy:

- Code in `local_brain/` (the Python package).
- The MCP server and its tool handlers.
- The Command Center dashboard and its FastAPI backend.
- Build/packaging files (`Dockerfile`, `docker-compose.yml`,
  `pyproject.toml`, CI workflows).
- Any scripts or recipes published in this repository.

Out of scope:

- Bugs that require local file system access already equivalent to the
  user the process is running as. (Anything you can do to your own box
  without exploiting AIIA is not an AIIA vulnerability.)
- Issues in upstream dependencies (Anthropic SDK, Google SDK, Supermemory
  SDK, ChromaDB, FastAPI, etc.) — please report those to the upstream
  projects. We'll update pins when fixes land.
- Social-engineering attacks against maintainers.

## Credit

Contributors who report valid vulnerabilities will be acknowledged in the
advisory and the CHANGELOG unless they ask to stay anonymous. We don't
currently run a bug bounty program.
