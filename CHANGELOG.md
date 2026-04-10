# Changelog

All notable changes to AIIA are documented here. This project adheres to
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] — 2026-04-10

### Added
- `pyproject.toml` as the single source of truth for the installable Python
  package. Declares Apache-2.0 metadata, Python ≥ 3.10, all runtime deps from
  `requirements.txt`, and a `dev` extra for `ruff`, `pytest`, and `mypy`.
- `local_brain/__version__.py` — single-line module that every runtime
  component (local_api, command_center) reads for its version string.
- `CHANGELOG.md` (this file).
- `SECURITY.md` — vulnerability disclosure process with supported-version
  table and a security@ contact.
- `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1.
- `.github/PULL_REQUEST_TEMPLATE.md` — standard checklist for contributors.
- `.github/workflows/ci.yml` — GitHub Actions CI that runs ruff format + lint,
  pytest collection, and a **sanitization guard** that fails the build if any
  of a known list of proprietary references is reintroduced.

### Changed
- **Unified version across all components.** Previously the brain API pinned
  `0.3.0`, the command center pinned `2.1.0`, and the dashboard pinned
  `0.0.0`, while only `v0.1.0` was git-tagged. All of these now roll forward
  into a single `0.4.0` release. The pre-release `2.1.0` command-center
  number was never published and has been merged into the unified version —
  anyone who was tracking it should read `0.4.0` as its successor.
- `local_brain/scripts/daily_report.py` and `syntax_checker.py` no longer
  hardcode a product → category classification map. The classifier now loads
  from the `AIIA_PRODUCT_MAP` environment variable (path to a JSON file
  mapping `{"product-name": "category"}`). When unset, `platform`, `root`,
  and `shared` collapse into a `Platform` category and everything else
  becomes `Other`.
- `local_brain/command_center/server.py` — `_DEFAULT_PRODUCTS` was already
  generic; `FUTURE_PRODUCTS` is now empty and populated from
  `AIIA_FUTURE_PRODUCTS_CONFIG` env var. The `AGENTS` list was reduced to
  the canonical `conductor` / `fast` / `rlm` / `finance` / `legal` set with
  no tenant-specific routers or specialists.
- `local_brain/README.md` — architecture diagram and 5-filter priority
  framework are no longer written as if for one specific company's
  customers; they describe the framework in generic terms.
- Command Center static HTML (`dashboard.html`, `work.html`, `console.html`)
  — product selectors, constellation graph, topology map, and
  `COMPONENTS` / `SERVICES` / `PRODUCT_COLORS` lists rewritten around
  AIIA's own services (Local Brain API, Command Center, MCP Server,
  Ollama Bridge) instead of a fixed tenant list.

### Removed
- Residual references to specific products and clients that leaked into the
  public repo when it was carved out of the private monorepo, including
  product cards, constellation graph nodes, category maps, and one
  real person's name that appeared in a product description.

### Security
- Ran a repo-wide scrub for proprietary strings. The CI sanitization guard
  added in this release will fail the build if any of them reappear in
  future pull requests.

## Between 0.1.0 and 0.4.0 (unreleased development)

This window of work was merged to `main` but never tagged. The highlights
that landed during it:

- **EQ brain port** — full persistent memory stack moved into
  `local_brain/eq_brain/`: structured memory, knowledge store, supermemory
  bridge, vault writer, session indexer, recursive engine, story prioritizer,
  morning briefing generator.
- **Story runner** — `local_brain/story_runner/` for decomposing backlog
  stories into action queues with safety-gated execution.
- **Execution engine** — `local_brain/execution/` with strategies, verifiers,
  git ops, and a subprocess pool.
- **Metered cloud sync** — quality-scored push from local memory to
  Supermemory with budget enforcement (`TokenLedger`) and dedup.
- **Dashboard (React + Vite)** — new React 19 front-end scaffolded in
  `dashboard/`, alongside the existing static HTML command center.
- **Geometric story prioritization** — vector-based scoring on top of the
  additive 5-filter weighting.
- **Background task safety** — hardened async task lifecycle and error
  handling across all runners.
- **Expanded APIs** — new FastAPI routes in `local_api.py` and additional
  MCP tools in `mcp_server.py`.
- **Model upgrades** — Ollama model selection parameterized; default model
  recommendations moved from a single hardcoded choice to configurable.
- **Obsidian bridge** — vault writer exports memory and notes as
  Obsidian-compatible Markdown for cross-tool reading.
- **Scan and sync automation** — nightly security scanning (bandit,
  semgrep, trivy, trufflehog, shellcheck, hadolint) and memory sync via
  the `brain` CLI and launchd plists.

## [0.1.0] — 2026-02

### Added
- Initial public release of AIIA Local Brain.
- FastAPI service on port 8100 for memory, knowledge, and Ollama routing.
- Command Center dashboard on port 8200.
- MCP server integration for Claude Code.
- Apache 2.0 license.
