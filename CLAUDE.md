# CLAUDE.md — Agent Context for AIIA

This file orients a coding agent landing in this repo. Read README.md for product framing; this doc is about *how the code is laid out and how to work with it*.

## What this repo is

`ericlovo/aiia` — the **Brain**: a Python/FastAPI service that holds persistent memory (decisions, patterns, sessions), routes between local LLMs (Ollama) and remote ones (Anthropic, Google), and exposes itself to AI tools (Claude Code, Cursor) via MCP.

The companion repo `ericlovo/aiia-console` is the Tauri desktop app that talks to this Brain at `localhost:8100` (with SQLite fallback if Brain is offline).

## Layout

```
local_brain/                  # The Python package — everything lives here
├── __version__.py            # Single source of version truth
├── cli.py                    # `aiia` CLI (installed via pyproject scripts)
├── local_api.py              # FastAPI Brain API on :8100
├── config.py                 # Settings via pydantic + env
├── ollama_client.py          # Local LLM dispatch
├── mcp_server.py             # MCP tools surface (aiia_ask, aiia_remember, ...)
├── vault_paths.py            # Where memory lives on disk
├── smart_conductor.py        # Cross-component coordination
│
├── command_center/           # :8200 dashboard server + WebSocket tasks
├── eq_brain/                 # Memory + knowledge layer (ChromaDB-backed)
├── execution/                # Safety-gated action execution (3-tier)
│   ├── safety.py             # Pre-execution gate
│   ├── executor.py           # The tiered executor
│   ├── strategies.py         # AUTO / SUPERVISED / GATED tiers
│   ├── verification.py       # Post-execution verify
│   └── git_ops.py            # Git-aware safe ops
├── autonomy/                 # Scheduled autonomous loops (Phase 2)
├── research/                 # Autonomous research loop — topics, prompt profiles (general/erdos), Erdős problem factory
├── a2a/                      # Google A2A protocol module
├── story_runner/             # Story-driven workflow engine
├── scripts/                  # CLI runners + indexers + reporters
└── tests/                    # See "Testing" below

dashboard/                    # Separate Vite frontend (no Tailwind yet — see design/TODO.md)
design/                       # Source-of-truth design tokens (aiia-console is canonical)
docs/                         # Architecture + design docs
docker-compose.yml            # Brain + Ollama + Command Center
```

## Dev commands

```bash
# Install editable + dev deps
pip install -e ".[dev]"

# Run the Brain API (:8100)
uvicorn local_brain.main:app --port 8100
# or
python -m local_brain.local_api

# Run the Command Center dashboard (:8200)
python -m local_brain.command_center.server

# CLI (installed by pyproject)
aiia                          # wordmark + quick-start
aiia next                     # top stories from Command Center
aiia ask "..."                # query Brain
aiia memory add "..." -c decisions
aiia status                   # Brain + CC health
aiia briefing                 # morning briefing CLI
```

## Linting + formatting

```bash
ruff check local_brain/        # lint (enforced as of v0.5.x, see PR #19)
ruff format local_brain/       # format
ruff check --fix local_brain/  # auto-fix what's safe
```

Config: `[tool.ruff]` in `pyproject.toml`. Selected rule families: E, F, I, UP, B, SIM. There's a documented backlog of ignored rules with site counts — drain them in focused follow-up PRs (one rule per PR is ideal).

## Testing

Tests live in `local_brain/tests/`. The current setup runs `pytest --collect-only` in CI because most tests are integration tests that need a live Ollama, Brain API, and Command Center.

The Track 2 work in the 24h plan is wiring up a `conftest.py` with fixtures that mock Ollama + the Brain HTTP layer, so the CI can flip from `--collect-only` to a real `pytest` run.

```bash
pytest --collect-only local_brain/tests/  # what CI runs today
pytest local_brain/tests/                 # once T4 lands
```

## Conventions

### Commits

Conventional Commits style — examples from the log:

```
feat: unified aiia CLI (v0.5.0 milestone) (#17)
docs(readme): mirror-first rewrite for v0.5.0 launch (#15)
chore: remove supermemory integration (#13)
ci: enforce ruff lint + format
design: add /design tokens + README
```

Prefixes: `feat`, `fix`, `docs`, `chore`, `ci`, `refactor`, `test`, `design`, `release`.

### Branches

- `main` — protected (or about to be)
- `claude/<slug>-sjym0` — agent-created working branches
- `feat/<slug>` — feature branches
- `docs/<slug>` — doc-only branches
- `v<version>-dev.<n>-<slug>` — release-train branches

### Wordmark

`⌬ AIIA` — appears in README headers, CLI startup output (`aiia` no-args), and console title bar. One glyph everywhere.

### Sanitization

CI has a `sanitization-guard` job that fails the build if banned proprietary strings reappear. List + exclusions in `.github/workflows/ci.yml`. If new code legitimately needs a banned reference (e.g. CHANGELOG documenting a removal), add it to the `--exclude` list with a comment.

## Security model

Real security infrastructure lives in `local_brain/execution/` — not aspirational. Three-tier execution model (AUTO / SUPERVISED / GATED) implemented in `executor.py` + `strategies.py`. Safety gate runs pre-execution from `safety.py`; verification runs post-execution from `verification.py`. See `SECURITY.md` for the reporting path and threat model.

The "scanner suite" referenced in older README copy was always the in-code safety/verification logic — not a separate set of tools. PR S4 in the 24h plan documents this precisely (and may add `docs/SECURITY-ARCHITECTURE.md`).

## Where things connect

| Surface | Module | Port |
|---|---|---|
| Brain API | `local_brain/local_api.py` | :8100 |
| Command Center | `local_brain/command_center/server.py` | :8200 |
| MCP server | `local_brain/mcp_server.py` | (stdio, invoked by Claude Code) |
| CLI | `local_brain/cli.py` | (PATH) |

aiia-console (Tauri) hits `:8100` for Brain queries. Browser dashboard in `dashboard/` hits `:8200`. Claude Code/MCP-aware tools invoke the MCP server directly.

## Common gotchas

- **`from local_brain import ...` requires editable install.** Run `pip install -e .` after a fresh clone.
- **Tests can't run without Ollama + Brain locally** (until T4 lands mocks). Use `--collect-only` for a syntax-only check.
- **Ruff target is `py310`.** Don't add Python 3.11-only syntax outside guarded blocks.
- **`v0.5.0-dev.*` branches on the remote** may or may not be stale — surface to the user, don't auto-delete.
<!-- writ:start -->
# Writ

You are **Writ** — a methodical AI development partner. You organize all work in `.writ/` folders.

## Commands

Run Writ commands by reading the command file and following its workflow:

| Command | File | Purpose |
|---------|------|---------|
| `/plan-product` | `.claude/commands/plan-product.md` | Product planning and roadmap |
| `/create-spec` | `.claude/commands/create-spec.md` | Contract-first feature specification |
| `/implement-spec` | `.claude/commands/implement-spec.md` | Orchestrate full spec implementation |
| `/implement-story` | `.claude/commands/implement-story.md` | Full SDLC via multi-agent pipeline |
| `/prototype` | `.claude/commands/prototype.md` | Lightweight execution for small changes |
| `/verify-spec` | `.claude/commands/verify-spec.md` | 8-check comprehensive validation |
| `/create-uat-plan` | `.claude/commands/create-uat-plan.md` | Generate UAT test plan from spec |
| `/review` | `.claude/commands/review.md` | Pre-landing code review |
| `/refactor` | `.claude/commands/refactor.md` | Scoped, verified refactoring |
| `/ship` | `.claude/commands/ship.md` | Branch to merged PR workflow |
| `/release` | `.claude/commands/release.md` | Changelog, version bump, git tag |
| `/security-audit` | `.claude/commands/security-audit.md` | Full security audit with auto-fix |
| `/create-adr` | `.claude/commands/create-adr.md` | Architecture Decision Records |
| `/create-issue` | `.claude/commands/create-issue.md` | Quick issue capture |
| `/research` | `.claude/commands/research.md` | Systematic research |
| `/status` | `.claude/commands/status.md` | Project status report |
| `/initialize` | `.claude/commands/initialize.md` | Project setup |
| `/retro` | `.claude/commands/retro.md` | Git-based retrospective |
| `/refresh-command` | `.claude/commands/refresh-command.md` | Self-improving command refinement |
| `/assess-spec` | `.claude/commands/assess-spec.md` | Spec assessment |
| `/edit-spec` | `.claude/commands/edit-spec.md` | Edit existing spec |
| `/design` | `.claude/commands/design.md` | UI/UX design workflows |
| `/new-command` | `.claude/commands/new-command.md` | Create new Writ command |
| `/migrate` | `.claude/commands/migrate.md` | Migration workflows |

## Agents

Writ uses Claude Code's native subagent system. Agents are defined in `.claude/agents/`:

- `writ-architect.md` — Pre-implementation design review (read-only, worktree)
- `writ-coder.md` — TDD implementation (worktree isolation)
- `writ-reviewer.md` — Quality + security gate (read-only, persistent memory)
- `writ-tester.md` — Test execution + coverage enforcement
- `writ-documenter.md` — Framework-adaptive documentation
- `writ-story-gen.md` — Parallel story file creation (fast model, worktree)

## Pipeline

```
/plan-product → /create-spec → /implement-spec → /verify-spec → /release
```

## Principles

- **Contract-first**: Establish agreement before creating files
- **TDD**: Tests first, then implementation
- **Challenge assumptions**: Push back on bad ideas with evidence
- **Commit incrementally**: Small commits, not big bangs
<!-- writ:end -->
