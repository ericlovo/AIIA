# Contributing to AIIA

Thank you for your interest in contributing to AIIA! This document provides guidelines and information for contributors.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/AIIA.git
   cd AIIA
   ```
3. **Create a branch** for your work:
   ```bash
   git checkout -b feat/your-feature-name
   ```
4. **Set up your environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Development Setup

### Prerequisites

- **Python 3.12+**
- **Ollama** — Install from [ollama.com](https://ollama.com) and pull a model:
  ```bash
  ollama pull llama3.1:8b-instruct-q8_0
  ```
- **ChromaDB** — Installed via requirements.txt (runs embedded, no separate server needed)

### Running Locally

```bash
# Start the Local Brain API
python -m local_brain.local_api

# In another terminal, start the Command Center
python -m local_brain.command_center.server
```

## Code Style

- **Python:** Follow PEP 8. We use `black` for formatting and `ruff` for linting.
- **Line length:** 88 characters (black default)
- **Imports:** Use `isort` with the `black` profile
- **Type hints:** Encouraged on public APIs

```bash
# Format code
black --line-length 88 .
isort --profile black .

# Lint
ruff check .
```

## Project Structure

```
local_brain/
├── __init__.py              # LocalBrain class
├── config.py                # Configuration (env vars, model assignments)
├── local_api.py             # FastAPI endpoints (port 8100)
├── ollama_client.py         # Ollama HTTP client
├── smart_conductor.py       # LLM-powered query routing
├── mcp_server.py            # MCP tool interface for Claude Code
├── command_center/          # Web dashboard (port 8200)
├── eq_brain/                # Memory, knowledge, sync, reasoning
├── execution/               # Safety-gated task execution engine
├── scripts/                 # Automation scripts (reports, scans, sync)
├── pilot/                   # Setup and deployment helpers
└── tests/                   # Test suite
```

## Making Changes

### Adding a New Feature

1. Check if there's an existing issue or discussion
2. Write your code with appropriate docstrings and type hints
3. Add tests if applicable
4. Update documentation (README, docstrings) as needed
5. Ensure all existing tests pass

### Modifying the Memory System

The 9-category memory system is a core architectural decision. If you're modifying memory categories, sync tiers, or decay policies, please open an issue for discussion first.

### Adding a New MCP Tool

MCP tools are defined in `local_brain/mcp_server.py`. Each tool should:
- Have a clear, descriptive name and docstring
- Handle errors gracefully (return error info, don't crash)
- Respect the AIIA API contract

### Working with the Execution Engine

The execution engine (`local_brain/execution/`) uses safety tiers:
- **AUTO:** Safe operations that can run without approval
- **SUPERVISED:** Logged and monitored, notify on completion
- **GATED:** Requires explicit user approval before executing

New execution tasks should default to SUPERVISED or GATED until proven safe.

## Pull Request Process

1. **Update your branch** with the latest `main`:
   ```bash
   git fetch origin
   git rebase origin/main
   ```
2. **Push your branch** and open a PR
3. **Describe your changes** clearly in the PR description:
   - What does this change do?
   - Why is it needed?
   - How was it tested?
4. **Link any related issues**

### PR Title Convention

Use conventional commit style:
- `feat: add new memory category for bookmarks`
- `fix: correct ChromaDB query limit handling`
- `docs: update API endpoint documentation`
- `refactor: simplify conductor routing logic`
- `chore: update dependencies`

## Reporting Issues

When opening an issue, please include:
- **Environment:** OS, Python version, Ollama version
- **Steps to reproduce** the problem
- **Expected behavior** vs actual behavior
- **Logs** if applicable (redact any API keys or sensitive data)

## Architecture Decisions

Major architectural changes should be discussed in an issue before implementation. The project follows these principles:

- **Local-first:** Ollama handles as much as possible for $0
- **Graceful degradation:** Cloud services (Supermemory, Anthropic, Google) are optional — AIIA works without them
- **Configuration over code:** Use env vars and config files, not hardcoded values
- **Safety by default:** Execution engine defaults to requiring approval

## License

By contributing to AIIA, you agree that your contributions will be licensed under the Apache License 2.0.
