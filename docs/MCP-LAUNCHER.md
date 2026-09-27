# Portable MCP launcher

Start Claude Code from the repository root. The checked-in `.mcp.json` invokes
`sh ./scripts/aiia-mcp`; it contains no personal paths and needs no executable-bit
setup. The launcher resolves the repository from its own location, changes to it,
and uses `.venv/bin/python3` when executable, otherwise `python3` on PATH.

Install dependencies first, for example:

```sh
python3 -m venv .venv
.venv/bin/python3 -m pip install -e .
claude mcp list
```

After approving the project MCP configuration in Claude Code, `claude mcp list`
should report `aiia` connected. A connected MCP transport does not establish that
the Brain API is available; tools that use it still need the Brain service.

The launcher checks the local_brain package and FastMCP import before starting.
An unusable repo virtualenv fails with a one-line installation instruction rather
than silently switching to an unrelated Python. Other startup/import errors can
still originate in the MCP server itself.

EQ_BRAIN_DATA_DIR defaults to `$HOME/.aiia/eq_data` when unset or empty; an explicit
value is preserved. HOME is required only when that default is needed. Existing
environment settings, including AIIA_AIRGAP, are inherited unchanged. The launcher
does not source .env, print keys, install dependencies, or start a network listener.
It execs the existing `python -m local_brain.mcp_server` stdio entry point.

For another MCP client, configure its working directory as the repo root or use
an absolute path to scripts/aiia-mcp as the argument to sh. Relative config paths
are resolved by the client, not by the launcher.
