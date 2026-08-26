#!/bin/bash
# SessionStart hook for Claude Code on the web.
# Ensures Node v24 + Vercel CLI (for Eve agent work) and the Python dev
# toolchain (ruff, pytest) are available before the session starts.
set -euo pipefail

# Only needed in remote (web) sessions — local machines manage their own toolchain.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

NODE_VERSION="24.19.0"
NODE_DIR="/opt/node24"

if [ ! -x "$NODE_DIR/bin/node" ] || [ "$("$NODE_DIR/bin/node" -v)" != "v$NODE_VERSION" ]; then
  curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" -o /tmp/node24.tar.xz
  mkdir -p "$NODE_DIR"
  tar -xJf /tmp/node24.tar.xz -C "$NODE_DIR" --strip-components=1
  rm -f /tmp/node24.tar.xz
fi

# Node 24 first on PATH for the rest of the session.
echo "export PATH=\"$NODE_DIR/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
export PATH="$NODE_DIR/bin:$PATH"

if ! command -v vercel >/dev/null 2>&1; then
  npm install -g vercel
fi

# Python side: editable install + dev deps so ruff/pytest run in-session.
# The base image ships debian-managed packages (PyYAML, PyJWT, ...) that pip
# can't uninstall (no RECORD file). When the install trips on one, reinstall
# that package pip-owned and retry.
for _ in 1 2 3 4 5 6 7 8; do
  if out=$(pip install -e "$CLAUDE_PROJECT_DIR[dev]" --quiet 2>&1); then
    break
  fi
  pkg=$(printf '%s' "$out" | sed -n 's/.*Cannot uninstall \([^ ,]*\).*/\1/p' | head -1)
  if [ -z "$pkg" ]; then
    printf '%s\n' "$out" >&2
    exit 1
  fi
  pip install --ignore-installed "$pkg" --quiet
done

echo "session-start: node $(node -v), npm $(npm -v), vercel $(vercel --version 2>/dev/null | tail -1), $(ruff --version)"
