#!/usr/bin/env bash
# Bootstrap the Writ source checkout the workshop sandbox runs against.
#
# Writ's harness is exercised against the *Writ repo itself* (self-dogfooding),
# not against AIIA. This clones it to a sibling path and installs the one
# dependency the unit suite needs.
#
# Usage: bash workshop/bootstrap.sh [target-dir]

set -euo pipefail

WRIT_DIR="${1:-${WRIT_DIR:-$HOME/sellke/writ}}"
WRIT_REPO="https://github.com/sellke/writ"

echo "Writ checkout: $WRIT_DIR"

if [ -d "$WRIT_DIR/.git" ]; then
  echo "  already present ($(git -C "$WRIT_DIR" rev-parse --short HEAD))"
else
  mkdir -p "$(dirname "$WRIT_DIR")"
  # NOT --depth 1. The archive-dogfood check reads real commit history and
  # reports a false FAIL on a shallow clone. See findings/01.
  git clone "$WRIT_REPO" "$WRIT_DIR"
fi

# A pre-existing shallow clone fails the same way — repair it in place.
if [ -f "$WRIT_DIR/.git/shallow" ]; then
  echo "  shallow clone detected — unshallowing (archive-dogfood needs full history)"
  git -C "$WRIT_DIR" fetch --unshallow
fi

echo "  commits: $(git -C "$WRIT_DIR" rev-list --count HEAD)"

python3 -m pytest --version >/dev/null 2>&1 || {
  echo "  installing pytest (required by scripts/tests/)"
  python3 -m pip install -q pytest
}

echo
echo "Ready. Next: bash workshop/run-harness.sh"
