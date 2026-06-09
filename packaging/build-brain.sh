#!/usr/bin/env bash
# Build the standalone AIIA Brain sidecar binary for aiia-console.
#
# Run on the Mac that matches the target arch (Apple Silicon mini → arm64).
# Produces: dist/aiia-brain-<target-triple>, ready to copy into
# aiia-console/src-tauri/binaries/ for Tauri's bundle.externalBin.
#
# Usage:
#   ./packaging/build-brain.sh
#   AIIA_CONSOLE_DIR=~/aiia-console ./packaging/build-brain.sh   # also copies it over

set -euo pipefail
cd "$(dirname "$0")/.."

command -v pyinstaller >/dev/null || {
    echo "pyinstaller not found — run: pip install pyinstaller" >&2
    exit 1
}

# Tauri externalBin naming: <name>-<rust target triple>
arch="$(uname -m)"
case "$(uname -s)-$arch" in
    Darwin-arm64)  triple="aarch64-apple-darwin" ;;
    Darwin-x86_64) triple="x86_64-apple-darwin" ;;
    Linux-x86_64)  triple="x86_64-unknown-linux-gnu" ;;
    *) echo "unsupported platform: $(uname -s)-$arch" >&2; exit 1 ;;
esac

echo "→ building aiia-brain for $triple"
pyinstaller --noconfirm packaging/aiia-brain.spec

out="dist/aiia-brain-$triple"
mv -f dist/aiia-brain "$out"
echo "→ built: $out ($(du -h "$out" | cut -f1))"

# Smoke test: the frozen binary must at least start and answer /health.
"$out" &
brain_pid=$!
trap 'kill "$brain_pid" 2>/dev/null || true' EXIT
for _ in $(seq 1 30); do
    if curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1; then
        echo "→ smoke test passed: /health responding"
        break
    fi
    sleep 1
done
kill "$brain_pid" 2>/dev/null || true
trap - EXIT

if [ -n "${AIIA_CONSOLE_DIR:-}" ]; then
    mkdir -p "$AIIA_CONSOLE_DIR/src-tauri/binaries"
    cp "$out" "$AIIA_CONSOLE_DIR/src-tauri/binaries/"
    echo "→ copied to $AIIA_CONSOLE_DIR/src-tauri/binaries/"
fi
