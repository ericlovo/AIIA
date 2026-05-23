#!/usr/bin/env bash
# One-shot installer for the AIIA journal watcher LaunchAgent.
# Copies the plist into ~/Library/LaunchAgents/ and (re)loads it.
# Re-runnable — bootout-then-bootstrap makes it idempotent.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.aplora.aiia.journal-watch"
SRC_PLIST="${REPO_DIR}/scripts/${LABEL}.plist"
DEST_PLIST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
UID_NUM="$(id -u)"

if [[ ! -f "$SRC_PLIST" ]]; then
    echo "ERROR: source plist not found at $SRC_PLIST" >&2
    exit 1
fi

# Sanity-check that aiia CLI is on PATH for the kind of shell launchd will
# spawn ('bash -lc'). If `command -v aiia` fails here, it will also fail
# inside the LaunchAgent.
if ! bash -lc "command -v aiia" >/dev/null; then
    cat >&2 <<'MSG'
WARNING: `aiia` not found in `bash -lc` PATH.

The journal watcher LaunchAgent invokes `aiia journal-watch` and will
fail to start until `aiia` is reachable in that shell. Two fixes:

  - Make sure `pip install -e .` (in this repo) has been run and that the
    resulting `aiia` script is on PATH (likely $HOME/.local/bin or your
    venv's bin/).

  - If aiia lives in a venv, edit the plist's PATH EnvironmentVariable
    to put that venv's bin/ first, then re-run this installer.

Proceeding anyway — fix the PATH and re-run when ready.
MSG
fi

# Create the iCloud inbox so launchd has something to poll on first start.
INBOX_DIR="${HOME}/Library/Mobile Documents/com~apple~CloudDocs/AIIA-Inbox"
mkdir -p "${INBOX_DIR}" 2>/dev/null || true
mkdir -p "${INBOX_DIR}/archived" 2>/dev/null || true

mkdir -p "${HOME}/Library/LaunchAgents"
cp "$SRC_PLIST" "$DEST_PLIST"
echo "Installed ${DEST_PLIST}"

launchctl bootout "gui/${UID_NUM}" "$DEST_PLIST" 2>/dev/null || true
launchctl bootstrap "gui/${UID_NUM}" "$DEST_PLIST"
launchctl enable "gui/${UID_NUM}/${LABEL}"

echo "Loaded ${LABEL}"
echo
echo "Watcher polls every 5s. Check status:"
echo "  launchctl print gui/${UID_NUM}/${LABEL}"
echo
echo "Tail logs:"
echo "  tail -f /tmp/aiia-journal-watch.err"
echo
echo "Inbox folder (drop voice memos here from your phone):"
echo "  ${INBOX_DIR}"
echo
echo "Set API keys (one-time, persists across reboots):"
echo "  launchctl setenv GROQ_API_KEY 'gsk_...'"
echo "  launchctl setenv ANTHROPIC_API_KEY 'sk-ant-...'"
echo "  launchctl kickstart -k gui/${UID_NUM}/${LABEL}"
echo
echo "Disable temporarily:"
echo "  launchctl disable gui/${UID_NUM}/${LABEL}"
