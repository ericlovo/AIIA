#!/usr/bin/env bash
# One-shot installer for the AIIA nightly research LaunchAgent.
# Copies the plist into ~/Library/LaunchAgents/ and (re)loads it.
# Re-runnable — bootout-then-bootstrap makes it idempotent.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.aplora.aiia.research"
SRC_PLIST="${REPO_DIR}/scripts/${LABEL}.plist"
DEST_PLIST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
UID_NUM="$(id -u)"

if [[ ! -f "$SRC_PLIST" ]]; then
    echo "ERROR: source plist not found at $SRC_PLIST" >&2
    exit 1
fi

mkdir -p "${HOME}/Library/LaunchAgents"
cp "$SRC_PLIST" "$DEST_PLIST"
echo "Installed ${DEST_PLIST}"

launchctl bootout "gui/${UID_NUM}" "$DEST_PLIST" 2>/dev/null || true
launchctl bootstrap "gui/${UID_NUM}" "$DEST_PLIST"
launchctl enable "gui/${UID_NUM}/${LABEL}"

echo "Loaded ${LABEL}"
echo
echo "Runs nightly at 03:30. The loop is armed via AIIA_AUTONOMY_LEVEL=phase2"
echo "+ AIIA_RESEARCH_ENABLED=true in the plist."
echo
echo "Check status:"
echo "  launchctl print gui/${UID_NUM}/${LABEL}"
echo
echo "Run it once now (kickstart):"
echo "  launchctl kickstart -k gui/${UID_NUM}/${LABEL}"
echo
echo "Or run a manual cycle without the agent (bypasses the gate):"
echo "  cd \"$REPO_DIR\" && python3 -m local_brain.scripts.research_runner --force"
echo
echo "Tail logs:"
echo "  tail -f \$HOME/.aiia/logs/research/research.log"
echo
echo "Disable temporarily (keeps the agent loaded):"
echo "  launchctl disable gui/${UID_NUM}/${LABEL}"
