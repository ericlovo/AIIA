#!/bin/bash
# install_obsidian_bridge.sh — Wire up the Obsidian Bridge on the Mac Mini.
#
# What this does:
#   1. Writes com.aiia.obsidiansync.plist to ~/Library/LaunchAgents/
#      (runs obsidian_bridge.py nightly at 11pm)
#   2. Loads the launchd agent
#   3. Adds `vault` alias to .zshrc if not already present
#
# Usage (run once on the Mini):
#   bash local_brain/local_brain/pilot/install_obsidian_bridge.sh
#
# After running, test with:
#   vault --dry-run

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PLIST_NAME="com.aiia.obsidiansync"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"
LOG_DIR="$HOME/aiia-local-brain/logs/vault"
PYTHON="$(which python3)"

echo "=== AIIA Obsidian Bridge Installer ==="
echo "Repo:     $REPO_DIR"
echo "Python:   $PYTHON"
echo "Vault:    $HOME/AIIAVault"
echo ""

# --- 1. Create log directory ---
mkdir -p "$LOG_DIR"
echo "[1/3] Log directory: $LOG_DIR"

# --- 2. Write launchd plist ---
cat > "$PLIST_PATH" << PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_NAME}</string>

    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON}</string>
        <string>-m</string>
        <string>local_brain.local_brain.scripts.obsidian_bridge</string>
    </array>

    <key>WorkingDirectory</key>
    <string>${REPO_DIR}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>${REPO_DIR}</string>
        <key>HOME</key>
        <string>${HOME}</string>
    </dict>

    <!-- Nightly at 11:00pm -->
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>23</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>${LOG_DIR}/vault.log</string>

    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/vault.log</string>

    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
PLIST_EOF

echo "[2/3] Plist written: $PLIST_PATH"

# Unload if already loaded (ignore errors)
launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load "$PLIST_PATH"
echo "      Loaded with launchctl"

# --- 3. Add shell aliases to .zshrc ---
ZSHRC="$HOME/.zshrc"
ALIAS_MARKER="# aiia-obsidian-bridge"

if grep -q "$ALIAS_MARKER" "$ZSHRC" 2>/dev/null; then
    echo "[3/3] Aliases already in .zshrc — skipping"
else
    cat >> "$ZSHRC" << 'ZSHRC_EOF'

# aiia-obsidian-bridge
alias vault='python3 -m local_brain.local_brain.scripts.obsidian_bridge'
alias vault-dry='python3 -m local_brain.local_brain.scripts.obsidian_bridge --dry-run'
ZSHRC_EOF
    echo "[3/3] Added 'vault' and 'vault-dry' aliases to .zshrc"
fi

echo ""
echo "=== Done ==="
echo ""
echo "Test now:   cd $REPO_DIR && python3 -m local_brain.local_brain.scripts.obsidian_bridge --dry-run"
echo "Run now:    cd $REPO_DIR && python3 -m local_brain.local_brain.scripts.obsidian_bridge"
echo "Logs:       $LOG_DIR/vault.log"
echo "Schedule:   Nightly at 11:00pm"
echo ""
echo "Vault files written:"
echo "  ~/AIIAVault/30-Decisions/AIIA-Decisions.md"
echo "  ~/AIIAVault/40-Sessions/AIIA-Sessions.md"
echo "  ~/AIIAVault/50-Stories/AIIA-Backlog.md"
echo "  ~/AIIAVault/80-Resources/AIIA-Patterns.md"
echo "  ~/AIIAVault/80-Resources/AIIA-Lessons.md"
