#!/usr/bin/env bash
# Auto-deploy hook for AIIA (the Brain) on the Mac mini.
# Polled by a LaunchAgent (see com.aplora.aiia.deploy.plist).
# Pulls main if SHA changed, reinstalls the package, runs a post-deploy
# hook if configured (the hook is where you restart the Brain + Command
# Center launchd services — kept out of this script so service
# orchestration stays your call).
#
# Environment overrides:
#   AIIA_REPO_DIR              Path to repo (default: $HOME/code/aiia)
#   AIIA_DEPLOY_LOG            Log file (default: $HOME/.aiia/logs/aiia-deploy.log)
#   AIIA_POST_DEPLOY_HOOK      Path to a script run after a successful pull+install.
#                              Typical contents: launchctl kickstart commands
#                              to restart your Brain + Command Center services.
#   AIIA_PIP                   Pip binary (default: pip)
#   AIIA_INSTALL_EXTRAS        Pip extras spec (default: "[dev]"; pass "" to skip extras)

set -euo pipefail

REPO_DIR="${AIIA_REPO_DIR:-$HOME/code/aiia}"
LOG_FILE="${AIIA_DEPLOY_LOG:-$HOME/.aiia/logs/aiia-deploy.log}"
PIP_BIN="${AIIA_PIP:-pip}"
INSTALL_EXTRAS="${AIIA_INSTALL_EXTRAS:-[dev]}"
LOCK_FILE="/tmp/aiia-deploy.lock"

mkdir -p "$(dirname "$LOG_FILE")"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG_FILE"; }

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    exit 0
fi

cd "$REPO_DIR" || { log "FAIL: repo not at $REPO_DIR"; exit 1; }

git fetch origin main --quiet
local_sha=$(git rev-parse HEAD)
remote_sha=$(git rev-parse origin/main)

if [[ "$local_sha" == "$remote_sha" ]]; then
    exit 0
fi

log "SHA changed: ${local_sha:0:7} → ${remote_sha:0:7}. Deploying..."

if ! git pull --ff-only origin main >> "$LOG_FILE" 2>&1; then
    log "FAIL: pull --ff-only failed (local diverged from main). Manual intervention needed."
    exit 1
fi

log "Running pip install -e .${INSTALL_EXTRAS}..."
if ! "$PIP_BIN" install -e ".${INSTALL_EXTRAS}" >> "$LOG_FILE" 2>&1; then
    log "FAIL: pip install failed."
    exit 1
fi

if [[ -n "${AIIA_POST_DEPLOY_HOOK:-}" && -x "$AIIA_POST_DEPLOY_HOOK" ]]; then
    log "Running post-deploy hook: $AIIA_POST_DEPLOY_HOOK"
    if ! "$AIIA_POST_DEPLOY_HOOK" "$remote_sha" >> "$LOG_FILE" 2>&1; then
        log "FAIL: post-deploy hook exited non-zero."
        exit 1
    fi
fi

log "OK: deployed ${remote_sha:0:7}"
