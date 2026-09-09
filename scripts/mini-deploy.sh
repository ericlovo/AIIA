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
#   AIIA_EXPECT_AIRGAP         Post-deploy assertion on air-gap state (1/true or
#                              0/false). Unset = no check. See "Air-gap drift
#                              check" below.
#   AIIA_HEALTH_URL            Brain health endpoint for that check
#                              (default: http://localhost:8100/health)
#
# NOTE: this script does NOT set AIIA_AIRGAP. It only deploys code — the Brain
# runs as a separate launchd service, so an `export` here would apply to this
# script and its hook, never to the Brain process. Air-gap must be set in the
# Brain service's launchd plist <EnvironmentVariables> (or .env under Docker,
# which is the only path that reads .env — nothing calls load_dotenv()).
# What this script can do is notice when the setting has gone missing, which is
# what AIIA_EXPECT_AIRGAP is for. Full runbook: docs/AIRGAP.md

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

# Air-gap drift check — catches the case where a rebuild, a reprovisioned
# launchd plist, or a hand-edited .env silently drops AIIA_AIRGAP and the Brain
# comes back up with cloud egress allowed. Advisory only: the deploy itself
# already succeeded, and a retry loop won't fix an environment misconfiguration,
# so a mismatch logs FAIL and exits 0. Grep the log for "airgap" to audit.
if [[ -n "${AIIA_EXPECT_AIRGAP:-}" ]]; then
    health_url="${AIIA_HEALTH_URL:-http://localhost:8100/health}"
    # tr, not ${x,,} — macOS ships bash 3.2, where that expansion is a syntax error.
    expect_lc=$(printf '%s' "$AIIA_EXPECT_AIRGAP" | tr '[:upper:]' '[:lower:]')
    case "$expect_lc" in
        1|true)  want="True" ;;
        0|false) want="False" ;;
        *) want="" ; log "FAIL: AIIA_EXPECT_AIRGAP='${AIIA_EXPECT_AIRGAP}' is not 1/true/0/false" ;;
    esac

    if [[ -n "$want" ]]; then
        health=$(curl -fsS --max-time 10 "$health_url" 2>/dev/null || true)
        if [[ -z "$health" ]]; then
            log "FAIL: airgap check — Brain unreachable at $health_url (is it restarted?)"
        else
            got=$(printf '%s' "$health" \
                | python3 -c "import json,sys; print(json.load(sys.stdin)['airgap']['enabled'])" \
                2>/dev/null || true)
            if [[ -z "$got" ]]; then
                log "FAIL: airgap check — no airgap.enabled in $health_url response"
            elif [[ "$got" != "$want" ]]; then
                log "FAIL: airgap DRIFT — expected enabled=$want, Brain reports enabled=$got." \
                    "Check the Brain launchd plist <EnvironmentVariables> for AIIA_AIRGAP."
            else
                log "OK: airgap check — enabled=$got (as expected)"
            fi
        fi
    fi
fi

log "OK: deployed ${remote_sha:0:7}"
