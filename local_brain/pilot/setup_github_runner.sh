#!/bin/bash
# ─────────────────────────────────────────────────────────────
# GitHub Actions Self-Hosted Runner — Mac Mini M4 Setup
#
# Replaces GitHub-hosted ubuntu runners with the Mini.
# $0/month, faster builds, unlimited minutes.
#
# Prerequisites (already installed via setup_mac_pilot.sh):
#   - Python 3.11+, Node 20+, npm
#   - ruff, mypy, bandit, semgrep
#   - trufflehog, trivy, shellcheck, hadolint
#   - gh CLI (authenticated)
#
# Usage:
#   chmod +x setup_github_runner.sh
#   ./setup_github_runner.sh
#
# Management (after install):
#   brain runner status    — check runner health
#   brain runner restart   — restart runner service
#   brain runner logs      — view runner logs
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config ──────────────────────────────────────────────────

RUNNER_DIR="$HOME/aiia-local-brain/github-runner"
REPO="ericlovo/AIIA"
RUNNER_NAME="aiia-mini-m4"
LABELS="self-hosted,macOS,ARM64,aiia-mini"
WORK_DIR="$RUNNER_DIR/_work"
PLIST_NAME="com.aiia.github-runner"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"
LOG_DIR="$HOME/aiia-local-brain/logs/runner"

# ── Colors ──────────────────────────────────────────────────

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

step() { echo -e "\n${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
fail() { echo -e "${RED}[✗]${NC} $1"; exit 1; }
info() { echo -e "${CYAN}[i]${NC} $1"; }

echo "╔═══════════════════════════════════════════════════╗"
echo "║  GitHub Actions Self-Hosted Runner Setup           ║"
echo "║  Mac Mini M4 — AIIA CI Infrastructure            ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""

# ── Preflight Checks ───────────────────────────────────────

echo "Checking prerequisites..."

# macOS + Apple Silicon
[[ "$(uname)" == "Darwin" ]] || fail "macOS only"
[[ "$(uname -m)" == "arm64" ]] || warn "Not Apple Silicon — builds will be slower"

# gh CLI authenticated
if ! gh auth status &>/dev/null; then
    fail "gh CLI not authenticated. Run: gh auth login"
fi
step "gh CLI authenticated"

# Required tools
MISSING=()
for tool in python3 node npm ruff mypy bandit trufflehog hadolint shellcheck; do
    if ! command -v "$tool" &>/dev/null; then
        MISSING+=("$tool")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    warn "Missing tools: ${MISSING[*]}"
    echo "  Install with: brew install ${MISSING[*]}"
    echo "  Or run setup_mac_pilot.sh first"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
else
    step "All CI tools found: python3, node, npm, ruff, mypy, bandit, trufflehog, hadolint, shellcheck"
fi

# Python + Node versions
PYTHON_VER=$(python3 --version 2>&1 | awk '{print $2}')
NODE_VER=$(node --version 2>&1)
step "Python ${PYTHON_VER}, Node ${NODE_VER}"

# ── Get Registration Token ─────────────────────────────────

info "Requesting runner registration token from GitHub..."
REG_TOKEN=$(gh api "repos/${REPO}/actions/runners/registration-token" -X POST --jq '.token' 2>&1)

if [[ -z "$REG_TOKEN" || "$REG_TOKEN" == *"error"* ]]; then
    fail "Could not get registration token. Check repo permissions."
fi
step "Registration token obtained"

# ── Download Runner ────────────────────────────────────────

mkdir -p "$RUNNER_DIR" "$LOG_DIR"

# Get latest runner version
info "Fetching latest runner release..."
RUNNER_VERSION=$(gh api repos/actions/runner/releases/latest --jq '.tag_name' | sed 's/^v//')
RUNNER_URL="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-osx-arm64-${RUNNER_VERSION}.tar.gz"
RUNNER_TAR="$RUNNER_DIR/actions-runner-osx-arm64-${RUNNER_VERSION}.tar.gz"

if [ -f "$RUNNER_DIR/.runner" ]; then
    warn "Runner already configured at $RUNNER_DIR"
    read -p "Re-install? This will stop the current runner. (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Stop existing service
        launchctl bootout "gui/$(id -u)/${PLIST_NAME}" 2>/dev/null || true
        # Remove existing config
        cd "$RUNNER_DIR"
        ./config.sh remove --token "$REG_TOKEN" 2>/dev/null || true
    else
        info "Skipping download — using existing installation"
        SKIP_DOWNLOAD=true
    fi
fi

if [ "${SKIP_DOWNLOAD:-}" != "true" ]; then
    step "Downloading runner v${RUNNER_VERSION} (ARM64 macOS)..."
    curl -sL "$RUNNER_URL" -o "$RUNNER_TAR"

    step "Extracting..."
    cd "$RUNNER_DIR"
    tar xzf "$RUNNER_TAR"
    rm -f "$RUNNER_TAR"

    # ── Configure Runner ───────────────────────────────────

    step "Configuring runner..."
    ./config.sh \
        --url "https://github.com/${REPO}" \
        --token "$REG_TOKEN" \
        --name "$RUNNER_NAME" \
        --labels "$LABELS" \
        --work "$WORK_DIR" \
        --replace \
        --unattended

    step "Runner configured: ${RUNNER_NAME}"
fi

# ── Create launchd Service ─────────────────────────────────

info "Installing launchd service..."

cat > "$PLIST_PATH" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_NAME}</string>

    <key>ProgramArguments</key>
    <array>
        <string>${RUNNER_DIR}/run.sh</string>
    </array>

    <key>WorkingDirectory</key>
    <string>${RUNNER_DIR}</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>ThrottleInterval</key>
    <integer>10</integer>

    <key>StandardOutPath</key>
    <string>${LOG_DIR}/runner.log</string>

    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/runner.err</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>HOME</key>
        <string>${HOME}</string>
        <key>LANG</key>
        <string>en_US.UTF-8</string>
    </dict>
</dict>
</plist>
PLIST

step "launchd plist created: ${PLIST_PATH}"

# ── Start Runner ───────────────────────────────────────────

info "Starting runner service..."
launchctl bootout "gui/$(id -u)/${PLIST_NAME}" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_PATH"

# Wait for startup
sleep 3

# Verify
if launchctl print "gui/$(id -u)/${PLIST_NAME}" &>/dev/null; then
    step "Runner service started successfully"
else
    warn "Runner service may not have started. Check: brain runner logs"
fi

# ── Verify with GitHub ─────────────────────────────────────

info "Verifying runner is online..."
sleep 5

RUNNER_STATUS=$(gh api "repos/${REPO}/actions/runners" --jq ".runners[] | select(.name == \"${RUNNER_NAME}\") | .status" 2>/dev/null || echo "unknown")

if [ "$RUNNER_STATUS" == "online" ]; then
    step "Runner is ONLINE and ready to accept jobs"
else
    warn "Runner status: ${RUNNER_STATUS}"
    info "It may take a moment to connect. Check: gh api repos/${REPO}/actions/runners --jq '.runners[]'"
fi

# ── Summary ────────────────────────────────────────────────

echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║  Setup Complete                                    ║"
echo "╠═══════════════════════════════════════════════════╣"
echo "║                                                    ║"
echo "║  Runner: ${RUNNER_NAME}                            ║"
echo "║  Labels: ${LABELS}                                 ║"
echo "║  Work:   ${WORK_DIR}                               ║"
echo "║  Logs:   ${LOG_DIR}/runner.log                     ║"
echo "║                                                    ║"
echo "║  Management:                                       ║"
echo "║    brain runner status   — health check            ║"
echo "║    brain runner restart  — restart service         ║"
echo "║    brain runner logs     — view logs               ║"
echo "║    brain runner stop     — stop service            ║"
echo "║                                                    ║"
echo "║  Cost: \$0/month (was ~\$24/3000 min)               ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""
info "Push a commit to trigger CI and verify the runner picks up jobs."
