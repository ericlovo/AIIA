#!/bin/bash
# ─────────────────────────────────────────────────────────────
# GitHub Actions Runner Control
#
# Standalone management script for the self-hosted runner.
# Can be called directly or integrated into the brain CLI.
#
# Usage:
#   ./runner_ctl.sh status   — runner health + GitHub API check
#   ./runner_ctl.sh start    — start the runner service
#   ./runner_ctl.sh stop     — stop the runner service
#   ./runner_ctl.sh restart  — restart the runner service
#   ./runner_ctl.sh logs     — tail runner logs
#   ./runner_ctl.sh logs -f  — follow runner logs
#
# To integrate into brain CLI, add to your brain script:
#   runner) source /path/to/runner_ctl.sh && runner_$2 ;;
# ─────────────────────────────────────────────────────────────

PLIST_NAME="com.aiia.github-runner"
RUNNER_DIR="$HOME/aiia-local-brain/github-runner"
LOG_DIR="$HOME/aiia-local-brain/logs/runner"
REPO="ericlovo/AIIA"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

runner_status() {
    echo -e "${CYAN}GitHub Actions Runner${NC}"
    echo "────────────────────────────────"

    # launchd status
    if launchctl print "gui/$(id -u)/${PLIST_NAME}" &>/dev/null; then
        PID=$(launchctl print "gui/$(id -u)/${PLIST_NAME}" 2>/dev/null | grep "pid =" | awk '{print $3}')
        echo -e "  Service:  ${GREEN}running${NC} (PID: ${PID:-?})"
    else
        echo -e "  Service:  ${RED}stopped${NC}"
    fi

    # GitHub API status
    if command -v gh &>/dev/null; then
        RUNNER_INFO=$(gh api "repos/${REPO}/actions/runners" --jq '.runners[] | select(.name == "aiia-mini-m4") | "\(.status) (\(.busy))"' 2>/dev/null || echo "unknown")
        if [[ "$RUNNER_INFO" == *"online"* ]]; then
            BUSY=$(echo "$RUNNER_INFO" | grep -o "(.*)")
            if [[ "$BUSY" == "(true)" ]]; then
                echo -e "  GitHub:   ${YELLOW}online — running a job${NC}"
            else
                echo -e "  GitHub:   ${GREEN}online — idle${NC}"
            fi
        else
            echo -e "  GitHub:   ${RED}${RUNNER_INFO}${NC}"
        fi
    fi

    # Runner directory
    if [ -f "$RUNNER_DIR/.runner" ]; then
        echo -e "  Install:  ${GREEN}configured${NC} at ${RUNNER_DIR}"
    else
        echo -e "  Install:  ${RED}not configured${NC}"
    fi

    # Log file
    if [ -f "$LOG_DIR/runner.log" ]; then
        LOG_SIZE=$(du -sh "$LOG_DIR/runner.log" 2>/dev/null | cut -f1)
        LAST_LINE=$(tail -1 "$LOG_DIR/runner.log" 2>/dev/null | cut -c1-80)
        echo "  Log:      ${LOG_SIZE} — ${LAST_LINE}"
    fi

    echo "────────────────────────────────"
}

runner_start() {
    echo "Starting GitHub Actions runner..."
    launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/${PLIST_NAME}.plist" 2>/dev/null
    sleep 2
    runner_status
}

runner_stop() {
    echo "Stopping GitHub Actions runner..."
    launchctl bootout "gui/$(id -u)/${PLIST_NAME}" 2>/dev/null
    echo -e "${YELLOW}Runner stopped${NC}"
}

runner_restart() {
    runner_stop
    sleep 1
    runner_start
}

runner_logs() {
    if [[ "$1" == "-f" ]]; then
        tail -f "$LOG_DIR/runner.log"
    elif [[ "$1" == "err" ]]; then
        tail -50 "$LOG_DIR/runner.err"
    else
        tail -50 "$LOG_DIR/runner.log"
    fi
}

# Direct invocation
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-status}" in
        status)  runner_status ;;
        start)   runner_start ;;
        stop)    runner_stop ;;
        restart) runner_restart ;;
        logs)    runner_logs "$2" ;;
        *)
            echo "Usage: runner_ctl.sh {status|start|stop|restart|logs [-f|err]}"
            exit 1
            ;;
    esac
fi
