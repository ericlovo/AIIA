#!/usr/bin/env bash
# Air-gap probe — proves the running Brain reports cloud egress denied and lists
# recent denials from Sanction's audit trail. Run with the Brain up under
# AIIA_AIRGAP=1. See docs/AIRGAP.md for the full evidence-pull runbook.
#
# Env:  BRAIN_URL (default http://localhost:8100)
#       LOCAL_BRAIN_API_KEY        — Brain API key (for authed endpoints)
#       SANCTION_API_URL / SANCTION_API_KEY — to check the audit trail (optional)
set -uo pipefail

BRAIN_URL="${BRAIN_URL:-http://localhost:8100}"
PASS=0
FAIL=0

check() { # check <name> <expected> <actual>
  if [ "$2" = "$3" ]; then
    echo "  ok   $1"
    PASS=$((PASS + 1))
  else
    echo "  FAIL $1 (expected $2, got $3)"
    FAIL=$((FAIL + 1))
  fi
}

echo "== /health airgap block"
health=$(curl -sf "$BRAIN_URL/health") || { echo "Brain not reachable at $BRAIN_URL"; exit 1; }
enabled=$(echo "$health" | python3 -c "import json,sys; print(json.load(sys.stdin)['airgap']['enabled'])")
check "airgap.enabled" "True" "$enabled"
echo "$health" | python3 -c "
import json, sys
a = json.load(sys.stdin)['airgap']
for name, state in sorted(a['egress'].items()):
    print(f'       {name}: {state}')
print('       permitted:', ', '.join(a['permitted']))"

echo "== Slack egress state (expect slack.post disabled)"
# General slack.post stays registered and denied. It no longer has a Brain route to
# provoke, so the probe reads the decision from /health. slack.capture_ack and
# slack.memory_post read "airgap-allowlisted" only when their opt-in flags are set.
slack_state=$(echo "$health" | python3 -c "import json,sys; print(json.load(sys.stdin)['airgap']['egress'].get('slack.post', 'unregistered'))")
check "slack.post egress state" "disabled" "$slack_state"

if [ -n "${SANCTION_API_URL:-}" ] && [ -n "${SANCTION_API_KEY:-}" ]; then
  echo "== Sanction audit trail (denied tool rows)"
  denied=$(curl -s "$SANCTION_API_URL/audit-events?type=authorization&limit=20" \
    -H "x-api-key: $SANCTION_API_KEY" | python3 -c "
import json, sys
events = json.load(sys.stdin).get('events', [])
rows = [e for e in events if e.get('status') == 'denied']
print(len(rows))
for e in rows[:5]:
    print(f\"       {e.get('created_at','?')}  {e.get('merchant', e.get('tool','?'))}  {e.get('decision_note','')}\", file=sys.stderr)
" 2>&2)
  echo "       denied rows in last 20 events: $denied"
else
  echo "== Sanction audit check skipped (SANCTION_API_URL/KEY not set)"
fi

echo
echo "passed=$PASS failed=$FAIL"
[ "$FAIL" -eq 0 ]
