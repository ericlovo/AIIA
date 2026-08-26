#!/usr/bin/env bash
# Run Writ's own quality harness end-to-end and write results to workshop/baseline/.
#
# Four layers, in the order they catch things:
#
#   1. scripts/eval.sh        45 static checks. This is what CI runs.
#   2. scripts/tests/         792 unit tests. CI does NOT run these.
#   3. measure-invocation.py  Per-command context load (the ADR-021 budget).
#   4. check-agent-parity.sh  agents/ vs claude-code/agents/ vs codex/agents/.
#
# Usage:
#   bash workshop/run-harness.sh            # all four layers
#   bash workshop/run-harness.sh eval       # just the eval checks
#   bash workshop/run-harness.sh tests      # just the unit suite
#   bash workshop/run-harness.sh measure    # just the context measurement

set -uo pipefail

OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/baseline"
LAYER="${1:-all}"

# Honor $WRIT_DIR if set; otherwise probe the usual checkout locations.
if [ -z "${WRIT_DIR:-}" ]; then
  for c in "$HOME/sellke/writ" /home/user/sellke/writ ../writ ../sellke/writ; do
    [ -d "$c/.git" ] && { WRIT_DIR="$(cd "$c" && pwd)"; break; }
  done
fi

if [ -z "${WRIT_DIR:-}" ] || [ ! -d "$WRIT_DIR/.git" ]; then
  echo "No Writ checkout found — run: bash workshop/bootstrap.sh" >&2
  echo "(or set WRIT_DIR=/path/to/writ)" >&2
  exit 1
fi

mkdir -p "$OUT"
cd "$WRIT_DIR"

REV="$(git rev-parse --short HEAD)"
DIRTY="$(git status --porcelain | wc -l)"
echo "Writ $(cat VERSION) @ $REV — python $(python3 -V 2>&1 | cut -d' ' -f2)"
if [ "$DIRTY" -eq 0 ]; then
  echo "Tree: pristine (as shipped)"
else
  echo "Tree: $DIRTY MODIFIED file(s) — results do NOT reflect upstream:"
  git status --short | sed 's/^/      /'
fi
echo "Results -> $OUT"
echo

# Read-only posture: this harness must not leave anything behind in his tree.
# .writ/state/ is gitignored but is his workspace — only remove it if we
# were the ones who created it.
STATE_PREEXISTED=false
[ -d "$WRIT_DIR/.writ/state" ] && STATE_PREEXISTED=true
cleanup() {
  rm -rf "$WRIT_DIR/.pytest_cache" "$WRIT_DIR/scripts/__pycache__" \
         "$WRIT_DIR/scripts/tests/__pycache__" 2>/dev/null || true
  $STATE_PREEXISTED || rm -rf "$WRIT_DIR/.writ/state" 2>/dev/null || true
}
trap cleanup EXIT

rc=0

run_eval() {
  echo "── [1/4] eval.sh — 45 static checks (what CI runs)"
  if [ -f "$WRIT_DIR/.git/shallow" ]; then
    echo "    WARNING: shallow clone — archive-dogfood will report a false FAIL (findings/01)"
  fi
  bash scripts/eval.sh --report="$OUT/eval.md" >/dev/null 2>&1
  local e=$?
  awk '/^## /{n=$2} /^(PASS|FAIL)/{printf "    %-32s %s\n", n, $0}' "$OUT/eval.md"
  echo "    $(grep -c '^PASS' "$OUT/eval.md") pass / $(grep -c '^FAIL' "$OUT/eval.md") fail"
  [ $e -ne 0 ] && rc=1
  echo
}

run_tests() {
  echo "── [2/4] pytest scripts/tests/ — unit suite (CI does NOT run this)"
  python3 -m pytest scripts/tests/ -q > "$OUT/tests.txt" 2>&1
  local e=$?
  tail -3 "$OUT/tests.txt" | sed 's/^/    /'
  [ $e -ne 0 ] && { rc=1; echo "    ^ see $OUT/tests.txt"; }
  echo
}

run_measure() {
  echo "── [3/4] measure-invocation.py — per-command context load"
  python3 scripts/measure-invocation.py --format=table > "$OUT/invocation.txt" 2>&1
  grep -E 'shared base|^commands:' "$OUT/invocation.txt" | sed 's/^/    /'
  echo "    full table -> $OUT/invocation.txt"
  echo
}

run_parity() {
  echo "── [4/4] check-agent-parity.sh — cross-platform agent alignment"
  bash scripts/check-agent-parity.sh 2>&1 | sed 's/^/    /'
  [ ${PIPESTATUS[0]} -ne 0 ] && rc=1
  echo
}

case "$LAYER" in
  eval)    run_eval ;;
  tests)   run_tests ;;
  measure) run_measure ;;
  parity)  run_parity ;;
  all)     run_eval; run_tests; run_measure; run_parity ;;
  *)       echo "Unknown layer: $LAYER (eval|tests|measure|parity|all)" >&2; exit 1 ;;
esac

[ $rc -eq 0 ] && echo "All green." || echo "Something failed — see $OUT/."
exit $rc
