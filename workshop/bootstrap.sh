#!/usr/bin/env bash
# Bootstrap a PRISTINE Writ checkout for the workshop.
#
# We are going into the author's session. The starting state is Writ exactly as
# he built it: HEAD == origin/main, zero diff, no stray files, no patches
# applied. This script establishes that and verifies it; it never modifies
# tracked files.
#
# Usage:
#   bash workshop/bootstrap.sh            # clone if needed, then verify pristine
#   bash workshop/bootstrap.sh --verify   # verify only, change nothing
#   bash workshop/bootstrap.sh --reset    # force back to pristine (DISCARDS local edits)

set -euo pipefail

WRIT_DIR="${WRIT_DIR:-$HOME/sellke/writ}"
[ -d "$WRIT_DIR/.git" ] || for c in "$HOME/sellke/writ" /home/user/sellke/writ ../writ ../sellke/writ; do
  [ -d "$c/.git" ] && { WRIT_DIR="$(cd "$c" && pwd)"; break; }
done

WRIT_REPO="https://github.com/sellke/writ"
MODE="${1:-default}"

echo "Writ checkout: $WRIT_DIR"

if [ ! -d "$WRIT_DIR/.git" ]; then
  [ "$MODE" = "--verify" ] && { echo "  MISSING — nothing to verify" >&2; exit 1; }
  mkdir -p "$(dirname "$WRIT_DIR")"
  # NOT --depth 1. archive-dogfood reads real commit history and reports a
  # false FAIL on a shallow clone (findings/01).
  git clone "$WRIT_REPO" "$WRIT_DIR"
fi

cd "$WRIT_DIR"

if [ -f .git/shallow ] && [ "$MODE" != "--verify" ]; then
  echo "  shallow clone — unshallowing (archive-dogfood needs full history)"
  git fetch --unshallow
fi

if [ "$MODE" = "--reset" ]; then
  echo "  --reset: discarding all local changes"
  git fetch origin main
  git checkout main 2>/dev/null || git checkout -B main origin/main
  git reset --hard origin/main
  git clean -fdx
fi

# ---- verify pristine -------------------------------------------------------

git fetch -q origin main 2>/dev/null || echo "  (offline — comparing against last-known origin/main)"

HEAD_SHA="$(git rev-parse HEAD)"
ORIGIN_SHA="$(git rev-parse origin/main 2>/dev/null || echo unknown)"
DIRTY="$(git status --porcelain | wc -l)"
STRAY="$(git status --porcelain --ignored | grep -c '^!!' || true)"

echo
echo "  version:  $(cat VERSION)  @ $(git rev-parse --short HEAD)"
echo "  commits:  $(git rev-list --count HEAD)"
echo "  HEAD:     $HEAD_SHA"
echo "  origin:   $ORIGIN_SHA"
echo "  modified: $DIRTY tracked file(s)"
echo "  stray:    $STRAY ignored path(s)"

ok=true
[ "$HEAD_SHA" = "$ORIGIN_SHA" ] || { echo "  ✗ HEAD is not origin/main"; ok=false; }
[ "$DIRTY" -eq 0 ] || { echo "  ✗ tracked files modified:"; git status --short | sed 's/^/      /'; ok=false; }

if $ok; then
  echo
  echo "PRISTINE — this is Writ exactly as shipped."
  [ "$STRAY" -gt 0 ] && echo "(note: $STRAY gitignored path(s) present — build artifacts, harmless)"
else
  echo
  echo "NOT PRISTINE. Run: bash workshop/bootstrap.sh --reset" >&2
  exit 1
fi

python3 -m pytest --version >/dev/null 2>&1 || {
  echo "installing pytest (scripts/tests/ needs it)"
  python3 -m pip install -q pytest
}

echo
echo "Next: bash workshop/run-harness.sh"
