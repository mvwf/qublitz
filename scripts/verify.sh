#!/usr/bin/env bash
# Local debugging gate for the Qublitz fork. Runs the same checks as CI:
# ruff lint + pytest smoke tests + (when vendored) the game physics test.
# Prints a single PASS/FAIL and exits non-zero if anything fails.
#
#   bash scripts/verify.sh
#
# Run this GREEN before any push or pull request.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

# Prefer the local venv interpreter if present, else fall back to python3.
PY=".venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

fail=0
run() {
  local name="$1"; shift
  echo "── $name ──────────────────────────────────────────────"
  if "$@"; then
    echo "  ✓ $name"
  else
    echo "  ✗ $name"
    fail=1
  fi
  echo
}

run "ruff (lint)"          "$PY" -m ruff check .
run "pytest (smoke tests)" "$PY" -m pytest -q

# Game physics regression test — only once the game is vendored (C02-T4).
GAME_TEST="pages/_assets/tests/qphysics.test.js"
GAME_HTML="pages/_assets/quantum_chess.html"
if [ -f "$GAME_TEST" ]; then
  if command -v node >/dev/null 2>&1; then
    run "node (game physics)" node "$GAME_TEST" "$GAME_HTML"
  else
    echo "── node (game physics) ── SKIPPED: node not installed"; echo
  fi
fi

echo "════════════════════════════════════════════════════════"
if [ "$fail" -eq 0 ]; then
  echo "VERIFY: PASS ✓  — safe to push / open a PR"
else
  echo "VERIFY: FAIL ✗  — fix the above before pushing"
fi
exit "$fail"
