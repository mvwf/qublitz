#!/usr/bin/env bash
# BE-7 — fails if pages/_assets/quantum_chess.html has drifted from the
# committed pages/_assets/VENDOR_SHA (written by the game repo's
# scripts/sync_vendor.sh at sync time). The vendored copy has no other
# enforcement — a "golden rule" comment is discipline, not a check — and
# this repo's own history already has one iCloud-sync incident that
# manufactured a divergent duplicate file. Called from CI (see
# .github/workflows/ci.yml) and from scripts/verify.sh.
#
# Usage: scripts/check_vendor.sh
set -euo pipefail
cd "$(dirname "$0")/.." || exit 2

GAME_HTML="pages/_assets/quantum_chess.html"
SHA_FILE="pages/_assets/VENDOR_SHA"

if [ ! -f "$GAME_HTML" ] || [ ! -f "$SHA_FILE" ]; then
  echo "vendor check: SKIPPED — $GAME_HTML not vendored yet"
  exit 0
fi

EXPECTED="$(tr -d '[:space:]' < "$SHA_FILE")"
ACTUAL="$(shasum -a 256 "$GAME_HTML" | awk '{print $1}')"

if [ "$EXPECTED" != "$ACTUAL" ]; then
  echo "vendor check: FAIL — $GAME_HTML does not match $SHA_FILE" >&2
  echo "  expected: $EXPECTED" >&2
  echo "  actual:   $ACTUAL" >&2
  echo "  run the game repo's scripts/sync_vendor.sh and commit the result." >&2
  exit 1
fi

echo "vendor check: PASS — $GAME_HTML matches $SHA_FILE ($ACTUAL)"
