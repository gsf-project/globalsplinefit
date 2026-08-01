#!/usr/bin/env bash
# Rebuild the globalsplinefit wheel for the web app from a clean source copy
# of THIS repo (rsync to a temp dir so no build artifacts land in the tree).
# Uses uv if available; otherwise a python with pip + setuptools + wheel
# (override with PYTHON=...).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${1:-$HERE/..}"
PYTHON="${PYTHON:-python3}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

rsync -a --exclude .git --exclude __pycache__ --exclude site --exclude .venv \
    --exclude webapp "$SRC/" "$TMP/src/"
if command -v uv >/dev/null 2>&1; then
    uv build --wheel --out-dir "$TMP/out" "$TMP/src"
else
    "$PYTHON" -m pip wheel --no-deps --no-build-isolation -w "$TMP/out" "$TMP/src"
fi

rm -f "$HERE"/globalsplinefit-*.whl
cp "$TMP"/out/globalsplinefit-*.whl "$HERE/"
echo "installed: $(ls "$HERE"/globalsplinefit-*.whl)"
echo "reminder: if the version changed, update WHEEL in main.js"
