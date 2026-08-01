#!/usr/bin/env bash
# Build the browser tutorial gallery into site/gallery/ (run after `mkdocs
# build`; used by docs.yml and docs-preview.yml, and runnable locally):
#
#  1. `marimo export html-wasm --mode edit` for each examples/*.py — full
#     in-browser editor so readers can change code and re-run cells.
#  2. Flip `auto_instantiate` to true in each export's embedded user config:
#     edit-mode exports do NOT run on load by default; this makes them.
#  3. Dedupe the marimo runtime: every export ships an identical ~27 MB
#     assets/ dir (content-hashed filenames). Keep ONE copy at
#     site/gallery/_assets/ and rewrite the references (222 MB -> ~29 MB,
#     and the browser cache carries over between tutorials). Identity is
#     asserted, not assumed — a marimo upgrade that breaks it fails loudly.
set -euo pipefail
cd "$(dirname "$0")/../.."

for nb in examples/*.py; do
  name="$(basename "$nb" .py)"
  echo "Exporting $name"
  uv run marimo export html-wasm "$nb" -o "site/gallery/$name" --mode edit
done

shared=site/gallery/_assets
rm -rf "$shared"
for d in site/gallery/*/; do
  [ -d "$d/assets" ] || continue
  if [ ! -d "$shared" ]; then
    mv "$d/assets" "$shared"
  else
    diff <(ls "$d/assets" | sort) <(ls "$shared" | sort) > /dev/null \
      || { echo "ERROR: $d assets differ from shared set — dedup unsafe"; exit 1; }
    rm -rf "$d/assets"
  fi
  sed -i 's|\./assets/|../_assets/|g' "$d/index.html"
  grep -q '"auto_instantiate": false' "$d/index.html" \
    || { echo "ERROR: auto_instantiate flag not found in $d/index.html"; exit 1; }
  sed -i 's|"auto_instantiate": false|"auto_instantiate": true|g' "$d/index.html"
done
echo "gallery: $(du -sh site/gallery | cut -f1), $(find site/gallery -type f | wc -l) files"
