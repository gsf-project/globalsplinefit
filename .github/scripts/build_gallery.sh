#!/usr/bin/env bash
# Build the browser tutorial gallery into site/gallery/ (run after `mkdocs
# build`; used by docs.yml and docs-preview.yml, and runnable locally).
#
# Each examples/*.py is published twice, plus its sources:
#
#   site/gallery/<name>/          app view  — `--mode run --show-code --execute`
#   site/gallery/<name>/edit/     editor    — `--mode edit`, auto-run patched on
#   site/gallery/<name>/<name>.py     source for the notebook's download button
#   site/gallery/<name>/<name>.ipynb  ditto, Jupyter conversion
#
#  * `--mode run` is the docs-example mode: cells auto-run and the reader sees
#    figures, not code. `--show-code` only ENABLES the code toggle; the gallery
#    cards link with `?show-code=false` so the page lands in app view and the
#    toggle reveals the code (marimo's own query parameter).
#  * `--execute` bakes the executed outputs into the export, so the figures are
#    on screen at first paint instead of ~15 s later, after pyodide boots.
#    `--no-sandbox` keeps the current uv env (which has globalsplinefit
#    installed from source); the default would re-resolve from PyPI, where the
#    package does not exist yet.
#  * The edit export does NOT auto-run on load, so flip `auto_instantiate` in
#    its embedded user config.
#  * Dedupe the marimo runtime: every export ships an identical ~27 MB assets/
#    dir (content-hashed filenames). Keep ONE copy at site/gallery/_assets/ and
#    rewrite the references (the browser cache then carries over between
#    tutorials). Identity is asserted, not assumed — a marimo upgrade that
#    breaks it fails loudly.
set -euo pipefail
cd "$(dirname "$0")/../.."

for nb in examples/*.py; do
  name="$(basename "$nb" .py)"
  echo "Exporting $name"
  uv run marimo export html-wasm "$nb" -o "site/gallery/$name" \
    --mode run --show-code --execute --no-sandbox -f
  uv run marimo export html-wasm "$nb" -o "site/gallery/$name/edit" \
    --mode edit --execute --no-sandbox -f
  cp "$nb" "site/gallery/$name/$name.py"
  uv run marimo export ipynb "$nb" -o "site/gallery/$name/$name.ipynb" -f
done

shared=site/gallery/_assets
rm -rf "$shared"
# depth-aware: <name>/index.html reaches _assets as ../_assets,
#              <name>/edit/index.html as ../../_assets
for d in site/gallery/*/ site/gallery/*/edit/; do
  [ -d "$d/assets" ] || continue
  if [ ! -d "$shared" ]; then
    mv "$d/assets" "$shared"
  else
    diff <(ls "$d/assets" | sort) <(ls "$shared" | sort) > /dev/null \
      || { echo "ERROR: $d assets differ from shared set — dedup unsafe"; exit 1; }
    rm -rf "$d/assets"
  fi
  case "$d" in
    */edit/) up=../../ ;;
    *)       up=../ ;;
  esac
  sed -i "s|\./assets/|${up}_assets/|g" "$d/index.html"
done

# The app-view pages must offer the code toggle; the editors must auto-run.
# Both exports must carry gallery.css + gallery_head.html: marimo only injects
# an app's css_file/html_head_file on the --execute code path (0.23.15), so a
# dropped --execute would silently lose the app-view default and the sidebar
# trim rather than fail.
for d in site/gallery/*/ site/gallery/*/edit/; do
  [ -f "$d/index.html" ] || continue
  grep -q 'chrome-sidebar' "$d/index.html" \
    || { echo "ERROR: gallery.css not inlined in $d/index.html"; exit 1; }
  grep -q 'replaceState' "$d/index.html" \
    || { echo "ERROR: gallery_head.html not inlined in $d/index.html"; exit 1; }
done
for d in site/gallery/*/; do
  [ -f "$d/index.html" ] || continue
  grep -q 'showAppCode[^,}]*true' "$d/index.html" \
    || { echo "ERROR: code toggle not enabled in $d/index.html"; exit 1; }
  # the notebook's download buttons resolve to these two, site-root relative
  name="$(basename "$d")"
  for ext in py ipynb; do
    [ -s "$d/$name.$ext" ] \
      || { echo "ERROR: missing download source $d$name.$ext"; exit 1; }
  done
done
for d in site/gallery/*/edit/; do
  [ -f "$d/index.html" ] || continue
  grep -q '"auto_instantiate": false' "$d/index.html" \
    || { echo "ERROR: auto_instantiate flag not found in $d/index.html"; exit 1; }
  sed -i 's|"auto_instantiate": false|"auto_instantiate": true|g' "$d/index.html"
done

echo "gallery: $(du -sh site/gallery | cut -f1), $(find site/gallery -type f | wc -l) files"
