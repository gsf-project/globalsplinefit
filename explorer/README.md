# GSF Explorer — in-browser web app

Interactive access to the Global Spline Fit model: the **actual
`globalsplinefit` package** (numpy + scipy) runs client-side in a
Pyodide/WebAssembly web worker; the front end is a hand-rolled reactive
Preact UI. No server-side compute, no build step — static hosting (GitHub
Pages) suffices. First visit downloads ~25 MB of runtime (cached afterwards).

## Architecture

Two-speed state model:

- **Display state** (spectral weight γ, flux/fraction normalization,
  log/linear, component visibility, line weight, bands + opacity, hover, x-window and
  y-range — i.e. ALL navigation: box-zoom, pan, trackpad pinch/scroll) is
  applied by `chart.js` in plain JS — instant, no Python. The model grid
  always covers the full 10⁰–10¹¹ range (default 480 pts), so zooming and
  panning slice a cached grid and never recompute.
- **Model state** (parameter sets, plotted quantity, abscissa, elements,
  solar modulation,
  energy scale, cutoff, grid resolution) round-trips to the Pyodide worker
  (debounced + pumped: one evaluation in flight, always against the latest
  params).

The Settings pane selects nucleus flux, nucleon flux, ⟨ln A⟩, or
σ²(ln A). Nucleus flux and both composition moments support all five
abscissae. Nucleon flux is available only versus total or kinetic energy per
nucleon; selecting it immediately constrains the axis selector to those two
physical choices. Composition curves are nucleus-weighted and their bands use
the package covariance propagation.

Fraction view shows Φᵢ/Φ_total (the paper's fraction plots). Its band is
drawn twice: σᵢ/Φ_total immediately, which neglects the correlation with the
total and overstates the band (median 3-10%, more where a group approaches
the whole total), then the covariance-aware `fraction_error` from the
`fractionErrors` RPC, which replaces it. That call costs ~440 ms in WASM for
480 points and 4 groups, so the status pill reads "refining bands…" while it
runs and the dock note says which band is on screen.
Publication exports combine all active models on shared axes and honor the
current window, y limits, flux/fraction view, and band visibility. Paper line
weights and band opacity are independent of the canvas display settings.
Filled bands identify the primary model; optional hatched bands
identify overlays. A generated, copyable caption describes the selected curves
and is included below the figure by default (it can be omitted for manuscripts).
CSV exports produce one file per active model; the data-table modal lists all models and has its own CSV.

Up to three parameter sets can be overlaid (Model panel): the first is
solid with ±1σ bands and carries the element curves; overlays are dotted /
dash-dot, groups + total only. Rows are drag-reorderable — order decides
which model is primary. Overlay bands are off by default and can be enabled
in Display as 45°/135° hatch fills (SVG patterns), so filled vs hatched
separates the models even where bands overlap. The plot has a horizontal
toolbar (box-zoom, pan, home); box selection sets both the decade window
(re-gridded at full resolution) and the y-range, double-click or home
resets. Where Φ−σ ≤ 0 (σ exceeds the flux at the highest energies) the band
clamps to the plot bottom — "consistent with zero" — rather than collapsing.

| file | role |
|---|---|
| `index.html` | shell: fonts, theme pre-paint stamp, error surface, splash |
| `main.js` | app: state, worker RPC, command overlays, display dock, exports, modals, citations |
| `chart.js` | reactive SVG spectrum chart (log axes, bands + hatches, legend, crosshair, box-zoom/pan) |
| `worker.js` | Pyodide web worker; lazy-loads matplotlib on first figure export |
| `bridge.py` | JSON adapter between the worker RPC and `gsf_explorer` |
| `gsf_explorer.py` | model/figure/CSV core (no UI imports — testable headless) |
| `style.css` | Liquid Canvas design tokens, responsive overlays/sheets, and self-hosted fonts |
| `fonts/` | IBM Plex Sans + IBM Plex Mono (self-hosted woff2, ~75 kB) |
| `globalsplinefit-*.whl` | the model package + all parameter sets |
| `update_wheel.sh` | rebuild the wheel from this repo (clean temp copy) |
| `serve.py` | local dev server (`Cache-Control: no-store`, port 8123) |
| `browser_smoke.py` | headless-chromium acceptance test (boot, interactions, all exports) |

Pinned externals (the only CDN dependencies): `pyodide v0.28.3` (classic
`importScripts` in `worker.js`) and `htm@3.1.1/preact` (direct URL imports
in `main.js`/`chart.js`). Everything else — fonts included — is
self-hosted. Deliberately NO import maps and NO module workers: both are
too new for older Firefox/Safari and fail as a silent blank page; early
failures are surfaced on the splash by an error handler in `index.html`.

## Run locally

```sh
python3.12 -m http.server 8123 -d webapp/
# open http://localhost:8123
```

`file://` does NOT work (module scripts, import maps, fetch need http).

## Test

```sh
pip install playwright && playwright install chromium
python test_ui.py            # FULL suite: every control element (~4 min)
python test_ui.py --fast     # same, minus matplotlib figure exports
python browser_smoke.py      # quick gate: boot + key interactions + exports
```

`test_ui.py` exercises every control on the page — model add/remove/drag-
reorder, all four plot quantities and their abscissa constraints, all five
abscissas (incl. deuterium on rigidity), every component and
element chip, all display/modulation/advanced inputs, box-zoom/pan/wheel/
home, hover, all exports, both themes, mobile — and fails on any error
toast, console error, or missing effect, with a per-control PASS/FAIL
summary. Run it after any change. The desktop Python is NOT the runtime —
Pyodide ships its own numpy/scipy/matplotlib builds, so only in-browser
tests count (lesson learned from the retired stlite front end).

## Update the model

```sh
./update_wheel.sh    # rebuilds the wheel from a clean source copy
```

then bump `WHEEL` in `main.js` if the version changed.

## Deploy

Deployed to this repo's GitHub Pages at `/explorer/` by the `explorer` job
in `.github/workflows/release.yml` — **on version tags only**, so the
public site always runs a released model. The docs (mkdocs, deployed on
every main push) set `keep_files: true` so they never wipe `/explorer/`.
CI (`.github/workflows/webapp.yml`) runs the UI suite on every webapp or
src change; the wheel is built fresh in CI (never committed) and the
`WHEEL` constant in `main.js` is rewritten to the built filename.

## Maintenance

Everything a maintainer needs to know, in one place:

**Updating the model package** (new parameter sets, renamed versions, new
elements): run `./update_wheel.sh`, bump `WHEEL` in `main.js` if the
version changed, run `python test_ui.py`. The page is meta-driven — it asks
the package at boot for the version list, the default version (first
registered), per-version notes, the element table (incl. deuterium
availability per abscissa) and the φ-table year range — so **renames and
additions need no app change**, with three exceptions to check:
1. `VERSION_NOTES` in `gsf_explorer.py` — short descriptions derived from the
   registry roles, with conference labels for historical releases.
2. `CITATIONS` in `main.js` — map new versions to their InspireHEP records
   (fetch BibTeX verbatim: `curl -H "Accept: application/x-bibtex"
   https://inspirehep.net/api/literature/<id>`).
3. `MAX_MODELS`/`MODEL_DASH` allow 3 overlaid models; extend `MODEL_DASH`
   in `chart.js` if more are ever wanted.

**Known workarounds tied to the package version** (drop when fixed
upstream):
- Ratio-view bands are σᵢ/Φ_total (correlation with the total neglected);
  proper `model.fraction_error` is ~10 s in WASM — revisit if it gets
  faster.

**Pinned externals** (bump deliberately, run the full suite after):
pyodide `v0.28.3` in `worker.js`; `htm@3.1.1/preact` URLs in
`main.js`/`chart.js`. Everything else is self-hosted. The evaluation domain
is `X_DOMAIN = [-1, 11]` decades in `chart.js` (0.1–10¹¹ GeV/GV); zeros
from below-validity regions (LIS < 1 GV, sub-threshold total energy) render
as gaps by design.

**Architecture invariants worth preserving**: navigation must stay
display-only (full-range cached grid); one evaluation in flight (the pump
in `main.js`); the browser is the only real runtime — desktop-Python tests
prove nothing; series palettes are validated per surface (CVD + contrast
checks) — don't tweak hues casually.

## Design notes

"Liquid Canvas" in dark (`#101317`) and light (`#f7f8fa`), defaulting to the
browser's `prefers-color-scheme` and toggleable in the header (explicit choice
persists in localStorage; stamped pre-paint to avoid a theme flash). The chart
is the full content plane. Navigation floats at the top, immediate display
controls sit in the bottom dock, and Series / Settings / Export open
consistent anchored command panes. Series combines parameter-set overlays with
component and element visibility (including a one-step element reset); Settings
contains plot quantity + Abscissa, Solar modulation, and Advanced controls.
All three panes
reserve a visible scrollbar and become scrollable bottom sheets on mobile.
Opening them does not change the chart size or trigger evaluation.
Typography is IBM Plex Sans for UI, wordmark and axis titles, with IBM Plex
Mono for numerics and the telegraphic state line
(`MODEL(s) · quantity · modulation · range · abscissa`). The top-right legend tightens
its spacing as the plot narrows, auto-collapses on compact canvases, and can
always be toggled with pointer or keyboard. Group colors are the paper's hues
re-stepped **per surface** and validated for CVD/contrast
(dataviz six checks, adjacent pairs PASS) — dark: p `#e64d6e`, He `#c08a00`,
O* `#1a9e70`, Fe* `#3d8ce0`; light: p `#c73558`, He `#ab8200`,
O* `#006e42`, Fe* `#2a6fc0`; all-particle is ink in both. Model overlays are
distinguished by line style (solid/dotted/dash-dot), never by repainting the
component hues. Dragging γ morphs the spectrum in real time; hover gives a
crosshair readout of every visible component across all models.

The paper aesthetic lives on **in the exports only**: publication figures
re-render via matplotlib with the main paper's Nimbus Roman text and math
(bundled under `fonts/`), red/yellow/green/blue group colors, 0.75 pt group
lines, 1.2 pt total, 0.3-opacity bands, and combined line/band legend keys.
Single-model curves use the paper's solid/dashed/dash-dot/dotted group styles;
comparisons use solid/dotted/dash-dot for model identity. The default is the
paper's 6×3.6 in wide figure; journal column sizes are also available (PDF/SVG/PNG).
CSV carries a provenance header, flux + 1σ
columns for every drawn series (groups AND individual elements), and
selectable covariance blocks — none / total flux / total + one N×N block per
drawn series, so single-element errors ship with their full covariance.
Also exportable: the live SVG view and an in-page data table.

The About modal carries the citation records (verbatim InspireHEP BibTeX
with copy buttons): `Dembinski:2017zsh` (GSF 2017/2019), `Fujisue:2025wnp`
(UHECR 2024), `Dembinski:2025nmp` (ICRC 2025), and the GSF 2026 paper
(reference text until its InspireHEP record exists).
