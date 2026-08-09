# GSF - Global Spline Fit

## Project Overview

Parametric model for cosmic ray flux and composition based on cubic B-spline fits to observational data. Provides separate proton/neutron nucleon fluxes with full uncertainty propagation via Jacobian-transformed parameter covariances.

## Architecture

- `src/globalsplinefit/model.py` - Core model classes: `GSFEnergy`, `GSFKineticEnergy`, `GSFRigidity`, `GSFEnergyPerNucleon`, `GSFKineticEnergyPerNucleon`. All inherit from `GSFBase`.
- `src/globalsplinefit/data_management.py` - `Parameters` class loads spline knots, coefficients, covariances, nuclear data, and solar modulation from `data/{version}/` directories.
- `src/globalsplinefit/reduced.py` - `ReducedGSF`: pivot-based flux nuisance parameters for downstream fits. The all-default constructor uses the published per-version grid `data/<version>/reduced_pivots.dat` (via `model.params.reduced_pivots`; theta components are defined at version-specific energies). A custom bundle without a pivot table raises; `optimize_pivots` derives one, and grids are regenerated on version promotion.
- `src/globalsplinefit/data/` - Data files, one directory per model version, named by year (`2026.0`, `2026.0-USO`, ...). Current sets, regenerated together after every promotion: `2026.0` (default; mixture + GMD), `2026.0-USO`, `2026.0-SIB23e`, `2026.0-EPOSLHCR`. Historical (static): `2025`, `2019`, `2017`. Each version has `parameters.dat`, `covariance.dat`, `knots.dat`, `nuclei.dat`, `reduced_pivots.dat` (current sets add `subleading.dat`, `fit_result.json`, and their own `solar_modulation.dat`). Shared Usoskin `solar_modulation.dat` in the parent `data/` directory (historical fallback). The `MODEL_VERSIONS` registry in `data_management.py` is the allow-list.
- `webapp/` - GSF Explorer (Pyodide + Preact). The **"Liquid Canvas" UI design is final** (maintainer-approved 2026-07-31): the plot is the content plane; Series/Settings/Export command panes, bottom display dock, dark+light tokens in `style.css`. Extend it — do not restyle or rebuild. `test_ui.py` (64 checks, incl. an iPad-size touch section: tap readout, pinch zoom, double-tap home, pane fit; plus the narrow-viewport dock and axis-gutter checks) asserts its invariants and is the acceptance gate for any webapp change. iPad/touch is a SUPPORTED target (maintainer, 2026-08-02) — touch regressions are release blockers.

## TEMPORARY pre-publication markers (strip at the GSF 2026 release)

- `webapp/chart.js` — `PRELIMINARY` const: diagonal watermark on the chart
  (and its SVG export). Set false / delete.
- `webapp/gsf_explorer.py` — `make_figure`: mirrored watermark on the
  publication-figure exports (PDF/SVG/PNG). Delete the marked `ax.text` block.
- `examples/*.py` — the `fig.text(...)` watermark block inside the `show()`
  helper in each import cell (marked, greps for `_gsf_preliminary`). Delete
  the block, KEEP `show()` and its `return fig` — the figures do not render
  in app view without it.
- `examples/*.py` + `docs.yml` — pyodide/micropip wheel-install cells and the
  `site/wheels/` build step, obsolete once the package is on PyPI.

## Key Concepts

- **4 element groups** (leaders), written **H\*, He\*, O\*, Fe\*** since
  2026-08-08 (the paper migrates to this nomenclature; the model's target keys
  are still `"H*"`, `"He"`, `"O*"`, `"Fe*"`): H (Z=1), He (Z=2), O* (Z=8), Fe* (Z=26). Subleading elements scale from their group leader.
- **Parameter covariance** stored as 10 block pairs between the 4 leaders.
- **Parameter trimming**: When building Jacobians/covariances, boundary parameters are trimmed: `params[1:-7]`, `cov[1:-3, 1:-3]` per element.
- **Nucleon flux**: `GSFEnergyPerNucleon.p_and_n_flux()` returns shape `(2, N)` for proton and neutron components. Each is summed over all 28 nuclei weighted by Z and A-Z.
- **Uncertainty propagation**: `Cov_flux = J @ Cov_params @ J.T` where J is the spline Jacobian.
- **Solar modulation**: Default is Solar Cycle 24 average (Dec 2008 - Dec 2019). Use `time_interval="LIS"` for unmodulated local interstellar spectrum.

## Example Notebooks

Located in `examples/` as **marimo notebooks** — plain `.py` files;
`marimo export ipynb` produces Jupyter copies on demand. Open interactively
with `uv run marimo edit examples/<name>.py`; running
`uv run python examples/<name>.py` executes the full cell DAG (that is the CI
smoke test in test.yml). Each notebook's import cell starts with a TEMPORARY
pyodide/micropip block installing the wheel from `/wheels/` on the Pages site
— remove these blocks (and the wheel step in docs.yml) once globalsplinefit is
on PyPI.

**Gallery publishing** (`.github/scripts/build_gallery.sh`, run by docs.yml;
cards in `docs/gallery.md`). Every notebook is exported twice — the app view
at `/gallery/<name>/` and an editor at `/gallery/<name>/edit/` — plus its
`.py`/`.ipynb` sources, which the notebook's own download buttons link to.
Rules that keep the app view working; break one and the page renders blank or
lands on raw code:

- **A plot cell must END with the figure** (`show(fig)`, which stamps the
  watermark and returns it). `plt.show()` renders NOTHING in app view.
- The app export needs `--show-code` (that is what puts "Show code" in the ⋯
  menu) and `examples/gallery_head.html` defaults marimo's `show-code` query
  parameter to false so the page still lands in app view.
- `--execute` is not optional: it bakes the figures into the export (no blank
  page while pyodide boots) and, in marimo 0.23.15, it is the ONLY export path
  that injects an app's `css_file`/`html_head_file`.
- `examples/gallery.css` trims the editor sidebar to the file panel.
- Controls (`mo.ui.dropdown` for the model version, `mo.ui.slider` for the
  display scaling) belong in their own cell, and y-limits that depend on the
  scaling must use the `autoscale()` helper, not fixed numbers.

`.github/scripts/test_gallery.py` (headless chromium) asserts all of this and
is the acceptance gate for gallery changes — the notebook counterpart of
`webapp/test_ui.py`.

- `paper_figures.py` - The GSF 2026 paper figures the model can draw on its
  own (main Figs. 1, 2, 3, 5, 6, 8-12; SM S1, S2, S3, S4, S7, S8, S9, S14),
  **without the measurements** (they belong to the publishing experiments).
  Not a free reproduction: each cell is a port of the function that generates
  that figure, keeping the figure size, spectral weighting, axis ranges, tick
  locators, palette, line styles and legends. Generators, in gsf-fitter-2 and
  the gsf-harness run dirs:
  `gsf_apps/deck/` (1, 2, 3, 5, 6, 8, 11, S1, S7, S14) ·
  `gsf_apps/paper_figures/massgroups.py` + `lv2024.py` (9) ·
  `comparison.py` (10, 12) ·
  `runs/2026-07-26_boron-toa-refit/inputs/` (S2, S3, S4) ·
  `runs/2026-08-05_reduced-fidelity-figure/inputs/` (S8) ·
  `runs/2026-08-06_reduced-sampling-nucleus/inputs/` (S9).
  Three things are easy to get wrong and are load-bearing:
  - **Every deck curve is the LIS.** The deck's `ModelAdapter` pins
    `default_time_interval="LIS"` on all three model classes; calling `flux()`
    without it applies the default modulation and is ~6x off at 2 GV. So
    `gsf_r`/`gsf_k`/`gsf_kn` are LIS-defaulted here, while `gsf_e`/`gsf_en`
    are not — matching `comparison.py`/`massgroups.py`/the SM scripts.
  - **Two font families.** `deck/config.py` sets a Times-like serif at import;
    the stand-alone SM run scripts don't and get DejaVu Sans. `paper_figures/*`
    runs in the same process as the deck, so Figs. 9, 10, 12 are serif too.
    Every plot cell opens its own `plt.rc_context(...)` so cell order cannot
    leak a font. STIXGeneral stands in for Nimbus Roman (bundled with
    matplotlib, so it works in pyodide).
  - **Nomenclature is p, He, O\*, Fe\*** (the paper's). Model target keys stay
    `"H*"`, `"He"`, `"O*"`, `"Fe*"`; note `flux(E, "p")` is the proton ELEMENT
    and is what Fig. 9's "p" row uses.

  Deliberate deviations from the printed figures: SM S1 draws the real
  B-spline basis functions (the paper's shows none — the deck's frozen model
  ignores the `par` argument `spline_plots` mutates); SM S4 has two of three
  curves (the third needs a re-fit that is not in the released model);
  data legends and in-panel notes about data are dropped. Fig. 3 spans each
  species' direct-data rigidity range via the `DIRECT_RANGE_GV` table
  (coverage intervals, not measurements). App-view-first; not a tutorial.
- `cosmic_ray_flux.py` - Basic cosmic ray flux calculations and plotting,
  ending with pseudo-experiment draws from `sample()`.
- `nucleon_flux.py` - Nucleon flux for atmospheric shower simulations.
- `solar_modulation.py` - Solar modulation effects on flux.
- `rigidity_cutoff.py` - Geomagnetic rigidity cutoff effects.
- `model_comparison_2017_vs_2025.py` - Comparing 2017 and 2025 model versions.
- `reduced_model.py` - Build, validate, and export the `ReducedGSF` pivot
  components.

## Development

Environments and workflows use [uv](https://docs.astral.sh/uv/); `uv.lock` is
committed. All CI workflows install via `astral-sh/setup-uv` + `uv sync`.

```bash
uv sync --all-extras             # Create/refresh .venv with all extras
uv run pytest tests/             # Run tests
uv run pytest tests/ -m "not slow"  # Skip slow tests
```

- Python >= 3.10, dependencies: numpy, scipy
- Linting: ruff (line-length 88)
- Tests: pytest with markers (slow, unit, integration, regression)

## Pre-commit checklist

Before committing any changes, always run the following and fix any issues:

```bash
uv run ruff check src tests examples     # Linting (must pass with zero errors)
uv run ruff format --check src tests examples  # Formatting check
```

This matches the CI Code Quality workflow and prevents lint/format failures on
push.
