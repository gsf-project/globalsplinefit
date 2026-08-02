# GSF - Global Spline Fit

## Project Overview

Parametric model for cosmic ray flux and composition based on cubic B-spline fits to observational data. Provides separate proton/neutron nucleon fluxes with full uncertainty propagation via Jacobian-transformed parameter covariances.

## Architecture

- `src/globalsplinefit/model.py` - Core model classes: `GSFEnergy`, `GSFKineticEnergy`, `GSFRigidity`, `GSFEnergyPerNucleon`, `GSFKineticEnergyPerNucleon`. All inherit from `GSFBase`.
- `src/globalsplinefit/data_management.py` - `Parameters` class loads spline knots, coefficients, covariances, nuclear data, and solar modulation from `data/{version}/` directories.
- `src/globalsplinefit/reduced.py` - `ReducedGSF`: pivot-based flux nuisance parameters (theta = delta f/f at pivot energies; local-cubic interpolation in log E, `basis="hat"` optional) with exact covariance penalty, for downstream fits (daemonflux-style). The all-default constructor uses the **published grid** `RECOMMENDED_PIVOTS["2026"]` (12 quotable pivots -> 24 named parameters like `p_9TeV`; worst-case coverage 1.29) so downstream results are citable without re-optimizing; `optimize_pivots` (minimax coordinate exchange) is only for custom windows/versions, and grids are regenerated on version promotion.
- `src/globalsplinefit/data/` - Data files, one directory per model version (year-only names, no "GSF" prefix). Current sets, regenerated together after every promotion: `2026` (default; mixture + GMD), `2026-USO`, `2026-UHE-S23e`, `2026-EPOS-LHCR`. Historical (static): `2025`, `2019`, `2017`. Each version has `parameters.dat`, `covariance.dat`, `knots.dat`, `nuclei.dat` (current sets add `subleading.dat`, `fit_result.json`, and their own `solar_modulation.dat`). Shared Usoskin `solar_modulation.dat` in the parent `data/` directory (historical fallback). The `MODEL_VERSIONS` registry in `data_management.py` is the allow-list.
- `webapp/` - GSF Explorer (Pyodide + Preact). The **"Liquid Canvas" UI design is final** (maintainer-approved 2026-07-31): the plot is the content plane; Series/Settings/Export command panes, bottom display dock, dark+light tokens in `style.css`. Extend it — do not restyle or rebuild. `test_ui.py` (55 checks, incl. an iPad-size touch section: tap readout, pinch zoom, double-tap home, pane fit) asserts its invariants and is the acceptance gate for any webapp change. iPad/touch is a SUPPORTED target (maintainer, 2026-08-02) — touch regressions are release blockers.

## TEMPORARY pre-publication markers (strip at the GSF 2026 release)

- `webapp/chart.js` — `PRELIMINARY` const: diagonal watermark on the chart
  (and its SVG export). Set false / delete.
- `webapp/gsf_explorer.py` — `make_figure`: mirrored watermark on the
  publication-figure exports (PDF/SVG/PNG). Delete the marked `ax.text` block.
- `examples/*.py` — watermark hook on `plt.show` in each import cell
  (marked block, greps for `_gsf_preliminary`). Delete.
- `examples/*.py` + `docs.yml` — pyodide/micropip wheel-install cells and the
  `site/wheels/` build step, obsolete once the package is on PyPI.

## Key Concepts

- **4 element groups** (leaders): H (Z=1), He (Z=2), O* (Z=8), Fe* (Z=26). Subleading elements scale from their group leader.
- **286 total spline parameters** across all elements (2025 version). Parameter covariance stored as 10 block pairs between the 4 leaders.
- **Parameter trimming**: When building Jacobians/covariances, boundary parameters are trimmed: `params[1:-7]`, `cov[1:-3, 1:-3]` per element.
- **Nucleon flux**: `GSFEnergyPerNucleon.p_and_n_flux()` returns shape `(2, N)` for proton and neutron components. Each is summed over all 28 nuclei weighted by Z and A-Z.
- **Uncertainty propagation**: `Cov_flux = J @ Cov_params @ J.T` where J is the spline Jacobian.
- **Solar modulation**: Default is Solar Cycle 24 average (Dec 2008 - Dec 2019). Use `time_interval="LIS"` for unmodulated local interstellar spectrum.

## Example Notebooks

Located in `examples/` as **marimo notebooks** — plain `.py` files, the only
copy (no `.ipynb` exports; users convert with `marimo export ipynb` if they
want Jupyter). The docs workflow (docs.yml) compiles each one with
`marimo export html-wasm` to a browser tutorial at `/gallery/<name>/` on the
Pages site (cards in `docs/gallery.md`). Each notebook's import cell starts
with a TEMPORARY pyodide/micropip block installing the wheel from `/wheels/`
on the Pages site — remove these blocks (and the wheel step in docs.yml) once
globalsplinefit is on PyPI. Open interactively with
`uv run marimo edit examples/<name>.py`; running
`uv run python examples/<name>.py` executes the full cell DAG (that is the CI
smoke test in test.yml).

- `model_deck.py` - Standard model summary deck (spectrum, fractions, ln A
  moments, relative uncertainty, nucleon flux, flux correlation) plus a
  back-to-back version comparison. Replaces the removed
  `globalsplinefit.plotting` module.
- `cosmic_ray_flux.py` - Basic cosmic ray flux calculations and plotting.
- `nucleon_flux.py` - Nucleon flux for atmospheric shower simulations.
- `solar_modulation.py` - Solar modulation effects on flux.
- `rigidity_cutoff.py` - Geomagnetic rigidity cutoff effects.
- `model_comparison_2017_vs_2025.py` - Comparing 2017 and 2025 model versions.
- `reduced_model.py` - `ReducedGSF` pivot components: flux nuisance parameters
  with exact covariance penalty for downstream fits (daemonflux-style).

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
