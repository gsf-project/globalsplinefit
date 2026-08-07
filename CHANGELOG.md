# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - Unreleased

### Added

- **Revisioned model versions.** Physical model versions are now named
  `<line>.<revision>[-<variant>]`: the 2026 family ships as `2026.0`,
  `2026.0-USO`, `2026.0-S23e`, `2026.0-EPOS-LHCR`, and a data or fit patch
  to a released line is published as a new revision (`2026.1`, ...) next to
  the old one. Unrevisioned names (`"2026"`, `"2026-USO"`) resolve to the
  newest registered revision via the new `resolve_version()`; revisioned
  names pin one. Historical releases (`2025`, `2019`, `2017`) keep their
  bare names. `version_info()` now includes the resolved `name`, and the
  package exposes `__version__` (the code release, distinct from the model
  version). Explorer CSV exports record both: the physical model version
  including its revision and the `globalsplinefit` code version.

- **`GSFBase.sample()` — pseudo-experiments from the native covariance.**
  Every model class can draw random flux realizations,
  `model.sample(energy, target, n_samples)`, with `target=None` giving the
  all-particle total. The flux is linear in the spline amplitudes, so each
  draw is a genuine model realization with the exact covariance at every
  energy and all cross-group correlations intact. One shared amplitude draw
  underlies all targets: identically seeded calls return the same
  pseudo-experiments (group draws sum to the all-particle draw exactly).
  Recommended for ensemble error propagation when nuclei fluxes or their
  correlation with the all-particle flux matter; Jacobian-based propagation
  via `ReducedGSF` remains the recommendation for nucleon fluxes.

### Changed

- **Docs consolidated into one Pages site; tutorials now run in the browser.**
  The landing page links the GSF Explorer (`/explorer/`), the tutorial gallery
  (`/gallery/<name>/` — `marimo export html-wasm` builds of `examples/*.py`,
  editable in the browser), the user guide, a new Getting Started page, and
  the API reference. The committed `.ipynb` exports, the `marimo-export`
  pre-commit hook, the CI freshness check, and the mkdocs-jupyter plugin are
  gone — the marimo `.py` files are the only notebook format (convert with
  `marimo export ipynb` for Jupyter). Until the package is on PyPI, the WASM
  notebooks install a wheel published under `/wheels/` on the docs site.
- **Removed `globalsplinefit.plotting` and the `plotting` extra.** The module
  was user-facing example code with no consumers inside the package; its deck
  and comparison figures live on as the interactive `examples/model_deck.py`
  notebook (browser version in the tutorial gallery). matplotlib moved into
  the `examples` extra.
- **Model versions are now named by bare year — the `GSF` prefix is gone.**
  `GSF2026` → `2026` and `GSF2026-USO` → `2026-USO` (data directories,
  `version=` strings, `DEFAULT_VERSION`, docs, webapp). The historical sets
  were already year-named.
- **The release state is documented and enforced by the registry**: after
  every promotion the four `current` sets (`2026`, `2026-USO`,
  `2026-S23e`, `2026-EPOS-LHCR`) are regenerated together from the same
  fit; `2025`/`2019`/`2017` are static historical releases.
  All seven directories constitute the release-complete package.

### Added

- **`2026-S23e`** and **`2026-EPOS-LHCR`**: the single-interpretation
  Auger FD-2026 SIBYLL-2.3e and EPOS-LHC-R variants of the 2026 fit (GMD
  potential) — the two halves of the mixture, for applications that need one
  definite hadronic-interaction model rather than the mixture band.

- **The distributed GSF2026 / GSF2026-USO parameter sets are now the mixture
  fits.** Both previously distributed *single-interpretation* (Auger SIBYLL-2.3e
  only) fits, which were intermediate products of the analysis: `GSF2026` is now
  the equal-weight SIBYLL-2.3e/EPOS-LHC-R parameter-level mixture with the
  Ghelfi-Maurin-Derome potential, and `GSF2026-USO` is the same mixture with the
  Usoskin 2017 potential. Central all-particle flux moves by <0.02% below
  10^8 GeV and by ~1.7% at 10^10 GeV; the 1-sigma band widens above ~10^8 GeV
  (x1.3 at 10^10 GeV) because the mixture covariance carries the rank-one
  between-model term, and `mean_lnA` above 10^9 GeV shifts by up to +0.34.

### Added

- `DEFAULT_VERSION`, `MODEL_VERSIONS` and `version_info()`: an explicit registry
  of distributable versions, marking each `current` (the default and its
  variants) or `historical` (an earlier release). Only registered
  directories are offered as versions, so an intermediate fit exported into
  `data/` cannot become distributable by accident -- unregistered directories
  warn instead. `get_available_versions(include_historical=False)` returns just
  the current sets
- `Parameters.provenance`: the `covering` (air-shower interpretation) and
  `solar_modulation_source` recorded with a parameter set, plus per-component fit
  quality for a mixture. `fit_result.json` is included in the wheel/sdist, so this
  provenance travels with an install
- `Model.version` now reports the version actually **resolved** -- a model built
  with no arguments reports `"GSF2026"` rather than `None`

- Complete rewrite of GSF model as modern Python package
- Object-oriented model classes (`GSFEnergy`, `GSFRigidity`, `GSFEnergyPerNucleon`, etc.) replacing functional interface
- Modern project structure with `src/globalsplinefit/` layout
- Comprehensive test suite with pytest (regression-anchored against the
  2017 reference fluxes/errors)
- Type hints throughout codebase
- Solar modulation (force-field, monthly phi table) on all model classes. Each
  parameter set is re-modulated with the potential it was demodulated with:
  GSF2026 includes a version-local Ghelfi-Maurin-Derome table, while GSF2026-USO
  and the historical sets use the bundled Usoskin table
- Isotope-aware (Z, A) species keying + FitResult v2 format (deuteron as a
  sub-leading Z=1 species); global `energy_scale` model parameter
- Data-anchored power-law extrapolation of the sub-leading element abundances
  above their top knot: the member-to-leader flux ratio is tilted by a fitted
  spectral slope and normalization and saturates at `R_sat = 5` PV
  (`SUBLEADING_SAT_LNR`). Parameters are in `data/<version>/subleading.dat`
  (`Z A norm slope`) for GSF2026 / GSF2026-USO. Previous releases held this
  ratio constant; legacy sets without `subleading.dat` fall back to that
  constant-ratio behavior bit-identically
- Data loading infrastructure with resource management
- PyPI packaging configuration
- Code quality tooling (ruff) and pre-commit hooks
- mkdocs documentation + tutorial notebooks
- CI/CD pipeline with GitHub Actions

### Added

- **Per-version reduced-pivot tables** are distributed as
  `data/<version>/reduced_pivots.dat` (plain text, one energy per line, the
  `optimize_pivots` parameters in the header). An all-default `ReducedGSF`
  uses its bundle's grid (`model.params.reduced_pivots`), and every version
  carries its own optimized table. Worst-case coverage vs the previously
  shared 2026 grid: 1.26/1.36/1.27 (was 1.28/1.53/1.38) for
  2026-USO/-S23e/-EPOS-LHCR and 1.24/1.55/1.21 (was 2.76/2.45/1.78) for
  2025/2019/2017. Pivots are version-specific, so theta components do not
  line up 1:1 across versions. Custom bundles without a table raise; derive
  a grid with `optimize_pivots` and pass it via `pivot_energies=`.

### Fixed

- **Kinetic-energy conversion uses each species' own rest mass.** Previously
  every member of a mass group was evaluated with the group leader's rest
  mass. Kinetic fluxes change by up to 3.9% below 1 TeV and by <0.1% above
  10 TeV; total-energy results are unaffected.
- **Nucleon counting separated from isotope mass.** The energy-variable
  Jacobian keeps the real isotope mass, but proton/neutron numbers are now
  integer counts. The old float arithmetic assigned a small spurious neutron
  contribution to protons (e.g. A=1.008 gave hydrogen 0.008 neutrons); total
  nucleon flux is 0.37-0.69% lower, its uncertainty 0.40-0.76% lower.
- **GSF Explorer energy-scale factor.** The Explorer's multiplicative
  energy-scale input (1 = unchanged) was assigned directly to the model's
  *fractional-shift* `energy_scale` property (0 = unchanged), so every
  Explorer evaluation ran at doubled energy. The bridge now maps the factor
  correctly (`energy_scale = factor - 1`).

### Changed

- **BREAKING**: Replaced functional interface with specialized model classes
- **BREAKING**: Moved data files to `src/globalsplinefit/data/{version}/` directory
- **BREAKING**: Changed import structure to `from globalsplinefit import GSFEnergy`
- Improved error messages and validation
- Enhanced solar modulation parameter handling
- Better separation of concerns across modules

### Removed

- Legacy `flux.py` functional interface
- Direct access to global variables
- Root-level data files

## [1.x.x] - Previous Versions

Legacy versions with functional interface. See git history for details.
