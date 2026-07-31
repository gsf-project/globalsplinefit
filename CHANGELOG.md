# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - Unreleased

### Changed

- **Model versions are now named by bare year — the `GSF` prefix is gone.**
  `GSF2026` → `2026` and `GSF2026-USO` → `2026-USO` (data directories,
  `version=` strings, `DEFAULT_VERSION`, docs, webapp). The historical sets
  were already year-named.
- **The shipping state is now documented and enforced by the registry**: after
  every promotion the four `current` sets (`2026`, `2026-USO`,
  `2026-UHE-S23e`, `2026-EPOS-LHCR`) are regenerated from the promoted run and
  re-shipped together; `2025`/`2019`/`2017` are static historical releases.
  All seven directories constitute the release-complete package.

### Added

- **`2026-UHE-S23e`** and **`2026-EPOS-LHCR`**: the single-interpretation
  Auger FD-2026 SIBYLL-2.3e and EPOS-LHC-R variants of the 2026 fit (GMD
  potential) — the two halves of the mixture, for applications that need one
  definite hadronic-interaction model rather than the mixture band.

- **The distributed GSF2026 / GSF2026-USO parameter sets are now the mixture
  fits.** Both previously shipped *single-interpretation* (Auger SIBYLL-2.3e
  only) fits, which were intermediate products of the analysis: `GSF2026` is now
  the equal-weight SIBYLL-2.3e/EPOS-LHC-R parameter-level mixture with the
  Ghelfi-Maurin-Derome potential, and `GSF2026-USO` is the same mixture with the
  Usoskin 2017 potential. Central all-particle flux moves by <0.02% below
  10^8 GeV and by ~1.7% at 10^10 GeV; the 1-sigma band widens above ~10^8 GeV
  (x1.3 at 10^10 GeV) because the mixture covariance carries the rank-one
  between-model term, and `mean_lnA` above 10^9 GeV shifts by up to +0.34.

### Added

- `DEFAULT_VERSION`, `MODEL_VERSIONS` and `version_info()`: an explicit registry
  of distributable versions, marking each `current` (the default and its one
  sanctioned alternative) or `historical` (a superseded release). Only registered
  directories are offered as versions, so an intermediate fit exported into
  `data/` cannot become distributable by accident -- unregistered directories
  warn instead. `get_available_versions(include_historical=False)` returns just
  the current sets
- `Parameters.provenance`: the `covering` (air-shower interpretation) and
  `solar_modulation_source` recorded with a parameter set, plus per-component fit
  quality for a mixture. `fit_result.json` now ships in the wheel/sdist, so this
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
  GSF2026 ships a version-local Ghelfi-Maurin-Derome table, while GSF2026-USO
  and the historical sets use the bundled Usoskin table
- Isotope-aware (Z, A) species keying + FitResult v2 format (deuteron as a
  sub-leading Z=1 species); global `energy_scale` model parameter
- Data-anchored power-law extrapolation of the sub-leading element abundances
  above their top knot: the member-to-leader flux ratio is tilted by a fitted
  spectral slope and normalization and saturates at `R_sat = 5` PV
  (`SUBLEADING_SAT_LNR`). Parameters ship in `data/<version>/subleading.dat`
  (`Z A norm slope`) for GSF2026 / GSF2026-USO. Previous releases held this
  ratio constant; legacy sets without `subleading.dat` fall back to that
  constant-ratio behavior bit-identically
- Data loading infrastructure with resource management
- PyPI packaging configuration
- Code quality tooling (ruff) and pre-commit hooks
- mkdocs documentation + tutorial notebooks
- CI/CD pipeline with GitHub Actions

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
