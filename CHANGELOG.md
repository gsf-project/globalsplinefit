# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-10-06

Nucleon fluxes change by about 1 %; nucleus fluxes are unchanged.

### Fixed

- Nucleon fluxes count nucleons by the integer mass number A. A nucleus
  carries A times the kinetic energy per nucleon, E = A (E_N - m_N) + m, and
  dE/dE_N = A. The atomic weight (1.008 for the proton) previously set the
  energy scale and the Jacobian; it now enters only the rest mass m. For
  2026.1 above 5 GeV per nucleon: proton nucleon flux +1.1 to +1.6 %,
  all-nucleon flux +0.8 to +1.3 %, n/p about -1 %; below 2 GeV per nucleon
  the change is under 1 %. Affects `GSFEnergyPerNucleon`,
  `GSFKineticEnergyPerNucleon` and `ReducedGSF`.
- A rigidity cutoff of 0 GV is no cutoff. The cutoff is a sigmoid in
  rigidity centred on the cutoff value, with width `cutoff_width` (default
  1 GV); at 0 GV it still suppressed low rigidities (2026.1 proton nucleon
  flux 38 % low at 1 GeV, 12 % at 2 GeV, 0.6 % at 5 GeV total energy per
  nucleon). The default remains no cutoff (`default_rigidity_cutoff=None`);
  cutoffs above 0 are unchanged. The user guide documents both arguments.

### Added

- Tests for the integer-A nucleon conversion and for the per-species
  rigidity cutoff in `GSFEnergy`, `GSFEnergyPerNucleon` and `ReducedGSF`.
  The 2017 nucleon-flux reference test rebuilds the table's legacy
  atomic-weight convention from the per-species flux (agreement 1e-5).

## [2.0.1] - 2026-09-29

### Changed

- GSF 2026 citation: `Fedynitch:2026ugq`
  ([arXiv:2609.32649](https://arxiv.org/abs/2609.32649),
  [InspireHEP](https://inspirehep.net/literature/3208618)) is the recommended
  reference in `CITATION.cff`, README, docs and Explorer.

### Fixed

- `version="2017"` loaded a later, unreleased parameter set. It now loads
  the October 2017 set (the fit tabulated in crflux `GlobalSplineFitBeta`,
  `GSF_spline_20171007`). Group fluxes change by up to tens of percent, and
  Ne moves to the O* group. The 2017 reference tables in `tests/data/` were
  regenerated with the original `flux.py`, and `reduced_pivots.dat` was
  re-optimized (coverage 1.195). Pin `globalsplinefit==2.0.0` to reproduce
  earlier `version="2017"` results.

## [2.0.0] - 2026-09-26

First release of `globalsplinefit` as a standalone package, accompanying the
GSF 2026 paper. It replaces the functional interface of the 1.x releases.

### Added

- **GSF 2026 model, revision 1** (`DEFAULT_VERSION = "2026.1"`): the
  equal-weight parameter-level mixture of the Auger FD-2026 SIBYLL-2.3e and
  EPOS-LHC-R interpretations with the Ghelfi-Maurin-Derome solar-modulation
  potential. Variants: `2026.1-USO` (Usoskin 2017 potential), `2026.1-SIB23e`
  and `2026.1-EPOSLHCR` (the two single-interpretation halves of the mixture).
  The mixture covariance carries the rank-one between-model term.
- **Model version registry.** Names follow
  `<line>.<revision>[-<physics classifier>]`; unrevisioned names (`"2026"`,
  `"2026-USO"`) resolve to the newest revision via `resolve_version()`, and
  revisioned names pin one. `MODEL_VERSIONS` and `version_info()` describe
  every distributable set as `current` or `historical`; only registered data
  directories are offered. The historical releases `2025`, `2019` and `2017`
  remain available under their bare names. `Model.version` reports the
  resolved name, and `__version__` the code release.
- **Model classes** for every common abscissa: `GSFEnergy`,
  `GSFKineticEnergy`, `GSFRigidity`, `GSFEnergyPerNucleon`,
  `GSFKineticEnergyPerNucleon`, with full covariance propagation (`error`,
  `covariance`, `jacobian`) for elements, mass groups (`H*`, `He*`, `O*`,
  `Fe*`) and the all-particle flux, plus composition moments
  (`mean_lnA`, `var_lnA`) and fractions with uncertainties.
- **`ReducedGSF`**: a compact nuisance-parameter representation of the
  nucleon-flux uncertainty (relative flux deviations of protons and neutrons
  at 12 pivot energies, exact covariance at the pivots), with `sample` and
  `penalty` for downstream fits and `optimize_pivots` for custom grids. Each
  parameter set ships its own optimized `reduced_pivots.dat`.
- **`sample()`**: pseudo-experiments drawn from the native amplitude
  covariance on every model class; identically seeded calls share one
  amplitude draw across targets.
- **Solar modulation** (force field, monthly potential tables) on all model
  classes, including time-interval averages. Each parameter set is
  re-modulated with the potential it was demodulated with.
- **Isotope-aware species** keyed by (Z, A), including the deuteron as a
  sub-leading Z=1 species; a global `energy_scale` parameter.
- **Sub-leading extrapolation**: above their top knot, sub-leading elements
  follow a fitted power-law tilt of the member-to-leader ratio that saturates
  at 5 PV (`subleading.dat`).
- **Provenance** (`Parameters.provenance`, `fit_result.json`): covering,
  solar-modulation source and fit quality travel with each parameter set.
- **GSF Explorer**: an in-browser (Pyodide) web app for plotting and
  exporting fluxes, fractions and uncertainties, and a tutorial gallery of
  marimo notebooks that run in the browser.

### Changed

- **BREAKING**: the functional interface of 1.x is replaced by the model
  classes above (`from globalsplinefit import GSFEnergy`).
- **BREAKING**: data files moved to `src/globalsplinefit/data/<version>/`.
- Kinetic-energy conversion uses each species' own rest mass (previously the
  group leader's): kinetic fluxes change by up to 3.9% below 1 TeV.
- Nucleon counting uses integer proton and neutron numbers; the previous
  float arithmetic gave hydrogen 0.008 spurious neutrons (total nucleon flux
  0.37-0.69% lower).

### Removed

- The legacy `flux.py` functional interface, global module state and
  root-level data files.

## [1.x.x] - Previous versions

Functional interface (`flux.py`).
