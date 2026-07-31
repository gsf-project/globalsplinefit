# GSF - Global Spline Fit

## Project Overview

Parametric model for cosmic ray flux and composition based on cubic B-spline fits to observational data. Provides separate proton/neutron nucleon fluxes with full uncertainty propagation via Jacobian-transformed parameter covariances.

## Architecture

- `src/globalsplinefit/model.py` - Core model classes: `GSFEnergy`, `GSFKineticEnergy`, `GSFRigidity`, `GSFEnergyPerNucleon`, `GSFKineticEnergyPerNucleon`. All inherit from `GSFBase`.
- `src/globalsplinefit/data_management.py` - `Parameters` class loads spline knots, coefficients, covariances, nuclear data, and solar modulation from `data/{version}/` directories.
- `src/globalsplinefit/pca.py` - `HybridPCA` class for low-rank PCA decomposition of flux covariances.
- `src/globalsplinefit/data/` - Data files, one directory per model version (year-only names, no "GSF" prefix). Current sets, regenerated together after every promotion: `2026` (default; mixture + GMD), `2026-USO`, `2026-UHE-S23e`, `2026-EPOS-LHCR`. Historical (static): `2025`, `2019`, `2017`. Each version has `parameters.dat`, `covariance.dat`, `knots.dat`, `nuclei.dat` (current sets add `subleading.dat`, `fit_result.json`, and their own `solar_modulation.dat`). Shared Usoskin `solar_modulation.dat` in the parent `data/` directory (historical fallback). The `MODEL_VERSIONS` registry in `data_management.py` is the allow-list.
- `webapp/` - GSF Explorer (Pyodide + Preact). The **"Liquid Canvas" UI design is final** (maintainer-approved 2026-07-31): the plot is the content plane; Series/Settings/Export command panes, bottom display dock, dark+light tokens in `style.css`. Extend it — do not restyle or rebuild. `test_ui.py` (50 checks) asserts its invariants and is the acceptance gate for any webapp change.

## Key Concepts

- **4 element groups** (leaders): H (Z=1), He (Z=2), O* (Z=8), Fe* (Z=26). Subleading elements scale from their group leader.
- **286 total spline parameters** across all elements (2025 version). Parameter covariance stored as 10 block pairs between the 4 leaders.
- **Parameter trimming**: When building Jacobians/covariances, boundary parameters are trimmed: `params[1:-7]`, `cov[1:-3, 1:-3]` per element.
- **Nucleon flux**: `GSFEnergyPerNucleon.p_and_n_flux()` returns shape `(2, N)` for proton and neutron components. Each is summed over all 28 nuclei weighted by Z and A-Z.
- **Uncertainty propagation**: `Cov_flux = J @ Cov_params @ J.T` where J is the spline Jacobian.
- **Solar modulation**: Default is Solar Cycle 24 average (Dec 2008 - Dec 2019). Use `time_interval="LIS"` for unmodulated local interstellar spectrum.

## Example Notebooks

Located in `examples/`:
- `cosmic_ray_flux.ipynb` - Basic cosmic ray flux calculations and plotting.
- `nucleon_flux.ipynb` - Nucleon flux for atmospheric shower simulations.
- `solar_modulation.ipynb` - Solar modulation effects on flux.
- `rigidity_cutoff.ipynb` - Geomagnetic rigidity cutoff effects.
- `model_comparison_2017_vs_2025.ipynb` - Comparing 2017 and 2025 model versions.
- `HybridPCA.ipynb` - Hybrid low-rank + diagonal PCA for uncertainty propagation. Outputs `PCA_Hybrid_GSF_2025.pkl`.

## Development

```bash
pip install -e ".[dev]"         # Install with dev dependencies
pytest tests/                    # Run tests
pytest tests/ -m "not slow"     # Skip slow tests
```

- Python >= 3.10, dependencies: numpy, scipy
- Linting: ruff (line-length 88)
- Tests: pytest with markers (slow, unit, integration, regression)

## Pre-commit checklist

Before committing any changes, always run the following and fix any issues:

```bash
ruff check src tests            # Linting (must pass with zero errors)
ruff format --check src tests   # Formatting check
```

This matches the CI Code Quality workflow and prevents lint/format failures on push.

## Current Branch

`major_version_gsf2025` - Major version update with 2025 data, low-rank + diagonal PCA approach, and new nucleon flux examples.
