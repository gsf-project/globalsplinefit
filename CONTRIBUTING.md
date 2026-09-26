# Contributing to globalsplinefit

Contributions are welcome: bug reports, fixes, documentation, examples, and
new features. This guide is written for human contributors and for AI coding
agents alike; agents should read it in full before editing anything. The
short version is at the end ("Checklist").

## What this repository is

`globalsplinefit` evaluates the **Global Spline Fit (GSF)**, a data-driven
model of the cosmic-ray flux and mass composition from ~1 GeV to 10^11 GeV.
It does **not** fit data. The fit is done elsewhere and its output (spline
knots, amplitudes, covariance) is shipped here as data files; this package
turns those into fluxes, composition, and uncertainties. Changes to the
physics of a parameter set therefore arrive as new data files, not as code.

## Repository map

| Path | Contents |
|---|---|
| `src/globalsplinefit/model.py` | Model classes: `GSFEnergy`, `GSFKineticEnergy`, `GSFRigidity`, `GSFEnergyPerNucleon`, `GSFKineticEnergyPerNucleon`, all derived from `GSFBase`. Flux, error, covariance, Jacobian, composition moments, sampling, solar modulation. |
| `src/globalsplinefit/data_management.py` | `Parameters` (loads a data directory), the version registry `MODEL_VERSIONS`, `DEFAULT_VERSION`, `resolve_version()`, `version_info()`. |
| `src/globalsplinefit/reduced.py` | `ReducedGSF` (pivot-based nuisance parameters for the nucleon flux) and `optimize_pivots`. |
| `src/globalsplinefit/data/<version>/` | One directory per parameter set; see `data/README.md`. Generated files: do not hand-edit. |
| `tests/` | pytest suite, including regression tests against reference fluxes in `tests/data/`. |
| `examples/` | Tutorials as [marimo](https://marimo.io) notebooks (plain `.py` files); also built into the browser gallery. |
| `webapp/` | GSF Explorer, an in-browser app (Pyodide + Preact). `webapp/README.md` documents it. |
| `docs/`, `mkdocs.yml`, `mkdocs_hooks/` | Documentation site (mkdocs-material + mkdocstrings). |
| `webapp/citations.json` | Single source of truth for citation metadata (docs and Explorer both read it). |
| `.github/workflows/` | CI: tests, lint, docs + gallery + Explorer deploy, release. |

## Key concepts

- **Mass groups.** Four leading elements carry their own splines: H (Z=1),
  He (Z=2), O (Z=8), Fe (Z=26). The groups are written H\*, He\*, O\*, Fe\*;
  the target keys are `"H*"`, `"He"`, `"O*"`, `"Fe*"`. `"p"`, `"O"`, `"Fe"` are
  single elements. Sub-leading elements follow their group leader through
  fitted ratios (and a power-law tilt above their top knot, `subleading.dat`).
- **Species keys** are `(Z, A)` tuples, so isotopes (the deuteron,
  `(1, 2.014)`) are distinct species.
- **Uncertainties** come from linear propagation, `Cov_flux = J Cov_par J^T`,
  with `J` the spline Jacobian. Only the group-leader blocks of
  `covariance.dat` enter propagation.
- **Pinned parameters.** The fit pins a coefficient by zeroing its covariance
  row and column; the pinned indices are not a contiguous slice. Always
  contract the full amplitude range; never trim positionally.
  `tests/test_reduced.py::TestPinningConvention` guards this.
- **Solar modulation.** Fluxes are modulated with the force-field
  approximation using monthly potential tables. The default is the Solar
  Cycle 24 average; `time_interval="LIS"` gives the local interstellar
  spectrum. Each parameter set is re-modulated with the potential it was
  demodulated with.
- **Nucleon flux.** `GSFEnergyPerNucleon.p_and_n_flux()` returns the proton
  and neutron components (shape `(2, N)`), summed over all nuclei with
  integer proton and neutron counts.
- **Versions.** Names are `<line>.<revision>[-<classifier>]` (`2026.1`,
  `2026.1-USO`, ...). `"2026"` resolves to the newest revision. The registry
  in `data_management.py` is the allow-list: an unregistered data directory
  is never offered as a version. `2025`, `2019`, `2017` are historical.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/); `uv.lock` is committed.

```bash
git clone https://github.com/gsf-project/globalsplinefit.git
cd globalsplinefit
uv sync --all-extras            # .venv with every optional dependency
uv run pytest tests/            # full suite (~30 s)
uv run pytest tests/ -m "not slow"
uv run pre-commit install       # hooks run on every commit
uv run pre-commit run --all-files   # what CI's lint job runs
```

Python >= 3.10; runtime dependencies are numpy and scipy only. Style is
enforced by ruff (line length 88) through the pre-commit hooks, which call the
project's own `uv run ruff`.

## Making changes

- **Library code** (`src/`): add or update tests in `tests/`, keep docstrings
  in numpy style (they render into the API reference), and keep the public
  API in `__init__.py`'s `__all__` consistent with the docs.
- **Numerical behaviour**: the regression tests compare against reference
  outputs. If a change moves numbers on purpose, say so in the pull request,
  quantify it, and update the reference together with an explanation.
- **Data files** (`src/globalsplinefit/data/`): generated by the fit; the
  pre-commit whitespace hooks deliberately skip them. Do not reformat or
  hand-edit them. A new parameter set needs a registry entry in
  `MODEL_VERSIONS`, a `reduced_pivots.dat`, a citation mapping in
  `webapp/citations.json`, and tests.
- **Examples** (`examples/*.py`, marimo): run `uv run python
  examples/<name>.py` (executes all cells; CI does this). For the browser
  gallery, a plot cell must end with the figure (`show(fig)`), because
  `plt.show()` renders nothing in marimo's app view; UI controls go in their
  own cell; y-limits that depend on a control use `autoscale()`. The
  headless-browser check `.github/scripts/test_gallery.py` is the acceptance
  test for gallery changes.
- **Explorer** (`webapp/`): the UI design is settled; extend it rather than
  restyling it. `python webapp/test_ui.py` (playwright) is the acceptance test
  and includes touch-device checks; touch regressions are bugs.
- **Documentation**: `uv run mkdocs build --strict` must pass.
- **Changelog**: add an entry under an "Unreleased" heading in
  `CHANGELOG.md` for user-visible changes.

## Pull requests

Open an issue first for anything larger than a fix. Keep pull requests
focused, describe the motivation and the observable effect, and make sure CI
is green (tests on Linux, macOS and Windows for Python 3.10-3.13, lint, docs).

## Notes for AI coding agents

- Read this file, `README.md`, and the docstring of the module you are
  changing before editing. `docs/user_guide.md` explains the physics-facing
  behaviour users rely on.
- Verify claims against the code: run the tests, or evaluate the model in a
  short script, rather than inferring numbers.
- Stay inside the task. Do not reformat unrelated code, regenerate data
  files, rename public API, or change numerical defaults unless asked.
- Physics conventions matter more than code style here: energies are in GeV
  (total energy per particle unless the class says otherwise), rigidities in
  GV, fluxes in (m^2 s sr GeV)^-1 or per GV, and the default time interval
  applies solar modulation. When unsure which convention a function uses,
  read its docstring and test.
- Report what you verified and what you did not.

## Checklist

1. `uv run pytest tests/` passes.
2. `uv run pre-commit run --all-files` passes.
3. Examples or Explorer touched? Their acceptance tests pass.
4. Docs build with `uv run mkdocs build --strict`.
5. `CHANGELOG.md` updated for user-visible changes.

## License

By contributing you agree that your contribution is released under the
repository's BSD 3-Clause license.
