# GlobalSplineFit

[![PyPI version](https://badge.fury.io/py/globalsplinefit.svg)](https://badge.fury.io/py/globalsplinefit)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Tests](https://github.com/gsf-project/globalsplinefit/workflows/Tests/badge.svg)](https://github.com/gsf-project/globalsplinefit/actions)

Parametric model of cosmic ray flux and composition based on cubic B-spline fits to observational data. Covers all elements Z=1--28, with full covariance propagation and solar modulation.

## Installation

```bash
pip install globalsplinefit
```

## Quick Start

```python
from globalsplinefit import GSFEnergy
import numpy as np

model = GSFEnergy()          # default 2026 fit
model_uso = GSFEnergy(version="2026-USO")  # Usoskin 2017 modulation
model_s23e = GSFEnergy(version="2026-SIB23e")  # SIBYLL-2.3e-only interpretation
model_epos = GSFEnergy(version="2026-EPOSLHCR")  # EPOS-LHC-R only (other half of the mixture)
model_2025 = GSFEnergy(version="2025")   # historical release
energy = np.logspace(0, 3, 100)  # 1 GeV to 1 TeV total energy

proton_flux = model.flux(energy, "p")
proton_error = model.error(energy, "p")
total_flux = model.total_flux(energy)

# Flux at Earth averaged over a time period (end month EXCLUSIVE):
# (200901, 201001) = calendar year 2009
flux_earth = model.flux(energy, "p", time_interval=(200901, 201001))
```

## Examples

Try the model without installing anything: the
[GSF Explorer](https://gsf-project.github.io/globalsplinefit/explorer/) and the
[tutorial gallery](https://gsf-project.github.io/globalsplinefit/gallery/)
run in the browser.

The tutorials in `examples/` are [marimo](https://marimo.io) notebooks — plain
Python files you can open interactively or run as scripts:

```bash
uv run marimo edit examples/reduced_model.py   # interactive
uv run python examples/reduced_model.py        # run top to bottom
```

Need a Jupyter notebook? `uvx marimo export ipynb examples/<name>.py -o <name>.ipynb`.

## Development

Development uses [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/gsf-project/globalsplinefit.git
cd globalsplinefit
uv sync --all-extras     # creates .venv with all optional dependencies
uv run pytest tests/
uv run pre-commit install
```

## Links

- [Documentation](https://gsf-project.github.io/globalsplinefit/)
- [Issues](https://github.com/gsf-project/globalsplinefit/issues)
- [CHANGELOG](CHANGELOG.md)

## License

BSD 3-Clause. See [LICENSE](LICENSE).
