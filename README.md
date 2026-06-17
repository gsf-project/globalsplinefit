# GlobalSplineFit

[![PyPI version](https://badge.fury.io/py/globalsplinefit.svg)](https://badge.fury.io/py/globalsplinefit)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Tests](https://github.com/gsf-project/gsf/workflows/Tests/badge.svg)](https://github.com/gsf-project/gsf/actions)

Parametric model of cosmic ray flux and composition based on cubic B-spline fits to observational data. Covers all elements Z=1--28, with full covariance propagation and solar modulation.

## Installation

```bash
pip install globalsplinefit
```

## Quick Start

```python
from globalsplinefit import GSFEnergy
import numpy as np

model = GSFEnergy()
energy = np.logspace(0, 3, 100)  # 1 GeV to 1 TeV total energy

proton_flux = model.flux(energy, "p")
proton_error = model.error(energy, "p")
total_flux = model.total_flux(energy)

# Flux at Earth during a specific time period
flux_earth = model.flux(energy, "p", time_interval=(200901, 200912))
```

## Development

```bash
git clone https://github.com/gsf-project/gsf.git
cd gsf
pip install -e ".[dev]"
pytest tests/
```

## Links

- [Documentation](https://gsf.readthedocs.io)
- [Issues](https://github.com/gsf-project/gsf/issues)
- [CHANGELOG](CHANGELOG.md)

## License

BSD 3-Clause. See [LICENSE](LICENSE).
