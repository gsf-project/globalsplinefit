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

model = GSFEnergy()          # promoted default: 2026 (SIBYLL/EPOS mixture, GMD modulation)
model_uso = GSFEnergy(version="2026-USO")  # the alternative: same mixture, Usoskin 2017 modulation
model_uhe = GSFEnergy(version="2026-UHE-S23e")  # SIBYLL-2.3e only, for UHE/air-shower work
model_epos = GSFEnergy(version="2026-EPOS-LHCR")  # EPOS-LHC-R only (other half of the mixture)
model_2025 = GSFEnergy(version="2025")   # superseded historical release (not an alternative)
energy = np.logspace(0, 3, 100)  # 1 GeV to 1 TeV total energy

proton_flux = model.flux(energy, "p")
proton_error = model.error(energy, "p")
total_flux = model.total_flux(energy)

# Flux at Earth averaged over a time period (end month EXCLUSIVE):
# (200901, 201001) = calendar year 2009
flux_earth = model.flux(energy, "p", time_interval=(200901, 201001))
```

## Development

```bash
git clone https://github.com/gsf-project/globalsplinefit.git
cd globalsplinefit
pip install -e ".[dev]"
pytest tests/
```

## Links

- [Documentation](https://gsf-project.github.io/globalsplinefit/)
- [Issues](https://github.com/gsf-project/globalsplinefit/issues)
- [CHANGELOG](CHANGELOG.md)

## License

BSD 3-Clause. See [LICENSE](LICENSE).
