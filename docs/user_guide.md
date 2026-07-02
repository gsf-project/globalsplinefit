# User Guide

Comprehensive guide to using GlobalSplineFit for cosmic ray flux calculations.

## Overview

GlobalSplineFit (GSF) provides accurate parameterizations of cosmic ray flux and composition based on spline fits to experimental data. The package includes uncertainty quantification and supports various input parameters.

## Installation

Install GlobalSplineFit using pip:

```bash
pip install globalsplinefit
```

For development installation:

```bash
git clone https://github.com/gsf-project/gsf.git
cd gsf
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from globalsplinefit import GSFEnergy

# Initialize the model
gsf = GSFEnergy()

# Define energy range (GeV)
energy = np.logspace(1, 6, 100)

# Calculate proton flux
proton_flux = gsf.flux(energy, "p")

# Calculate uncertainties
proton_error = gsf.error(energy, "p")

# Print results
print(f"Energy range: {energy[0]:.1e} - {energy[-1]:.1e} GeV")
print(f"Flux range: {proton_flux.min():.2e} - {proton_flux.max():.2e}")
print(f"Relative error: {(proton_error/proton_flux).mean()*100:.1f}%")
```

## Model Classes

### GSFEnergy

The [`GSFEnergy`][globalsplinefit.GSFEnergy] class calculates flux as a function of total energy per nucleus.

**Input**: Total energy per nucleus [GeV]
**Output**: Differential flux [particles/(m² s sr GeV)]

```python
from globalsplinefit import GSFEnergy
import numpy as np

gsf = GSFEnergy()
energy = np.logspace(1, 6, 100)  # 10 GeV to 1 PeV
flux = gsf.flux(energy, "p")  # Proton flux
```

### GSFEnergyPerNucleon

The [`GSFEnergyPerNucleon`][globalsplinefit.GSFEnergyPerNucleon] class calculates nucleon flux for atmospheric shower simulations.

**Input**: Total energy per nucleon [GeV/nucleon] (use
[`GSFKineticEnergyPerNucleon`][globalsplinefit.GSFKineticEnergyPerNucleon]
for kinetic energy per nucleon)
**Output**: Nucleon flux [nucleons/(m² s sr GeV)]

```python
from globalsplinefit import GSFEnergyPerNucleon

gsf_nucleon = GSFEnergyPerNucleon()
energy_per_nucleon = np.logspace(0, 6, 100)
nucleon_flux = gsf_nucleon.total_flux(energy_per_nucleon)
```

### GSFRigidity

The [`GSFRigidity`][globalsplinefit.GSFRigidity] class works in magnetic rigidity space with solar modulation.

**Input**: Magnetic rigidity [GV]
**Output**: Modulated flux [particles/(m² s sr GV)]

```python
from globalsplinefit import GSFRigidity

gsf_rigidity = GSFRigidity()
rigidity = np.logspace(0, 3, 100)  # 1 GV to 1 TV
flux = gsf_rigidity.flux(rigidity, "p")
```

## Particle Groups

All models support these cosmic ray groups:

| Group | Description | Atomic Numbers |
|-------|-------------|----------------|
| `"p"` | Proton group | Z = 1 |
| `"He"` | Helium group | Z = 2 |
| `"O"` | Oxygen group | Z = 3--9 |
| `"Fe"` | Iron group | Z = 10--28 |

## Uncertainty Quantification

GSF provides full uncertainty quantification including correlations:

```python
# Calculate flux and uncertainties
flux = gsf.flux(energy, "p")
error = gsf.error(energy, "p")

# Relative uncertainty
rel_error = error / flux

# Covariance matrix (for advanced users)
cov_matrix = gsf.covariance("p", "He", energy)
```

## Solar Modulation

All models support time-dependent solar modulation via the `time_interval` parameter:

```python
# Local interstellar spectrum (no modulation)
flux_lis = gsf.flux(energy, "p", time_interval="LIS")

# Specific time period (YYYYMM format)
flux_2009 = gsf.flux(energy, "p", time_interval=(200901, 201001))  # end EXCLUSIVE: calendar year 2009

# Default: Solar Cycle 24 average (Dec 2008 - Dec 2019)
flux_default = gsf.flux(energy, "p")
```

See the [Solar Modulation tutorial](examples/solar_modulation.ipynb) for detailed examples.

## Geomagnetic Rigidity Cutoff

Apply a geomagnetic cutoff to suppress low-rigidity cosmic rays:

```python
# Cutoff at 20 GV (smooth sigmoid transition by default, cutoff_width=1 GV;
# construct the model with cutoff_width=0.0 for a sharp Heaviside cutoff)
flux_cut = gsf.flux(energy, "p", rigidity_cutoff=20.0)
```

See the [Rigidity Cutoff tutorial](examples/rigidity_cutoff.ipynb) for more details.

## Performance Considerations

- **Vectorization**: All methods support vectorized calculations
- **Energy ranges**: The fit spans ~1 GeV -- 10^11 GeV total energy per nucleus
- **Caching**: Jacobian matrices are cached for repeated calculations
- **Memory**: Consider chunking for very large arrays (>10^6 points)

## Data Export

Export results for use in other applications:

```python
import numpy as np

# Prepare data
data = np.column_stack([energy, flux, error])

# Save to file
np.savetxt("cosmic_ray_flux.dat", data,
          header="energy[GeV] flux[1/(GeV m2 s sr)] error[1/(GeV m2 s sr)]",
          fmt="%.6e")
```

## Advanced Features

### Jacobian Access

For sensitivity studies and error propagation:

```python
jacobian = gsf.jacobian(energy, "p")
print(f"Jacobian shape: {jacobian.shape}")
```

### Model Information

Access model metadata:

```python
# Available groups
print(f"Particle groups: {gsf.GROUP_NAMES}")
```

## Best Practices

1. **Choose the right model**: Use `GSFEnergy` for most applications, `GSFEnergyPerNucleon` for shower simulations
2. **Vectorize calculations**: Pass arrays instead of loops for better performance
3. **Handle uncertainties**: Always consider flux uncertainties in your analysis
4. **Energy ranges**: Stay within the fitted energy range (~1 GeV -- 10^11 GeV per nucleus); the flux is zero below the first knot and extrapolated above the last
5. **Solar modulation**: Use time intervals for time-dependent studies

## Tutorials

See the tutorials for detailed, worked examples:

- [Cosmic Ray Flux](examples/cosmic_ray_flux.ipynb) -- Basic flux calculations
- [Nucleon Flux](examples/nucleon_flux.ipynb) -- Nucleon flux for simulations
- [Solar Modulation](examples/solar_modulation.ipynb) -- Solar modulation effects
- [Rigidity Cutoff](examples/rigidity_cutoff.ipynb) -- Geomagnetic cutoff effects
- [Model Comparison](examples/model_comparison_2017_vs_2025.ipynb) -- Comparing model versions
- [Hybrid PCA](examples/HybridPCA.ipynb) -- Dimensionality reduction for uncertainties
