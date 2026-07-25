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

## Model Versions

All model classes accept a `version` parameter selecting the fitted parameter set:

```python
gsf = GSFEnergy()                        # promoted default: GSF2026
gsf_uso = GSFEnergy(version="GSF2026-USO")   # the one sanctioned alternative
gsf_2025 = GSFEnergy(version="2025")     # superseded historical release
```

There is **one promoted model** and **one alternative to it**. Both describe the
same fit and differ only in the solar modulation potential; everything else in
the table is a superseded release kept so that older work can be reproduced.

| Version | Status | Covering | Modulation | Description |
|---------|--------|----------|------------|-------------|
| `"GSF2026"` | **current -- default** | SIBYLL-2.3e / EPOS-LHC-R mixture | Ghelfi--Maurin--Derome (**baseline**) | **The promoted model.** Use this unless you have a specific reason not to |
| `"GSF2026-USO"` | current -- alternative | SIBYLL-2.3e / EPOS-LHC-R mixture | Usoskin 2017 (alternative) | The same fit with the other potential; use it to gauge the solar-modulation systematic |
| `"2025"` | historical | superseded | Usoskin | Previous published release |
| `"2019"` | historical | superseded | Usoskin | Legacy published release |
| `"2017"` | historical | superseded | Usoskin | Original GSF release (Dembinski et al. 2017) |

The historical sets are **not** alternatives to the current fit: they are earlier
published releases, retained only for reproducing results that cited them.

`get_available_versions(include_historical=False)` returns just the two current
sets, and `version_info(version)` reports any version's status, covering and
modulation potential. A loaded model also carries its own provenance:

```python
gsf.version                      # -> "GSF2026", the version actually loaded
gsf.params.provenance["covering"]                  # the air-shower interpretation
gsf.params.provenance["solar_modulation_source"]   # GMD or USO
```

### What "covering" means, and why both current sets are a mixture

Above ~10^8 GeV the mass composition inferred from air-shower data depends on the
hadronic interaction model used to interpret it. Rather than committing to one,
both current sets use an equal-weight **parameter-level mixture** of the Auger
FD-2026 SIBYLL-2.3e and EPOS-LHC-R interpretations: parameters are the mean of
the two fits, and the covariance carries an additional rank-one between-model
term. The published band therefore spans both interpretations where they diverge
and collapses to the ordinary fit covariance where they agree (below
~2x10^8 GeV, where the two coincide). Single-interpretation fits exist inside the
analysis but are intermediate products and are not distributed.

The GSF2026 sets are fitted in isotope format (deuterium and the ³He/⁴He
split are carried explicitly). The choice of solar modulation potential
affects only the local interstellar spectrum below ~10 GV: relative to
GSF2026, the Usoskin potential yields a 10--14% lower interstellar
spectrum below 2 GV, with the two converging above ~10 GV. All
higher-energy results are identical.

Each set is re-modulated with the potential it was demodulated with:
GSF2026 ships its own Ghelfi--Maurin--Derome monthly `phi(t)` table
(`GSF2026/solar_modulation.dat`), while GSF2026-USO and the historical sets
use the bundled Usoskin table at the package data root. Because the two
LIS are paired with their respective potentials, the resulting fluxes at
Earth agree far better than the LIS do (the residual below a few GV is
the solar-modulation systematic captured by the fit's `phi` nuisance).

## Particle Groups

All models support these cosmic ray groups:

| Group | Description | Atomic Numbers |
|-------|-------------|----------------|
| `"p"` | Proton group | Z = 1 |
| `"He"` | Helium group | Z = 2 |
| `"O"` | Oxygen group | Z = 3--9 |
| `"Fe"` | Iron group | Z = 10--28 |

## Sub-leading Elements and High-Energy Extrapolation

Within each mass group only the *leading* element carries its own B-spline; the
remaining *sub-leading* species are tied to their group leader through a
rigidity-dependent flux ratio. Inside the range covered by direct data this
ratio follows the fitted splines. **Above each species' top knot**
$R_{\text{max}}$ (the highest rigidity at which that element has data), the
ratio is extrapolated as a power law in rigidity that saturates to a constant:

$$
\frac{J_j(R)}{J_L(R)} = w_j \left(\frac{\min(R,\,R_{\text{sat}})}{R_{\text{max},j}}\right)^{s_j},
\qquad R_{\text{sat}} = 5\ \text{PV},
$$

where $J_L$ is the group-leader flux. The normalization $w_j$ and the slope
$s_j$ are **anchored to the data** — obtained from an error-weighted power-law
fit to the measured member-to-leader ratio over the last decade of the
element's direct data. The saturation rigidity $R_{\text{sat}} = 5$ PV sits
near the proton knee, motivated by the common galactic origin and the assumed
similarity of transport effects above it.

For the `GSF2026` and `GSF2026-USO` sets these per-element parameters ship in
`data/<version>/subleading.dat` (columns `Z A norm slope`). The legacy sets
(`2017`, `2019`, `2025`) omit the file; a missing file — or a zero slope —
reproduces the historical **constant-ratio** extrapolation bit-identically.

!!! note
    The norm and slope are best-fit point estimates; their uncertainty is not
    propagated (`covariance.dat` stores the four group leaders only). The
    saturation constant is exposed as
    `globalsplinefit.model.SUBLEADING_SAT_LNR`.

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
