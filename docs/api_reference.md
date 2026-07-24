# API Reference

Complete API documentation for the GlobalSplineFit package.

## Model Classes

The main interface to GlobalSplineFit functionality.

::: globalsplinefit.GSFEnergy
    options:
      members:
        - flux
        - error
        - covariance
        - jacobian
        - total_flux
        - total_error
      show_bases: true

::: globalsplinefit.GSFKineticEnergy
    options:
      members:
        - flux
        - error
        - covariance
        - jacobian
        - total_flux
        - total_error
      show_bases: true

::: globalsplinefit.GSFRigidity
    options:
      members:
        - flux
        - error
        - covariance
        - jacobian
        - total_flux
        - total_error
      show_bases: true

::: globalsplinefit.GSFEnergyPerNucleon
    options:
      members:
        - flux
        - error
        - covariance
        - jacobian
        - total_flux
        - total_error
        - p_and_n_flux
        - p_and_n_jacobian
        - p_and_n_covariance
        - p_and_n_error
        - p_and_n_total_flux
      show_bases: true

::: globalsplinefit.GSFKineticEnergyPerNucleon
    options:
      members:
        - flux
        - error
      show_bases: true

## Data Management

::: globalsplinefit.data_management.Parameters
    options:
      members:
        - get_solar_cycle_24_interval
        - get_solar_cycle_24_phi_average
      show_bases: true

## PCA

::: globalsplinefit.pca.HybridPCA
    options:
      show_bases: true

## Constants

### GROUP_NAMES

Dictionary mapping group names to atomic numbers.

Maps string group names (`"p"`, `"He"`, `"O*"`, `"Fe*"`, etc.) to their
corresponding group leader atomic numbers.

```python
from globalsplinefit.model import GSFBase
print(GSFBase.GROUP_NAMES)
```

### SUBLEADING_SAT_LNR

Saturation rigidity for the sub-leading high-energy extrapolation, stored as
$\ln(R/\text{GV})$ — i.e. `log(5e6)`, with $R_{\text{sat}} = 5$ PV.

Above a sub-leading species' top knot the member-to-leader ratio is tilted by
its fitted power-law slope and held constant beyond $R_{\text{sat}}$:
`ratio(R) = norm * (min(R, R_sat)/Rmax)**slope`. See
[Sub-leading Elements and High-Energy Extrapolation](user_guide.md#sub-leading-elements-and-high-energy-extrapolation).

```python
import numpy as np
from globalsplinefit.model import SUBLEADING_SAT_LNR
print(np.exp(SUBLEADING_SAT_LNR))  # 5e6 GV = 5 PV
```
