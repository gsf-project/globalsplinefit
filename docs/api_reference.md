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

## Data Management

::: globalsplinefit.data_management.Parameters
    options:
      members:
        - get_solar_cycle_24_interval
        - get_solar_cycle_24_phi_average
        - as_json
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
