"""Global Spline Fit (GSF) model for cosmic ray flux and composition.

Spline-fit parametrizations of cosmic-ray flux and composition, with one
model class per input variable: energy, kinetic energy, energy per
nucleon, kinetic energy per nucleon, and rigidity.

Solar Cycle 24 (December 2008 to December 2019) is used as the default
reference period for solar modulation calculations. The "LIS" time_interval
can be used to obtain the Local Interstellar Spectrum without modulation.

Examples
--------
Calculate cosmic ray flux using total energy per nucleus:

>>> from globalsplinefit import GSFEnergy
>>> import numpy as np
>>> energy_model = GSFEnergy()
>>> energy = np.logspace(0, 3, 100)  # 1 GeV to 1 TeV total energy
>>> proton_flux = energy_model.flux(energy, "p")  # proton flux (Solar Cycle 24 avg)
>>> total_flux = energy_model.total_flux(energy)  # all groups combined
>>> lis_flux = energy_model.flux(energy, "p", time_interval="LIS")  # LIS flux

Calculate cosmic ray flux using kinetic energy per nucleus:

>>> from globalsplinefit import GSFKineticEnergy
>>> kinetic_model = GSFKineticEnergy()
>>> kinetic_energy = np.logspace(0, 3, 100)  # 1 GeV to 1 TeV kinetic energy
>>> proton_flux = kinetic_model.flux(kinetic_energy, "p")  # proton flux
>>> total_flux = kinetic_model.total_flux(kinetic_energy)  # all groups combined

Calculate flux at Earth during specific time period:

>>> flux_earth = energy_model.flux(energy, "p", time_interval=(200901, 201001))

Calculate nucleon flux using energy per nucleon. ``flux()`` returns the
sum of proton and neutron contributions; use ``p_and_n_flux()`` to get
them separately:

>>> from globalsplinefit import GSFEnergyPerNucleon
>>> nucleon_model = GSFEnergyPerNucleon()
>>> energy_per_nucleon = np.logspace(0, 2, 50)
>>> total_nucleons = nucleon_model.flux(energy_per_nucleon, "He")  # shape (N,)
>>> p_and_n = nucleon_model.p_and_n_flux(energy_per_nucleon, "He")  # shape (2, N)
>>> proton_nucleons, neutron_nucleons = p_and_n[0], p_and_n[1]

For kinetic energy per nucleon calculations:

>>> from globalsplinefit import GSFKineticEnergyPerNucleon
>>> kinetic_nucleon_model = GSFKineticEnergyPerNucleon()

Calculate flux using rigidity:

>>> from globalsplinefit import GSFRigidity
>>> rigidity_model = GSFRigidity()
>>> rigidity = np.logspace(0, 3, 100)  # GV
>>> proton_flux = rigidity_model.flux(rigidity, "p")
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

from .data_management import (
    DEFAULT_VERSION,
    MODEL_VERSIONS,
    SOLAR_CYCLE_24_DURATION_YEARS,
    SOLAR_CYCLE_24_END,
    SOLAR_CYCLE_24_START,
    get_available_versions,
    list_versions,
    resolve_version,
    version_info,
)
from .model import (
    GSFEnergy,
    GSFEnergyPerNucleon,
    GSFKineticEnergy,
    GSFKineticEnergyPerNucleon,
    GSFRigidity,
)
from .reduced import ReducedGSF, optimize_pivots

try:
    #: Installed code version. The model parameter-set version is reported
    #: by ``Parameters.version``.
    __version__ = _dist_version("globalsplinefit")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "unknown"

__all__ = [
    "__version__",
    "GSFEnergy",
    "GSFKineticEnergy",
    "GSFEnergyPerNucleon",
    "GSFKineticEnergyPerNucleon",
    "GSFRigidity",
    "ReducedGSF",
    "optimize_pivots",
    "DEFAULT_VERSION",
    "MODEL_VERSIONS",
    "SOLAR_CYCLE_24_DURATION_YEARS",
    "SOLAR_CYCLE_24_END",
    "SOLAR_CYCLE_24_START",
    "get_available_versions",
    "list_versions",
    "resolve_version",
    "version_info",
]
