"""Test configuration and fixtures for GSF package."""

from pathlib import Path

import numpy as np
import pytest

from globalsplinefit.model import GSFEnergy, GSFEnergyPerNucleon, GSFRigidity


# The shared model fixtures are pinned to the 2017 (non-isotope) set: the bulk
# of the suite compares against the gsf_*_2017 reference data and uses bare
# integer charge access (z_to_a[1], etc.), which is only defined for
# single-species (non-isotope) charges. The package default is GSF2026 (an
# isotope set); it is covered separately in test_versioned_sets.py.
@pytest.fixture
def gsf_energy():
    """Create a GSF energy model instance for testing (2017 reference set)."""
    return GSFEnergy(version="2017")


@pytest.fixture
def gsf_rigidity():
    """Create a GSF rigidity model instance for testing (2017 reference set)."""
    return GSFRigidity(version="2017")


@pytest.fixture
def gsf_nucleon():
    """Create a GSF nucleon model instance for testing (2017 reference set)."""
    return GSFEnergyPerNucleon(version="2017")


@pytest.fixture
def sample_energies():
    """Sample energy values for testing (GeV) - chosen to be above flux thresholds."""
    return np.array([10.0, 100.0, 1000.0, 10000.0])


@pytest.fixture
def sample_rigidities():
    """Sample rigidity values in GV."""
    return np.array([1.0, 10.0, 100.0, 1000.0])


@pytest.fixture
def sample_energies_per_nucleon():
    """Sample energies per nucleon in GeV."""
    return np.array([1.0, 10.0, 100.0, 1000.0])


@pytest.fixture
def sample_time_interval():
    """Sample time interval for testing."""
    return (200901, 200912)


@pytest.fixture
def extended_energy_range():
    """Extended energy range for comprehensive testing."""
    return 10 ** np.linspace(0, 11, 100)


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def reference_particle_flux_2017(test_data_dir):
    """Load reference particle flux data from 2017."""
    data = np.loadtxt(test_data_dir / "gsf_particle_flux_2017.dat")
    return {
        "energy": data[:, 0],
        "proton": data[:, 1],
        "helium": data[:, 2],
        "oxygen": data[:, 3],
        "iron": data[:, 4],
        "total": data[:, 5],
    }


@pytest.fixture
def reference_nucleon_flux_2017(test_data_dir):
    """Load reference nucleon flux data from 2017."""
    data = np.loadtxt(test_data_dir / "gsf_nucleon_flux_2017.dat")
    return {
        "energy_per_nucleon": data[:, 0],
        "proton_group": data[:, 1],
        "helium_group": data[:, 2],
        "oxygen_group": data[:, 3],
        "iron_group": data[:, 4],
        "total": data[:, 5],
    }


@pytest.fixture
def reference_particle_flux_error_2017(test_data_dir):
    """Load reference particle flux error data from 2017."""
    data = np.loadtxt(test_data_dir / "gsf_particle_flux_error_2017.dat")
    return {
        "energy": data[:, 0],
        "proton": data[:, 1],
        "helium": data[:, 2],
        "oxygen": data[:, 3],
        "iron": data[:, 4],
        "total": data[:, 5],
    }


@pytest.fixture
def reference_nucleon_flux_error_2017(test_data_dir):
    """Load reference nucleon flux error data from 2017."""
    data = np.loadtxt(test_data_dir / "gsf_nucleon_flux_error_2017.dat")
    return {
        "energy_per_nucleon": data[:, 0],
        "proton_group": data[:, 1],
        "helium_group": data[:, 2],
        "oxygen_group": data[:, 3],
        "iron_group": data[:, 4],
        "total": data[:, 5],
    }


@pytest.fixture
def reference_solar_modulation_2017(test_data_dir):
    """Load reference solar modulation data from 2017."""
    data = np.loadtxt(test_data_dir / "gsf_solar_modulation_2017.dat")
    return {
        "energy": data[:, 0],
        "LIS_flux": data[:, 1],
        "LIS_flux_error": data[:, 2],
        "Oct_2009_min": data[:, 3],
        "Oct_2009_min_error": data[:, 4],
        "June_1991_max": data[:, 5],
        "June_1991_max_error": data[:, 6],
    }
