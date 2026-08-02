"""The public deuterium target: flux()/error() accept "D" and species-id tuples.

2026 carries deuterium as its own species (1, 2.014) at Z=1. These tests
pin the target-resolution contract: the string "D" (and an explicit species
id) selects that single isotope on every model class, uncertainties propagate
through the leader covariance, mean/var lnA iterate all species on every
class, and one-species-per-charge model versions reject the target cleanly.
"""

import numpy as np
import pytest

from globalsplinefit.model import (
    NUCLEON_MASS_GEV,
    GSFEnergy,
    GSFEnergyPerNucleon,
    GSFKineticEnergy,
    GSFKineticEnergyPerNucleon,
    GSFRigidity,
)

ALL_MODEL_CLASSES = [
    GSFEnergy,
    GSFKineticEnergy,
    GSFRigidity,
    GSFEnergyPerNucleon,
    GSFKineticEnergyPerNucleon,
]

D_SID = (1, 2.014)
P_SID = (1, 1.008)

X = np.array([10.0, 100.0, 1000.0])  # GeV / GV / GeV-per-nucleon


@pytest.mark.parametrize("cls", ALL_MODEL_CLASSES)
def test_d_flux_positive_and_matches_species_id(cls):
    """flux("D") works on every model class and equals the explicit sid."""
    model = cls()
    flux_name = model.flux(X, "D")
    flux_sid = model.flux(X, D_SID)
    assert np.all(flux_name > 0)
    assert np.all(np.isfinite(flux_name))
    np.testing.assert_allclose(flux_name, flux_sid)


@pytest.mark.parametrize("cls", ALL_MODEL_CLASSES)
def test_species_fluxes_sum_to_charge(cls):
    """p + D species fluxes reproduce the Z=1 charge flux."""
    model = cls()
    np.testing.assert_allclose(
        model.flux(X, 1),
        model.flux(X, P_SID) + model.flux(X, D_SID),
        rtol=1e-12,
    )


def test_kinetic_energy_uses_deuteron_mass():
    """A kinetic-energy D target converts with A_D, not the proton mass."""
    kin = GSFKineticEnergy()
    tot = GSFEnergy()
    np.testing.assert_allclose(
        kin.flux(X, "D"),
        tot.flux(X + D_SID[1] * NUCLEON_MASS_GEV, "D"),
    )


@pytest.mark.parametrize("cls", ALL_MODEL_CLASSES)
def test_d_error_propagates_from_leader_covariance(cls):
    """error("D") is finite, positive, and scales with the p ratio."""
    model = cls()
    err = model.error(X, "D")
    assert np.all(err > 0)
    assert np.all(np.isfinite(err))
    # D rides on the proton spline: its relative error cannot exceed a few
    # times the proton group's (identical up to the extrapolation tilt and,
    # on kinetic models, the rest-mass shift).
    rel_d = err / model.flux(X, "D")
    rel_p = model.error(X, "p") / model.flux(X, "p")
    assert np.all(rel_d < 5 * rel_p)


@pytest.mark.parametrize("cls", ALL_MODEL_CLASSES)
def test_mean_lnA_iterates_all_species(cls):
    """mean_lnA/var_lnA work on every class for the isotope-format default set."""
    model = cls()
    mean = model.mean_lnA(X)
    var = model.var_lnA(X)
    assert np.all(np.isfinite(mean))
    assert np.all(mean > 0)
    assert np.all(var >= 0)


def test_non_isotope_version_rejects_d():
    """Model versions without a separate D species reject the target."""
    model = GSFEnergy(version="2025")
    with pytest.raises(ValueError, match="not carried by model version"):
        model.flux(X, "D")


def test_charge_tuple_semantics_unchanged():
    """A tuple of charges is still read as charges, not as a species id."""
    model = GSFEnergy()
    # (1, 2) = H + He -> different groups -> rejected, exactly as before
    with pytest.raises(ValueError, match="Multiple groups"):
        model.flux(X, (1, 2))
    # same-group charge tuple still works
    np.testing.assert_allclose(model.flux(X, (6, 7, 8)), model.flux(X, [6, 7, 8]))


def test_unknown_species_name_lists_valid_names():
    model = GSFEnergy()
    with pytest.raises(ValueError, match="unknown target"):
        model.flux(X, "T")


def test_named_proton_excludes_deuterium():
    model = GSFEnergy()
    np.testing.assert_allclose(model.flux(X, "p"), model.flux(X, P_SID))
    assert np.all(model.flux(X, "H") > model.flux(X, "p"))
