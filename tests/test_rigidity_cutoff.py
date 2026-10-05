"""Geomagnetic rigidity cutoff: per-species thresholds and the 0 GV case."""

import numpy as np
import pytest

from globalsplinefit import (
    GSFEnergy,
    GSFEnergyPerNucleon,
    GSFRigidity,
    ReducedGSF,
)
from globalsplinefit.model import NUCLEON_MASS_GEV

RC = 8.0


def _rigidity(model, sid, total_energy):
    mass = model.z_to_a[sid] * NUCLEON_MASS_GEV
    return np.sqrt(np.maximum(total_energy**2 - mass**2, 0.0)) / sid[0]


@pytest.fixture(scope="module")
def energy_models():
    return (
        GSFEnergy(),
        GSFEnergy(default_rigidity_cutoff=RC, cutoff_width=0.0),
    )


def test_sharp_cutoff_per_species(energy_models):
    """Every species is cut at its own rigidity, with its own (Z, A)."""
    ref, cut = energy_models
    energy = np.logspace(0, 4, 400)
    for sid in ref.species:
        kept = _rigidity(ref, sid, energy) >= RC
        np.testing.assert_array_equal(
            cut.flux(energy, sid), np.where(kept, ref.flux(energy, sid), 0.0)
        )


def test_group_flux_and_jacobian_sum_cut_species(energy_models):
    """Group fluxes and Jacobians are sums of the individually cut species."""
    _ref, cut = energy_models
    energy = np.logspace(0.5, 3, 60)
    for group in cut.active_groups:
        sids = cut._target_sids(cut._resolve_z(group)[0])
        np.testing.assert_allclose(
            cut.flux(energy, group), sum(cut.flux(energy, s) for s in sids)
        )
        np.testing.assert_allclose(
            cut.jacobian(energy, group), sum(cut.jacobian(energy, s) for s in sids)
        )


def test_nucleon_flux_cut_per_species():
    ref = GSFEnergyPerNucleon()
    cut = GSFEnergyPerNucleon(default_rigidity_cutoff=RC, cutoff_width=0.0)
    e_n = np.logspace(0, 3, 300)
    expected = np.zeros((2, e_n.size))
    for sid in ref.species:
        e_tot = e_n * ref.z_to_a[sid]
        kept = _rigidity(ref, sid, e_tot) >= RC
        expected += np.where(kept, ref.p_and_n_flux(e_n, sid), 0.0)
    np.testing.assert_allclose(cut.p_and_n_total_flux(e_n), expected, rtol=1e-12)
    # the cut p and n Jacobians carry the same per-species mask
    jp, jn = cut.p_and_n_jacobian(e_n, "He")
    fp, fn = cut.p_and_n_flux(e_n, "He")
    assert np.all(jp[fp == 0] == 0) and np.all(jn[fn == 0] == 0)


def test_reduced_with_cutoff_is_cut_nucleon_flux():
    model = GSFEnergyPerNucleon(default_rigidity_cutoff=3.0)
    red = ReducedGSF(model)
    e_n = np.logspace(0, 6, 40)
    np.testing.assert_allclose(red.flux(e_n), model.p_and_n_total_flux(e_n))
    uncut = ReducedGSF(GSFEnergyPerNucleon())
    assert np.all(red.flux(e_n)[0, :3] < uncut.flux(e_n)[0, :3])


@pytest.mark.parametrize("width", [0.0, 1.0])
def test_zero_cutoff_is_no_cutoff(width):
    e = np.logspace(0, 2, 50)
    for cls, target in ((GSFEnergy, "p"), (GSFRigidity, "p")):
        ref = cls()
        zero = cls(default_rigidity_cutoff=0.0, cutoff_width=width)
        np.testing.assert_array_equal(zero.flux(e, target), ref.flux(e, target))
        np.testing.assert_array_equal(
            ref.flux(e, target, rigidity_cutoff=0.0), ref.flux(e, target)
        )
        np.testing.assert_array_equal(zero.jacobian(e, target), ref.jacobian(e, target))
    ref = GSFEnergyPerNucleon()
    zero = GSFEnergyPerNucleon(default_rigidity_cutoff=0.0, cutoff_width=width)
    np.testing.assert_array_equal(zero.p_and_n_total_flux(e), ref.p_and_n_total_flux(e))


def test_explicit_zero_overrides_default():
    cut = GSFEnergy(default_rigidity_cutoff=RC)
    e = np.logspace(0, 2, 30)
    np.testing.assert_array_equal(
        cut.flux(e, "p", rigidity_cutoff=0.0), GSFEnergy().flux(e, "p")
    )
    assert np.all(cut.flux(e, "p")[:5] < GSFEnergy().flux(e, "p")[:5])
