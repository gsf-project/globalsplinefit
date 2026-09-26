"""Tests for GSFBase.sample (pseudo-experiments from the native covariance)."""

import numpy as np
import pytest

from globalsplinefit import (
    GSFEnergy,
    GSFEnergyPerNucleon,
    GSFKineticEnergy,
    GSFKineticEnergyPerNucleon,
    GSFRigidity,
)

N_DRAWS = 3000
SEED = 7


@pytest.fixture(scope="module")
def gsf():
    return GSFEnergy()


@pytest.fixture(scope="module")
def energy():
    return np.logspace(2, 10, 40)


def test_shape_and_finite(gsf, energy):
    draws = gsf.sample(energy, "He", 11, rng=np.random.default_rng(SEED))
    assert draws.shape == (11, len(energy))
    assert np.all(np.isfinite(draws))


def test_mean_matches_central(gsf, energy):
    draws = gsf.sample(energy, "He", N_DRAWS, rng=np.random.default_rng(SEED))
    central = gsf.flux(energy, "He")
    sigma = gsf.error(energy, "He")
    # mean of N draws is central within ~5 standard errors
    pull = (draws.mean(axis=0) - central) / (sigma / np.sqrt(N_DRAWS))
    assert np.all(np.abs(pull) < 5.0)


@pytest.mark.parametrize("target", ["H", "He", "O*", "Fe*"])
def test_empirical_sigma_matches_error(gsf, energy, target):
    draws = gsf.sample(energy, target, N_DRAWS, rng=np.random.default_rng(SEED))
    ratio = draws.std(axis=0) / gsf.error(energy, target)
    # sigma estimate from N draws: relative error ~ 1/sqrt(2N) ~ 1.3%
    assert np.all(np.abs(ratio - 1.0) < 0.10)


def test_total_empirical_sigma_matches_total_error(gsf, energy):
    draws = gsf.sample(energy, None, N_DRAWS, rng=np.random.default_rng(SEED))
    ratio = draws.std(axis=0) / gsf.total_error(energy)
    assert np.all(np.abs(ratio - 1.0) < 0.10)


def test_same_seed_groups_sum_to_total(gsf, energy):
    """One shared amplitude draw: same-seeded group samples sum to the
    same-seeded all-particle sample (exact, by linearity)."""
    total = gsf.sample(energy, None, 7, rng=np.random.default_rng(SEED))
    group_sum = sum(
        gsf.sample(energy, g, 7, rng=np.random.default_rng(SEED))
        for g in gsf.active_groups
    )
    np.testing.assert_allclose(group_sum, total, rtol=1e-10, atol=0.0)


def test_element_target(gsf, energy):
    draws = gsf.sample(energy, 6, 9, rng=np.random.default_rng(SEED))
    assert draws.shape == (9, len(energy))


@pytest.mark.parametrize(
    "cls,x",
    [
        (GSFEnergy, np.logspace(2, 10, 10)),
        (GSFKineticEnergy, np.logspace(2, 10, 10)),
        (GSFRigidity, np.logspace(1, 9, 10)),
        (GSFEnergyPerNucleon, np.logspace(1, 8, 10)),
        (GSFKineticEnergyPerNucleon, np.logspace(1, 8, 10)),
    ],
)
def test_all_model_classes(cls, x):
    model = cls()
    draws = model.sample(x, "He", 5, rng=np.random.default_rng(SEED))
    assert draws.shape == (5, len(x))
    assert np.all(np.isfinite(draws))


def test_historical_version(energy):
    model = GSFEnergy(version="2019")
    draws = model.sample(energy, None, N_DRAWS, rng=np.random.default_rng(SEED))
    ratio = draws.std(axis=0) / model.total_error(energy)
    assert np.all(np.abs(ratio - 1.0) < 0.10)


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "10"])
def test_invalid_n_samples(gsf, energy, bad):
    with pytest.raises(ValueError, match="n_samples"):
        gsf.sample(energy, "He", bad)


def test_reproducible_with_seed(gsf, energy):
    a = gsf.sample(energy, "He", 4, rng=np.random.default_rng(SEED))
    b = gsf.sample(energy, "He", 4, rng=np.random.default_rng(SEED))
    np.testing.assert_array_equal(a, b)
