"""Regression coverage for the streamlined public model implementation."""

import importlib.util
import shutil

import numpy as np
import pytest

import globalsplinefit
from globalsplinefit import GSFEnergy
from globalsplinefit.data_management import Parameters


def test_pca_api_is_retired():
    assert not hasattr(globalsplinefit, "HybridPCA")
    assert importlib.util.find_spec("globalsplinefit.pca") is None


def test_packaged_parameters_are_shared_and_immutable():
    default = GSFEnergy()
    explicit = GSFEnergy(version="2026", solar_cycle_average_bins=1)

    assert default.params is explicit.params
    assert default.solar_cycle_average_bins == 6
    assert explicit.solar_cycle_average_bins == 1

    sid = default.species[0]
    assert not default.pars[sid].flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        default.pars[sid][0] = 0.0
    with pytest.raises(TypeError):
        default.pars[sid] = np.zeros(1)
    with pytest.raises(AttributeError, match="immutable"):
        default.params.version = "changed"


def test_custom_data_path_and_version_are_mutually_exclusive():
    data_path = GSFEnergy().params.data_path
    with pytest.raises(ValueError, match="either version or data_path"):
        Parameters(data_path=data_path, version="2026")


def test_provenance_is_validated_and_returned_by_copy(tmp_path):
    model = GSFEnergy()
    provenance = model.params.provenance
    provenance["version"] = "changed"
    assert model.params.provenance["version"] == "2026"

    bundle = tmp_path / "bad-provenance"
    shutil.copytree(GSFEnergy(version="2017").params.data_path, bundle)
    (bundle / "fit_result.json").write_text("[]")
    with pytest.raises(ValueError, match="invalid provenance"):
        Parameters(data_path=bundle)


def test_public_numeric_inputs_have_clear_domains():
    model = GSFEnergy()
    with pytest.raises(ValueError, match="one-dimensional"):
        model.flux(np.ones((2, 2)), "p")
    with pytest.raises(ValueError, match="non-negative"):
        model.flux([-1.0], "p")

    for value in (-1.0, -2.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="energy_scale"):
            model.energy_scale = value

    with pytest.raises(ValueError, match="default_rigidity_cutoff"):
        GSFEnergy(default_rigidity_cutoff=-1.0)
    with pytest.raises(ValueError, match="cutoff_width"):
        GSFEnergy(cutoff_width=np.nan)
    with pytest.raises(ValueError, match="valid YYYYMM"):
        model.flux([10.0], "p", time_interval=(200913, 201001))
    with pytest.raises(ValueError, match="missing"):
        model.flux([10.0], "p", time_interval=(180001, 180002))


def test_jacobian_cache_is_byte_bounded():
    model = GSFEnergy()
    model._cache_max_bytes = 100_000
    sid = model._leader_by_charge[1]

    for offset in range(20):
        rigidity = np.geomspace(1.0 + offset, 100.0 + offset, 64)
        model._rigidity_flux_jacobian(sid, rigidity)

    assert 0 < model._jacobian_cache_bytes <= model._cache_max_bytes
    assert model._jacobian_cache
    assert all(not value.flags.writeable for value in model._jacobian_cache.values())
