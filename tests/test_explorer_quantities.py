"""Physics and validation checks for the browser Explorer evaluation core."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from globalsplinefit import DEFAULT_VERSION

sys.path.insert(0, str(Path(__file__).parents[1] / "webapp"))
import bridge  # noqa: E402
import gsf_explorer as gx  # noqa: E402


@pytest.fixture(scope="module")
def grid():
    return {
        "dmin": 1.0,
        "dmax": 3.0,
        "npts": 12,
        "groups": gx.GROUPS,
        "elements": [],
        "time_interval": "LIS",
        "rigidity_cutoff": None,
        "energy_scale": 1.0,
    }


def test_nucleon_flux_accepts_only_per_nucleon_axes(grid):
    for basis in gx.NUCLEON_BASES:
        result = gx.evaluate(
            gx.make_model(basis, DEFAULT_VERSION, 1), basis, quantity="nucleon", **grid
        )
        assert result["quantity"] == "nucleon"
        assert result["total"] is not None

    with pytest.raises(ValueError, match="energy-per-nucleon"):
        gx.evaluate(
            gx.make_model("etot", DEFAULT_VERSION, 1),
            "etot",
            quantity="nucleon",
            **grid,
        )


def test_nucleus_and_nucleon_flux_are_distinct_on_per_nucleon_axis(grid):
    model = gx.make_model("en", DEFAULT_VERSION, 1)
    nucleus = gx.evaluate(model, "en", quantity="nucleus", **grid)
    nucleon = gx.evaluate(model, "en", quantity="nucleon", **grid)

    nucleus_he = nucleus["series"]["He*"][0]
    nucleon_he = nucleon["series"]["He*"][0]
    assert np.all(nucleon_he > nucleus_he)


@pytest.mark.parametrize("quantity", ["mean_lna", "var_lna"])
def test_composition_moments_are_dimensionless_with_propagated_errors(quantity, grid):
    result = gx.evaluate(
        gx.make_model("etot", DEFAULT_VERSION, 1), "etot", quantity=quantity, **grid
    )
    value, error = result["series"][quantity]
    assert result["total"] is None
    assert np.all(np.isfinite(value))
    assert np.all(np.isfinite(error))
    assert np.all(error >= 0)


def test_energy_scale_is_a_factor_not_a_fractional_shift(grid):
    model = gx.make_model("etot", DEFAULT_VERSION, 1)
    gx.evaluate(model, "etot", quantity="nucleus", **grid)
    assert model.energy_scale == 0.0


def test_composition_bridge_emits_strict_json_gaps():
    payload = {
        "version": DEFAULT_VERSION,
        "quantity": "mean_lna",
        "basis": "etot",
        "dmin": -1,
        "dmax": 1,
        "npts": 8,
        "groups": gx.GROUPS,
        "elements": [],
        "mod": "LIS",
        "cutoff": 0,
        "escale": 1.0,
        "phiBins": 1,
    }
    encoded = bridge.evaluate(json.dumps(payload))
    assert "NaN" not in encoded
    assert None in json.loads(encoded)["series"][0]["flux"]
