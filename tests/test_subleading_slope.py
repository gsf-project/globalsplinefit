"""Tests for the sub-leading norm+slope extrapolation (subleading.dat).

Above a sub-leading species' top knot Rmax the flux is
norm * (R/Rmax)**slope * leader_flux(R). Bundles without subleading.dat
(2017/2019/2025) must behave exactly as before: ratio recomputed from the
splines at Rmax, slope 0.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from globalsplinefit.model import GSFRigidity

DATA_2025 = Path(__file__).parent.parent / "src/globalsplinefit/data/2025"

LI = 3  # sub-leading test species (leader O, Z=8)


@pytest.fixture
def base_model():
    return GSFRigidity(data_path=DATA_2025, default_time_interval="LIS")


def _bundle_with_subleading(tmp_path, rows):
    """Copy the 2025 bundle and add a subleading.dat with the given
    (z, a, norm, slope) rows."""
    dst = tmp_path / "bundle"
    shutil.copytree(DATA_2025, dst)
    with open(dst / "subleading.dat", "w") as f:
        f.write("#Z\tA\tnorm\tslope\n")
        for z, a, norm, slope in rows:
            f.write(f"{z:d}\t{a:.3f}\t{norm:.8e}\t{slope:.6f}\n")
    return dst


def _li_sid(model):
    return model.params.z_to_sids[LI][0]


def test_no_file_gives_zero_slopes(base_model):
    """Pre-slope bundles: every species has slope 0 and the historic ratio."""
    assert all(s == 0.0 for s in base_model.flux_slope.values())


def test_stored_norm_and_slope_are_applied(base_model, tmp_path):
    """Above Rmax the flux must be norm * (R/Rmax)**slope * leader_flux."""
    sid = _li_sid(base_model)
    leader_sid, ratio = base_model.flux_ratio[sid]
    slope = -0.30
    bundle = _bundle_with_subleading(tmp_path, [(sid[0], sid[1], ratio, slope)])
    m = GSFRigidity(data_path=bundle, default_time_interval="LIS")
    assert m.flux_slope[sid] == slope

    xmax = base_model.kx[sid][-1]
    R = np.exp(xmax + np.array([0.3, 1.0, 2.0]))  # GV, above Rmax
    base = base_model.flux(R, LI)  # = ratio * leader
    tilt = np.exp(slope * (np.log(R) - xmax))
    got = m.flux(R, LI)
    # rtol bounded by the %.8e round-trip of the stored norm
    np.testing.assert_allclose(got, base * tilt, rtol=1e-8)

    # below Rmax the species' own spline is untouched
    R_lo = np.exp(xmax - np.array([1.0, 0.3]))
    np.testing.assert_allclose(m.flux(R_lo, LI), base_model.flux(R_lo, LI), rtol=1e-12)


def test_continuity_at_xmax_with_stored_values(base_model, tmp_path):
    """With the stored norm equal to the spline ratio at Rmax, the flux stays
    continuous across the matching point for any slope."""
    sid = _li_sid(base_model)
    _, ratio = base_model.flux_ratio[sid]
    bundle = _bundle_with_subleading(tmp_path, [(sid[0], sid[1], ratio, -0.42)])
    m = GSFRigidity(data_path=bundle, default_time_interval="LIS")
    xmax = m.kx[sid][-1]
    eps = 1e-9
    lo = m.flux(np.exp(xmax - eps), LI)
    hi = m.flux(np.exp(xmax + eps), LI)
    np.testing.assert_allclose(lo, hi, rtol=1e-6)


def test_zero_slope_file_matches_no_file(base_model, tmp_path):
    """A subleading.dat carrying the recomputed ratio and slope 0 must be
    bit-identical to the historical no-file behavior."""
    rows = []
    for sid in base_model.species:
        leader_sid, ratio = base_model.flux_ratio[sid]
        if sid != leader_sid:
            rows.append((sid[0], sid[1], ratio, 0.0))
    bundle = _bundle_with_subleading(tmp_path, rows)
    m = GSFRigidity(data_path=bundle, default_time_interval="LIS")
    R = np.geomspace(1.0, 1e7, 60)
    for z in (3, 5, 7, 12, 24):  # sub-leadings across both groups
        np.testing.assert_allclose(
            m.flux(R, z), base_model.flux(R, z), rtol=1e-7, err_msg=f"Z={z}"
        )


def test_tilt_saturates_at_5pv(base_model, tmp_path):
    """Above R_sat = 5e6 GV the extrapolated ratio is constant (the tilt
    exponent saturates at ln(R_sat/Rmax))."""
    from globalsplinefit.model import SUBLEADING_SAT_LNR

    sid = _li_sid(base_model)
    _, ratio = base_model.flux_ratio[sid]
    slope = -0.30
    bundle = _bundle_with_subleading(tmp_path, [(sid[0], sid[1], ratio, slope)])
    m = GSFRigidity(data_path=bundle, default_time_interval="LIS")
    xmax = m.kx[sid][-1]
    R = np.array([5.0e6, 5.0e7])  # at and above saturation, GV
    r = m.flux(R, LI) / base_model.flux(R, LI)  # tilt factor vs constant-ratio
    expect = np.exp(slope * (SUBLEADING_SAT_LNR - xmax))
    np.testing.assert_allclose(r, expect, rtol=1e-7)


def test_jacobian_carries_the_tilt(base_model, tmp_path):
    """The rigidity-space Jacobian of a sub-leading species must gain the same
    clamped tilt factor above Rmax and stay unchanged below."""
    sid = _li_sid(base_model)
    _, ratio = base_model.flux_ratio[sid]
    slope = -0.30
    bundle = _bundle_with_subleading(tmp_path, [(sid[0], sid[1], ratio, slope)])
    m = GSFRigidity(data_path=bundle, default_time_interval="LIS")

    xmax = base_model.kx[sid][-1]
    R = np.exp(xmax + np.array([-0.5, 0.5, 1.5]))  # below / above Rmax
    j_base = base_model.jacobian(R, LI)
    j_new = m.jacobian(R, LI)
    tilt = np.exp(slope * np.clip(np.log(R) - xmax, 0.0, None))
    np.testing.assert_allclose(j_new, j_base * tilt[:, None], rtol=1e-8)
