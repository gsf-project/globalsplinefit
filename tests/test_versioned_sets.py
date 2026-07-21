"""Coverage for the shipped parameter sets and the default version.

GSF2026 is the default set (Ghelfi-Maurin-Derome modulation potential);
GSF2026-USO is the same fit with the Usoskin 2017 potential, which yields a
lower low-rigidity local interstellar spectrum. Both are isotope-format sets
(they carry deuterium and the He-isotope split), so bare integer-charge access
is not defined for the multi-species charges Z=1 (p+D) and Z=2 (3He+4He); the
name/tuple flux API is used throughout here.
"""
import numpy as np
import pytest

from globalsplinefit import GSFEnergy, GSFRigidity
from globalsplinefit.data_management import get_available_versions


def test_new_sets_available():
    versions = get_available_versions()
    assert "GSF2026" in versions
    assert "GSF2026-USO" in versions


def test_default_is_gsf2026():
    """A bare GSFEnergy() must resolve to the GSF2026 (GMD) set."""
    default = GSFEnergy()
    explicit = GSFEnergy(version="GSF2026")
    E = np.logspace(0, 4, 50)
    for g in ("p", "He", "O*", "Fe*"):
        np.testing.assert_allclose(default.flux(E, g), explicit.flux(E, g), rtol=1e-12)


@pytest.mark.parametrize("version", ["GSF2026", "GSF2026-USO"])
def test_set_loads_and_is_positive(version):
    m = GSFEnergy(version=version)
    E = np.logspace(0, 6, 80)
    for g in ("p", "He", "O*", "Fe*"):
        f = m.flux(E, g)
        assert np.all(np.isfinite(f))          # never NaN/inf, even below threshold
    # above the heaviest group's threshold all four groups are populated
    E_hi = np.logspace(np.log10(30), 6, 60)
    for g in ("p", "He", "O*", "Fe*"):
        assert np.all(m.flux(E_hi, g) > 0)


def test_usoskin_lowers_low_rigidity_lis():
    """The Usoskin potential runs lower than the GMD default, so its
    demodulated (LIS) flux is lower at low rigidity and converges to the
    default at high rigidity."""
    gmd = GSFRigidity(version="GSF2026")        # default
    uso = GSFRigidity(version="GSF2026-USO")
    R = np.array([1.5, 5.0, 50.0])
    for g in ("p", "He"):
        jg = gmd.flux(R, g, time_interval="LIS")
        ju = uso.flux(R, g, time_interval="LIS")
        assert ju[0] / jg[0] - 1.0 < -0.05      # >5% lower at 1.5 GV
        assert abs(ju[-1] / jg[-1] - 1.0) < 0.02  # converged by 50 GV


def test_gsf2026_ships_its_own_phi_table():
    """GSF2026 (GMD) carries a version-local solar_modulation.dat; the fitted
    LIS must be re-modulated with the potential it was demodulated with. The
    Usoskin variant and the legacy sets fall back to the shared Usoskin table.
    """
    gmd = GSFRigidity(version="GSF2026")
    uso = GSFRigidity(version="GSF2026-USO")
    leg = GSFRigidity(version="2025")
    # GMD runs a distinctly higher potential than Usoskin (~60-70 MV) over the
    # neutron-monitor era, so its monthly table differs at the tens-of-MV level.
    assert abs(gmd.phi[2015].mean() - uso.phi[2015].mean()) * 1e3 > 40.0
    # Usoskin variant and legacy set share the bundled Usoskin table exactly.
    np.testing.assert_allclose(uso.phi[2015], leg.phi[2015], rtol=0, atol=0)


def test_toa_agrees_across_phi_sources():
    """Although the LIS differ ~10-14% at low R, re-modulating each set with
    its OWN table brings the fluxes at Earth much closer together — the residual
    is the solar-modulation systematic, well below the LIS gap and shrinking
    with rigidity."""
    gmd = GSFRigidity(version="GSF2026")
    uso = GSFRigidity(version="GSF2026-USO")
    R = np.array([1.5, 5.0, 20.0])
    epoch = (201105, 201805)  # AMS-02 era
    for g in ("p", "He"):
        lis_gap = abs(uso.flux(R, g, time_interval="LIS")[0]
                      / gmd.flux(R, g, time_interval="LIS")[0] - 1.0)
        toa = uso.flux(R, g, time_interval=epoch) / gmd.flux(R, g, time_interval=epoch)
        assert lis_gap > 0.08                       # LIS differ markedly at 1.5 GV
        assert abs(toa[0] - 1.0) < 0.5 * lis_gap    # TOA gap at least halved
        assert abs(toa[-1] - 1.0) < 0.01            # ~converged by 20 GV
