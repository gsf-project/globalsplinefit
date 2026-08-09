"""Coverage for the shipped parameter sets and the default version.

2026 is the default equal-weight SIBYLL-2.3e/EPOS-LHC-R
mixture covering with the Ghelfi-Maurin-Derome modulation potential.
2026-USO is the same mixture fit with the Usoskin 2017 potential, which
yields a lower low-rigidity local interstellar spectrum. 2026-SIB23e and
2026-EPOSLHCR are the single-interpretation variants (SIBYLL-2.3e and
EPOS-LHC-R, both GMD). All 2026 sets are isotope-format sets
(they carry deuterium and the He-isotope split), so bare integer-charge access
is not defined for the multi-species charges Z=1 (p+D) and Z=2 (3He+4He); the
name/tuple flux API is used throughout here.
"""

import numpy as np
import pytest

from globalsplinefit import GSFEnergy, GSFRigidity
from globalsplinefit.data_management import (
    DEFAULT_VERSION,
    MODEL_VERSIONS,
    Parameters,
    get_available_versions,
    resolve_version,
    version_info,
)


def test_new_sets_available():
    versions = get_available_versions()
    assert "2026.0" in versions
    assert "2026.0-USO" in versions
    assert "2026.0-SIB23e" in versions
    assert "2026.0-EPOSLHCR" in versions


def test_unrevisioned_names_resolve_to_newest_revision():
    """ "2026" / "2026-USO" float to the newest registered revision; the
    revisioned name pins one. Unknown names fail loudly."""
    assert resolve_version("2026") == "2026.0"
    assert resolve_version("2026-USO") == "2026.0-USO"
    assert resolve_version("2026.0") == "2026.0"  # exact pin
    assert resolve_version("2025") == "2025"  # pre-scheme release
    assert resolve_version(None) == resolve_version(DEFAULT_VERSION)
    with pytest.raises(ValueError, match="not found"):
        resolve_version("2026.9")
    with pytest.raises(ValueError, match="not found"):
        resolve_version("2031")
    # alias and revisioned name share one cached parameter bundle
    assert Parameters.for_model(version="2026") is Parameters.for_model(
        version="2026.0"
    )


def test_default_is_gsf2026():
    """A bare GSFEnergy() must resolve to the 2026 (GMD) set."""
    default = GSFEnergy()
    explicit = GSFEnergy(version="2026")
    E = np.logspace(0, 4, 50)
    for g in default.active_groups:
        np.testing.assert_allclose(default.flux(E, g), explicit.flux(E, g), rtol=1e-12)
    assert DEFAULT_VERSION == "2026.0"


def test_resolved_version_is_reported():
    """A model must report the version it actually loaded, not the argument.

    Regression guard: `self.version` used to be the raw constructor argument, so
    a default-constructed model reported None and nothing could tell which set
    was in use.
    """
    assert GSFEnergy().version == "2026.0"
    assert GSFEnergy(version="2026-USO").version == "2026.0-USO"
    assert GSFEnergy(version="2017").version == "2017"


def test_only_registered_versions_are_offered():
    """get_available_versions() is an allow-list, not a directory listing.

    Guards the failure this registry was introduced for: a transient
    single-interpretation fit exported into data/ must not become a
    distributable version by accident.
    """
    for name in get_available_versions():
        assert name in MODEL_VERSIONS, f"{name} is offered but not registered"
    current = get_available_versions(include_historical=False)
    assert set(current) == {
        "2026.0",
        "2026.0-USO",
        "2026.0-SIB23e",
        "2026.0-EPOSLHCR",
    }
    for name in current:
        assert MODEL_VERSIONS[name]["status"] == "current"


def test_unregistered_data_dir_warns_and_is_not_offered(tmp_path, monkeypatch):
    """An unregistered directory under data/ warns and stays unavailable."""
    import globalsplinefit.data_management as dm

    fake_data = tmp_path / "data"
    for name in ("2026.0", "rogue_variant"):
        d = fake_data / name
        d.mkdir(parents=True)
        for f in ("knots.dat", "nuclei.dat", "parameters.dat", "covariance.dat"):
            (d / f).write_text("")
    monkeypatch.setattr(dm, "__file__", str(tmp_path / "data_management.py"))
    with pytest.warns(UserWarning, match="rogue_variant"):
        versions = dm.get_available_versions()
    assert versions == ["2026.0"]
    assert "rogue_variant" not in versions


def test_current_sets_are_the_mixture_and_declare_provenance():
    """Both distributed sets must be the mixture, and must say so.

    The shipped sets were once single-interpretation (SIBYLL-only) fits while the
    docs advertised the paper's mixture; nothing in the data recorded the
    covering, so the mismatch was invisible. This asserts the provenance ships.
    """
    for version, phi in (("2026", "GMD"), ("2026-USO", "USO")):
        prov = GSFEnergy(version=version).params.provenance
        assert "mixture" in prov["covering"].lower(), version
        assert "SIBYLL" in prov["covering"] and "EPOS" in prov["covering"], version
        assert prov["solar_modulation_source"].startswith(phi), version
        assert set(prov["mixture_components"]) == {"A", "B"}, version
        assert prov["registry"]["status"] == "current"
        assert version_info(version)["covering"] == prov["registry"]["covering"]


@pytest.mark.parametrize(
    "version, model, absent",
    [
        ("2026-SIB23e", "SIBYLL", "EPOS"),
        ("2026-EPOSLHCR", "EPOS", "SIBYLL"),
    ],
)
def test_single_interpretation_variants(version, model, absent):
    """The single-interpretation sets must be pure one-model fits (no mixture)
    under the same GMD potential as the default, and must say so."""
    prov = GSFEnergy(version=version).params.provenance
    assert model in prov["covering"] and "mixture" not in prov["covering"].lower()
    assert absent not in prov["covering"]
    assert prov["solar_modulation_source"].startswith("GMD")
    assert prov["registry"]["status"] == "current"


def test_historical_sets_are_marked_historical():
    """Legacy releases must not read as alternatives to the current fit."""
    for name in ("2017", "2019", "2025"):
        assert MODEL_VERSIONS[name]["status"] == "historical"


@pytest.mark.parametrize(
    "version", ["2026", "2026-USO", "2026-SIB23e", "2026-EPOSLHCR"]
)
def test_set_loads_and_is_positive(version):
    m = GSFEnergy(version=version)
    E = np.logspace(0, 6, 80)
    for g in m.active_groups:
        f = m.flux(E, g)
        assert np.all(np.isfinite(f))  # never NaN/inf, even below threshold
    # above the heaviest group's threshold all four groups are populated
    E_hi = np.logspace(np.log10(30), 6, 60)
    for g in m.active_groups:
        assert np.all(m.flux(E_hi, g) > 0)


def test_usoskin_lowers_low_rigidity_lis():
    """The Usoskin potential runs lower than the GMD default, so its
    demodulated (LIS) flux is lower at low rigidity and converges to the
    default at high rigidity."""
    gmd = GSFRigidity(version="2026")  # default
    uso = GSFRigidity(version="2026-USO")
    R = np.array([1.5, 5.0, 50.0])
    for g in ("p", "He"):
        jg = gmd.flux(R, g, time_interval="LIS")
        ju = uso.flux(R, g, time_interval="LIS")
        assert ju[0] / jg[0] - 1.0 < -0.05  # >5% lower at 1.5 GV
        assert abs(ju[-1] / jg[-1] - 1.0) < 0.02  # converged by 50 GV


def test_gsf2026_ships_its_own_phi_table():
    """2026 (GMD) carries a version-local solar_modulation.dat; the fitted
    LIS must be re-modulated with the potential it was demodulated with. The
    Usoskin variant ships its own copy of the Usoskin table (identical to the
    shared one the legacy sets fall back to).
    """
    gmd = GSFRigidity(version="2026")
    uso = GSFRigidity(version="2026-USO")
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
    gmd = GSFRigidity(version="2026")
    uso = GSFRigidity(version="2026-USO")
    R = np.array([1.5, 5.0, 20.0])
    epoch = (201105, 201805)  # AMS-02 era
    for g in ("p", "He"):
        lis_gap = abs(
            uso.flux(R, g, time_interval="LIS")[0]
            / gmd.flux(R, g, time_interval="LIS")[0]
            - 1.0
        )
        toa = uso.flux(R, g, time_interval=epoch) / gmd.flux(R, g, time_interval=epoch)
        assert lis_gap > 0.08  # LIS differ markedly at 1.5 GV
        assert abs(toa[0] - 1.0) < 0.5 * lis_gap  # TOA gap at least halved
        assert abs(toa[-1] - 1.0) < 0.01  # ~converged by 20 GV
