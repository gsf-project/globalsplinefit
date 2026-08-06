"""Tests for the pivot-based reduced representation (ReducedGSF)."""

import json

import numpy as np
import pytest

from globalsplinefit import (
    RECOMMENDED_PIVOTS,
    GSFEnergy,
    GSFEnergyPerNucleon,
    GSFKineticEnergyPerNucleon,
    ReducedGSF,
    optimize_pivots,
)


@pytest.fixture(scope="module")
def gsf():
    return GSFEnergyPerNucleon()


@pytest.fixture(scope="module", params=["spline", "hat"])
def red(gsf, request):
    return ReducedGSF(gsf, basis=request.param)


@pytest.fixture(scope="module")
def red_groups(gsf):
    return ReducedGSF(gsf, n_pivots=6, energy_range=(2.0, 1e8), per_group=True)


class TestConstruction:
    def test_default_is_published_grid(self, red):
        """All-default construction uses the citable RECOMMENDED_PIVOTS."""
        assert red.species == ["p", "n"]
        assert np.allclose(red.pivot_energies, RECOMMENDED_PIVOTS["2026.0"])
        assert red.n_params == 24
        assert red.cov.shape == (24, 24)
        assert len(red.labels) == 24
        assert red.labels[0] == "p_1GeV"
        assert red.labels[3] == "p_9TeV"
        assert red.labels[4] == "p_25TeV"
        assert red.labels[12] == "n_1GeV"
        assert red.labels[-1] == "n_1EeV"

    def test_explicit_n_pivots_gives_log_spaced(self, gsf):
        r = ReducedGSF(gsf, n_pivots=10)
        assert r.n_params == 20
        assert np.allclose(r.pivot_energies, np.logspace(0, 9, 10))

    def test_custom_energy_range_gives_log_spaced(self, gsf):
        r = ReducedGSF(gsf, energy_range=(2.0, 1e8))
        assert len(r.pivot_energies) == 10
        assert r.pivot_energies[0] == pytest.approx(2.0)

    def test_historical_version_uses_fixed_grid(self):
        """Every supported version constructs immediately from a fixed grid."""
        model = GSFEnergyPerNucleon(version="2025")
        reduced = ReducedGSF(model)
        assert np.array_equal(reduced.pivot_energies, RECOMMENDED_PIVOTS["2025"])

    def test_every_shipped_version_has_a_pivot_table(self):
        from globalsplinefit.data_management import MODEL_VERSIONS

        assert set(RECOMMENDED_PIVOTS) == set(MODEL_VERSIONS)
        for version, pivots in RECOMMENDED_PIVOTS.items():
            grid = np.asarray(pivots)
            assert len(grid) == 12, version
            assert np.all(np.diff(grid) > 0), version
            assert grid[0] == 1.0 and grid[-1] == 1e9, version

    def test_every_version_has_its_own_grid(self):
        """Each shipped set carries its own optimized pivot table."""
        grids = list(RECOMMENDED_PIVOTS.values())
        assert len(grids) == len(set(grids))

    def test_shipped_grid_matches_bundle_file(self):
        model = GSFEnergyPerNucleon(version="2019")
        reduced = ReducedGSF(model)
        assert np.array_equal(reduced.pivot_energies, RECOMMENDED_PIVOTS["2019"])
        assert np.array_equal(model.params.reduced_pivots, reduced.pivot_energies)

    def test_custom_bundle_without_table_raises(self, tmp_path):
        import shutil
        from pathlib import Path

        source = Path(GSFEnergyPerNucleon(version="2025").params.data_path)
        bundle = tmp_path / "bundle"
        shutil.copytree(source, bundle)
        (bundle / "reduced_pivots.dat").unlink()
        model = GSFEnergyPerNucleon(data_path=bundle)
        assert model.params.reduced_pivots is None
        with pytest.raises(ValueError, match="reduced_pivots.dat"):
            ReducedGSF(model)
        # explicit pivots (as optimize_pivots would supply) still work
        reduced = ReducedGSF(model, pivot_energies=RECOMMENDED_PIVOTS["2025"])
        assert reduced.n_params == 24

    def test_custom_bundle_with_table_uses_it(self, tmp_path):
        import shutil
        from pathlib import Path

        source = Path(GSFEnergyPerNucleon(version="2025").params.data_path)
        bundle = tmp_path / "bundle"
        shutil.copytree(source, bundle)
        (bundle / "reduced_pivots.dat").write_text(
            "# custom grid\n1.0\n1e3\n1e6\n1e9\n"
        )
        model = GSFEnergyPerNucleon(data_path=bundle)
        reduced = ReducedGSF(model)
        assert np.array_equal(reduced.pivot_energies, [1.0, 1e3, 1e6, 1e9])

    def test_per_group_layout(self, red_groups):
        assert len(red_groups.species) == 8
        assert red_groups.n_params == 48
        assert red_groups.species[0] == "H_p"
        assert red_groups.species[-1] == "Fe*_n"

    def test_cov_is_symmetric_psd(self, red):
        assert np.allclose(red.cov, red.cov.T)
        w = np.linalg.eigvalsh(red.cov)
        assert w.min() >= -1e-14

    def test_explicit_pivots(self, gsf):
        pv = np.array([10.0, 1e3, 1e5])
        r = ReducedGSF(gsf, pivot_energies=pv)
        assert np.allclose(r.pivot_energies, pv)
        assert r.n_params == 6

    def test_rejects_non_nucleon_model(self):
        with pytest.raises(TypeError):
            ReducedGSF(GSFEnergy())

    def test_rejects_bad_pivots(self, gsf):
        with pytest.raises(ValueError):
            ReducedGSF(gsf, pivot_energies=[1e3])
        with pytest.raises(ValueError):
            ReducedGSF(gsf, pivot_energies=[1e3, 1e2])
        with pytest.raises(ValueError):
            ReducedGSF(gsf, pivot_energies=[[1.0, 10.0], [100.0, 1000.0]])

    @pytest.mark.parametrize("n_pivots", [True, 1, 2.5])
    def test_rejects_bad_pivot_count(self, gsf, n_pivots):
        with pytest.raises(ValueError, match="n_pivots"):
            ReducedGSF(gsf, n_pivots=n_pivots)

    def test_arrays_are_read_only(self, red):
        assert not red.pivot_energies.flags.writeable
        assert not red.cov.flags.writeable

    def test_rejects_unknown_basis(self, gsf):
        with pytest.raises(ValueError, match="basis"):
            ReducedGSF(gsf, basis="fourier")

    def test_cov_independent_of_basis(self, gsf):
        a = ReducedGSF(gsf, n_pivots=5, basis="spline")
        b = ReducedGSF(gsf, n_pivots=5, basis="hat")
        assert np.allclose(a.cov, b.cov)

    def test_kinetic_model(self):
        r = ReducedGSF(GSFKineticEnergyPerNucleon(), n_pivots=4)
        assert r.n_params == 8

    def test_rejects_vanishing_central_flux(self, gsf):
        # heavy-group fluxes underflow far beyond 1e9 GeV/nucleon
        with pytest.raises(ValueError, match="vanishes"):
            ReducedGSF(gsf, energy_range=(1.0, 1e11), per_group=True)


class TestBasis:
    def test_partition_of_unity(self, red):
        E = np.logspace(0.1, 8.9, 137)
        H = red.basis(E)
        assert np.allclose(H.sum(axis=1), 1.0)
        if red.basis_type == "hat":
            assert H.min() >= 0.0
        else:
            # cardinal cubic splines ring, but the side lobes are bounded
            assert H.min() > -0.25

    def test_spline_is_smooth_hat_is_not(self, gsf):
        """C1 continuity at an interior pivot for the spline basis only."""
        pv = ReducedGSF(gsf).pivot_energies[4]
        eps = 1e-4
        E = pv * np.exp(np.array([-2 * eps, -eps, 0.0, eps, 2 * eps]))
        for basis, smooth in [("spline", True), ("hat", False)]:
            H = ReducedGSF(gsf, basis=basis).basis(E)
            d_left = (H[2] - H[0]) / (2 * eps)
            d_right = (H[4] - H[2]) / (2 * eps)
            kink = np.abs(d_right - d_left).max()
            assert (kink < 1e-3) == smooth

    def test_exact_at_pivots(self, red):
        H = red.basis(red.pivot_energies)
        assert np.allclose(H, np.eye(len(red.pivot_energies)), atol=1e-12)

    def test_frozen_outside_range(self, red):
        n = len(red.pivot_energies)
        H = red.basis([1e-2, 1e11])
        assert np.allclose(H[0], np.eye(n)[0])
        assert np.allclose(H[1], np.eye(n)[-1])


class TestFlux:
    def test_central_matches_model(self, red, gsf):
        E = np.logspace(1, 6, 40)
        assert np.allclose(red.flux(E), gsf.p_and_n_total_flux(E))

    def test_theta_zero_is_central(self, red):
        E = np.logspace(1, 6, 40)
        assert np.allclose(red.flux(E, np.zeros(red.n_params)), red.flux(E))

    def test_physical_configuration_cannot_be_changed_after_construction(self, gsf):
        red = ReducedGSF(gsf, n_pivots=4, energy_range=(2.0, 1e9), time_interval="LIS")
        red.flux([10.0], time_interval="LIS")
        with pytest.raises(ValueError, match="construct a new reduction"):
            red.flux([10.0], time_interval=(200901, 201001))

    def test_unit_bump_at_pivot(self, red):
        theta = np.zeros(red.n_params)
        theta[3] = 0.1  # p component at pivot 3
        f = red.flux(red.pivot_energies, theta)
        f0 = red.flux(red.pivot_energies)
        ratio = f / f0
        assert ratio[0, 3] == pytest.approx(1.1)
        # other pivots of p and all of n unaffected
        assert np.allclose(np.delete(ratio[0], 3), 1.0)
        assert np.allclose(ratio[1], 1.0)

    def test_jacobian_matches_finite_difference(self, red):
        E = np.logspace(1, 8, 25)
        jac = red.flux_jacobian(E)
        rng = np.random.default_rng(42)
        theta = 0.05 * rng.standard_normal(red.n_params)
        eps = 1e-6
        for j in [0, 7, 13, 19]:
            dt = np.zeros(red.n_params)
            dt[j] = eps
            fd = (red.flux(E, theta + dt) - red.flux(E, theta - dt)) / (2 * eps)
            assert np.allclose(
                jac[:, :, j], fd, rtol=1e-5, atol=1e-8 * np.abs(fd).max()
            )


class TestPublishedGrid:
    # Measured worst-case coverage factors of each shipped per-version grid
    # (150-point check grid) plus a small margin. With the 2026 grid applied
    # everywhere these were 1.28 / 1.53 / 1.38 (USO / S23e / EPOS-LHCR) and
    # 2.76 / 2.45 / 1.78 (2025 / 2019 / 2017).
    COVERAGE_BOUNDS = {
        "2026.0": 1.30,
        "2026.0-USO": 1.27,
        "2026.0-S23e": 1.37,
        "2026.0-EPOS-LHCR": 1.28,
        "2025": 1.25,
        "2019": 1.56,
        "2017": 1.22,
    }

    @pytest.mark.parametrize("version", sorted(COVERAGE_BOUNDS))
    def test_coverage_bound(self, version):
        """The quoted worst-case coverage factor of each published grid."""
        model = GSFEnergyPerNucleon(version=version)
        E = np.logspace(0, 9, 150)
        exact = _exact_total_error(model, E)
        red = ReducedGSF(model)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.log(red.error(E) / exact)
        dev = np.exp(np.abs(ratio[np.isfinite(ratio)]).max())
        assert dev < self.COVERAGE_BOUNDS[version]

    def test_min_separation_respected(self, gsf):
        pivots, _ = optimize_pivots(
            gsf, n_pivots=6, n_grid=80, n_restarts=1, max_sweeps=3
        )
        assert np.diff(np.log10(pivots)).min() >= 0.15 * 0.9


class TestCovariance:
    def test_error_exact_at_pivots_total(self, red, gsf):
        """At the pivots the reduced sigma equals the full model's."""
        pv = red.pivot_energies
        n = len(pv)
        # exact total p / total n variance: sum group-pair covariance diagonals
        var_p = np.zeros(n)
        var_n = np.zeros(n)
        for g1 in gsf.active_groups:
            for g2 in gsf.active_groups:
                cpp, cnn = gsf.p_and_n_covariance(g1, g2, pv)
                var_p += np.diag(cpp)
                var_n += np.diag(cnn)
        sig = red.error(pv)
        assert np.allclose(sig[0], np.sqrt(var_p), rtol=1e-8)
        assert np.allclose(sig[1], np.sqrt(var_n), rtol=1e-8)

    def test_error_exact_at_pivots_per_group(self, red_groups, gsf):
        pv = red_groups.pivot_energies
        for s, g in enumerate(gsf.active_groups):
            exact = gsf.p_and_n_error(pv, g)
            assert np.allclose(red_groups.error(pv)[2 * s], exact[0], rtol=1e-8)
            assert np.allclose(red_groups.error(pv)[2 * s + 1], exact[1], rtol=1e-8)

    def test_sample_covariance_converges(self, red):
        theta = red.sample(40000, rng=np.random.default_rng(7))
        assert theta.shape == (40000, red.n_params)
        assert np.allclose(theta.mean(axis=0), 0.0, atol=4 * red.sigma.max() / 200)
        emp = np.cov(theta.T)
        assert np.allclose(np.sqrt(np.diag(emp)), red.sigma, rtol=0.05)

    def test_penalty_whitened_identity(self, red):
        """penalty(cov^1/2 z) == |z|^2."""
        rng = np.random.default_rng(3)
        w, u = np.linalg.eigh(red.cov)
        a = u * np.sqrt(np.maximum(w, 0.0))
        z = rng.standard_normal(red.n_params)
        assert red.penalty(a @ z) == pytest.approx(z @ z, rel=1e-8)

    def test_penalty_zero(self, red):
        assert red.penalty(np.zeros(red.n_params)) == 0.0


class TestOptimizePivots:
    """Small grids / few sweeps to keep the exchange loop fast."""

    def test_improves_on_log_spaced(self, gsf):
        n_grid, n_piv = 80, 6
        pivots, worst = optimize_pivots(
            gsf, n_pivots=n_piv, n_grid=n_grid, n_restarts=0, max_sweeps=5
        )
        assert len(pivots) == n_piv
        assert np.all(np.diff(pivots) > 0)
        assert pivots[0] == pytest.approx(1.0)
        assert pivots[-1] == pytest.approx(1e9)
        # compare against the log-spaced start in the same metric
        E = np.logspace(0, 9, 200)
        r_opt = ReducedGSF(gsf, pivot_energies=pivots)
        r_log = ReducedGSF(gsf, n_pivots=n_piv)
        exact = _exact_total_error(gsf, E)
        dev = lambda r: np.abs(np.log(r.error(E) / exact)).max()  # noqa: E731
        assert dev(r_opt) < dev(r_log)
        assert worst > 1.0  # the reported factor is meaningful

    def test_reported_factor_matches_grid(self, gsf):
        pivots, worst = optimize_pivots(
            gsf, n_pivots=5, n_grid=60, n_restarts=0, max_sweeps=3
        )
        E = np.logspace(0, 9, 60)
        r = ReducedGSF(gsf, pivot_energies=pivots)
        dev = np.abs(np.log(r.error(E) / _exact_total_error(gsf, E))).max()
        assert np.exp(dev) == pytest.approx(worst, rel=1e-6)

    def test_deterministic(self, gsf):
        a, _ = optimize_pivots(gsf, n_pivots=5, n_grid=60, n_restarts=1, max_sweeps=3)
        b, _ = optimize_pivots(gsf, n_pivots=5, n_grid=60, n_restarts=1, max_sweeps=3)
        assert np.allclose(a, b)

    def test_rejects_bad_args(self, gsf):
        with pytest.raises(TypeError):
            optimize_pivots(GSFEnergy())
        with pytest.raises(ValueError):
            optimize_pivots(gsf, basis="fourier")
        with pytest.raises(ValueError):
            optimize_pivots(gsf, n_pivots=2)


def _exact_total_error(gsf, energy):
    var_p = np.zeros(len(energy))
    var_n = np.zeros(len(energy))
    for g1 in gsf.active_groups:
        for g2 in gsf.active_groups:
            cpp, cnn = gsf.p_and_n_covariance(g1, g2, energy)
            var_p += np.diag(cpp)
            var_n += np.diag(cnn)
    return np.vstack([np.sqrt(var_p), np.sqrt(var_n)])


class TestExport:
    def test_to_dict_roundtrips_json(self, red):
        d = red.to_dict()
        s = json.dumps(d)
        d2 = json.loads(s)
        assert d2["species"] == ["p", "n"]
        assert len(d2["cov"]) == red.n_params
        assert np.allclose(d2["cov"], red.cov)
        assert d2["model_version"] == red.model.version
