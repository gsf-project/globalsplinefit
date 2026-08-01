"""Tests for the HybridPCA dimensionality reduction module."""

import numpy as np
import pytest

from globalsplinefit import (
    GSFEnergy,
    GSFEnergyPerNucleon,
    GSFKineticEnergyPerNucleon,
    GSFRigidity,
)
from globalsplinefit.pca import HybridPCA, _build_stacked_system


@pytest.fixture(scope="module")
def gsf_nucleon():
    return GSFEnergyPerNucleon(version="2025")


@pytest.fixture(scope="module")
def gsf_energy():
    return GSFEnergy(version="2025")


@pytest.fixture(scope="module")
def gsf_rigidity():
    return GSFRigidity(version="2025")


@pytest.fixture(scope="module")
def pca_nucleon(gsf_nucleon):
    return HybridPCA(gsf_nucleon, n_components=12)


@pytest.fixture(scope="module")
def pca_energy(gsf_energy):
    return HybridPCA(gsf_energy, n_components=12)


class TestConstruction:
    """Test HybridPCA construction and basic attributes."""

    def test_default_construction(self, pca_nucleon):
        assert pca_nucleon.n_components == 12
        assert pca_nucleon.L_param.shape[1] == 12
        assert pca_nucleon.gauge == "correlation"
        assert pca_nucleon.D.shape[0] > 0
        assert pca_nucleon.B.shape[1:] == (8, 8)
        assert 0 < pca_nucleon.variance_explained <= 1.0

    def test_default_n_components(self, gsf_energy):
        assert HybridPCA(gsf_energy).n_components == 8

    def test_covariance_gauge_option(self, gsf_energy):
        pca = HybridPCA(gsf_energy, n_components=6, gauge="covariance")
        assert pca.gauge == "covariance"
        assert pca.B.shape[1:] == (4, 4)

    def test_invalid_gauge_raises(self, gsf_energy):
        with pytest.raises(ValueError):
            HybridPCA(gsf_energy, gauge="nope")

    def test_works_with_energy_model(self, gsf_energy):
        pca = HybridPCA(gsf_energy, n_components=8)
        assert pca.n_components == 8
        assert pca.L_param.shape[1] == 8

    def test_works_with_rigidity_model(self, gsf_rigidity):
        pca = HybridPCA(gsf_rigidity, n_components=10)
        assert pca.n_components == 10

    def test_custom_energy_grid(self, gsf_nucleon):
        grid = np.logspace(1, 8, 50)
        pca = HybridPCA(gsf_nucleon, n_components=6, energy_grid=grid)
        assert pca.n_components == 6
        assert pca.L_param.shape[1] == 6

    def test_variance_explained_reasonable(self, pca_nucleon, gsf_nucleon):
        # Correlation gauge: honest, slowly converging spectrum (~0.6 at 12).
        assert 0.3 < pca_nucleon.variance_explained < 0.95
        # Covariance gauge: trace dominated by the data-free region (>0.9).
        pca_cov = HybridPCA(gsf_nucleon, n_components=12, gauge="covariance")
        assert pca_cov.variance_explained > 0.90


class TestVariancePreservation:
    """Test that the decomposition preserves exact variances."""

    def test_variance_identity_on_reference_grid(self, gsf_nucleon):
        """sum(L**2, axis=1) + D == diag(flux_rel_cov) (up to the PSD
        projection of the residual blocks, which is at the level of the
        least-squares reprojection error)."""
        grid = np.logspace(np.log10(1.0), 11, 100)
        pca = HybridPCA(gsf_nucleon, n_components=12, energy_grid=grid)

        # Rebuild the flux-space relative covariance from scratch
        jac_stack, cov_stack, par_stack, central_flux, _, _ = _build_stacked_system(
            gsf_nucleon, grid
        )
        nonzero = central_flux != 0.0
        rel_jac = np.zeros_like(jac_stack)
        rel_jac[nonzero] = jac_stack[nonzero] / central_flux[nonzero, np.newaxis]
        flux_rel_cov = rel_jac @ cov_stack @ rel_jac.T
        exact_var = np.diag(flux_rel_cov)

        # Reconstructed variance from L_param
        L_reconstructed = rel_jac @ pca.L_param
        reconstructed_var = np.sum(L_reconstructed**2, axis=1) + pca.D

        # Compare with tolerance for floating point (some rows have zero exact variance)
        np.testing.assert_allclose(
            reconstructed_var[nonzero], exact_var[nonzero], rtol=1e-6, atol=1e-12
        )

    def test_L_reconstruction(self, gsf_nucleon):
        """rel_jac @ L_param should reconstruct L (covariance-gauge factor)."""
        grid = np.logspace(np.log10(1.0), 11, 100)
        pca = HybridPCA(
            gsf_nucleon, n_components=12, energy_grid=grid, gauge="covariance"
        )

        jac_stack, cov_stack, _, central_flux, _, _ = _build_stacked_system(
            gsf_nucleon, grid
        )
        nonzero = central_flux != 0.0
        rel_jac = np.zeros_like(jac_stack)
        rel_jac[nonzero] = jac_stack[nonzero] / central_flux[nonzero, np.newaxis]

        # Reconstruct L from eigendecomposition for comparison
        flux_rel_cov = rel_jac @ cov_stack @ rel_jac.T
        eigvals, eigvecs = np.linalg.eigh(flux_rel_cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        L = eigvecs[:, :12] * np.sqrt(np.maximum(eigvals[:12], 0.0))

        L_from_param = rel_jac @ pca.L_param
        np.testing.assert_allclose(L_from_param, L, atol=1e-7)


class TestModelInterface:
    """Test that HybridPCA mirrors the original model interface."""

    def test_flux_delegates_to_model(self, gsf_nucleon, pca_nucleon):
        """flux() should return exactly the same values as the model."""
        E = np.logspace(1, 6, 30)
        np.testing.assert_array_equal(
            pca_nucleon.flux(E, "p"),
            gsf_nucleon.flux(E, "p"),
        )

    def test_jacobian_delegates_to_model(self, gsf_nucleon, pca_nucleon):
        E = np.logspace(1, 5, 10)
        np.testing.assert_array_equal(
            pca_nucleon.jacobian(E, "He"),
            gsf_nucleon.jacobian(E, "He"),
        )

    def test_p_and_n_flux_delegates(self, gsf_nucleon, pca_nucleon):
        E = np.logspace(1, 5, 10)
        np.testing.assert_array_equal(
            pca_nucleon.p_and_n_flux(E, "He"),
            gsf_nucleon.p_and_n_flux(E, "He"),
        )

    def test_error_returns_absolute(self, gsf_nucleon, pca_nucleon):
        """error() should return absolute uncertainties, not relative."""
        E = np.logspace(2, 5, 10)
        pca_err = pca_nucleon.error(E, "p")
        model_err = gsf_nucleon.error(E, "p")
        # Should be same order of magnitude (PCA is an approximation)
        ratio = pca_err / model_err
        assert np.all(ratio > 0.5) and np.all(ratio < 2.0)

    def test_covariance_signature_matches_model(self, pca_nucleon):
        """covariance(target1, target2, energy) — same arg order as model."""
        E = np.logspace(2, 5, 10)
        cov = pca_nucleon.covariance("p", "p", E)
        assert cov.shape == (len(E), len(E))


class TestErrorAndCovariance:
    """Test error and covariance methods."""

    def test_error_positive(self, pca_nucleon):
        E = np.logspace(1, 6, 30)
        sigma = pca_nucleon.error(E, "p")
        assert np.all(sigma > 0)
        assert np.all(np.isfinite(sigma))

    def test_covariance_symmetric(self, pca_nucleon):
        E = np.logspace(2, 5, 10)
        cov = pca_nucleon.covariance("p", "p", E)
        np.testing.assert_allclose(cov, cov.T, atol=1e-15)

    def test_covariance_diagonal_matches_error(self, pca_nucleon):
        E = np.logspace(2, 5, 10)
        cov = pca_nucleon.covariance("He", "He", E)
        sigma = pca_nucleon.error(E, "He")
        np.testing.assert_allclose(np.sqrt(np.diag(cov)), sigma, rtol=1e-10)

    @pytest.mark.slow
    def test_error_close_to_full_model(self, gsf_nucleon, pca_nucleon):
        """PCA per-group error should approximate full model per-group error."""
        E = np.logspace(1, 5, 30)  # 10 GeV to 100 TeV (moderate range)
        groups = ["p", "He", "O*", "Fe*"]

        for g in groups:
            pca_err = pca_nucleon.error(E, g)
            model_err = gsf_nucleon.error(E, g)
            ratio = pca_err / model_err
            assert np.median(np.abs(ratio - 1)) < 0.02, (
                f"Group {g} median error ratio too far from 1"
            )
            assert np.all(ratio > 0.95) and np.all(ratio < 1.05), (
                f"Group {g} error out of range"
            )

    @pytest.mark.slow
    def test_total_error_close_to_full_model(self, gsf_nucleon, pca_nucleon):
        """total_error should approximate the full model's total_error.

        Per-group errors are exact to <0.2% thanks to the D_pn cross-term
        correction.  The total error sums all 16 group pairs including
        cross-group covariances (e.g. p-He, O*-Fe*) where the rank-12
        low-rank factor has limited accuracy at very high energies.
        """
        E = np.logspace(1, 6, 30)
        pca_total = pca_nucleon.total_error(E)
        model_total = gsf_nucleon.total_error(E)
        ratio = pca_total / model_total
        assert np.median(np.abs(ratio - 1)) < 0.01
        assert np.all(ratio > 0.97) and np.all(ratio < 1.03)

    def test_p_and_n_covariance(self, pca_nucleon):
        E = np.logspace(2, 5, 10)
        cov_pp, cov_nn = pca_nucleon.p_and_n_covariance("He", "He", E)
        assert cov_pp.shape == (len(E), len(E))
        assert cov_nn.shape == (len(E), len(E))
        # Both should be PSD
        assert np.all(np.diag(cov_pp) >= 0)
        assert np.all(np.diag(cov_nn) >= 0)

    @pytest.mark.slow
    def test_cross_term_covariance(self, gsf_nucleon, pca_nucleon):
        """PCA total covariance should match the full model for all groups.

        This verifies that the D_pn cross-term correction correctly captures
        the p-n covariance contribution for nuclei where Z ~ A-Z.
        """
        E = np.logspace(1, 5, 20)
        groups = ["p", "He", "O*", "Fe*"]

        for g in groups:
            pca_cov = pca_nucleon.covariance(g, g, E)
            model_cov = gsf_nucleon.covariance(g, g, E)
            ratio = np.diag(pca_cov) / np.diag(model_cov)
            np.testing.assert_allclose(
                ratio,
                1.0,
                atol=0.02,
                err_msg=f"Group {g} covariance diagonal mismatch",
            )

    def test_p_and_n_error(self, pca_nucleon):
        E = np.logspace(2, 5, 10)
        err = pca_nucleon.p_and_n_error(E, "He")
        assert err.shape == (2, len(E))
        assert np.all(err >= 0)


class TestReducedJacobian:
    """Test reduced_jacobian PCA-specific method."""

    def test_full_shape_nucleon_model(self, pca_nucleon):
        E = np.logspace(1, 6, 20)
        M = pca_nucleon.reduced_jacobian(E)
        # 4 groups * 2 (p+n) * 20 energies = 160 rows
        assert M.shape == (160, 12)

    def test_per_group_shape(self, pca_nucleon):
        E = np.logspace(1, 6, 20)
        M = pca_nucleon.reduced_jacobian(E, target="p")
        # 1 group * 2 (p+n) * 20 energies = 40 rows
        assert M.shape == (40, 12)

    def test_full_shape_energy_model(self, pca_energy):
        E = np.logspace(1, 6, 20)
        M = pca_energy.reduced_jacobian(E)
        # 4 groups * 20 energies = 80 rows
        assert M.shape == (80, 12)

    def test_off_grid_evaluation(self, pca_nucleon):
        E_off = np.array([3.7, 42.0, 1337.0, 5e7])
        M = pca_nucleon.reduced_jacobian(E_off)
        assert np.all(np.isfinite(M))


class TestSampling:
    """Test Monte Carlo sampling."""

    def test_sample_shape(self, pca_nucleon):
        E = np.logspace(2, 5, 10)
        samples = pca_nucleon.sample(100, E)
        assert samples.shape == (100, 80)  # 4 groups * 2 * 10

    def test_sample_reproducible(self, pca_nucleon):
        E = np.logspace(2, 5, 10)
        s1 = pca_nucleon.sample(50, E, rng=np.random.default_rng(42))
        s2 = pca_nucleon.sample(50, E, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(s1, s2)

    @pytest.mark.slow
    def test_total_error_exact_energy_model(self, gsf_energy):
        """All-particle error must match the full model (block correction)."""
        pca = HybridPCA(gsf_energy, n_components=4)
        E = np.logspace(1.5, 8.5, 25)
        ratio = pca.total_error(E) / gsf_energy.total_error(E)
        assert np.all(ratio > 0.97) and np.all(ratio < 1.05)

    @pytest.mark.slow
    def test_rank_independence_of_totals(self, gsf_energy):
        """Fixed-energy observables must not depend on n_components."""
        E = np.logspace(2, 8, 15)
        t4 = HybridPCA(gsf_energy, n_components=4).total_error(E)
        t8 = HybridPCA(gsf_energy, n_components=8).total_error(E)
        np.testing.assert_allclose(t4, t8, rtol=0.02)

    @pytest.mark.slow
    def test_default_samples_match_reduced_covariance(self, pca_nucleon):
        """Default samples include B and match the complete reduced covariance."""
        E = np.logspace(2, 5, 8)
        rng = np.random.default_rng(1)
        samples = pca_nucleon.sample(30000, E, rng=rng)
        M, central = pca_nucleon._reduced_jacobian_full(E)

        # Assemble the same relative covariance as covariance(): low-rank
        # cross-energy structure plus the complete same-energy B blocks.
        expected = M @ M.T
        B = pca_nucleon._interpolate_B(E)
        offsets = pca_nucleon._sub_row_offsets(len(E))
        idx = np.arange(len(E))
        for a, off_a in enumerate(offsets):
            for b, off_b in enumerate(offsets):
                expected[off_a + idx, off_b + idx] += B[:, a, b]

        relative_samples = samples / central - 1.0
        observed = np.cov(relative_samples, rowvar=False, ddof=0)
        scale = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
        normalized_difference = np.divide(
            observed - expected,
            scale,
            out=np.zeros_like(expected),
            where=scale > 0,
        )
        assert np.max(np.abs(normalized_difference)) < 0.04

    def test_smooth_sampling_is_explicit_opt_out(self, pca_nucleon):
        """residual_noise=False drops the per-energy residual term."""
        E = np.logspace(2, 5, 8)
        seed = 1234
        smooth = pca_nucleon.sample(
            20, E, rng=np.random.default_rng(seed), residual_noise=False
        )
        full = pca_nucleon.sample(
            20, E, rng=np.random.default_rng(seed), residual_noise=True
        )
        # same low-rank draw, so any difference is the residual noise alone
        assert not np.array_equal(smooth, full)
        assert smooth.shape == full.shape
        assert np.std(full - smooth) > 0

    def test_retired_diagonal_noise_alias_is_gone(self, pca_nucleon):
        """The old ``diagonal_noise`` spelling is not silently swallowed."""
        with pytest.raises(TypeError, match="diagonal_noise"):
            pca_nucleon.sample(
                2, np.logspace(2, 5, 4), rng=np.random.default_rng(0),
                diagonal_noise=False,
            )

    @pytest.mark.slow
    def test_sample_statistics(self, pca_nucleon):
        """Mean of many samples should approximate central flux."""
        E = np.logspace(2, 5, 10)
        rng = np.random.default_rng(0)
        samples = pca_nucleon.sample(10000, E, rng=rng)
        mean = samples.mean(axis=0)

        from globalsplinefit.pca import _build_jacobian_at_energy

        _, central = _build_jacobian_at_energy(pca_nucleon.model, E)

        np.testing.assert_allclose(mean, central, rtol=0.05)


class TestAllModelTypes:
    """Test that HybridPCA works with all GSF model types."""

    def test_gsf_energy(self, gsf_energy):
        pca = HybridPCA(gsf_energy, n_components=8)
        E = np.logspace(1, 5, 10)
        err = pca.error(E, "p")
        assert err.shape == (10,)
        assert np.all(err > 0)

    def test_gsf_rigidity(self, gsf_rigidity):
        pca = HybridPCA(gsf_rigidity, n_components=8)
        R = np.logspace(0, 4, 10)
        err = pca.error(R, "p")
        assert err.shape == (10,)
        assert np.all(err > 0)

    def test_gsf_kinetic_energy_per_nucleon(self):
        gsf = GSFKineticEnergyPerNucleon(version="2025")
        pca = HybridPCA(gsf, n_components=8)
        E = np.logspace(1, 5, 10)
        err = pca.error(E, "p")
        assert err.shape == (10,)
        assert np.all(err > 0)

    def test_energy_model_error_close_to_full(self, gsf_energy):
        """PCA error should approximate full model error for GSFEnergy too."""
        pca = HybridPCA(gsf_energy, n_components=12)
        E = np.logspace(1, 6, 20)
        pca_err = pca.error(E, "p")
        model_err = gsf_energy.error(E, "p")
        np.testing.assert_allclose(pca_err, model_err, rtol=0.10)
