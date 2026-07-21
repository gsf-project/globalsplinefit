"""
Test suite to validate the vectorized solar modulation implementation.

This test suite serves two purposes:
1. Ensures the vectorized implementation produces correc        # Calculate using vectorized method
        flux_jacobian_vectorized = gsf_energy._element_flux_jacobian(z, energies, time_interval)results
2. Documents how the explicit (loop-based) implementation works for reference

The explicit implementations shown here are the reference implementations
that the vectorized versions should match exactly.
"""

import numpy as np
import pytest

from globalsplinefit.model import GSFEnergy, GSFEnergyPerNucleon


class TestVectorizedImplementation:
    """Test vectorized solar modulation implementation against reference explicit versions."""

    @pytest.fixture
    def gsf_energy(self):
        """Create GSFEnergy model for testing (2017 set: bare-int charge access)."""
        return GSFEnergy(version="2017")

    @pytest.fixture
    def gsf_nucleon(self):
        """Create GSFEnergyPerNucleon model for testing (2017 set)."""
        return GSFEnergyPerNucleon(version="2017")

    def _reference_rigidity_from_energy(self, model, z, energy, phi):
        """Reference implementation of rigidity conversion (explicit, single phi)."""
        nucleon_mass = 0.93891872965  # GeV
        mass = model.z_to_a[z] * nucleon_mass
        energy_is = energy + z * phi

        # Check for invalid energies (below rest mass)
        invalid_mask = energy_is <= mass

        # Handle potential division by zero or invalid values
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = (energy**2 - mass**2) / (energy_is**2 - mass**2)

        p2 = energy_is**2 - mass**2
        p2[p2 < 0] = 0.0
        rig = p2**0.5 / z

        # Handle potential division by zero or invalid values
        with np.errstate(divide="ignore", invalid="ignore"):
            factor *= energy_is / (rig * z**2 + 1e-300)

        # Set invalid results to zero (when energy is too low or factor is negative)
        factor[invalid_mask | (factor < 0) | ~np.isfinite(factor)] = 0.0
        rig[invalid_mask | ~np.isfinite(rig)] = 0.0

        return rig, factor

    def _reference_element_flux_explicit(self, model, z, energy, time_interval=None):
        """Reference implementation of element flux calculation (explicit loop over phi).

        This is the reference implementation that shows how solar modulation
        averaging is done explicitly by looping over phi values.
        """
        energy = np.atleast_1d(energy)

        result = 0.0
        phis = model._phi_list(time_interval)

        # This is the key part: explicit loop over phi values
        for phi in phis:
            rig, factor = self._reference_rigidity_from_energy(model, z, energy, phi)
            result += model._rigidity_flux_lis(z, rig) * factor

        return result / len(phis)

    def _reference_element_flux_jacobian_explicit(
        self, model, z, energy, time_interval=None
    ):
        """Reference implementation of Jacobian calculation (explicit loop over phi).

        This shows how the Jacobian (used for uncertainty propagation) is calculated
        explicitly by looping over phi values.
        """
        energy = np.atleast_1d(energy)
        leading, ratio = model.flux_ratio[z]

        jac = 0.0
        phis = model._phi_list(time_interval)

        # This is the key part: explicit loop over phi values
        for phi in phis:
            rig, factor = self._reference_rigidity_from_energy(model, z, energy, phi)
            jac += model._rigidity_flux_jacobian(leading, rig) * factor[:, np.newaxis]

        return ratio * jac / len(phis)

    def test_rigidity_from_energy_vectorized_vs_explicit(self, gsf_energy):
        """Test that vectorized rigidity conversion matches explicit version."""
        z = 8  # Oxygen
        energies = np.logspace(1, 3, 10)  # 10 GeV to 1 TeV
        phis = np.array([0.0, 300.0, 600.0])  # Range of phi values

        # Test each phi value individually (explicit style)
        explicit_results = []
        for phi in phis:
            rig, factor = self._reference_rigidity_from_energy(
                gsf_energy, z, energies, phi
            )
            explicit_results.append((rig, factor))

        # Test vectorized version
        rig_vec, factor_vec = gsf_energy._rigidity_from_energy_vectorized(
            z, energies, phis
        )

        # Compare results
        for i, phi in enumerate(phis):
            rig_exp, factor_exp = explicit_results[i]
            rig_v = rig_vec[:, i]
            factor_v = factor_vec[:, i]

            np.testing.assert_allclose(
                rig_v,
                rig_exp,
                rtol=1e-14,
                atol=1e-16,
                err_msg=f"Rigidity mismatch for phi={phi}",
            )
            np.testing.assert_allclose(
                factor_v,
                factor_exp,
                rtol=1e-14,
                atol=1e-16,
                err_msg=f"Factor mismatch for phi={phi}",
            )

    @pytest.mark.parametrize(
        "z,name",
        [
            (1, "Proton"),
            (2, "Helium"),
            (8, "Oxygen"),
            (26, "Iron"),
        ],
    )
    @pytest.mark.parametrize(
        "time_interval",
        [
            None,  # Solar cycle average
            "LIS",  # Local interstellar spectrum
            (200901, 200912),  # Specific period
        ],
    )
    def test_element_flux_vectorized_vs_explicit(
        self, gsf_energy, z, name, time_interval
    ):
        """Test that vectorized element flux matches explicit implementation."""
        energies = np.logspace(1, 3, 15)  # 10 GeV to 1 TeV

        # Calculate using reference explicit method
        flux_explicit = self._reference_element_flux_explicit(
            gsf_energy, z, energies, time_interval
        )

        # Calculate using vectorized method
        flux_vectorized = gsf_energy._element_flux(z, energies, time_interval)

        # Compare results (very tight tolerance since these should be identical)
        np.testing.assert_allclose(
            flux_vectorized,
            flux_explicit,
            rtol=1e-14,
            atol=1e-16,
            err_msg=f"{name} flux mismatch for time_interval={time_interval}",
        )

    @pytest.mark.parametrize(
        "z,name",
        [
            (1, "Proton"),
            (8, "Oxygen"),  # Test both leading and sub-leading elements
        ],
    )
    @pytest.mark.parametrize(
        "time_interval",
        [
            "LIS",  # Local interstellar spectrum (fastest)
            None,  # Solar cycle average
        ],
    )
    def test_element_flux_jacobian_vectorized_vs_explicit(
        self, gsf_energy, z, name, time_interval
    ):
        """Test that vectorized Jacobian matches explicit implementation."""
        energies = np.logspace(1, 2, 8)  # Fewer points for Jacobian test

        # Calculate using reference explicit method
        jac_explicit = self._reference_element_flux_jacobian_explicit(
            gsf_energy, z, energies, time_interval
        )

        # Calculate using vectorized method
        jac_vectorized = gsf_energy._element_flux_jacobian(z, energies, time_interval)

        # Compare results (slightly looser tolerance for Jacobian due to numerical precision)
        np.testing.assert_allclose(
            jac_vectorized,
            jac_explicit,
            rtol=1e-12,
            atol=1e-14,
            err_msg=f"{name} Jacobian mismatch for time_interval={time_interval}",
        )

    @pytest.mark.parametrize("target", ["p", "He", "O", "Fe"])
    @pytest.mark.parametrize(
        "time_interval",
        [
            "LIS",
            None,
            (200901, 200903),  # Short period for speed
        ],
    )
    def test_model_flux_consistency(self, gsf_energy, target, time_interval):
        """Test that high-level model.flux gives consistent results."""
        energies = np.logspace(1, 3, 10)  # 10 GeV to 1 TeV

        # Calculate flux twice - should be identical
        flux1 = gsf_energy.flux(energies, target, time_interval=time_interval)
        flux2 = gsf_energy.flux(energies, target, time_interval=time_interval)

        # Results should be bit-for-bit identical
        np.testing.assert_array_equal(
            flux1,
            flux2,
            err_msg=f"{target} flux is not deterministic for time_interval={time_interval}",
        )

        # Flux should be positive (or zero) and finite
        assert np.all(flux1 >= 0), f"{target} flux has negative values"
        assert np.all(np.isfinite(flux1)), f"{target} flux has non-finite values"

    @pytest.mark.parametrize("target", ["p", "He"])
    def test_nucleon_flux_consistency(self, gsf_nucleon, target):
        """Test nucleon flux model consistency."""
        energies = np.logspace(1, 2, 8)  # Fewer points for speed

        # Calculate flux twice - should be identical
        flux1 = gsf_nucleon.flux(energies, target, time_interval="LIS")
        flux2 = gsf_nucleon.flux(energies, target, time_interval="LIS")

        # Results should be bit-for-bit identical
        np.testing.assert_array_equal(
            flux1, flux2, err_msg=f"Nucleon {target} flux is not deterministic"
        )

        # Flux should be positive (or zero) and finite
        assert np.all(flux1 >= 0), f"Nucleon {target} flux has negative values"
        assert np.all(np.isfinite(flux1)), (
            f"Nucleon {target} flux has non-finite values"
        )

    def test_performance_improvement(self, gsf_energy):
        """Test that vectorized implementation provides performance benefit."""
        import time

        z = 8  # Oxygen
        energies = np.logspace(1, 3, 100)  # More points to see timing difference
        time_interval = None  # Solar cycle average (many phi values)

        # Best-of-N timing: the minimum over repeats is robust against the
        # scheduling noise of shared CI runners, where a mean-based timing
        # flakes (observed on windows/macos runners).
        n_loops = 10

        explicit_times = []
        for _ in range(n_loops):
            start_time = time.perf_counter()
            flux_explicit = self._reference_element_flux_explicit(
                gsf_energy, z, energies, time_interval
            )
            explicit_times.append(time.perf_counter() - start_time)
        explicit_time = min(explicit_times)

        vectorized_times = []
        for _ in range(n_loops):
            start_time = time.perf_counter()
            flux_vectorized = gsf_energy._element_flux(z, energies, time_interval)
            vectorized_times.append(time.perf_counter() - start_time)
        vectorized_time = min(vectorized_times)

        # Results should be identical
        np.testing.assert_allclose(
            flux_vectorized, flux_explicit, rtol=1e-14, atol=1e-16
        )

        # Vectorized should be faster (though this is not a strict requirement for correctness)
        if vectorized_time > 0 and explicit_time > 0:
            speedup = explicit_time / vectorized_time
            print(
                f"Performance (best of {n_loops} loops): Explicit={explicit_time:.4f}s, "
                f"Vectorized={vectorized_time:.4f}s, Speedup={speedup:.1f}x"
            )

            # This is informational - we don't enforce a specific speedup
            # We just check that vectorized is not pathologically slower
            assert speedup > 0.5, (
                "Vectorized implementation is significantly slower than explicit"
            )

    def test_edge_cases(self, gsf_energy):
        """Test edge cases that might cause numerical issues."""
        z = 1  # Proton

        # Test very low energies (near rest mass)
        proton_mass = gsf_energy.z_to_a[z] * 0.93891872965  # GeV
        low_energies = np.array(
            [proton_mass + 1e-6, proton_mass + 1e-3, proton_mass + 0.1]
        )

        flux_explicit = self._reference_element_flux_explicit(
            gsf_energy, z, low_energies, "LIS"
        )
        flux_vectorized = gsf_energy._element_flux(z, low_energies, "LIS")

        np.testing.assert_allclose(
            flux_vectorized,
            flux_explicit,
            rtol=1e-12,
            atol=1e-16,
            err_msg="Edge case (low energy) mismatch",
        )

        # Test very high energies
        high_energies = np.array([1e6, 1e9, 1e12])  # Very high energies

        flux_explicit = self._reference_element_flux_explicit(
            gsf_energy, z, high_energies, "LIS"
        )
        flux_vectorized = gsf_energy._element_flux(z, high_energies, "LIS")

        np.testing.assert_allclose(
            flux_vectorized,
            flux_explicit,
            rtol=1e-12,
            atol=1e-16,
            err_msg="Edge case (high energy) mismatch",
        )

        # Test single energy point
        single_energy = np.array([100.0])

        flux_explicit = self._reference_element_flux_explicit(
            gsf_energy, z, single_energy, "LIS"
        )
        flux_vectorized = gsf_energy._element_flux(z, single_energy, "LIS")

        np.testing.assert_allclose(
            flux_vectorized,
            flux_explicit,
            rtol=1e-14,
            atol=1e-16,
            err_msg="Edge case (single energy) mismatch",
        )
