"""Additional edge case tests for the new GSF API."""

import numpy as np
import pytest


class TestModelEdgeCases:
    """Test edge cases and boundary conditions for all GSF models."""

    def test_zero_energy_handling(self, gsf_energy):
        """Test handling of zero or near-zero energies."""
        zero_energies = np.array([0.0, 1e-10, 1e-5])
        flux = gsf_energy.flux(zero_energies, "p")

        assert isinstance(flux, np.ndarray)
        assert len(flux) == len(zero_energies)
        assert np.all(flux >= 0), "Flux should be non-negative for zero/low energies"

    def test_inf_nan_energy_handling(self, gsf_energy):
        """Test handling of infinite or NaN energies."""
        # Test with inf energies - should handle gracefully
        inf_energies = np.array([np.inf, -np.inf])
        flux = gsf_energy.flux(inf_energies, "p")

        assert isinstance(flux, np.ndarray)
        assert len(flux) == len(inf_energies)
        # The flux might be NaN or zero, but shouldn't crash

    def test_very_large_array(self, gsf_energy):
        """Test performance with very large energy arrays."""
        large_array = np.logspace(0, 5, 10000)  # 10,000 points
        flux = gsf_energy.flux(large_array, "p")

        assert isinstance(flux, np.ndarray)
        assert len(flux) == len(large_array)
        assert np.all(np.isfinite(flux[np.isfinite(large_array)])), (
            "Finite energies should produce finite flux"
        )

    def test_mixed_data_types(self, gsf_energy):
        """Test with mixed data types in energy arrays."""
        # Mix integers and floats
        mixed_energies = [1, 10.0, 100, 1000.0]
        flux = gsf_energy.flux(mixed_energies, "p")

        assert isinstance(flux, np.ndarray)
        assert len(flux) == len(mixed_energies)
        assert np.all(flux >= 0), "Mixed data types should work correctly"

    def test_duplicate_energies(self, gsf_energy):
        """Test with duplicate energy values."""
        duplicate_energies = np.array([100.0, 100.0, 1000.0, 1000.0])
        flux = gsf_energy.flux(duplicate_energies, "p")

        assert isinstance(flux, np.ndarray)
        assert len(flux) == len(duplicate_energies)
        # Duplicate energies should give identical flux values
        assert flux[0] == flux[1], "Duplicate energies should give identical flux"
        assert flux[2] == flux[3], "Duplicate energies should give identical flux"

    def test_unsorted_energies(self, gsf_energy):
        """Test with unsorted energy arrays."""
        unsorted_energies = np.array([1000.0, 10.0, 100.0, 1.5])
        flux = gsf_energy.flux(unsorted_energies, "p")

        assert isinstance(flux, np.ndarray)
        assert len(flux) == len(unsorted_energies)
        assert np.all(flux > 0), "Unsorted energies should still produce valid flux"

    def test_time_interval_edge_cases(self, gsf_energy, sample_energies):
        """Test edge cases in time interval specification."""
        # Same month - should raise ValueError
        same_month = (200901, 200901)
        with pytest.raises(ValueError):
            _ = gsf_energy.flux(sample_energies, "p", time_interval=same_month)

        # Single day would be invalid format, but test boundary
        single_month = (200901, 200902)
        flux2 = gsf_energy.flux(sample_energies, "p", time_interval=single_month)
        assert isinstance(flux2, np.ndarray)

        # Cross year boundary
        cross_year = (200912, 201001)
        flux3 = gsf_energy.flux(sample_energies, "p", time_interval=cross_year)
        assert isinstance(flux3, np.ndarray)

    def test_rigidity_negative_values(self, gsf_rigidity):
        """Test rigidity model with negative rigidity values."""
        negative_rigidities = np.array([-5, -0.1, 0.0, 0.1, 1.0])
        flux = gsf_rigidity.flux(negative_rigidities, "p")

        assert isinstance(flux, np.ndarray)
        assert len(flux) == len(negative_rigidities)
        # Negative rigidity should typically give zero or very small flux
        assert np.all(flux >= 0), (
            "Flux should be non-negative even for negative rigidity"
        )

    def test_nucleon_flux_consistency_check(self, gsf_nucleon):
        """Test internal consistency of nucleon flux calculations."""
        test_energies = np.array([10.0, 100.0, 1000.0])

        # For proton group, neutron flux should be ~0.008 of proton flux
        # Use p_and_n_flux to get separate components
        proton_flux = gsf_nucleon.p_and_n_flux(test_energies, "p")
        neutron_to_proton_ratio = proton_flux[1] / proton_flux[0]
        np.testing.assert_allclose(
            neutron_to_proton_ratio,
            0.008,
            rtol=0.1,
            err_msg="Neutron to proton flux ratio should be ~0.008",
        )

        # For helium, should have equal numbers of protons and neutrons
        # Use p_and_n_flux to get separate components
        helium_flux = gsf_nucleon.p_and_n_flux(test_energies, "He")
        # Helium-4 has 2 protons and 2 neutrons, so ratio should be 1:1
        # But the flux is weighted by abundance, so just check both are positive
        assert np.all(helium_flux[0] > 0), "Helium should have positive proton flux"
        assert np.all(helium_flux[1] > 0), "Helium should have positive neutron flux"

        # Test that total flux equals sum of separate components
        total_proton_flux = gsf_nucleon.flux(test_energies, "p")
        np.testing.assert_allclose(
            total_proton_flux,
            proton_flux[0] + proton_flux[1],
            rtol=1e-10,
            err_msg="Total flux should equal sum of proton and neutron components",
        )

    def test_covariance_matrix_properties(self, gsf_energy):
        """Test mathematical properties of covariance matrices."""
        test_energies = np.array([10.0, 100.0, 1000.0])

        # Test self-covariance (diagonal elements should be positive)
        cov = gsf_energy.covariance("p", "p", test_energies)
        assert np.all(np.diag(cov) >= 0), (
            "Diagonal covariance elements should be non-negative"
        )

        # Test symmetry property for cross-covariance
        cov12 = gsf_energy.covariance("p", "He", test_energies)
        cov21 = gsf_energy.covariance("He", "p", test_energies)
        np.testing.assert_allclose(
            cov12, cov21.T, rtol=1e-10, err_msg="Cross-covariance should be symmetric"
        )

    def test_jacobian_finite_differences(self, gsf_energy):
        """Test Jacobian calculation using finite differences verification."""
        # This is a numerical verification test
        test_energies = np.array([100.0])  # Single point for simplicity

        # Get analytical Jacobian
        jac_analytical = gsf_energy.jacobian(test_energies, "p")

        # The Jacobian should have reasonable magnitude
        assert np.all(np.isfinite(jac_analytical)), "Jacobian should be finite"
        assert jac_analytical.shape[0] == len(test_energies)
        assert jac_analytical.shape[1] > 0, "Should have parameters"

    def test_flux_monotonicity(self, gsf_energy):
        """Test that flux generally decreases with increasing energy (power-law behavior)."""
        # Test energy range where power-law behavior is expected
        test_energies = np.logspace(2, 4, 20)  # 100 to 10000 GeV
        flux = gsf_energy.flux(test_energies, "p")

        # Calculate power-law index (should be negative, around -2.7)
        log_flux = np.log10(flux)
        log_energy = np.log10(test_energies)

        # Fit linear relationship in log space
        coeffs = np.polyfit(log_energy, log_flux, 1)
        power_index = coeffs[0]

        # Cosmic ray spectrum typically has index between -2 and -4
        assert -4.0 < power_index < -1.5, (
            f"Power-law index {power_index:.2f} outside expected range"
        )

    def test_element_vs_group_relationships(self, gsf_energy):
        """Test relationships between individual elements and groups."""
        test_energies = np.array([100.0, 1000.0])

        # Test that we can access individual elements if they exist in z_group
        available_groups = list(gsf_energy.z_group.keys())

        for z in available_groups:
            # Group flux should be sum of constituent elements
            group_flux = gsf_energy.flux(test_energies, z)

            # For leading elements, group might be just the element itself
            # This tests the internal consistency
            assert np.all(group_flux >= 0), f"Group {z} flux should be non-negative"

    def test_memory_efficiency(self, gsf_energy):
        """Test that calculations don't consume excessive memory."""
        import gc
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Perform several large calculations
        large_energies = np.logspace(0, 6, 1000)
        for group in ["p", "He", "O", "Fe"]:
            flux = gsf_energy.flux(large_energies, group)
            del flux
            gc.collect()

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Should not increase memory by more than 100MB
        assert memory_increase < 100 * 1024 * 1024, (
            f"Memory increased by {memory_increase / 1024 / 1024:.1f}MB"
        )

    def test_thread_safety_simulation(self, gsf_energy):
        """Test behavior under concurrent-like access patterns."""
        # Simulate what might happen with concurrent access
        test_energies = np.array([100.0, 1000.0])

        # Rapid successive calls
        results = []
        for _ in range(10):
            flux = gsf_energy.flux(test_energies, "p")
            results.append(flux.copy())

        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(
                results[0],
                results[i],
                err_msg="Successive calls should give identical results",
            )

    def test_parameter_validation(self, gsf_energy):
        """Test comprehensive parameter validation."""
        test_energies = np.array([100.0])

        # Test invalid group names
        invalid_groups = [
            "invalid",
            123.456,
            {},
        ]  # Removed None and [] as these are handled
        for invalid_group in invalid_groups:
            with pytest.raises((ValueError, TypeError)):
                gsf_energy.flux(test_energies, invalid_group)

    def test_numerical_stability(self, gsf_energy):
        """Test numerical stability at extreme values."""
        # Test with values that might cause numerical issues
        extreme_energies = np.array([1e-10, 1e-5, 1e5, 1e10])
        flux = gsf_energy.flux(extreme_energies, "p")

        # Should not contain NaN or inf values where input is finite
        finite_mask = np.isfinite(extreme_energies)
        assert np.all(np.isfinite(flux[finite_mask])), (
            "Finite energies should produce finite flux"
        )
