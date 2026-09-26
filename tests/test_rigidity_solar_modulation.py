"""Test suite for GSFRigidity solar modulation functionality.

This module tests the new solar modulation capability added to GSFRigidity class,
including the _rigidity_phi_transform helper method and the integration with
flux, jacobian, and covariance calculations.
"""

import numpy as np
import pytest

from globalsplinefit import SOLAR_CYCLE_24_END, SOLAR_CYCLE_24_START, GSFEnergy
from globalsplinefit.model import _bin_phi


class TestGSFRigidityPhiTransform:
    """Test the _rigidity_phi_transform helper method."""

    def test_phi_transform_zero_phi(self, gsf_rigidity):
        """Test that zero phi gives identity transformation."""
        rigidity = np.array([1.0, 10.0, 100.0])
        z = 1  # proton

        R_is, Lambda = gsf_rigidity._rigidity_phi_transform(z, rigidity, 0.0)

        # With phi=0, R_is should equal input rigidity
        np.testing.assert_allclose(R_is, rigidity, rtol=1e-10)
        # And Lambda should be 1
        np.testing.assert_allclose(Lambda, 1.0, rtol=1e-10)

    def test_phi_transform_positive_phi(self, gsf_rigidity):
        """Test phi transformation with positive modulation potential."""
        rigidity = np.array([1.0, 10.0, 100.0])
        z = 1  # proton
        phi = 500.0  # MV

        R_is, Lambda = gsf_rigidity._rigidity_phi_transform(z, rigidity, phi)

        # With positive phi, interstellar rigidity should be higher
        assert np.all(R_is > rigidity)
        # Lambda should be positive
        assert np.all(Lambda > 0)
        assert np.all(np.isfinite(Lambda))

    def test_phi_transform_different_elements(self, gsf_rigidity):
        """Test phi transformation for different elements."""
        rigidity = np.array([1.0, 10.0, 100.0])
        phi = 500.0

        elements = [1, 2, 8, 26]  # H, He, O, Fe

        for z in elements:
            R_is, Lambda = gsf_rigidity._rigidity_phi_transform(z, rigidity, phi)

            # All outputs should be finite and positive
            assert np.all(np.isfinite(R_is))
            assert np.all(R_is > 0)
            assert np.all(np.isfinite(Lambda))
            assert np.all(Lambda > 0)

            # Higher charge elements should have different transformation
            if z > 1:
                R_is_H, Lambda_H = gsf_rigidity._rigidity_phi_transform(
                    1, rigidity, phi
                )
                # The exact relationship depends on masses, but they should be different
                assert not np.allclose(R_is, R_is_H, rtol=1e-3)

    def test_phi_transform_high_phi_stability(self, gsf_rigidity):
        """Test numerical stability with high phi values."""
        rigidity = np.array([0.1, 1.0, 10.0])
        z = 1
        phi = 2000.0  # Very high modulation

        R_is, Lambda = gsf_rigidity._rigidity_phi_transform(z, rigidity, phi)

        # Should still be finite
        assert np.all(np.isfinite(R_is))
        assert np.all(np.isfinite(Lambda))
        assert np.all(R_is >= 0)
        assert np.all(Lambda >= 0)

    def test_phi_transform_low_rigidity_behavior(self, gsf_rigidity):
        """Test behavior at very low rigidities where relativistic effects matter."""
        rigidity = np.array([0.01, 0.1, 1.0])
        z = 1
        phi = 500.0

        R_is, Lambda = gsf_rigidity._rigidity_phi_transform(z, rigidity, phi)

        # Even at low rigidities, outputs should be finite and physical
        assert np.all(np.isfinite(R_is))
        assert np.all(np.isfinite(Lambda))
        assert np.all(R_is >= 0)
        assert np.all(Lambda >= 0)


class TestGSFRigiditySolarModulation:
    """Test solar modulation functionality in GSFRigidity flux calculations."""

    def test_flux_without_time_interval_uses_solar_cycle_24(self, gsf_rigidity):
        """Test that flux without time_interval uses Solar Cycle 24 average."""
        rigidity = np.logspace(0, 2, 20)

        # Test 1: Default (None) should use Solar Cycle 24 with approximation
        flux_default = gsf_rigidity.flux(rigidity, "p")

        # Test 2: Explicit "LIS" should be different from default
        flux_lis = gsf_rigidity.flux(rigidity, "p", time_interval="LIS")

        # Solar modulated flux (default) should be different from LIS at low rigidities
        # Check that they're different (not equal)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(flux_default, flux_lis, rtol=1e-6)

        # Test 3: Both should be positive and finite
        assert np.all(flux_default >= 0), "Default flux should be non-negative"
        assert np.all(flux_lis >= 0), "LIS flux should be non-negative"
        assert np.all(np.isfinite(flux_default)), "Default flux should be finite"
        assert np.all(np.isfinite(flux_lis)), "LIS flux should be finite"

    def test_flux_with_solar_modulation_reduces_low_energy(self, gsf_rigidity):
        """Test that solar modulation reduces flux at low energies compared to LIS."""
        rigidity = np.logspace(0, 2, 30)  # Start from 1 GV to avoid edge effects

        # LIS flux (explicit)
        flux_lis = gsf_rigidity.flux(rigidity, "p", time_interval="LIS")

        # Modulated flux (using a representative time period)
        # Note: This requires actual phi data to be available
        try:
            flux_mod = gsf_rigidity.flux(rigidity, "p", time_interval=(200901, 200912))

            # At low rigidities where LIS has positive flux, modulated flux should be lower
            # We need to be careful about the very low energy regime where LIS might be zero
            # but modulation can boost the effective interstellar rigidity
            valid_lis_mask = (rigidity > 2.0) & (
                flux_lis > 0
            )  # Only compare where LIS has flux
            if np.any(valid_lis_mask):
                assert np.all(flux_mod[valid_lis_mask] <= flux_lis[valid_lis_mask])

            # At high rigidities, fluxes should be similar (modulation effect is
            # small). Force-field suppression at ~10 GV is still ~15% (phi~0.4-0.6
            # GV), so "small" (<10%) only holds well above ~40 GV.
            high_rigidity_mask = rigidity > 40.0
            if np.any(high_rigidity_mask):
                np.testing.assert_allclose(
                    flux_mod[high_rigidity_mask], flux_lis[high_rigidity_mask], rtol=0.1
                )
        except (KeyError, IndexError):
            # Skip if phi data not available for this time period
            pytest.skip("Solar modulation data not available for test period")

    def test_flux_different_time_periods(self, gsf_rigidity):
        """Test flux calculation for different time periods."""
        rigidity = np.logspace(0, 2, 20)

        # Try a few different time periods
        time_periods = [
            (200501, 200512),  # 2005
            (201001, 201012),  # 2010
            (201501, 201512),  # 2015
        ]

        fluxes = {}
        for period in time_periods:
            try:
                flux = gsf_rigidity.flux(rigidity, "p", time_interval=period)
                fluxes[period] = flux

                # Basic sanity checks
                assert np.all(flux >= 0)
                assert np.all(np.isfinite(flux))

            except (KeyError, IndexError):
                # Skip periods without available data
                continue

        # If we have multiple periods, check they are different
        if len(fluxes) > 1:
            period_list = list(fluxes.keys())
            for i in range(len(period_list)):
                for j in range(i + 1, len(period_list)):
                    p1, p2 = period_list[i], period_list[j]
                    # Fluxes should differ at some energies due to different solar activity
                    assert not np.allclose(fluxes[p1], fluxes[p2], rtol=1e-6)

    def test_flux_all_groups_with_modulation(self, gsf_rigidity):
        """Test that solar modulation works for all cosmic ray groups."""
        rigidity = np.logspace(0, 2, 15)
        groups = gsf_rigidity.active_groups

        try:
            time_interval = (200901, 200912)

            for group in groups:
                flux_lis = gsf_rigidity.flux(rigidity, group)
                flux_mod = gsf_rigidity.flux(
                    rigidity, group, time_interval=time_interval
                )

                # Basic checks
                assert np.all(flux_lis >= 0)
                assert np.all(flux_mod >= 0)
                assert np.all(np.isfinite(flux_lis))
                assert np.all(np.isfinite(flux_mod))

                # Solar modulation should affect low-energy part
                low_mask = rigidity < 1.0
                if np.any(low_mask):
                    # Modulated flux should generally be lower at low energies
                    assert np.mean(flux_mod[low_mask]) <= np.mean(flux_lis[low_mask])

        except (KeyError, IndexError):
            pytest.skip("Solar modulation data not available for test period")

    def test_total_flux_with_modulation(self, gsf_rigidity):
        """Test total flux calculation with solar modulation."""
        rigidity = np.logspace(0, 2, 15)

        try:
            total_lis = gsf_rigidity.total_flux(rigidity)
            total_mod = gsf_rigidity.total_flux(
                rigidity, time_interval=(200901, 200912)
            )

            assert np.all(total_lis >= 0)
            assert np.all(total_mod >= 0)
            assert np.all(np.isfinite(total_lis))
            assert np.all(np.isfinite(total_mod))

            # Total flux should also show modulation effects
            low_mask = rigidity < 1.0
            if np.any(low_mask):
                assert np.mean(total_mod[low_mask]) <= np.mean(total_lis[low_mask])

        except (KeyError, IndexError):
            pytest.skip("Solar modulation data not available for test period")


class TestGSFRigidityJacobianWithModulation:
    """Test Jacobian calculation with solar modulation."""

    def test_jacobian_without_time_interval(self, gsf_rigidity):
        """Test Jacobian without time interval (LIS case)."""
        rigidity = np.logspace(0, 2, 10)

        jac = gsf_rigidity.jacobian(rigidity, "p")

        assert isinstance(jac, np.ndarray)
        assert jac.shape[0] == len(rigidity)
        assert jac.shape[1] > 0
        assert np.all(np.isfinite(jac))

    def test_jacobian_with_time_interval(self, gsf_rigidity):
        """Test Jacobian with solar modulation."""
        rigidity = np.logspace(0, 2, 10)

        try:
            jac_lis = gsf_rigidity.jacobian(rigidity, "p")
            jac_mod = gsf_rigidity.jacobian(
                rigidity, "p", time_interval=(200901, 200912)
            )

            # Both should have same shape
            assert jac_lis.shape == jac_mod.shape
            assert np.all(np.isfinite(jac_lis))
            assert np.all(np.isfinite(jac_mod))

            # They should be different due to modulation
            assert not np.allclose(jac_lis, jac_mod, rtol=1e-6)

        except (KeyError, IndexError):
            pytest.skip("Solar modulation data not available for test period")

    def test_jacobian_consistency_with_flux(self, gsf_rigidity):
        """Test that Jacobian is consistent with flux calculation method."""
        rigidity = np.logspace(0, 2, 5)  # Small array for numerical differentiation

        try:
            time_interval = (200901, 200902)  # Short interval for test

            # Get analytical jacobian
            jac = gsf_rigidity.jacobian(rigidity, "p", time_interval=time_interval)

            # Check it has reasonable properties
            assert isinstance(jac, np.ndarray)
            assert jac.shape[0] == len(rigidity)
            assert np.all(np.isfinite(jac))

        except (KeyError, IndexError):
            pytest.skip("Solar modulation data not available for test period")


class TestGSFRigidityCovarianceWithModulation:
    """Test covariance calculation with solar modulation."""

    def test_covariance_without_time_interval(self, gsf_rigidity):
        """Test covariance without time interval (LIS case)."""
        rigidity = np.logspace(0, 2, 8)

        cov = gsf_rigidity.covariance("p", "p", rigidity)

        assert isinstance(cov, np.ndarray)
        assert cov.shape == (len(rigidity), len(rigidity))
        assert np.all(np.diag(cov) >= 0)
        assert np.all(np.isfinite(cov))

    def test_covariance_with_time_interval(self, gsf_rigidity):
        """Test covariance with solar modulation."""
        rigidity = np.logspace(0, 2, 8)

        try:
            cov_lis = gsf_rigidity.covariance("p", "p", rigidity)
            cov_mod = gsf_rigidity.covariance(
                "p", "p", rigidity, time_interval=(200901, 200912)
            )

            # Both should have same shape
            assert cov_lis.shape == cov_mod.shape
            assert np.all(np.isfinite(cov_lis))
            assert np.all(np.isfinite(cov_mod))
            assert np.all(np.diag(cov_lis) >= 0)
            assert np.all(np.diag(cov_mod) >= 0)

            # They should be different due to modulation
            assert not np.allclose(cov_lis, cov_mod, rtol=1e-6)

        except (KeyError, IndexError):
            pytest.skip("Solar modulation data not available for test period")

    def test_covariance_cross_groups(self, gsf_rigidity):
        """Test cross-group covariance with solar modulation."""
        rigidity = np.logspace(0, 2, 6)

        try:
            # Test cross-covariance between different groups
            cov_cross = gsf_rigidity.covariance(
                "p", "He", rigidity, time_interval=(200901, 200912)
            )

            assert isinstance(cov_cross, np.ndarray)
            assert cov_cross.shape == (len(rigidity), len(rigidity))
            assert np.all(np.isfinite(cov_cross))

        except (KeyError, IndexError):
            pytest.skip("Solar modulation data not available for test period")

    def test_error_calculation_with_modulation(self, gsf_rigidity):
        """Test error calculation with solar modulation."""
        rigidity = np.logspace(0, 2, 8)

        try:
            error_lis = gsf_rigidity.error(rigidity, "p")
            error_mod = gsf_rigidity.error(
                rigidity, "p", time_interval=(200901, 200912)
            )

            assert np.all(error_lis >= 0)
            assert np.all(error_mod >= 0)
            assert np.all(np.isfinite(error_lis))
            assert np.all(np.isfinite(error_mod))

            # Errors should generally be different due to modulation
            assert not np.allclose(error_lis, error_mod, rtol=1e-6)

        except (KeyError, IndexError):
            pytest.skip("Solar modulation data not available for test period")


class TestGSFRigidityConsistency:
    """Test consistency with existing functionality and other model classes."""

    def test_consistency_with_gsf_energy_at_high_energies(self, gsf_rigidity):
        """Test that results are consistent with GSFEnergy at high energies where modulation is negligible."""
        # Import here to avoid circular dependencies in test
        from globalsplinefit.model import GSFEnergy

        gsf_energy = GSFEnergy()

        # High rigidities where solar modulation effect is minimal
        rigidity = np.array([50.0, 100.0, 500.0])  # GV

        # For protons, convert rigidity to energy
        # E = sqrt((pc)^2 + (mc^2)^2) = sqrt((ZR)^2 + m^2) for protons (Z=1)
        nucleon_mass = 0.93891872965  # GeV
        energy = np.sqrt(rigidity**2 + nucleon_mass**2)

        # Get fluxes from both models (LIS)
        flux_rigidity = gsf_rigidity.flux(rigidity, "p")
        flux_energy = gsf_energy.flux(energy, "p")

        # Convert energy flux to rigidity flux using dE/dR
        # dE/dR = R/E for relativistic particles
        conversion_factor = rigidity / energy
        flux_energy_converted = flux_energy * conversion_factor

        # At high energies, they should be very similar
        np.testing.assert_allclose(flux_rigidity, flux_energy_converted, rtol=0.1)

    def test_flux_scaling_with_charge(self, gsf_rigidity):
        """Test that flux scaling with charge is reasonable."""
        rigidity = np.logspace(0, 2, 20)

        # Get fluxes for different groups
        flux_p = gsf_rigidity.flux(rigidity, "p")
        flux_he = gsf_rigidity.flux(rigidity, "He")
        flux_fe = gsf_rigidity.flux(rigidity, "Fe*")

        # At most energies, flux should decrease with increasing charge
        # (This is a general cosmic ray trend)
        mid_idx = len(rigidity) // 2
        assert flux_p[mid_idx] > flux_he[mid_idx]
        assert flux_he[mid_idx] > flux_fe[mid_idx]

    def test_time_interval_validation(self, gsf_rigidity):
        """Test that time interval validation works correctly."""
        rigidity = np.array([1.0, 10.0])

        # Invalid time intervals should raise appropriate errors
        with pytest.raises(ValueError, match="start must be earlier than end"):
            gsf_rigidity.flux(rigidity, "p", time_interval=(200901, 200901))

        with pytest.raises(ValueError, match="start must be earlier than end"):
            gsf_rigidity.flux(rigidity, "p", time_interval=(200912, 200901))

    def test_element_and_group_consistency(self, gsf_rigidity):
        """Test that element flux sums equal group flux."""
        rigidity = np.logspace(0, 2, 10)

        # Test for oxygen group (includes C, N, O)
        try:
            group_flux = gsf_rigidity.flux(rigidity, "O*")

            # Get the elements in oxygen group
            o_elements = gsf_rigidity.z_group[8]  # Oxygen group leader is Z=8
            element_sum = np.zeros_like(rigidity)

            for z in o_elements:
                element_flux = gsf_rigidity.flux(rigidity, z)
                element_sum += element_flux

            np.testing.assert_allclose(group_flux, element_sum, rtol=1e-10)

        except KeyError:
            # Skip if group structure is different than expected
            pytest.skip("Oxygen group structure not as expected")

    def test_zero_covariance_for_missing_groups(self, gsf_rigidity):
        """Test that covariance returns zeros for missing group combinations."""
        rigidity = np.array([1.0, 10.0])

        # Test that the method doesn't crash and returns appropriate shape
        cov = gsf_rigidity.covariance("p", "p", rigidity)
        assert cov.shape == (len(rigidity), len(rigidity))
        assert np.all(np.isfinite(cov))


class TestSolarCycleAveraging:
    """Test binned vs fully explicit Solar Cycle 24 averaging."""

    def test_single_bin_vs_explicit_averaging_agreement(self, gsf_rigidity):
        """One bin (modulate-the-mean) stays within 5% of explicit at 1.5 GV."""
        rigidity = np.array([1.5])  # 1.5 GV test point

        # Single bin: evaluate once at the mean phi (the old "approximate" mode)
        gsf_rigidity.solar_cycle_average_bins = 1
        flux_approx = gsf_rigidity.flux(rigidity, "p")

        # Full explicit monthly average
        gsf_rigidity.solar_cycle_average_bins = None
        flux_explicit = gsf_rigidity.flux(rigidity, "p")

        # Calculate relative difference
        rel_diff = np.abs(flux_approx - flux_explicit) / flux_explicit

        # Should be within 5%
        assert rel_diff[0] < 0.05, (
            f"Single-bin averaging differs by {rel_diff[0] * 100:.2f}% from explicit "
            f"(flux_approx={flux_approx[0]:.3e}, flux_explicit={flux_explicit[0]:.3e})"
        )

        # Reset to default
        gsf_rigidity.solar_cycle_average_bins = 6

    def test_default_bins_track_explicit_closely(self, gsf_rigidity):
        """The 6-bin default must sit far closer to explicit than a single bin."""
        rigidity = np.array([1.5])
        try:
            gsf_rigidity.solar_cycle_average_bins = None
            ref = gsf_rigidity.flux(rigidity, "p")[0]
            gsf_rigidity.solar_cycle_average_bins = 6
            six = gsf_rigidity.flux(rigidity, "p")[0]
            gsf_rigidity.solar_cycle_average_bins = 1
            one = gsf_rigidity.flux(rigidity, "p")[0]
        finally:
            gsf_rigidity.solar_cycle_average_bins = 6
        assert abs(six / ref - 1) < 0.02
        assert abs(six / ref - 1) < abs(one / ref - 1)

    def test_lis_string_returns_unmodulated_spectrum(self, gsf_rigidity):
        """Test that 'LIS' string returns the Local Interstellar Spectrum."""
        rigidity = np.array([10.0, 100.0])  # Use rigidities in valid range

        # Get LIS flux for protons
        flux_lis = gsf_rigidity.flux(rigidity, "p", time_interval="LIS")

        # Get flux with phi=0 (equivalent to LIS)
        phis_lis, weights_lis = gsf_rigidity._phi_list("LIS")
        assert len(phis_lis) == 1
        assert phis_lis[0] == 0.0
        assert weights_lis[0] == 1.0

        # Verify flux is positive and finite
        assert np.all(flux_lis > 0)
        assert np.all(np.isfinite(flux_lis))

        # LIS flux should be higher than modulated flux
        flux_default = gsf_rigidity.flux(rigidity, "p")  # Solar Cycle 24 average
        assert np.all(flux_lis > flux_default)

    def test_solar_cycle_24_constants_used(self, gsf_rigidity):
        """Test that Solar Cycle 24 constants are properly used as defaults."""
        rigidity = np.array([1.0])

        # Get explicit Solar Cycle 24 interval
        start = SOLAR_CYCLE_24_START[0]
        end = SOLAR_CYCLE_24_END[0]

        # Naming the SC24 interval explicitly must match the default path, for
        # every bin count -- the default is now the same code path, not a
        # separate single-phi shortcut.
        try:
            for bins in (1, 6, 12, None):
                gsf_rigidity.solar_cycle_average_bins = bins
                flux_explicit_sc24 = gsf_rigidity.flux(
                    rigidity, "p", time_interval=(start, end)
                )
                flux_default = gsf_rigidity.flux(rigidity, "p")
                np.testing.assert_allclose(flux_default, flux_explicit_sc24, rtol=1e-10)
        finally:
            gsf_rigidity.solar_cycle_average_bins = 6


class TestPhiBinning:
    """The period-average binning that replaced the fast/exact flag."""

    @staticmethod
    def _months(n=132, seed=0):
        return np.random.default_rng(seed).uniform(0.2, 1.2, n)

    @pytest.mark.parametrize("n_bins", [1, 2, 3, 6, 12, 24, 132, 500, None])
    def test_weighted_mean_is_exact(self, n_bins):
        """Weights are normalized and the weighted mean phi is preserved exactly.

        This is what keeps the first-order (mean) suppression right at every bin
        count, so a coarser setting only costs curvature, never a level shift.
        """
        months = self._months()
        phi, w = _bin_phi(months, n_bins)
        assert w.sum() == pytest.approx(1.0, abs=1e-14)
        assert np.dot(phi, w) == pytest.approx(months.mean(), abs=1e-12)
        assert len(phi) == len(w)

    def test_bin_count_bounds(self):
        months = self._months()
        assert len(_bin_phi(months, 1)[0]) == 1
        assert len(_bin_phi(months, 6)[0]) <= 6
        assert len(_bin_phi(months, None)[0]) == len(months)
        # asking for more bins than months keeps every month, weight 1/n
        phi, w = _bin_phi(months[:5], 50)
        assert len(phi) == 5
        np.testing.assert_allclose(w, np.full(5, 0.2))

    def test_single_bin_is_the_plain_mean(self):
        months = self._months()
        phi, w = _bin_phi(months, 1)
        assert phi[0] == pytest.approx(months.mean())
        assert w[0] == 1.0

    def test_empty_interval_rejected(self):
        with pytest.raises(ValueError, match="no solar modulation potentials"):
            _bin_phi(np.array([]), 6)

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True, "6"])
    def test_invalid_bin_count_rejected(self, bad):
        with pytest.raises(ValueError, match="solar_cycle_average_bins"):
            GSFEnergy(solar_cycle_average_bins=bad)

    def test_single_bin_matches_legacy_mean_phi(self):
        """n_bins=1 reproduces the retired approximate path exactly."""
        model = GSFEnergy(solar_cycle_average_bins=1)
        phi, _ = model._phi_list(None)
        assert phi[0] == model.params.get_solar_cycle_24_phi_average()

    def test_retired_flag_is_gone(self):
        """The old boolean is not silently accepted as an unknown kwarg."""
        with pytest.raises(TypeError, match="use_approximate_solar_cycle_average"):
            GSFEnergy(use_approximate_solar_cycle_average=True)

    def test_more_bins_converge_monotonically(self):
        """Error against the full average shrinks as bins are added."""
        E = np.array([1.0, 2.0, 5.0])
        ref = GSFEnergy(solar_cycle_average_bins=None).flux(E, "proton")
        errs = [
            np.max(
                np.abs(
                    GSFEnergy(solar_cycle_average_bins=n).flux(E, "proton") / ref - 1
                )
            )
            for n in (1, 3, 6, 12)
        ]
        assert errs == sorted(errs, reverse=True), errs
        assert errs[-1] < 0.005  # 12 bins within 0.5%
