"""Test suite for default_time_interval parameter functionality.

This module tests that the default_time_interval parameter works correctly
across all GSF model classes and that it can be properly overridden when needed.
"""

import numpy as np
import pytest

from globalsplinefit import (
    GSFEnergy,
    GSFEnergyPerNucleon,
    GSFKineticEnergy,
    GSFKineticEnergyPerNucleon,
    GSFRigidity,
)


class TestDefaultTimeInterval:
    """Test default_time_interval parameter across all model classes."""

    @pytest.fixture(
        params=[
            (GSFEnergy, "energy"),
            (GSFKineticEnergy, "kinetic_energy"),
            (GSFRigidity, "rigidity"),
            (GSFEnergyPerNucleon, "energy_per_nucleon"),
        ]
    )
    def model_class(self, request):
        """Provide all non-kinetic nucleon model classes for testing."""
        return request.param

    def test_default_initialization(self):
        """Test that default initialization sets default_time_interval to None."""
        model = GSFEnergy()
        assert model.default_time_interval is None, (
            "Default time interval should be None for Solar Cycle 24 average"
        )
        assert model.solar_cycle_average_bins == 6, (
            "Default should period-average the solar cycle over 6 phi bins"
        )

    def test_custom_default_time_interval_tuple(self, model_class):
        """Test setting custom default_time_interval with tuple."""
        ModelClass, _ = model_class
        custom_interval = (200901, 200912)
        model = ModelClass(default_time_interval=custom_interval)
        assert model.default_time_interval == custom_interval, (
            f"Custom default should be {custom_interval}"
        )

    def test_custom_default_time_interval_lis(self, model_class):
        """Test setting custom default_time_interval to LIS."""
        ModelClass, _ = model_class
        model = ModelClass(default_time_interval="LIS")
        assert model.default_time_interval == "LIS", "LIS default should be set"

    def test_custom_default_time_interval_none(self, model_class):
        """Test explicitly setting default_time_interval to None."""
        ModelClass, _ = model_class
        model = ModelClass(default_time_interval=None)
        assert model.default_time_interval is None, "None default should be set"

    def test_flux_uses_default(self, model_class):
        """Test that flux calculation uses default_time_interval when not specified."""
        ModelClass, energy_name = model_class
        energy = np.array([10.0, 100.0])
        custom_interval = (200901, 200912)

        # Model with custom default
        model_default = ModelClass(default_time_interval=custom_interval)
        flux_default = model_default.flux(energy, "p")

        # Model with explicit time interval in method call
        model_explicit = ModelClass()
        flux_explicit = model_explicit.flux(energy, "p", time_interval=custom_interval)

        np.testing.assert_allclose(
            flux_default,
            flux_explicit,
            rtol=1e-14,
            err_msg=f"Flux should match when using default vs explicit for {ModelClass.__name__}",
        )

    def test_explicit_time_interval_overrides_default(self, model_class):
        """Test that explicit time_interval parameter overrides default."""
        ModelClass, _ = model_class
        energy = np.array([10.0, 100.0])
        custom_interval = (200901, 200912)

        model = ModelClass(default_time_interval=custom_interval)
        flux_default = model.flux(energy, "p")  # Uses default
        flux_override = model.flux(
            energy, "p", time_interval="LIS"
        )  # Overrides with LIS

        # These should be different (modulated vs LIS)
        assert not np.allclose(flux_default, flux_override, rtol=1e-10), (
            f"Flux should differ when overriding default for {ModelClass.__name__}"
        )

    def test_total_flux_uses_default(self, model_class):
        """Test that total_flux uses default_time_interval."""
        ModelClass, _ = model_class
        energy = np.array([10.0, 100.0])
        custom_interval = (200901, 200912)

        model_default = ModelClass(default_time_interval=custom_interval)
        total_default = model_default.total_flux(energy)

        model_explicit = ModelClass()
        total_explicit = model_explicit.total_flux(
            energy, time_interval=custom_interval
        )

        np.testing.assert_allclose(
            total_default,
            total_explicit,
            rtol=1e-14,
            err_msg=f"total_flux should use default for {ModelClass.__name__}",
        )

    def test_error_uses_default(self, model_class):
        """Test that error calculation uses default_time_interval."""
        ModelClass, _ = model_class
        energy = np.array([10.0, 100.0])
        custom_interval = (200901, 200912)

        model_default = ModelClass(default_time_interval=custom_interval)
        error_default = model_default.error(energy, "p")

        model_explicit = ModelClass()
        error_explicit = model_explicit.error(
            energy, "p", time_interval=custom_interval
        )

        np.testing.assert_allclose(
            error_default,
            error_explicit,
            rtol=1e-14,
            err_msg=f"error should use default for {ModelClass.__name__}",
        )

    def test_total_error_uses_default(self, model_class):
        """Test that total_error uses default_time_interval."""
        ModelClass, _ = model_class
        energy = np.array([10.0, 100.0])
        custom_interval = (200901, 200912)

        model_default = ModelClass(default_time_interval=custom_interval)
        error_default = model_default.total_error(energy)

        model_explicit = ModelClass()
        error_explicit = model_explicit.total_error(
            energy, time_interval=custom_interval
        )

        np.testing.assert_allclose(
            error_default,
            error_explicit,
            rtol=1e-14,
            err_msg=f"total_error should use default for {ModelClass.__name__}",
        )

    def test_covariance_uses_default(self, model_class):
        """Test that covariance calculation uses default_time_interval."""
        ModelClass, _ = model_class
        energy = np.array([10.0, 100.0])
        custom_interval = (200901, 200912)

        model_default = ModelClass(default_time_interval=custom_interval)
        cov_default = model_default.covariance("p", "He", energy)

        model_explicit = ModelClass()
        cov_explicit = model_explicit.covariance(
            "p", "He", energy, time_interval=custom_interval
        )

        np.testing.assert_allclose(
            cov_default,
            cov_explicit,
            rtol=1e-14,
            err_msg=f"covariance should use default for {ModelClass.__name__}",
        )


class TestGSFKineticEnergyPerNucleonDefaultTimeInterval:
    """Test default_time_interval for GSFKineticEnergyPerNucleon specifically.

    This class has additional methods (p_and_n_flux, p_and_n_total_flux, etc.)
    that also need to respect the default_time_interval parameter.
    """

    @pytest.fixture
    def model(self):
        """Create a model with custom default_time_interval."""
        return GSFKineticEnergyPerNucleon(default_time_interval=(200901, 200912))

    @pytest.fixture
    def model_explicit(self):
        """Create a model without custom default for comparison."""
        return GSFKineticEnergyPerNucleon()

    @pytest.fixture
    def energy(self):
        """Sample energy values for testing."""
        return np.array([10.0, 100.0])

    def test_initialization(self):
        """Test initialization with default_time_interval."""
        model = GSFKineticEnergyPerNucleon(default_time_interval=(200901, 200912))
        assert model.default_time_interval == (200901, 200912), (
            "Custom default should be set"
        )

    def test_p_and_n_flux_uses_default(self, model, model_explicit, energy):
        """Test that p_and_n_flux uses default_time_interval."""
        flux_default = model.p_and_n_flux(energy, "He")
        flux_explicit = model_explicit.p_and_n_flux(
            energy, "He", time_interval=(200901, 200912)
        )

        np.testing.assert_allclose(
            flux_default,
            flux_explicit,
            rtol=1e-14,
            err_msg="p_and_n_flux should use default",
        )
        assert flux_default.shape == (2, len(energy)), (
            "p_and_n_flux should return (2, N) array"
        )

    def test_flux_uses_default(self, model, model_explicit, energy):
        """Test that flux uses default_time_interval."""
        flux_default = model.flux(energy, "He")
        flux_explicit = model_explicit.flux(
            energy, "He", time_interval=(200901, 200912)
        )

        np.testing.assert_allclose(
            flux_default, flux_explicit, rtol=1e-14, err_msg="flux should use default"
        )

    def test_total_flux_uses_default(self, model, model_explicit, energy):
        """Test that total_flux uses default_time_interval."""
        total_default = model.total_flux(energy)
        total_explicit = model_explicit.total_flux(
            energy, time_interval=(200901, 200912)
        )

        np.testing.assert_allclose(
            total_default,
            total_explicit,
            rtol=1e-14,
            err_msg="total_flux should use default",
        )

    def test_p_and_n_total_flux_uses_default(self, model, model_explicit, energy):
        """Test that p_and_n_total_flux uses default_time_interval."""
        pn_total_default = model.p_and_n_total_flux(energy)
        pn_total_explicit = model_explicit.p_and_n_total_flux(
            energy, time_interval=(200901, 200912)
        )

        np.testing.assert_allclose(
            pn_total_default,
            pn_total_explicit,
            rtol=1e-14,
            err_msg="p_and_n_total_flux should use default",
        )
        assert pn_total_default.shape == (2, len(energy)), (
            "p_and_n_total_flux should return (2, N) array"
        )

    def test_total_error_uses_default(self, model, model_explicit, energy):
        """Test that total_error uses default_time_interval."""
        error_default = model.total_error(energy)
        error_explicit = model_explicit.total_error(
            energy, time_interval=(200901, 200912)
        )

        np.testing.assert_allclose(
            error_default,
            error_explicit,
            rtol=1e-14,
            err_msg="total_error should use default",
        )

    def test_p_and_n_error_uses_default(self, model, model_explicit, energy):
        """Test that p_and_n_error uses default_time_interval."""
        pn_error_default = model.p_and_n_error(energy, "He")
        pn_error_explicit = model_explicit.p_and_n_error(
            energy, "He", time_interval=(200901, 200912)
        )

        np.testing.assert_allclose(
            pn_error_default,
            pn_error_explicit,
            rtol=1e-14,
            err_msg="p_and_n_error should use default",
        )
        assert pn_error_default.shape == (2, len(energy)), (
            "p_and_n_error should return (2, N) array"
        )

    def test_error_uses_default(self, model, model_explicit, energy):
        """Test that error uses default_time_interval."""
        err_default = model.error(energy, "He")
        err_explicit = model_explicit.error(
            energy, "He", time_interval=(200901, 200912)
        )

        np.testing.assert_allclose(
            err_default, err_explicit, rtol=1e-14, err_msg="error should use default"
        )

    def test_can_override_default(self, model, energy):
        """Test that explicit time_interval can override default."""
        flux_lis = model.flux(energy, "He", time_interval="LIS")
        flux_default = model.flux(energy, "He")

        assert not np.allclose(flux_lis, flux_default, rtol=1e-10), (
            "Flux should differ when overriding default"
        )

    def test_all_methods_work_with_lis_default(self, energy):
        """Test that all methods work when default_time_interval is LIS."""
        model_lis = GSFKineticEnergyPerNucleon(default_time_interval="LIS")

        # These should all execute without error
        flux = model_lis.flux(energy, "p")
        pn_flux = model_lis.p_and_n_flux(energy, "p")
        total = model_lis.total_flux(energy)
        pn_total = model_lis.p_and_n_total_flux(energy)
        error = model_lis.error(energy, "p")
        pn_error = model_lis.p_and_n_error(energy, "p")
        total_err = model_lis.total_error(energy)

        # Basic sanity checks
        assert flux.shape == energy.shape
        assert pn_flux.shape == (2, len(energy))
        assert total.shape == energy.shape
        assert pn_total.shape == (2, len(energy))
        assert error.shape == energy.shape
        assert pn_error.shape == (2, len(energy))
        assert total_err.shape == energy.shape

    def test_consistency_across_different_defaults(self, energy):
        """Test that results are consistent regardless of how default is set."""
        # Three equivalent ways to get LIS flux
        model1 = GSFKineticEnergyPerNucleon(default_time_interval="LIS")
        flux1 = model1.flux(energy, "p")

        model2 = GSFKineticEnergyPerNucleon()
        flux2 = model2.flux(energy, "p", time_interval="LIS")

        model3 = GSFKineticEnergyPerNucleon(default_time_interval=(200901, 200912))
        flux3 = model3.flux(energy, "p", time_interval="LIS")

        np.testing.assert_allclose(flux1, flux2, rtol=1e-14)
        np.testing.assert_allclose(flux2, flux3, rtol=1e-14)


class TestDefaultTimeIntervalEdgeCases:
    """Test edge cases and special scenarios for default_time_interval."""

    def test_none_default_uses_solar_cycle_24(self):
        """Test that None default uses Solar Cycle 24 average."""
        energy = np.array([10.0, 100.0])

        # Default (None) should give Solar Cycle 24 average
        model_none = GSFEnergy(default_time_interval=None)
        flux_none = model_none.flux(energy, "p")

        # Explicitly passing None should do the same
        model_explicit = GSFEnergy()
        flux_explicit = model_explicit.flux(energy, "p", time_interval=None)

        np.testing.assert_allclose(flux_none, flux_explicit, rtol=1e-14)

    def test_different_defaults_for_different_models(self):
        """Test that different model instances can have different defaults."""
        energy = np.array([10.0])

        model1 = GSFEnergy(default_time_interval="LIS")
        model2 = GSFEnergy(default_time_interval=(200901, 200912))
        model3 = GSFEnergy(default_time_interval=None)

        flux1 = model1.flux(energy, "p")
        flux2 = model2.flux(energy, "p")
        flux3 = model3.flux(energy, "p")

        # All three should be different
        assert not np.allclose(flux1, flux2, rtol=1e-10)
        assert not np.allclose(flux1, flux3, rtol=1e-10)
        assert not np.allclose(flux2, flux3, rtol=1e-10)

    def test_default_persists_across_multiple_calls(self):
        """Test that default_time_interval persists across multiple method calls."""
        energy = np.array([10.0, 100.0])
        custom_interval = (200901, 200912)
        model = GSFEnergy(default_time_interval=custom_interval)

        # Multiple calls should all use the same default
        flux1 = model.flux(energy, "p")
        flux2 = model.flux(energy, "p")
        total1 = model.total_flux(energy)
        total2 = model.total_flux(energy)

        np.testing.assert_array_equal(flux1, flux2)
        np.testing.assert_array_equal(total1, total2)
