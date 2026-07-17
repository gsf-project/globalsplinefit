"""Test suite for GSFKineticEnergyPerNucleon class."""

import numpy as np
import pytest

from globalsplinefit import GSFEnergyPerNucleon, GSFKineticEnergyPerNucleon


class TestGSFKineticEnergyPerNucleon:
    """Test the GSFKineticEnergyPerNucleon class for kinetic energy per nucleon calculations."""

    @pytest.fixture
    def model_kinetic(self):
        """Create GSFKineticEnergyPerNucleon model for testing."""
        return GSFKineticEnergyPerNucleon()

    @pytest.fixture
    def model_energy(self):
        """Create GSFEnergyPerNucleon model for comparison."""
        return GSFEnergyPerNucleon()

    def test_initialization(self, model_kinetic):
        """Test that GSFKineticEnergyPerNucleon initializes correctly."""
        assert hasattr(model_kinetic, "GROUP_NAMES")
        assert hasattr(model_kinetic, "z_group")
        assert hasattr(model_kinetic, "z_to_a")
        assert hasattr(model_kinetic, "phi")

        # Check that it's a subclass of GSFEnergyPerNucleon
        assert isinstance(model_kinetic, GSFEnergyPerNucleon)

    def test_kinetic_to_total_energy_conversion(self, model_kinetic):
        """Test the kinetic to total energy conversion is correct."""
        kinetic_energies = np.array([1.0, 10.0, 100.0])
        target = "p"

        total_energies = model_kinetic._transform_energy_per_nucleon(
            kinetic_energies, target
        )

        # Should add nucleon mass (0.93892 GeV)
        nucleon_mass = 0.93891872965
        expected_total_energies = kinetic_energies + nucleon_mass

        np.testing.assert_allclose(total_energies, expected_total_energies, rtol=1e-10)

    def test_consistency_with_energy_per_nucleon_model(
        self, model_kinetic, model_energy
    ):
        """Test that GSFKineticEnergyPerNucleon is consistent with GSFEnergyPerNucleon."""
        kinetic_energies = np.array([1.0, 10.0, 100.0])
        target = "p"

        # Get kinetic model flux
        flux_kinetic = model_kinetic.flux(kinetic_energies, target)

        # Convert to total energy per nucleon for comparison
        total_energies = model_kinetic._transform_energy_per_nucleon(
            kinetic_energies, target
        )
        flux_energy = model_energy.flux(total_energies, target)

        # Should be identical (within numerical precision)
        np.testing.assert_allclose(flux_kinetic, flux_energy, rtol=1e-12)

    def test_all_groups_work(self, model_kinetic):
        """Test that all cosmic ray groups work with kinetic energy per nucleon."""
        kinetic_energy = np.array([10.0])
        groups = ["p", "He", "O*", "Fe*"]

        for group in groups:
            flux = model_kinetic.flux(kinetic_energy, group)
            error = model_kinetic.error(kinetic_energy, group)
            jacobian = model_kinetic.jacobian(kinetic_energy, group)
            covariance = model_kinetic.covariance(group, group, kinetic_energy)

            # Basic sanity checks
            assert np.all(flux > 0), f"Flux must be positive for {group}"
            assert np.all(error > 0), f"Error must be positive for {group}"
            assert np.all(np.isfinite(flux)), f"Flux must be finite for {group}"
            assert np.all(np.isfinite(error)), f"Error must be finite for {group}"
            assert jacobian.shape[0] == len(kinetic_energy), (
                f"Jacobian shape incorrect for {group}"
            )
            assert covariance.shape == (len(kinetic_energy), len(kinetic_energy)), (
                f"Covariance shape incorrect for {group}"
            )

    def test_p_and_n_flux_method(self, model_kinetic):
        """Test the p_and_n_flux method specific to nucleon models."""
        kinetic_energy = np.array([10.0, 100.0])
        groups = ["p", "He", "O*"]

        for group in groups:
            p_and_n = model_kinetic.p_and_n_flux(kinetic_energy, group)
            total_flux = model_kinetic.flux(kinetic_energy, group)

            # p_and_n should be shape (2, N)
            assert p_and_n.shape == (2, len(kinetic_energy)), (
                f"p_and_n shape incorrect for {group}"
            )

            # Sum of proton and neutron flux should equal total flux
            sum_flux = p_and_n[0] + p_and_n[1]
            np.testing.assert_allclose(sum_flux, total_flux, rtol=1e-10)

            # Both proton and neutron fluxes should be non-negative
            assert np.all(p_and_n >= 0), (
                f"p_and_n flux should be non-negative for {group}"
            )

    def test_p_and_n_error_method(self, model_kinetic):
        """Test the p_and_n_error method."""
        kinetic_energy = np.array([10.0])
        target = "He"

        p_and_n_error = model_kinetic.p_and_n_error(kinetic_energy, target)

        # Should be shape (2, N)
        assert p_and_n_error.shape == (2, len(kinetic_energy))

        # Errors should be positive
        assert np.all(p_and_n_error > 0)

    def test_total_flux_and_error(self, model_kinetic):
        """Test total flux and error calculations."""
        kinetic_energy = np.logspace(0, 2, 10)

        total_flux = model_kinetic.total_flux(kinetic_energy)
        total_error = model_kinetic.total_error(kinetic_energy)

        # Calculate sum of individual groups for comparison
        individual_sum = np.zeros_like(kinetic_energy)
        for group in ["p", "He", "O*", "Fe*"]:
            individual_sum += model_kinetic.flux(kinetic_energy, group)

        # Total flux should equal sum of individual groups
        np.testing.assert_allclose(total_flux, individual_sum, rtol=1e-12)

        # Total error should be positive and finite
        assert np.all(total_error > 0)
        assert np.all(np.isfinite(total_error))

    def test_time_intervals(self, model_kinetic):
        """Test different time intervals work correctly."""
        kinetic_energy = np.array([10.0])
        target = "p"

        # Test LIS
        flux_lis = model_kinetic.flux(kinetic_energy, target, time_interval="LIS")

        # Test specific time interval
        flux_2009 = model_kinetic.flux(
            kinetic_energy, target, time_interval=(200901, 200912)
        )

        # Test default (solar cycle average)
        flux_default = model_kinetic.flux(kinetic_energy, target)

        # LIS should be higher than modulated flux
        assert flux_lis[0] > flux_2009[0], (
            "LIS flux should be higher than modulated flux"
        )
        assert flux_lis[0] > flux_default[0], (
            "LIS flux should be higher than solar cycle average"
        )

        # All should be positive and finite
        assert np.all(flux_lis > 0)
        assert np.all(flux_2009 > 0)
        assert np.all(flux_default > 0)
        assert np.all(np.isfinite(flux_lis))
        assert np.all(np.isfinite(flux_2009))
        assert np.all(np.isfinite(flux_default))

    def test_energy_conversion_consistency_across_groups(
        self, model_kinetic, model_energy
    ):
        """Test that energy conversion is consistent across all groups."""
        kinetic_energy = np.array([10.0])

        for group in ["p", "He", "O*", "Fe*"]:
            # Get flux from kinetic model
            flux_kinetic = model_kinetic.flux(kinetic_energy, group)

            # Convert using the model's own conversion method
            total_energy = model_kinetic._transform_energy_per_nucleon(
                kinetic_energy, group
            )
            flux_energy = model_energy.flux(total_energy, group)

            # Should be identical
            np.testing.assert_allclose(
                flux_kinetic,
                flux_energy,
                rtol=1e-12,
                err_msg=f"Energy conversion inconsistent for {group}",
            )

    def test_covariance_cross_groups(self, model_kinetic):
        """Test covariance calculations between different groups."""
        kinetic_energy = np.array([10.0, 100.0])

        # Test covariance between different groups
        cov_p_he = model_kinetic.covariance("p", "He", kinetic_energy)
        cov_he_o = model_kinetic.covariance("He", "O*", kinetic_energy)

        # Covariance matrices should be square
        assert cov_p_he.shape == (len(kinetic_energy), len(kinetic_energy))
        assert cov_he_o.shape == (len(kinetic_energy), len(kinetic_energy))

        # Should be finite
        assert np.all(np.isfinite(cov_p_he))
        assert np.all(np.isfinite(cov_he_o))

    def test_scalar_input(self, model_kinetic):
        """Test that scalar inputs work correctly."""
        kinetic_energy = 10.0  # Scalar
        target = "p"

        flux = model_kinetic.flux(kinetic_energy, target)
        error = model_kinetic.error(kinetic_energy, target)

        # Should return arrays even for scalar input
        assert isinstance(flux, np.ndarray)
        assert isinstance(error, np.ndarray)
        assert len(flux) == 1
        assert len(error) == 1

        # Values should be positive
        assert flux[0] > 0
        assert error[0] > 0

    def test_physical_reasonableness(self, model_kinetic):
        """Test that flux values are physically reasonable."""
        kinetic_energy = np.logspace(0, 3, 100)  # 1 GeV/nucleon to 1 TeV/nucleon

        for group in ["p", "He", "O*", "Fe*"]:
            flux = model_kinetic.flux(kinetic_energy, group)

            # Flux should decrease with energy (cosmic ray spectrum)
            assert flux[0] > flux[-1], f"Flux should decrease with energy for {group}"

            # Flux should be in reasonable range (rough order of magnitude check)
            assert np.all(flux > 1e-10), f"Flux too small for {group}"
            assert np.all(flux < 1e10), f"Flux too large for {group}"

    def test_nucleon_mass_consistency(self, model_kinetic):
        """Test that the nucleon mass used is consistent."""
        nucleon_mass = 0.93891872965  # Standard nucleon mass

        # Test energy conversion for different groups
        kinetic_energy = np.array([10.0])

        for group in ["p", "He", "O*", "Fe*"]:
            total_energy = model_kinetic._transform_energy_per_nucleon(
                kinetic_energy, group
            )
            expected_total = kinetic_energy + nucleon_mass

            np.testing.assert_allclose(
                total_energy,
                expected_total,
                rtol=1e-10,
                err_msg=f"Nucleon mass inconsistent for {group}",
            )
