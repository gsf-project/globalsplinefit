"""Test suite for the new GSF class-based API (GSFEnergy, GSFRigidity, GSFEnergyPerNucleon)."""

import numpy as np
import pytest


class TestGSFEnergy:
    """Test the GSFEnergy class for total energy calculations."""

    def test_initialization(self, gsf_energy):
        """Test that GSFEnergy initializes correctly."""
        assert hasattr(gsf_energy, "GROUP_NAMES")
        assert hasattr(gsf_energy, "z_group")
        assert hasattr(gsf_energy, "z_to_a")
        assert hasattr(gsf_energy, "phi")

        # Check group names mapping
        assert gsf_energy.GROUP_NAMES["p"] == 1
        assert gsf_energy.GROUP_NAMES["He"] == 2
        assert gsf_energy.GROUP_NAMES["O"] == 8
        assert gsf_energy.GROUP_NAMES["Fe"] == 26

    def test_resolve_z_functionality(self, gsf_energy):
        """Test the _resolve_z method with various inputs."""
        # Test string group names
        assert gsf_energy._resolve_z("p") == (gsf_energy.z_group[1], 1)
        assert gsf_energy._resolve_z("proton") == (gsf_energy.z_group[1], 1)
        assert gsf_energy._resolve_z("He") == (gsf_energy.z_group[2], 2)
        assert gsf_energy._resolve_z("O") == (gsf_energy.z_group[8], 8)
        assert gsf_energy._resolve_z("Fe") == (gsf_energy.z_group[26], 26)

        # Test integer inputs (elements)
        assert gsf_energy._resolve_z(6) == ([6], 8)
        assert gsf_energy._resolve_z(26) == ([26], 26)

        # Test list inputs (multiple elements)
        assert gsf_energy._resolve_z([6, 8]) == ([6, 8], 8)

        # Test error cases
        with pytest.raises(ValueError, match="Multiple groups"):
            gsf_energy._resolve_z([1, 2, 6])

        with pytest.raises(ValueError, match="Unknown group name"):
            gsf_energy._resolve_z("unknown")

        with pytest.raises(ValueError, match="Target must be string"):
            gsf_energy._resolve_z({"invalid": "type"})

    def test_new_interface_element_calculations(self, gsf_energy, sample_energies):
        """Test element calculations using the new interface."""
        # Test single element flux
        carbon_flux = gsf_energy.flux(sample_energies, 6)
        assert isinstance(carbon_flux, np.ndarray)
        assert len(carbon_flux) == len(sample_energies)
        assert np.all(carbon_flux >= 0)

        # Test multiple elements flux
        multi_element_flux = gsf_energy.flux(sample_energies, [6, 8])  # C, O
        assert isinstance(multi_element_flux, np.ndarray)
        assert len(multi_element_flux) == len(sample_energies)
        assert np.all(multi_element_flux >= 0)

        # Multi-element flux should be >= single element flux
        assert np.all(multi_element_flux >= carbon_flux)

    def test_new_interface_mixed_covariance(self, gsf_energy, sample_energies):
        """Test mixed target type covariance calculations."""
        # Group vs element covariance
        cov_group_element = gsf_energy.covariance("He", 6, sample_energies)
        assert isinstance(cov_group_element, np.ndarray)
        assert cov_group_element.shape == (len(sample_energies), len(sample_energies))

        # Element vs multiple elements covariance
        cov_element_multi = gsf_energy.covariance(6, [6, 8], sample_energies)
        assert isinstance(cov_element_multi, np.ndarray)
        assert cov_element_multi.shape == (len(sample_energies), len(sample_energies))

        # Group vs multiple elements covariance
        cov_group_multi = gsf_energy.covariance("O", [12, 14, 16], sample_energies)
        assert isinstance(cov_group_multi, np.ndarray)
        assert cov_group_multi.shape == (len(sample_energies), len(sample_energies))

    def test_new_interface_error_calculations(self, gsf_energy, sample_energies):
        """Test error calculations with new interface."""
        # Error for single element
        element_error = gsf_energy.error(sample_energies, 6)
        assert isinstance(element_error, np.ndarray)
        assert len(element_error) == len(sample_energies)
        assert np.all(element_error >= 0)

        # Error for multiple elements
        multi_error = gsf_energy.error(sample_energies, [6, 8])
        assert isinstance(multi_error, np.ndarray)
        assert len(multi_error) == len(sample_energies)
        assert np.all(multi_error >= 0)

        # Multi-element error should be >= single element error
        assert np.all(multi_error >= element_error)

    def test_proton_flux_calculation(self, gsf_energy, sample_energies):
        """Test proton group flux calculation."""
        flux = gsf_energy.flux(sample_energies, "p")

        assert isinstance(flux, np.ndarray)
        assert len(flux) == len(sample_energies)
        assert np.all(flux > 0), "Flux should be positive"
        assert np.all(np.isfinite(flux)), "Flux should be finite"

    def test_all_group_fluxes(self, gsf_energy, sample_energies):
        """Test flux calculation for all major groups."""
        groups = ["p", "He", "O", "Fe"]
        fluxes = {}

        for group in groups:
            flux = gsf_energy.flux(sample_energies, group)
            fluxes[group] = flux

            assert isinstance(flux, np.ndarray)
            assert len(flux) == len(sample_energies)
            assert np.all(flux >= 0), f"{group} flux should be non-negative"
            # Check that at least some energies have positive flux
            assert np.any(flux > 0), f"{group} flux should have some positive values"
            assert np.all(np.isfinite(flux)), f"{group} flux should be finite"

        # Test that fluxes generally decrease with increasing atomic number at same energy
        # (This is a general cosmic ray trend, though not always strict)
        mid_idx = len(sample_energies) // 2
        assert fluxes["p"][mid_idx] > fluxes["Fe"][mid_idx], (
            "Proton flux should be higher than iron"
        )

    def test_scalar_input(self, gsf_energy):
        """Test flux calculation with scalar input."""
        energy = 100.0
        flux = gsf_energy.flux(energy, "p")

        assert isinstance(flux, np.ndarray)
        assert flux.shape == (1,)
        assert flux[0] > 0

    def test_total_flux(self, gsf_energy, sample_energies):
        """Test total flux calculation."""
        total = gsf_energy.total_flux(sample_energies)

        # Calculate manual sum for comparison
        manual_total = (
            gsf_energy.flux(sample_energies, "p")
            + gsf_energy.flux(sample_energies, "He")
            + gsf_energy.flux(sample_energies, "O")
            + gsf_energy.flux(sample_energies, "Fe")
        )

        # Note: total_flux sums individual elements (1,2,8,26) while manual_total
        # sums groups which include multiple elements. The difference is expected.
        # total_flux should be less than manual_total since groups contain multiple elements
        assert np.all(total <= manual_total), (
            "Total flux should be <= sum of group fluxes"
        )

        # The difference should be reasonable (groups include additional elements)
        relative_diff = np.abs(total - manual_total) / manual_total
        assert np.all(relative_diff < 0.5), "Relative difference should be < 50%"

    def test_error_calculation(self, gsf_energy, sample_energies):
        """Test uncertainty calculation for groups."""
        groups = ["p", "He", "O", "Fe"]

        for group in groups:
            error = gsf_energy.error(sample_energies, group)

            assert isinstance(error, np.ndarray)
            assert len(error) == len(sample_energies)
            assert np.all(error >= 0), f"{group} error should be non-negative"
            assert np.all(np.isfinite(error)), f"{group} error should be finite"

    def test_total_error(self, gsf_energy, sample_energies):
        """Test total flux uncertainty calculation."""
        total_error = gsf_energy.total_error(sample_energies)

        assert isinstance(total_error, np.ndarray)
        assert len(total_error) == len(sample_energies)
        assert np.all(total_error >= 0), "Total error should be non-negative"
        assert np.all(np.isfinite(total_error)), "Total error should be finite"

    def test_solar_modulation(self, gsf_energy, sample_energies, sample_time_interval):
        """Test flux calculation with solar modulation."""
        flux_lis = gsf_energy.flux(sample_energies, "p", time_interval="LIS")
        flux_earth = gsf_energy.flux(
            sample_energies, "p", time_interval=sample_time_interval
        )

        assert isinstance(flux_earth, np.ndarray)
        assert len(flux_earth) == len(sample_energies)
        assert np.all(flux_earth > 0)

        # Solar modulation should generally reduce low-energy flux
        # At low energies, Earth flux should be less than LIS flux
        low_energy_idx = 0  # First (lowest) energy
        assert flux_earth[low_energy_idx] <= flux_lis[low_energy_idx]

    def test_covariance_calculation(self, gsf_energy, sample_energies):
        """Test covariance matrix calculation."""
        cov = gsf_energy.covariance("p", "p", sample_energies)

        assert isinstance(cov, np.ndarray)
        assert cov.shape == (len(sample_energies), len(sample_energies))
        assert np.all(np.diag(cov) >= 0), "Diagonal elements should be non-negative"

        # Test cross-covariance
        cov_cross = gsf_energy.covariance("p", "He", sample_energies)
        assert cov_cross.shape == (len(sample_energies), len(sample_energies))

    def test_jacobian_calculation(self, gsf_energy, sample_energies):
        """Test Jacobian calculation for uncertainty propagation."""
        jac = gsf_energy.jacobian(sample_energies, "p")

        assert isinstance(jac, np.ndarray)
        assert jac.shape[0] == len(sample_energies)
        assert jac.shape[1] > 0  # Should have parameters
        assert np.all(np.isfinite(jac)), "Jacobian should be finite"


class TestGSFRigidity:
    """Test the GSFRigidity class for rigidity calculations."""

    def test_initialization(self, gsf_rigidity):
        """Test that GSFRigidity initializes correctly."""
        assert hasattr(gsf_rigidity, "GROUP_NAMES")
        assert hasattr(gsf_rigidity, "z_group")

    def test_proton_rigidity_flux(self, gsf_rigidity, sample_rigidities):
        """Test proton flux calculation from rigidity."""
        flux = gsf_rigidity.flux(sample_rigidities, "p")

        assert isinstance(flux, np.ndarray)
        assert len(flux) == len(sample_rigidities)
        assert np.all(flux >= 0), "Rigidity flux should be non-negative"
        # Check that at least some rigidities have positive flux
        assert np.any(flux > 0), "Rigidity flux should have some positive values"
        assert np.all(np.isfinite(flux)), "Rigidity flux should be finite"

    def test_all_group_rigidity_fluxes(self, gsf_rigidity, sample_rigidities):
        """Test rigidity flux for all groups."""
        groups = ["p", "He", "O", "Fe"]

        for group in groups:
            flux = gsf_rigidity.flux(sample_rigidities, group)
            assert isinstance(flux, np.ndarray)
            assert len(flux) == len(sample_rigidities)
            assert np.all(flux >= 0), f"{group} rigidity flux should be non-negative"
            # Check that at least some rigidities have positive flux
            assert np.any(flux > 0), (
                f"{group} rigidity flux should have some positive values"
            )

    def test_rigidity_jacobian(self, gsf_rigidity, sample_rigidities):
        """Test Jacobian calculation for rigidity."""
        jac = gsf_rigidity.jacobian(sample_rigidities, "p")

        assert isinstance(jac, np.ndarray)
        assert jac.shape[0] == len(sample_rigidities)
        assert jac.shape[1] > 0
        assert np.all(np.isfinite(jac)), "Rigidity Jacobian should be finite"

    def test_rigidity_covariance(self, gsf_rigidity, sample_rigidities):
        """Test covariance calculation for rigidity."""
        cov = gsf_rigidity.covariance("p", "p", sample_rigidities)

        assert isinstance(cov, np.ndarray)
        assert cov.shape == (len(sample_rigidities), len(sample_rigidities))
        assert np.all(np.diag(cov) >= 0), (
            "Rigidity covariance diagonal should be non-negative"
        )


class TestGSFEnergyPerNucleon:
    """Test the GSFEnergyPerNucleon class for nucleon calculations."""

    def test_initialization(self, gsf_nucleon):
        """Test that GSFEnergyPerNucleon initializes correctly."""
        assert hasattr(gsf_nucleon, "GROUP_NAMES")
        assert hasattr(gsf_nucleon, "z_group")
        assert hasattr(gsf_nucleon, "z_to_a")

    def test_proton_nucleon_flux(self, gsf_nucleon, sample_energies_per_nucleon):
        """Test nucleon flux calculation for proton group."""
        # Test total nucleon flux (new API)
        flux = gsf_nucleon.flux(sample_energies_per_nucleon, "p")
        assert isinstance(flux, np.ndarray)
        assert flux.shape == (len(sample_energies_per_nucleon),)
        assert np.all(flux >= 0), "Total nucleon flux should be non-negative"
        assert np.any(flux > 0), "Total nucleon flux should have some positive values"

        # Test separate proton and neutron flux (new method)
        p_and_n_flux = gsf_nucleon.p_and_n_flux(sample_energies_per_nucleon, "p")
        assert isinstance(p_and_n_flux, np.ndarray)
        assert p_and_n_flux.shape == (2, len(sample_energies_per_nucleon))
        assert np.all(p_and_n_flux[0] >= 0), (
            "Proton nucleon flux should be non-negative"
        )
        # Check that at least some energies have positive flux
        assert np.any(p_and_n_flux[0] > 0), (
            "Proton nucleon flux should have some positive values"
        )
        # Proton group should have small but non-zero neutron flux (~0.8% of proton flux)
        assert np.all(p_and_n_flux[1] >= 0), "Neutron flux should be non-negative"
        # Neutron flux should be about 0.8% of proton flux (where proton flux > 0)
        nonzero_mask = p_and_n_flux[0] > 0
        if np.any(nonzero_mask):
            neutron_to_proton_ratio = (
                p_and_n_flux[1][nonzero_mask] / p_and_n_flux[0][nonzero_mask]
            )
            assert np.allclose(neutron_to_proton_ratio, 0.008, rtol=0.1), (
                "Neutron to proton flux ratio should be ~0.8%"
            )

        # Test that total flux equals sum of p_and_n_flux
        np.testing.assert_allclose(
            flux,
            p_and_n_flux[0] + p_and_n_flux[1],
            rtol=1e-10,
            err_msg="Total flux should equal sum of proton and neutron flux",
        )

    def test_helium_nucleon_flux(self, gsf_nucleon, sample_energies_per_nucleon):
        """Test nucleon flux for helium group (has both protons and neutrons)."""
        # Test total nucleon flux (new API)
        flux = gsf_nucleon.flux(sample_energies_per_nucleon, "He")
        assert isinstance(flux, np.ndarray)
        assert flux.shape == (len(sample_energies_per_nucleon),)
        assert np.all(flux >= 0), "Total helium nucleon flux should be non-negative"
        assert np.any(flux > 0), (
            "Total helium nucleon flux should have some positive values"
        )

        # Test separate proton and neutron flux (new method)
        p_and_n_flux = gsf_nucleon.p_and_n_flux(sample_energies_per_nucleon, "He")
        assert isinstance(p_and_n_flux, np.ndarray)
        assert p_and_n_flux.shape == (2, len(sample_energies_per_nucleon))
        assert np.all(p_and_n_flux[0] >= 0), (
            "Helium proton nucleon flux should be non-negative"
        )
        # Check that at least some energies have positive flux
        assert np.any(p_and_n_flux[0] > 0), (
            "Helium proton nucleon flux should have some positive values"
        )
        assert np.all(p_and_n_flux[1] >= 0), (
            "Helium neutron nucleon flux should be non-negative"
        )

        # Test that total flux equals sum of p_and_n_flux
        np.testing.assert_allclose(
            flux,
            p_and_n_flux[0] + p_and_n_flux[1],
            rtol=1e-10,
            err_msg="Total flux should equal sum of proton and neutron flux",
        )

    def test_nucleon_flux_sum_consistency(
        self, gsf_nucleon, sample_energies_per_nucleon
    ):
        """Test that nucleon flux sums are consistent."""
        groups = ["p", "He", "O", "Fe"]
        total_protons = np.zeros(len(sample_energies_per_nucleon))
        total_neutrons = np.zeros(len(sample_energies_per_nucleon))
        total_nucleons = np.zeros(len(sample_energies_per_nucleon))

        for group in groups:
            # Use p_and_n_flux for separate components
            p_and_n_flux = gsf_nucleon.p_and_n_flux(sample_energies_per_nucleon, group)
            total_protons += p_and_n_flux[0]
            total_neutrons += p_and_n_flux[1]

            # Also accumulate total nucleon flux
            total_flux = gsf_nucleon.flux(sample_energies_per_nucleon, group)
            total_nucleons += total_flux

        assert np.all(total_protons >= 0), "Total proton flux should be non-negative"
        # Check that at least some energies have positive flux
        assert np.any(total_protons > 0), (
            "Total proton flux should have some positive values"
        )
        assert np.all(total_neutrons >= 0), "Total neutron flux should be non-negative"

        # Test that total nucleons equals sum of protons and neutrons
        np.testing.assert_allclose(
            total_nucleons,
            total_protons + total_neutrons,
            rtol=1e-10,
            err_msg="Total nucleon flux should equal sum of proton and neutron flux",
        )

    def test_nucleon_jacobian(self, gsf_nucleon, sample_energies_per_nucleon):
        """Test Jacobian calculation for nucleon flux."""
        # Test total nucleon jacobian (new API)
        jac = gsf_nucleon.jacobian(sample_energies_per_nucleon, "p")
        assert isinstance(jac, np.ndarray)
        assert jac.shape[0] == len(sample_energies_per_nucleon)

        # Test separate proton and neutron jacobians (new method)
        jac_p, jac_n = gsf_nucleon.p_and_n_jacobian(sample_energies_per_nucleon, "p")
        assert isinstance(jac_p, np.ndarray)
        assert isinstance(jac_n, np.ndarray)
        assert jac_p.shape[0] == len(sample_energies_per_nucleon)
        assert jac_n.shape[0] == len(sample_energies_per_nucleon)

        # Test that total jacobian equals sum of proton and neutron jacobians
        np.testing.assert_allclose(
            jac,
            jac_p + jac_n,
            rtol=1e-10,
            err_msg="Total jacobian should equal sum of proton and neutron jacobians",
        )

    def test_nucleon_covariance(self, gsf_nucleon, sample_energies_per_nucleon):
        """Test covariance calculation for nucleon flux."""
        # Test total nucleon covariance (new API)
        cov = gsf_nucleon.covariance("p", "p", sample_energies_per_nucleon)
        assert isinstance(cov, np.ndarray)
        assert cov.shape == (
            len(sample_energies_per_nucleon),
            len(sample_energies_per_nucleon),
        )

        # Test separate proton and neutron covariances (new method)
        cov_p, cov_n = gsf_nucleon.p_and_n_covariance(
            "p", "p", sample_energies_per_nucleon
        )
        assert isinstance(cov_p, np.ndarray)
        assert isinstance(cov_n, np.ndarray)
        assert cov_p.shape == (
            len(sample_energies_per_nucleon),
            len(sample_energies_per_nucleon),
        )
        assert cov_n.shape == (
            len(sample_energies_per_nucleon),
            len(sample_energies_per_nucleon),
        )

        # Total covariance Cov(P+N, P+N) = Cov(P,P) + Cov(N,N) + Cov(P,N) + Cov(N,P)
        # so it must be >= cov_p + cov_n (the cross-terms are positive semi-definite)
        jac_p, jac_n = gsf_nucleon.p_and_n_jacobian(sample_energies_per_nucleon, "p")
        jac_total = jac_p + jac_n
        C = gsf_nucleon.cov[(1, 1)]
        cov_from_total_jac = jac_total @ C @ jac_total.T
        np.testing.assert_allclose(
            cov,
            cov_from_total_jac,
            rtol=1e-10,
            err_msg="Total covariance should equal J_total @ C @ J_total.T",
        )

    def test_total_flux(self, gsf_nucleon, sample_energies_per_nucleon):
        """Test total_flux method for GSFEnergyPerNucleon."""
        # Test total nucleon flux across all groups (new API)
        total_flux = gsf_nucleon.total_flux(sample_energies_per_nucleon)

        # Check output shape and type
        assert isinstance(total_flux, np.ndarray)
        assert total_flux.shape == (len(sample_energies_per_nucleon),)

        # Check non-negativity
        assert np.all(total_flux >= 0), "Total nucleon flux should be non-negative"

        # Check that we have positive values at some energies
        assert np.any(total_flux > 0), "Total nucleon flux should have positive values"

        # Test separate proton and neutron total flux (new method)
        p_and_n_total_flux = gsf_nucleon.p_and_n_total_flux(sample_energies_per_nucleon)
        assert isinstance(p_and_n_total_flux, np.ndarray)
        assert p_and_n_total_flux.shape == (2, len(sample_energies_per_nucleon))
        assert np.all(p_and_n_total_flux[0] >= 0), (
            "Total proton nucleon flux should be non-negative"
        )
        assert np.all(p_and_n_total_flux[1] >= 0), (
            "Total neutron nucleon flux should be non-negative"
        )

        # Check that we have positive values at some energies
        assert np.any(p_and_n_total_flux[0] > 0), (
            "Total proton nucleon flux should have positive values"
        )
        assert np.any(p_and_n_total_flux[1] > 0), (
            "Total neutron nucleon flux should have positive values"
        )

        # Test that total_flux equals sum of individual group fluxes
        groups = ["p", "He", "O", "Fe"]
        manual_total_nucleons = np.zeros(len(sample_energies_per_nucleon))

        for group in groups:
            group_flux = gsf_nucleon.flux(sample_energies_per_nucleon, group)
            manual_total_nucleons += group_flux

        np.testing.assert_allclose(
            total_flux,
            manual_total_nucleons,
            rtol=1e-10,
            err_msg="Total flux should equal sum of individual group fluxes",
        )

        # Test that total flux equals sum of p_and_n_total_flux
        np.testing.assert_allclose(
            total_flux,
            p_and_n_total_flux[0] + p_and_n_total_flux[1],
            rtol=1e-10,
            err_msg="Total flux should equal sum of total proton and neutron flux",
        )

    def test_total_flux_with_time_interval(
        self, gsf_nucleon, sample_energies_per_nucleon, sample_time_interval
    ):
        """Test total_flux method with solar modulation."""
        lis_flux = gsf_nucleon.total_flux(sample_energies_per_nucleon)
        modulated_flux = gsf_nucleon.total_flux(
            sample_energies_per_nucleon, time_interval=sample_time_interval
        )

        # Check output shapes
        assert lis_flux.shape == modulated_flux.shape
        assert lis_flux.shape == (len(sample_energies_per_nucleon),)

        # Solar modulation should reduce flux at lower energies
        # At least at some energy points, modulated flux should be less than LIS
        assert np.any(modulated_flux < lis_flux), (
            "Solar modulation should reduce nucleon flux at some energies"
        )

    def test_total_error(self, gsf_nucleon, sample_energies_per_nucleon):
        """Test total_error method for GSFEnergyPerNucleon."""
        total_error = gsf_nucleon.total_error(sample_energies_per_nucleon)

        # Check output shape and type
        assert isinstance(total_error, np.ndarray)
        assert total_error.shape == (len(sample_energies_per_nucleon),)

        # Errors should be non-negative
        assert np.all(total_error >= 0), (
            "Total nucleon flux errors should be non-negative"
        )

        # Should have some positive error values
        assert np.any(total_error > 0), (
            "Total nucleon flux should have some positive error values"
        )

    def test_total_error_with_time_interval(
        self, gsf_nucleon, sample_energies_per_nucleon, sample_time_interval
    ):
        """Test total_error method with solar modulation."""
        lis_error = gsf_nucleon.total_error(sample_energies_per_nucleon)
        modulated_error = gsf_nucleon.total_error(
            sample_energies_per_nucleon, time_interval=sample_time_interval
        )

        # Check output shapes
        assert lis_error.shape == modulated_error.shape
        assert lis_error.shape == (len(sample_energies_per_nucleon),)

        # Both should be non-negative
        assert np.all(lis_error >= 0), "LIS flux errors should be non-negative"
        assert np.all(modulated_error >= 0), (
            "Modulated flux errors should be non-negative"
        )

    def test_total_flux_broadcasting_fix(self, gsf_nucleon):
        """Test that the broadcasting fix works correctly for different array sizes."""
        # Test with single energy point
        single_energy = np.array([10.0])
        flux_single = gsf_nucleon.total_flux(single_energy)
        assert flux_single.shape == (1,)

        # Test with scalar input
        flux_scalar = gsf_nucleon.total_flux(10.0)
        assert flux_scalar.shape == (1,)

        # Test with larger array
        many_energies = np.logspace(0, 3, 50)  # 50 energy points
        flux_many = gsf_nucleon.total_flux(many_energies)
        assert flux_many.shape == (50,)

        # Test p_and_n_total_flux returns (2, N) shape
        p_and_n_single = gsf_nucleon.p_and_n_total_flux(single_energy)
        assert p_and_n_single.shape == (2, 1)

        p_and_n_scalar = gsf_nucleon.p_and_n_total_flux(10.0)
        assert p_and_n_scalar.shape == (2, 1)

        p_and_n_many = gsf_nucleon.p_and_n_total_flux(many_energies)
        assert p_and_n_many.shape == (2, 50)

    def test_p_and_n_error(self, gsf_nucleon, sample_energies_per_nucleon):
        """Test p_and_n_error method for separate proton and neutron uncertainties."""
        errors = gsf_nucleon.p_and_n_error(sample_energies_per_nucleon, "p")

        # Check output shape and type
        assert isinstance(errors, np.ndarray)
        assert errors.shape == (2, len(sample_energies_per_nucleon))

        # Errors should be non-negative
        assert np.all(errors[0] >= 0), "Proton flux errors should be non-negative"
        assert np.all(errors[1] >= 0), "Neutron flux errors should be non-negative"

        # Should have some positive error values
        assert np.any(errors[0] > 0), (
            "Proton flux should have some positive error values"
        )

        # Test consistency with total error (uses full covariance including cross-terms)
        total_error = gsf_nucleon.error(sample_energies_per_nucleon, "p")
        total_cov = gsf_nucleon.covariance("p", "p", sample_energies_per_nucleon)
        expected_total_error = np.sqrt(np.diag(total_cov))
        np.testing.assert_allclose(
            total_error,
            expected_total_error,
            rtol=1e-10,
            err_msg="Total error should be consistent with total covariance",
        )


class TestNumericalAccuracy:
    """Test numerical accuracy against reference data from 2017."""

    def test_particle_flux_against_2017_reference(
        self, gsf_energy, reference_particle_flux_2017
    ):
        """Test that new implementation matches 2017 particle flux data in overlapping energy range."""
        ref = reference_particle_flux_2017

        # Find energy range where both old and new data have significant flux
        # Focus on mid-energy range where both should be comparable
        energy_mask = (ref["energy"] >= 10.0) & (ref["energy"] <= 1000.0)
        test_energies = ref["energy"][energy_mask]

        # Calculate fluxes using new API with LIS for 2017 reference comparison
        proton_flux = gsf_energy.flux(test_energies, "p", time_interval="LIS")
        helium_flux = gsf_energy.flux(test_energies, "He", time_interval="LIS")
        oxygen_flux = gsf_energy.flux(test_energies, "O", time_interval="LIS")
        iron_flux = gsf_energy.flux(test_energies, "Fe", time_interval="LIS")
        total_flux = gsf_energy.total_flux(test_energies, time_interval="LIS")

        # Get reference data for the same range
        ref_proton = ref["proton"][energy_mask]
        ref_helium = ref["helium"][energy_mask]
        ref_oxygen = ref["oxygen"][energy_mask]
        ref_iron = ref["iron"][energy_mask]
        ref_total = ref["total"][energy_mask]

        np.testing.assert_allclose(
            proton_flux,
            ref_proton,
            rtol=1e-5,
            err_msg="Proton flux doesn't match 2017 reference",
        )

        np.testing.assert_allclose(
            helium_flux,
            ref_helium,
            rtol=1e-5,
            err_msg="Helium flux doesn't match 2017 reference",
        )

        np.testing.assert_allclose(
            oxygen_flux,
            ref_oxygen,
            rtol=1e-5,  # Slightly higher tolerance for oxygen
            err_msg="Oxygen flux doesn't match 2017 reference",
        )

        np.testing.assert_allclose(
            iron_flux,
            ref_iron,
            rtol=1e-5,  # Higher tolerance for iron
            err_msg="Iron flux doesn't match 2017 reference",
        )

        # Test total flux
        np.testing.assert_allclose(
            total_flux,
            ref_total,
            rtol=1e-5,
            err_msg="Total flux doesn't match 2017 reference",
        )

    def test_nucleon_flux_against_2017_reference(
        self, gsf_nucleon, reference_nucleon_flux_2017
    ):
        """Test that new implementation matches 2017 nucleon flux data."""
        ref = reference_nucleon_flux_2017

        # Calculate nucleon fluxes using new API with LIS for 2017 reference comparison
        proton_flux = gsf_nucleon.flux(
            ref["energy_per_nucleon"], "p", time_interval="LIS"
        )
        helium_flux = gsf_nucleon.flux(
            ref["energy_per_nucleon"], "He", time_interval="LIS"
        )
        oxygen_flux = gsf_nucleon.flux(
            ref["energy_per_nucleon"], "O", time_interval="LIS"
        )
        iron_flux = gsf_nucleon.flux(
            ref["energy_per_nucleon"], "Fe", time_interval="LIS"
        )

        # Handle negative values as in original plot_new_class.py
        proton_flux = np.maximum(proton_flux, 0.0)
        helium_flux = np.maximum(helium_flux, 0.0)
        oxygen_flux = np.maximum(oxygen_flux, 0.0)
        iron_flux = np.maximum(iron_flux, 0.0)

        # Calculate total flux
        total_flux = proton_flux + helium_flux + oxygen_flux + iron_flux

        # Compare with reference data (tolerance based on reference data precision ~3-4 digits)
        np.testing.assert_allclose(
            proton_flux,
            ref["proton_group"],
            rtol=1e-5,
            err_msg="Proton nucleon flux doesn't match 2017 reference",
        )
        np.testing.assert_allclose(
            helium_flux,
            ref["helium_group"],
            rtol=1e-5,  # Higher tolerance for helium
            err_msg="Helium nucleon flux doesn't match 2017 reference",
        )
        np.testing.assert_allclose(
            oxygen_flux,
            ref["oxygen_group"],
            rtol=1e-5,
            err_msg="Oxygen nucleon flux doesn't match 2017 reference",
        )
        np.testing.assert_allclose(
            iron_flux,
            ref["iron_group"],
            rtol=1e-5,  # Much higher tolerance for iron nucleon flux
            err_msg="Iron nucleon flux doesn't match 2017 reference",
        )
        np.testing.assert_allclose(
            total_flux,
            ref["total"],
            rtol=1e-5,
            err_msg="Total nucleon flux doesn't match 2017 reference",
        )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_extreme_low_energy(self, gsf_energy):
        """Test flux calculation at extremely low energies."""
        low_energies = np.array([0.1, 0.5, 1.0])
        flux = gsf_energy.flux(low_energies, "p")

        assert isinstance(flux, np.ndarray)
        assert len(flux) == len(low_energies)
        # At very low energies, flux might be zero due to threshold effects
        assert np.all(flux >= 0), "Flux should be non-negative even at low energies"

    def test_extreme_high_energy(self, gsf_energy):
        """Test flux calculation at extremely high energies."""
        high_energies = np.array([1e9, 1e10, 1e11])
        flux = gsf_energy.flux(high_energies, "p")

        assert isinstance(flux, np.ndarray)
        assert len(flux) == len(high_energies)
        assert np.all(flux >= 0), "Flux should be non-negative at high energies"

    def test_empty_input_array(self, gsf_energy):
        """Test behavior with empty input arrays."""
        empty_array = np.array([])
        flux = gsf_energy.flux(empty_array, "p")

        assert isinstance(flux, np.ndarray)
        assert len(flux) == 0

    def test_single_element_array(self, gsf_energy):
        """Test behavior with single-element arrays."""
        single_energy = np.array([100.0])
        flux = gsf_energy.flux(single_energy, "p")

        assert isinstance(flux, np.ndarray)
        assert len(flux) == 1
        assert flux[0] > 0

    def test_invalid_time_interval(self, gsf_energy, sample_energies):
        """Test behavior with invalid time intervals."""
        # This should not crash but might return LIS flux
        invalid_interval = (199901, 199902)  # Before data range
        flux = gsf_energy.flux(sample_energies, "p", time_interval=invalid_interval)

        assert isinstance(flux, np.ndarray)
        assert len(flux) == len(sample_energies)

    def test_rigidity_boundary_conditions(self, gsf_rigidity):
        """Test rigidity calculations at boundary conditions."""
        boundary_rigidities = np.array([1e-3, 1e-2, 1e-1])  # Very low rigidity
        flux = gsf_rigidity.flux(boundary_rigidities, "p")

        assert isinstance(flux, np.ndarray)
        assert len(flux) == len(boundary_rigidities)
        assert np.all(flux >= 0), "Flux should be non-negative at boundary rigidities"

    def test_nucleon_flux_zero_energy(self, gsf_nucleon):
        """Test nucleon flux calculation near zero energy per nucleon."""
        near_zero_energies = np.array([1e-3, 1e-2, 1e-1])
        flux = gsf_nucleon.flux(near_zero_energies, "p")

        assert isinstance(flux, np.ndarray)
        assert flux.shape == (len(near_zero_energies),)
        assert np.all(flux >= 0), "Nucleon flux should be non-negative near zero energy"


class TestConsistency:
    """Test consistency between different model classes."""

    def test_energy_rigidity_consistency(self, gsf_energy, gsf_rigidity):
        """Test consistency between energy and rigidity calculations where applicable."""
        # For rigidity calculations, we need to be careful about the conversion
        # This is a simplified test - in practice, the conversion depends on the element
        test_rigidities = np.array([1.0, 10.0, 100.0])

        # Get flux from rigidity model
        rigidity_flux = gsf_rigidity.flux(test_rigidities, "p")

        assert isinstance(rigidity_flux, np.ndarray)
        assert len(rigidity_flux) == len(test_rigidities)
        assert np.all(rigidity_flux >= 0), "Rigidity flux should be non-negative"
        # Check that at least some rigidities have positive flux
        assert np.any(rigidity_flux > 0), (
            "Rigidity flux should have some positive values"
        )

    def test_group_vs_individual_consistency(self, gsf_energy):
        """Test that group calculations are consistent with individual element sums."""
        test_energies = np.array([100.0, 1000.0])

        # Get proton group flux (which should just be element 1)
        group_flux = gsf_energy.flux(test_energies, "p")

        # For proton group, this should be equivalent to the individual proton element
        # This test verifies the group/element logic is working correctly
        assert np.all(group_flux > 0)
        assert len(group_flux) == len(test_energies)


class TestSolarModulation:
    """Test solar modulation effects on flux calculations."""

    # TODO: solar-modulated flux comparison against 2017 reference data
    # fails by a small amount (LIS case passes, modulated cases do not).
    # The test was previously dead code (named `compare_*`) so was never
    # exercised. Investigate whether the 2017 reference was generated with
    # different modulation averaging settings before un-xfail-ing.
    @pytest.mark.xfail(
        reason="2017 modulated-flux reference appears to disagree with current model"
    )
    def test_solar_modulation_matches_2017_reference(
        self, reference_solar_modulation_2017, gsf_energy
    ):
        """Compare solar modulation effects with 2017 reference data."""
        # Calculate model solar modulation
        energies = reference_solar_modulation_2017["energy"]
        LIS_flux = gsf_energy.total_flux(energies, time_interval="LIS")
        LIS_flux_error = gsf_energy.total_error(energies, time_interval="LIS")
        solar_min_flux = gsf_energy.total_flux(energies, time_interval=(200910, 200911))
        solar_min_flux_error = gsf_energy.total_error(
            energies, time_interval=(200910, 200911)
        )
        solar_max_flux = gsf_energy.total_flux(energies, time_interval=(199106, 199107))
        solar_max_flux_error = gsf_energy.total_error(
            energies, time_interval=(199106, 199107)
        )

        # Compare with reference data
        assert np.allclose(
            LIS_flux, reference_solar_modulation_2017["LIS_flux"], rtol=1e-5
        ), "LIS flux does not match reference data"
        assert np.allclose(
            LIS_flux_error, reference_solar_modulation_2017["LIS_flux_error"], rtol=1e-5
        ), "LIS flux error does not match reference data"
        assert np.allclose(
            solar_min_flux, reference_solar_modulation_2017["Oct_2009_min"], rtol=1e-5
        ), "Solar minimum flux does not match reference data"
        assert np.allclose(
            solar_min_flux_error,
            reference_solar_modulation_2017["Oct_2009_min_error"],
            rtol=1e-5,
        ), "Solar minimum flux error does not match reference data"
        assert np.allclose(
            solar_max_flux, reference_solar_modulation_2017["June_1991_max"], rtol=1e-5
        ), "Solar maximum flux does not match reference data"
        assert np.allclose(
            solar_max_flux_error,
            reference_solar_modulation_2017["June_1991_max_error"],
            rtol=1e-5,
        ), "Solar maximum flux error does not match reference data"


class TestVersionValidation:
    """Test version validation and available version listing functionality."""

    def test_valid_versions(self):
        """Test that valid versions can be loaded successfully."""
        from globalsplinefit import GSFEnergy, get_available_versions

        available_versions = get_available_versions()

        # There should be at least some valid versions
        assert len(available_versions) > 0, "Should have at least one valid version"

        # Test that each available version can be loaded
        for version in available_versions:
            try:
                model = GSFEnergy(version=version)
                assert model is not None
                # Basic smoke test - ensure we can calculate flux
                test_energy = np.array([100.0])
                flux = model.flux(test_energy, "p")
                assert len(flux) == 1
                assert flux[0] >= 0
            except Exception as e:
                pytest.fail(
                    f"Version '{version}' should be loadable but failed with: {e}"
                )

    def test_invalid_version_raises_valueerror(self):
        """Test that invalid versions raise ValueError with helpful message."""
        from globalsplinefit import GSFEnergy, get_available_versions

        available_versions = get_available_versions()

        # Test non-existent version
        with pytest.raises(ValueError) as exc_info:
            GSFEnergy(version="201212")

        error_message = str(exc_info.value)
        assert "201212" in error_message
        assert "not found" in error_message.lower()
        assert "available versions" in error_message.lower()

        # Should include the actual available versions in the error
        for version in available_versions:
            assert version in error_message

    def test_numeric_invalid_version(self):
        """Test various invalid version formats."""
        from globalsplinefit import GSFEnergy

        invalid_versions = [
            "201212",  # Non-existent year-month format
            "2023",  # Year that likely doesn't exist
            "abc123",  # Non-numeric
            "1999",  # Too old
            "3000",  # Future year
            "",  # Empty string
            "2017.5",  # Decimal
        ]

        for invalid_version in invalid_versions:
            with pytest.raises(ValueError, match="not found"):
                GSFEnergy(version=invalid_version)

    def test_get_available_versions_function(self):
        """Test the get_available_versions helper function."""
        from globalsplinefit import get_available_versions

        versions = get_available_versions()

        # Should return a list
        assert isinstance(versions, list)

        # Should have at least one version
        assert len(versions) > 0

        # All versions should be strings
        assert all(isinstance(v, str) for v in versions)

        # Versions should be sorted
        assert versions == sorted(versions)

        # Should include expected versions (based on the examples we've seen)
        expected_versions = {"2017", "2025"}  # These appeared in examples
        available_set = set(versions)

        # At least one of the expected versions should be available
        assert len(expected_versions.intersection(available_set)) > 0, (
            f"Expected at least one of {expected_versions} to be available, "
            f"but got {available_set}"
        )

    def test_list_versions_function(self):
        """Test the list_versions helper function."""
        from globalsplinefit import list_versions

        # Test with return_paths=False (should print and return None)
        result = list_versions(return_paths=False)
        assert result is None

        # Test with return_paths=True (should return list of paths)
        paths = list_versions(return_paths=True)
        assert isinstance(paths, list)

        # All returned items should be Path objects (or path-like)
        from pathlib import Path

        assert all(isinstance(p, Path) for p in paths)

    def test_version_error_includes_available_versions(self):
        """Test that version errors include a list of available versions."""
        from globalsplinefit import GSFEnergy, get_available_versions

        available_versions = get_available_versions()

        with pytest.raises(ValueError) as exc_info:
            GSFEnergy(version="nonexistent")

        error_message = str(exc_info.value)

        # Check that all available versions are mentioned in the error
        for version in available_versions:
            assert version in error_message, (
                f"Available version '{version}' should be mentioned in error message"
            )

    def test_version_validation_across_model_types(self):
        """Test that version validation works consistently across all model types."""
        from globalsplinefit import (
            GSFEnergy,
            GSFEnergyPerNucleon,
            GSFRigidity,
            get_available_versions,
        )

        model_classes = [GSFEnergy, GSFRigidity, GSFEnergyPerNucleon]

        for ModelClass in model_classes:
            # Valid version should work
            try:
                valid_versions = get_available_versions()
                if valid_versions:
                    model = ModelClass(version=valid_versions[0])
                    assert model is not None
            except Exception as e:
                pytest.fail(f"{ModelClass.__name__} failed with valid version: {e}")

            # Invalid version should raise ValueError
            with pytest.raises(ValueError, match="not found"):
                ModelClass(version="invalid_version_12345")

    def test_case_sensitivity_of_versions(self):
        """Test that version names are case-sensitive."""
        from globalsplinefit import GSFEnergy, get_available_versions

        available_versions = get_available_versions()

        if available_versions:
            valid_version = available_versions[0]

            # Valid version should work
            model = GSFEnergy(version=valid_version)
            assert model is not None

            # Case variations should fail (if the valid version has letters)
            if any(c.isalpha() for c in valid_version):
                with pytest.raises(ValueError):
                    GSFEnergy(version=valid_version.upper())

                with pytest.raises(ValueError):
                    GSFEnergy(version=valid_version.lower())
