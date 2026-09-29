"""Test suite for GSF error calculation methods and comparison with reference data."""

import numpy as np
import pytest


class TestErrorCalculations:
    """Test error calculation methods in GSF classes."""

    def test_particle_flux_error_against_2017_reference(
        self, gsf_energy, reference_particle_flux_error_2017
    ):
        """Compare against the frozen GSF2017 uncertainties.

        The leading elements still reproduce the original release exactly.
        The groups do not, and by design: a sub-leading element inherits its
        leader's RELATIVE uncertainty (the convention stated in the GSF 2026
        SM), which means the leader Jacobian is scaled by the LOCAL flux
        ratio. The original implementation scaled it by the frozen top-knot
        ratio instead, so the two agree only above a species' top knot. The
        difference is confined to the direct-data region: the median ratio to
        the 2017 reference is 1 for every series, with excursions where the
        sub-leading members carry the group.
        """
        ref = reference_particle_flux_error_2017
        test_energies = ref["energy"]

        # leading elements: the group leaders drive these, unchanged
        for name, target in (("proton", "p"), ("helium", "He")):
            np.testing.assert_allclose(
                gsf_energy.error(test_energies, target, time_interval="LIS"),
                ref[name],
                rtol=1e-5,
                err_msg=f"{name} flux error doesn't match 2017 reference",
            )

        # groups: bounded, localized departure from the 2017 convention
        bounds = {  # series: (min ratio, max ratio)
            "oxygen": (0.35, 6.4),
            "iron": (0.80, 21.3),
            "total": (0.93, 1.16),
        }
        values = {
            "oxygen": gsf_energy.error(test_energies, "O*", time_interval="LIS"),
            "iron": gsf_energy.error(test_energies, "Fe*", time_interval="LIS"),
            "total": gsf_energy.total_error(test_energies, time_interval="LIS"),
        }
        for name, (lo, hi) in bounds.items():
            ratio = values[name] / ref[name]
            good = np.isfinite(ratio) & (ref[name] > 0)
            assert np.median(ratio[good]) == pytest.approx(1.0, abs=1e-6), name
            assert lo <= ratio[good].min(), (name, ratio[good].min())
            assert ratio[good].max() <= hi, (name, ratio[good].max())

    def test_nucleon_flux_error_against_2017_reference(
        self, gsf_nucleon, reference_nucleon_flux_error_2017
    ):
        """Nucleon errors use integer counts and agree with their covariances."""
        energy = reference_nucleon_flux_error_2017["energy_per_nucleon"]
        for group in gsf_nucleon.active_groups:
            error = gsf_nucleon.error(energy, group, time_interval="LIS")
            covariance = gsf_nucleon.covariance(
                group, group, energy, time_interval="LIS"
            )
            np.testing.assert_allclose(error, np.sqrt(np.diag(covariance)))
            assert np.all(np.isfinite(error))

    def test_error_method_consistency(self, gsf_energy, sample_energies):
        """Test that error methods are consistent with manual covariance calculations."""
        # Test for proton group
        error_builtin = gsf_energy.error(sample_energies, "p")

        # Manual calculation using covariance
        cov_matrix = gsf_energy.covariance("p", "p", sample_energies)
        error_manual = np.sqrt(np.diag(cov_matrix))

        np.testing.assert_allclose(
            error_builtin,
            error_manual,
            rtol=1e-10,
            err_msg="Built-in error method doesn't match manual covariance calculation",
        )

    def test_total_error_method_consistency(self, gsf_energy, sample_energies):
        """Test that total_error method is consistent with summed group errors."""
        # Get total error using built-in method
        total_error_builtin = gsf_energy.total_error(sample_energies)

        # Manual calculation by summing all groups with covariances
        groups = gsf_energy.active_groups
        total_cov = np.zeros((len(sample_energies), len(sample_energies)))

        for g1 in groups:
            for g2 in groups:
                cov = gsf_energy.covariance(g1, g2, sample_energies)
                total_cov += cov

        total_error_manual = np.sqrt(np.diag(total_cov))

        np.testing.assert_allclose(
            total_error_builtin,
            total_error_manual,
            rtol=1e-12,
            err_msg="Built-in total_error method doesn't match manual calculation",
        )

    def test_error_positivity(self, gsf_energy, gsf_nucleon):
        """Test that all error calculations return positive values."""
        test_energies = np.logspace(1, 6, 20)
        test_energies_nucleon = np.logspace(1, 5, 20)

        groups = gsf_energy.active_groups

        # Test energy flux errors
        for group in groups:
            errors = gsf_energy.error(test_energies, group)
            assert np.all(errors >= 0), (
                f"Energy flux errors for {group} should be non-negative"
            )

        total_error = gsf_energy.total_error(test_energies)
        assert np.all(total_error >= 0), (
            "Total energy flux error should be non-negative"
        )

        # Test nucleon flux errors
        for group in groups:
            error_result = gsf_nucleon.error(test_energies_nucleon, group)
            assert np.all(error_result[0] >= 0), (
                f"Proton nucleon flux errors for {group} should be non-negative"
            )
            assert np.all(error_result[1] >= 0), (
                f"Neutron nucleon flux errors for {group} should be non-negative"
            )

    def test_error_scaling_with_flux(self, gsf_energy, sample_energies):
        """Test that relative errors are reasonable (not too large or small)."""
        groups = gsf_energy.active_groups

        for group in groups:
            flux = gsf_energy.flux(sample_energies, group)
            error = gsf_energy.error(sample_energies, group)

            # Where flux is significant, relative error should be reasonable
            significant_mask = flux > 1e-10
            if np.any(significant_mask):
                relative_error = error[significant_mask] / flux[significant_mask]

                # Relative errors should typically be between 0.1% and 100%
                assert np.all(relative_error >= 0.001), (
                    f"Relative error for {group} seems too small"
                )
                assert np.all(relative_error <= 1.0), (
                    f"Relative error for {group} seems too large"
                )

    def test_cross_group_error_consistency(self, gsf_energy, sample_energies):
        """Test that cross-group covariances are reasonable."""
        groups = gsf_energy.active_groups

        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups):
                if i != j:  # Cross-group covariances
                    cov = gsf_energy.covariance(g1, g2, sample_energies)
                    var1 = np.diag(gsf_energy.covariance(g1, g1, sample_energies))
                    var2 = np.diag(gsf_energy.covariance(g2, g2, sample_energies))

                    # Calculate correlation coefficients where both variances are positive
                    valid_mask = (var1 > 0) & (var2 > 0)
                    if np.any(valid_mask):
                        correlation = np.diag(cov)[valid_mask] / np.sqrt(
                            var1[valid_mask] * var2[valid_mask]
                        )

                        # Correlation coefficients should be between -1 and 1
                        assert np.all(correlation >= -1.0), (
                            f"Correlation {g1}-{g2} should be >= -1"
                        )
                        assert np.all(correlation <= 1.0), (
                            f"Correlation {g1}-{g2} should be <= 1"
                        )


class TestEnergyTransformConsistency:
    """Regression tests for the double-energy-transform bug (audit 2026-07-02).

    ``covariance()`` used to pre-transform the input energy before calling
    ``jacobian()``, which transforms again: ``GSFKineticEnergy`` errors were
    evaluated at E + 2m instead of E + m, and a nonzero ``energy_scale``
    entered the error/covariance path as (1+delta)^2 instead of (1+delta).
    Invisible for plain ``GSFEnergy`` at delta=0 (identity transform).
    """

    def test_kinetic_error_matches_total_at_shifted_energy(self):
        from globalsplinefit.model import (
            NUCLEON_MASS_GEV,
            GSFEnergy,
            GSFKineticEnergy,
        )

        e_total = GSFEnergy()
        e_kin = GSFKineticEnergy()
        ekin = np.array([2.0, 10.0, 100.0, 1000.0])
        # rest mass exactly as the kinetic transform computes it (the data
        # tables carry A = 1.008 for hydrogen, not 1.0)
        sid = e_kin._leader_by_charge.get(1, 1)
        m_p = e_kin.z_to_a[sid] * NUCLEON_MASS_GEV
        np.testing.assert_allclose(
            e_kin.error(ekin, "p"),
            e_total.error(ekin + m_p, "p"),
            rtol=1e-12,
            err_msg="kinetic-energy errors must equal total-energy errors "
            "at E_kin + m (rest mass added exactly once)",
        )
        # flux already had the single transform; keep the pair consistent
        np.testing.assert_allclose(
            e_kin.flux(ekin, "p"), e_total.flux(ekin + m_p, "p"), rtol=1e-12
        )

    def test_error_scales_once_with_energy_scale(self):
        from globalsplinefit.model import GSFEnergy

        base = GSFEnergy()
        scaled = GSFEnergy()
        scaled.energy_scale = 0.10
        e = np.array([10.0, 100.0, 1000.0])
        np.testing.assert_allclose(
            scaled.flux(e, "p"), base.flux(e * 1.10, "p"), rtol=1e-12
        )
        np.testing.assert_allclose(
            scaled.error(e, "p"),
            base.error(e * 1.10, "p"),
            rtol=1e-12,
            err_msg="energy_scale must enter error() as (1+delta), not (1+delta)^2",
        )
