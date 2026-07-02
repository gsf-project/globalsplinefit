"""Additional physics validation tests for GSFRigidity solar modulation.

These tests verify that the implementation follows known physical principles
and mathematical relationships for solar modulation.

Force field (Gleeson-Axford): total-energy loss Z*phi (phi in GV), dN/dR
prefactor Lambda = (E_IS/E) * (R/R_IS)**3. The rigidity shift R_IS - R ~ phi is
~charge-independent (the basis of a single "universal" modulation potential).
"""

import numpy as np

M_NUCLEON = 0.93891872965  # GeV


class TestGSFRigidityPhysicsValidation:
    """Physics-consistency checks for GSFRigidity."""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _force_field_reference(z, a, R, phi):
        """Return (R_IS, Lambda) from the analytic force-field formula."""
        m = a * M_NUCLEON
        E = np.sqrt((z * R) ** 2 + m**2)
        E_is = E + z * phi
        p2_is = np.clip(E_is**2 - m**2, 0.0, None)
        R_is = np.sqrt(p2_is) / z
        Λ = (E_is / E) * (np.abs(R) / (R_is + 1e-300)) ** 3
        return R_is, Λ

    # ------------------------------------------------------------------
    # tests
    # ------------------------------------------------------------------
    def test_force_field_approximation_consistency(self, gsf_rigidity):
        R = np.array([1.0, 5.0, 10.0])
        z, phi = 1, 0.5
        R_is_mod, Λ_mod = gsf_rigidity._rigidity_phi_transform(z, R, phi)
        a = gsf_rigidity.z_to_a[z]
        R_is_ref, Λ_ref = self._force_field_reference(z, a, R, phi)
        np.testing.assert_allclose(R_is_mod, R_is_ref, rtol=1e-12)
        np.testing.assert_allclose(Λ_mod, Λ_ref, rtol=1e-12)

    def test_energy_momentum_conservation(self, gsf_rigidity):
        R, z, phi = np.array([5.0, 10.0, 20.0]), 1, 0.3
        R_is, _ = gsf_rigidity._rigidity_phi_transform(z, R, phi)
        a = gsf_rigidity.z_to_a[z]
        m = a * M_NUCLEON
        p_is = z * R_is
        E_is = np.sqrt(p_is**2 + m**2)
        E = np.sqrt((z * R) ** 2 + m**2)
        E_is_direct = E + z * phi  # total-energy loss Z*phi
        np.testing.assert_allclose(E_is, E_is_direct, rtol=1e-12)

    def test_modulation_intensity_scaling(self, gsf_rigidity):
        R = np.array([2.0, 5.0, 10.0])
        phi_low, phi_high = 0.2, 0.6
        R_is_low, Λ_low = gsf_rigidity._rigidity_phi_transform(1, R, phi_low)
        R_is_high, Λ_high = gsf_rigidity._rigidity_phi_transform(1, R, phi_high)
        # stronger modulation shifts to higher interstellar rigidity ...
        assert np.all(R_is_high > R_is_low)
        # ... and suppresses more, so the dN/dR prefactor is smaller at low R.
        mask = R < 5.0
        if np.any(mask):
            assert np.all(Λ_high[mask] < Λ_low[mask])

    def test_charge_dependence(self, gsf_rigidity):
        # The rigidity shift R_IS - R ~ phi is ~charge-independent (universal),
        # in contrast to a per-nucleon (Z/A)*phi shift which would scale ~1/A.
        R, phi = np.array([5.0, 10.0, 20.0]), 0.4
        charges = [1, 2, 8, 26]
        shifts = []
        for z in charges:
            R_is, _ = gsf_rigidity._rigidity_phi_transform(z, R, phi)
            s = R_is - R
            # each species' shift is within 20% of phi (NOT ~phi/A)
            assert np.all(np.abs(s - phi) < 0.2 * phi)
            shifts.append(s)
        shifts = np.array(shifts)
        # the shift is ~universal across charges (spread well under 2x)
        assert np.all(shifts.max(axis=0) / shifts.min(axis=0) < 1.3)

    def test_relativistic_limits(self, gsf_rigidity):
        z, phi = 1, 0.2
        R = np.array([0.1, 0.5, 5.0, 100.0, 1000.0])
        a = gsf_rigidity.z_to_a[z]
        R_is_ref, Λ_ref = self._force_field_reference(z, a, R, phi)
        R_is_mod, Λ_mod = gsf_rigidity._rigidity_phi_transform(z, R, phi)
        np.testing.assert_allclose(R_is_mod, R_is_ref, rtol=1e-12)
        np.testing.assert_allclose(Λ_mod, Λ_ref, rtol=1e-12)
        ratio = R_is_ref / R
        assert np.all(np.diff(ratio) < 0)              # shift fades with rigidity
        assert ratio[0] > 1.0 and Λ_ref[-1] > 0.99     # -> LIS at high rigidity

    def test_flux_modulation_symmetry(self, gsf_rigidity):
        R = np.geomspace(1.0, 100.0, 40)
        groups = ["p", "He", "O*", "Fe*"]
        lis = {g: gsf_rigidity.flux(R, g) for g in groups}
        mod = {
            g: gsf_rigidity.flux(R, g, time_interval=(200901, 200912)) for g in groups
        }
        high = R > 50.0
        factors = [np.mean(mod[g][high] / lis[g][high]) for g in groups]
        if len(factors) > 1:
            rel_spread = (max(factors) - min(factors)) / np.mean(factors)
            assert rel_spread < 0.10

    def test_jacobian_transformation_consistency(self, gsf_rigidity):
        """Average over *all* φ values in the interval before comparison."""
        R_grid = np.geomspace(1, 100, 12)
        interval = (200901, 200912)
        jac_from_method = gsf_rigidity.jacobian(R_grid, "p", time_interval=interval)

        # Re-compute analytically, averaging over the same φ list
        z, leader, ratio = 1, 1, 1.0
        phis = gsf_rigidity._phi_list(interval)
        jac_manual = 0.0
        for phi in phis:
            R_is, Λ = gsf_rigidity._rigidity_phi_transform(z, R_grid, phi)
            jac_manual += (
                ratio * gsf_rigidity._rigidity_flux_jacobian(leader, R_is) * Λ[:, None]
            )
        jac_manual /= len(phis)

        np.testing.assert_allclose(jac_manual, jac_from_method, rtol=5e-5)
