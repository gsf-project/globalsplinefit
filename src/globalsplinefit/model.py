"""Main GSF model class and functionality.

This module contains the core Global Spline Fit (GSF) model classes for cosmic ray
flux calculations. The GSF model provides parametrizations of cosmic ray flux and
composition based on spline fits to observational data.

The module includes:
- GSFBase: Abstract base class with common functionality
- GSFEnergy: Model using total energy per nucleus as input
- GSFKineticEnergy: Model using kinetic energy per nucleus as input
- GSFRigidity: Model using magnetic rigidity as input
- GSFEnergyPerNucleon: Model for nucleon flux calculations
- GSFKineticEnergyPerNucleon: Model for nucleon flux using kinetic energy per nucleon

All models support the same target specification system for cosmic ray groups
and individual elements, with optional solar modulation for time-dependent calculations.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from scipy.interpolate import splev

from .data_management import SPLINE_DEGREE, Parameters, _collect_phi_values

# Physical constants
NUCLEON_MASS_GEV = 0.93891872965
"""Nucleon mass in GeV."""


_ZERO_GUARD = 1e-300
"""Small value to prevent division by zero in rigidity calculations."""

SUBLEADING_SAT_LNR = float(np.log(5.0e6))
"""ln(R/GV) above which the tilted sub-leading extrapolation saturates to a
constant ratio: ratio(R) = norm * (min(R, 5 PV)/Rmax)^slope. R_sat = 5 PV is
around the proton knee, motivated by the galactic origin and assumed
similarity of transport effects above it. Must match the fitter
(gsffit.model.flux_jax._SLOPE_XSAT)."""


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    x = np.clip(x, -500, 500)
    pos = x >= 0
    result = np.empty_like(x, dtype=float)
    result[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    result[~pos] = np.exp(x[~pos]) / (1.0 + np.exp(x[~pos]))
    return result


# Type alias for array-like inputs
ArrayLike = np.ndarray | list[float] | float
"""Type alias for inputs that can be scalars, lists, or numpy arrays."""


class GSFBase(ABC):
    """Base class for Global Spline Fit model for cosmic ray flux calculations.

    This abstract base class provides the common functionality for all GSF model variants.
    The GSF model uses spline interpolation to represent cosmic ray flux measurements
    and provides methods to calculate flux, uncertainties, and covariances for different
    cosmic ray groups and individual elements.

    The model supports four main element groups:
    - Protons (H): charge Z=1
    - Helium (He): charge Z=2
    - Oxygen group (O*): represents CNO elements
    - Iron group (Fe*): represents heavy elements

    Attributes
    ----------
        GROUP_NAMES: Dictionary mapping group names to charge numbers
        active_groups: List of active element groups ["p", "He", "O*", "Fe*"]
    """

    # Species (single-isotope) name mappings: name -> charge of the species.
    # Resolved against the loaded data in _resolve_z; model versions without
    # a separate species of that name reject the target with ValueError.
    SPECIES_NAMES = {
        "D": 1,
        "deuterium": 1,
    }

    # Group name mappings
    GROUP_NAMES = {
        "p": 1,
        "proton": 1,
        "protons": 1,
        "H": 1,
        "He": 2,
        "helium": 2,
        "alpha": 2,
        "O": 8,
        "oxygen": 8,
        "CNO": 8,
        "O*": 8,
        "Fe": 26,
        "iron": 26,
        "Fe*": 26,
    }

    # --- global energy-scale nuisance (carried as a model parameter) -----------
    # A single fractional shift applied to the INPUT energy at evaluation:
    # the model is read at E*(1+energy_scale). Default 0.0 -> exact no-op. Lets a
    # user explore the (data-insensitive) global scale that propagates from the
    # low-energy anchor up to high energy. Usage: ``model.energy_scale = delta``.
    # ``energy_scale_prior`` is the advisory 1-sigma fractional prior width.
    # NOTE: GSFRigidity is INTENTIONALLY not scaled — the rigidity/LIS model is
    # the fit's low-energy anchor, and a rigidity scaling is not an energy
    # scaling. The knob acts on the energy-based evaluators only (GSFEnergy,
    # GSFKineticEnergy, the per-nucleon variants).
    _energy_scale = 0.0
    energy_scale_prior = 0.10

    @property
    def energy_scale(self) -> float:
        """Global fractional energy-scale shift applied at evaluation (0 = none)."""
        return self._energy_scale

    @energy_scale.setter
    def energy_scale(self, value: float) -> None:
        self._energy_scale = float(value)

    def _scale_energy(self, e: np.ndarray) -> np.ndarray:
        """Apply the global energy-scale nuisance to an input energy/rigidity array."""
        return e * (1.0 + self._energy_scale) if self._energy_scale else e

    def __init__(
        self,
        data_path: str | Path | None = None,
        version: str | None = None,
        use_approximate_solar_cycle_average: bool = True,
        default_time_interval: tuple[int, int] | str | None = None,
        default_rigidity_cutoff: float | None = None,
        cutoff_width: float = 1.0,
    ):
        """Initialize GSF base model.

        Parameters
        ----------
        data_path
            Optional path to custom data files. If None, uses default
            data files included with the package.
        version
            Optional model version. Defaults to ``"2026"``, the promoted
            default: the mixture covering (an equal-weight combination of the
            Auger FD-2026 SIBYLL-2.3e and EPOS-LHC-R interpretations) with the
            Ghelfi-Maurin-Derome modulation potential. ``"2026-USO"`` is the
            one sanctioned alternative -- the same mixture fit with the Usoskin
            2017 potential -- and is how the solar-modulation systematic is
            gauged. ``"2026-UHE-S23e"`` and ``"2026-EPOS-LHCR"`` are the
            single-interpretation variants (Auger FD-2026 SIBYLL-2.3e and
            EPOS-LHC-R, respectively; GMD potential) for applications that
            need one hadronic model rather than the mixture
            band. ``"2025"``, ``"2019"`` and ``"2017"`` are superseded
            historical releases, kept only to reproduce older work; they are not
            alternatives to the current fit. See
            :data:`~globalsplinefit.data_management.MODEL_VERSIONS` and
            :func:`~globalsplinefit.data_management.version_info`. If specified,
            overrides data_path and uses the corresponding package data directory.
        use_approximate_solar_cycle_average
            If True (default), solar cycle averages
            are calculated approximately from average of monthly phi values.
            If False, averages are calculated explicitly by averaging monthly
            fluxes over the solar cycle.
        default_time_interval
            Default time period specification to use when not explicitly
            provided in method calls:
            - None: Solar Cycle 24 average (December 2008 to December 2019)
            - "LIS": Local Interstellar Spectrum (phi=0, no modulation)
            - tuple[int, int]: (start, end) in YYYYMM format.
            Default is None for Solar Cycle 24 average.
        default_rigidity_cutoff
            Default geomagnetic rigidity cutoff in GV.
            If None (default), no cutoff is applied. Nuclei with rigidity
            below this value are excluded from the nucleon flux summation.
        cutoff_width
            Width of the cutoff transition in GV. Default 1.0 (a smooth
            sigmoid transition modeling the geomagnetic penumbra); use 0.0
            for a sharp (Heaviside) cutoff.
        """
        self.params = Parameters(
            data_path, version, use_approximate_solar_cycle_average
        )

        # Store default time interval
        self.default_time_interval = default_time_interval

        # Store default rigidity cutoff
        self.default_rigidity_cutoff = default_rigidity_cutoff
        self.cutoff_width = cutoff_width

        # The RESOLVED version name, not the constructor argument: a model built
        # with no arguments reports the default it actually loaded rather than
        # None. None only when data_path points outside the packaged versions.
        self.version = self.params.version

        # Copy frequently used parameters for convenience. Species are keyed by
        # ``sid = (Z, A)`` so isotopes (e.g. D and p both at Z=1) coexist; for a
        # v1 (one-species-per-Z) model each charge maps to a single sid, so the
        # behaviour is identical.
        self.kx = self.params.kx                  # sid (+int charge alias) -> knots
        self.pars = self.params.pars              # sid (+int alias) -> coeffs
        self.npar = self.params.npar              # sid (+int alias) -> n coeffs
        self.z_group = self.params.z_group        # leader charge -> [member charges]
        self.z_to_a = self.params.z_to_a          # sid (+int alias) -> A
        self.z_to_sids = self.params.z_to_sids    # charge Z -> [sids]
        self.z_ungroup = self.params.z_ungroup    # charge -> leader charge
        self.cov = self.params.cov                # (sid,sid) (+int alias) -> block
        self.phi = self.params.phi
        self.flux_ratio = self.params.flux_ratio  # sid (+int alias) -> (leader_sid, ratio)
        self.flux_slope = self.params.flux_slope  # sid (+int alias) -> extrapolation slope
        self.species = self.params.species        # sorted list of all sids
        self._leaders = self.params._leaders      # set of leader sids
        self._leader_by_charge = self.params.leader_sid   # int charge -> leader_sid
        # Helpers
        self.active_groups = ["p", "He", "O*", "Fe*"]

        # Cache for _rigidity_flux_jacobian results
        self._jacobian_cache = {}
        self._cache_max_size = 1000  # Limit cache size to prevent memory issues

    def _as_sid(self, x):
        """Accept either a species id ``(Z, A)`` or a bare charge ``Z``. A bare
        charge resolves to the (unique, for single-species charges) species at
        that charge — preserving the integer-charge calling style used by callers
        and tests. Internal callers pass explicit sids for isotope disambiguation."""
        if isinstance(x, tuple):
            return x
        return self.z_to_sids[int(x)][0]

    def _make_cache_key(
        self, sid, rigidity: np.ndarray
    ) -> tuple:
        """Build a cache key from the species id (Z, A) and rigidity array."""
        sid = self._as_sid(sid)
        arr = np.atleast_1d(rigidity)
        return (sid, arr.shape, arr.dtype, arr.tobytes())

    def _manage_cache_size(self):
        """Keep cache size under control by removing oldest entries."""
        if len(self._jacobian_cache) >= self._cache_max_size:
            # Remove about 20% of oldest entries (simple FIFO strategy)
            items_to_remove = len(self._jacobian_cache) // 5
            for _ in range(items_to_remove):
                # Remove first (oldest) item
                oldest_key = next(iter(self._jacobian_cache))
                del self._jacobian_cache[oldest_key]

    def _resolve_z(
        self, target: str | int | tuple[int, float] | list[int]
    ) -> tuple[list, int]:
        """Resolve group/element/species specification to charges and group leader.

        Parameters
        ----------
        target
            Target specification, which can be:
            - String: Group name ("p", "proton", "H", "He", "helium", "alpha",
            "O", "oxygen", "CNO", "O*", "Fe", "iron", "Fe*") or species name
            ("D", "deuterium" — model versions carrying it as its own species)
            - Integer: Single element atomic number (e.g., 1 for hydrogen, 2 for helium)
            - Species id ``(Z, A)``: a single isotope, e.g. ``(1, 2.014)`` for D
            - List/array: Multiple element atomic numbers from the same group

        Returns
        -------
            Tuple of:
            - List of atomic numbers included in the target; a single-species
              target yields a one-element list holding the species id tuple
              instead of a bare charge (see ``_target_sids``)
            - Group leader atomic number (1 for H, 2 for He, 8 for O*, 26 for Fe*)

        Raises
        ------
            ValueError: If group name is unknown, element is not in GSF model,
            or multiple elements are from different groups.

        Examples
        --------
            >>> model._resolve_z("p")  # returns ([1], 1)
            >>> model._resolve_z(1)    # returns ([1], 1)
            >>> model._resolve_z([6, 7, 8])  # returns ([6, 7, 8], 8) for CNO group
            >>> model._resolve_z("D")  # returns ([(1, 2.014)], 1)
        """
        if isinstance(target, str):
            # String input -> resolve to group (returns member element charges)
            if target in self.GROUP_NAMES:
                group_leader = self.GROUP_NAMES[target]
                return list(self.z_group[group_leader]), group_leader
            elif target in self.SPECIES_NAMES:
                # Named single species: the non-leader isotope at its charge
                # (currently only D at Z=1; absent in one-species-per-charge
                # model versions).
                z = self.SPECIES_NAMES[target]
                leader_sid = self._leader_by_charge.get(z)
                sids = [s for s in self.z_to_sids.get(z, []) if s != leader_sid]
                if len(sids) != 1:
                    raise ValueError(
                        f"Species '{target}' is not carried by model version "
                        f"{self.version} - it has no separate isotope at Z={z}"
                    )
                return sids, self.z_ungroup[z]
            else:
                raise ValueError(
                    f"Unknown group name: {target} - valid names are "
                    f"{list(self.GROUP_NAMES.keys()) + list(self.SPECIES_NAMES.keys())}"
                )

        elif isinstance(target, tuple) and target in self.species:
            # Explicit species id (Z, A) -> that single isotope. Checked before
            # the generic tuple branch below, which reads a tuple as charges.
            return [target], self.z_ungroup[target[0]]

        elif isinstance(target, int | np.integer):
            # Single integer charge -> that element charge (its species are
            # expanded from the charge in the flux loops).
            z = int(target)
            if z not in self.z_ungroup:
                raise ValueError(f"Unknown element: {z} - not in GSF model")
            return [z], self.z_ungroup[z]

        elif isinstance(target, list | tuple | np.ndarray):
            # List/array of integer charges -> multiple elements (same group)
            zlist = [int(z) for z in target]
            unknown = [z for z in zlist if z not in self.z_ungroup]
            if unknown:
                raise ValueError(f"Unknown elements: {unknown} - not in GSF model")
            leaders = {self.z_ungroup[z] for z in zlist}
            if len(leaders) != 1:
                raise ValueError(
                    f"Multiple groups found for elements {zlist} - must be from the same group"
                )
            return zlist, self.z_ungroup[zlist[0]]

        else:
            raise ValueError(
                f"Target must be string (group), integer (element), or list of integers (elements). "
                f"Got {type(target)}"
            )

    def _target_sids(self, zlist: list) -> list:
        """Expand ``_resolve_z`` output to species ids.

        Bare charges expand to every species at that charge (p, D, … at Z=1);
        explicit species-id tuples pass through unchanged.
        """
        sids = []
        for zi in zlist:
            if isinstance(zi, tuple):
                sids.append(zi)
            else:
                sids.extend(self.z_to_sids[zi])
        return sids

    def _resolve_time_interval(
        self, time_interval: tuple[int, int] | str | None
    ) -> tuple[int, int] | str | None:
        """Resolve time interval, using default if None provided by user.

        Parameters
        ----------
        time_interval
            Time period specification provided by user, or None.

        Returns
        -------
            The user-provided time_interval if not None, otherwise the default.
        """
        if time_interval is None:
            return self.default_time_interval
        return time_interval

    def _resolve_rigidity_cutoff(self, rigidity_cutoff: float | None) -> float | None:
        """Resolve rigidity cutoff, using default if None provided by user."""
        if rigidity_cutoff is None:
            return self.default_rigidity_cutoff
        return rigidity_cutoff

    def _rigidity_cutoff_mask(
        self, sid, energy: np.ndarray, rigidity_cutoff: float | None
    ) -> np.ndarray:
        """Return mask for geomagnetic rigidity cutoff.

        Parameters
        ----------
        z
            Atomic number.
        energy
            Total energy per nucleus in GeV (at Earth).
        rigidity_cutoff
            Cutoff rigidity in GV, or None for no cutoff.

        Returns
        -------
            Float array with same shape as energy. Values are 0.0/1.0 for sharp
            cutoff (cutoff_width=0), or smooth sigmoid values in [0, 1] when
            cutoff_width > 0.
        """
        if rigidity_cutoff is None:
            return np.ones_like(energy)
        sid = self._as_sid(sid)
        z = sid[0]
        mass = self.z_to_a[sid] * NUCLEON_MASS_GEV
        p2 = np.maximum(energy**2 - mass**2, 0.0)
        rig = np.sqrt(p2) / z

        if self.cutoff_width > 0:
            # Smooth sigmoid transition (models geomagnetic penumbra)
            x = (rig - rigidity_cutoff) / self.cutoff_width
            return _sigmoid(x)

        return (rig >= rigidity_cutoff).astype(float)

    def _spline(self, sid, x: np.ndarray) -> np.ndarray:
        """Evaluate spline for species ``sid=(Z, A)`` at log rigidity x.

        Returns the spline-evaluated flux with the (R/GV)^-3 power-law factor.
        """
        sid = self._as_sid(sid)
        return splev(x, (self.kx[sid], self.pars[sid], SPLINE_DEGREE)) * np.exp(-3.0 * x)

    def _propagate_cov(
        self, j1: np.ndarray, j2: np.ndarray, c: np.ndarray
    ) -> np.ndarray:
        """Compute J1 @ C @ J2.T for covariance propagation."""
        return np.linalg.multi_dot((j1, c, j2.T))

    def _phi_list(self, time_interval: tuple[int, int] | str | None) -> np.ndarray:
        """Get list of solar modulation parameters for time interval.

        Parameters
        ----------
        time_interval
            Time period specification:
            - None: Solar Cycle 24 average (December 2008 to December 2019) —
              the default (restored 2026-07-17; comparisons against
              modulation-unaware models expect a flux at Earth, not the LIS)
            - "LIS": Local Interstellar Spectrum (phi=0, no modulation)
            - tuple[int, int]: (start, end) in YYYYMM format; the end month
            is EXCLUSIVE. Example: (200901, 201001) for Jan-Dec 2009.
            Monthly phi values of an interval are reduced to at most 12
            representative values (mean-preserving), matching the fitter's
            12-bin period average.

        Returns
        -------
            Array of solar modulation potential values (in GV) for the time period.

        Raises
        ------
            ValueError: If start equals end, or start > end in time interval tuple.
        """
        if time_interval is None:
            # Default: Solar Cycle 24 average (December 2008 to December 2019)
            if self.params.use_approximate_solar_cycle_average:
                # Approximate averaging: use a single average phi value
                phi_avg = self.params.get_solar_cycle_24_phi_average()
                return np.array([phi_avg])
            # Explicit averaging: fall through with the SC24 interval
            time_interval = self.params.get_solar_cycle_24_interval()
        elif time_interval == "LIS":
            # Local Interstellar Spectrum: no solar modulation
            return np.array([0.0])
        elif isinstance(time_interval, str):
            raise ValueError(
                f"Invalid string time_interval '{time_interval}'. Only 'LIS' is supported."
            )

        t_a, t_b = time_interval
        if t_a == t_b:
            raise ValueError("Time interval start and end cannot be the same")
        if t_a > t_b:
            raise ValueError("Time interval start must be less than end")
        phis = _collect_phi_values(self.phi, t_a, t_b)
        # Reduce long monthly lists to 12 representative phi values (equal-count
        # chunks of the sorted list, chunk means, shifted to preserve the full
        # monthly mean exactly), mirroring the fitter's 12-bin period average.
        # Downstream averages weight each phi equally, so this keeps the
        # period-averaged flux within ~0.1% of the full monthly sum.
        if len(phis) > 12:
            chunk_means = np.array(
                [c.mean() for c in np.array_split(np.sort(phis), 12)]
            )
            phis = chunk_means + (np.mean(phis) - chunk_means.mean())
        return phis

    def _rigidity_flux_lis(self, sid, rigidity: ArrayLike) -> np.ndarray:
        """Calculate LIS flux of species ``sid=(Z, A)`` as a function of rigidity."""
        sid = self._as_sid(sid)
        rigidity = np.atleast_1d(rigidity)

        with np.errstate(divide="ignore", invalid="ignore"):
            log_rigidity = np.log(rigidity)

        # Get valid rigidity range for this species
        min_log_rigidity = self.kx[sid][0]
        max_log_rigidity = self.kx[sid][-1]

        # Initialize result array
        result = np.zeros_like(log_rigidity)

        # Only calculate flux for valid rigidity range
        valid_mask = log_rigidity >= min_log_rigidity
        if not np.any(valid_mask):
            return result

        valid_log_rigidity = log_rigidity[valid_mask]

        if sid in self._leaders:  # Leading species (its own group leader)
            result[valid_mask] = self._spline(sid, valid_log_rigidity)
        else:  # Subleading species
            leading, ratio = self.flux_ratio[sid]

            # Split into regions: within spline range vs extrapolation region
            within_range = valid_log_rigidity <= max_log_rigidity
            extrapolation = ~within_range

            if np.any(within_range):
                # Use species' own spline within its range
                within_indices = valid_mask.copy()
                within_indices[valid_mask] = within_range
                result[within_indices] = self._spline(
                    sid, valid_log_rigidity[within_range]
                )

            if np.any(extrapolation):
                # Group leader's spline scaled by the ratio at xmax, tilted by
                # the stored power-law slope and SATURATING at R_sat:
                # ratio(R>Rmax) = ratio * (min(R, R_sat)/Rmax)^s.
                # slope == 0 (pre-slope bundles) reproduces the historical
                # constant-ratio extrapolation bit-identically.
                extrap_indices = valid_mask.copy()
                extrap_indices[valid_mask] = extrapolation
                x_ex = valid_log_rigidity[extrapolation]
                result[extrap_indices] = ratio * self._spline(leading, x_ex)
                slope = self.flux_slope[sid]
                if slope != 0.0:
                    dx_sat = max(SUBLEADING_SAT_LNR - max_log_rigidity, 0.0)
                    result[extrap_indices] *= np.exp(slope * np.clip(
                        x_ex - max_log_rigidity, 0.0, dx_sat))

        return result

    def _element_flux(
        self,
        sid,
        energy: ArrayLike,
        time_interval: tuple[int, int] | str | None = None,
    ) -> np.ndarray:
        """Calculate flux for species ``sid=(Z, A)``."""
        sid = self._as_sid(sid)
        energy = np.atleast_1d(energy)
        time_interval = self._resolve_time_interval(time_interval)
        phis = np.array(self._phi_list(time_interval))

        # Get vectorized rigidity and factors for all phi values
        # Shape: [n_energy, n_phi]
        rigidity, factor = self._rigidity_from_energy_vectorized(sid, energy, phis)

        # Calculate LIS flux for all rigidities at once
        # Reshape rigidity to 1D for LIS calculation, then reshape back
        rig_flat = rigidity.flatten()
        lis_flux_flat = self._rigidity_flux_lis(sid, rig_flat)
        lis_flux = lis_flux_flat.reshape(rigidity.shape)  # [n_energy, n_phi]

        # Apply factors and average over phi dimension
        modulated_flux = lis_flux * factor  # [n_energy, n_phi]
        averaged_flux = np.mean(modulated_flux, axis=1)  # [n_energy]

        return averaged_flux

    def _element_flux_jacobian(
        self,
        sid,
        energy: ArrayLike,
        time_interval: tuple[int, int] | str | None = None,
    ) -> np.ndarray:
        """Calculate Jacobian of flux for uncertainty propagation."""
        sid = self._as_sid(sid)
        energy = np.atleast_1d(energy)
        leading, ratio = self.flux_ratio[sid]
        time_interval = self._resolve_time_interval(time_interval)
        phis = np.array(self._phi_list(time_interval))

        # Get vectorized rigidity and factors
        rigidity, factor = self._rigidity_from_energy_vectorized(sid, energy, phis)

        # Calculate Jacobian for all rigidities
        # This requires careful reshaping to handle the parameter dimension
        rig_flat = rigidity.flatten()
        jac_flat = self._rigidity_flux_jacobian(
            leading, rig_flat
        )  # [n_energy*n_phi, n_params]

        # Reshape to [n_energy, n_phi, n_params]
        jac = jac_flat.reshape(energy.shape[0], len(phis), -1)

        # Apply factors (broadcasting over parameter dimension); above the
        # sub-leading species' top knot the extrapolation carries the stored
        # power-law tilt (clamped at xmax: below it the historical
        # ratio*leader-Jacobian approximation is unchanged).
        slope = self.flux_slope[sid]
        if slope != 0.0:
            xmax = self.kx[sid][-1]
            dx_sat = max(SUBLEADING_SAT_LNR - xmax, 0.0)
            with np.errstate(divide="ignore"):
                tilt = np.exp(
                    slope * np.clip(np.log(rigidity) - xmax, 0.0, dx_sat))
            factor = factor * tilt
        jac_weighted = jac * factor[:, :, np.newaxis]

        # Average over phi dimension
        jac_averaged = np.mean(jac_weighted, axis=1)  # [n_energy, n_params]

        return ratio * jac_averaged

    def _rigidity_from_energy_vectorized(
        self, sid, energy: np.ndarray, phis: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized version that handles multiple phi values at once."""
        sid = self._as_sid(sid)
        # Convert inputs to proper shapes for broadcasting
        energy = np.atleast_1d(energy)
        phis = np.atleast_1d(phis)

        # Reshape for broadcasting: energy[n_energy, 1], phis[1, n_phi]
        energy_bc = energy[:, np.newaxis]
        phis_bc = phis[np.newaxis, :]

        # Get atomic mass and nucleon mass; Z is the charge (sid = (Z, A))
        z = sid[0]
        nucleon_mass = NUCLEON_MASS_GEV
        mass = self.z_to_a[sid] * nucleon_mass

        # Apply solar modulation to energy
        energy_is = energy_bc + z * phis_bc

        # Check for invalid energies (below rest mass)
        invalid_mask = energy_is <= mass

        # Handle potential division by zero or invalid values
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = (energy_bc**2 - mass**2) / (energy_is**2 - mass**2)

        p2 = energy_is**2 - mass**2
        p2[p2 < 0] = 0.0
        rigidity = p2**0.5 / z

        # Handle potential division by zero or invalid values
        with np.errstate(divide="ignore", invalid="ignore"):
            factor *= energy_is / (rigidity * z**2 + _ZERO_GUARD)

        # Set invalid results to zero (when energy is too low or factor is negative)
        factor[invalid_mask | (factor < 0) | ~np.isfinite(factor)] = 0.0
        rigidity[invalid_mask | ~np.isfinite(rigidity)] = 0.0

        return rigidity, factor

    def _rigidity_flux_jacobian(self, sid, rigidity: np.ndarray) -> np.ndarray:
        """Calculate Jacobian of LIS flux for the (leader) species ``sid``.

        This method is cached to improve performance for repeated calls with
        the same parameters.
        """
        sid = self._as_sid(sid)
        cache_key = self._make_cache_key(sid, rigidity)
        if cache_key in self._jacobian_cache:
            return self._jacobian_cache[cache_key].copy()

        self._manage_cache_size()
        result = self._compute_rigidity_flux_jacobian(sid, rigidity)
        self._jacobian_cache[cache_key] = result.copy()
        return result

    def _compute_rigidity_flux_jacobian(
        self, sid, rigidity: np.ndarray
    ) -> np.ndarray:
        """Compute Jacobian of LIS flux for species ``sid`` (uncached).

        This is the original implementation extracted to a separate method
        to maintain clean separation between caching logic and computation.
        """
        sid = self._as_sid(sid)
        with np.errstate(divide="ignore"):
            x = np.log(rigidity)

        jac = np.zeros((len(x), self.npar[sid]))
        pi = np.zeros(self.npar[sid] + 4)  # splev needs 4 extra zeros

        for ipar in range(self.npar[sid]):
            pi[ipar] = 1.0
            v = splev(x, (self.kx[sid], pi, SPLINE_DEGREE))
            v[x < self.kx[sid][0]] = 0.0
            jac[:, ipar] = v
            pi[ipar] = 0.0

        x[x < self.kx[sid][0]] = 0.0
        jac *= np.exp(-3.0 * x)[:, np.newaxis]

        return jac

    def total_flux(
        self,
        energy_or_rigidity: ArrayLike,
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Calculate total flux from all element groups.

        Computes the sum of flux from all active cosmic ray groups:
        protons (p), helium (He), oxygen group (O*), and iron group (Fe*).

        Parameters
        ----------
        energy_or_rigidity
            Input energy or rigidity values. Units depend on subclass:
            - GSFEnergy: Total energy per nucleus in GeV
            - GSFRigidity: Rigidity in GV
            - GSFEnergyPerNucleon: Energy per nucleon in GeV
        time_interval
            Time period specification:
            - None: Uses the default_time_interval set during initialization
            - "LIS": Local Interstellar Spectrum (no modulation)
            - tuple: (start, end) in YYYYMM format, end month EXCLUSIVE, e.g. (200901, 201001) for calendar year 2009
        rigidity_cutoff
            Geomagnetic rigidity cutoff in GV. Nuclei with rigidity
            below this value are excluded. None uses the default.

        Returns
        -------
            Array of total cosmic ray flux values. Units are particles/(m²·s·sr·GeV)
            for energy models, particles/(m²·s·sr·GV) for rigidity model.
            For GSFEnergyPerNucleon and GSFKineticEnergyPerNucleon, the
            proton and neutron contributions are summed; use
            ``p_and_n_total_flux`` to get them separately as shape (2, N).
        """
        energy_or_rigidity = np.atleast_1d(energy_or_rigidity)
        total_flux = np.zeros_like(energy_or_rigidity, dtype=float)
        for group in self.active_groups:
            total_flux += self.flux(
                energy_or_rigidity,
                group,
                time_interval=time_interval,
                rigidity_cutoff=rigidity_cutoff,
            )

        return total_flux

    def total_error(
        self,
        energy_or_rigidity: ArrayLike,
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Calculate uncertainty of total flux.

        Computes the standard deviation of the total flux from all element groups,
        accounting for correlations between groups through the covariance matrix.

        Parameters
        ----------
        energy_or_rigidity
            Input energy or rigidity values. Units depend on subclass.
        time_interval
            Time period specification:
            - None: Uses the default_time_interval set during initialization
            - "LIS": Local Interstellar Spectrum (no modulation)
            - tuple: (start, end) in YYYYMM format
        rigidity_cutoff
            Geomagnetic rigidity cutoff in GV. Nuclei with rigidity
            below this value are excluded. None uses the default.

        Returns
        -------
            Array of total flux uncertainties (1-sigma).
        """
        energy_or_rigidity = np.atleast_1d(energy_or_rigidity)
        n = len(energy_or_rigidity)
        total_cov = np.zeros((n, n), dtype=float)

        for l1 in self.active_groups:
            for l2 in self.active_groups:
                total_cov += self.covariance(
                    l1,
                    l2,
                    energy_or_rigidity,
                    time_interval=time_interval,
                    rigidity_cutoff=rigidity_cutoff,
                )

        return np.sqrt(np.diag(total_cov))

    # ------------------------------------------------------------------
    # Composition helpers (derived from per-group / per-element flux).
    # Added so plotting/comparison can be built on the package natively.
    # ------------------------------------------------------------------
    def fraction(
        self,
        energy_or_rigidity: ArrayLike,
        target: str | int | list[int],
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Flux fraction of ``target`` relative to the all-particle total."""
        tot = self.total_flux(
            energy_or_rigidity,
            time_interval=time_interval,
            rigidity_cutoff=rigidity_cutoff,
        )
        grp = self.flux(
            energy_or_rigidity,
            target,
            time_interval=time_interval,
            rigidity_cutoff=rigidity_cutoff,
        )
        return grp / np.where(tot > 0, tot, np.nan)

    def _flux_one_species(
        self, energy_or_rigidity, sid, *, time_interval=None, rigidity_cutoff=None
    ):
        """Flux of a single species ``sid=(Z, A)``.

        Equivalent to ``flux()`` with an explicit species-id target: isolates
        one isotope of a multi-species charge (e.g. D vs p at Z=1).
        """
        return self.flux(
            energy_or_rigidity, sid,
            time_interval=time_interval, rigidity_cutoff=rigidity_cutoff,
        )

    def _element_lnA_fluxes(
        self, energy_or_rigidity, *, time_interval=None, rigidity_cutoff=None
    ):
        """(ln A, per-species flux array, species-id list) over every species."""
        sids = self.species
        fl = np.array(
            [
                self._flux_one_species(
                    energy_or_rigidity, sid,
                    time_interval=time_interval, rigidity_cutoff=rigidity_cutoff,
                )
                for sid in sids
            ]
        )
        lnA = np.log(np.array([sid[1] for sid in sids]))
        return lnA, fl, sids

    def _lnA_moments(self, energy_or_rigidity, time_interval, rigidity_cutoff):
        """(num1=Σf·lnA, num2=Σf·lnA², den=Σf) over all species."""
        lnA, fl, sids = self._element_lnA_fluxes(
            energy_or_rigidity,
            time_interval=time_interval, rigidity_cutoff=rigidity_cutoff,
        )
        num1 = (fl * lnA[:, None]).sum(0)
        num2 = (fl * (lnA**2)[:, None]).sum(0)
        den = fl.sum(0)
        return num1, num2, den

    def mean_lnA(
        self,
        energy_or_rigidity: ArrayLike,
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Flux-weighted mean of ln A over all species."""
        num1, _num2, den = self._lnA_moments(
            energy_or_rigidity, time_interval, rigidity_cutoff)
        return num1 / np.where(den > 0, den, np.nan)

    def var_lnA(
        self,
        energy_or_rigidity: ArrayLike,
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Flux-weighted variance of ln A over all species."""
        num1, num2, den = self._lnA_moments(
            energy_or_rigidity, time_interval, rigidity_cutoff)
        denw = np.where(den > 0, den, np.nan)
        m = num1 / denw
        return num2 / denw - m**2

    # --- covariance propagation for DIMENSIONLESS derived quantities -------
    def _leading_param_cov(self):
        """Assemble the dense leading-parameter covariance and the per-leader
        (sid, n_coeff, col0) layout from the sparse per-block ``self.cov``."""
        leads = sorted(self._leaders, key=lambda s: s[0])
        layout, col = [], 0
        for sid in leads:
            n = self.npar[sid]
            layout.append((sid, n, col))
            col += n
        cov = np.zeros((col, col))
        for si, ni, ci in layout:
            for sj, nj, cj in layout:
                blk = self.cov.get((si, sj))
                if blk is not None:
                    cov[ci:ci + ni, cj:cj + nj] = np.asarray(blk)[:ni, :nj]
        return cov, layout

    def _scalar_error(self, fn, x):
        """1-sigma error of a scalar field ``fn(x)`` (e.g. mean_lnA, fraction):
        finite-difference fn w.r.t. each LEADING spline coefficient, propagate the
        stored leading covariance. fn must read the current ``self.pars``."""
        cov, layout = self._leading_param_cov()
        base = np.atleast_1d(np.asarray(fn(x), float))
        J = np.zeros((base.shape[0], cov.shape[0]))
        for sid, n, col0 in layout:
            coeffs = self.pars[sid]
            for k in range(n):
                old = coeffs[k]
                h = 1e-6 * (abs(old) if old != 0.0 else 1.0)
                coeffs[k] = old + h
                try:
                    plus = np.atleast_1d(np.asarray(fn(x), float))
                finally:
                    coeffs[k] = old
                J[:, col0 + k] = (plus - base) / h
        var = np.einsum("ei,ij,ej->e", J, cov, J)
        return np.sqrt(np.clip(var, 0.0, None))

    def mean_lnA_error(self, x, **kw):
        """1-sigma band of <lnA> from the stored covariance."""
        return self._scalar_error(lambda y: self.mean_lnA(y, **kw), x)

    def var_lnA_error(self, x, **kw):
        return self._scalar_error(lambda y: self.var_lnA(y, **kw), x)

    def fraction_error(self, x, target, **kw):
        """1-sigma band of a group's flux fraction from the stored covariance."""
        return self._scalar_error(lambda y: self.fraction(y, target, **kw), x)

    def error(
        self,
        energy_or_rigidity: ArrayLike,
        target: str | int | list[int],
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Calculate uncertainty for specific target (group or elements).

        Parameters
        ----------
        energy_or_rigidity
            Input energy or rigidity values. Units depend on subclass.
        target
            Target specification (see _resolve_z for details):
            - String: Group name ("p", "He", "O*", "Fe*", etc.) or species
              name ("D")
            - Integer: Single element atomic number
            - Species id ``(Z, A)``: a single isotope
            - List: Multiple element atomic numbers from same group
        time_interval
            Time period specification:
            - None: Uses the default_time_interval set during initialization
            - "LIS": Local Interstellar Spectrum (no modulation)
            - tuple: (start, end) in YYYYMM format
        rigidity_cutoff
            Geomagnetic rigidity cutoff in GV. Nuclei with rigidity
            below this value are excluded. None uses the default.

        Returns
        -------
            Array of flux uncertainties (1-sigma) for the specified target.
        """
        cov = self.covariance(
            target,
            target,
            energy_or_rigidity,
            time_interval=time_interval,
            rigidity_cutoff=rigidity_cutoff,
        )
        return np.sqrt(np.diag(cov))

    @abstractmethod
    def flux(
        self,
        energy_or_rigidity: ArrayLike,
        target: str | int | list[int],
        *,
        time_interval: tuple[int, int] | str | None = None,
    ) -> np.ndarray:
        """Calculate flux for target (group or elements).

        Parameters
        ----------
        energy_or_rigidity
            Input energy or rigidity values. Units depend on subclass.
        target
            Target specification (see _resolve_z for details).
        time_interval
            Optional time period as (start, end) in YYYYMM format.

        Returns
        -------
            Array of flux values for the specified target.
        """
        pass

    @abstractmethod
    def jacobian(
        self,
        energy_or_rigidity: ArrayLike,
        target: str | int | list[int],
        *,
        time_interval: tuple[int, int] | str | None = None,
    ) -> np.ndarray:
        """Calculate Jacobian of flux for uncertainty propagation.

        Parameters
        ----------
        energy_or_rigidity
            Input energy or rigidity values.
        target
            Target specification.
        time_interval
            Optional time period as (start, end) in YYYYMM format.

        Returns
        -------
            Jacobian matrix for uncertainty propagation.
        """
        pass

    @abstractmethod
    def covariance(
        self,
        target1: str | int | list[int],
        target2: str | int | list[int],
        energy_or_rigidity: ArrayLike,
        *,
        time_interval: tuple[int, int] | str | None = None,
    ) -> np.ndarray:
        """Calculate covariance matrix of flux.

        Parameters
        ----------
        target1
            First target specification.
        target2
            Second target specification.
        energy_or_rigidity
            Input energy or rigidity values.
        time_interval
            Optional time period as (start, end) in YYYYMM format.

        Returns
        -------
            Covariance matrix between the two targets.
        """
        pass


class GSFEnergy(GSFBase):
    """GSF model expecting total energy per nucleus in GeV.

    This class implements the GSF model for cosmic ray calculations using
    total energy per nucleus as input. The energy should be the total
    kinetic + rest mass energy of the nucleus.

    Examples
    --------
        >>> from globalsplinefit import GSFEnergy
        >>> import numpy as np
        >>> model = GSFEnergy()
        >>> energy = np.logspace(0, 3, 100)  # 1 GeV to 1 TeV
        >>> proton_flux = model.flux(energy, "p")
        >>> he_flux = model.flux(energy, "He")
        >>> total_flux = model.total_flux(energy)
    """

    def _transform_energy(
        self,
        energy: ArrayLike,
        target: str | int | list[int],  # noqa: ARG002
    ) -> np.ndarray:
        """Transform input energy to total energy per nucleus.

        Subclasses override this to convert from kinetic energy, etc.
        """
        return self._scale_energy(np.atleast_1d(energy).astype(float))

    def flux(
        self,
        energy: ArrayLike,
        target: str | int | list[int],
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Calculate flux for target (group or elements).

        Parameters
        ----------
        energy
            Total energy per nucleus in GeV. Can be scalar, list, or numpy array.
        target
            Target specification:
            - String: Group name ("p", "proton", "H", "He", "O*", "Fe*", etc.)
              or species name ("D" for deuterium, in model versions that
              carry it as its own species)
            - Integer: Single element atomic number (e.g., 1 for H, 2 for He)
            - Species id ``(Z, A)``: a single isotope, e.g. ``(1, 2.014)``
            - List: Multiple elements from same group (e.g., [6,7,8] for CNO)
        time_interval
            Time period specification:
            - None: Uses the default_time_interval set during initialization
            - "LIS": Local Interstellar Spectrum (no modulation)
            - tuple: (start, end) in YYYYMM format, end month EXCLUSIVE, e.g. (200901, 201001) for calendar year 2009
        rigidity_cutoff
            Geomagnetic rigidity cutoff in GV. Nuclei with rigidity
            below this value are excluded. None uses the default.

        Returns
        -------
            Array of differential flux values in units of particles/(m²·s·sr·GeV).
            Shape matches the input energy array.

        Examples
        --------
            >>> flux_p = model.flux([1, 10, 100], "p")  # proton flux at 1, 10, 100 GeV
            >>> flux_he = model.flux(energy_array, "He")  # helium flux
            >>> flux_cno = model.flux(energy_array, [6, 7, 8])  # combined CNO flux
            >>> flux_lis = model.flux(energy_array, "p", time_interval="LIS")  # LIS flux
        """
        rigidity_cutoff = self._resolve_rigidity_cutoff(rigidity_cutoff)
        zlist, group_leader = self._resolve_z(target)
        energy = self._transform_energy(energy, target)

        flux = np.zeros_like(energy, dtype=float)
        for sid in self._target_sids(zlist):   # charges expand to species (p, D, …)
            mask = self._rigidity_cutoff_mask(sid, energy, rigidity_cutoff)
            flux += self._element_flux(sid, energy, time_interval) * mask
        return flux

    def jacobian(
        self,
        energy: ArrayLike,
        target: str | int | list[int],
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Calculate Jacobian of flux for uncertainty propagation."""
        rigidity_cutoff = self._resolve_rigidity_cutoff(rigidity_cutoff)
        zlist, group_leader = self._resolve_z(target)
        energy = self._transform_energy(energy, target)

        jac = 0.0
        for sid in self._target_sids(zlist):
            mask = self._rigidity_cutoff_mask(sid, energy, rigidity_cutoff)
            jac += (
                self._element_flux_jacobian(sid, energy, time_interval)
                * mask[:, np.newaxis]
            )
        return np.asarray(jac)

    def covariance(
        self,
        target1: str | int | list[int],
        target2: str | int | list[int],
        energy: ArrayLike,
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Calculate covariance matrix of flux."""
        zlist1, leader1 = self._resolve_z(target1)
        zlist2, leader2 = self._resolve_z(target2)
        # NOTE: pass the RAW input through — jacobian() applies
        # _transform_energy itself. Transforming here too double-applies the
        # kinetic->total rest-mass shift (GSFKineticEnergy) and squares the
        # energy_scale factor.

        # Calculate total jacobian for each group (sum over all elements)
        jac1 = self.jacobian(
            energy,
            target1,
            time_interval=time_interval,
            rigidity_cutoff=rigidity_cutoff,
        )
        jac2 = jac1
        if zlist2 != zlist1:
            jac2 = self.jacobian(
                energy,
                target2,
                time_interval=time_interval,
                rigidity_cutoff=rigidity_cutoff,
            )

        # Use group leaders for covariance matrix lookup
        # key the covariance by the leader SPECIES id (Z, A). _resolve_z returns a
        # charge; a charge with a single species has a cov alias under the bare int,
        # but a charge carrying >1 species (p + D at Z=1) does not — so resolve to
        # the leader sid, which is always a real cov key.
        cov_key = (self._leader_by_charge.get(leader1, leader1),
                   self._leader_by_charge.get(leader2, leader2))

        # Check if covariance matrix entry exists
        if cov_key in self.cov:
            return self._propagate_cov(jac1, jac2, self.cov[cov_key])
        else:
            n_energies = len(np.atleast_1d(energy))
            return np.zeros((n_energies, n_energies))


class GSFKineticEnergy(GSFEnergy):
    """GSF model expecting kinetic energy per nucleus in GeV.

    Converts kinetic energy to total energy (kinetic + rest mass) internally
    before using the GSFEnergy calculation methods.

    Examples
    --------
        >>> from globalsplinefit import GSFKineticEnergy
        >>> import numpy as np
        >>> model = GSFKineticEnergy()
        >>> kinetic_energy = np.logspace(0, 3, 100)  # 1 GeV to 1 TeV kinetic energy
        >>> proton_flux = model.flux(kinetic_energy, "p")
        >>> he_flux = model.flux(kinetic_energy, "He")
        >>> total_flux = model.total_flux(kinetic_energy)
    """

    def _transform_energy(
        self, kinetic_energy: ArrayLike, target: str | int | list[int]
    ) -> np.ndarray:
        """Convert kinetic energy per nucleus to total energy per nucleus."""
        kinetic_energy = self._scale_energy(np.atleast_1d(kinetic_energy).astype(float))
        zlist, leader = self._resolve_z(target)
        if len(zlist) == 1 and isinstance(zlist[0], tuple):
            # Explicit single-species target ("D", (1, 2.014)): its own mass.
            sid = zlist[0]
        else:
            # leader is a charge; resolve to its species id (z, a) so a
            # multi-species charge (p + D at Z=1) — which has no bare-int
            # z_to_a alias — still works.
            sid = self._leader_by_charge.get(leader, leader)
        rest_mass = self.z_to_a[sid] * NUCLEON_MASS_GEV
        return kinetic_energy + rest_mass


class GSFRigidity(GSFBase):
    """GSF model expecting rigidity in GV.

    This class implements the GSF model for cosmic ray calculations using
    magnetic rigidity as input parameter. Rigidity is defined as
    R = pc/Z where p is momentum, c is speed of light, and Z is charge.

    The model supports both Local Interstellar Spectrum (LIS) calculations
    and solar modulation effects using the force-field approximation to
    transform between Earth and interstellar rigidity spectra.

    Note: the global ``energy_scale`` parameter is intentionally a no-op on
    this class — the rigidity/LIS model is the fit's low-energy anchor, and
    a rigidity scaling is not an energy scaling.

    Examples
    --------
        >>> from globalsplinefit import GSFRigidity
        >>> import numpy as np
        >>> model = GSFRigidity()
        >>> rigidity = np.logspace(0, 3, 100)  # 1 GV to 1 TV
        >>> proton_flux_lis = model.flux(rigidity, "p")  # LIS
        >>> proton_flux_2009 = model.flux(rigidity, "p", time_interval=(200901, 201001))
        >>> total_flux = model.total_flux(rigidity)
    """

    # ---------- new helpers ----------
    def _rigidity_phi_transform(
        self, sid, rigidity: np.ndarray, phi: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert Earth rigidity to interstellar rigidity under force-field phi.

        Force field: total-energy loss Z*phi (Gleeson-Axford). Returns R_IS and the
        dN/dR prefactor Lambda = (E_IS/E) * (R/R_IS)**3.
        """
        sid = self._as_sid(sid)
        nucleon_mass = NUCLEON_MASS_GEV
        z = sid[0]
        a = self.z_to_a[sid]
        m = a * nucleon_mass

        # Earth energy from input R; interstellar energy after Z*phi loss
        E = np.sqrt((z * rigidity) ** 2 + m**2)
        E_is = E + z * phi

        # Interstellar rigidity
        with np.errstate(divide="ignore", invalid="ignore"):
            p2_is = E_is**2 - m**2
            p2_is[p2_is < 0] = 0.0
            R_is = np.sqrt(p2_is) / z

            # dN/dR force-field prefactor (abs keeps unphysical R<0 non-negative)
            Lambda = (E_is / E) * (np.abs(rigidity) / (R_is + _ZERO_GUARD)) ** 3
        return R_is, Lambda

    # ---------- public API ----------
    def flux(
        self,
        rigidity: ArrayLike,
        target: str | int | list[int],
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Calculate flux for target (group or elements).

        Parameters
        ----------
        rigidity
            Magnetic rigidity in GV. Can be scalar, list, or numpy array.
        target
            Target specification (same as GSFEnergy.flux).
        time_interval
            Time period specification:
            - None: Uses the default_time_interval set during initialization
            - "LIS": Local Interstellar Spectrum (no modulation)
            - tuple: (start, end) in YYYYMM format, end month EXCLUSIVE, e.g. (200901, 201001) for calendar year 2009
        rigidity_cutoff
            Geomagnetic rigidity cutoff in GV. Flux at rigidities
            below this value is set to zero. None uses the default.

        Returns
        -------
            Array of differential flux values in units of particles/(m²·s·sr·GV).
            Shape matches the input rigidity array.

        Notes
        -----
            Solar modulation is now supported for rigidity-based calculations.
            The transformation uses the force-field approximation to convert
            between Earth and interstellar rigidity spectra.
        """
        time_interval = self._resolve_time_interval(time_interval)
        rigidity_cutoff = self._resolve_rigidity_cutoff(rigidity_cutoff)
        zlist, _ = self._resolve_z(target)
        rigidity = np.atleast_1d(rigidity)

        # φ list (length 1 with 0.0 for LIS)
        phis = self._phi_list(time_interval)

        flux = np.zeros_like(rigidity, dtype=float)
        for phi in phis:  # loop version (safe, readable)
            for sid in self._target_sids(zlist):   # charges expand to species (p, D, …)
                if phi == 0.0:
                    flux += self._rigidity_flux_lis(sid, rigidity)
                else:
                    R_is, fac = self._rigidity_phi_transform(sid, rigidity, phi)
                    flux += self._rigidity_flux_lis(sid, R_is) * fac

        flux /= len(phis)

        # Apply rigidity cutoff (in rigidity space, cutoff is Z-independent)
        if rigidity_cutoff is not None:
            if self.cutoff_width > 0:
                mask = _sigmoid((rigidity - rigidity_cutoff) / self.cutoff_width)
                flux *= mask
            else:
                flux[rigidity < rigidity_cutoff] = 0.0

        return flux

    def jacobian(
        self,
        rigidity: ArrayLike,
        target: str | int | list[int],
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Jacobian including solar modulation."""
        time_interval = self._resolve_time_interval(time_interval)
        rigidity_cutoff = self._resolve_rigidity_cutoff(rigidity_cutoff)
        zlist, _ = self._resolve_z(target)
        rigidity = np.atleast_1d(rigidity)

        phis = self._phi_list(time_interval)
        jac = 0.0
        for phi in phis:
            for sid in self._target_sids(zlist):
                leading, ratio = self.flux_ratio[sid]
                slope = self.flux_slope[sid]

                def _tilt(R, slope=slope, sid=sid):
                    # extrapolation tilt above the species' top knot
                    # (clamped below xmax and saturated at R_sat;
                    # 1.0 for slope 0)
                    if slope == 0.0:
                        return 1.0
                    xmax = self.kx[sid][-1]
                    dx_sat = max(SUBLEADING_SAT_LNR - xmax, 0.0)
                    with np.errstate(divide="ignore"):
                        return np.exp(slope * np.clip(
                            np.log(R) - xmax, 0.0, dx_sat))

                if phi == 0.0:
                    contrib = ratio * self._rigidity_flux_jacobian(
                        leading, rigidity)
                    if slope != 0.0:
                        contrib = contrib * _tilt(rigidity)[:, None]
                else:
                    R_is, fac = self._rigidity_phi_transform(sid, rigidity, phi)
                    contrib = (
                        ratio
                        * self._rigidity_flux_jacobian(leading, R_is)
                        * (fac * _tilt(R_is))[:, None]
                    )
                jac += contrib
        jac = np.asarray(jac) / len(phis)

        # Apply rigidity cutoff (in rigidity space, cutoff is Z-independent)
        if rigidity_cutoff is not None:
            if self.cutoff_width > 0:
                mask = _sigmoid((rigidity - rigidity_cutoff) / self.cutoff_width)
                jac *= mask[:, None]
            else:
                jac[rigidity < rigidity_cutoff, :] = 0.0

        return jac

    def covariance(
        self,
        target1: str | int | list[int],
        target2: str | int | list[int],
        rigidity: ArrayLike,
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Calculate covariance matrix of flux."""
        zlist1, leader1 = self._resolve_z(target1)
        zlist2, leader2 = self._resolve_z(target2)

        jac1 = self.jacobian(
            rigidity,
            target1,
            time_interval=time_interval,
            rigidity_cutoff=rigidity_cutoff,
        )
        jac2 = (
            jac1
            if zlist2 == zlist1
            else self.jacobian(
                rigidity,
                target2,
                time_interval=time_interval,
                rigidity_cutoff=rigidity_cutoff,
            )
        )

        # Use group leaders for covariance lookup
        # key the covariance by the leader SPECIES id (Z, A). _resolve_z returns a
        # charge; a charge with a single species has a cov alias under the bare int,
        # but a charge carrying >1 species (p + D at Z=1) does not — so resolve to
        # the leader sid, which is always a real cov key.
        cov_key = (self._leader_by_charge.get(leader1, leader1),
                   self._leader_by_charge.get(leader2, leader2))

        # Check if covariance matrix entry exists
        if cov_key in self.cov:
            return self._propagate_cov(jac1, jac2, self.cov[cov_key])
        else:
            # If no covariance matrix entry exists, return zero covariance
            # This happens for elements not in the main groups (H, He, O, Fe)
            rigidity = np.atleast_1d(rigidity)
            n_rigidities = len(rigidity)
            return np.zeros((n_rigidities, n_rigidities))


class GSFEnergyPerNucleon(GSFBase):
    """GSF model for nucleon flux calculations (energy per nucleon).

    This class implements the GSF model for calculating nucleon (proton + neutron)
    flux from cosmic ray nuclei. Input energy is specified per nucleon, and the
    output separates proton and neutron contributions.

    The nucleon flux is calculated by:
    - Proton flux = sum over nuclei: flux(nucleus) × A × Z
    - Neutron flux = sum over nuclei: flux(nucleus) × A × (A-Z)

    Where A is atomic mass number and Z is atomic number.

    Examples
    --------
        >>> from globalsplinefit import GSFEnergyPerNucleon
        >>> import numpy as np
        >>> model = GSFEnergyPerNucleon()
        >>> energy_per_nucleon = np.logspace(0, 2, 50)  # GeV/nucleon
        >>> total_nucleons = model.flux(energy_per_nucleon, "He")  # shape (N,)
        >>> p_and_n = model.p_and_n_flux(energy_per_nucleon, "He")  # shape (2, N)
        >>> proton_nucleons = p_and_n[0]
        >>> neutron_nucleons = p_and_n[1]
    """

    def _transform_energy_per_nucleon(
        self,
        energy_per_nucleon: ArrayLike,
        target: str | int | list[int],  # noqa: ARG002
    ) -> np.ndarray:
        """Transform input energy per nucleon. Subclasses override for kinetic energy."""
        return self._scale_energy(np.atleast_1d(energy_per_nucleon).astype(float))

    def p_and_n_flux(
        self,
        energy_per_nucleon: ArrayLike,
        target: str | int | list[int],
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Calculate separate proton and neutron flux from target (group or elements).

        Parameters
        ----------
        energy_per_nucleon
            Energy per nucleon in GeV. Can be scalar, list, or array.
        target
            Target specification (same as GSFEnergy.flux).
        time_interval
            Time period specification:
            - None: Uses the default_time_interval set during initialization
            - "LIS": Local Interstellar Spectrum (no modulation)
            - tuple: (start, end) in YYYYMM format
        rigidity_cutoff
            Geomagnetic rigidity cutoff in GV. Nuclei with rigidity
            below this value are excluded. None uses the default.

        Returns
        -------
            Array of shape (2, N) where N is the number of energy points:
            - [0, :]: Proton nucleon flux in particles/(m²·s·sr·GeV)
            - [1, :]: Neutron nucleon flux in particles/(m²·s·sr·GeV)

        Examples
        --------
            >>> nucleon_flux = model.p_and_n_flux([1, 10, 100], "He")
            >>> proton_flux = nucleon_flux[0]  # 2 protons per He nucleus
            >>> neutron_flux = nucleon_flux[1]  # 2 neutrons per He nucleus
        """
        time_interval = self._resolve_time_interval(time_interval)
        rigidity_cutoff = self._resolve_rigidity_cutoff(rigidity_cutoff)
        zlist, group_leader = self._resolve_z(target)
        energy_per_nucleon = self._transform_energy_per_nucleon(
            energy_per_nucleon, target
        )

        flux = np.zeros((2, len(energy_per_nucleon)))
        for sid in self._target_sids(zlist):
            zi, ai = sid[0], self.z_to_a[sid]
            energy = energy_per_nucleon * ai
            mask = self._rigidity_cutoff_mask(sid, energy, rigidity_cutoff)
            fl = self._element_flux(sid, energy, time_interval)
            flux[0] += fl * ai * zi * mask  # protons
            flux[1] += fl * ai * (ai - zi) * mask  # neutrons
        return flux

    def flux(
        self,
        energy_per_nucleon: ArrayLike,
        target: str | int | list[int],
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Calculate total nucleon flux (protons + neutrons).

        Equivalent to the sum of the two components returned by `p_and_n_flux`.

        Parameters
        ----------
        energy_per_nucleon
            Energy per nucleon in GeV.
        target
            Target specification (group name or list of Z values).
        time_interval
            Time period for solar modulation, or "LIS" for unmodulated.
            ``None`` uses the default set during initialization.
        rigidity_cutoff
            Geomagnetic rigidity cutoff in GV. ``None`` uses the default.

        Returns
        -------
            Array of total nucleon flux in particles/(m²·s·sr·GeV).
        """
        p_and_n = self.p_and_n_flux(
            energy_per_nucleon,
            target,
            time_interval=time_interval,
            rigidity_cutoff=rigidity_cutoff,
        )
        return p_and_n[0] + p_and_n[1]

    def p_and_n_jacobian(
        self,
        energy_per_nucleon: ArrayLike,
        target: str | int | list[int],
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Jacobian of separate proton and neutron flux.

        Parameters
        ----------
        energy_per_nucleon
            Energy per nucleon in GeV.
        target
            Target specification (group name or list of Z values).
        time_interval
            Time period for solar modulation, or "LIS" for unmodulated.
            ``None`` uses the default set during initialization.
        rigidity_cutoff
            Geomagnetic rigidity cutoff in GV. ``None`` uses the default.

        Returns
        -------
            Tuple ``(jac_p, jac_n)`` of proton and neutron flux Jacobians,
            each of shape ``(N, P)`` where ``P`` is the number of parameters.
        """
        time_interval = self._resolve_time_interval(time_interval)
        rigidity_cutoff = self._resolve_rigidity_cutoff(rigidity_cutoff)
        zlist, group_leader = self._resolve_z(target)
        energy_per_nucleon = self._transform_energy_per_nucleon(
            energy_per_nucleon, target
        )

        jac_p = 0.0
        jac_n = 0.0
        for sid in self._target_sids(zlist):
            zi, ai = sid[0], self.z_to_a[sid]
            energy = energy_per_nucleon * ai
            mask = self._rigidity_cutoff_mask(sid, energy, rigidity_cutoff)
            j = self._element_flux_jacobian(sid, energy, time_interval)
            jac_p += j * (ai * zi * mask)[:, np.newaxis]
            jac_n += j * (ai * (ai - zi) * mask)[:, np.newaxis]
        return np.asarray(jac_p), np.asarray(jac_n)

    def jacobian(
        self,
        energy_per_nucleon: ArrayLike,
        target: str | int | list[int],
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Jacobian of total nucleon flux (protons + neutrons).

        Parameters
        ----------
        energy_per_nucleon
            Energy per nucleon in GeV.
        target
            Target specification (group name or list of Z values).
        time_interval
            Time period for solar modulation, or "LIS" for unmodulated.
            ``None`` uses the default set during initialization.
        rigidity_cutoff
            Geomagnetic rigidity cutoff in GV. ``None`` uses the default.

        Returns
        -------
            Jacobian of total nucleon flux, shape ``(N, P)``.
        """
        jac_p, jac_n = self.p_and_n_jacobian(
            energy_per_nucleon,
            target,
            time_interval=time_interval,
            rigidity_cutoff=rigidity_cutoff,
        )
        return jac_p + jac_n

    def p_and_n_covariance(
        self,
        target1: str | int | list[int],
        target2: str | int | list[int],
        energy_per_nucleon: ArrayLike,
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Separate covariance matrices for proton and neutron flux.

        Parameters
        ----------
        target1, target2
            Target specifications for the two targets being correlated.
        energy_per_nucleon
            Energy per nucleon in GeV.
        time_interval
            Time period for solar modulation. ``None`` uses the default.
        rigidity_cutoff
            Geomagnetic rigidity cutoff in GV. ``None`` uses the default.

        Returns
        -------
            Tuple ``(cov_pp, cov_nn)`` of the proton-proton and
            neutron-neutron covariance matrices, each shape ``(N, N)``.
        """
        zlist1, leader1 = self._resolve_z(target1)
        zlist2, leader2 = self._resolve_z(target2)

        # Calculate jacobians using the helper method
        jac1_p, jac1_n = self.p_and_n_jacobian(
            energy_per_nucleon,
            target1,
            time_interval=time_interval,
            rigidity_cutoff=rigidity_cutoff,
        )
        jac2_p, jac2_n = (
            (jac1_p, jac1_n)
            if zlist2 == zlist1
            else self.p_and_n_jacobian(
                energy_per_nucleon,
                target2,
                time_interval=time_interval,
                rigidity_cutoff=rigidity_cutoff,
            )
        )

        # Use group leaders for covariance lookup
        # key the covariance by the leader SPECIES id (Z, A). _resolve_z returns a
        # charge; a charge with a single species has a cov alias under the bare int,
        # but a charge carrying >1 species (p + D at Z=1) does not — so resolve to
        # the leader sid, which is always a real cov key.
        cov_key = (self._leader_by_charge.get(leader1, leader1),
                   self._leader_by_charge.get(leader2, leader2))

        # Check if covariance matrix entry exists
        if cov_key in self.cov:
            return (
                self._propagate_cov(jac1_p, jac2_p, self.cov[cov_key]),
                self._propagate_cov(jac1_n, jac2_n, self.cov[cov_key]),
            )
        else:
            # If no covariance matrix entry exists, return zero covariance
            # This happens for elements not in the main groups (H, He, O, Fe)
            energy_per_nucleon = np.atleast_1d(energy_per_nucleon)
            n_energies = len(energy_per_nucleon)
            zero_cov = np.zeros((n_energies, n_energies))
            return zero_cov, zero_cov

    def covariance(
        self,
        target1: str | int | list[int],
        target2: str | int | list[int],
        energy_per_nucleon: ArrayLike,
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Covariance matrix of total nucleon flux (protons + neutrons).

        Uses the total Jacobian ``J = J_p + J_n``, which correctly includes all
        cross-terms ``Cov(P,P) + Cov(P,N) + Cov(N,P) + Cov(N,N)``.

        Parameters
        ----------
        target1, target2
            Target specifications for the two targets being correlated.
        energy_per_nucleon
            Energy per nucleon in GeV.
        time_interval
            Time period for solar modulation. ``None`` uses the default.
        rigidity_cutoff
            Geomagnetic rigidity cutoff in GV. ``None`` uses the default.

        Returns
        -------
            Covariance matrix of shape ``(N, N)``.
        """
        zlist1, leader1 = self._resolve_z(target1)
        zlist2, leader2 = self._resolve_z(target2)

        # Use total Jacobian J = J_p + J_n directly.
        # Cov(P+N, P+N) = J_total @ C @ J_total.T which correctly includes
        # all four terms: Cov(P,P) + Cov(P,N) + Cov(N,P) + Cov(N,N).
        jac1 = self.jacobian(
            energy_per_nucleon,
            target1,
            time_interval=time_interval,
            rigidity_cutoff=rigidity_cutoff,
        )
        jac2 = (
            jac1
            if zlist2 == zlist1
            else self.jacobian(
                energy_per_nucleon,
                target2,
                time_interval=time_interval,
                rigidity_cutoff=rigidity_cutoff,
            )
        )

        # key the covariance by the leader SPECIES id (Z, A). _resolve_z returns a
        # charge; a charge with a single species has a cov alias under the bare int,
        # but a charge carrying >1 species (p + D at Z=1) does not — so resolve to
        # the leader sid, which is always a real cov key.
        cov_key = (self._leader_by_charge.get(leader1, leader1),
                   self._leader_by_charge.get(leader2, leader2))
        if cov_key in self.cov:
            return self._propagate_cov(jac1, jac2, self.cov[cov_key])
        else:
            energy_per_nucleon = np.atleast_1d(energy_per_nucleon)
            n_energies = len(energy_per_nucleon)
            return np.zeros((n_energies, n_energies))

    def total_flux(
        self,
        energy_or_rigidity: ArrayLike,
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Total nucleon flux summed over all element groups.

        Sums the nucleon flux from all active cosmic ray groups
        (protons, helium, oxygen group, iron group).

        Parameters
        ----------
        energy_or_rigidity
            Energy per nucleon in GeV.
        time_interval
            Time period for solar modulation. ``None`` uses the default.
        rigidity_cutoff
            Geomagnetic rigidity cutoff in GV. ``None`` uses the default.

        Returns
        -------
            Array of total nucleon flux in particles/(m²·s·sr·GeV).
        """
        energy_or_rigidity = np.atleast_1d(energy_or_rigidity)
        # Initialize with proper shape for total nucleon flux (N,)
        total_flux = np.zeros(len(energy_or_rigidity), dtype=float)

        for group in self.active_groups:
            group_flux = self.flux(
                energy_or_rigidity,
                group,
                time_interval=time_interval,
                rigidity_cutoff=rigidity_cutoff,
            )
            total_flux += group_flux

        return total_flux

    def p_and_n_total_flux(
        self,
        energy_or_rigidity: ArrayLike,
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Separate proton and neutron flux summed over all element groups.

        Parameters
        ----------
        energy_or_rigidity
            Energy per nucleon in GeV.
        time_interval
            Time period for solar modulation. ``None`` uses the default.
        rigidity_cutoff
            Geomagnetic rigidity cutoff in GV. ``None`` uses the default.

        Returns
        -------
            Array of shape ``(2, N)``: row 0 is total proton flux,
            row 1 is total neutron flux, both in particles/(m²·s·sr·GeV).
        """
        energy_or_rigidity = np.atleast_1d(energy_or_rigidity)
        # Initialize with proper shape for nucleon flux (2, N)
        total_flux = np.zeros((2, len(energy_or_rigidity)), dtype=float)

        for group in self.active_groups:
            group_flux = self.p_and_n_flux(
                energy_or_rigidity,
                group,
                time_interval=time_interval,
                rigidity_cutoff=rigidity_cutoff,
            )
            total_flux += group_flux

        return total_flux

    def total_error(
        self,
        energy_or_rigidity: ArrayLike,
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """1-sigma uncertainty of total nucleon flux from all element groups.

        Accounts for correlations between groups through the full covariance
        matrix.

        Parameters
        ----------
        energy_or_rigidity
            Energy per nucleon in GeV.
        time_interval
            Time period for solar modulation. ``None`` uses the default.
        rigidity_cutoff
            Geomagnetic rigidity cutoff in GV. ``None`` uses the default.

        Returns
        -------
            Array of total nucleon flux uncertainties (1-sigma), shape ``(N,)``.
        """
        energy_or_rigidity = np.atleast_1d(energy_or_rigidity)
        n_points = len(energy_or_rigidity)

        # Initialize covariance matrix for total nucleon flux
        total_cov = np.zeros((n_points, n_points), dtype=float)

        # Sum covariances across all group pairs
        for l1 in self.active_groups:
            for l2 in self.active_groups:
                cov = self.covariance(
                    l1,
                    l2,
                    energy_or_rigidity,
                    time_interval=time_interval,
                    rigidity_cutoff=rigidity_cutoff,
                )
                total_cov += cov

        # Return uncertainties as (N,) array
        return np.sqrt(np.diag(total_cov))

    def p_and_n_error(
        self,
        energy_per_nucleon: ArrayLike,
        target: str | int | list[int],
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Separate proton and neutron flux uncertainties (1-sigma).

        Parameters
        ----------
        energy_per_nucleon
            Energy per nucleon in GeV.
        target
            Target specification (group name or list of Z values).
        time_interval
            Time period for solar modulation. ``None`` uses the default.
        rigidity_cutoff
            Geomagnetic rigidity cutoff in GV. ``None`` uses the default.

        Returns
        -------
            Array of shape ``(2, N)``: row 0 is proton uncertainties,
            row 1 is neutron uncertainties.
        """
        cov_pp, cov_nn = self.p_and_n_covariance(
            target,
            target,
            energy_per_nucleon,
            time_interval=time_interval,
            rigidity_cutoff=rigidity_cutoff,
        )
        return np.array([np.sqrt(np.diag(cov_pp)), np.sqrt(np.diag(cov_nn))])

    def error(
        self,
        energy_or_rigidity: ArrayLike,
        target: str | int | list[int],
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """1-sigma uncertainty of total nucleon flux (protons + neutrons).

        Use `p_and_n_error` to get separate proton and neutron uncertainties.

        Parameters
        ----------
        energy_or_rigidity
            Energy per nucleon in GeV.
        target
            Target specification (group name or list of Z values).
        time_interval
            Time period for solar modulation. ``None`` uses the default.
        rigidity_cutoff
            Geomagnetic rigidity cutoff in GV. ``None`` uses the default.

        Returns
        -------
            Array of total nucleon flux uncertainties (1-sigma), shape ``(N,)``.
        """
        # Get covariance for total nucleon flux
        total_cov = self.covariance(
            target,
            target,
            energy_or_rigidity,
            time_interval=time_interval,
            rigidity_cutoff=rigidity_cutoff,
        )
        return np.sqrt(np.diag(total_cov))


class GSFKineticEnergyPerNucleon(GSFEnergyPerNucleon):
    """GSF model for nucleon flux calculations using kinetic energy per nucleon.

    Converts kinetic energy per nucleon to total energy per nucleon internally
    before using the GSFEnergyPerNucleon calculation methods.

    Examples
    --------
        >>> from globalsplinefit import GSFKineticEnergyPerNucleon
        >>> import numpy as np
        >>> model = GSFKineticEnergyPerNucleon()
        >>> kinetic_energy_per_nucleon = np.logspace(0, 2, 50)  # GeV/nucleon kinetic
        >>> total_nucleons = model.flux(kinetic_energy_per_nucleon, "He")  # shape (N,)
        >>> p_and_n = model.p_and_n_flux(kinetic_energy_per_nucleon, "He")
    """

    def _transform_energy_per_nucleon(
        self,
        kinetic_energy_per_nucleon: ArrayLike,
        target: str | int | list[int],  # noqa: ARG002
    ) -> np.ndarray:
        """Convert kinetic energy per nucleon to total energy per nucleon."""
        return (
            self._scale_energy(np.atleast_1d(kinetic_energy_per_nucleon).astype(float))
            + NUCLEON_MASS_GEV
        )
