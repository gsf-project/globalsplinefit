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
            Optional model version ("2017", "2019", "2025"). If specified,
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
            Width of the cutoff transition in GV. Default 0.0 gives
            a sharp (Heaviside) cutoff. Values > 0 produce a smooth sigmoid
            transition modeling the geomagnetic penumbra.
        """
        self.params = Parameters(
            data_path, version, use_approximate_solar_cycle_average
        )

        # Store default time interval
        self.default_time_interval = default_time_interval

        # Store default rigidity cutoff
        self.default_rigidity_cutoff = default_rigidity_cutoff
        self.cutoff_width = cutoff_width

        # Store version for compatibility decisions
        self.version = version

        # Copy frequently used parameters for convenience
        self.kx = self.params.kx
        self.pars = self.params.pars
        self.npar = self.params.npar
        self.z_group = self.params.z_group
        self.z_to_a = self.params.z_to_a
        self.cov = self.params.cov
        self.phi = self.params.phi
        self.flux_ratio = self.params.flux_ratio
        # Helpers
        self.active_groups = ["p", "He", "O*", "Fe*"]
        self.z_ungroup = {
            z: group for group, elements in self.z_group.items() for z in elements
        }

        # Cache for _rigidity_flux_jacobian results
        self._jacobian_cache = {}
        self._cache_max_size = 1000  # Limit cache size to prevent memory issues

    def _make_cache_key(
        self, z: int, rigidity: np.ndarray
    ) -> tuple[int, tuple, np.dtype, bytes]:
        """Build a cache key from element charge and rigidity array."""
        arr = np.atleast_1d(rigidity)
        return (z, arr.shape, arr.dtype, arr.tobytes())

    def _manage_cache_size(self):
        """Keep cache size under control by removing oldest entries."""
        if len(self._jacobian_cache) >= self._cache_max_size:
            # Remove about 20% of oldest entries (simple FIFO strategy)
            items_to_remove = len(self._jacobian_cache) // 5
            for _ in range(items_to_remove):
                # Remove first (oldest) item
                oldest_key = next(iter(self._jacobian_cache))
                del self._jacobian_cache[oldest_key]

    def _resolve_z(self, target: str | int | list[int]) -> tuple[list[int], int]:
        """Resolve group/element specification to a list of atomic numbers and group leader.

        Parameters
        ----------
        target
            Target specification, which can be:
            - String: Group name ("p", "proton", "H", "He", "helium", "alpha",
            "O", "oxygen", "CNO", "O*", "Fe", "iron", "Fe*")
            - Integer: Single element atomic number (e.g., 1 for hydrogen, 2 for helium)
            - List/array: Multiple element atomic numbers from the same group

        Returns
        -------
            Tuple of:
            - List of atomic numbers included in the target
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
        """
        if isinstance(target, str):
            # String input -> resolve to group
            if target in self.GROUP_NAMES:
                group_leader = self.GROUP_NAMES[target]
                return list(self.z_group[group_leader]), group_leader
            else:
                raise ValueError(
                    f"Unknown group name: {target} - valid names are {list(self.GROUP_NAMES.keys())}"
                )

        elif isinstance(target, int | np.integer):
            # Single integer -> single element
            z = int(target)
            if z not in self.z_ungroup:
                raise ValueError(f"Unknown element: {z} - not in GSF model")
            return [z], self.z_ungroup[z]

        elif isinstance(target, list | tuple | np.ndarray):
            # List/array of integers -> multiple elements
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
        self, z: int, energy: np.ndarray, rigidity_cutoff: float | None
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
        mass = self.z_to_a[z] * NUCLEON_MASS_GEV
        p2 = np.maximum(energy**2 - mass**2, 0.0)
        rig = np.sqrt(p2) / z

        if self.cutoff_width > 0:
            # Smooth sigmoid transition (models geomagnetic penumbra)
            x = (rig - rigidity_cutoff) / self.cutoff_width
            return _sigmoid(x)

        return (rig >= rigidity_cutoff).astype(float)

    def _spline(self, z: int, x: np.ndarray) -> np.ndarray:
        """Evaluate spline for element z at log rigidity x.

        Parameters
        ----------
        z
            Atomic number of the element.
        x
            Natural logarithm of rigidity values.

        Returns
        -------
            Spline-evaluated flux values with power-law correction factor.
        """
        return splev(x, (self.kx[z], self.pars[z], SPLINE_DEGREE)) * np.exp(-3.0 * x)

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
            - None: Solar Cycle 24 average (default behavior)
            - "LIS": Local Interstellar Spectrum (phi=0, no modulation)
            - tuple[int, int]: (start, end) in YYYYMM format.
            Example: (200901, 200912) for Jan-Dec 2009.

        Returns
        -------
            Array of solar modulation potential values (in MV) for the time period.

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
        return _collect_phi_values(self.phi, t_a, t_b)

    def _rigidity_flux_lis(self, z: int, rigidity: ArrayLike) -> np.ndarray:
        """Calculate LIS flux as a function of rigidity."""
        rigidity = np.atleast_1d(rigidity)

        with np.errstate(divide="ignore", invalid="ignore"):
            log_rigidity = np.log(rigidity)

        # Get valid rigidity range for this element
        min_log_rigidity = self.kx[z][0]
        max_log_rigidity = self.kx[z][-1]

        # Initialize result array
        result = np.zeros_like(log_rigidity)

        # Only calculate flux for valid rigidity range
        valid_mask = log_rigidity >= min_log_rigidity
        if not np.any(valid_mask):
            return result

        valid_log_rigidity = log_rigidity[valid_mask]

        if z in self.z_group:  # Leading element
            result[valid_mask] = self._spline(z, valid_log_rigidity)
        else:  # Subleading element
            leading, ratio = self.flux_ratio[z]

            # Split into regions: within spline range vs extrapolation region
            within_range = valid_log_rigidity <= max_log_rigidity
            extrapolation = ~within_range

            if np.any(within_range):
                # Use element's own spline within its range
                within_indices = valid_mask.copy()
                within_indices[valid_mask] = within_range
                result[within_indices] = self._spline(
                    z, valid_log_rigidity[within_range]
                )

            if np.any(extrapolation):
                # Use leading element's spline scaled by ratio for extrapolation
                extrap_indices = valid_mask.copy()
                extrap_indices[valid_mask] = extrapolation
                result[extrap_indices] = ratio * self._spline(
                    leading, valid_log_rigidity[extrapolation]
                )

        return result

    def _element_flux(
        self,
        z: int,
        energy: ArrayLike,
        time_interval: tuple[int, int] | str | None = None,
    ) -> np.ndarray:
        """Calculate flux for element with charge z."""
        energy = np.atleast_1d(energy)
        time_interval = self._resolve_time_interval(time_interval)
        phis = np.array(self._phi_list(time_interval))

        # Get vectorized rigidity and factors for all phi values
        # Shape: [n_energy, n_phi]
        rigidity, factor = self._rigidity_from_energy_vectorized(z, energy, phis)

        # Calculate LIS flux for all rigidities at once
        # Reshape rigidity to 1D for LIS calculation, then reshape back
        rig_flat = rigidity.flatten()
        lis_flux_flat = self._rigidity_flux_lis(z, rig_flat)
        lis_flux = lis_flux_flat.reshape(rigidity.shape)  # [n_energy, n_phi]

        # Apply factors and average over phi dimension
        modulated_flux = lis_flux * factor  # [n_energy, n_phi]
        averaged_flux = np.mean(modulated_flux, axis=1)  # [n_energy]

        return averaged_flux

    def _element_flux_jacobian(
        self,
        z: int,
        energy: ArrayLike,
        time_interval: tuple[int, int] | str | None = None,
    ) -> np.ndarray:
        """Calculate Jacobian of flux for uncertainty propagation."""
        energy = np.atleast_1d(energy)
        leading, ratio = self.flux_ratio[z]
        time_interval = self._resolve_time_interval(time_interval)
        phis = np.array(self._phi_list(time_interval))

        # Get vectorized rigidity and factors
        rigidity, factor = self._rigidity_from_energy_vectorized(z, energy, phis)

        # Calculate Jacobian for all rigidities
        # This requires careful reshaping to handle the parameter dimension
        rig_flat = rigidity.flatten()
        jac_flat = self._rigidity_flux_jacobian(
            leading, rig_flat
        )  # [n_energy*n_phi, n_params]

        # Reshape to [n_energy, n_phi, n_params]
        jac = jac_flat.reshape(energy.shape[0], len(phis), -1)

        # Apply factors (broadcasting over parameter dimension)
        jac_weighted = jac * factor[:, :, np.newaxis]

        # Average over phi dimension
        jac_averaged = np.mean(jac_weighted, axis=1)  # [n_energy, n_params]

        return ratio * jac_averaged

    def _rigidity_from_energy_vectorized(
        self, z: int, energy: np.ndarray, phis: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized version that handles multiple phi values at once."""
        # Convert inputs to proper shapes for broadcasting
        energy = np.atleast_1d(energy)
        phis = np.atleast_1d(phis)

        # Reshape for broadcasting: energy[n_energy, 1], phis[1, n_phi]
        energy_bc = energy[:, np.newaxis]
        phis_bc = phis[np.newaxis, :]

        # Get atomic mass and nucleon mass
        nucleon_mass = NUCLEON_MASS_GEV
        mass = self.z_to_a[z] * nucleon_mass

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

    def _rigidity_flux_jacobian(self, z: int, rigidity: np.ndarray) -> np.ndarray:
        """Calculate Jacobian of LIS flux for leading element.

        This method is cached to improve performance for repeated calls with
        the same parameters.
        """
        cache_key = self._make_cache_key(z, rigidity)
        if cache_key in self._jacobian_cache:
            return self._jacobian_cache[cache_key].copy()

        self._manage_cache_size()
        result = self._compute_rigidity_flux_jacobian(z, rigidity)
        self._jacobian_cache[cache_key] = result.copy()
        return result

    def _compute_rigidity_flux_jacobian(
        self, z: int, rigidity: np.ndarray
    ) -> np.ndarray:
        """Compute Jacobian of LIS flux for leading element (uncached implementation).

        This is the original implementation extracted to a separate method
        to maintain clean separation between caching logic and computation.
        """
        with np.errstate(divide="ignore"):
            x = np.log(rigidity)

        jac = np.zeros((len(x), self.npar[z]))
        pi = np.zeros(self.npar[z] + 4)  # splev needs 4 extra zeros

        for ipar in range(self.npar[z]):
            pi[ipar] = 1.0
            v = splev(x, (self.kx[z], pi, SPLINE_DEGREE))
            v[x < self.kx[z][0]] = 0.0
            jac[:, ipar] = v
            pi[ipar] = 0.0

        x[x < self.kx[z][0]] = 0.0
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
            - tuple: (start, end) in YYYYMM format, e.g. (200901, 200912)
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

    def _element_lnA_fluxes(
        self, energy_or_rigidity, *, time_interval=None, rigidity_cutoff=None
    ):
        """(ln A, per-element flux array) over every element in the model."""
        zs = sorted(self.z_to_a)
        fl = np.array(
            [
                self.flux(
                    energy_or_rigidity,
                    z,
                    time_interval=time_interval,
                    rigidity_cutoff=rigidity_cutoff,
                )
                for z in zs
            ]
        )
        lnA = np.log(np.array([self.z_to_a[z] for z in zs]))
        return lnA, fl

    def mean_lnA(
        self,
        energy_or_rigidity: ArrayLike,
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Flux-weighted mean of ln A over all elements."""
        lnA, fl = self._element_lnA_fluxes(
            energy_or_rigidity,
            time_interval=time_interval,
            rigidity_cutoff=rigidity_cutoff,
        )
        tot = fl.sum(0)
        return (fl * lnA[:, None]).sum(0) / np.where(tot > 0, tot, np.nan)

    def var_lnA(
        self,
        energy_or_rigidity: ArrayLike,
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Flux-weighted variance of ln A over all elements."""
        lnA, fl = self._element_lnA_fluxes(
            energy_or_rigidity,
            time_interval=time_interval,
            rigidity_cutoff=rigidity_cutoff,
        )
        tot = fl.sum(0)
        m = (fl * lnA[:, None]).sum(0) / np.where(tot > 0, tot, np.nan)
        return (fl * (lnA[:, None] - m[None, :]) ** 2).sum(0) / np.where(
            tot > 0, tot, np.nan
        )

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
            - String: Group name ("p", "He", "O*", "Fe*", etc.)
            - Integer: Single element atomic number
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
        return np.atleast_1d(energy).astype(float)

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
            - Integer: Single element atomic number (e.g., 1 for H, 2 for He)
            - List: Multiple elements from same group (e.g., [6,7,8] for CNO)
        time_interval
            Time period specification:
            - None: Uses the default_time_interval set during initialization
            - "LIS": Local Interstellar Spectrum (no modulation)
            - tuple: (start, end) in YYYYMM format, e.g. (200901, 200912)
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
        for zi in zlist:
            mask = self._rigidity_cutoff_mask(zi, energy, rigidity_cutoff)
            flux += self._element_flux(zi, energy, time_interval) * mask
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
        for zi in zlist:
            mask = self._rigidity_cutoff_mask(zi, energy, rigidity_cutoff)
            jac += (
                self._element_flux_jacobian(zi, energy, time_interval)
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
        energy = self._transform_energy(energy, target1)

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
        cov_key = (leader1, leader2)

        # Check if covariance matrix entry exists
        if cov_key in self.cov:
            return self._propagate_cov(jac1, jac2, self.cov[cov_key])
        else:
            n_energies = len(energy)
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
        kinetic_energy = np.atleast_1d(kinetic_energy).astype(float)
        _zlist, leader = self._resolve_z(target)
        rest_mass = self.z_to_a[leader] * NUCLEON_MASS_GEV
        return kinetic_energy + rest_mass


class GSFRigidity(GSFBase):
    """GSF model expecting rigidity in GV.

    This class implements the GSF model for cosmic ray calculations using
    magnetic rigidity as input parameter. Rigidity is defined as
    R = pc/Z where p is momentum, c is speed of light, and Z is charge.

    The model supports both Local Interstellar Spectrum (LIS) calculations
    and solar modulation effects using the force-field approximation to
    transform between Earth and interstellar rigidity spectra.

    Examples
    --------
        >>> from globalsplinefit import GSFRigidity
        >>> import numpy as np
        >>> model = GSFRigidity()
        >>> rigidity = np.logspace(0, 3, 100)  # 1 GV to 1 TV
        >>> proton_flux_lis = model.flux(rigidity, "p")  # LIS
        >>> proton_flux_2009 = model.flux(rigidity, "p", time_interval=(200901, 200912))
        >>> total_flux = model.total_flux(rigidity)
    """

    # ---------- new helpers ----------
    def _rigidity_phi_transform(
        self, z: int, rigidity: np.ndarray, phi: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert Earth rigidity to interstellar rigidity under force-field phi.

        Returns Lambda(R, phi) = dR_IS/dR prefactor (Eq. 3).
        """
        nucleon_mass = NUCLEON_MASS_GEV
        a = self.z_to_a[z]
        m = a * nucleon_mass

        # Earth energy from input R
        E = np.sqrt((z * rigidity) ** 2 + m**2)
        E_is = E + z * phi / a

        # Interstellar rigidity
        with np.errstate(divide="ignore", invalid="ignore"):
            p2_is = E_is**2 - m**2
            p2_is[p2_is < 0] = 0.0
            R_is = np.sqrt(p2_is) / z

            # Jacobian prefactor Λ
            Lambda = (R_is**2 / (rigidity**2 + _ZERO_GUARD)) * (
                E / (E_is + _ZERO_GUARD)
            )
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
            - tuple: (start, end) in YYYYMM format, e.g. (200901, 200912)
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
            for zi in zlist:
                if phi == 0.0:
                    flux += self._rigidity_flux_lis(zi, rigidity)
                else:
                    R_is, fac = self._rigidity_phi_transform(zi, rigidity, phi)
                    flux += self._rigidity_flux_lis(zi, R_is) * fac

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
            for zi in zlist:
                leading, ratio = self.flux_ratio[zi]
                if phi == 0.0:
                    jac += ratio * self._rigidity_flux_jacobian(leading, rigidity)
                else:
                    R_is, fac = self._rigidity_phi_transform(zi, rigidity, phi)
                    jac += (
                        ratio
                        * self._rigidity_flux_jacobian(leading, R_is)
                        * fac[:, None]
                    )
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
        cov_key = (leader1, leader2)

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
        return np.atleast_1d(energy_per_nucleon).astype(float)

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
        for zi in zlist:
            ai = self.z_to_a[zi]
            energy = energy_per_nucleon * ai
            mask = self._rigidity_cutoff_mask(zi, energy, rigidity_cutoff)
            fl = self._element_flux(zi, energy, time_interval)
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
        for zi in zlist:
            ai = self.z_to_a[zi]
            energy = energy_per_nucleon * ai
            mask = self._rigidity_cutoff_mask(zi, energy, rigidity_cutoff)
            j = self._element_flux_jacobian(zi, energy, time_interval)
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
        cov_key = (leader1, leader2)

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

        cov_key = (leader1, leader2)
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
            np.atleast_1d(kinetic_energy_per_nucleon).astype(float) + NUCLEON_MASS_GEV
        )
