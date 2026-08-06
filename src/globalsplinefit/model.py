"""Cosmic-ray flux models in energy, energy per nucleon, and rigidity."""

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from hashlib import blake2b
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import BSpline, splev

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


def _bin_phi(phis: np.ndarray, n_bins: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Reduce monthly phi values to weighted representatives for averaging.

    Mirrors the fitter's period average
    (``gsffit.data.solarmod.window_phi_samples`` in gsf-fitter-2): histogram the
    monthly potentials into ``n_bins`` bins and keep each populated bin's **mean**
    phi — not its center — weighted by the number of months in it.  The weighted
    mean of the representatives then equals the true monthly mean exactly, so the
    first-order (mean) suppression is preserved and only the curvature of
    flux(phi) is approximated.

    Because the flux is nonlinear in phi, averaging the modulated flux over these
    bins ("mean of modulated") is not the same as modulating at the mean phi
    ("modulate the mean"); the bin count sets how much of that curvature is kept.

    Parameters
    ----------
    phis
        Monthly potentials in GV.
    n_bins
        Number of bins.  ``None`` keeps every month (the full explicit average).
        ``1`` collapses to the plain period mean, i.e. modulate-the-mean.  Bins
        that end up empty are dropped, so fewer than ``n_bins`` values may be
        returned.

    Returns
    -------
        ``(phi, weights)`` with ``weights`` normalized to sum to 1.
    """
    phis = np.atleast_1d(np.asarray(phis, dtype=float))
    if phis.size == 0:
        raise ValueError("no solar modulation potentials in the requested interval")
    if n_bins is None or phis.size <= n_bins:
        return phis, np.full(phis.size, 1.0 / phis.size)
    if n_bins == 1:
        return np.array([phis.mean()]), np.array([1.0])
    counts, edges = np.histogram(phis, bins=n_bins)
    sums, _ = np.histogram(phis, bins=edges, weights=phis)
    nz = counts > 0
    return sums[nz] / counts[nz], counts[nz] / counts[nz].sum()


FloatArray: TypeAlias = NDArray[np.float64]
SpeciesID: TypeAlias = tuple[int, float]
Target: TypeAlias = str | int | SpeciesID | Sequence[int] | NDArray[np.integer]


class GSFBase(ABC):
    """Shared implementation for the four GSF composition groups."""

    # Named individual species. The lightest isotope at the given charge is used.
    PARTICLE_NAMES = {
        "p": 1,
        "proton": 1,
        "protons": 1,
        "O": 8,
        "oxygen": 8,
        "Fe": 26,
        "iron": 26,
    }

    # Named non-leading isotopes.
    SPECIES_NAMES = {
        "D": 1,
        "deuterium": 1,
    }

    # Canonical composition groups. Starred names are used where a group spans
    # several element charges; H and He contain isotopes of one element.
    GROUP_NAMES = {
        "H": 1,
        "H*": 1,
        "hydrogen": 1,
        "He": 2,
        "helium": 2,
        "alpha": 2,
        "CNO": 8,
        "O*": 8,
        "Fe*": 26,
        "heavy": 26,
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
        value = float(value)
        if not np.isfinite(value) or value <= -1.0:
            raise ValueError("energy_scale must be finite and greater than -1")
        self._energy_scale = value

    def _scale_energy(self, e: np.ndarray) -> np.ndarray:
        """Apply the global energy-scale nuisance to an input energy/rigidity array."""
        return e * (1.0 + self._energy_scale) if self._energy_scale else e

    def __init__(
        self,
        data_path: str | Path | None = None,
        version: str | None = None,
        solar_cycle_average_bins: int | None = 12,
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
            Optional registered model version. Defaults to ``"2026"``. See
            :data:`~globalsplinefit.data_management.MODEL_VERSIONS` and
            :func:`~globalsplinefit.data_management.version_info`. If specified,
            overrides data_path and uses the corresponding package data directory.
        solar_cycle_average_bins
            Weighted bins used for a solar-period average. The default 12-bin
            approximation is within 0.1% of the full monthly average for
            E >= 1 GeV. Use ``None`` for every month or ``1`` for the mean
            modulation potential.
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
        if solar_cycle_average_bins is not None and (
            isinstance(solar_cycle_average_bins, bool)
            or not isinstance(solar_cycle_average_bins, (int, np.integer))
            or solar_cycle_average_bins < 1
        ):
            raise ValueError(
                "solar_cycle_average_bins must be a positive int or None, got "
                f"{solar_cycle_average_bins!r}"
            )
        self.solar_cycle_average_bins = (
            None if solar_cycle_average_bins is None else int(solar_cycle_average_bins)
        )
        self.params = Parameters.for_model(data_path, version)

        # Store default time interval
        self.default_time_interval = default_time_interval

        # Store default rigidity cutoff
        if default_rigidity_cutoff is not None and (
            not np.isfinite(default_rigidity_cutoff) or default_rigidity_cutoff < 0
        ):
            raise ValueError("default_rigidity_cutoff must be finite and non-negative")
        if not np.isfinite(cutoff_width) or cutoff_width < 0:
            raise ValueError("cutoff_width must be finite and non-negative")
        self.default_rigidity_cutoff = default_rigidity_cutoff
        self.cutoff_width = float(cutoff_width)

        # The RESOLVED version name, not the constructor argument: a model built
        # with no arguments reports the default it actually loaded rather than
        # None. None only when data_path points outside the packaged versions.
        self.version = self.params.version

        # Copy frequently used parameters for convenience. Species are keyed by
        # ``sid = (Z, A)`` so isotopes (e.g. D and p both at Z=1) coexist; for a
        # v1 (one-species-per-Z) model each charge maps to a single sid, so the
        # behaviour is identical.
        self.kx = self.params.kx  # sid (+int charge alias) -> knots
        self.pars = self.params.pars  # sid (+int alias) -> coeffs
        self.npar = self.params.npar  # sid (+int alias) -> n coeffs
        self.z_group = self.params.z_group  # leader charge -> [member charges]
        self.z_to_a = self.params.z_to_a  # sid (+int alias) -> A
        self.mass_number = self.params.mass_number
        self.z_to_sids = self.params.z_to_sids  # charge Z -> [sids]
        self.z_ungroup = self.params.z_ungroup  # charge -> leader charge
        self.cov = self.params.cov  # (sid,sid) (+int alias) -> block
        self.phi = self.params.phi
        self.flux_ratio = (
            self.params.flux_ratio
        )  # sid (+int alias) -> (leader_sid, ratio)
        self.flux_slope = (
            self.params.flux_slope
        )  # sid (+int alias) -> extrapolation slope
        self.species = self.params.species  # sorted list of all sids
        self._leaders = self.params._leaders  # set of leader sids
        self._leader_by_charge = self.params.leader_sid  # int charge -> leader_sid
        # Helpers
        self.active_groups = ("H", "He", "O*", "Fe*")

        # Cache for _rigidity_flux_jacobian results
        self._jacobian_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._jacobian_cache_bytes = 0
        self._cache_max_bytes = 64 * 1024 * 1024

        # Lazy factorization of the stacked leader-amplitude covariance
        # used by ``sample`` (see _amplitude_sample_factor).
        self._sample_factor_cache: tuple[dict, np.ndarray] | None = None

    def _as_sid(self, x):
        """Normalize a species ID or unambiguous bare charge.

        Charges containing multiple isotopes require an explicit ``(Z, A)`` ID.
        """
        if isinstance(x, tuple):
            if x not in self.species:
                raise ValueError(f"unknown species id {x!r}")
            return x
        charge = int(x)
        sids = self.z_to_sids.get(charge, ())
        if len(sids) != 1:
            raise ValueError(
                f"charge Z={charge} has {len(sids)} species; pass an explicit (Z, A) id"
            )
        return sids[0]

    @staticmethod
    def _as_1d_values(
        values: ArrayLike,
        name: str,
        *,
        non_negative: bool = True,
        positive: bool = False,
    ) -> FloatArray:
        """Normalize a public numerical input and enforce its physical domain."""
        array = np.asarray(values, dtype=float)
        if array.ndim == 0:
            array = array.reshape(1)
        if array.ndim != 1:
            raise ValueError(f"{name} must be a scalar or one-dimensional array")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        if positive and np.any(array <= 0):
            raise ValueError(f"{name} must contain only positive values")
        if non_negative and np.any(array < 0):
            raise ValueError(f"{name} must contain only non-negative values")
        return array

    def _make_cache_key(self, sid, rigidity: np.ndarray) -> tuple:
        """Build a cache key from the species id (Z, A) and rigidity array."""
        sid = self._as_sid(sid)
        arr = np.ascontiguousarray(np.atleast_1d(rigidity))
        digest = blake2b(arr.view(np.uint8), digest_size=16).digest()
        return (sid, arr.shape, arr.dtype.str, digest)

    def _cache_jacobian(self, key: tuple, value: np.ndarray) -> None:
        """Insert a read-only Jacobian into the byte-bounded LRU cache."""
        if value.nbytes > self._cache_max_bytes // 4:
            return
        value.setflags(write=False)
        while self._jacobian_cache and (
            self._jacobian_cache_bytes + value.nbytes > self._cache_max_bytes
        ):
            _old_key, old_value = self._jacobian_cache.popitem(last=False)
            self._jacobian_cache_bytes -= old_value.nbytes
        self._jacobian_cache[key] = value
        self._jacobian_cache_bytes += value.nbytes

    def _resolve_z(self, target: Target) -> tuple[list, int]:
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
            if target in self.GROUP_NAMES:
                group_leader = self.GROUP_NAMES[target]
                return list(self.z_group[group_leader]), group_leader
            if target in self.PARTICLE_NAMES:
                charge = self.PARTICLE_NAMES[target]
                return [self._leader_by_charge[charge]], self.z_ungroup[charge]
            if target in self.SPECIES_NAMES:
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
            valid = (
                list(self.PARTICLE_NAMES)
                + list(self.SPECIES_NAMES)
                + list(self.GROUP_NAMES)
            )
            raise ValueError(f"unknown target {target!r}; valid names are {valid}")

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
            if len(target) == 0:
                raise ValueError("target charge list cannot be empty")
            if any(
                isinstance(z, bool) or not isinstance(z, (int, np.integer))
                for z in target
            ):
                raise ValueError("target charge lists must contain only integers")
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
        value = float(rigidity_cutoff)
        if not np.isfinite(value) or value < 0:
            raise ValueError("rigidity_cutoff must be finite and non-negative")
        return value

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
        return splev(x, (self.kx[sid], self.pars[sid], SPLINE_DEGREE)) * np.exp(
            -3.0 * x
        )

    def _propagate_cov(
        self, j1: np.ndarray, j2: np.ndarray, c: np.ndarray
    ) -> np.ndarray:
        """Compute J1 @ C @ J2.T for covariance propagation."""
        return np.linalg.multi_dot((j1, c, j2.T))

    def _phi_list(
        self, time_interval: tuple[int, int] | str | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solar modulation potentials and their weights for a time interval.

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

        The monthly values of an interval are binned into
        ``solar_cycle_average_bins`` weighted representatives — see
        :func:`_bin_phi`.  Every interval, including the Solar Cycle 24
        default, goes through the same reduction.

        Returns
        -------
            ``(phi, weights)`` — potentials in GV and normalized weights
            (summing to 1) for the period average.

        Raises
        ------
            ValueError: If start equals end, or start > end in time interval tuple.
        """
        if time_interval is None:
            # Default: Solar Cycle 24 average (December 2008 to December 2019)
            time_interval = self.params.get_solar_cycle_24_interval()
        elif time_interval == "LIS":
            # Local Interstellar Spectrum: no solar modulation
            return np.array([0.0]), np.array([1.0])
        elif isinstance(time_interval, str):
            raise ValueError(
                f"Invalid string time_interval '{time_interval}'. Only 'LIS' is supported."
            )

        if not isinstance(time_interval, tuple) or len(time_interval) != 2:
            raise ValueError("time_interval must be 'LIS' or a (start, end) tuple")
        t_a, t_b = time_interval
        phis = _collect_phi_values(self.phi, t_a, t_b)
        return _bin_phi(phis, self.solar_cycle_average_bins)

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
                    result[extrap_indices] *= np.exp(
                        slope * np.clip(x_ex - max_log_rigidity, 0.0, dx_sat)
                    )

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
        phis, weights = self._phi_list(time_interval)

        # Get vectorized rigidity and factors for all phi values
        # Shape: [n_energy, n_phi]
        rigidity, factor = self._rigidity_from_energy_vectorized(sid, energy, phis)

        # Calculate LIS flux for all rigidities at once
        # Reshape rigidity to 1D for LIS calculation, then reshape back
        rig_flat = rigidity.flatten()
        lis_flux_flat = self._rigidity_flux_lis(sid, rig_flat)
        lis_flux = lis_flux_flat.reshape(rigidity.shape)  # [n_energy, n_phi]

        # Apply factors and take the weighted period average over phi
        modulated_flux = lis_flux * factor  # [n_energy, n_phi]
        averaged_flux = modulated_flux @ weights  # [n_energy]

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
        phis, weights = self._phi_list(time_interval)

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
                tilt = np.exp(slope * np.clip(np.log(rigidity) - xmax, 0.0, dx_sat))
            factor = factor * tilt
        jac_weighted = jac * factor[:, :, np.newaxis]

        # Weighted period average over the phi dimension
        jac_averaged = np.tensordot(  # [n_energy, n_params]
            jac_weighted, weights, axes=([1], [0])
        )

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
            self._jacobian_cache.move_to_end(cache_key)
            return self._jacobian_cache[cache_key]

        result = self._compute_rigidity_flux_jacobian(sid, rigidity)
        self._cache_jacobian(cache_key, result)
        return result

    def _compute_rigidity_flux_jacobian(self, sid, rigidity: np.ndarray) -> np.ndarray:
        """Compute Jacobian of LIS flux for species ``sid`` (uncached).

        This is the original implementation extracted to a separate method
        to maintain clean separation between caching logic and computation.
        """
        sid = self._as_sid(sid)
        with np.errstate(divide="ignore"):
            x = np.log(rigidity)

        jac = np.zeros((len(x), self.npar[sid]))
        valid = x >= self.kx[sid][0]
        if np.any(valid):
            basis = BSpline.design_matrix(
                x[valid], self.kx[sid], SPLINE_DEGREE, extrapolate=True
            ).toarray()
            jac[valid] = basis * np.exp(-3.0 * x[valid])[:, np.newaxis]

        return jac

    def total_flux(
        self,
        energy_or_rigidity: ArrayLike,
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Calculate total flux from all element groups.

        Computes the sum of the H, He, O*, and Fe* group fluxes.

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
        values = self._as_1d_values(energy_or_rigidity, "energy or rigidity")
        jacobians = {
            group: self.jacobian(
                values,
                group,
                time_interval=time_interval,
                rigidity_cutoff=rigidity_cutoff,
            )
            for group in self.active_groups
        }
        variance = np.zeros(len(values), dtype=float)
        for group1, jac1 in jacobians.items():
            for group2, jac2 in jacobians.items():
                block = self._covariance_block(group1, group2)
                if block is not None:
                    variance += np.einsum("ni,ij,nj->n", jac1, block, jac2)
        return np.sqrt(np.clip(variance, 0.0, None))

    # ------------------------------------------------------------------
    # Composition helpers (derived from per-group / per-element flux).
    # Added so plotting/comparison can be built on the package natively.
    # ------------------------------------------------------------------
    def fraction(
        self,
        energy_or_rigidity: ArrayLike,
        target: Target,
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
            energy_or_rigidity,
            sid,
            time_interval=time_interval,
            rigidity_cutoff=rigidity_cutoff,
        )

    def _element_lnA_fluxes(
        self, energy_or_rigidity, *, time_interval=None, rigidity_cutoff=None
    ):
        """(ln A, per-species flux array, species-id list) over every species."""
        sids = self.species
        fl = np.array(
            [
                self._flux_one_species(
                    energy_or_rigidity,
                    sid,
                    time_interval=time_interval,
                    rigidity_cutoff=rigidity_cutoff,
                )
                for sid in sids
            ]
        )
        lnA = np.log(np.array([self.mass_number[sid] for sid in sids]))
        return lnA, fl, sids

    def _lnA_moments(self, energy_or_rigidity, time_interval, rigidity_cutoff):
        """(num1=Σf·lnA, num2=Σf·lnA², den=Σf) over all species."""
        lnA, fl, sids = self._element_lnA_fluxes(
            energy_or_rigidity,
            time_interval=time_interval,
            rigidity_cutoff=rigidity_cutoff,
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
            energy_or_rigidity, time_interval, rigidity_cutoff
        )
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
            energy_or_rigidity, time_interval, rigidity_cutoff
        )
        denw = np.where(den > 0, den, np.nan)
        m = num1 / denw
        return num2 / denw - m**2

    # --- covariance propagation for dimensionless composition quantities ---
    def _composition_state(self, x, **kwargs):
        """Return species fluxes and local leader-parameter Jacobians."""
        values = self._as_1d_values(x, "energy or rigidity")
        fluxes = []
        jacobians = []
        leaders = []
        for sid in self.species:
            fluxes.append(self.flux(values, sid, **kwargs))
            jacobians.append(self.jacobian(values, sid, **kwargs))
            leaders.append(self.z_ungroup[sid[0]])
        return np.asarray(fluxes), jacobians, leaders

    def _derived_error(self, weights, jacobians, leaders) -> np.ndarray:
        """Propagate per-species derivatives through leader covariance blocks."""
        grouped = {}
        for weight, jac, leader in zip(weights, jacobians, leaders, strict=True):
            contribution = jac * weight[:, np.newaxis]
            grouped[leader] = grouped.get(leader, 0.0) + contribution
        variance = np.zeros(weights.shape[1])
        for leader1, jac1 in grouped.items():
            sid1 = self._leader_by_charge[leader1]
            for leader2, jac2 in grouped.items():
                sid2 = self._leader_by_charge[leader2]
                block = self.cov.get((sid1, sid2))
                if block is not None:
                    variance += np.einsum("ni,ij,nj->n", jac1, block, jac2)
        return np.sqrt(np.clip(variance, 0.0, None))

    def mean_lnA_error(self, x, **kw):
        """1-sigma band of <lnA> from the stored covariance."""
        fluxes, jacobians, leaders = self._composition_state(x, **kw)
        ln_a = np.log(np.asarray([self.mass_number[sid] for sid in self.species]))[
            :, None
        ]
        total = fluxes.sum(axis=0)
        mean = (fluxes * ln_a).sum(axis=0) / np.where(total > 0, total, np.nan)
        weights = (ln_a - mean) / np.where(total > 0, total, np.nan)
        return self._derived_error(weights, jacobians, leaders)

    def var_lnA_error(self, x, **kw):
        """1-sigma band of Var(ln A) from the stored covariance."""
        fluxes, jacobians, leaders = self._composition_state(x, **kw)
        ln_a = np.log(np.asarray([self.mass_number[sid] for sid in self.species]))[
            :, None
        ]
        total = fluxes.sum(axis=0)
        denominator = np.where(total > 0, total, np.nan)
        mean = (fluxes * ln_a).sum(axis=0) / denominator
        second = (fluxes * ln_a**2).sum(axis=0) / denominator
        weights = ((ln_a**2 - second) - 2.0 * mean * (ln_a - mean)) / denominator
        return self._derived_error(weights, jacobians, leaders)

    def fraction_error(self, x, target, **kw):
        """1-sigma band of a group's flux fraction from the stored covariance."""
        fluxes, jacobians, leaders = self._composition_state(x, **kw)
        zlist, _leader = self._resolve_z(target)
        selected = set(self._target_sids(zlist))
        indicator = np.asarray([sid in selected for sid in self.species])[:, None]
        total = fluxes.sum(axis=0)
        numerator = (fluxes * indicator).sum(axis=0)
        denominator = np.where(total > 0, total, np.nan)
        weights = (indicator * denominator - numerator) / denominator**2
        return self._derived_error(weights, jacobians, leaders)

    def error(
        self,
        energy_or_rigidity: ArrayLike,
        target: Target,
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
        jac = self.jacobian(
            energy_or_rigidity,
            target,
            time_interval=time_interval,
            rigidity_cutoff=rigidity_cutoff,
        )
        block = self._covariance_block(target, target)
        if block is None:
            return np.zeros(jac.shape[0])
        variance = np.einsum("ni,ij,nj->n", jac, block, jac)
        return np.sqrt(np.clip(variance, 0.0, None))

    def _covariance_block(self, target1: Target, target2: Target) -> np.ndarray | None:
        """Return the leader-parameter covariance block for two targets."""
        _zlist1, leader1 = self._resolve_z(target1)
        _zlist2, leader2 = self._resolve_z(target2)
        sid1 = self._leader_by_charge[leader1]
        sid2 = self._leader_by_charge[leader2]
        return self.cov.get((sid1, sid2))

    @abstractmethod
    def flux(
        self,
        energy_or_rigidity: ArrayLike,
        target: Target,
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
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
        target: Target,
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
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

    def covariance(
        self,
        target1: Target,
        target2: Target,
        energy_or_rigidity: ArrayLike,
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Calculate the absolute flux covariance between two targets."""
        jac1 = self.jacobian(
            energy_or_rigidity,
            target1,
            time_interval=time_interval,
            rigidity_cutoff=rigidity_cutoff,
        )
        jac2 = (
            jac1
            if target1 is target2
            else self.jacobian(
                energy_or_rigidity,
                target2,
                time_interval=time_interval,
                rigidity_cutoff=rigidity_cutoff,
            )
        )
        block = self._covariance_block(target1, target2)
        if block is None:
            return np.zeros((jac1.shape[0], jac2.shape[0]))
        return self._propagate_cov(jac1, jac2, block)

    def _amplitude_sample_factor(self) -> tuple[dict, np.ndarray]:
        """Factorize the stacked leader-amplitude covariance for ``sample``.

        Returns a dict mapping each group-leader sid to its column slice in
        the stacked amplitude vector, and a matrix ``F`` with ``F @ F.T``
        equal to the stacked covariance (eigendecomposition, negative
        round-off eigenvalues clipped to zero). Cached on the instance.
        """
        if self._sample_factor_cache is None:
            sids = []
            widths = []
            for group in self.active_groups:
                _zlist, leader = self._resolve_z(group)
                sid = self._leader_by_charge[leader]
                sids.append(sid)
                widths.append(self.cov[(sid, sid)].shape[0])
            edges = np.concatenate([[0], np.cumsum(widths)])
            stacked = np.zeros((edges[-1], edges[-1]))
            for i, sid1 in enumerate(sids):
                for j, sid2 in enumerate(sids):
                    block = self.cov.get((sid1, sid2))
                    if block is not None:
                        stacked[
                            edges[i] : edges[i + 1], edges[j] : edges[j + 1]
                        ] = block
            stacked = 0.5 * (stacked + stacked.T)
            eigenvalues, eigenvectors = np.linalg.eigh(stacked)
            factor = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
            slices = {
                sid: slice(edges[i], edges[i + 1]) for i, sid in enumerate(sids)
            }
            self._sample_factor_cache = (slices, factor)
        return self._sample_factor_cache

    def sample(
        self,
        energy_or_rigidity: ArrayLike,
        target: Target | None = None,
        n_samples: int = 100,
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw random flux realizations from the parameter covariance.

        The flux is linear in the spline amplitudes, so drawing amplitude
        vectors from the fitted covariance and evaluating the model is the
        exact model response: each sample is a genuine model realization
        (a pseudo-experiment), with the full covariance at every energy and
        all cross-group correlations intact. This is the recommended way to
        propagate the model uncertainty into ensembles (Monte Carlo error
        bands, pseudo-experiment studies).

        One shared amplitude draw underlies all targets: calling this
        method several times with an ``rng`` seeded identically returns
        views of the *same* pseudo-experiments, so per-group and
        all-particle samples obtained that way are mutually consistent
        (the group draws sum to the all-particle draw).

        Parameters
        ----------
        energy_or_rigidity
            Input energy or rigidity values. Units depend on subclass.
        target
            Any target accepted by :meth:`flux`, or ``None`` (default) for
            the all-particle total flux.
        n_samples : int, optional
            Number of realizations to draw. Default 100.
        time_interval
            Time period specification, as for :meth:`flux`.
        rigidity_cutoff
            Geomagnetic rigidity cutoff in GV, as for :meth:`flux`.
        rng : numpy.random.Generator, optional
            Source of randomness. Default: a fresh
            ``numpy.random.default_rng()``.

        Returns
        -------
        flux : ndarray, shape (n_samples, n_E)
            One flux realization per row, same units as :meth:`flux`.

        Notes
        -----
        Samples follow the Gaussian parameter covariance of the fit.
        Individual group fluxes can fluctuate below zero where their
        relative uncertainty is of order one (the data-free tails); the
        all-particle flux is protected by the cross-group correlations.

        Examples
        --------
        >>> model = GSFEnergy()
        >>> energy = np.logspace(2, 10, 100)
        >>> realizations = model.sample(energy, "He", n_samples=200)
        >>> band = np.percentile(realizations, [16, 84], axis=0)
        """
        if (
            isinstance(n_samples, bool)
            or not isinstance(n_samples, int | np.integer)
            or n_samples < 1
        ):
            raise ValueError("n_samples must be a positive integer")
        if rng is None:
            rng = np.random.default_rng()
        values = self._as_1d_values(energy_or_rigidity, "energy or rigidity")

        slices, factor = self._amplitude_sample_factor()
        delta = rng.standard_normal((n_samples, factor.shape[0])) @ factor.T

        if target is None:
            central = self.total_flux(
                values,
                time_interval=time_interval,
                rigidity_cutoff=rigidity_cutoff,
            )
            targets = self.active_groups
        else:
            central = self.flux(
                values,
                target,
                time_interval=time_interval,
                rigidity_cutoff=rigidity_cutoff,
            )
            targets = (target,)

        out = np.tile(central, (n_samples, 1))
        for tgt in targets:
            _zlist, leader = self._resolve_z(tgt)
            sid = self._leader_by_charge[leader]
            jac = self.jacobian(
                values,
                tgt,
                time_interval=time_interval,
                rigidity_cutoff=rigidity_cutoff,
            )
            out += delta[:, slices[sid]] @ jac.T
        return out


class GSFEnergy(GSFBase):
    """GSF model evaluated at total energy per nucleus in GeV.

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
        target: Target,  # noqa: ARG002
    ) -> np.ndarray:
        """Transform input energy to total energy per nucleus.

        Subclasses override this to convert from kinetic energy, etc.
        """
        return self._scale_energy(self._as_1d_values(energy, "energy"))

    def _total_energy_for_sid(self, energy: np.ndarray, sid) -> np.ndarray:  # noqa: ARG002
        """Convert prepared input values to total energy for one species."""
        return energy

    def flux(
        self,
        energy: ArrayLike,
        target: Target,
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
        zlist, _group_leader = self._resolve_z(target)
        input_energy = self._transform_energy(energy, target)

        flux = np.zeros_like(input_energy, dtype=float)
        for sid in self._target_sids(zlist):  # charges expand to species (p, D, …)
            total_energy = self._total_energy_for_sid(input_energy, sid)
            mask = self._rigidity_cutoff_mask(sid, total_energy, rigidity_cutoff)
            flux += self._element_flux(sid, total_energy, time_interval) * mask
        return flux

    def jacobian(
        self,
        energy: ArrayLike,
        target: Target,
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Calculate Jacobian of flux for uncertainty propagation."""
        rigidity_cutoff = self._resolve_rigidity_cutoff(rigidity_cutoff)
        zlist, _group_leader = self._resolve_z(target)
        input_energy = self._transform_energy(energy, target)

        jac = 0.0
        for sid in self._target_sids(zlist):
            total_energy = self._total_energy_for_sid(input_energy, sid)
            mask = self._rigidity_cutoff_mask(sid, total_energy, rigidity_cutoff)
            jac += (
                self._element_flux_jacobian(sid, total_energy, time_interval)
                * mask[:, np.newaxis]
            )
        return np.asarray(jac)


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
        self, kinetic_energy: ArrayLike, _target: Target
    ) -> np.ndarray:
        """Prepare kinetic energy; species rest masses are added during summation."""
        return self._scale_energy(self._as_1d_values(kinetic_energy, "kinetic energy"))

    def _total_energy_for_sid(self, energy: np.ndarray, sid) -> np.ndarray:
        """Convert kinetic to total energy using the evaluated species' mass."""
        return energy + self.z_to_a[sid] * NUCLEON_MASS_GEV


class GSFRigidity(GSFBase):
    """GSF model evaluated at magnetic rigidity in GV.

    Solar modulation uses the force-field approximation. ``energy_scale`` has
    no effect because rigidity is the low-energy anchor, not an energy
    coordinate.

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

            # dN/dR force-field prefactor. Public inputs are non-negative.
            Lambda = (E_is / E) * (rigidity / (R_is + _ZERO_GUARD)) ** 3
        return R_is, Lambda

    def flux(
        self,
        rigidity: ArrayLike,
        target: Target,
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
        rigidity = self._as_1d_values(rigidity, "rigidity")

        # φ list and period-average weights (length 1 with 0.0 for LIS)
        phis, phi_weights = self._phi_list(time_interval)

        flux = np.zeros_like(rigidity, dtype=float)
        for phi, w in zip(phis, phi_weights, strict=True):
            for sid in self._target_sids(zlist):  # charges expand to species (p, D, …)
                if phi == 0.0:
                    flux += w * self._rigidity_flux_lis(sid, rigidity)
                else:
                    R_is, fac = self._rigidity_phi_transform(sid, rigidity, phi)
                    flux += w * self._rigidity_flux_lis(sid, R_is) * fac

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
        target: Target,
        *,
        time_interval: tuple[int, int] | str | None = None,
        rigidity_cutoff: float | None = None,
    ) -> np.ndarray:
        """Jacobian including solar modulation."""
        time_interval = self._resolve_time_interval(time_interval)
        rigidity_cutoff = self._resolve_rigidity_cutoff(rigidity_cutoff)
        zlist, _ = self._resolve_z(target)
        rigidity = self._as_1d_values(rigidity, "rigidity")

        phis, phi_weights = self._phi_list(time_interval)
        jac = 0.0
        for phi, w in zip(phis, phi_weights, strict=True):
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
                        return np.exp(slope * np.clip(np.log(R) - xmax, 0.0, dx_sat))

                if phi == 0.0:
                    contrib = ratio * self._rigidity_flux_jacobian(leading, rigidity)
                    if slope != 0.0:
                        contrib = contrib * _tilt(rigidity)[:, None]
                else:
                    R_is, fac = self._rigidity_phi_transform(sid, rigidity, phi)
                    contrib = (
                        ratio
                        * self._rigidity_flux_jacobian(leading, R_is)
                        * (fac * _tilt(R_is))[:, None]
                    )
                jac += w * contrib
        jac = np.asarray(jac)

        # Apply rigidity cutoff (in rigidity space, cutoff is Z-independent)
        if rigidity_cutoff is not None:
            if self.cutoff_width > 0:
                mask = _sigmoid((rigidity - rigidity_cutoff) / self.cutoff_width)
                jac *= mask[:, None]
            else:
                jac[rigidity < rigidity_cutoff, :] = 0.0

        return jac


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
        target: Target,  # noqa: ARG002
    ) -> np.ndarray:
        """Transform input energy per nucleon. Subclasses override for kinetic energy."""
        return self._scale_energy(
            self._as_1d_values(energy_per_nucleon, "energy per nucleon")
        )

    def p_and_n_flux(
        self,
        energy_per_nucleon: ArrayLike,
        target: Target,
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
        zlist, _group_leader = self._resolve_z(target)
        energy_per_nucleon = self._transform_energy_per_nucleon(
            energy_per_nucleon, target
        )

        flux = np.zeros((2, len(energy_per_nucleon)))
        for sid in self._target_sids(zlist):
            charge = sid[0]
            mass_scale = self.z_to_a[sid]
            nucleons = self.mass_number[sid]
            energy = energy_per_nucleon * mass_scale
            mask = self._rigidity_cutoff_mask(sid, energy, rigidity_cutoff)
            fl = self._element_flux(sid, energy, time_interval)
            flux[0] += fl * mass_scale * charge * mask
            flux[1] += fl * mass_scale * (nucleons - charge) * mask
        return flux

    def flux(
        self,
        energy_per_nucleon: ArrayLike,
        target: Target,
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
        target: Target,
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
        zlist, _group_leader = self._resolve_z(target)
        energy_per_nucleon = self._transform_energy_per_nucleon(
            energy_per_nucleon, target
        )

        jac_p = 0.0
        jac_n = 0.0
        for sid in self._target_sids(zlist):
            charge = sid[0]
            mass_scale = self.z_to_a[sid]
            nucleons = self.mass_number[sid]
            energy = energy_per_nucleon * mass_scale
            mask = self._rigidity_cutoff_mask(sid, energy, rigidity_cutoff)
            j = self._element_flux_jacobian(sid, energy, time_interval)
            jac_p += j * (mass_scale * charge * mask)[:, np.newaxis]
            jac_n += j * (mass_scale * (nucleons - charge) * mask)[:, np.newaxis]
        return np.asarray(jac_p), np.asarray(jac_n)

    def jacobian(
        self,
        energy_per_nucleon: ArrayLike,
        target: Target,
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
        target1: Target,
        target2: Target,
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
        cov_key = (
            self._leader_by_charge.get(leader1, leader1),
            self._leader_by_charge.get(leader2, leader2),
        )

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

    def p_and_n_error(
        self,
        energy_per_nucleon: ArrayLike,
        target: Target,
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
        jac_p, jac_n = self.p_and_n_jacobian(
            energy_per_nucleon,
            target,
            time_interval=time_interval,
            rigidity_cutoff=rigidity_cutoff,
        )
        block = self._covariance_block(target, target)
        if block is None:
            return np.zeros((2, jac_p.shape[0]))
        var_p = np.einsum("ni,ij,nj->n", jac_p, block, jac_p)
        var_n = np.einsum("ni,ij,nj->n", jac_n, block, jac_n)
        return np.sqrt(np.clip(np.vstack([var_p, var_n]), 0.0, None))


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
        target: Target,  # noqa: ARG002
    ) -> np.ndarray:
        """Convert kinetic energy per nucleon to total energy per nucleon."""
        return (
            self._scale_energy(
                self._as_1d_values(
                    kinetic_energy_per_nucleon, "kinetic energy per nucleon"
                )
            )
            + NUCLEON_MASS_GEV
        )
