"""Pivot-based reduced representation of the GSF flux uncertainty.

Provides a small set of flux-deformation parameters ``theta`` designed to be
carried as nuisance parameters in downstream analyses (atmospheric lepton
calculations, detector fits): vary one component at a time to build a
Jacobian of your observable, then constrain the components with the exact
``N x N`` covariance penalty of the reduction.

Construction
------------
Each species (total proton and total neutron flux by default; the four
mass-group p/n fluxes with ``per_group=True``) is deformed relative to the
central model at ``n_pivots`` log-spaced pivot energies:

    f_s(E; theta) = f_central,s(E) * (1 + sum_k H_k(log E) theta_{s,k})

where ``H_k`` are cardinal interpolation functions in log-energy — a local
cubic (Catmull-Rom) spline by default, or piecewise-linear hats with
``basis="hat"``.  Either way ``H`` is the identity at the pivots, so the
components ``theta`` are *relative flux deviations at the pivot energies* —
directly interpretable knobs — and their covariance is evaluated **exactly**
from the full GSF parameter covariance at the pivots:

    C = J_rel(pivots) Cov_param J_rel(pivots)^T

At the pivot energies the reduced variance — and every
cross-species/cross-energy correlation between pivots — equals the full
model's.  Between pivots the deformation is interpolated.

Intended use in a fit::

    red = ReducedGSF()
    flux = red.flux(E, theta)                # vary theta -> observable Jacobian
    chi2_penalty = red.penalty(theta)        # theta^T C^-1 theta

References
----------
GSF 2026 supplemental material.  The construction follows the approach used
for the GSF nuisance parameters in daemonflux (Yanez & Fedynitch 2023).
"""

import numpy as np

from .model import (
    ArrayLike,
    GSFEnergyPerNucleon,
    GSFKineticEnergyPerNucleon,
)

_NUCLEON_MODELS = (GSFEnergyPerNucleon, GSFKineticEnergyPerNucleon)
_GROUPS = ("H", "He", "O*", "Fe*")


def _build_stacked_system(model, energy_grid, **kwargs):
    """Build the block Jacobian and parameter covariance for four mass groups."""
    jac_blocks = []
    leader_sids = []
    for group in _GROUPS:
        _charges, leader = model._resolve_z(group)
        leader_sids.append(model._leader_by_charge[leader])
        jac_p, jac_n = model.p_and_n_jacobian(energy_grid, group, **kwargs)
        # Full amplitude range: the fit pins a coefficient by zeroing its
        # covariance row and column (NNLS-zero coefficients, and free ones
        # whose relative sigma exceeds 50 in the data-free extrapolation),
        # so pinned parameters contribute nothing to the contraction.
        jac_blocks.append(np.vstack([jac_p, jac_n]))

    widths = [block.shape[1] for block in jac_blocks]
    rows = sum(block.shape[0] for block in jac_blocks)
    cols = sum(widths)
    jacobian = np.zeros((rows, cols))
    row = col = 0
    for block in jac_blocks:
        nr, nc = block.shape
        jacobian[row : row + nr, col : col + nc] = block
        row += nr
        col += nc

    covariance = np.zeros((cols, cols))
    col1 = 0
    for sid1, width1 in zip(leader_sids, widths, strict=True):
        col2 = 0
        for sid2, width2 in zip(leader_sids, widths, strict=True):
            block = model.cov.get((sid1, sid2))
            if block is not None:
                covariance[col1 : col1 + width1, col2 : col2 + width2] = block
            col2 += width2
        col1 += width1

    return jacobian, covariance


_DEFAULT_ENERGY_RANGE = (1.0, 1e9)


def _format_energy(e: float) -> str:
    """Quotable energy label: 80GeV, 9TeV, 4PeV, 1EeV.

    Three significant figures, so an :func:`optimize_pivots` grid still
    labels readably (``1.52GeV``).
    """
    for unit, scale in (("EeV", 1e9), ("PeV", 1e6), ("TeV", 1e3)):
        if e >= scale:
            return f"{e / scale:.3g}{unit}"
    return f"{e:.3g}GeV"


def _interp_basis(log_pivots: np.ndarray, log_x: np.ndarray, basis: str) -> np.ndarray:
    """Cardinal interpolation matrix H, shape (len(log_x), len(log_pivots)).

    Clamped to the pivot range (edge deformation held constant outside).
    """
    log_x = np.clip(log_x, log_pivots[0], log_pivots[-1])
    n_piv = len(log_pivots)
    if basis == "spline":
        from scipy.interpolate import CubicHermiteSpline

        # Hermite interpolation with finite-difference slopes (Catmull-Rom).
        # Local, because a global cubic on a non-uniform grid rings with side
        # lobes larger than the bump.
        slopes = np.zeros((n_piv, n_piv))
        slopes[0, :2] = [-1.0, 1.0] / (log_pivots[1] - log_pivots[0])
        slopes[-1, -2:] = [-1.0, 1.0] / (log_pivots[-1] - log_pivots[-2])
        for j in range(1, n_piv - 1):
            dx = log_pivots[j + 1] - log_pivots[j - 1]
            slopes[j, j - 1] = -1.0 / dx
            slopes[j, j + 1] = 1.0 / dx
        return CubicHermiteSpline(log_pivots, np.eye(n_piv), slopes, axis=0)(log_x)
    idx = np.clip(np.searchsorted(log_pivots, log_x) - 1, 0, n_piv - 2)
    lo = log_pivots[idx]
    hi = log_pivots[idx + 1]
    t = np.clip((log_x - lo) / (hi - lo), 0.0, 1.0)
    H = np.zeros((len(log_x), n_piv))
    rows = np.arange(len(log_x))
    H[rows, idx] = 1.0 - t
    H[rows, idx + 1] += t
    return H


def _relative_species_system(model, energies: np.ndarray, per_group: bool, **kwargs):
    """Relative species Jacobian rows, parameter covariance, central flux.

    Returns
    -------
    jac_rel : ndarray, shape (n_species * n_E, P)
        Relative-flux Jacobian rows, species-major.
    cov_par : ndarray, shape (P, P)
        Trimmed parameter covariance.
    central : ndarray, shape (n_species * n_E,)
        Central flux, species-major.
    species : list of str
        Species labels.
    """
    n_e = len(energies)
    jac, cov_par = _build_stacked_system(model, energies, **kwargs)

    if per_group:
        species = [f"{g}_{q}" for g in _GROUPS for q in ("p", "n")]
        jac_rows = jac  # already (8 * n_E, P) in species order
        central = np.concatenate(
            [model.p_and_n_flux(energies, g, **kwargs).ravel() for g in _GROUPS]
        )
    else:
        species = ["p", "n"]
        jp = sum(jac[2 * ig * n_e : (2 * ig + 1) * n_e] for ig in range(4))
        jn = sum(jac[(2 * ig + 1) * n_e : (2 * ig + 2) * n_e] for ig in range(4))
        jac_rows = np.vstack([jp, jn])
        central = model.p_and_n_total_flux(energies, **kwargs).ravel()

    if np.any(central <= 0.0):
        bad = np.flatnonzero(central <= 0.0)[0]
        s, k = divmod(bad, n_e)
        raise ValueError(
            f"central flux of species {species[s]!r} vanishes at "
            f"E = {energies[k]:.3g} GeV; shrink the energy range"
        )

    return jac_rows / central[:, np.newaxis], cov_par, central, species


class ReducedGSF:
    """Pivot-component reduction of a GSF nucleon-flux model.

    Parameters
    ----------
    model : GSFEnergyPerNucleon or GSFKineticEnergyPerNucleon, optional
        Nucleon model to reduce. Default: ``GSFEnergyPerNucleon()`` using the
        2026 set and Solar Cycle 24 average.
    n_pivots : int, optional
        Number of log-spaced pivot energies per species.  By default the
        model version's published pivot grid
        (``model.params.reduced_pivots``) is used.
    energy_range : tuple of float, optional
        ``(E_min, E_max)`` of the pivot grid in GeV per nucleon.
        Default ``(1.0, 1e9)`` — beyond ~1e9 the heavy-group fluxes
        underflow and relative deviations lose meaning.
    pivot_energies : array-like, optional
        Explicit pivot energies (overrides ``n_pivots``/``energy_range``).
        Must be strictly increasing.
    per_group : bool, optional
        If False (default), the species are the total proton and total
        neutron flux — sufficient for atmospheric-cascade applications, and
        much lower-dimensional.  If True, every mass group contributes its
        own p and n species (8 species; for composition-sensitive users).
        The eight species come from four leader amplitude blocks, so their
        component covariance is rank-deficient (68 of 96 on the default
        grid) and :meth:`penalty` leaves that nullspace unconstrained.  Use
        it to inspect composition correlations, not as a fit prior.
    basis : {"spline", "hat"}, optional
        Interpolation between pivots (in log-energy).  ``"spline"``
        (default) is a local cubic (Catmull-Rom) spline: smooth (C1)
        deformations, each component confined to its two neighboring
        intervals, at the cost of small side lobes there (the cardinal
        functions dip to ~-0.12).  ``"hat"`` is piecewise-linear: strictly
        local and non-negative, but the deformations are kinked at the
        pivots.  The component covariance is identical for both — only the
        behavior *between* pivots differs, and the coverage of the full
        model's variance is comparable.
    **kwargs
        Passed to model methods (e.g. ``time_interval``,
        ``rigidity_cutoff``).

    Attributes
    ----------
    pivot_energies : ndarray, shape (N,)
        The pivot grid.
    species : list of str
        Species row labels: ``["p", "n"]``, or ``["H_p", "H_n", "He_p", ...]``
        (group then p/n) with ``per_group=True``.
    n_params : int
        ``len(species) * N`` — the length of ``theta``.
    labels : list of str
        One quotable name per component (``"p_9TeV"``, ``"n_30PeV"``, ...),
        in ``theta`` order (species-major: all pivots of species 0, then
        species 1, ...).
    cov : ndarray, shape (n_params, n_params)
        Exact covariance of ``theta`` (relative-flux units).
    sigma : ndarray, shape (n_params,)
        ``sqrt(diag(cov))`` — the 1-sigma prior width of each component.
    correlation : ndarray, shape (n_params, n_params)
        The correlation matrix of ``theta``.

    Examples
    --------
    >>> from globalsplinefit.reduced import ReducedGSF
    >>> red = ReducedGSF()                    # 24 parameters (2 x 12)
    >>> E = np.logspace(1, 6, 50)
    >>> f_central = red.flux(E)               # (2, 50): [p, n], theta = 0
    >>> theta = np.zeros(red.n_params)
    >>> theta[3] = red.sigma[3]               # +1 sigma on one component
    >>> f_varied = red.flux(E, theta)
    >>> chi2 = red.penalty(theta)             # Gaussian prior contribution
    """

    def __init__(
        self,
        model: GSFEnergyPerNucleon | None = None,
        n_pivots: int | None = None,
        energy_range: tuple[float, float] = (1.0, 1e9),
        pivot_energies: ArrayLike | None = None,
        per_group: bool = False,
        basis: str = "spline",
        **kwargs,
    ):
        if model is None:
            model = GSFEnergyPerNucleon()
        if not isinstance(model, _NUCLEON_MODELS):
            raise TypeError(
                "ReducedGSF requires a nucleon model "
                "(GSFEnergyPerNucleon or GSFKineticEnergyPerNucleon), "
                f"got {type(model).__name__}"
            )
        if basis not in ("spline", "hat"):
            raise ValueError(f"basis must be 'spline' or 'hat', got {basis!r}")

        if pivot_energies is None:
            if n_pivots is None and energy_range == _DEFAULT_ENERGY_RANGE:
                pivot_energies = model.params.reduced_pivots
                if pivot_energies is None:
                    raise ValueError(
                        "this model bundle has no reduced_pivots.dat. Derive "
                        "a grid once with optimize_pivots(model, n_pivots=12) "
                        "and pass it via pivot_energies= (or store it as "
                        "reduced_pivots.dat in the bundle directory), or "
                        "request a log-spaced grid with n_pivots=."
                    )
            else:
                if n_pivots is not None and (
                    isinstance(n_pivots, bool)
                    or not isinstance(n_pivots, (int, np.integer))
                    or n_pivots < 2
                ):
                    raise ValueError("n_pivots must be an integer >= 2")
                if (
                    len(energy_range) != 2
                    or not np.all(np.isfinite(energy_range))
                    or energy_range[0] <= 0
                    or energy_range[0] >= energy_range[1]
                ):
                    raise ValueError(
                        "energy_range must be two positive increasing values"
                    )
                pivot_energies = np.logspace(
                    np.log10(energy_range[0]),
                    np.log10(energy_range[1]),
                    n_pivots if n_pivots is not None else 10,
                )
        pivot_energies = np.array(pivot_energies, dtype=float, copy=True)
        if (
            pivot_energies.ndim != 1
            or len(pivot_energies) < 2
            or not np.all(np.isfinite(pivot_energies))
            or np.any(pivot_energies <= 0)
            or np.any(np.diff(pivot_energies) <= 0)
        ):
            raise ValueError(
                "pivot_energies must be at least two positive, finite, increasing values"
            )

        self.model = model
        self.per_group = per_group
        self.pivot_energies = pivot_energies
        self.basis_type = basis
        self._log_pivots = np.log(pivot_energies)
        self._kwargs = dict(kwargs)

        n_piv = len(pivot_energies)
        jac_rel, cov_par, central, self.species = _relative_species_system(
            model, pivot_energies, per_group, **kwargs
        )
        covariance = jac_rel @ cov_par @ jac_rel.T
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        tolerance = (
            np.finfo(float).eps
            * max(covariance.shape)
            * max(float(np.max(np.abs(eigenvalues))), 1.0)
        )
        if np.min(eigenvalues) < -100 * tolerance:
            raise ValueError("reduced covariance is not positive semidefinite")
        self.cov = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        self._central_pivots = central.reshape(len(self.species), n_piv)
        self._precision = None
        self._sample_factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))

        self.n_params = len(self.species) * n_piv
        self.labels = [
            f"{s}_{_format_energy(e)}" for s in self.species for e in pivot_energies
        ]
        for array in (
            self.pivot_energies,
            self._log_pivots,
            self.cov,
            self._central_pivots,
            self._sample_factor,
        ):
            array.setflags(write=False)

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    @property
    def sigma(self) -> np.ndarray:
        """1-sigma prior width of each component (relative flux units)."""
        return np.sqrt(np.diag(self.cov))

    @property
    def correlation(self) -> np.ndarray:
        """Correlation matrix of the components."""
        s = self.sigma
        denominator = np.outer(s, s)
        return np.divide(
            self.cov,
            denominator,
            out=np.zeros_like(self.cov),
            where=denominator > 0,
        )

    # ------------------------------------------------------------------
    # Basis
    # ------------------------------------------------------------------

    def basis(self, energy: ArrayLike) -> np.ndarray:
        """Interpolation basis H, shape (n_E, N).

        See the ``basis`` constructor parameter.
        """
        energy = self.model._as_1d_values(energy, "energy", positive=True)
        log_e = np.log(energy)
        return _interp_basis(self._log_pivots, log_e, self.basis_type)

    # ------------------------------------------------------------------
    # Model evaluation
    # ------------------------------------------------------------------

    def _effective_kwargs(self, overrides: dict) -> dict:
        for key, value in overrides.items():
            if key not in self._kwargs or self._kwargs[key] != value:
                raise ValueError(
                    f"cannot override {key!r} on an existing ReducedGSF; "
                    "construct a new reduction for different physical settings"
                )
        return self._kwargs

    def _central_flux(self, energy: np.ndarray, **kwargs) -> np.ndarray:
        """Central flux per species, shape (S, n_E)."""
        kw = self._effective_kwargs(kwargs)
        if self.per_group:
            return np.vstack(
                [self.model.p_and_n_flux(energy, g, **kw) for g in _GROUPS]
            )
        return np.asarray(self.model.p_and_n_total_flux(energy, **kw))

    def flux(
        self,
        energy: ArrayLike,
        theta: ArrayLike | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Deformed flux for parameter vector ``theta``.

        Parameters
        ----------
        energy : array-like
            Energy per nucleon in GeV.
        theta : array-like, shape (n_params,), optional
            Component values (relative deviations at the pivots).
            ``None`` (default) returns the central flux.
        **kwargs
            Override kwargs for model methods.

        Returns
        -------
        flux : ndarray, shape (S, n_E)
            One row per entry of :attr:`species`.
        """
        energy = self.model._as_1d_values(energy, "energy", positive=True)
        central = self._central_flux(energy, **kwargs)
        if theta is None:
            return central
        theta = np.asarray(theta, dtype=float)
        if (
            theta.ndim != 1
            or theta.size != self.n_params
            or not np.all(np.isfinite(theta))
        ):
            raise ValueError(f"theta must be a finite vector of length {self.n_params}")
        theta = theta.reshape(len(self.species), len(self.pivot_energies))
        H = self.basis(energy)
        return central * (1.0 + theta @ H.T)

    def flux_jacobian(self, energy: ArrayLike, **kwargs) -> np.ndarray:
        """Compute the derivative of :meth:`flux` w.r.t. ``theta``.

        Returns
        -------
        jac : ndarray, shape (S, n_E, n_params)
            ``jac[s, i, j] = d flux[s, i] / d theta[j]``.  Species ``s``
            only responds to its own block of components.
        """
        energy = self.model._as_1d_values(energy, "energy", positive=True)
        central = self._central_flux(energy, **kwargs)
        H = self.basis(energy)
        n_s = len(self.species)
        n_piv = len(self.pivot_energies)
        jac = np.zeros((n_s, len(energy), self.n_params))
        for s in range(n_s):
            jac[s, :, s * n_piv : (s + 1) * n_piv] = central[s][:, np.newaxis] * H
        return jac

    def error(self, energy: ArrayLike, **kwargs) -> np.ndarray:
        """Absolute 1-sigma flux uncertainty of the reduced model.

        Exact at the pivot energies and interpolated with the selected basis.

        Returns
        -------
        sigma : ndarray, shape (S, n_E)
        """
        energy = self.model._as_1d_values(energy, "energy", positive=True)
        central = self._central_flux(energy, **kwargs)
        H = self.basis(energy)
        n_piv = len(self.pivot_energies)
        out = np.empty_like(central)
        for s in range(len(self.species)):
            block = self.cov[s * n_piv : (s + 1) * n_piv, s * n_piv : (s + 1) * n_piv]
            out[s] = np.sqrt(np.einsum("ij,jk,ik->i", H, block, H))
        return central * out

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def sample(
        self,
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw component vectors ``theta ~ N(0, cov)``.

        Returns
        -------
        theta : ndarray, shape (n_samples, n_params)
        """
        if rng is None:
            rng = np.random.default_rng()
        if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
            raise ValueError("n_samples must be a positive integer")
        if n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        return rng.standard_normal((n_samples, self.n_params)) @ self._sample_factor.T

    def penalty(self, theta: ArrayLike) -> float:
        """Gaussian penalty ``theta^T cov^-1 theta`` for a fit.

        Add this to the fit's chi-square to constrain the components to the
        GSF uncertainty.  The default (p, n) covariance is full rank.  With
        ``per_group=True`` it is not, and the pseudo-inverse charges nothing
        along the nullspace, so a fit is free to move there.
        """
        theta = np.asarray(theta, dtype=float)
        if (
            theta.ndim != 1
            or theta.size != self.n_params
            or not np.all(np.isfinite(theta))
        ):
            raise ValueError(f"theta must be a finite vector of length {self.n_params}")
        if self._precision is None:
            self._precision = np.linalg.pinv(self.cov, hermitian=True)
        return float(theta @ self._precision @ theta)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-serializable description of the reduction."""
        return {
            "description": (
                "GSF reduced flux representation: theta are relative flux "
                "deviations at the pivot energies, interpolated in log(E) "
                f"with a {self.basis_type!r} basis; "
                "penalty = theta^T cov^-1 theta"
            ),
            "basis": self.basis_type,
            "model_version": self.model.version,
            "species": list(self.species),
            "pivot_energies_GeV": self.pivot_energies.tolist(),
            "labels": list(self.labels),
            "central_flux_at_pivots": self._central_pivots.tolist(),
            "cov": self.cov.tolist(),
        }


def optimize_pivots(
    model: GSFEnergyPerNucleon | None = None,
    n_pivots: int = 12,
    energy_range: tuple[float, float] = (1.0, 1e9),
    per_group: bool = False,
    basis: str = "spline",
    n_grid: int = 300,
    n_restarts: int = 2,
    max_sweeps: int = 40,
    min_separation: float = 0.15,
    seed: int = 0,
    **kwargs,
) -> tuple[np.ndarray, float]:
    """Optimize pivot placement for :class:`ReducedGSF` coverage.

    Minimizes the worst-case mismatch between the reduced and the exact
    flux uncertainty over a dense log-energy grid,

        max over (species, E) of |log(sigma_reduced / sigma_exact)|,

    by coordinate exchange: candidate pivots are snapped to the dense grid,
    so the pivot covariance of every trial is a submatrix of one
    precomputed dense-grid covariance and each trial costs only linear
    algebra — no model re-evaluations.  One pivot at a time is moved to its
    best available grid slot; sweeps repeat until no move improves the
    objective.  The search restarts from the log-spaced grid and
    ``n_restarts`` random configurations, keeping the best result.  The
    endpoint pivots stay fixed at ``energy_range``.

    Two guards keep the search honest: candidate pivots live on every
    *second* grid point, so the objective always samples between any two
    pivots (otherwise the exchange can hide spline ringing between its own
    evaluation points), and pivots must stay ``min_separation`` decades
    apart (near-duplicate pivots are statistically useless and make the
    cardinal spline ring violently).

    Roughly half a minute with the defaults; scales as
    ``n_pivots * n_grid * n_restarts``.  The result is deterministic for a
    given ``seed``.

    Parameters
    ----------
    model, energy_range, per_group, basis, **kwargs
        As for :class:`ReducedGSF`.
    n_pivots : int, optional
        Number of pivots to place.  Default 12.
    n_grid : int, optional
        Dense-grid resolution; pivots are quantized to this grid.
        Default 300 (about 0.03 decades over the default range).
    n_restarts : int, optional
        Random restarts in addition to the log-spaced start.  Default 2.
    max_sweeps : int, optional
        Maximum exchange sweeps per start.  Default 40.
    min_separation : float, optional
        Minimum pivot separation in decades.  Default 0.15.
    seed : int, optional
        Seed for the restart configurations.  Default 0.

    Returns
    -------
    pivot_energies : ndarray, shape (n_pivots,)
        Optimized pivot grid — pass to ``ReducedGSF(pivot_energies=...)``
        (with the same ``basis``, ``per_group``, and model kwargs).
    max_ratio : float
        Achieved worst-case coverage factor: ``sigma_reduced/sigma_exact``
        lies within ``[1/max_ratio, max_ratio]`` on the dense grid.

    Examples
    --------
    >>> pivots, worst = optimize_pivots(n_pivots=12)
    >>> red = ReducedGSF(pivot_energies=pivots)
    """
    if model is None:
        model = GSFEnergyPerNucleon()
    if not isinstance(model, _NUCLEON_MODELS):
        raise TypeError(
            "optimize_pivots requires a nucleon model "
            "(GSFEnergyPerNucleon or GSFKineticEnergyPerNucleon), "
            f"got {type(model).__name__}"
        )
    if basis not in ("spline", "hat"):
        raise ValueError(f"basis must be 'spline' or 'hat', got {basis!r}")
    if not 3 <= n_pivots < n_grid // 2:
        raise ValueError(f"n_pivots must be in [3, n_grid/2), got {n_pivots}")

    energies = np.logspace(np.log10(energy_range[0]), np.log10(energy_range[1]), n_grid)
    log_x = np.log(energies)
    jac_rel, cov_par, _, species = _relative_species_system(
        model, energies, per_group, **kwargs
    )
    S = jac_rel @ cov_par @ jac_rel.T
    sig2_exact = np.diag(S).reshape(len(species), n_grid)

    span_decades = np.log10(energy_range[1] / energy_range[0])
    min_sep = max(2, int(np.ceil(min_separation * (n_grid - 1) / span_decades)))
    candidates = np.arange(2, n_grid - 1, 2)
    if (n_pivots - 1) * min_sep >= n_grid - 1:
        raise ValueError(
            f"cannot place {n_pivots} pivots {min_separation} decades apart "
            f"within {span_decades:.2g} decades"
        )

    def objective(idx: np.ndarray) -> float:
        H = _interp_basis(log_x[idx], log_x, basis)
        worst = 0.0
        for s in range(len(species)):
            rows = s * n_grid + idx
            var = np.einsum("ij,jk,ik->i", H, S[np.ix_(rows, rows)], H)
            dev = np.abs(0.5 * np.log(var / sig2_exact[s]))
            worst = max(worst, float(dev.max()))
        return worst

    def exchange(idx: np.ndarray) -> tuple[np.ndarray, float]:
        best = objective(idx)
        for _ in range(max_sweeps):
            improved = False
            for j in range(1, n_pivots - 1):
                others = np.delete(idx, j)
                free = candidates[
                    np.min(np.abs(candidates[:, None] - others[None, :]), axis=1)
                    >= min_sep
                ]
                if len(free) == 0:
                    continue
                vals = [objective(np.sort(np.append(others, c))) for c in free]
                k = int(np.argmin(vals))
                if vals[k] < best - 1e-12:
                    idx = np.sort(np.append(others, free[k]))
                    best = vals[k]
                    improved = True
            if not improved:
                break
        return idx, best

    def random_start(rng: np.random.Generator) -> np.ndarray:
        # sequential rejection keeps the separation constraint satisfied
        idx = [0, n_grid - 1]
        pool = list(candidates)
        while len(idx) < n_pivots and pool:
            c = pool[rng.integers(len(pool))]
            if all(abs(c - i) >= min_sep for i in idx):
                idx.append(c)
            pool.remove(c)
        if len(idx) < n_pivots:
            raise ValueError("could not place pivots with the given separation")
        return np.sort(np.array(idx))

    rng = np.random.default_rng(seed)
    log_start = np.unique(np.round(np.linspace(0, n_grid - 1, n_pivots)).astype(int))
    starts = [log_start]
    starts += [random_start(rng) for _ in range(n_restarts)]

    results = [exchange(idx) for idx in starts]
    best_idx, best = min(results, key=lambda t: t[1])
    return energies[best_idx], float(np.exp(best))
