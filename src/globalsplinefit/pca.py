"""Hybrid PCA dimensionality reduction for GSF models.

Reduces the retained interior of the spline-amplitude covariance to a small
number of uncorrelated latent parameters phi ~ N(0, I), while preserving the
exact covariance between all mass groups at every reference-grid energy via
a per-energy cross-group block correction.  The first and final three
coefficients of every group are omitted; this limits the reduction in the
data-free high-energy tail.

Reduced relative covariance:

    Cov[delta f_i / f_i, delta f_j / f_j]
        = M(E_i) M(E_j)^T + delta_ij B(E_i)

where M(E) = J_rel(E) @ L_param is the reduced relative Jacobian,
exact at any energy (no interpolation).

Construction
------------
On a reference energy grid the relative flux covariance
``Sigma = J_rel C J_rel^T`` (C = retained interior parameter covariance) is
reduced in two steps:

1. **Low-rank factor.** With the default ``gauge="correlation"`` the
   *correlation* matrix ``R = D_s^-1/2 Sigma D_s^-1/2`` is diagonalized and
   the leading ``n_components`` eigenmodes are un-whitened into a factor
   ``L`` (``Sigma ~ L L^T``).  Diagonalizing R weights every grid row
   equally, so components resolve correlation structure across the whole
   energy range instead of being spent on the few rows with the largest
   relative variance (the data-free heavy-group region, which dominates
   the trace of Sigma).  ``gauge="covariance"`` diagonalizes Sigma itself
   (the classic PCA; retained as an option).  Either factor is projected
   to parameter space by least squares (``J_rel L_param = L`` exactly).

2. **Block correction.** The residual ``Sigma - L L^T`` is kept exactly on
   the block-diagonal over energy: for every grid energy the full
   cross-group residual block (4x4 per-particle, 8x8 with the p/n split
   for nucleon models) is stored, projected to the nearest positive
   semi-definite matrix.  Any observable formed from fluxes at a single
   energy — each group band, the all-particle flux, the total nucleon
   flux, lnA-type sums — then has exact variance on the grid regardless
   of ``n_components`` and gauge.  Only correlations *between different
   energies* beyond the retained components are truncated; those matter
   for multi-energy functionals and smooth sampling, which is what
   ``n_components`` buys.

The scalar diagonal correction D and the p-n cross term D_pn of the
original construction (GSF 2025 analysis) are the diagonal entries of the
block; they remain available as read-only properties.

Monte Carlo samples include both terms by default and therefore draw from
the same covariance returned by ``covariance`` and ``error``.  The block
term is independent between requested energies and can look like bin-to-bin
jitter.  Pass ``residual_noise=False`` only when a smooth, low-rank-only
spectral deformation is desired.

References
----------
GSF 2026 supplemental material, "Reduced representation of the
covariance".  The original diagonal construction followed the hybrid
approach of the GSF 2025 analysis.
"""

import numpy as np

from .model import (
    ArrayLike,
    GSFBase,
    GSFEnergyPerNucleon,
    GSFKineticEnergyPerNucleon,
)

_NUCLEON_MODELS = (GSFEnergyPerNucleon, GSFKineticEnergyPerNucleon)

# Groups in canonical order, matching the model's active_groups
_GROUPS = ["p", "He", "O*", "Fe*"]


def _build_block_diagonal_jacobian(model, energy_grid, **kwargs):
    """Build block-diagonal Jacobian and central flux across all groups.

    Parameters
    ----------
    model : GSFBase
        Any GSF model instance.
    energy_grid : np.ndarray
        Energy grid.
    **kwargs
        Passed to model Jacobian/flux methods (time_interval, rigidity_cutoff).

    Returns
    -------
    jac_stack : ndarray, shape (M, P)
        Stacked trimmed Jacobian.
    central_flux : ndarray, shape (M,)
        Central flux at the grid.
    group_leaders : list of int
        Z values of the 4 group leaders, in order.
    npar_trimmed : list of int
        Number of trimmed parameters per group.
    """
    is_nucleon = isinstance(model, _NUCLEON_MODELS)

    jac_blocks = []
    flux_parts = []
    group_leaders = []
    npar_trimmed_list = []

    for g in _GROUPS:
        z = model.GROUP_NAMES[g]
        group_leaders.append(z)

        if is_nucleon:
            jp, jn = model.p_and_n_jacobian(energy_grid, g, **kwargs)
            jac_blocks.append(np.vstack([jp[:, 1:-3], jn[:, 1:-3]]))
            pn = model.p_and_n_flux(energy_grid, g, **kwargs)
            flux_parts.append(pn.ravel())
        else:
            jac = model.jacobian(energy_grid, g, **kwargs)
            jac_blocks.append(jac[:, 1:-3])
            flux_parts.append(model.flux(energy_grid, g, **kwargs))

        npar_trimmed_list.append(jac_blocks[-1].shape[1])

    # Assemble block-diagonal Jacobian (rows stacked, columns block-diagonal)
    n_rows = sum(j.shape[0] for j in jac_blocks)
    n_cols = sum(j.shape[1] for j in jac_blocks)
    jac_stack = np.zeros((n_rows, n_cols))

    row_offset = 0
    col_offset = 0
    for j in jac_blocks:
        r, c = j.shape
        jac_stack[row_offset : row_offset + r, col_offset : col_offset + c] = j
        row_offset += r
        col_offset += c

    central_flux = np.concatenate(flux_parts)
    return jac_stack, central_flux, group_leaders, npar_trimmed_list


def _build_stacked_system(model, energy_grid, **kwargs):
    """Build stacked Jacobian, covariance, parameters, and central flux.

    Assembles the block system across all 4 element groups.

    Parameters
    ----------
    model : GSFBase
        Any GSF model instance.
    energy_grid : np.ndarray
        Reference energy grid.
    **kwargs
        Passed to model Jacobian/flux methods (time_interval, rigidity_cutoff).

    Returns
    -------
    jac_stack : ndarray, shape (M, P)
        Stacked trimmed Jacobian (rows over energies/nucleons, columns over parameters).
    cov_stack : ndarray, shape (P, P)
        Block covariance matrix of trimmed parameters.
    par_stack : ndarray, shape (P,)
        Stacked trimmed parameter vector.
    central_flux : ndarray, shape (M,)
        Central flux at the reference grid.
    group_leaders : list of int
        Z values of the 4 group leaders, in order.
    npar_trimmed : list of int
        Number of trimmed parameters per group.
    """
    jac_stack, central_flux, group_leaders, npar_trimmed_list = (
        _build_block_diagonal_jacobian(model, energy_grid, **kwargs)
    )

    # Trimmed parameter vectors (key by the group's leader species id (Z, A), so
    # this works when a charge carries >1 species, e.g. p+D at Z=1).
    par_vecs = [
        model.pars[model._leader_by_charge[model.GROUP_NAMES[g]]][1:-7] for g in _GROUPS
    ]
    par_stack = np.hstack(par_vecs)

    # Assemble block covariance matrix
    n_cols = jac_stack.shape[1]
    cov_stack = np.zeros((n_cols, n_cols))
    col_i = 0
    for i, zi in enumerate(group_leaders):
        ni = npar_trimmed_list[i]
        col_j = 0
        for j, zj in enumerate(group_leaders):
            nj = npar_trimmed_list[j]
            # covariance is keyed by leader species ids (Z, A), not bare charge
            cov_key = (model._leader_by_charge[zi], model._leader_by_charge[zj])
            if cov_key in model.cov:
                cov_stack[col_i : col_i + ni, col_j : col_j + nj] = model.cov[cov_key][
                    1:-3, 1:-3
                ]
            col_j += nj
        col_i += ni

    return (
        jac_stack,
        cov_stack,
        par_stack,
        central_flux,
        group_leaders,
        npar_trimmed_list,
    )


def _build_jacobian_at_energy(model, energy, **kwargs):
    """Build stacked trimmed Jacobian at arbitrary energy.

    Same block structure as _build_stacked_system but only returns the Jacobian
    and central flux.
    """
    jac_stack, central_flux, _, _ = _build_block_diagonal_jacobian(
        model, energy, **kwargs
    )
    return jac_stack, central_flux


def _group_index(target: str | int | list[int], model: GSFBase) -> int:
    """Return the index (0-3) of the group that target belongs to."""
    if isinstance(target, str):
        z = model.GROUP_NAMES[target]
    elif isinstance(target, list | tuple | np.ndarray):
        z = model.z_ungroup[int(target[0])]
    else:
        z = model.z_ungroup[int(target)]
    return [model.GROUP_NAMES[g] for g in _GROUPS].index(z)


def _nearest_psd(mat: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix to the nearest PSD matrix (eigenvalue clip)."""
    w, u = np.linalg.eigh(mat)
    return (u * np.maximum(w, 0.0)) @ u.T


class HybridPCA:
    """Hybrid PCA dimensionality reduction for any GSF model.

    Provides the same interface as the underlying model (``flux``, ``error``,
    ``covariance``, ``jacobian``) but uses the low-rank + per-energy-block
    approximation for uncertainty propagation.  Flux values are delegated
    unchanged to the underlying model; only the uncertainty methods use the
    reduction.

    Parameters
    ----------
    model : GSFBase
        Any GSF model instance (GSFEnergy, GSFRigidity, GSFEnergyPerNucleon, etc.).
    n_components : int, optional
        Number of retained components. Default 8.  Fixed-energy observables
        (group bands, all-particle flux, total nucleon flux) are exact on the
        reference grid for ANY value; ``n_components`` controls the fidelity
        of cross-energy correlations and of smooth samples.
    energy_grid : array-like, optional
        Reference energy grid for the decomposition.
        Default: 300 log-spaced points from 1 to 10^11 GeV.
    gauge : {"correlation", "covariance"}, optional
        Metric in which the low-rank factor is extracted.  The default
        "correlation" diagonalizes the correlation matrix of the relative
        flux covariance (every grid row weighted equally); "covariance"
        diagonalizes the covariance itself (classic PCA, dominated by the
        rows with the largest relative variance).
    **kwargs
        Passed to model methods (e.g. ``time_interval``, ``rigidity_cutoff``).

    Attributes
    ----------
    L_param : ndarray, shape (n_params, n_components)
        Parameter-space low-rank factor.
    B : ndarray, shape (n_ref, n_sub, n_sub)
        Per-energy cross-group residual blocks on the reference grid
        (n_sub = 4, or 8 for nucleon models: p/n per group), PSD-projected.
    n_components : int
        Number of retained components.
    variance_explained : float
        Fraction of the total variance in the chosen gauge captured by the
        low-rank part (in the correlation gauge: average fraction of the
        per-row variance).

    Examples
    --------
    >>> from globalsplinefit import GSFEnergyPerNucleon
    >>> from globalsplinefit.pca import HybridPCA
    >>> gsf = GSFEnergyPerNucleon()          # promoted default (GSF2026)
    >>> pca = HybridPCA(gsf, n_components=8)
    >>> E = np.logspace(1, 6, 50)
    >>> flux = pca.flux(E, "p")             # same as gsf.flux(E, "p")
    >>> sigma = pca.error(E, "p")           # absolute 1-sigma (like gsf.error)
    >>> cov = pca.covariance("p", "p", E)   # absolute covariance matrix
    """

    def __init__(
        self,
        model: GSFBase,
        n_components: int = 8,
        energy_grid: np.ndarray | list | float | None = None,
        gauge: str = "correlation",
        **kwargs,
    ):
        if gauge not in ("correlation", "covariance"):
            raise ValueError(
                f"gauge must be 'correlation' or 'covariance', got {gauge!r}"
            )
        if energy_grid is None:
            energy_grid = np.logspace(np.log10(1.0), 11, 300)
        energy_grid = np.atleast_1d(np.asarray(energy_grid, dtype=float))

        self.model = model
        self.n_components = n_components
        self.gauge = gauge
        self._energy_grid = energy_grid
        self._kwargs = kwargs
        self._is_nucleon_model = isinstance(model, _NUCLEON_MODELS)

        # Build the stacked system on the reference grid
        (jac_stack, cov_stack, par_stack, central_flux, group_leaders, npar_trimmed) = (
            _build_stacked_system(model, energy_grid, **kwargs)
        )

        self.par_stack = par_stack
        self.central_flux = central_flux
        self._group_leaders = group_leaders
        self._npar_trimmed = npar_trimmed

        # Mask out rows with zero central flux (e.g. neutron flux from H group)
        nonzero = central_flux != 0.0
        self._nonzero_mask = nonzero

        # Relative Jacobian: J_rel = J / f_central (only on non-zero rows)
        rel_jac = np.zeros_like(jac_stack)
        rel_jac[nonzero] = jac_stack[nonzero] / central_flux[nonzero, np.newaxis]

        # Relative flux covariance
        flux_rel_cov = rel_jac @ cov_stack @ rel_jac.T

        # Low-rank factor in the chosen gauge
        k = n_components
        if gauge == "correlation":
            s = np.sqrt(np.maximum(np.diag(flux_rel_cov), 0.0))
            snz = s > 0
            corr = np.zeros_like(flux_rel_cov)
            corr[np.ix_(snz, snz)] = flux_rel_cov[np.ix_(snz, snz)] / np.outer(
                s[snz], s[snz]
            )
            eigvals, eigvecs = np.linalg.eigh(corr)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            # un-whiten the retained factor back to covariance scale
            L = (s[:, np.newaxis] * eigvecs[:, :k]) * np.sqrt(
                np.maximum(eigvals[:k], 0.0)
            )
        else:
            eigvals, eigvecs = np.linalg.eigh(flux_rel_cov)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            L = eigvecs[:, :k] * np.sqrt(np.maximum(eigvals[:k], 0.0))

        self._eigvals = eigvals

        # Project to parameter space via least-squares (robust to rank-deficiency
        # when the energy grid has fewer points than parameters)
        self.L_param, _, _, _ = np.linalg.lstsq(rel_jac, L, rcond=None)

        # Store layout info
        self._ref_log_energy = np.log(energy_grid)
        self._n_ref = len(energy_grid)
        self._n_groups = len(_GROUPS)
        self._n_sub_per_group = 2 if self._is_nucleon_model else 1
        self._n_sub = self._n_groups * self._n_sub_per_group

        # Precompute row layout: rows_per_group for the reference grid
        self._rows_per_group_ref = (
            2 * self._n_ref if self._is_nucleon_model else self._n_ref
        )

        # Per-energy cross-group residual blocks: everything the low-rank part
        # misses BETWEEN sub-rows (group x p/n) at the SAME energy, kept
        # exactly (PSD-projected).  Cross-energy residuals are dropped.
        L_grid = rel_jac @ self.L_param
        n_ref = self._n_ref
        nsub = self._n_sub
        B = np.empty((n_ref, nsub, nsub))
        sub_offsets = self._sub_row_offsets(n_ref)
        for i in range(n_ref):
            rows = sub_offsets + i
            resid = flux_rel_cov[np.ix_(rows, rows)] - L_grid[rows] @ L_grid[rows].T
            B[i] = _nearest_psd(0.5 * (resid + resid.T))
        self.B = B

    def _sub_row_offsets(self, n_e: int) -> np.ndarray:
        """Row offset of each sub-row (group-major, p before n) for n_e energies."""
        rpg = self._n_sub_per_group * n_e
        return np.array(
            [
                ig * rpg + q * n_e
                for ig in range(self._n_groups)
                for q in range(self._n_sub_per_group)
            ]
        )

    # ------------------------------------------------------------------
    # Backward-compatible views of the block correction
    # ------------------------------------------------------------------

    @property
    def D(self) -> np.ndarray:
        """Diagonal of the block correction in stacked-row layout.

        Equivalent to the scalar diagonal correction of the original
        construction (relative variance not carried by the low-rank part).
        """
        n_ref = self._n_ref
        D = np.empty(self._n_sub * n_ref)
        offsets = self._sub_row_offsets(n_ref)
        for s, off in enumerate(offsets):
            D[off : off + n_ref] = self.B[:, s, s]
        return D

    @property
    def D_pn(self) -> np.ndarray | None:
        """p-n cross-term correction per group (nucleon models), from the block."""
        if not self._is_nucleon_model:
            return None
        n_ref = self._n_ref
        D_pn = np.empty(self._n_groups * n_ref)
        for ig in range(self._n_groups):
            D_pn[ig * n_ref : (ig + 1) * n_ref] = self.B[:, 2 * ig, 2 * ig + 1]
        return D_pn

    @property
    def variance_explained(self) -> float:
        """Fraction of gauge-metric variance captured by the low-rank part.

        In the correlation gauge this is the average fraction of the
        per-row variance resolved by the components; in the covariance
        gauge, the classic explained-variance fraction of the trace.
        """
        total = np.sum(np.maximum(self._eigvals, 0.0))
        captured = np.sum(np.maximum(self._eigvals[: self.n_components], 0.0))
        return float(captured / total) if total > 0 else 1.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reduced_jacobian_full(
        self, energy: np.ndarray, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute stacked reduced relative Jacobian M and central flux."""
        kw = {**self._kwargs, **kwargs}
        jac_stack, central_flux = _build_jacobian_at_energy(self.model, energy, **kw)
        nonzero = central_flux != 0.0
        rel_jac = np.zeros_like(jac_stack)
        rel_jac[nonzero] = jac_stack[nonzero] / central_flux[nonzero, np.newaxis]
        M = rel_jac @ self.L_param
        return M, central_flux

    def _interpolate_B(self, energy: np.ndarray) -> np.ndarray:
        """Interpolate the residual blocks to arbitrary energies.

        Linear interpolation in log-energy; a convex combination of PSD
        blocks is PSD.  Outside the reference grid the edge block is held
        constant.
        """
        log_e = np.log(energy)
        idx = np.clip(
            np.searchsorted(self._ref_log_energy, log_e) - 1, 0, self._n_ref - 2
        )
        lo = self._ref_log_energy[idx]
        hi = self._ref_log_energy[idx + 1]
        t = np.clip((log_e - lo) / (hi - lo), 0.0, 1.0)
        return (1.0 - t)[:, np.newaxis, np.newaxis] * self.B[idx] + t[
            :, np.newaxis, np.newaxis
        ] * self.B[idx + 1]

    def _group_row_slice(self, ig: int, n_e: int) -> slice:
        """Return the row slice for group index ig in the stacked layout."""
        rpg = 2 * n_e if self._is_nucleon_model else n_e
        return slice(ig * rpg, (ig + 1) * rpg)

    # ------------------------------------------------------------------
    # Public interface — mirrors the original GSF model API
    # ------------------------------------------------------------------

    def flux(
        self,
        energy: ArrayLike,
        target: str | int | list[int],
        **kwargs,
    ) -> np.ndarray:
        """Calculate flux (delegated to the underlying model).

        Identical to ``model.flux(energy, target, ...)``.
        """
        kw = {**self._kwargs, **kwargs}
        return self.model.flux(energy, target, **kw)

    def jacobian(
        self,
        energy: ArrayLike,
        target: str | int | list[int],
        **kwargs,
    ) -> np.ndarray:
        """Calculate flux Jacobian (delegated to the underlying model).

        Identical to ``model.jacobian(energy, target, ...)``.
        """
        kw = {**self._kwargs, **kwargs}
        return self.model.jacobian(energy, target, **kw)

    def _group_pair_absolute_covariance(
        self,
        ig1: int,
        ig2: int,
        energy: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Compute full absolute covariance between two group indices.

        Returns the full abs_cov matrix including all nucleon blocks if
        applicable.  The low-rank part carries all cross-energy structure;
        the interpolated residual block is added on the same-energy
        diagonal for every sub-row pair of the two groups.
        """
        n_e = len(energy)
        nspg = self._n_sub_per_group

        M_full, cf = self._reduced_jacobian_full(energy, **kwargs)
        Bx = self._interpolate_B(energy)

        s1 = self._group_row_slice(ig1, n_e)
        s2 = self._group_row_slice(ig2, n_e)

        M1 = M_full[s1]
        M2 = M_full[s2]
        cf1 = cf[s1]
        cf2 = cf[s2]

        # Relative covariance for this group pair: low rank ...
        rel_cov = M1 @ M2.T
        # ... plus the same-energy residual block entries
        diag_idx = np.arange(n_e)
        for a in range(nspg):
            for b in range(nspg):
                rel_cov[a * n_e + diag_idx, b * n_e + diag_idx] += Bx[
                    :, ig1 * nspg + a, ig2 * nspg + b
                ]

        # Convert to absolute: Cov_abs[i,j] = rel_cov[i,j] * f1[i] * f2[j]
        return rel_cov * cf1[:, np.newaxis] * cf2[np.newaxis, :]

    def covariance(
        self,
        target1: str | int | list[int],
        target2: str | int | list[int],
        energy: ArrayLike,
        **kwargs,
    ) -> np.ndarray:
        """Approximate absolute flux covariance via low rank + energy blocks.

        Same signature and return units as the original model's ``covariance``.

        Parameters
        ----------
        target1, target2 : str, int, or list[int]
            Target specification (group name, element Z, or list of Z values).
        energy : array-like
            Energy values.
        **kwargs
            Override kwargs for model methods.

        Returns
        -------
        cov : ndarray, shape (N, N)
            Absolute flux covariance matrix.
        """
        energy = np.atleast_1d(np.asarray(energy, dtype=float))
        n_e = len(energy)
        ig1 = _group_index(target1, self.model)
        ig2 = _group_index(target2, self.model)

        abs_cov = self._group_pair_absolute_covariance(ig1, ig2, energy, **kwargs)

        if self._is_nucleon_model:
            # For nucleon models, the stacked layout is [p-rows, n-rows].
            # Sum the 4 blocks: Cov(p+n, p+n) = Cov(p,p) + Cov(p,n) + Cov(n,p) + Cov(n,n)
            cov_pp = abs_cov[:n_e, :n_e]
            cov_pn = abs_cov[:n_e, n_e:]
            cov_np = abs_cov[n_e:, :n_e]
            cov_nn = abs_cov[n_e:, n_e:]
            return cov_pp + cov_pn + cov_np + cov_nn
        return abs_cov

    def error(
        self,
        energy: ArrayLike,
        target: str | int | list[int],
        **kwargs,
    ) -> np.ndarray:
        """Approximate absolute 1-sigma flux uncertainty via the reduction.

        Same signature and return units as the original model's ``error``.

        Parameters
        ----------
        energy : array-like
            Energy values.
        target : str, int, or list[int]
            Target specification.
        **kwargs
            Override kwargs for model methods.

        Returns
        -------
        sigma : ndarray, shape (N,)
            Absolute 1-sigma flux uncertainties.
        """
        cov = self.covariance(target, target, energy, **kwargs)
        return np.sqrt(np.diag(cov))

    def total_flux(
        self,
        energy: ArrayLike,
        **kwargs,
    ) -> np.ndarray:
        """Total flux summed over all groups (delegated to model)."""
        kw = {**self._kwargs, **kwargs}
        return self.model.total_flux(energy, **kw)

    def total_error(
        self,
        energy: ArrayLike,
        **kwargs,
    ) -> np.ndarray:
        """Total flux uncertainty summed over all group pairs.

        Same as the original model's ``total_error`` but using the reduced
        representation.  Exact on the reference grid (the residual blocks
        carry all same-energy cross-group covariance).
        """
        energy = np.atleast_1d(np.asarray(energy, dtype=float))
        n_e = len(energy)
        total_cov = np.zeros((n_e, n_e))
        for g1 in _GROUPS:
            for g2 in _GROUPS:
                total_cov += self.covariance(g1, g2, energy, **kwargs)
        return np.sqrt(np.diag(total_cov))

    # ------------------------------------------------------------------
    # Nucleon-model-specific methods
    # ------------------------------------------------------------------

    def p_and_n_flux(
        self,
        energy: ArrayLike,
        target: str | int | list[int],
        **kwargs,
    ) -> np.ndarray:
        """Separate proton and neutron flux (delegated to model).

        Only available for nucleon-model backends.
        """
        kw = {**self._kwargs, **kwargs}
        return self.model.p_and_n_flux(energy, target, **kw)

    def p_and_n_covariance(
        self,
        target1: str | int | list[int],
        target2: str | int | list[int],
        energy: ArrayLike,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Separate proton and neutron covariance matrices via the reduction.

        Returns
        -------
        cov_pp, cov_nn : ndarray, shape (N, N) each
            Proton-proton and neutron-neutron absolute covariance matrices.
        """
        energy = np.atleast_1d(np.asarray(energy, dtype=float))
        n_e = len(energy)
        ig1 = _group_index(target1, self.model)
        ig2 = _group_index(target2, self.model)

        abs_cov = self._group_pair_absolute_covariance(ig1, ig2, energy, **kwargs)

        # Extract p-p and n-n blocks
        cov_pp = abs_cov[:n_e, :n_e]
        cov_nn = abs_cov[n_e:, n_e:]
        return cov_pp, cov_nn

    def p_and_n_error(
        self,
        energy: ArrayLike,
        target: str | int | list[int],
        **kwargs,
    ) -> np.ndarray:
        """Separate proton and neutron uncertainties via the reduction.

        Returns
        -------
        errors : ndarray, shape (2, N)
            [0, :] proton flux uncertainties, [1, :] neutron flux uncertainties.
        """
        cov_pp, cov_nn = self.p_and_n_covariance(target, target, energy, **kwargs)
        return np.array([np.sqrt(np.diag(cov_pp)), np.sqrt(np.diag(cov_nn))])

    # ------------------------------------------------------------------
    # PCA-specific extras
    # ------------------------------------------------------------------

    def reduced_jacobian(
        self,
        energy: ArrayLike,
        target: str | int | list[int] | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Reduced relative Jacobian M(E) = J_rel(E) @ L_param.

        Exact at any energy — no interpolation needed.

        Parameters
        ----------
        energy : array-like
            Energy values.
        target : str, int, list[int], or None
            If given, return only rows for this group.
            If None, return the full stacked matrix over all groups.
        **kwargs
            Override kwargs for model methods.

        Returns
        -------
        M : ndarray
            Reduced relative Jacobian. Shape depends on target:
            - target=None: (n_groups * rows_per_group, n_components)
            - target given: (rows_per_group, n_components) for that group
        """
        energy = np.atleast_1d(np.asarray(energy, dtype=float))
        M, _ = self._reduced_jacobian_full(energy, **kwargs)
        if target is not None:
            ig = _group_index(target, self.model)
            s = self._group_row_slice(ig, len(energy))
            return M[s]
        return M

    def sample(
        self,
        n_samples: int,
        energy: ArrayLike,
        rng: np.random.Generator | None = None,
        residual_noise: bool = True,
        diagonal_noise: bool | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Draw random flux realizations from the reduced model.

        By default, samples are drawn from the same low-rank + per-energy
        block covariance used by :meth:`covariance` and :meth:`error`:

            f_i = f_central,i * (1 + M_i @ phi + eps_i)

        Here ``phi ~ N(0, I_k)`` is shared by all energies and
        ``eps_i ~ N(0, B_i)`` is independent between requested energies.
        The residual term makes the sample covariance match the exact
        cross-group covariance of the approximation at every energy.

        Set ``residual_noise=False`` to draw only the correlated low-rank
        modes.  Those draws are smooth, but they do not sample the covariance
        returned by :meth:`covariance` and generally underestimate the
        pointwise variance.

        Parameters
        ----------
        n_samples : int
            Number of realizations to draw.
        energy : array-like
            Energy values.
        rng : numpy.random.Generator, optional
            Random number generator. Default: ``np.random.default_rng()``.
        residual_noise : bool, optional
            If True, include per-energy noise from the residual blocks B so
            that samples match the same-energy covariance returned by the
            uncertainty methods. Default True. Set to False for smooth
            low-rank-only draws.
        diagonal_noise : bool, optional
            Deprecated alias for ``residual_noise``.
        **kwargs
            Override kwargs for model methods.

        Returns
        -------
        samples : ndarray, shape (n_samples, rows)
            Absolute flux realizations (stacked over all groups).
        """
        if diagonal_noise is not None:
            residual_noise = diagonal_noise
        if rng is None:
            rng = np.random.default_rng()

        energy = np.atleast_1d(np.asarray(energy, dtype=float))
        n_e = len(energy)

        M, central_flux = self._reduced_jacobian_full(energy, **kwargs)

        n_rows = M.shape[0]
        phi = rng.standard_normal((n_samples, self.n_components))
        rel_variation = phi @ M.T

        if residual_noise:
            Bx = self._interpolate_B(energy)  # (n_e, nsub, nsub)
            # matrix square root per energy (PSD by construction)
            w, u = np.linalg.eigh(Bx)
            A = u * np.sqrt(np.maximum(w, 0.0))[:, np.newaxis, :]
            z = rng.standard_normal((n_samples, n_e, self._n_sub))
            # eps[s, i, a] = sum_b A[i, a, b] * z[s, i, b]
            eps_sub = np.einsum("iab,sib->sia", A, z)
            eps = np.zeros((n_samples, n_rows))
            offsets = self._sub_row_offsets(n_e)
            for s_idx, off in enumerate(offsets):
                eps[:, off : off + n_e] = eps_sub[:, :, s_idx]
            rel_variation += eps

        return central_flux[np.newaxis, :] * (1.0 + rel_variation)
