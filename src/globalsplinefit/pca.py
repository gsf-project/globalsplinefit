"""Hybrid PCA dimensionality reduction for GSF models.

Reduces the full parameter covariance (88 correlated spline parameters)
to a small number of uncorrelated latent parameters phi ~ N(0, I),
while preserving exact marginal variances at every energy via a
diagonal correction term.

Flux model: f(E) = f_central(E) * (1 + M(E) @ phi)
where M(E) = J_rel(E) @ L_param is the reduced relative Jacobian,
exact at any energy (no interpolation).

References
----------
Construction follows the hybrid approach from the GSF 2025 analysis:
eigendecompose in flux space, then project the low-rank factor back
to parameter space via normal equations.
"""

import numpy as np
from scipy.interpolate import interp1d

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
    par_vecs = [model.pars[model._leader_by_charge[model.GROUP_NAMES[g]]][1:-7]
                for g in _GROUPS]
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


class HybridPCA:
    """Hybrid PCA dimensionality reduction for any GSF model.

    Provides the same interface as the underlying model (``flux``, ``error``,
    ``covariance``, ``jacobian``) but uses the low-rank + diagonal approximation
    for uncertainty propagation.  Flux values are delegated unchanged to the
    underlying model; only the uncertainty methods use the PCA reduction.

    Parameters
    ----------
    model : GSFBase
        Any GSF model instance (GSFEnergy, GSFRigidity, GSFEnergyPerNucleon, etc.).
    n_components : int, optional
        Number of PCA components. Default 12.
    energy_grid : array-like, optional
        Reference energy grid for the decomposition.
        Default: 300 log-spaced points from 1 to 10^11 GeV.
    **kwargs
        Passed to model methods (e.g. ``time_interval``, ``rigidity_cutoff``).

    Attributes
    ----------
    L_param : ndarray, shape (n_params, n_components)
        Parameter-space low-rank factor.
    D : ndarray, shape (M,)
        Relative diagonal variance correction on the reference grid.
    n_components : int
        Number of retained components.
    variance_explained : float
        Fraction of total variance captured by the low-rank part.

    Examples
    --------
    >>> from globalsplinefit import GSFEnergyPerNucleon
    >>> from globalsplinefit.pca import HybridPCA
    >>> gsf = GSFEnergyPerNucleon(version="2025")
    >>> pca = HybridPCA(gsf, n_components=12)
    >>> E = np.logspace(1, 6, 50)
    >>> flux = pca.flux(E, "p")             # same as gsf.flux(E, "p")
    >>> sigma = pca.error(E, "p")           # absolute 1-sigma (like gsf.error)
    >>> cov = pca.covariance("p", "p", E)   # absolute covariance matrix
    """

    def __init__(
        self,
        model: GSFBase,
        n_components: int = 12,
        energy_grid: np.ndarray | list | float | None = None,
        **kwargs,
    ):
        if energy_grid is None:
            energy_grid = np.logspace(np.log10(1.0), 11, 300)
        energy_grid = np.atleast_1d(np.asarray(energy_grid, dtype=float))

        self.model = model
        self.n_components = n_components
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

        # Eigendecomposition (ascending order from eigh, reverse to descending)
        eigvals, eigvecs = np.linalg.eigh(flux_rel_cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        self._eigvals = eigvals

        # Low-rank factor in flux space
        k = n_components
        L = eigvecs[:, :k] * np.sqrt(np.maximum(eigvals[:k], 0.0))

        # Diagonal correction: exact variance minus low-rank contribution
        self.D = np.maximum(np.diag(flux_rel_cov) - np.sum(L**2, axis=1), 0.0)

        # Project to parameter space via least-squares (robust to rank-deficiency
        # when the energy grid has fewer points than parameters)
        self.L_param, _, _, _ = np.linalg.lstsq(rel_jac, L, rcond=None)

        # Store for interpolation of D
        self._ref_log_energy = np.log(energy_grid)
        self._n_ref = len(energy_grid)
        self._n_groups = len(_GROUPS)

        # Precompute row layout: rows_per_group for the reference grid
        self._rows_per_group_ref = (
            2 * self._n_ref if self._is_nucleon_model else self._n_ref
        )

        # Cross-term diagonal correction for nucleon models.
        # For each group, D_pn captures the residual diagonal of the p-n
        # cross-block: D_pn[i] = Sigma_pn[i,i] - sum_k L_p[i,k]*L_n[i,k].
        # Without this, covariance cross-terms (and thus total errors) are
        # severely underestimated for nuclei where Z ≈ A-Z (e.g. He, O, Fe).
        if self._is_nucleon_model:
            n_ref = self._n_ref
            rpg = 2 * n_ref
            D_pn = np.zeros(self._n_groups * n_ref)
            for ig in range(self._n_groups):
                p_idx = np.arange(n_ref) + ig * rpg
                n_idx = p_idx + n_ref
                exact_pn_diag = flux_rel_cov[p_idx, n_idx]
                lr_pn_diag = np.sum(L[p_idx] * L[n_idx], axis=1)
                d_pn = exact_pn_diag - lr_pn_diag
                # Clamp for PSD safety: D_pn <= sqrt(D_p * D_n)
                D_p = self.D[p_idx]
                D_n = self.D[n_idx]
                d_pn = np.minimum(d_pn, np.sqrt(D_p * D_n))
                D_pn[ig * n_ref : (ig + 1) * n_ref] = d_pn
            self.D_pn = D_pn
        else:
            self.D_pn = None

    @property
    def variance_explained(self) -> float:
        """Fraction of total variance captured by the low-rank components."""
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

    def _interpolate_D(self, energy: np.ndarray) -> np.ndarray:
        """Interpolate diagonal correction D from reference grid to new energies."""
        log_e = np.log(energy)
        n_e = len(energy)

        rows_per_group = self._rows_per_group_ref
        n_out_per_group = 2 * n_e if self._is_nucleon_model else n_e

        D_interp = np.empty(self._n_groups * n_out_per_group)

        for ig in range(self._n_groups):
            d_group = self.D[ig * rows_per_group : (ig + 1) * rows_per_group]

            if self._is_nucleon_model:
                d_p = d_group[: self._n_ref]
                d_n = d_group[self._n_ref :]
                for k, d_half in enumerate([d_p, d_n]):
                    log_d = np.log(np.maximum(d_half, 1e-300))
                    interp_fn = interp1d(
                        self._ref_log_energy,
                        log_d,
                        kind="linear",
                        fill_value="extrapolate",  # type: ignore[arg-type]
                    )
                    offset = ig * n_out_per_group + k * n_e
                    D_interp[offset : offset + n_e] = np.maximum(
                        np.exp(interp_fn(log_e)), 0.0
                    )
            else:
                log_d = np.log(np.maximum(d_group, 1e-300))
                interp_fn = interp1d(
                    self._ref_log_energy,
                    log_d,
                    kind="linear",
                    fill_value="extrapolate",  # type: ignore[arg-type]
                )
                offset = ig * n_out_per_group
                D_interp[offset : offset + n_e] = np.maximum(
                    np.exp(interp_fn(log_e)), 0.0
                )

        return D_interp

    def _interpolate_D_pn(self, energy: np.ndarray) -> np.ndarray:
        """Interpolate cross-term diagonal correction D_pn to new energies.

        Only meaningful for nucleon models. Layout: ``n_groups * n_e`` values,
        indexed as ``[ig * n_e : (ig + 1) * n_e]``.
        """
        log_e = np.log(energy)
        n_e = len(energy)
        n_ref = self._n_ref

        D_pn_interp = np.empty(self._n_groups * n_e)
        for ig in range(self._n_groups):
            d_pn_ref = self.D_pn[ig * n_ref : (ig + 1) * n_ref]
            log_d = np.log(np.maximum(d_pn_ref, 1e-300))
            interp_fn = interp1d(
                self._ref_log_energy,
                log_d,
                kind="linear",
                fill_value="extrapolate",  # type: ignore[arg-type]
            )
            offset = ig * n_e
            D_pn_interp[offset : offset + n_e] = np.maximum(
                np.exp(interp_fn(log_e)), 0.0
            )
        return D_pn_interp

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

        Returns the full abs_cov matrix including all nucleon blocks if applicable.
        """
        n_e = len(energy)

        M_full, cf = self._reduced_jacobian_full(energy, **kwargs)
        D_interp = self._interpolate_D(energy)

        s1 = self._group_row_slice(ig1, n_e)
        s2 = self._group_row_slice(ig2, n_e)

        M1 = M_full[s1]
        M2 = M_full[s2]
        cf1 = cf[s1]
        cf2 = cf[s2]

        # Relative covariance for this group pair
        rel_cov = M1 @ M2.T
        if ig1 == ig2:
            rel_cov += np.diag(D_interp[s1])
            # Add cross-term diagonal correction for nucleon models
            if self._is_nucleon_model:
                D_pn_interp = self._interpolate_D_pn(energy)
                d_pn = D_pn_interp[ig1 * n_e : (ig1 + 1) * n_e]
                rel_cov[:n_e, n_e:] += np.diag(d_pn)
                rel_cov[n_e:, :n_e] += np.diag(d_pn)

        # Convert to absolute: Cov_abs[i,j] = rel_cov[i,j] * f1[i] * f2[j]
        return rel_cov * cf1[:, np.newaxis] * cf2[np.newaxis, :]

    def covariance(
        self,
        target1: str | int | list[int],
        target2: str | int | list[int],
        energy: ArrayLike,
        **kwargs,
    ) -> np.ndarray:
        """Approximate absolute flux covariance via PCA low-rank + diagonal.

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
        """Approximate absolute 1-sigma flux uncertainty via PCA.

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

        Same as the original model's ``total_error`` but using PCA approximation.
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
        """Separate proton and neutron covariance matrices via PCA.

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
        """Separate proton and neutron uncertainties via PCA.

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
        diagonal_noise: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Draw random flux realizations from the reduced model.

        By default, samples are smooth spectral perturbations driven by
        the k correlated PCA components:

            f(E) = f_central(E) * (1 + M(E) @ phi),  phi ~ N(0, I_k)

        With ``diagonal_noise=True``, an additional per-bin noise term is
        added to match the exact marginal variances.  This noise treats
        the residual covariance as uncorrelated across energy bins, which
        introduces bin-to-bin jitter that is not physical.  Use this only
        when correct per-bin variances matter more than spectral smoothness.

        Parameters
        ----------
        n_samples : int
            Number of realizations to draw.
        energy : array-like
            Energy values.
        rng : numpy.random.Generator, optional
            Random number generator. Default: ``np.random.default_rng()``.
        diagonal_noise : bool, optional
            If True, add per-bin diagonal noise from D (and D_pn for
            nucleon models) to match exact marginal variances.
            Default False (smooth samples only).
        **kwargs
            Override kwargs for model methods.

        Returns
        -------
        samples : ndarray, shape (n_samples, rows)
            Absolute flux realizations (stacked over all groups).
        """
        if rng is None:
            rng = np.random.default_rng()

        energy = np.atleast_1d(np.asarray(energy, dtype=float))

        M, central_flux = self._reduced_jacobian_full(energy, **kwargs)

        n_rows = M.shape[0]
        phi = rng.standard_normal((n_samples, self.n_components))
        rel_variation = phi @ M.T

        if diagonal_noise:
            D_interp = self._interpolate_D(energy)
            if self._is_nucleon_model:
                n_e = len(energy)
                D_pn_interp = self._interpolate_D_pn(energy)
                eps = np.zeros((n_samples, n_rows))
                rpg = 2 * n_e
                for ig in range(self._n_groups):
                    p_off = ig * rpg
                    n_off = p_off + n_e
                    D_p = D_interp[p_off : p_off + n_e]
                    D_n = D_interp[n_off : n_off + n_e]
                    D_pn = D_pn_interp[ig * n_e : (ig + 1) * n_e]
                    L00 = np.sqrt(D_p)
                    L10 = np.where(L00 > 0, D_pn / np.maximum(L00, 1e-300), 0.0)
                    L11 = np.sqrt(np.maximum(D_n - L10**2, 0.0))
                    z = rng.standard_normal((n_samples, 2, n_e))
                    eps[:, p_off : p_off + n_e] = z[:, 0, :] * L00
                    eps[:, n_off : n_off + n_e] = z[:, 0, :] * L10 + z[:, 1, :] * L11
            else:
                eps = rng.standard_normal((n_samples, n_rows)) * np.sqrt(D_interp)
            rel_variation += eps

        return central_flux[np.newaxis, :] * (1.0 + rel_variation)
