import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Hybrid PCA: a compact representation of the GSF uncertainty

    GSF propagates flux uncertainties through ~90 correlated spline parameters
    (the retained interior coefficients of the four mass groups H, He, O\*, Fe\*).
    `HybridPCA` compresses this into a small number of uncorrelated standard-normal
    components $\phi_k \sim \mathcal{N}(0, 1)$ (default 8) plus a per-energy
    cross-group **residual block** $B(E)$:

    $$
    \mathrm{Cov}\!\left[\frac{\delta f_i}{f_i}, \frac{\delta f_j}{f_j}\right]
        = M(E_i)\, M(E_j)^{\mathsf T} + \delta_{ij}\, B(E_i)
    $$

    - The **low-rank part** $M(E) = J_\mathrm{rel}(E)\, L_\mathrm{param}$ carries all
      correlations *between energies* — smooth spectral deformations, exact at any
      energy (no interpolation).
    - The **blocks** $B(E)$ keep whatever the components miss at each single energy,
      exactly on the reference grid. Any observable formed from fluxes at one energy
      (a group band, the all-particle flux, the total nucleon flux, $\langle\ln A\rangle$-type
      sums) has the exact variance there — for *any* number of components.

    **What changed vs. the original GSF 2025 construction** (shown in the old version
    of this notebook, which built the decomposition by hand):

    1. The residual correction is a full cross-group block $B(E)$ per energy
       ($8\times 8$ for nucleon models with the p/n split), not a scalar diagonal $D$.
       This makes cross-group sums (all-particle, total nucleon) exact on the grid.
    2. The factor is extracted in the **correlation gauge** by default: the
       *correlation* matrix of the relative flux covariance is diagonalized, so every
       grid row counts equally and the components resolve correlation structure across
       the whole energy range — instead of being spent on the few rows with the largest
       relative variance (the data-free heavy-group tail). The classic PCA is retained
       as `gauge="covariance"`; both are compared below.

    Everything here uses the shipped class — no hand-built stacked systems.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from globalsplinefit import GSFEnergy, GSFEnergyPerNucleon
    from globalsplinefit.pca import HybridPCA

    plt.rcParams["figure.dpi"] = 100
    return GSFEnergy, GSFEnergyPerNucleon, HybridPCA, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Build the reduced model

    One call. The decomposition is built on a reference grid (default: 300 log-spaced
    points from 1 GeV to $10^{11}$ GeV) from the model's stacked relative Jacobian and
    parameter covariance. `HybridPCA` then mirrors the model API — `flux` is delegated
    unchanged; `error`, `covariance`, and `sample` use the reduction.

    The 2026 default model is used (mixture covering, GMD modulation potential,
    Solar Cycle 24 average).
    """
    )
    return


@app.cell
def _(GSFEnergyPerNucleon, HybridPCA, np):
    xkin = np.logspace(0, 11, 300)  # reference grid [GeV/nucleon]
    gsf = GSFEnergyPerNucleon()
    pca = HybridPCA(gsf, n_components=8, energy_grid=xkin)

    print(f"components:         {pca.n_components}  (gauge: {pca.gauge})")
    print(f"parameter factor:   L_param {pca.L_param.shape}")
    print(f"residual blocks:    B {pca.B.shape}  (p/n x 4 groups per energy)")
    print(f"variance explained: {pca.variance_explained:.1%} (correlation gauge)")
    return gsf, pca, xkin


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Validation: same-energy observables are exact on the grid

    Group fluxes with the reduced $\pm1\sigma$ band (filled) against the exact
    full-parameter band (dashed), and the error ratio reduced/exact per group and for
    the total nucleon flux. On reference-grid energies the ratios are 1 to numerical
    precision — the blocks carry everything the 8 components miss at a single energy.

    (The comparison stops at $10^9$ GeV/nucleon: beyond that the central flux of the
    heavy groups underflows to zero — the data-free tail trimmed by the construction —
    and relative quantities lose meaning.)
    """
    )
    return


@app.cell
def _(gsf, pca, plt, xkin):
    E_val = xkin[(xkin >= 2.0) & (xkin <= 1e9)][::3]  # on-grid points

    _fig, (_ax1, _ax2) = plt.subplots(
        1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.3, 1]}
    )

    for _g, _c in zip(["p", "He", "O*", "Fe*"], ["C0", "C1", "C2", "C3"], strict=True):
        _f = pca.flux(E_val, _g)
        _s_pca = pca.error(E_val, _g)
        _s_ex = gsf.error(E_val, _g)
        _scale = E_val**2.7
        _ax1.loglog(E_val, _scale * _f, color=_c, lw=1.5, label=_g)
        _ax1.fill_between(
            E_val,
            _scale * (_f - _s_pca),
            _scale * (_f + _s_pca),
            color=_c,
            alpha=0.3,
        )
        _ax1.loglog(E_val, _scale * (_f + _s_ex), color="k", lw=0.7, ls="--")
        _ax1.loglog(E_val, _scale * (_f - _s_ex), color="k", lw=0.7, ls="--")
        _ax2.semilogx(E_val, _s_pca / _s_ex, color=_c, lw=1.5, label=_g)

    _ax2.semilogx(
        E_val,
        pca.total_error(E_val) / gsf.total_error(E_val),
        color="k",
        lw=2,
        label="total nucleon",
    )

    _ax1.set_xlabel("Energy per nucleon [GeV]")
    _ax1.set_ylabel(r"Nucleon flux $\times\, E^{2.7}$")
    _ax1.set_title(r"Reduced band (filled) vs. exact $\pm1\sigma$ (dashed)")
    _ax1.legend()
    _ax1.grid(True, alpha=0.3)

    _ax2.axhline(1.0, color="grey", lw=0.8)
    _ax2.set_ylim(0.98, 1.02)
    _ax2.set_xlabel("Energy per nucleon [GeV]")
    _ax2.set_ylabel(r"$\sigma_\mathrm{reduced} / \sigma_\mathrm{exact}$")
    _ax2.set_title("Error ratio on grid energies")
    _ax2.legend(fontsize=8)
    _ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return (E_val,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The components

    Each component is an independent deformation mode: setting $\phi_k = \pm 1$
    perturbs the flux by $\delta f / f = \pm M_k(E)$. Shown for the H group —
    protons (blue) and neutrons (orange; the two nearly coincide since both trace
    the same group parameters, with small differences from the deuterium
    admixture). The modes concentrate where the flux uncertainty is largest: the
    solar-modulated region below ~10 GeV and the transition region above
    $10^6$ GeV, with the leading mode carrying the oscillatory structure of the
    poorly-constrained high-energy proton spectrum.
    """
    )
    return


@app.cell
def _(np, pca, plt, xkin):
    # restrict to the data-constrained range; above ~1e9 GeV/nucleon the modes
    # are dominated by the huge relative variance of the data-free tail
    _E_c = xkin[(xkin >= 2.0) & (xkin <= 1e9)]
    _M = pca.reduced_jacobian(_E_c, "p")  # H group: rows = [p at E; n at E]
    _n_e = len(_E_c)

    _fig, _axes = plt.subplots(2, 4, figsize=(14, 5), sharex=True, sharey=True)
    _fig.subplots_adjust(hspace=0.15, wspace=0.08)

    for _k, _ax in enumerate(_axes.flat):
        _mp = _M[:_n_e, _k]
        _mn = _M[_n_e:, _k]
        _ax.fill_between(_E_c, 1 + _mp, 1 - _mp, color="C0", alpha=0.5)
        _ax.fill_between(_E_c, 1 + _mn, 1 - _mn, color="C1", alpha=0.35)
        _ax.set_xscale("log")
        _ax.axhline(1.0, color="grey", lw=0.6)
        _ax.set_title(rf"$\phi_{{{_k + 1}}}$", fontsize=10)
        _ax.grid(True, alpha=0.3)

    _amp = 1.12 * np.abs(_M).max()
    _axes[0, 0].set_ylim(1 - _amp, 1 + _amp)
    for _ax in _axes[1]:
        _ax.set_xlabel("E/nucleon [GeV]")
    for _ax in _axes[:, 0]:
        _ax.set_ylabel(r"$1 \pm M_k$")

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## What the residual blocks carry

    The low-rank part alone (dashed) underestimates the pointwise uncertainty —
    by construction it only resolves the leading correlated modes. Adding the
    block term (solid) recovers the exact band (black). The gap is largest where
    the covariance is locally structured rather than long-range correlated.

    The scalar diagonal $D$ and p–n cross term $D_{pn}$ of the original 2025
    construction are the diagonal entries of $B$ and remain available as
    read-only properties (`pca.D`, `pca.D_pn`).
    """
    )
    return


@app.cell
def _(E_val, gsf, pca, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    for _ax, _g in zip(_axes, ["p", "Fe*"], strict=True):
        _n_e = len(E_val)
        _f_p = gsf.p_and_n_flux(E_val, _g)[0]
        _lowrank = (pca.reduced_jacobian(E_val, _g)[:_n_e] ** 2).sum(axis=1) ** 0.5
        _full = pca.p_and_n_error(E_val, _g)[0] / _f_p
        _exact = gsf.p_and_n_error(E_val, _g)[0] / _f_p

        _ax.loglog(E_val, _exact, color="k", lw=2.5, alpha=0.4, label="exact")
        _ax.loglog(E_val, _lowrank, color="C0", ls="--", label="low rank only")
        _ax.loglog(E_val, _full, color="C0", label="low rank + block")
        _ax.set_xlabel("Energy per nucleon [GeV]")
        _ax.set_title(f"{_g} group, protons")
        _ax.grid(True, alpha=0.3)
        _ax.legend()

    _axes[0].set_ylabel(r"relative uncertainty $\sigma_f / f$")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Monte Carlo samples

    `sample` draws flux realizations
    $f_i = f_{\mathrm{central},i}\,(1 + M_i \phi + \varepsilon_i)$ with
    $\phi \sim \mathcal{N}(0, I_8)$ shared across energies and
    $\varepsilon_i \sim \mathcal{N}(0, B_i)$ independent between energies.
    By default (`residual_noise=True`) the draws reproduce the full covariance
    returned by `covariance`/`error` — the block term is visible as bin-to-bin
    jitter. With `residual_noise=False` only the smooth correlated modes are
    drawn: use this for smooth spectral deformations, but note those draws
    underestimate the pointwise variance (dashed curve on the right).
    """
    )
    return


@app.cell
def _(E_val, gsf, np, pca, plt):
    _rng = np.random.default_rng(0)
    _n_e = len(E_val)
    samples = pca.sample(2000, E_val, rng=_rng)
    samples_smooth = pca.sample(2000, E_val, rng=_rng, residual_noise=False)

    # H-group proton rows are the first n_e columns of the stacked layout
    _f_p = gsf.p_and_n_flux(E_val, "p")[0]
    _sig_p = pca.p_and_n_error(E_val, "p")[0]

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for _s in samples[:12, :_n_e]:
        _ax1.semilogx(E_val, _s / _f_p, color="grey", lw=0.6, alpha=0.5)
    for _s in samples_smooth[:12, :_n_e]:
        _ax1.semilogx(E_val, _s / _f_p, color="C0", lw=1.0, alpha=0.8)
    _ax1.semilogx([], [], color="grey", label="with residual noise")
    _ax1.semilogx([], [], color="C0", label="smooth (low rank only)")
    _ax1.set_xlabel("Energy per nucleon [GeV]")
    _ax1.set_ylabel("sample / central (H protons)")
    _ax1.legend()
    _ax1.grid(True, alpha=0.3)

    _ax2.semilogx(
        E_val,
        samples[:, :_n_e].std(axis=0) / _sig_p,
        color="grey",
        label="with residual noise",
    )
    _ax2.semilogx(
        E_val,
        samples_smooth[:, :_n_e].std(axis=0) / _sig_p,
        color="C0",
        ls="--",
        label="smooth only",
    )
    _ax2.axhline(1.0, color="k", lw=0.8)
    _ax2.set_ylim(0, 1.3)
    _ax2.set_xlabel("Energy per nucleon [GeV]")
    _ax2.set_ylabel(r"sample std / $\sigma_\mathrm{reduced}$")
    _ax2.legend()
    _ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return (samples_smooth,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Neutron-to-proton ratio

    The $n/p$ ratio drives atmospheric lepton charge ratios and composition
    observables. Multi-energy functionals like this live in the correlated
    low-rank part, so the smooth samples are the right tool: total proton and
    neutron fluxes are summed over the four groups per draw, and the band is the
    16–84% percentile range of the ratio.
    """
    )
    return


@app.cell
def _(E_val, gsf, np, plt, samples_smooth):
    _n_e = len(E_val)
    # stacked layout: per group [p rows; n rows], group-major
    _tot_p = sum(
        samples_smooth[:, 2 * _ig * _n_e : (2 * _ig + 1) * _n_e] for _ig in range(4)
    )
    _tot_n = sum(
        samples_smooth[:, (2 * _ig + 1) * _n_e : (2 * _ig + 2) * _n_e]
        for _ig in range(4)
    )
    _ratio = _tot_n / _tot_p

    _pn = gsf.p_and_n_total_flux(E_val)
    _central = _pn[1] / _pn[0]

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _lo, _hi = np.percentile(_ratio, [16, 84], axis=0)
    _ax.fill_between(E_val, _lo, _hi, color="C2", alpha=0.3, label="16–84% of samples")
    _ax.semilogx(E_val, _central, color="C2", lw=2, label="central")
    _ax.set_xlabel("Energy per nucleon [GeV]")
    _ax.set_ylabel("neutron / proton flux")
    _ax.grid(True, alpha=0.3)
    _ax.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Correlation gauge vs. classic PCA

    Why the new default? The fraction of the pointwise relative variance captured
    by the 8 components alone, per group. The classic covariance-gauge PCA
    (dashed) concentrates on the rows with the largest relative variance — the
    Fe\* group and the high-energy tail — and captures almost nothing of the
    proton uncertainty at low energies. The correlation gauge (solid) weights
    every grid row equally and spreads the fidelity across the whole range.
    Either way the blocks make single-energy observables exact; the gauge decides
    how well *cross-energy correlations* are resolved, which is what the
    components are for.
    """
    )
    return


@app.cell
def _(HybridPCA, gsf, xkin):
    pca_cov = HybridPCA(gsf, n_components=8, energy_grid=xkin, gauge="covariance")
    return (pca_cov,)


@app.cell
def _(E_val, gsf, pca, pca_cov, plt):
    _n_e = len(E_val)
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for _ax, _g in zip(_axes, ["p", "Fe*"], strict=True):
        _f_p = gsf.p_and_n_flux(E_val, _g)[0]
        _var_exact = (gsf.p_and_n_error(E_val, _g)[0] / _f_p) ** 2
        for _p, _ls, _lbl in [
            (pca, "-", 'gauge="correlation" (default)'),
            (pca_cov, "--", 'gauge="covariance" (classic PCA)'),
        ]:
            _cap = (_p.reduced_jacobian(E_val, _g)[:_n_e] ** 2).sum(axis=1) / _var_exact
            _ax.semilogx(E_val, _cap, ls=_ls, color="C3", label=_lbl)
        _ax.set_ylim(0, 1.05)
        _ax.set_xlabel("Energy per nucleon [GeV]")
        _ax.set_title(f"{_g} group, protons")
        _ax.grid(True, alpha=0.3)

    _axes[0].set_ylabel("variance fraction captured\nby 8 components")
    _axes[0].legend(loc="lower right", fontsize=9)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Cutoff-aware uncertainty propagation

    The decomposition is a property of the *spline parameters*, so a geomagnetic
    rigidity cutoff enters only through the Jacobian at evaluation time:
    $M(E, R_\mathrm{cut}) = J_\mathrm{rel}(E, R_\mathrm{cut})\, L_\mathrm{param}$.
    No rebuild needed — pass `rigidity_cutoff` as a keyword.

    A sharp Heaviside cutoff (`cutoff_width=0`) makes $M(E)$ discontinuous — hard
    steps as each species drops out (deuterium at half the proton energy per
    nucleon) and a dead zone below the cutoff. The default smooth penumbral
    transition (1 GV width here) stays well-behaved. Always prefer a finite
    `cutoff_width` in practice.
    """
    )
    return


@app.cell
def _(GSFEnergyPerNucleon, HybridPCA, np, pca, plt, xkin):
    R_CUT = 20.0  # GV, typical equatorial cutoff

    gsf_sharp = GSFEnergyPerNucleon(cutoff_width=0.0)
    pca_sharp = HybridPCA(gsf_sharp, n_components=8, energy_grid=xkin)

    E_cut = np.logspace(np.log10(2.0), np.log10(300.0), 400)
    _n_e = len(E_cut)
    # H-group proton rows, evaluated WITH the cutoff via kwargs
    _M_smooth = pca.reduced_jacobian(E_cut, "p", rigidity_cutoff=R_CUT)[:_n_e]
    _M_sharp = pca_sharp.reduced_jacobian(E_cut, "p", rigidity_cutoff=R_CUT)[:_n_e]

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    for _k, _ax in enumerate(_axes):
        _ax.semilogx(E_cut, _M_sharp[:, _k], color="C3", label="sharp (width 0)")
        _ax.semilogx(E_cut, _M_smooth[:, _k], color="C0", label="smooth (1 GV)")
        _ax.axvline(R_CUT, color="grey", lw=0.8, ls=":")
        _ax.set_title(rf"$M_{{{_k + 1}}}(E)$ with $R_\mathrm{{cut}}$ = {R_CUT:g} GV")
        _ax.set_xlabel("Energy per nucleon [GeV]")
        _ax.grid(True, alpha=0.3)
    _axes[0].legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Appendix: Hybrid PCA with `GSFEnergy`

    Nothing above is specific to nucleon models. For per-particle models there is
    no p/n split — the residual blocks are $4\times4$ per energy and carry the
    (anticorrelated) cross-group covariance, so the all-particle band is exact on
    the grid as well.
    """
    )
    return


@app.cell
def _(GSFEnergy, HybridPCA, np, plt):
    gsf_energy = GSFEnergy()
    pca_energy = HybridPCA(gsf_energy, n_components=8)

    _E = np.logspace(2, 9, 120)  # total energy per particle [GeV]
    _scale = _E**2.6

    _fig, (_ax1, _ax2) = plt.subplots(
        1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.3, 1]}
    )

    for _g, _c in zip(["p", "He", "O*", "Fe*"], ["C0", "C1", "C2", "C3"], strict=True):
        _f = pca_energy.flux(_E, _g)
        _s = pca_energy.error(_E, _g)
        _ax1.loglog(_E, _scale * _f, color=_c, lw=1.5, label=_g)
        _ax1.fill_between(
            _E, _scale * (_f - _s), _scale * (_f + _s), color=_c, alpha=0.3
        )
        _ax2.semilogx(_E, _s / gsf_energy.error(_E, _g), color=_c, lw=1.5, label=_g)

    _f_tot = pca_energy.total_flux(_E)
    _s_tot = pca_energy.total_error(_E)
    _ax1.loglog(_E, _scale * _f_tot, color="k", lw=2, label="all-particle")
    _ax1.fill_between(
        _E,
        _scale * (_f_tot - _s_tot),
        _scale * (_f_tot + _s_tot),
        color="k",
        alpha=0.2,
    )
    _ax2.semilogx(
        _E,
        _s_tot / gsf_energy.total_error(_E),
        color="k",
        lw=2,
        label="all-particle",
    )

    _ax1.set_xlabel("Energy [GeV]")
    _ax1.set_ylabel(r"Particle flux $\times\, E^{2.6}$")
    _ax1.legend()
    _ax1.grid(True, alpha=0.3)
    _ax2.axhline(1.0, color="grey", lw=0.8)
    _ax2.set_ylim(0.9, 1.1)
    _ax2.set_xlabel("Energy [GeV]")
    _ax2.set_ylabel(r"$\sigma_\mathrm{reduced} / \sigma_\mathrm{exact}$")
    _ax2.legend(fontsize=8)
    _ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
