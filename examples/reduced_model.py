import marimo

__generated_with = "0.23.15"
app = marimo.App(
    width="full",
    app_title="GSF — Reduced Model",
    css_file="gallery.css",
    html_head_file="gallery_head.html",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reduced GSF: flux nuisance parameters for downstream fits

    Analyses that consume the GSF flux — atmospheric lepton calculations,
    detector fits, air-shower interpretations — need the model uncertainty as
    something they can *vary*: a small set of parameters $\theta$ to build a
    Jacobian of their own observable, plus a covariance to use as a penalty
    term. `ReducedGSF` provides exactly that.

    Each species (total proton and total neutron flux by default) is deformed
    at $N$ **pivot energies**:

    $$
    f_s(E;\theta) = f_{\mathrm{central},s}(E)\,
        \Bigl(1 + \sum_k H_k(\log E)\, \theta_{s,k}\Bigr)
    $$

    where $H_k$ are cardinal interpolation functions in $\log E$ — a local
    cubic (Catmull–Rom) spline by default (smooth $C^1$ deformations, each
    component confined to its two neighboring intervals), or piecewise-linear
    hats with `basis="hat"`. Either way $H$ is the identity at the pivots, so
    the components are *relative flux deviations at the pivots* — interpretable
    knobs — and their covariance $C$ is evaluated **exactly** from the full GSF
    parameter covariance at the pivot energies. At the pivots the reduced
    model carries the full model variance and all cross-energy / p–n
    correlations exactly; between them the deformation is interpolated,
    which is median-unbiased and errs conservative.

    Intended fit loop — no setup, the default is the published grid:

    ```python
    red = ReducedGSF()                    # published 12-pivot grid, 24 named pars
    flux = red.flux(E, theta)             # -> your observable's Jacobian
    chi2 = chi2_data(theta) + red.penalty(theta)
    ```

    This is the same construction used for the GSF nuisance parameters in
    daemonflux (Yanez & Fedynitch 2023).
    """)
    return


@app.cell
async def _():
    import sys

    NOTEBOOK = "reduced_model"  # used by the download buttons at the end
    IN_WASM = "pyodide" in sys.modules
    SITE = ""

    if IN_WASM:
        # WASM (docs gallery): install the wheel built together with these
        # docs and published under /wheels/ on the same site, so the notebook
        # always runs the documented version (works on any host).
        from urllib.parse import urljoin

        import marimo as _mo
        import micropip

        # notebook_location() resolution depth differs between marimo export
        # layouts — try site-root /wheels/ from every plausible depth, and
        # keep the depth that worked as the site root for other assets.
        _base = str(_mo.notebook_location()) + "/"
        _err = None
        for _up in ("", "../", "../../", "../../../"):
            try:
                await micropip.install(
                    urljoin(
                        _base,
                        _up + "wheels/globalsplinefit-2.0.0-py3-none-any.whl",
                    )
                )
                SITE = urljoin(_base, _up)
                _err = None
                break
            except Exception as _e:  # noqa: BLE001 — 404 lands as generic error
                _err = _e
        if _err is not None:
            raise _err

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from globalsplinefit import MODEL_VERSIONS, GSFEnergyPerNucleon, ReducedGSF

    plt.rcParams["figure.dpi"] = 110

    def show(fig=None):
        """Return a figure as the cell output.

        marimo renders a cell's last expression; `plt.show()` produces no
        output at all in app view, so every plot cell ends with `show(fig)`.
        """
        fig = plt.gcf() if fig is None else fig
        return fig

    def autoscale(ax, *series, pad=1.6):
        """Fit the y-axis to the positive, finite values in `series`."""
        _v = np.concatenate([np.asarray(s, float).ravel() for s in series])
        _v = _v[np.isfinite(_v) & (_v > 0)]
        if _v.size:
            ax.set_ylim(_v.min() / pad, _v.max() * pad)

    return (
        GSFEnergyPerNucleon,
        MODEL_VERSIONS,
        NOTEBOOK,
        ReducedGSF,
        SITE,
        autoscale,
        mo,
        np,
        plt,
        show,
    )


@app.cell(hide_code=True)
def _(MODEL_VERSIONS, mo):
    version = mo.ui.dropdown(
        options=list(MODEL_VERSIONS), value="2026.1", label="model version"
    )
    plot_size = mo.ui.slider(
        start=0.6,
        stop=1.6,
        step=0.1,
        value=1.0,
        label="plot size",
        show_value=True,
    )
    mo.hstack([version, plot_size], justify="start", gap=2)
    return plot_size, version


@app.cell(hide_code=True)
def _(mo, plot_size):
    # Drives --gsf-plot-scale, which gallery.css multiplies into every figure's
    # width. Scaling the rendered raster (rather than the figsize) is what keeps
    # it proportional: fonts, line widths and the aspect ratio all ride along.
    mo.Html(f"<style>:root{{--gsf-plot-scale:{plot_size.value}}}</style>")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The published parameter set

    With all-default arguments, `ReducedGSF` uses the pivot grid of its
    model version (`model.params.reduced_pivots`): twelve energies per
    species — 1, 4, 90 GeV, 9, 25, 100 TeV, 3, 6, 30, 100, 300 PeV, 1 EeV —
    derived once with `optimize_pivots`. Every component has a citable name
    (`p_9TeV`, `n_30PeV`, ...), so results quoted against them are
    reproducible. The worst-case coverage of the exact uncertainty is a
    factor 1.29 anywhere in 1–$10^9$ GeV (validated below and enforced by
    a unit test).
    """)
    return


@app.cell
def _(
    GSFEnergyPerNucleon,
    ReducedGSF,
    version,
):
    gsf = GSFEnergyPerNucleon(
        version=version.value
    )  # 2026 default, Solar Cycle 24 average
    red = ReducedGSF(gsf)  # published grid for "2026.1" -> 24 parameters

    print(f"pivots [GeV]: {red.pivot_energies}")
    print(f"n_params:     {red.n_params}")
    print(f"labels:       {red.labels[:12]}")
    print(f"              {red.labels[12:]}")
    return gsf, red


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Validation against the full model

    The reduced $\pm1\sigma$ band (from `red.error`, i.e. sampling $\theta$
    from $C$) against the exact full-covariance band. At the pivot energies
    (markers) the two agree to numerical precision by construction; between
    pivots the interpolation over- or under-covers by a bounded amount. The
    dotted curve shows a naive 12-pivot log-spaced grid: same parameter
    count, but up to 1.8x mis-coverage — this is what the published grid's
    optimization buys.
    """)
    return


@app.cell
def _(
    ReducedGSF,
    gsf,
    np,
    plt,
    red,
    show,
):
    def exact_error(energy):
        """Exact total p/n error: sum group-pair covariance diagonals."""
        var_p = np.zeros(len(energy))
        var_n = np.zeros(len(energy))
        for g1 in gsf.active_groups:
            for g2 in gsf.active_groups:
                cpp, cnn = gsf.p_and_n_covariance(g1, g2, energy)
                var_p += np.diag(cpp)
                var_n += np.diag(cnn)
        return np.vstack([np.sqrt(var_p), np.sqrt(var_n)])

    red_log = ReducedGSF(gsf, n_pivots=12)  # naive grid, for comparison

    _E = np.logspace(0, 9, 250)
    _exact = exact_error(_E)
    _model = red.error(_E)
    _central = red.flux(_E)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    for _s, _c, _lbl in [(0, "C0", "p"), (1, "C1", "n")]:
        _ax1.loglog(_E, _exact[_s] / _central[_s], color="k", lw=2.2, alpha=0.35)
        _ax1.loglog(_E, _model[_s] / _central[_s], color=_c, lw=1.4, label=_lbl)
        _ax2.semilogx(_E, _model[_s] / _exact[_s], color=_c, lw=1.4, label=_lbl)
        _ax2.semilogx(
            _E,
            red_log.error(_E)[_s] / _exact[_s],
            color=_c,
            lw=1.0,
            ls=":",
            alpha=0.7,
            label=f"{_lbl}, log-spaced" if _s == 0 else None,
        )
    _piv_ratio = red.error(red.pivot_energies) / np.vstack(
        [
            np.interp(red.pivot_energies, _E, _exact[0]),
            np.interp(red.pivot_energies, _E, _exact[1]),
        ]
    )
    _ax2.plot(red.pivot_energies, _piv_ratio[0], "o", color="C0", ms=5)
    _ax2.plot(red.pivot_energies, _piv_ratio[1], "s", color="C1", ms=5)

    _ax1.set_xlabel("Energy per nucleon [GeV]")
    _ax1.set_ylabel(r"relative uncertainty $\sigma_f/f$")
    _ax1.set_title("reduced (colored) vs exact (grey)")
    _ax1.grid(True, alpha=0.3)
    _ax1.legend()

    _ax2.axhline(1.0, color="grey", lw=0.8)
    _ax2.set_xlabel("Energy per nucleon [GeV]")
    _ax2.set_ylabel(r"$\sigma_\mathrm{reduced}/\sigma_\mathrm{exact}$")
    _ax2.set_title("markers: published pivots (exact)")
    _ax2.grid(True, alpha=0.3)
    _ax2.legend(fontsize=8)

    plt.tight_layout()
    show()
    return (exact_error,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    (`optimize_pivots` derives a grid for custom energy windows, per-group
    reductions, or other model versions.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Component prior widths

    The prior width of each component — sub-percent where AMS-02/CREAM
    constrain the flux, ~8% at 1 GeV (solar modulation), growing to 20–30%
    at $10^9$ GeV. Note how the optimization crowded pivots into the
    poorly-constrained $10^6$–$10^9$ GeV region.
    """)
    return


@app.cell
def _(
    plt,
    red,
    show,
):
    _fig, _ax1 = plt.subplots(figsize=(7, 4.4))

    _n = len(red.pivot_energies)
    _ax1.loglog(red.pivot_energies, red.sigma[:_n], "o-", color="C0", label="p")
    _ax1.loglog(red.pivot_energies, red.sigma[_n:], "s-", color="C1", label="n")
    _ax1.set_xlabel("Pivot energy per nucleon [GeV]")
    _ax1.set_ylabel(r"prior width $\sigma_{\theta}$ (relative flux)")
    _ax1.grid(True, alpha=0.3)
    _ax1.legend()

    plt.tight_layout()
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reference: flux bands

    The classic view — $E^{2.7}$-scaled total proton and neutron flux with the
    reduced $\pm1\sigma$ band (filled) against the exact full-covariance band
    (dashed). This is the reduced model a downstream fit actually spans.
    """)
    return


@app.cell
def _(
    exact_error,
    np,
    plt,
    red,
    show,
):
    _E = np.logspace(0, 9, 250)
    _central = red.flux(_E)
    _sig = red.error(_E)
    _exact = exact_error(_E)
    _scale = _E**2.7

    _fig, _ax = plt.subplots(figsize=(9, 4.6))
    for _s, _c, _lbl in [(0, "C0", "protons"), (1, "C1", "neutrons")]:
        _ax.loglog(_E, _scale * _central[_s], color=_c, lw=1.8, label=_lbl)
        _ax.fill_between(
            _E,
            _scale * (_central[_s] - _sig[_s]),
            _scale * (_central[_s] + _sig[_s]),
            color=_c,
            alpha=0.3,
        )
        _ax.loglog(_E, _scale * (_central[_s] + _exact[_s]), "k--", lw=0.7)
        _ax.loglog(_E, _scale * (_central[_s] - _exact[_s]), "k--", lw=0.7)
    _ax.set_xlabel("Energy per nucleon [GeV]")
    _ax.set_ylabel(r"Nucleon flux $\times\, E^{2.7}$")
    _ax.set_title(r"reduced band (filled) vs exact $\pm1\sigma$ (dashed)")
    _ax.grid(True, alpha=0.3)
    _ax.legend()
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The components

    Each panel shows the flux deformation $1 \pm \sigma_k H_k(E)$ produced
    by moving one component by its prior width — protons (blue) and
    neutrons (orange). This is the operation a downstream fit performs to
    build its Jacobian (`red.flux_jacobian(E)` returns it analytically).
    The dotted curve in the first panel is the same component with
    `basis="hat"`.
    """)
    return


@app.cell
def _(
    ReducedGSF,
    gsf,
    np,
    plt,
    red,
    show,
):
    red_hat = ReducedGSF(gsf, pivot_energies=red.pivot_energies, basis="hat")

    _pv = red.pivot_energies
    _E = np.logspace(0, 9, 400)
    _H = red.basis(_E)
    _H_hat = red_hat.basis(_E)
    _n = len(_pv)
    _sig_p = red.sigma[:_n]
    _sig_n = red.sigma[_n:]

    _fig, _axes = plt.subplots(2, 6, figsize=(15, 4.6), sharey=True)
    _fig.subplots_adjust(hspace=0.35, wspace=0.08)

    for _k, _ax in enumerate(_axes.flat):
        _ax.fill_between(
            _E,
            1 + _sig_p[_k] * _H[:, _k],
            1 - _sig_p[_k] * _H[:, _k],
            color="C0",
            alpha=0.5,
        )
        _ax.fill_between(
            _E,
            1 + _sig_n[_k] * _H[:, _k],
            1 - _sig_n[_k] * _H[:, _k],
            color="C1",
            alpha=0.35,
        )
        if _k == 0:
            _ax.plot(_E, 1 + _sig_p[_k] * _H_hat[:, _k], "k:", lw=1)
            _ax.plot(_E, 1 - _sig_p[_k] * _H_hat[:, _k], "k:", lw=1)
        _ax.set_xscale("log")
        _ax.axhline(1.0, color="grey", lw=0.6)
        _ax.axvline(_pv[_k], color="grey", lw=0.5, ls=":")
        # zoom to +-2.5 decades around the pivot so narrow bumps stay visible
        _ax.set_xlim(max(_pv[_k] / 316.0, 1.0), min(_pv[_k] * 316.0, 1e9))
        _ax.set_title(red.labels[_k], fontsize=9)
        _ax.grid(True, alpha=0.3)

    for _ax in _axes[1]:
        _ax.set_xlabel("E/nucleon [GeV]")
    for _ax in _axes[:, 0]:
        _ax.set_ylabel(r"$1 \pm \sigma_k H_k$")

    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sampling

    Flux deformations from $\theta \sim \mathcal{N}(0, C)$ — smooth, correlated
    realizations suitable for Monte Carlo propagation. Left: protons, with the
    sample-std/error check confirming the draws reproduce the reduced covariance.
    Right: neutrons, against the $\pm1\sigma$ (dashed) and $\pm3\sigma$ (dotted)
    equivalent envelopes of the reduced model — the deformation is linear in
    $\theta$, so the draws are Gaussian at every energy and the dotted band is
    exactly three times the dashed one.

    The Gaussian penalty $\theta^{\mathsf T} C^{-1} \theta$ evaluated on these
    samples follows a $\chi^2_{24}$ distribution (`red.penalty`), which is what
    makes it usable directly as an additive penalty in a fit.
    """)
    return


@app.cell
def _(
    np,
    plt,
    red,
    show,
):
    _rng = np.random.default_rng(0)
    thetas = red.sample(2000, rng=_rng)
    _E = np.logspace(0, 9, 300)
    _central = red.flux(_E)
    _sig = red.error(_E)

    # vectorized sampled fluxes: ratio_s = 1 + theta_s @ H^T  per species
    _H = red.basis(_E)
    _n = len(red.pivot_energies)
    _ratio_p = 1 + thetas[:, :_n] @ _H.T  # (n_samples, n_E)
    _ratio_n = 1 + thetas[:, _n:] @ _H.T

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    for _r in _ratio_p[:25]:
        _ax1.semilogx(_E, _r, color="C0", lw=0.7, alpha=0.5)
    _ax1.semilogx(_E, 1 + _sig[0] / _central[0], "k--", lw=1.2)
    _ax1.semilogx(_E, 1 - _sig[0] / _central[0], "k--", lw=1.2, label=r"$\pm1\sigma$")
    _ax1.semilogx(
        _E,
        1 + _ratio_p.std(axis=0),
        color="C3",
        lw=1.0,
        label="sample std (2000 draws)",
    )
    _ax1.set_xlabel("Energy per nucleon [GeV]")
    _ax1.set_ylabel("proton flux ratio to central")
    _ax1.grid(True, alpha=0.3)
    _ax1.legend(fontsize=9)

    # neutrons, with 1-sigma (dashed) and 3-sigma (dotted) equivalent envelopes
    _rel_n = _sig[1] / _central[1]
    for _r in _ratio_n[:25]:
        _ax2.semilogx(_E, _r, color="C1", lw=0.7, alpha=0.5)
    _ax2.semilogx(_E, 1 + _rel_n, "k--", lw=1.2)
    _ax2.semilogx(_E, 1 - _rel_n, "k--", lw=1.2, label=r"$\pm1\sigma$")
    _ax2.semilogx(_E, 1 + 3 * _rel_n, "k:", lw=1.2)
    _ax2.semilogx(_E, 1 - 3 * _rel_n, "k:", lw=1.2, label=r"$\pm3\sigma$")
    _ax2.set_xlabel("Energy per nucleon [GeV]")
    _ax2.set_ylabel("neutron flux ratio to central")
    _ax2.grid(True, alpha=0.3)
    _ax2.legend(fontsize=9)

    plt.tight_layout()
    show()
    return (thetas,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The component correlation

    The 24 parameters are jointly Gaussian with covariance $C$ — the matrix
    that `to_dict()` exports and that the penalty term inverts. Left:
    neighbouring pivots correlate strongly, and the p–n off-diagonal blocks
    carry the common group parameters, so the penalty must use the full
    matrix. Right: the empirical correlation of the 2000 samples drawn
    above reproduces the same structure.
    """)
    return


@app.cell
def _(
    np,
    plt,
    red,
    show,
    thetas,
):
    _R = red.correlation
    _R_emp = np.corrcoef(thetas.T)
    _n = len(red.pivot_energies)

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for _ax, _mat, _title in [
        (_axes[0], _R, "analytic correlation"),
        (_axes[1], _R_emp, "sample correlation (2000 draws)"),
    ]:
        _im = _ax.imshow(_mat, cmap="RdBu_r", vmin=-1, vmax=1)
        _ax.axhline(_n - 0.5, color="k", lw=0.7)
        _ax.axvline(_n - 0.5, color="k", lw=0.7)
        _ax.set_xticks([_n / 2, 1.5 * _n], ["p", "n"])
        _ax.set_yticks([_n / 2, 1.5 * _n], ["p", "n"])
        _ax.set_title(_title)
    _fig.colorbar(_im, ax=_axes, fraction=0.03, label="correlation")
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Neutron-to-proton ratio

    The $n/p$ ratio drives atmospheric lepton charge ratios. Each sample
    deforms protons and neutrons coherently through the correlated p–n blocks
    of $C$, so the ratio uncertainty is much smaller than the individual
    bands: the common normalization cancels, and what remains is the genuine
    p–n decorrelation of the underlying group parameters.
    """)
    return


@app.cell
def _(
    np,
    plt,
    red,
    show,
):
    _rng = np.random.default_rng(1)
    _thetas = red.sample(2000, rng=_rng)
    _E = np.logspace(0, 9, 300)
    _central = red.flux(_E)
    _H = red.basis(_E)
    _n = len(red.pivot_energies)

    _f_p = _central[0] * (1 + _thetas[:, :_n] @ _H.T)
    _f_n = _central[1] * (1 + _thetas[:, _n:] @ _H.T)
    _ratio = _f_n / _f_p
    _lo, _hi = np.percentile(_ratio, [16, 84], axis=0)

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.fill_between(_E, _lo, _hi, color="C2", alpha=0.3, label="16–84% of samples")
    _ax.semilogx(_E, _central[1] / _central[0], color="C2", lw=2, label="central")
    _ax.set_xlabel("Energy per nucleon [GeV]")
    _ax.set_ylabel("neutron / proton flux")
    _ax.grid(True, alpha=0.3)
    _ax.legend()
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Export

    `to_dict()` emits a JSON-serializable record — pivots, species, basis,
    central flux at the pivots, covariance — so a fitter can consume it with
    only a JSON reader.
    """)
    return


@app.cell
def _(red):
    import json

    _d = red.to_dict()
    print(json.dumps(_d, indent=1)[:600] + "\n ...")
    return


@app.cell(hide_code=True)
def _(NOTEBOOK, SITE, mo):
    if SITE:  # running in the browser: link the sources published next to it
        _py = f"{SITE}gallery/{NOTEBOOK}/{NOTEBOOK}.py"
        _ipynb = f"{SITE}gallery/{NOTEBOOK}/{NOTEBOOK}.ipynb"
    else:  # running locally: hand over the file on disk
        _py = (mo.notebook_dir() / f"{NOTEBOOK}.py").read_bytes()
        _ipynb = None

    _buttons = [
        mo.download(
            _py,
            filename=f"{NOTEBOOK}.py",
            label="marimo notebook (.py)",
            mimetype="text/x-python",
        )
    ]
    if _ipynb is not None:
        _buttons.append(
            mo.download(
                _ipynb,
                filename=f"{NOTEBOOK}.ipynb",
                label="Jupyter notebook (.ipynb)",
                mimetype="application/x-ipynb+json",
            )
        )
    mo.vstack(
        [
            mo.md(r"""### Take this notebook with you"""),
            mo.hstack(_buttons, justify="start"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
