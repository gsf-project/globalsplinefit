import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GSF Model Deck

    The standard one-look summary of a GSF model: flux spectrum, composition
    fractions, ln A moments, relative uncertainty, nucleon flux, and the
    all-particle flux correlation across energy — plus a back-to-back
    comparison between two model versions. Pick the versions below; every
    panel updates.
    """)
    return


@app.cell
async def _():
    import sys

    if "pyodide" in sys.modules:
        # WASM (docs gallery): TEMPORARY until globalsplinefit is on PyPI —
        # install the wheel published under /wheels/ on the same site the
        # notebook is served from (also works on staging or local hosts).
        from urllib.parse import urljoin

        import marimo as _mo
        import micropip

        # notebook_location() resolution depth differs between marimo export
        # layouts — try site-root /wheels/ from both plausible depths
        _err = None
        for _up in ("../../", "../../../"):
            try:
                await micropip.install(
                    urljoin(
                        str(_mo.notebook_location()) + "/",
                        _up + "wheels/globalsplinefit-2.0.0a1-py3-none-any.whl",
                    )
                )
                _err = None
                break
            except Exception as _e:  # noqa: BLE001 — 404 lands as generic error
                _err = _e
        if _err is not None:
            raise _err

    import matplotlib.pyplot as plt
    import numpy as np

    from globalsplinefit import GSFEnergy, GSFEnergyPerNucleon
    from globalsplinefit.data_management import MODEL_VERSIONS

    plt.rcParams["figure.figsize"] = (9, 6)

    # TEMPORARY pre-publication marker: stamp every figure this notebook
    # shows with a diagonal PRELIMINARY watermark. Delete at the GSF 2026
    # release (guarded so a cell re-run does not stack wrappers).
    if not getattr(plt.show, "_gsf_preliminary", False):
        _orig_show = plt.show

        def _show_with_watermark(*args, **kwargs):
            plt.gcf().text(
                0.5,
                0.5,
                "PRELIMINARY",
                ha="center",
                va="center",
                rotation=30,
                fontsize=42,
                fontweight="bold",
                color="gray",
                alpha=0.18,
                zorder=1000,
            )
            return _orig_show(*args, **kwargs)

        _show_with_watermark._gsf_preliminary = True
        plt.show = _show_with_watermark

    return GSFEnergy, GSFEnergyPerNucleon, MODEL_VERSIONS, np, plt


@app.cell
def _(MODEL_VERSIONS, mo):
    version = mo.ui.dropdown(options=list(MODEL_VERSIONS), value="2026.0", label="model")
    ref_version = mo.ui.dropdown(
        options=list(MODEL_VERSIONS), value="2025", label="reference (comparison)"
    )
    mo.hstack([version, ref_version], justify="start")
    return ref_version, version


@app.cell
def _(GSFEnergy, np, version):
    gsf = GSFEnergy(version=version.value)
    energy = np.logspace(0, 11, 300)  # GeV, full GSF range
    energy_exponent = 2.6  # E^2.6 flux scaling for display
    scaled = energy**energy_exponent
    groups = ["p", "He", "O*", "Fe*"]
    colors = {
        "p": "#d62728",
        "He": "#ff7f0e",
        "O*": "#2ca02c",
        "Fe*": "#1f77b4",
        "all": "k",
    }
    return colors, energy, energy_exponent, groups, gsf, scaled


@app.cell
def _(energy, np):
    def flux_err(model, group):
        """Flux and 1-sigma error for one mass group on the shared grid."""
        _f = np.asarray(model.flux(energy, group), float)
        _e = np.asarray(model.error(energy, group), float)
        return _f, _e

    return (flux_err,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Flux spectrum

    All-particle flux and the four mass groups, scaled by E^2.6, with 1-sigma
    bands propagated from the parameter covariance.
    """)
    return


@app.cell
def _(colors, energy, flux_err, groups, gsf, np, plt, scaled, version):
    _fig, _ax = plt.subplots()
    _tf = np.asarray(gsf.total_flux(energy), float)
    _te = np.asarray(gsf.total_error(energy), float)
    _ax.plot(energy, _tf * scaled, color="k", lw=2.2, label="all-particle")
    _ax.fill_between(
        energy, (_tf - _te) * scaled, (_tf + _te) * scaled, color="k", alpha=0.18, lw=0
    )
    for _g in groups:
        _f, _e = flux_err(gsf, _g)
        _ax.plot(energy, _f * scaled, color=colors[_g], lw=1.6, label=_g)
        _ax.fill_between(
            energy,
            (_f - _e) * scaled,
            (_f + _e) * scaled,
            color=colors[_g],
            alpha=0.18,
            lw=0,
        )
    _ax.set_xscale("log")
    _ax.set_yscale("log")
    _ax.set_ylim(1e2, 6e4)
    _ax.set_xlabel("E [GeV]")
    _ax.set_ylabel(r"$E^{2.6}\,\Phi$ [GeV$^{1.6}$m$^{-2}$s$^{-1}$sr$^{-1}$]")
    _ax.set_title(f"{version.value}: flux spectrum (×E$^{{2.6}}$, ±1σ)")
    _ax.legend(ncol=2, fontsize=9)
    _ax.grid(alpha=0.3)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Composition fractions""")
    return


@app.cell
def _(colors, energy, groups, gsf, plt, version):
    _fig, _ax = plt.subplots()
    for _g in groups:
        _ax.plot(energy, gsf.fraction(energy, _g), color=colors[_g], lw=1.8, label=_g)
    _ax.set_xscale("log")
    _ax.set_ylim(0, 1)
    _ax.set_xlabel("E [GeV]")
    _ax.set_ylabel("flux fraction")
    _ax.set_title(f"{version.value}: composition fractions")
    _ax.legend(ncol=2, fontsize=9)
    _ax.grid(alpha=0.3)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## ln A moments""")
    return


@app.cell
def _(energy, gsf, plt, version):
    _fig, _ax = plt.subplots(1, 2, figsize=(15, 5.5))
    _ax[0].plot(energy, gsf.mean_lnA(energy), "k-", lw=1.8)
    _ax[0].set_title(f"{version.value}: ⟨ln A⟩")
    _ax[0].set_ylabel("⟨ln A⟩")
    _ax[1].plot(energy, gsf.var_lnA(energy), "k-", lw=1.8)
    _ax[1].set_title(f"{version.value}: Var(ln A)")
    _ax[1].set_ylabel("Var(ln A)")
    for _a in _ax:
        _a.set_xscale("log")
        _a.set_xlabel("E [GeV]")
        _a.grid(alpha=0.3)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Relative uncertainty""")
    return


@app.cell
def _(colors, energy, flux_err, groups, gsf, np, plt, version):
    _fig, _ax = plt.subplots()
    _tf = np.asarray(gsf.total_flux(energy), float)
    _te = np.asarray(gsf.total_error(energy), float)
    _ax.plot(energy, _te / _tf, "k-", lw=2.0, label="all-particle")
    for _g in groups:
        _f, _e = flux_err(gsf, _g)
        _ax.plot(
            energy,
            _e / np.where(_f > 0, _f, np.nan),
            color=colors[_g],
            lw=1.4,
            label=_g,
        )
    _ax.set_xscale("log")
    _ax.set_yscale("log")
    _ax.set_ylim(1e-3, 3)
    _ax.set_xlabel("E [GeV]")
    _ax.set_ylabel(r"relative uncertainty $\sigma/\Phi$")
    _ax.set_title(f"{version.value}: relative uncertainty")
    _ax.legend(ncol=2, fontsize=9)
    _ax.grid(alpha=0.3)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Nucleon flux

    Proton + neutron flux per energy-per-nucleon, summed over all nuclei —
    the quantity that drives atmospheric lepton production.
    """)
    return


@app.cell
def _(GSFEnergyPerNucleon, energy_exponent, np, plt, version):
    _nuc = GSFEnergyPerNucleon(version=version.value)
    _en = np.logspace(0, 11, 300)
    _tot = np.asarray(_nuc.total_flux(_en), float)
    _fig, _ax = plt.subplots()
    _ax.plot(_en, _tot * _en**energy_exponent, "k-", lw=2.0, label="nucleon (p+n)")
    _ax.set_xscale("log")
    _ax.set_yscale("log")
    _ax.set_xlabel("E/nucleon [GeV]")
    _ax.set_ylabel(r"$E^{2.6}\,\Phi_{\rm nucleon}$")
    _ax.set_title(f"{version.value}: nucleon flux")
    _ax.legend(fontsize=9)
    _ax.grid(alpha=0.3)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## All-particle flux correlation

    Correlation of the all-particle flux between energies, from the summed
    group-pair covariances.
    """)
    return


@app.cell
def _(groups, gsf, np, plt, version):
    _xc = np.logspace(1, 11, 80)
    _cov = np.zeros((len(_xc), len(_xc)))
    for _l1 in groups:
        for _l2 in groups:
            _cov += np.asarray(gsf.covariance(_l1, _l2, _xc), float)
    _d = np.sqrt(np.clip(np.diag(_cov), 1e-300, None))
    _corr = _cov / np.outer(_d, _d)
    _fig, _ax = plt.subplots(figsize=(7, 6))
    _im = _ax.pcolormesh(
        np.log10(_xc), np.log10(_xc), _corr, cmap="RdBu_r", vmin=-1, vmax=1
    )
    _fig.colorbar(_im, ax=_ax, label="correlation")
    _ax.set_xlabel("log10 E [GeV]")
    _ax.set_ylabel("log10 E [GeV]")
    _ax.set_title(f"{version.value}: all-particle flux correlation")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo, ref_version, version):
    mo.md(rf"""
    ## Back-to-back comparison: {version.value} vs {ref_version.value}

    Main panel: both models ×E^2.6 (reference solid, model dotted).
    Sub-panels: per-group model/reference ratio (colored, with the model's
    error band) against the reference's own relative uncertainty (grey band).
    """)
    return


@app.cell
def _(
    GSFEnergy,
    colors,
    energy,
    flux_err,
    groups,
    gsf,
    np,
    plt,
    ref_version,
    scaled,
    version,
):
    from matplotlib.lines import Line2D

    _ref = GSFEnergy(version=ref_version.value)
    _items = groups + ["all"]

    def _fe(model, group):
        if group == "all":
            return (
                np.asarray(model.total_flux(energy), float),
                np.asarray(model.total_error(energy), float),
            )
        return flux_err(model, group)

    _fn = {_g: _fe(gsf, _g) for _g in _items}
    _fr = {_g: _fe(_ref, _g) for _g in _items}

    _fig = plt.figure(figsize=(8.5, 11))
    _gs = _fig.add_gridspec(6, 1, height_ratios=[3.2, 1, 1, 1, 1, 1], hspace=0.12)
    _ax0 = _fig.add_subplot(_gs[0])
    _rax = [_fig.add_subplot(_gs[_i + 1], sharex=_ax0) for _i in range(5)]
    for _g in _items:
        _c = colors[_g]
        _ax0.plot(energy, _fr[_g][0] * scaled, _c, ls="-", lw=1.8)
        _ax0.fill_between(
            energy,
            (_fr[_g][0] - _fr[_g][1]) * scaled,
            (_fr[_g][0] + _fr[_g][1]) * scaled,
            color=_c,
            alpha=0.18,
            lw=0,
        )
        _ax0.plot(energy, _fn[_g][0] * scaled, _c, ls=":", lw=1.8)
    _ax0.set_xscale("log")
    _ax0.set_yscale("log")
    _ax0.set_ylim(1e2, 6e4)
    _ax0.set_ylabel(r"$E^{2.6}\,\Phi$")
    _ax0.set_title(
        f"Back-to-back: {ref_version.value} (solid) vs {version.value} (dotted)"
    )
    _leg = [Line2D([], [], color=colors[_g], label=_g) for _g in _items]
    _leg += [
        Line2D([], [], color="gray", ls="-", label=ref_version.value),
        Line2D([], [], color="gray", ls=":", label=version.value),
    ]
    _ax0.legend(handles=_leg, ncol=2, fontsize=8)
    for _ax, _g in zip(_rax, _items, strict=True):
        _c = colors[_g]
        (_fnf, _fne), (_frf, _fre) = _fn[_g], _fr[_g]
        with np.errstate(divide="ignore", invalid="ignore"):
            _ratio = _fnf / np.where(_frf > 0, _frf, np.nan)
            _rel = _fre / np.where(_frf > 0, _frf, np.nan)
            _band = _fne / np.where(_frf > 0, _frf, np.nan)
        _ax.fill_between(energy, 1 - _rel, 1 + _rel, color="0.8", lw=0)
        _ax.plot(energy, _ratio, _c, lw=1.6)
        _ax.fill_between(
            energy, _ratio - _band, _ratio + _band, color=_c, alpha=0.25, lw=0
        )
        _ax.axhline(1.0, color="k", lw=0.6)
        _ax.set_ylabel(f"{_g}\nnew/ref", fontsize=8)
        _ax.set_ylim(0.6, 1.4)
        _ax.set_xscale("log")
        _ax.grid(alpha=0.25)
        for _t in _ax.get_xticklabels():
            _t.set_visible(False)
    for _t in _rax[-1].get_xticklabels():
        _t.set_visible(True)
    _rax[-1].set_xlabel("E [GeV]")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Custom fits

    Every panel works the same for a parameter directory that is not in the
    registry — construct the model with `GSFEnergy(data_path="my_fit/")`
    instead of `version=...` (run the notebook locally for that; the browser
    version has no filesystem access). To save any figure, use
    `fig.savefig("name.png", dpi=120, bbox_inches="tight")` in a cell.
    """)
    return


if __name__ == "__main__":
    app.run()
