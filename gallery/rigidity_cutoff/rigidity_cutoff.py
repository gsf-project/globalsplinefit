import marimo

__generated_with = "0.23.15"
app = marimo.App(
    width="full",
    app_title="GSF — Rigidity Cutoff",
    css_file="gallery.css",
    html_head_file="gallery_head.html",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Geomagnetic Rigidity Cutoff

    Earth's magnetic field deflects cosmic rays below a location-dependent rigidity threshold $R_\text{cut}$ (~0 GV at the poles, ~15–17 GV near the equator). Because rigidity depends on charge $Z$, the cutoff affects nuclear species differently. We compare its effect on three GSF model types using $R_\text{cut} = 20$ GV (high, for visibility).
    """)
    return


@app.cell
async def _():
    import sys

    NOTEBOOK = "rigidity_cutoff"  # used by the download buttons at the end
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

    import matplotlib.pyplot as plt
    import numpy as np

    from globalsplinefit import (
        MODEL_VERSIONS,
        GSFEnergy,
        GSFEnergyPerNucleon,
        GSFRigidity,
    )

    plt.rcParams["figure.figsize"] = (10, 6)

    R_CUT = 20.0  # GV
    groups = [
        ("p", "r", "H*"),
        ("He", "orange", "He*"),
        ("O*", "g", "O*"),
        ("Fe*", "b", "Fe*"),
    ]

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
        GSFEnergy,
        GSFEnergyPerNucleon,
        GSFRigidity,
        MODEL_VERSIONS,
        NOTEBOOK,
        R_CUT,
        SITE,
        autoscale,
        groups,
        np,
        plt,
        show,
    )


@app.cell(hide_code=True)
def _(MODEL_VERSIONS, mo):
    version = mo.ui.dropdown(
        options=list(MODEL_VERSIONS), value="2026.1", label="model version"
    )
    exponent = mo.ui.slider(
        start=0.0,
        stop=3.0,
        step=0.1,
        value=2.6,
        label="flux scaling  E^k",
        show_value=True,
    )
    plot_size = mo.ui.slider(
        start=0.6,
        stop=1.6,
        step=0.1,
        value=1.0,
        label="plot size",
        show_value=True,
    )
    mo.hstack([version, exponent, plot_size], justify="start", gap=2)
    return exponent, plot_size, version


@app.cell(hide_code=True)
def _(mo, plot_size):
    # Drives --gsf-plot-scale, which gallery.css multiplies into every figure's
    # width. Scaling the rendered raster (rather than the figsize) is what keeps
    # it proportional: fonts, line widths and the aspect ratio all ride along.
    mo.Html(f"<style>:root{{--gsf-plot-scale:{plot_size.value}}}</style>")
    return


@app.cell
def _(exponent):
    energy_exponent = exponent.value  # display scaling, driven by the slider
    return (energy_exponent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Nucleus Flux (`GSFEnergy`)

    At total energy $E$, a nucleus with charge $Z$ and mass $A$ has rigidity $R = \sqrt{E^2 - (A m_N)^2} / Z$. A fixed rigidity cutoff therefore sits at a total energy $E_\text{cut} \simeq Z R_\text{cut}$, higher for heavier nuclei: at $R_\text{cut} = 20$ GV it falls at 20 GeV for p, 40 GeV for He, 161 GeV for O and 523 GeV for Fe. Left panel: absolute fluxes with (dashed) and without (solid) cutoff. Right panel: suppression ratio.
    """)
    return


@app.cell
def _(
    GSFEnergy,
    R_CUT,
    autoscale,
    energy_exponent,
    groups,
    np,
    plt,
    show,
    version,
):
    energy = np.logspace(0, 3, 500)  # 1 GeV to 1 TeV
    gsf = GSFEnergy(version=version.value)
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(16, 7))
    for _name, _color, _label in groups:
        _f0 = gsf.flux(energy, _name)
        _f1 = gsf.flux(energy, _name, rigidity_cutoff=R_CUT)
        _ax1.plot(
            energy,
            _f0 * energy**energy_exponent,
            "-",
            color=_color,
            lw=1.5,
            label=_label,
        )
        _ax1.plot(energy, _f1 * energy**energy_exponent, "--", color=_color, lw=1.5)
    _f0_tot = gsf.total_flux(energy)
    _f1_tot = gsf.total_flux(energy, rigidity_cutoff=R_CUT)
    _ax1.plot(
        energy, _f0_tot * energy**energy_exponent, "-", color="k", lw=2, label="Total"
    )
    _ax1.plot(energy, _f1_tot * energy**energy_exponent, "--", color="k", lw=2)
    _ax1.set_xscale("log")
    _ax1.set_yscale("log")
    _ax1.set_xlabel("Energy [GeV]")
    _ax1.set_ylabel(f"Nucleus Flux x (E/GeV)$^{{{energy_exponent}}}$ [1/(GeV m² s sr)]")
    _ax1.set_xlim(energy[0], energy[-1])
    autoscale(_ax1, _f0_tot * energy**energy_exponent)
    _ax1.legend()
    _ax1.grid(True, alpha=0.3)
    _ax1.set_title(f"Nucleus Flux (solid: no cutoff, dashed: $R_{{cut}}$ = {R_CUT} GV)")
    for _name, _color, _label in groups:
        _f0 = gsf.flux(energy, _name)
        _f1 = gsf.flux(energy, _name, rigidity_cutoff=R_CUT)
        with np.errstate(divide="ignore", invalid="ignore"):
            _ratio = np.where(_f0 > 0, _f1 / _f0, np.nan)
        _ax2.plot(energy, _ratio, "-", color=_color, lw=1.5, label=_label)
    _ax2.set_xscale("log")
    # Ratio plot
    _ax2.set_xlabel("Energy [GeV]")
    _ax2.set_ylabel("Flux ratio (with / without cutoff)")
    _ax2.set_xlim(energy[0], energy[-1])
    _ax2.set_ylim(-0.05, 1.15)
    _ax2.legend()
    _ax2.grid(True, alpha=0.3)
    _ax2.set_title(f"Suppression by $R_{{cut}}$ = {R_CUT} GV")
    _ax2.axhline(1.0, color="gray", ls="--", alpha=0.5)
    plt.tight_layout()
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rigidity Flux (`GSFRigidity`)

    In rigidity space the cutoff sits at $R_\text{cut}$ for every species alike, as a sigmoid of width `cutoff_width` (1 GV by default) that models the geomagnetic penumbra: for a 20 GV cutoff the transmission runs 0.27 / 0.50 / 0.73 at 19 / 20 / 21 GV. Pass `cutoff_width=0` for a hard edge.
    """)
    return


@app.cell
def _(
    GSFRigidity,
    R_CUT,
    autoscale,
    energy_exponent,
    groups,
    np,
    plt,
    show,
    version,
):
    rigidity = np.logspace(-1, 4, 500)  # 0.1 GV to 10 TV
    rig_exponent = energy_exponent
    gsf_r = GSFRigidity(version=version.value)
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(16, 7))
    for _name, _color, _label in groups:
        _f0 = gsf_r.flux(rigidity, _name)
        _f1 = gsf_r.flux(rigidity, _name, rigidity_cutoff=R_CUT)
        _ax1.plot(
            rigidity,
            _f0 * rigidity**rig_exponent,
            "-",
            color=_color,
            lw=1.5,
            label=_label,
        )
        _ax1.plot(rigidity, _f1 * rigidity**rig_exponent, "--", color=_color, lw=1.5)
    _f0_tot = gsf_r.total_flux(rigidity)
    _f1_tot = gsf_r.total_flux(rigidity, rigidity_cutoff=R_CUT)
    _ax1.plot(
        rigidity, _f0_tot * rigidity**rig_exponent, "-", color="k", lw=2, label="Total"
    )
    _ax1.plot(rigidity, _f1_tot * rigidity**rig_exponent, "--", color="k", lw=2)
    _ax1.axvline(
        R_CUT, color="gray", ls=":", lw=2, alpha=0.7, label=f"$R_{{cut}}$ = {R_CUT} GV"
    )
    _ax1.set_xscale("log")
    _ax1.set_yscale("log")
    _ax1.set_xlabel("Rigidity [GV]")
    _ax1.set_ylabel(f"Rigidity Flux x (R/GV)$^{{{rig_exponent}}}$ [1/(GV m² s sr)]")
    _ax1.set_xlim(rigidity[0], rigidity[-1])
    autoscale(_ax1, _f0_tot * rigidity**rig_exponent)
    _ax1.legend(fontsize=9)
    _ax1.grid(True, alpha=0.3)
    _ax1.set_title(
        f"Rigidity Flux (solid: no cutoff, dashed: $R_{{cut}}$ = {R_CUT} GV)"
    )
    for _name, _color, _label in groups:
        _f0 = gsf_r.flux(rigidity, _name)
        _f1 = gsf_r.flux(rigidity, _name, rigidity_cutoff=R_CUT)
        with np.errstate(divide="ignore", invalid="ignore"):
            _ratio = np.where(_f0 > 0, _f1 / _f0, np.nan)
        _ax2.plot(rigidity, _ratio, "-", color=_color, lw=1.5, label=_label)
    _ax2.axvline(
        R_CUT, color="gray", ls=":", lw=2, alpha=0.7, label=f"$R_{{cut}}$ = {R_CUT} GV"
    )
    _ax2.set_xscale("log")
    _ax2.set_xlabel("Rigidity [GV]")
    # Ratio
    _ax2.set_ylabel("Flux ratio (with / without cutoff)")
    _ax2.set_xlim(rigidity[0], rigidity[-1])
    _ax2.set_ylim(-0.05, 1.15)
    _ax2.legend()
    _ax2.grid(True, alpha=0.3)
    _ax2.set_title(f"Sharp cutoff at $R_{{cut}}$ = {R_CUT} GV")
    _ax2.axhline(1.0, color="gray", ls="--", alpha=0.5)
    plt.tight_layout()
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Nucleon Flux (`GSFEnergyPerNucleon`)

    The cutoff is applied per element during the nucleon-flux summation. At a given $E_N$, nucleus $i$ has rigidity $R_i = (A_i/Z_i) \sqrt{E_N^2 - m_N^2}$, so the cutoff in $E_N$ is roughly twice as high for protons as for heavier nuclei. Left: absolute flux with 1-sigma bands; uncertainties propagate through the cutoff via the Jacobian. Right: relative uncertainty with (dashed) and without (solid) cutoff.
    """)
    return


@app.cell
def _(
    GSFEnergyPerNucleon,
    R_CUT,
    autoscale,
    energy_exponent,
    groups,
    np,
    plt,
    show,
    version,
):
    ekin = np.logspace(0, 3, 500)  # 1 GeV to 1 TeV per nucleon
    en_exponent = energy_exponent
    gsf_n = GSFEnergyPerNucleon(version=version.value)
    _fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for _name, _color, _label in groups:
        _f0 = gsf_n.flux(ekin, _name)
        _f1 = gsf_n.flux(ekin, _name, rigidity_cutoff=R_CUT)
        e1 = gsf_n.error(ekin, _name, rigidity_cutoff=R_CUT)
        axes[0].plot(
            ekin, _f0 * ekin**en_exponent, "-", color=_color, lw=1.5, label=_label
        )
        axes[0].plot(ekin, _f1 * ekin**en_exponent, "--", color=_color, lw=1.5)
        axes[0].fill_between(
            ekin,
            (_f1 - e1) * ekin**en_exponent,
            (_f1 + e1) * ekin**en_exponent,
            facecolor=_color,
            alpha=0.15,
        )
    _f0_tot = gsf_n.total_flux(ekin)
    _f1_tot = gsf_n.total_flux(ekin, rigidity_cutoff=R_CUT)
    e1_tot = gsf_n.total_error(ekin, rigidity_cutoff=R_CUT)
    axes[0].plot(ekin, _f0_tot * ekin**en_exponent, "-", color="k", lw=2, label="Total")
    axes[0].plot(ekin, _f1_tot * ekin**en_exponent, "--", color="k", lw=2)
    axes[0].fill_between(
        ekin,
        (_f1_tot - e1_tot) * ekin**en_exponent,
        (_f1_tot + e1_tot) * ekin**en_exponent,
        facecolor="k",
        alpha=0.1,
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("$E_N$ [GeV/nucleon]")
    axes[0].set_ylabel(
        f"Nucleon Flux x $(E_N/\\mathrm{{GeV}})^{{{en_exponent}}}$ [1/(GeV m² s sr)]"
    )
    axes[0].set_xlim(ekin[0], ekin[-1])
    autoscale(axes[0], _f0_tot * ekin**en_exponent)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(
        f"Nucleon Flux (solid: no cutoff, dashed: $R_{{cut}}$ = {R_CUT} GV)"
    )
    for _name, _color, _label in groups:
        _f0 = gsf_n.flux(ekin, _name)
        e0 = gsf_n.error(ekin, _name)
        _f1 = gsf_n.flux(ekin, _name, rigidity_cutoff=R_CUT)
        e1 = gsf_n.error(ekin, _name, rigidity_cutoff=R_CUT)
        with np.errstate(divide="ignore", invalid="ignore"):
            re0 = e0 / np.maximum(_f0, 1e-90)
            re1 = e1 / np.maximum(_f1, 1e-90)
        valid = (_f1 > 0) & np.isfinite(re1)
        axes[1].plot(ekin, re0, "-", color=_color, lw=1.5, alpha=0.4)
        if np.any(valid):
            axes[1].plot(
                ekin[valid], re1[valid], "--", color=_color, lw=1.5, label=_label
            )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("$E_N$ [GeV/nucleon]")
    axes[1].set_ylabel("Relative Uncertainty")
    axes[1].set_xlim(ekin[0], ekin[-1])
    axes[1].set_ylim(0.003, 1)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Relative Uncertainty (solid: no cutoff, dashed: with cutoff)")
    plt.tight_layout()
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `rigidity_cutoff` is accepted by all flux and error methods.
    """)
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
