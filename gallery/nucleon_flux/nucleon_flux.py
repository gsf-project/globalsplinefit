import marimo

__generated_with = "0.23.15"
app = marimo.App(
    width="full",
    app_title="GSF — Nucleon Flux",
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
    # Nucleon Flux for Air Shower Simulations

    Air shower codes (e.g. CORSIKA) take cosmic ray fluxes as nucleon fluxes per GeV/nucleon. `GSFEnergyPerNucleon` provides this from the fitted all-particle spectra, summed over the 28 nuclear species into proton and neutron components.
    """)
    return


@app.cell
async def _():
    import sys

    NOTEBOOK = "nucleon_flux"  # used by the download buttons at the end
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

    from globalsplinefit import MODEL_VERSIONS, GSFEnergyPerNucleon

    plt.rcParams["figure.figsize"] = (10, 6)

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
        SITE,
        autoscale,
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
        stop=3.5,
        step=0.1,
        value=3.0,
        label="flux scaling  E_N^k",
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
def _(GSFEnergyPerNucleon, np, version):
    energy_per_nucleon = 10 ** np.linspace(0, 10, 200)  # GeV/nucleon
    gsf_nucleon = GSFEnergyPerNucleon(version=version.value)
    return energy_per_nucleon, gsf_nucleon


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Nucleon fluxes

    `flux()` returns the proton + neutron sum for a group or element; `total_flux()` sums over all 28 species. Use `p_and_n_flux()` for the components separately (shape `(2, N)`).
    """)
    return


@app.cell
def _(energy_per_nucleon, gsf_nucleon):
    # "H*", "O*", "Fe*" select the mass groups; bare "p", "O", "Fe" select
    # the single leading elements
    proton_flux = gsf_nucleon.flux(energy_per_nucleon, "H*")
    helium_flux = gsf_nucleon.flux(energy_per_nucleon, "He")
    oxygen_flux = gsf_nucleon.flux(energy_per_nucleon, "O*")
    iron_flux = gsf_nucleon.flux(energy_per_nucleon, "Fe*")
    total_flux = gsf_nucleon.total_flux(energy_per_nucleon)
    return helium_flux, iron_flux, oxygen_flux, proton_flux, total_flux


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Uncertainties

    `error()` and `total_error()` return 1-sigma absolute uncertainties on the nucleon flux.
    """)
    return


@app.cell
def _(energy_per_nucleon, gsf_nucleon):
    proton_error = gsf_nucleon.error(energy_per_nucleon, "H*")
    helium_error = gsf_nucleon.error(energy_per_nucleon, "He")
    oxygen_error = gsf_nucleon.error(energy_per_nucleon, "O*")
    iron_error = gsf_nucleon.error(energy_per_nucleon, "Fe*")
    total_error = gsf_nucleon.total_error(energy_per_nucleon)
    return helium_error, iron_error, oxygen_error, proton_error, total_error


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Export to text files
    """)
    return


@app.cell
def _(
    energy_per_nucleon,
    helium_error,
    helium_flux,
    iron_error,
    iron_flux,
    gsf_nucleon,
    np,
    oxygen_error,
    oxygen_flux,
    proton_error,
    proton_flux,
    total_error,
    total_flux,
):
    # Export nucleon flux data. Stamp the provenance: the physical model
    # version incl. its revision (gsf_nucleon.version) and the code version.
    import globalsplinefit

    provenance = (
        f"parameter set {gsf_nucleon.version}, "
        f"globalsplinefit {globalsplinefit.__version__}\n"
    )
    np.savetxt(
        "gsf_nucleon_flux.dat",
        np.transpose(
            (
                energy_per_nucleon,
                proton_flux,
                helium_flux,
                oxygen_flux,
                iron_flux,
                total_flux,
            )
        ),
        fmt="%.10e",
        header=provenance
        + "energy, hydrogen group, helium, oxygen group, iron group, total\n"
        "[units: energy in GeV per nucleon, flux in 1/(GeV m2 s sr)]",
    )

    np.savetxt(
        "gsf_nucleon_flux_error.dat",
        np.transpose(
            (
                energy_per_nucleon,
                proton_error,
                helium_error,
                oxygen_error,
                iron_error,
                total_error,
            )
        ),
        fmt="%.10e",
        header=provenance
        + "energy, hydrogen group, helium, oxygen group, iron group, total\n"
        "[units: energy in GeV per nucleon, flux error in 1/(GeV m2 s sr)]",
    )

    print(f"Saved nucleon flux and error data ({len(energy_per_nucleon)} points each)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spectrum

    Flux scaled by E_N^k (slider above) to flatten the steep falloff. Shaded bands are 1-sigma; the axis follows the scaling.
    """)
    return


@app.cell
def _(
    autoscale,
    energy_per_nucleon,
    exponent,
    helium_error,
    helium_flux,
    iron_error,
    iron_flux,
    oxygen_error,
    oxygen_flux,
    plt,
    proton_error,
    proton_flux,
    show,
    total_error,
    total_flux,
):
    # Plot nucleon flux with error bands
    _fig, _ax = plt.subplots(figsize=(12, 8))
    _scale = energy_per_nucleon**exponent.value
    _components = [
        (proton_flux, proton_error, "r", "H*", 1),
        (helium_flux, helium_error, "orange", "He*", 1),
        (oxygen_flux, oxygen_error, "g", "O*", 1),
        (iron_flux, iron_error, "b", "Fe*", 1),
        (total_flux, total_error, "k", "Total", 2),
    ]
    for _flux, _error, _color, _label, _lw in _components:
        _ax.plot(
            energy_per_nucleon, _flux * _scale, "-", color=_color, lw=_lw, label=_label
        )
        _ax.fill_between(
            energy_per_nucleon,
            (_flux - _error) * _scale,
            (_flux + _error) * _scale,
            facecolor=_color,
            alpha=0.2,
        )
    _ax.loglog()
    _ax.set_xlabel("$E_N$ [GeV/nucleon]")
    _ax.set_ylabel(
        f"Nucleon Flux × $(E_N/\\mathrm{{GeV}})^{{{exponent.value:g}}}$ [1/(GeV m² s sr)]"
    )
    _ax.set_xlim(energy_per_nucleon[0], energy_per_nucleon[-1])
    autoscale(_ax, total_flux * _scale, (total_flux + total_error) * _scale)
    _ax.set_title("Cosmic Ray Nucleon Flux (GSF)")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Relative uncertainties

    A few percent in the well-measured 10 GeV – 100 TeV per nucleon range, growing toward the boundaries.

    Each curve stops where $\sigma$ exceeds $10\,\Phi$, the point beyond which the group carries no information; once its flux underflows to zero the ratio diverges outright. Fe\* is the first to go, just below $3\times10^9$ GeV/nucleon, which is $1.7\times10^{11}$ GeV of total energy, past the end of the fitted range.
    """)
    return


@app.cell
def _(
    autoscale,
    energy_per_nucleon,
    helium_error,
    helium_flux,
    iron_error,
    iron_flux,
    np,
    oxygen_error,
    oxygen_flux,
    plt,
    proton_error,
    proton_flux,
    show,
    total_error,
    total_flux,
):
    # Plot nucleon relative uncertainties.
    #
    # Past the end of its fitted range a group's flux falls off a cliff and
    # eventually underflows to exactly zero, while its error stays finite, so
    # sigma/flux runs away to 1e60 and flattens every real curve into a line.
    # Cut the curve where it stops carrying information.
    UNINFORMATIVE = 10.0

    def rel_uncertainty(_flux, _error):
        with np.errstate(divide="ignore", invalid="ignore"):
            _rel = np.where(_flux > 0, _error / _flux, np.inf)
        return np.where(np.isfinite(_rel) & (_rel <= UNINFORMATIVE), _rel, np.nan)

    plt.figure(figsize=(12, 8))
    error_components = [
        (rel_uncertainty(proton_flux, proton_error), "r", "H*"),
        (rel_uncertainty(helium_flux, helium_error), "orange", "He*"),
        (rel_uncertainty(oxygen_flux, oxygen_error), "g", "O*"),
        (rel_uncertainty(iron_flux, iron_error), "b", "Fe*"),
        (rel_uncertainty(total_flux, total_error), "k", "Total"),
    ]
    for rel_error, _color, _label in error_components:
        # NaN-gapped rather than index-filtered, so a break stays a break
        plt.plot(
            energy_per_nucleon,
            rel_error,
            "-",
            color=_color,
            label=_label,
            lw=2,
        )
    plt.loglog()
    plt.xlabel("$E_N$ [GeV/nucleon]")
    plt.ylabel("Relative Uncertainty (σ/flux)")
    plt.xlim(energy_per_nucleon[0], energy_per_nucleon[-1])
    autoscale(plt.gca(), *[_e for _e, _c, _l in error_components], pad=1.3)
    plt.title("Nucleon Flux Relative Uncertainties (GSF)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.01, color="gray", linestyle="--", alpha=0.5)
    plt.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5)
    # Reference lines
    show()
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
