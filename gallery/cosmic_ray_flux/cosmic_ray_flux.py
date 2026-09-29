import marimo

__generated_with = "0.23.15"
app = marimo.App(
    width="full",
    app_title="GSF — Cosmic Ray Flux",
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
    # Cosmic Ray Flux by Mass Group

    Compute and plot differential fluxes for the four GSF mass groups (H\*, He\*, O\*, Fe\*) as a function of **total energy per nucleus**, with 1-sigma error bands propagated from the parameter covariance.

    Pick a model version and the display scaling below; every figure updates.
    """)
    return


@app.cell
async def _():
    import sys

    NOTEBOOK = "cosmic_ray_flux"  # used by the download buttons at the end
    IN_WASM = "pyodide" in sys.modules
    SITE = ""

    if IN_WASM:
        # WASM (docs gallery): install the wheel built together with these
        # docs and published under /wheels/ on the same site, so the notebook
        # always runs the documented version (works on any host).
        from urllib.parse import urljoin

        import marimo as _mo
        import micropip
        from pyodide.http import pyfetch

        # notebook_location() resolution depth differs between marimo export
        # layouts — try site-root /wheels/ from every plausible depth, and
        # keep the depth that worked as the site root for other assets.
        _base = str(_mo.notebook_location()) + "/"
        _err = None
        for _up in ("", "../", "../../", "../../../"):
            try:
                # wheels/latest.json names the wheel the docs build published
                _wheels = urljoin(_base, _up + "wheels/")
                _resp = await pyfetch(_wheels + "latest.json")
                if not _resp.ok:
                    raise OSError(f"no wheel manifest at {_wheels}")
                await micropip.install(_wheels + (await _resp.json())["wheel"])
                SITE = urljoin(_base, _up)
                _err = None
                break
            except Exception as _e:  # noqa: BLE001 — 404 lands as generic error
                _err = _e
        if _err is not None:
            raise _err

    import matplotlib.pyplot as plt
    import numpy as np

    from globalsplinefit import MODEL_VERSIONS, GSFEnergy

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
        GSFEnergy,
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
def _(GSFEnergy, np, version):
    energy = 10 ** np.linspace(0, 11, 1000)  # 1 GeV to 1e11 GeV
    gsf = GSFEnergy(version=version.value)
    return energy, gsf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fluxes

    Four mass groups led by H\*, He\*, O\*, Fe\* (asterisks mark groups that include neighboring elements scaled from the leader). `flux()` returns the per-group differential flux in 1/(GeV m² s sr); `total_flux()` returns the all-particle sum.
    """)
    return


@app.cell
def _(energy, gsf):
    # "H*", "O*", "Fe*" select the mass groups; bare "p", "O", "Fe" select
    # the single leading elements
    proton_flux = gsf.flux(energy, "H*")
    helium_flux = gsf.flux(energy, "He")
    oxygen_flux = gsf.flux(energy, "O*")
    iron_flux = gsf.flux(energy, "Fe*")
    total_flux = gsf.total_flux(energy)
    return helium_flux, iron_flux, oxygen_flux, proton_flux, total_flux


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Uncertainties

    `error()` and `total_error()` return 1-sigma uncertainties propagated from the parameter covariance via the spline Jacobian.
    """)
    return


@app.cell
def _(energy, gsf):
    proton_error = gsf.error(energy, "H*")
    helium_error = gsf.error(energy, "He")
    oxygen_error = gsf.error(energy, "O*")
    iron_error = gsf.error(energy, "Fe*")
    total_error = gsf.total_error(energy)
    return helium_error, iron_error, oxygen_error, proton_error, total_error


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Flux spectra

    Differential flux scaled by E^k (slider above) to flatten the steep power law. Shaded bands are 1-sigma; the axis follows the scaling.
    """)
    return


@app.cell
def _(
    autoscale,
    energy,
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
    _fig, _ax = plt.subplots()
    _scale = energy**exponent.value
    _components = [
        (proton_flux, proton_error, "r", "H*", 1),
        (helium_flux, helium_error, "orange", "He*", 1),
        (oxygen_flux, oxygen_error, "g", "O*", 1),
        (iron_flux, iron_error, "b", "Fe*", 1),
        (total_flux, total_error, "k", "Total", 2),
    ]
    for _flux, _error, _color, _label, _lw in _components:
        _ax.plot(energy, _flux * _scale, "-", color=_color, lw=_lw, label=_label)
        _ax.fill_between(
            energy,
            (_flux - _error) * _scale,
            (_flux + _error) * _scale,
            facecolor=_color,
            alpha=0.2,
        )
    _ax.loglog()
    _ax.set_xlabel("Energy [GeV]")
    _ax.set_ylabel(f"Flux × (E/GeV)^{exponent.value:g} [1/(GeV m² s sr)]")
    _ax.set_xlim(energy[0], energy[-1])
    autoscale(_ax, total_flux * _scale, (total_flux + total_error) * _scale)
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Relative uncertainties

    Sigma divided by central flux. Solar modulation widens the low-energy band; sparser data widens the high-energy band. Dashed grey lines mark the 1% and 10% levels.
    """)
    return


@app.cell
def _(
    autoscale,
    energy,
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
    _fig, _ax = plt.subplots()
    with np.errstate(divide="ignore", invalid="ignore"):
        _error_components = [
            (proton_error / proton_flux, "r", "H*"),
            (helium_error / helium_flux, "orange", "He*"),
            (oxygen_error / oxygen_flux, "g", "O*"),
            (iron_error / iron_flux, "b", "Fe*"),
            (total_error / total_flux, "k", "Total"),
        ]
    for _rel_error, _color, _label in _error_components:
        _ax.plot(energy, _rel_error, "-", color=_color, label=_label, lw=2)
    _ax.loglog()
    _ax.set_xlabel("Energy [GeV]")
    _ax.set_ylabel("Relative Uncertainty (σ/flux)")
    _ax.set_xlim(energy[0], energy[-1])
    autoscale(_ax, *[_e for _e, _c, _l in _error_components], pad=1.3)
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _ax.axhline(y=0.01, color="gray", linestyle="--", alpha=0.5)
    _ax.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Export to text files
    """)
    return


@app.cell
def _(
    energy,
    helium_error,
    helium_flux,
    iron_error,
    iron_flux,
    np,
    oxygen_error,
    oxygen_flux,
    proton_error,
    proton_flux,
    total_error,
    total_flux,
):
    flux_file_name = "gsf_particle_flux.dat"
    error_file_name = "gsf_particle_flux_error.dat"

    precision = 3  # Number of decimal places for flux

    np.savetxt(
        flux_file_name,
        np.transpose(
            (energy, proton_flux, helium_flux, oxygen_flux, iron_flux, total_flux)
        ),
        fmt=f"%.{precision}e",
        header="energy, proton, helium, oxygen group, iron group, total\n"
        "[units: energy in GeV, flux in 1/(GeV m2 s sr)]",
    )

    np.savetxt(
        error_file_name,
        np.transpose(
            (energy, proton_error, helium_error, oxygen_error, iron_error, total_error)
        ),
        fmt=f"%.{precision}e",
        header="energy, proton, helium, oxygen group, iron group, total\n"
        "[units: energy in GeV, flux error in 1/(GeV m2 s sr)]",
    )

    print(f"Saved flux and error data ({len(energy)} points each)")
    print(f"Flux file: {flux_file_name}")
    print(f"Error file: {error_file_name}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pseudo-experiments

    `sample()` draws whole flux realizations from the parameter covariance. The flux is linear in the spline amplitudes, so each draw is an exact model realization, correlated across energy and across groups. For a nonlinear analysis, run it on each realization and take the spread of the results.

    One shared amplitude draw underlies every target: seed the generator identically and the per-group draws sum to the all-particle draw.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    n_draws = mo.ui.slider(
        start=10,
        stop=200,
        step=10,
        value=60,
        label="pseudo-experiments",
        show_value=True,
    )
    mo.hstack([n_draws], justify="start")
    return (n_draws,)


@app.cell
def _(energy, gsf, n_draws, np):
    # Fixed seed: the same draws on every re-run, and the same draws for
    # every target — that is what makes the group samples add up to the
    # all-particle one.
    total_draws = gsf.sample(
        energy, None, n_samples=n_draws.value, rng=np.random.default_rng(20260809)
    )
    return (total_draws,)


@app.cell
def _(
    autoscale,
    energy,
    exponent,
    np,
    plt,
    show,
    total_draws,
    total_error,
    total_flux,
):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _scale = energy**exponent.value

    for _draw in total_draws:
        _ax1.plot(energy, _draw * _scale, "-", color="C0", lw=0.5, alpha=0.35)
    _ax1.plot(energy, total_flux * _scale, "k-", lw=2, label="central value")
    _ax1.fill_between(
        energy,
        (total_flux - total_error) * _scale,
        (total_flux + total_error) * _scale,
        facecolor="k",
        alpha=0.15,
        label="1-sigma band",
    )
    _ax1.loglog()
    _ax1.set_xlabel("Energy [GeV]")
    _ax1.set_ylabel(f"Flux × (E/GeV)^{exponent.value:g} [1/(GeV m² s sr)]")
    _ax1.set_xlim(energy[0], energy[-1])
    autoscale(_ax1, total_flux * _scale, (total_flux + total_error) * _scale)
    _ax1.set_title(f"{len(total_draws)} all-particle realizations")
    _ax1.legend()
    _ax1.grid(True, alpha=0.3)

    # The draws carry the same covariance the analytic error comes from, so
    # their scatter reproduces it — up to the 1/sqrt(2N) noise on an
    # N-sample standard deviation.
    _ratio = total_draws / total_flux
    for _r in _ratio:
        _ax2.semilogx(energy, _r, "-", color="C0", lw=0.5, alpha=0.35)
    _rel = total_error / total_flux
    _ax2.semilogx(energy, 1 + _rel, "k--", lw=1.5, label="analytic ±1σ")
    _ax2.semilogx(energy, 1 - _rel, "k--", lw=1.5)
    _ax2.semilogx(
        energy, 1 + _ratio.std(axis=0), "-", color="crimson", lw=1.5, label="sample ±1σ"
    )
    _ax2.semilogx(energy, 1 - _ratio.std(axis=0), "-", color="crimson", lw=1.5)
    _ax2.set_xlabel("Energy [GeV]")
    _ax2.set_ylabel("Realization / central value")
    _ax2.set_xlim(energy[0], energy[-1])
    _ax2.set_ylim(1 - 2.2 * np.max(_rel), 1 + 2.2 * np.max(_rel))
    _ax2.set_title("Scatter vs. propagated uncertainty")
    _ax2.legend()
    _ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    show(_fig)
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
