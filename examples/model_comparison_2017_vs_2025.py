import marimo

__generated_with = "0.23.15"
app = marimo.App(
    width="full",
    app_title="GSF — Model Comparison",
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
    # Model Comparison: GSF 2017 vs 2025

    The 2025 update extends the input dataset and refits the splines, changing both central fluxes and uncertainties. We compare the two versions for `GSFEnergy` and `GSFEnergyPerNucleon`, scaled by $E^{2.7}$.
    """)
    return


@app.cell
async def _():
    import sys

    NOTEBOOK = (
        "model_comparison_2017_vs_2025"  # used by the download buttons at the end
    )
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
                        _up + "wheels/globalsplinefit-2.0.1-py3-none-any.whl",
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

    from globalsplinefit import MODEL_VERSIONS, GSFEnergy, GSFEnergyPerNucleon

    plt.rcParams["figure.figsize"] = (8, 5)
    plt.rcParams["font.size"] = 11
    plt.rcParams["lines.linewidth"] = 1.2
    plt.rcParams["grid.alpha"] = 0.3

    def show(fig=None):
        """Return a figure as the cell output.

        marimo renders a cell's last expression; `plt.show()` produces no
        output at all in app view, so every plot cell ends with `show(fig)`.
        """
        fig = plt.gcf() if fig is None else fig
        return fig

    def autoscale(ax, *series, pad=1.6, decades=None):
        """Fit the y-axis to the positive, finite values in `series`.

        With `decades`, the lower limit is that many decades below the top
        instead of the smallest value: every group here ends in a cutoff
        tail that falls another eight decades, and letting those set the
        bottom of the axis flattens the entire comparison into a strip.
        """
        _v = np.concatenate([np.asarray(s, float).ravel() for s in series])
        _v = _v[np.isfinite(_v) & (_v > 0)]
        if _v.size:
            _top = _v.max() * pad
            ax.set_ylim(_top / 10**decades if decades else _v.min() / pad, _top)

    def relative(flux, error, cap=3.0):
        """sigma/flux up to `cap`, blank above it.

        Past the end of a group's fitted range the flux collapses (and
        eventually underflows to exactly zero) while its error stays finite,
        so the ratio runs away.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            _r = np.where(flux > 0, error / flux, np.inf)
        return np.where(np.isfinite(_r) & (_r <= cap), _r, np.nan)

    return (
        GSFEnergy,
        GSFEnergyPerNucleon,
        MODEL_VERSIONS,
        NOTEBOOK,
        SITE,
        autoscale,
        np,
        plt,
        relative,
        show,
    )


@app.cell(hide_code=True)
def _(mo):
    exponent = mo.ui.slider(
        start=0.0,
        stop=3.0,
        step=0.1,
        value=2.7,
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
    mo.hstack([exponent, plot_size], justify="start", gap=2)
    return exponent, plot_size


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


@app.cell
def _(GSFEnergy, GSFEnergyPerNucleon, np):
    # 200 points is plenty on a log grid, and the browser runtime pays for
    # every one of them twice over (two versions x flux + covariance error)
    energy = np.logspace(np.log10(0.5), np.log10(100000000000.0), 200)
    energy_per_nucleon = energy
    compared_versions = ["2017", "2025"]
    models_e = {_v: GSFEnergy(version=_v) for _v in compared_versions}
    models_n = {_v: GSFEnergyPerNucleon(version=_v) for _v in compared_versions}
    groups = ["p", "He", "O*", "Fe*"]
    group_labels = ["H*", "He*", "O*", "Fe*"]
    group_colors = ["red", "orange", "green", "blue"]
    line_styles = {"2017": "--", "2025": "-"}

    def collect(models, grid):
        _flux = {_v: {g: m.flux(grid, g) for g in groups} for _v, m in models.items()}
        err = {_v: {g: m.error(grid, g) for g in groups} for _v, m in models.items()}
        for _v, m in models.items():
            _flux[_v]["Total"] = m.total_flux(grid)
            err[_v]["Total"] = m.total_error(grid)
        return (_flux, err)

    energy_fluxes, energy_errors = collect(models_e, energy)
    nucleon_fluxes, nucleon_errors = collect(models_n, energy_per_nucleon)
    return (
        energy,
        energy_errors,
        energy_fluxes,
        energy_per_nucleon,
        group_colors,
        group_labels,
        groups,
        line_styles,
        nucleon_errors,
        nucleon_fluxes,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Total-energy flux comparison

    Solid: 2025. Dashed: 2017.

    1. Per-group + total flux.
    2. Relative uncertainty $\sigma/\Phi$.
    3. Flux with 1-sigma error bands.
    """)
    return


@app.cell
def _(
    autoscale,
    energy,
    energy_exponent,
    energy_fluxes,
    group_colors,
    group_labels,
    groups,
    line_styles,
    plt,
    show,
):
    _fig, _ax = plt.subplots()
    for _group, _label, _color in zip(groups, group_labels, group_colors, strict=False):
        for _v in ("2017", "2025"):
            _flux = energy_fluxes[_v][_group]
            _ax.loglog(
                energy,
                _flux * energy**energy_exponent,
                ls=line_styles[_v],
                color=_color,
                alpha=0.8,
                label=f"{_label} ({_v})" if _v == "2025" else None,
            )
    for _v in ("2017", "2025"):
        _flux = energy_fluxes[_v]["Total"]
        _ax.loglog(
            energy,
            _flux * energy**energy_exponent,
            ls=line_styles[_v],
            color="black",
            lw=2,
            alpha=0.9,
            label=f"Total ({_v})",
        )
    _ax.set_xlabel("Energy [GeV]")
    _ax.set_ylabel(f"Flux × $E^{{{energy_exponent}}}$")
    _ax.grid(True, alpha=0.3)
    _ax.legend(loc="lower center", ncol=2)
    _ax.set_xlim(energy[0], energy[-1])
    autoscale(_ax, *[_l.get_ydata() for _l in _ax.get_lines()], decades=3)
    plt.tight_layout()
    show()
    return


@app.cell
def _(
    energy,
    energy_errors,
    energy_fluxes,
    group_colors,
    group_labels,
    groups,
    line_styles,
    plt,
    relative,
    show,
):
    _fig, _ax = plt.subplots()
    for _group, _label, _color in zip(groups, group_labels, group_colors, strict=False):
        for _v in ("2017", "2025"):
            _rel = relative(energy_fluxes[_v][_group], energy_errors[_v][_group])
            _ax.loglog(
                energy,
                _rel,
                ls=line_styles[_v],
                color=_color,
                alpha=0.8,
                label=f"{_label} ({_v})" if _v == "2025" else None,
            )
    for _v in ("2017", "2025"):
        _rel = relative(energy_fluxes[_v]["Total"], energy_errors[_v]["Total"])
        _ax.loglog(
            energy,
            _rel,
            ls=line_styles[_v],
            color="black",
            lw=2,
            alpha=0.9,
            label=f"Total ({_v})",
        )
    _ax.set_xlabel("Energy [GeV]")
    _ax.set_ylabel("Relative error σ/Φ")
    _ax.grid(True, alpha=0.3)
    _ax.legend(loc="lower right", ncol=2)
    _ax.set_xlim(energy[0], energy[-1])
    _ax.set_ylim(0.003, 3)
    plt.tight_layout()
    show()
    return


@app.cell
def _(
    autoscale,
    energy,
    energy_errors,
    energy_exponent,
    energy_fluxes,
    group_colors,
    group_labels,
    groups,
    line_styles,
    plt,
    show,
):
    _fig, _ax = plt.subplots()

    def plot_with_band(ax, x, flux, err, color, ls, lw=1, label=None):
        scaled = flux * x**energy_exponent
        scaled_err = err * x**energy_exponent
        ax.loglog(x, scaled, color=color, ls=ls, lw=lw, alpha=0.8, label=label)
        ax.fill_between(
            x, scaled - scaled_err, scaled + scaled_err, color=color, alpha=0.2
        )

    for _group, _label, _color in zip(groups, group_labels, group_colors, strict=False):
        plot_with_band(
            _ax,
            energy,
            energy_fluxes["2017"][_group],
            energy_errors["2017"][_group],
            _color,
            "--",
        )
        plot_with_band(
            _ax,
            energy,
            energy_fluxes["2025"][_group],
            energy_errors["2025"][_group],
            _color,
            "-",
            label=_label,
        )
    for _v, _ls in line_styles.items():
        plot_with_band(
            _ax,
            energy,
            energy_fluxes[_v]["Total"],
            energy_errors[_v]["Total"],
            "black",
            _ls,
            lw=2,
            label=f"Total {_v}",
        )
    _ax.set_xlabel("Energy [GeV]")
    _ax.set_ylabel(f"Flux × $E^{{{energy_exponent}}}$")
    _ax.grid(True, alpha=0.3)
    _ax.legend(loc="lower center", ncol=2)
    _ax.set_xlim(energy[0], energy[-1])
    autoscale(_ax, *[_l.get_ydata() for _l in _ax.get_lines()], decades=3)
    plt.tight_layout()
    show()
    return (plot_with_band,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Energy-per-nucleon flux comparison

    Same three plots in the `GSFEnergyPerNucleon` view.
    """)
    return


@app.cell
def _(
    autoscale,
    energy_exponent,
    energy_per_nucleon,
    group_colors,
    group_labels,
    groups,
    line_styles,
    nucleon_fluxes,
    plt,
    show,
):
    _fig, _ax = plt.subplots()
    for _group, _label, _color in zip(groups, group_labels, group_colors, strict=False):
        for _v in ("2017", "2025"):
            _flux = nucleon_fluxes[_v][_group]
            _ax.loglog(
                energy_per_nucleon,
                _flux * energy_per_nucleon**energy_exponent,
                ls=line_styles[_v],
                color=_color,
                alpha=0.8,
                label=f"{_label} ({_v})" if _v == "2025" else None,
            )
    for _v in ("2017", "2025"):
        _flux = nucleon_fluxes[_v]["Total"]
        _ax.loglog(
            energy_per_nucleon,
            _flux * energy_per_nucleon**energy_exponent,
            ls=line_styles[_v],
            color="black",
            lw=2,
            alpha=0.9,
            label=f"Total ({_v})",
        )
    _ax.set_xlabel("Energy per nucleon [GeV/nucleon]")
    _ax.set_ylabel(f"Nucleon flux × $E^{{{energy_exponent}}}$")
    _ax.grid(True, alpha=0.3)
    _ax.legend(loc="upper right", ncol=2)
    _ax.set_xlim(energy_per_nucleon[0], energy_per_nucleon[-1])
    autoscale(_ax, *[_l.get_ydata() for _l in _ax.get_lines()], decades=3)
    plt.tight_layout()
    show()
    return


@app.cell
def _(
    energy_per_nucleon,
    group_colors,
    group_labels,
    groups,
    line_styles,
    nucleon_errors,
    nucleon_fluxes,
    plt,
    relative,
    show,
):
    _fig, _ax = plt.subplots()
    for _group, _label, _color in zip(groups, group_labels, group_colors, strict=False):
        for _v in ("2017", "2025"):
            _rel = relative(nucleon_fluxes[_v][_group], nucleon_errors[_v][_group])
            _ax.loglog(
                energy_per_nucleon,
                _rel,
                ls=line_styles[_v],
                color=_color,
                alpha=0.8,
                label=f"{_label} ({_v})" if _v == "2025" else None,
            )
    for _v in ("2017", "2025"):
        _rel = relative(nucleon_fluxes[_v]["Total"], nucleon_errors[_v]["Total"])
        _ax.loglog(
            energy_per_nucleon,
            _rel,
            ls=line_styles[_v],
            color="black",
            lw=2,
            alpha=0.9,
            label=f"Total ({_v})",
        )
    _ax.set_xlabel("Energy per nucleon [GeV/nucleon]")
    _ax.set_ylabel("Relative error σ/Φ")
    _ax.grid(True, alpha=0.3)
    _ax.legend(loc="lower right", ncol=2)
    _ax.set_xlim(energy_per_nucleon[0], energy_per_nucleon[-1])
    _ax.set_ylim(0.003, 3)
    plt.tight_layout()
    show()
    return


@app.cell
def _(
    autoscale,
    energy_exponent,
    energy_per_nucleon,
    group_colors,
    group_labels,
    groups,
    line_styles,
    nucleon_errors,
    nucleon_fluxes,
    plot_with_band,
    plt,
    show,
):
    _fig, _ax = plt.subplots()
    for _group, _label, _color in zip(groups, group_labels, group_colors, strict=False):
        plot_with_band(
            _ax,
            energy_per_nucleon,
            nucleon_fluxes["2017"][_group],
            nucleon_errors["2017"][_group],
            _color,
            "--",
        )
        plot_with_band(
            _ax,
            energy_per_nucleon,
            nucleon_fluxes["2025"][_group],
            nucleon_errors["2025"][_group],
            _color,
            "-",
            label=_label,
        )
    for _v, _ls in line_styles.items():
        plot_with_band(
            _ax,
            energy_per_nucleon,
            nucleon_fluxes[_v]["Total"],
            nucleon_errors[_v]["Total"],
            "black",
            _ls,
            lw=2,
            label=f"Total {_v}",
        )
    _ax.set_xlabel("Energy per nucleon [GeV/nucleon]")
    _ax.set_ylabel(f"Nucleon flux × $E^{{{energy_exponent}}}$")
    _ax.grid(True, alpha=0.3)
    _ax.legend(loc="lower center", ncol=2)
    _ax.set_xlim(energy_per_nucleon[0], energy_per_nucleon[-1])
    autoscale(_ax, *[_l.get_ydata() for _l in _ax.get_lines()], decades=3)
    plt.tight_layout()
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ratio of 2017 to 2025

    Top: total nucleon flux ratio $\Phi_{2017}/\Phi_{2025}$ with the relative uncertainty of each version shaded. Bottom: ratio of absolute errors.
    """)
    return


@app.cell
def _(
    energy_per_nucleon,
    np,
    nucleon_errors,
    nucleon_fluxes,
    plt,
    show,
):
    _fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # NaN out the underflowed tail (both totals reach zero past the fitted
    # range) to keep the ratios finite; the axis stops well below it anyway
    _pos = (nucleon_fluxes["2017"]["Total"] > 0) & (nucleon_fluxes["2025"]["Total"] > 0)
    flux_2017 = np.where(_pos, nucleon_fluxes["2017"]["Total"], np.nan)
    flux_2025 = np.where(_pos, nucleon_fluxes["2025"]["Total"], np.nan)
    err_2017 = np.where(_pos, nucleon_errors["2017"]["Total"], np.nan)
    err_2025 = np.where(_pos, nucleon_errors["2025"]["Total"], np.nan)
    flux_ratio = flux_2017 / flux_2025
    err_ratio = err_2017 / err_2025
    ax1.semilogx(energy_per_nucleon, flux_ratio, "b-", label="GSF2017")
    ax1.fill_between(
        energy_per_nucleon,
        flux_ratio - err_2017 / flux_2017,
        flux_ratio + err_2017 / flux_2017,
        color="b",
        alpha=0.2,
    )
    ax1.fill_between(
        energy_per_nucleon,
        1 - err_2025 / flux_2025,
        1 + err_2025 / flux_2025,
        color="gray",
        alpha=0.2,
    )
    ax1.axhline(1, color="k", ls="--", alpha=0.5, label="GSF2025")
    ax1.set_ylabel("Flux ratio (2017/2025)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(1.0, 3000000000.0)
    ax1.set_ylim(0.75, 1.25)
    ax2.semilogx(energy_per_nucleon, err_ratio, "k-")
    ax2.axhline(1, color="k", ls="--", alpha=0.5)
    ax2.set_xlabel("Energy per nucleon [GeV/nucleon]")
    ax2.set_ylabel("Error 2017 / Error 2025")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.0, 5.5)
    plt.tight_layout()
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
