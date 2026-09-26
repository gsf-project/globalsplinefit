import marimo

__generated_with = "0.23.15"
app = marimo.App(
    width="full",
    app_title="GSF — Solar Modulation",
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
    # Solar Modulation Effects on Cosmic Ray Flux

    Galactic cosmic rays are decelerated by the solar wind before reaching Earth. GSF uses the force-field approximation, parameterized by a single potential $\phi$ (GV): higher $\phi$ → stronger modulation, lower observed flux. Typical values: $\phi \approx 0.4$ GV at solar minimum, $\phi \approx 1.0$ GV at solar maximum. The unmodulated Local Interstellar Spectrum (LIS) corresponds to $\phi = 0$.
    """)
    return


@app.cell
async def _():
    import sys

    NOTEBOOK = "solar_modulation"  # used by the download buttons at the end
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
    from matplotlib.colors import Normalize

    from globalsplinefit import (
        MODEL_VERSIONS,
        GSFEnergy,
        GSFEnergyPerNucleon,
        GSFRigidity,
    )

    plt.rcParams["figure.figsize"] = (10, 6)

    def show(fig=None):
        """Return a figure as the cell output.

        marimo renders a cell's last expression; `plt.show()` produces no
        output at all in app view, so every plot cell ends with `show(fig)`.
        """
        fig = plt.gcf() if fig is None else fig
        return fig

    def autoscale(ax, *series, pad=1.6, lift=1.0):
        """Fit the y-axis to the positive, finite values in `series`.

        `lift` raises the lower limit by that factor — the decades a steeply
        falling group spends near the bottom carry no information and only
        compress the part of the plot that does.
        """
        _v = np.concatenate([np.asarray(s, float).ravel() for s in series])
        _v = _v[np.isfinite(_v) & (_v > 0)]
        if _v.size:
            ax.set_ylim(_v.min() / pad * lift, _v.max() * pad)

    return (
        GSFEnergy,
        GSFEnergyPerNucleon,
        GSFRigidity,
        MODEL_VERSIONS,
        NOTEBOOK,
        Normalize,
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
def _(exponent):
    energy_exponent = exponent.value  # display scaling, driven by the slider
    return (energy_exponent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Three model types, three solar conditions

    Solar conditions: LIS, solar minimum (2009), solar maximum (1991).
    """)
    return


@app.cell
def _(
    GSFEnergy,
    GSFEnergyPerNucleon,
    GSFRigidity,
    np,
    version,
):
    grid = np.logspace(np.log10(0.5), np.log10(200), 200)  # GeV or GV
    energy = rigidity = energy_per_nucleon = grid

    gsf_energy = GSFEnergy(version=version.value)
    gsf_rigidity = GSFRigidity(version=version.value)
    gsf_nucleon = GSFEnergyPerNucleon(version=version.value)

    # time_interval ends are EXCLUSIVE, so a calendar year runs to
    # January of the next one
    time_periods = {
        "LIS": "LIS",
        "Solar Minimum (2009)": (200901, 201001),
        "Solar Maximum (1991)": (199101, 199201),
    }
    groups = ["p", "He", "O*", "Fe*"]
    group_colors = ["red", "orange", "green", "blue"]
    line_styles = {
        "LIS": "-",
        "Solar Minimum (2009)": "--",
        "Solar Maximum (1991)": ":",
    }
    return (
        energy,
        energy_per_nucleon,
        group_colors,
        groups,
        gsf_energy,
        gsf_nucleon,
        gsf_rigidity,
        line_styles,
        rigidity,
        time_periods,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Flux comparisons

    Compute and plot per-group + total flux for each model and time period. Fluxes are scaled by $E^{2.6}$ to flatten the spectrum.
    """)
    return


@app.cell
def _(
    energy,
    energy_per_nucleon,
    groups,
    gsf_energy,
    gsf_nucleon,
    gsf_rigidity,
    rigidity,
    time_periods,
):
    # Collect fluxes for each (model, period, group) combination
    models = {
        "GSFEnergy": (gsf_energy, energy),
        "GSFRigidity": (gsf_rigidity, rigidity),
        "GSFEnergyPerNucleon": (gsf_nucleon, energy_per_nucleon),
    }
    fluxes = {}
    for _mname, (model, _grid_) in models.items():
        fluxes[_mname] = {}
        for _period, ti in time_periods.items():
            fluxes[_mname][_period] = {
                g: model.flux(_grid_, g, time_interval=ti) for g in groups
            }
            fluxes[_mname][_period]["Total"] = model.total_flux(
                _grid_, time_interval=ti
            )  # fluxes[model_name][period][group_or_"Total"]
    return (fluxes,)


@app.cell
def _(energy_exponent, fluxes, group_colors, groups, line_styles):
    def plot_groups(ax, model_name, grid_, xlabel, scale_label):
        period_handles = []
        for i, (group, _color) in enumerate(zip(groups, group_colors, strict=False)):
            for _period, style in line_styles.items():
                _flux = fluxes[model_name][_period][group]
                (line,) = ax.loglog(
                    grid_,
                    _flux * grid_**energy_exponent,
                    ls=style,
                    color=_color,
                    alpha=0.8,
                )
                if i == 0:
                    period_handles.append((line, _period))
        group_handles = [
            ax.plot([], [], color=c, lw=3, label=lbl)[0]
            for c, lbl in zip(group_colors, ["H*", "He*", "O*", "Fe*"], strict=False)
        ]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(scale_label)
        ax.set_title(model_name)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(grid_[0], grid_[-1])
        period_legend = ax.legend(
            [h for h, _ in period_handles],
            [lbl for _, lbl in period_handles],
            loc="lower right",
        )
        ax.legend(handles=group_handles, loc="upper right", ncol=4)
        ax.add_artist(period_legend)

    return (plot_groups,)


@app.cell
def _(
    autoscale,
    energy,
    energy_exponent,
    energy_per_nucleon,
    plot_groups,
    plt,
    rigidity,
    show,
):
    _fig, axes = plt.subplots(3, 1, figsize=(9, 12))
    plot_groups(
        axes[0],
        "GSFEnergy",
        energy,
        "Energy [GeV]",
        f"Flux × $E^{{{energy_exponent}}}$",
    )
    plot_groups(
        axes[1],
        "GSFRigidity",
        rigidity,
        "Rigidity [GV]",
        f"Flux × $R^{{{energy_exponent}}}$",
    )
    plot_groups(
        axes[2],
        "GSFEnergyPerNucleon",
        energy_per_nucleon,
        "Energy/nucleon [GeV/nucleon]",
        f"Nucleon flux × $E^{{{energy_exponent}}}$",
    )
    for _ax in axes:
        autoscale(_ax, *[_l.get_ydata() for _l in _ax.get_lines()], lift=10)
    plt.tight_layout()
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Total flux across model types

    Overlay the total flux from all three models on a single axis. Note the x-axis units differ by model (GeV vs GV vs GeV/nucleon).
    """)
    return


@app.cell
def _(
    autoscale,
    energy,
    energy_exponent,
    energy_per_nucleon,
    fluxes,
    line_styles,
    plt,
    rigidity,
    show,
):
    _fig, _ax = plt.subplots(figsize=(10, 6))
    plot_models = [
        ("GSFEnergy", energy, "black"),
        ("GSFRigidity", rigidity, "purple"),
        ("GSFEnergyPerNucleon", energy_per_nucleon, "brown"),
    ]
    for _mname, _grid_, _color in plot_models:
        for _period, style in line_styles.items():
            _flux = fluxes[_mname][_period]["Total"]
            label = _mname if _period == "LIS" else None
            _ax.loglog(
                _grid_,
                _flux * _grid_**energy_exponent,
                ls=style,
                color=_color,
                label=label,
                alpha=0.8,
            )
    _ax.set_xlabel("Energy / Rigidity / Energy per nucleon")
    _ax.set_ylabel(f"Total flux × $X^{{{energy_exponent}}}$ (units vary by model)")
    _ax.grid(True, alpha=0.3)
    _ax.legend(loc="lower right")
    autoscale(_ax, *[_l.get_ydata() for _l in _ax.get_lines()], lift=10)
    plt.tight_layout()
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Solar cycle evolution

    Sample 17 monthly time points across 2013–2024 and compute the total nucleon flux at each. The color scale maps to $\phi$ (warmer = stronger modulation). The second figure shows $\phi$ and the flux-suppression ratio (modulated / LIS) at 1.5, 10, and 100 GeV/nucleon.
    """)
    return


@app.cell
def _(gsf_energy, np):
    # 17 points spanning 2013-01 to 2024-12. The end of a time_interval is
    # exclusive, so a December point ends in January of the next year.
    time_points = []
    phi_values = []
    dates = []
    for _idx in np.linspace(0, (2024 - 2013 + 1) * 12 - 1, 17):
        month_idx = int(round(_idx))
        year = 2013 + month_idx // 12
        month = (month_idx % 12) + 1
        if year in gsf_energy.phi:
            end_year, end_month = (year + 1, 1) if month == 12 else (year, month + 1)
            time_points.append((year * 100 + month, end_year * 100 + end_month))
            phi_values.append(float(gsf_energy.phi[year][month - 1]))
            dates.append(f"{year}-{month:02d}")

    print(
        f"{len(time_points)} time points, φ range: {min(phi_values):.3f}–{max(phi_values):.3f} GV"
    )
    return dates, phi_values, time_points


@app.cell
def _(energy_per_nucleon, gsf_nucleon, np, time_points):
    time_series_fluxes = np.array(
        [
            gsf_nucleon.total_flux(energy_per_nucleon, time_interval=ti)
            for ti in time_points
        ]
    )
    flux_lis = gsf_nucleon.total_flux(energy_per_nucleon, time_interval="LIS")
    return flux_lis, time_series_fluxes


@app.cell
def _(
    Normalize,
    autoscale,
    dates,
    energy_exponent,
    energy_per_nucleon,
    flux_lis,
    phi_values,
    plt,
    show,
    time_series_fluxes,
):
    cmap = plt.get_cmap("coolwarm")
    norm = Normalize(vmin=min(phi_values), vmax=max(phi_values))
    _fig, _ax = plt.subplots(figsize=(8, 5))
    for _flux, phi, _date in zip(time_series_fluxes, phi_values, dates, strict=False):
        _ax.loglog(
            energy_per_nucleon,
            _flux * energy_per_nucleon**energy_exponent,
            color=cmap(norm(phi)),
            alpha=0.7,
            lw=1.5,
        )
    _ax.loglog(
        energy_per_nucleon,
        flux_lis * energy_per_nucleon**energy_exponent,
        "k-",
        lw=3,
        label="LIS (φ=0)",
    )
    _ax.set_xlabel("Energy [GeV/nucleon]")
    _ax.set_ylabel(f"Nucleon flux × $E^{{{energy_exponent}}}$")
    _ax.set_title("Nucleon flux through the solar cycle")
    _ax.grid(True, alpha=0.3)
    _ax.set_xlim(energy_per_nucleon[0], energy_per_nucleon[-1])
    autoscale(_ax, flux_lis * energy_per_nucleon**energy_exponent)
    _ax.legend(loc="lower right")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=_ax, label="φ [GV]")
    plt.tight_layout()
    show()
    return


@app.cell
def _(
    dates,
    energy_per_nucleon,
    flux_lis,
    np,
    phi_values,
    plt,
    show,
    time_series_fluxes,
):
    _fig, _ax = plt.subplots(figsize=(9, 5))
    _ax2 = _ax.twinx()
    years = [int(d[:4]) + (int(d[5:7]) - 1) / 12 for d in dates]
    _ax.plot(years, phi_values, "ro-", lw=2, ms=6, label="φ")
    _ax.set_xlabel("Year")
    _ax.set_ylabel("φ [GV]", color="red")
    _ax.tick_params(axis="y", labelcolor="red")
    _ax.grid(True, alpha=0.3)
    for energy_val, _color in zip(
        [1.5, 10.0, 100.0], ["blue", "green", "orange"], strict=False
    ):
        idx = np.argmin(np.abs(energy_per_nucleon - energy_val))
        ratio = time_series_fluxes[:, idx] / flux_lis[idx]
        _ax2.plot(
            years,
            ratio,
            "o-",
            color=_color,
            lw=2,
            ms=4,
            label=f"{energy_per_nucleon[idx]:.1f} GeV/nuc",
        )
    _ax2.axhline(1.0, color="gray", ls="--", alpha=0.5)
    _ax2.set_ylabel("Flux ratio (modulated / LIS)")
    _ax2.set_ylim(0, 1.2)
    _ax2.legend(loc="lower right")
    plt.tight_layout()
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Solar-cycle averaging: how many $\phi$ bins?

    `time_interval=None` returns a Solar Cycle 24 average. The monthly $\phi$ values
    of the interval are binned into `solar_cycle_average_bins` weighted
    representatives (each bin's mean $\phi$, weighted by its month count) and the
    *modulated flux* is averaged over them — the same period average the fitter
    uses. Because flux is nonlinear in $\phi$, this "mean of modulated" is not
    $\mathrm{Flux}(\langle\phi\rangle)$.

    The weighted mean $\phi$ is exact at every setting, so the first-order
    suppression is always right and only the curvature is approximated. At 1 GeV,
    1 bin is off by 5.2% from the full monthly average, 6 bins by 0.5%, and the default 12 by 0.1%; all are below 0.3% above
    10 GeV. `None` averages every month explicitly.
    """)
    return


@app.cell
def _(
    GSFEnergy,
    energy_exponent,
    np,
    plt,
    show,
):
    gsf_one = GSFEnergy(solar_cycle_average_bins=1)
    gsf_def = GSFEnergy(solar_cycle_average_bins=12)  # the default
    gsf_exact = GSFEnergy(solar_cycle_average_bins=None)
    energy_subset = np.logspace(0, 1, 500)

    flux_one = gsf_one.total_flux(energy_subset)
    flux_def = gsf_def.total_flux(energy_subset)
    flux_exact = gsf_exact.total_flux(energy_subset)
    rel_one = 100 * np.abs(flux_one - flux_exact) / flux_exact
    rel_def = 100 * np.abs(flux_def - flux_exact) / flux_exact

    _fig, (ax1, _ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    ax1.loglog(
        energy_subset,
        energy_subset**energy_exponent * flux_one,
        "r-",
        lw=2,
        label="1 bin: Flux$(\\langle\\phi\\rangle)$",
    )
    ax1.loglog(
        energy_subset,
        energy_subset**energy_exponent * flux_def,
        "-",
        color="C2",
        lw=2,
        label="12 bins (default)",
    )
    ax1.loglog(
        energy_subset,
        energy_subset**energy_exponent * flux_exact,
        "b--",
        lw=2,
        label="all months: $\\langle$Flux$(\\phi)\\rangle$",
    )
    ax1.set_ylabel("Total flux × $E^{2.6}$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    _ax2.semilogx(energy_subset, rel_one, "r-", lw=2, label="1 bin")
    _ax2.semilogx(energy_subset, rel_def, "-", color="C2", lw=2, label="12 bins")
    _ax2.set_xlabel("Energy [GeV]")
    _ax2.set_ylabel("Deviation from full average [%]")
    _ax2.grid(True, alpha=0.3)
    _ax2.legend(fontsize=8, loc="upper right")
    _ax2.set_ylim(0, rel_one.max() * 1.1)
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
