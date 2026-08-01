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
    # Solar Modulation Effects on Cosmic Ray Flux

    Galactic cosmic rays are decelerated by the solar wind before reaching Earth. GSF uses the force-field approximation, parameterized by a single potential $\phi$ (GV): higher $\phi$ → stronger modulation, lower observed flux. Typical values: $\phi \approx 0.4$ GV at solar minimum, $\phi \approx 1.0$ GV at solar maximum. The unmodulated Local Interstellar Spectrum (LIS) corresponds to $\phi = 0$.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize

    from globalsplinefit import GSFEnergy, GSFEnergyPerNucleon, GSFRigidity

    plt.rcParams["figure.figsize"] = (10, 6)
    return GSFEnergy, GSFEnergyPerNucleon, GSFRigidity, Normalize, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Three model types, three solar conditions

    We compare three GSF model classes against three solar conditions: LIS, solar minimum (2009), solar maximum (1991).
    """)
    return


@app.cell
def _(GSFEnergy, GSFEnergyPerNucleon, GSFRigidity, np):
    grid = np.logspace(np.log10(0.5), np.log10(200), 200)  # GeV or GV
    energy = rigidity = energy_per_nucleon = grid

    gsf_energy = GSFEnergy()
    gsf_rigidity = GSFRigidity()
    gsf_nucleon = GSFEnergyPerNucleon()

    time_periods = {
        "LIS": "LIS",
        "Solar Minimum (2009)": (200901, 200912),
        "Solar Maximum (1991)": (199101, 199112),
    }
    groups = ["p", "He", "O*", "Fe*"]
    group_colors = ["red", "orange", "green", "blue"]
    line_styles = {
        "LIS": "-",
        "Solar Minimum (2009)": "--",
        "Solar Maximum (1991)": ":",
    }
    energy_exponent = 2.6
    return (
        energy,
        energy_exponent,
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
            for c, lbl in zip(
                group_colors, ["Proton", "Helium", "Oxygen*", "Iron*"], strict=False
            )
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
def _(energy, energy_exponent, energy_per_nucleon, plot_groups, plt, rigidity):
    _fig, axes = plt.subplots(1, 3, figsize=(20, 6))
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
        _ax.set_ylim(1.0, 30000.0)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Total flux across model types

    Overlay the total flux from all three models on a single axis. Note the x-axis units differ by model (GeV vs GV vs GeV/nucleon), so this exposes structural differences rather than direct comparisons.
    """)
    return


@app.cell
def _(
    energy,
    energy_exponent,
    energy_per_nucleon,
    fluxes,
    line_styles,
    plt,
    rigidity,
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
    _ax.set_ylim(500.0, 30000.0)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Solar cycle evolution

    Sample 17 monthly time points across 2013–2024 and compute the total nucleon flux at each. The color scale maps to $\phi$ (warmer = stronger modulation). The second figure shows $\phi$ and the flux-suppression ratio (modulated / LIS) at 1.5, 10, and 100 GeV/nucleon.
    """)
    return


@app.cell
def _(gsf_energy):
    # Sample 17 monthly time points between 2013-01 and 2024-12
    total_months = (2024 - 2013) * 12 + 12
    step = total_months // 16
    time_points = []
    phi_values = []
    dates = []
    month_idx = 0
    for _ in range(17):
        year = 2013 + month_idx // 12
        month = (month_idx % 12) + 1
        if year in gsf_energy.phi:
            time_points.append((year * 100 + month, year * 100 + month + 1))
            phi_values.append(float(gsf_energy.phi[year][month - 1]))
            dates.append(f"{year}-{month:02d}")
        month_idx += step

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
    dates,
    energy_exponent,
    energy_per_nucleon,
    flux_lis,
    phi_values,
    plt,
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
    _ax.set_ylim(500.0, 40000.0)
    _ax.legend(loc="lower right")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=_ax, label="φ [GV]")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    dates,
    energy_per_nucleon,
    flux_lis,
    np,
    phi_values,
    plt,
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
    plt.show()
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
    1 bin (the retired "approximate" mode) is off by 5.2% from the full monthly
    average, the default 6 by 0.5%, and 12 by 0.1%; all are below 0.3% above
    10 GeV. `None` averages every month explicitly.
    """)
    return


@app.cell
def _(GSFEnergy, energy_exponent, np, plt):
    import time as _time

    gsf_one = GSFEnergy(solar_cycle_average_bins=1)
    gsf_six = GSFEnergy(solar_cycle_average_bins=6)
    gsf_exact = GSFEnergy(solar_cycle_average_bins=None)
    energy_subset = np.logspace(0, 1, 500)


    def _timed(model):
        _t0 = _time.time()
        return model.total_flux(energy_subset), _time.time() - _t0


    flux_one, t_one = _timed(gsf_one)
    flux_six, t_six = _timed(gsf_six)
    flux_exact, t_exact = _timed(gsf_exact)
    rel_one = 100 * np.abs(flux_one - flux_exact) / flux_exact
    rel_six = 100 * np.abs(flux_six - flux_exact) / flux_exact

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
        energy_subset**energy_exponent * flux_six,
        "-",
        color="C2",
        lw=2,
        label="6 bins (default)",
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
    _ax2.semilogx(energy_subset, rel_six, "-", color="C2", lw=2, label="6 bins")
    _ax2.set_xlabel("Energy [GeV]")
    _ax2.set_ylabel("Deviation from full average [%]")
    _ax2.grid(True, alpha=0.3)
    _ax2.legend(fontsize=8, loc="upper right")
    _ax2.set_ylim(0, rel_one.max() * 1.1)
    _ax2.text(
        0.05,
        0.95,
        f"1 bin:  {t_one:.3f} s, max {rel_one.max():.2f}%\n"
        f"6 bins: {t_six:.3f} s, max {rel_six.max():.2f}%\n"
        f"full:   {t_exact:.3f} s ({len(gsf_exact._phi_list(None)[0])} months)",
        transform=_ax2.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        fontfamily="monospace",
        fontsize=9,
    )
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
