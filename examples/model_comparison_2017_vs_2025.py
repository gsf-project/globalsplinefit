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
    # Model Comparison: GSF 2017 vs 2025

    The 2025 update extends the input dataset and refits the splines, changing both central fluxes and uncertainties. We compare the two versions for `GSFEnergy` and `GSFEnergyPerNucleon`, scaled by $E^{2.7}$.
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

    plt.rcParams["figure.figsize"] = (8, 5)
    plt.rcParams["font.size"] = 11
    plt.rcParams["lines.linewidth"] = 1.2
    plt.rcParams["grid.alpha"] = 0.3

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

    return GSFEnergy, GSFEnergyPerNucleon, np, plt


@app.cell
def _(GSFEnergy, GSFEnergyPerNucleon, np):
    energy = np.logspace(np.log10(0.5), np.log10(100000000000.0), 500)
    energy_per_nucleon = energy
    compared_versions = ["2017", "2025", "2026"]
    models_e = {_v: GSFEnergy(version=_v) for _v in compared_versions}
    models_n = {_v: GSFEnergyPerNucleon(version=_v) for _v in compared_versions}
    groups = ["p", "He", "O*", "Fe*"]
    group_labels = ["Proton", "Helium", "Oxygen*", "Iron*"]
    group_colors = ["red", "orange", "green", "blue"]
    line_styles = {"2025": "--", "2017": ":", "2026": "-"}
    energy_exponent = 2.7

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
        energy_exponent,
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
    energy,
    energy_exponent,
    energy_fluxes,
    group_colors,
    group_labels,
    groups,
    line_styles,
    plt,
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
    _ax.set_ylim(8.0, 90000.0)
    plt.tight_layout()
    plt.show()
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
):
    _fig, _ax = plt.subplots()
    for _group, _label, _color in zip(groups, group_labels, group_colors, strict=False):
        for _v in ("2017", "2025"):
            _rel = energy_errors[_v][_group] / energy_fluxes[_v][_group]
            _ax.loglog(
                energy,
                _rel,
                ls=line_styles[_v],
                color=_color,
                alpha=0.8,
                label=f"{_label} ({_v})" if _v == "2025" else None,
            )
    for _v in ("2017", "2025"):
        _rel = energy_errors[_v]["Total"] / energy_fluxes[_v]["Total"]
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
    _ax.set_ylim(0.001, 0.98)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    energy,
    energy_exponent,
    energy_fluxes,
    group_colors,
    group_labels,
    groups,
    line_styles,
    plt,
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
    _ax.set_ylim(8.0, 90000.0)
    plt.tight_layout()
    plt.show()
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
):
    _fig, _ax = plt.subplots()
    for _group, _label, _color in zip(groups, group_labels, group_colors, strict=False):
        for _v in ("2017", "2025"):
            _rel = energy_errors[_v][_group] / energy_fluxes[_v][_group]
            _ax.loglog(
                energy,
                _rel,
                ls=line_styles[_v],
                color=_color,
                alpha=0.8,
                label=f"{_label} ({_v})" if _v == "2025" else None,
            )
    for _v in ("2017", "2025"):
        _rel = energy_errors[_v]["Total"] / energy_fluxes[_v]["Total"]
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
    _ax.set_ylim(0.001, 0.98)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    energy,
    energy_errors,
    energy_exponent,
    energy_fluxes,
    group_colors,
    group_labels,
    groups,
    line_styles,
    plt,
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
    _ax.set_ylim(8.0, 90000.0)
    plt.tight_layout()
    plt.show()
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
    energy_exponent,
    energy_per_nucleon,
    group_colors,
    group_labels,
    groups,
    line_styles,
    nucleon_fluxes,
    plt,
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
    _ax.set_ylim(80.0, 90000.0)
    plt.tight_layout()
    plt.show()
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
):
    _fig, _ax = plt.subplots()
    for _group, _label, _color in zip(groups, group_labels, group_colors, strict=False):
        for _v in ("2017", "2025"):
            _rel = nucleon_errors[_v][_group] / nucleon_fluxes[_v][_group]
            _ax.loglog(
                energy_per_nucleon,
                _rel,
                ls=line_styles[_v],
                color=_color,
                alpha=0.8,
                label=f"{_label} ({_v})" if _v == "2025" else None,
            )
    for _v in ("2017", "2025"):
        _rel = nucleon_errors[_v]["Total"] / nucleon_fluxes[_v]["Total"]
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
    _ax.set_ylim(0.003, 1)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
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
    _ax.set_ylim(8.0, 90000.0)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ratio of 2017 to 2025

    Top: total nucleon flux ratio $\Phi_{2017}/\Phi_{2025}$ with the relative uncertainty of each version shaded. Bottom: ratio of absolute errors.
    """)
    return


@app.cell
def _(energy_per_nucleon, nucleon_errors, nucleon_fluxes, plt):
    _fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    flux_2017 = nucleon_fluxes["2017"]["Total"]
    flux_2025 = nucleon_fluxes["2025"]["Total"]
    err_2017 = nucleon_errors["2017"]["Total"]
    err_2025 = nucleon_errors["2025"]["Total"]
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
    plt.show()
    return


if __name__ == "__main__":
    app.run()
