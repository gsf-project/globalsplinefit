import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cosmic Ray Flux by Mass Group

    Compute and plot differential fluxes for the four GSF mass groups (proton, helium, oxygen, iron) as a function of **total energy per nucleus**, with 1-sigma error bands propagated from the parameter covariance.
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

    from globalsplinefit import GSFEnergy

    plt.rcParams["figure.figsize"] = (10, 6)

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

    return GSFEnergy, np, plt


@app.cell
def _(GSFEnergy, np):
    energy = 10 ** np.linspace(0, 11, 1000)  # 1 GeV to 100 TeV
    energy_exponent = 2.6
    gsf = GSFEnergy(version="2025")
    return energy, energy_exponent, gsf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fluxes

    Four mass groups led by H, He, O\*, Fe\* (asterisks mark groups that include neighboring elements scaled from the leader). `flux()` returns the per-group differential flux in 1/(GeV m² s sr); `total_flux()` returns the all-particle sum.
    """)
    return


@app.cell
def _(energy, gsf):
    # Calculate fluxes
    proton_flux = gsf.flux(energy, "p")
    helium_flux = gsf.flux(energy, "He")
    oxygen_flux = gsf.flux(energy, "O")
    iron_flux = gsf.flux(energy, "Fe")
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
    proton_error = gsf.error(energy, "p")
    helium_error = gsf.error(energy, "He")
    oxygen_error = gsf.error(energy, "O")
    iron_error = gsf.error(energy, "Fe")
    total_error = gsf.total_error(energy)
    return helium_error, iron_error, oxygen_error, proton_error, total_error


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Flux spectra

    Differential flux scaled by E^2.6 to flatten the steep power law. Shaded bands are 1-sigma.
    """)
    return


@app.cell
def _(
    energy,
    energy_exponent,
    helium_error,
    helium_flux,
    iron_error,
    iron_flux,
    oxygen_error,
    oxygen_flux,
    plt,
    proton_error,
    proton_flux,
    total_error,
    total_flux,
):
    # Plot flux with error bands
    plt.figure()
    components = [
        (proton_flux, proton_error, "r", "Proton", 1),
        (helium_flux, helium_error, "orange", "Helium", 1),
        (oxygen_flux, oxygen_error, "g", "Oxygen*", 1),
        (iron_flux, iron_error, "b", "Iron*", 1),
        (total_flux, total_error, "k", "Total", 2),
    ]
    for flux, error, _color, _label, lw in components:
        scaled_flux = flux * energy**energy_exponent
        plt.plot(energy, scaled_flux, "-", color=_color, lw=lw, label=_label)
        plt.fill_between(
            energy,
            (flux - error) * energy**energy_exponent,
            (flux + error) * energy**energy_exponent,
            facecolor=_color,
            alpha=0.2,
        )
    plt.loglog()
    plt.xlabel("Energy [GeV]")
    plt.ylabel(f"Flux × (E/GeV)^{energy_exponent} [1/(GeV m² s sr)]")
    plt.xlim(energy[0], energy[-1])
    plt.ylim(10.0, 30000.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
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
    total_error,
    total_flux,
):
    # Plot relative uncertainties
    plt.figure()
    with np.errstate(divide="ignore", invalid="ignore"):
        error_components = [
            (proton_error / proton_flux, "r", "Proton"),
            (helium_error / helium_flux, "orange", "Helium"),
            (oxygen_error / oxygen_flux, "g", "Oxygen*"),
            (iron_error / iron_flux, "b", "Iron*"),
            (total_error / total_flux, "k", "Total"),
        ]
    for rel_error, _color, _label in error_components:
        plt.plot(energy, rel_error, "-", color=_color, label=_label, lw=2)
    plt.loglog()
    plt.xlabel("Energy [GeV]")
    plt.ylabel("Relative Uncertainty (σ/flux)")
    plt.xlim(energy[0], energy[-1])
    plt.ylim(0.003, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.01, color="gray", linestyle="--", alpha=0.5)
    plt.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5)
    # Reference lines
    plt.show()
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
    # Export data
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


if __name__ == "__main__":
    app.run()
