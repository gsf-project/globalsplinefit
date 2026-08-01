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
    # Nucleon Flux for Air Shower Simulations

    Air shower codes (e.g. CORSIKA) take cosmic ray fluxes as nucleon fluxes per GeV/nucleon. `GSFEnergyPerNucleon` provides this from the fitted all-particle spectra, summed over the 28 nuclear species into proton and neutron components.
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

        await micropip.install(
            urljoin(
                str(_mo.notebook_location()) + "/",
                "../../wheels/globalsplinefit-2.0.0a1-py3-none-any.whl",
            )
        )

    import matplotlib.pyplot as plt
    import numpy as np

    from globalsplinefit import GSFEnergyPerNucleon

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

    return GSFEnergyPerNucleon, np, plt


@app.cell
def _(GSFEnergyPerNucleon, np):
    energy_per_nucleon = 10 ** np.linspace(0, 10, 200)  # GeV/nucleon
    energy_exponent = 3.0
    gsf_nucleon = GSFEnergyPerNucleon()
    return energy_exponent, energy_per_nucleon, gsf_nucleon


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Nucleon fluxes

    `flux()` returns the proton + neutron sum for a group or element; `total_flux()` sums over all 28 species. Use `p_and_n_flux()` for the components separately (shape `(2, N)`).
    """)
    return


@app.cell
def _(energy_per_nucleon, gsf_nucleon):
    proton_flux = gsf_nucleon.flux(energy_per_nucleon, "p")
    helium_flux = gsf_nucleon.flux(energy_per_nucleon, "He")
    oxygen_flux = gsf_nucleon.flux(energy_per_nucleon, "O")
    iron_flux = gsf_nucleon.flux(energy_per_nucleon, "Fe")
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
    proton_error = gsf_nucleon.error(energy_per_nucleon, "p")
    helium_error = gsf_nucleon.error(energy_per_nucleon, "He")
    oxygen_error = gsf_nucleon.error(energy_per_nucleon, "O")
    iron_error = gsf_nucleon.error(energy_per_nucleon, "Fe")
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
    np,
    oxygen_error,
    oxygen_flux,
    proton_error,
    proton_flux,
    total_error,
    total_flux,
):
    # Export nucleon flux data
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
        header="energy, proton, helium, oxygen group, iron group, total\n"
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
        header="energy, proton, helium, oxygen group, iron group, total\n"
        "[units: energy in GeV per nucleon, flux error in 1/(GeV m2 s sr)]",
    )

    print(f"Saved nucleon flux and error data ({len(energy_per_nucleon)} points each)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spectrum

    Flux scaled by E^3 to flatten the steep falloff. Shaded bands are 1-sigma.
    """)
    return


@app.cell
def _(
    energy_exponent,
    energy_per_nucleon,
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
    # Plot nucleon flux with error bands
    plt.figure(figsize=(12, 8))
    components = [
        (proton_flux, proton_error, "r", "Proton", 1),
        (helium_flux, helium_error, "orange", "Helium", 1),
        (oxygen_flux, oxygen_error, "g", "Oxygen", 1),
        (iron_flux, iron_error, "b", "Iron", 1),
        (total_flux, total_error, "k", "Total", 2),
    ]
    for flux, error, _color, _label, lw in components:
        scaled_flux = flux * energy_per_nucleon**energy_exponent
        plt.plot(
            energy_per_nucleon, scaled_flux, "-", color=_color, lw=lw, label=_label
        )
        plt.fill_between(
            energy_per_nucleon,
            (flux - error) * energy_per_nucleon**energy_exponent,
            (flux + error) * energy_per_nucleon**energy_exponent,
            facecolor=_color,
            alpha=0.2,
        )
    plt.loglog()
    plt.xlabel("$E_N$ [GeV/nucleon]")
    plt.ylabel(
        f"Nucleon Flux × $(E_N/\\mathrm{{GeV}})^{{{energy_exponent}}}$ [1/(GeV m² s sr)]"
    )
    plt.xlim(energy_per_nucleon[0], energy_per_nucleon[-1])
    plt.ylim(500.0, 10000000.0)
    plt.title("Cosmic Ray Nucleon Flux (GSF 2025)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Relative uncertainties

    A few percent in the well-measured 10 GeV – 100 TeV per nucleon range, growing toward the boundaries.
    """)
    return


@app.cell
def _(
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
    total_error,
    total_flux,
):
    # Plot nucleon relative uncertainties
    plt.figure(figsize=(12, 8))
    error_components = [
        (proton_error / np.maximum(proton_flux, 1e-90), "r", "Proton"),
        (helium_error / np.maximum(helium_flux, 1e-90), "orange", "Helium"),
        (oxygen_error / np.maximum(oxygen_flux, 1e-90), "g", "Oxygen"),
        (iron_error / np.maximum(iron_flux, 1e-90), "b", "Iron"),
        (total_error / np.maximum(total_flux, 1e-90), "k", "Total"),
    ]
    for rel_error, _color, _label in error_components:
        valid_mask = np.isfinite(rel_error) & (rel_error > 0)
        if np.any(valid_mask):
            plt.plot(
                energy_per_nucleon[valid_mask],
                rel_error[valid_mask],
                "-",
                color=_color,
                label=_label,
                lw=2,
            )
    plt.loglog()
    plt.xlabel("$E_N$ [GeV/nucleon]")
    plt.ylabel("Relative Uncertainty (σ/flux)")
    plt.xlim(energy_per_nucleon[0], energy_per_nucleon[-1])
    plt.ylim(0.003, 1)
    plt.title("Nucleon Flux Relative Uncertainties (GSF 2025)")  # & (rel_error < 10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.01, color="gray", linestyle="--", alpha=0.5)
    plt.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5)
    # Reference lines
    plt.show()
    return


if __name__ == "__main__":
    app.run()
