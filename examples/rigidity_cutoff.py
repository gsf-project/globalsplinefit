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
    # Geomagnetic Rigidity Cutoff

    Earth's magnetic field deflects cosmic rays below a location-dependent rigidity threshold $R_\text{cut}$ (~0 GV at the poles, ~15–17 GV near the equator). Because rigidity depends on charge $Z$, the cutoff affects nuclear species differently. We compare its effect on three GSF model types using $R_\text{cut} = 20$ GV (high, for visibility).
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    from globalsplinefit import GSFEnergy, GSFEnergyPerNucleon, GSFRigidity

    plt.rcParams["figure.figsize"] = (10, 6)

    R_CUT = 20.0  # GV
    groups = [
        ("p", "r", "Proton"),
        ("He", "orange", "Helium"),
        ("O*", "g", "Oxygen*"),
        ("Fe*", "b", "Iron*"),
    ]
    return GSFEnergy, GSFEnergyPerNucleon, GSFRigidity, R_CUT, groups, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Nucleus Flux (`GSFEnergy`)

    At total energy $E$, a nucleus with charge $Z$ and mass $A$ has rigidity $R = \sqrt{E^2 - (A m_N)^2} / Z$. Protons ($A/Z = 1$) hit the cutoff at higher total energies than heavier nuclei ($A/Z \approx 2$). Left panel: absolute fluxes with (dashed) and without (solid) cutoff. Right panel: suppression ratio.
    """)
    return


@app.cell
def _(GSFEnergy, R_CUT, groups, np, plt):
    energy = np.logspace(0, 3, 500)  # 1 GeV to 10 PeV
    energy_exponent = 2.6
    gsf = GSFEnergy()
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
    _ax1.set_ylim(1.0, 30000.0)
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
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rigidity Flux (`GSFRigidity`)

    In rigidity space the cutoff is a sharp vertical boundary at $R_\text{cut}$, identical for all species.
    """)
    return


@app.cell
def _(GSFRigidity, R_CUT, groups, np, plt):
    rigidity = np.logspace(-1, 4, 500)  # 0.1 GV to 10 TV
    rig_exponent = 2.6
    gsf_r = GSFRigidity()
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
    _ax1.set_ylim(0.1, 100000.0)
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
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Nucleon Flux (`GSFEnergyPerNucleon`)

    The cutoff is applied per element during the nucleon-flux summation. At a given $E_N$, nucleus $i$ has rigidity $R_i = (A_i/Z_i) \sqrt{E_N^2 - m_N^2}$, so the cutoff in $E_N$ is roughly twice as high for protons as for heavier nuclei. Left: absolute flux with 1-sigma bands; uncertainties propagate through the cutoff via the Jacobian. Right: relative uncertainty with (dashed) and without (solid) cutoff.
    """)
    return


@app.cell
def _(GSFEnergyPerNucleon, R_CUT, groups, np, plt):
    ekin = np.logspace(0, 3, 500)  # 1 GeV to 10 PeV per nucleon
    en_exponent = 3.0
    gsf_n = GSFEnergyPerNucleon()
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
    axes[0].set_ylim(500.0, 100000.0)
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
    # Relative error comparison
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - In **rigidity space** the cutoff is a universal sharp threshold.
    - In **energy space** the cutoff energy depends on $Z/A$; protons are suppressed up to higher energies than heavier nuclei.
    - **Uncertainties propagate correctly** through the cutoff with no user intervention.
    - The `rigidity_cutoff` parameter is available on all flux/error methods.
    """)
    return


if __name__ == "__main__":
    app.run()
