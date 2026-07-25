"""Plotting / deck generation for GSF models.

Renders the standard model deck (flux spectrum, composition fractions, ln A
moments, relative uncertainty, nucleon flux, flux correlation) directly from a
loaded model — i.e. from a parameter-file directory via any GSF* model class —
plus a back-to-back comparison of two models. matplotlib is required only for
this module (kept out of the core package import).

Examples
--------
>>> from globalsplinefit import GSFEnergy
>>> from globalsplinefit.plotting import make_deck, compare_models
>>> make_deck(GSFEnergy(), "deck_gsf2026", label="GSF2026")
>>> make_deck(GSFEnergy(data_path="my_fit/"), "deck_mine", label="my fit")
>>> compare_models(GSFEnergy(data_path="my_fit/"), GSFEnergy(),
...                "cmp", label="my fit", ref_label="GSF2026")
"""
from __future__ import annotations

import os
import traceback

import numpy as np

GROUPS = ["p", "He", "O*", "Fe*"]
COLORS = {"p": "#d62728", "He": "#ff7f0e", "O*": "#2ca02c", "Fe*": "#1f77b4", "all": "k"}
EXP = 2.6  # E^EXP flux scaling for display


def _grid(model, energy):
    if energy is not None:
        return np.asarray(energy, float)
    # rigidity model -> GV, energy models -> GeV; both span the GSF range
    return np.logspace(0, 11, 300)


def _save(fig, outdir, name, formats):
    os.makedirs(outdir, exist_ok=True)
    for fmt in formats:
        fig.savefig(os.path.join(outdir, f"{name}.{fmt}"), dpi=120, bbox_inches="tight")


def _flux_err(model, target, x, ti):
    f = np.asarray(model.flux(x, target, time_interval=ti), float)
    try:
        e = np.asarray(model.error(x, target, time_interval=ti), float)
    except Exception:
        e = np.zeros_like(f)
    return f, e


def make_deck(model, outdir, label="GSF", *, time_interval=None, energy=None,
              formats=("png",)):
    """Render the standard model deck into ``outdir``. Returns list of figure names."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = _grid(model, energy)
    lx = np.log10(x)
    sca = x ** EXP
    ti = time_interval
    done = []

    def panel(name, fn):
        try:
            fig = fn()
            _save(fig, outdir, name, formats)
            plt.close(fig)
            done.append(name)
            print(f"  OK   {name}", flush=True)
        except Exception:
            print(f"  FAIL {name}: {traceback.format_exc()[-200:]}", flush=True)

    # 1) flux spectrum: all-particle + groups, x E^2.6, with +-1sigma bands
    def f_spectrum():
        fig, ax = plt.subplots(figsize=(9, 6))
        tf = np.asarray(model.total_flux(x, time_interval=ti), float)
        te = np.asarray(model.total_error(x, time_interval=ti), float)
        ax.plot(x, tf * sca, color="k", lw=2.2, label="all-particle")
        ax.fill_between(x, (tf - te) * sca, (tf + te) * sca, color="k", alpha=0.18, lw=0)
        for g in GROUPS:
            f, e = _flux_err(model, g, x, ti)
            ax.plot(x, f * sca, color=COLORS[g], lw=1.6, label=g)
            ax.fill_between(x, (f - e) * sca, (f + e) * sca, color=COLORS[g], alpha=0.18, lw=0)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e2, 6e4)
        ax.set_xlabel("E [GeV]"); ax.set_ylabel(r"$E^{2.6}\,\Phi$ [GeV$^{1.6}$m$^{-2}$s$^{-1}$sr$^{-1}$]")
        ax.set_title(f"{label}: flux spectrum (×E$^{{2.6}}$, ±1σ)")
        ax.legend(ncol=2, fontsize=9); ax.grid(alpha=0.3)
        return fig
    panel("cr_flux_spectrum", f_spectrum)

    # 2) composition fractions
    def f_fractions():
        fig, ax = plt.subplots(figsize=(9, 6))
        for g in GROUPS:
            ax.plot(x, model.fraction(x, g, time_interval=ti), color=COLORS[g], lw=1.8, label=g)
        ax.set_xscale("log"); ax.set_ylim(0, 1); ax.set_xlabel("E [GeV]")
        ax.set_ylabel("flux fraction"); ax.set_title(f"{label}: composition fractions")
        ax.legend(ncol=2, fontsize=9); ax.grid(alpha=0.3)
        return fig
    panel("cr_fractions", f_fractions)

    # 3) ln A moments
    def f_lna():
        fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
        ax[0].plot(x, model.mean_lnA(x, time_interval=ti), "k-", lw=1.8)
        ax[0].set_title(f"{label}: ⟨ln A⟩"); ax[0].set_ylabel("⟨ln A⟩")
        ax[1].plot(x, model.var_lnA(x, time_interval=ti), "k-", lw=1.8)
        ax[1].set_title(f"{label}: Var(ln A)"); ax[1].set_ylabel("Var(ln A)")
        for a in ax:
            a.set_xscale("log"); a.set_xlabel("E [GeV]"); a.grid(alpha=0.3)
        return fig
    panel("cr_lna_moments", f_lna)

    # 4) relative uncertainty
    def f_relunc():
        fig, ax = plt.subplots(figsize=(9, 6))
        tf = np.asarray(model.total_flux(x, time_interval=ti), float)
        te = np.asarray(model.total_error(x, time_interval=ti), float)
        ax.plot(x, te / tf, "k-", lw=2.0, label="all-particle")
        for g in GROUPS:
            f, e = _flux_err(model, g, x, ti)
            ax.plot(x, e / np.where(f > 0, f, np.nan), color=COLORS[g], lw=1.4, label=g)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-3, 3)
        ax.set_xlabel("E [GeV]"); ax.set_ylabel(r"relative uncertainty $\sigma/\Phi$")
        ax.set_title(f"{label}: relative uncertainty"); ax.legend(ncol=2, fontsize=9); ax.grid(alpha=0.3)
        return fig
    panel("cr_rel_uncertainty", f_relunc)

    # 5) nucleon flux (build a per-nucleon model from the same parameter dir)
    def f_nucleon():
        from .model import GSFEnergyPerNucleon
        dp = str(model.params.data_path)
        nuc = GSFEnergyPerNucleon(data_path=dp,
                                  default_time_interval=getattr(model, "default_time_interval", "LIS"))
        xn = np.logspace(0, 11, 300)
        fig, ax = plt.subplots(figsize=(9, 6))
        tot = np.asarray(nuc.total_flux(xn, time_interval=ti), float)
        ax.plot(xn, tot * xn ** EXP, "k-", lw=2.0, label="nucleon (p+n)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("E/nucleon [GeV]")
        ax.set_ylabel(r"$E^{2.6}\,\Phi_{\rm nucleon}$"); ax.set_title(f"{label}: nucleon flux")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        return fig
    panel("cr_nucleon_flux", f_nucleon)

    # 6) all-particle flux correlation across energy
    def f_corr():
        xc = np.logspace(1, 11, 80)
        cov = np.zeros((len(xc), len(xc)))
        for l1 in GROUPS:
            for l2 in GROUPS:
                cov += np.asarray(model.covariance(l1, l2, xc, time_interval=ti), float)
        d = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
        corr = cov / np.outer(d, d)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.pcolormesh(np.log10(xc), np.log10(xc), corr, cmap="RdBu_r", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, label="correlation")
        ax.set_xlabel("log10 E [GeV]"); ax.set_ylabel("log10 E [GeV]")
        ax.set_title(f"{label}: all-particle flux correlation")
        return fig
    panel("cr_flux_correlation", f_corr)

    print(f"deck '{label}' -> {outdir}: {len(done)} figures", flush=True)
    return done


def compare_models(model, ref, outdir, label="model", ref_label="reference", *,
                   time_interval=None, energy=None, formats=("png",)):
    """Back-to-back (Kozo-style): main E^2.6 panel (both, solid/dotted) + per-group
    new/ref ratio sub-panels with error bands. Curves+covariance only (no data)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    x = _grid(model, energy); sca = x ** EXP; ti = time_interval
    items = GROUPS + ["all"]

    def fe(m, g):
        if g == "all":
            return (np.asarray(m.total_flux(x, time_interval=ti), float),
                    np.asarray(m.total_error(x, time_interval=ti), float))
        return _flux_err(m, g, x, ti)
    fn = {g: fe(model, g) for g in items}; fr = {g: fe(ref, g) for g in items}

    fig = plt.figure(figsize=(8.5, 11))
    gs = fig.add_gridspec(6, 1, height_ratios=[3.2, 1, 1, 1, 1, 1], hspace=0.12)
    ax0 = fig.add_subplot(gs[0]); rax = [fig.add_subplot(gs[i + 1], sharex=ax0) for i in range(5)]
    for g in items:
        c = COLORS[g]
        ax0.plot(x, fr[g][0] * sca, c, ls="-", lw=1.8)
        ax0.fill_between(x, (fr[g][0] - fr[g][1]) * sca, (fr[g][0] + fr[g][1]) * sca, color=c, alpha=0.18, lw=0)
        ax0.plot(x, fn[g][0] * sca, c, ls=":", lw=1.8)
    ax0.set_xscale("log"); ax0.set_yscale("log"); ax0.set_ylim(1e2, 6e4)
    ax0.set_ylabel(r"$E^{2.6}\,\Phi$"); ax0.set_title(f"Back-to-back: {ref_label} (solid) vs {label} (dotted)")
    leg = [Line2D([], [], color=COLORS[g], label=g) for g in items]
    leg += [Line2D([], [], color="gray", ls="-", label=ref_label), Line2D([], [], color="gray", ls=":", label=label)]
    ax0.legend(handles=leg, ncol=2, fontsize=8)
    for ax, g in zip(rax, items):
        c = COLORS[g]; (fnf, fne), (frf, fre) = fn[g], fr[g]
        ratio = fnf / np.where(frf > 0, frf, np.nan)
        ax.fill_between(x, 1 - fre / frf, 1 + fre / frf, color="0.8", lw=0)
        ax.plot(x, ratio, c, lw=1.6)
        ax.fill_between(x, ratio - fne / frf, ratio + fne / frf, color=c, alpha=0.25, lw=0)
        ax.axhline(1.0, color="k", lw=0.6); ax.set_ylabel(f"{g}\nnew/ref", fontsize=8)
        ax.set_ylim(0.6, 1.4); ax.set_xscale("log"); ax.grid(alpha=0.25)
        for t in ax.get_xticklabels():
            t.set_visible(False)
    for t in rax[-1].get_xticklabels():
        t.set_visible(True)
    rax[-1].set_xlabel("E [GeV]")
    _save(fig, outdir, "cr_compare_backtoback", formats); plt.close(fig)
    print(f"comparison {label} vs {ref_label} -> {outdir}/cr_compare_backtoback", flush=True)
