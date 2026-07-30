"""Core evaluation and publication-figure code for the GSF Explorer web app.

Pure functions over ``globalsplinefit`` models — no UI imports here, so this
module runs headless (tests, scripts) and in the browser (Pyodide worker)
unchanged. The figure style reproduces the paper's matplotlib conventions:
serif/mathtext fonts, framed axes with inward major+minor ticks, the GSF group
colors, no chartjunk.
"""

from __future__ import annotations

import io
import re

import numpy as np

from globalsplinefit import (
    GSFEnergy,
    GSFEnergyPerNucleon,
    GSFKineticEnergy,
    GSFKineticEnergyPerNucleon,
    GSFRigidity,
    MODEL_VERSIONS,
    version_info,
)

# ---------------------------------------------------------------- registry

GROUPS = ["p", "He", "O*", "Fe*"]

# Paper convention (globalsplinefit.plotting)
GROUP_COLORS = {
    "p": "#d62728",
    "He": "#ff7f0e",
    "O*": "#2ca02c",
    "Fe*": "#1f77b4",
    "all": "#000000",
}
# Muted cycle for individual (subleading) elements, dashed
ELEMENT_COLORS = ["#8c564b", "#9467bd", "#7f7f7f", "#bcbd22", "#17becf", "#e377c2"]

ELEMENT_SYMBOLS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti",
    23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni",
}

BASES = {
    "etot": dict(
        cls=GSFEnergy, unit="GeV", sym="E",
        label=r"Total energy per nucleus $E$",
        phrase="total energy per nucleus",
    ),
    "ekin": dict(
        cls=GSFKineticEnergy, unit="GeV", sym=r"E_{\mathrm{kin}}",
        label=r"Kinetic energy per nucleus $E_{\mathrm{kin}}$",
        phrase="kinetic energy per nucleus",
    ),
    "rig": dict(
        cls=GSFRigidity, unit="GV", sym="R",
        label=r"Rigidity $R$",
        phrase="rigidity",
    ),
    "en": dict(
        cls=GSFEnergyPerNucleon, unit="GeV", sym=r"E_{N}",
        label=r"Total energy per nucleon $E_{N}$",
        phrase="total energy per nucleon",
    ),
    "ekn": dict(
        cls=GSFKineticEnergyPerNucleon, unit="GeV", sym=r"E_{\mathrm{kin},N}",
        label=r"Kinetic energy per nucleon $E_{\mathrm{kin},N}$",
        phrase="kinetic energy per nucleon",
    ),
}

VERSION_NOTES = {
    "GSF2026": "default — mixture covering, GMD potential",
    "GSF2026-USO": "Usoskin-2017 potential (modulation systematic)",
    "2025": "historical release",
    "2019": "historical release",
    "2017": "historical release (Dembinski et al., ICRC2017)",
}

# Paper-style matplotlib rc — DejaVu Serif ships with matplotlib everywhere,
# including the Pyodide build, so screen and export are pixel-identical.
PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9.0,
    "axes.labelsize": 10.0,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.2,
    "ytick.minor.size": 2.2,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.frameon": False,
    "legend.fontsize": 8.0,
    "axes.formatter.use_mathtext": True,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "path",  # text as outlines: no font dependency in exports
}


def make_model(basis: str, version: str):
    """Construct the GSF model class for an abscissa basis + parameter set."""
    return BASES[basis]["cls"](version=version)


def phi_year_range(model) -> tuple[int, int]:
    """First and last calendar year covered by the modulation-potential data."""
    years = sorted(model.phi.keys())
    return int(years[0]), int(years[-1])


def evaluate(
    model,
    dmin: float,
    dmax: float,
    npts: int,
    groups: list[str],
    elements: list[int],
    time_interval,
    rigidity_cutoff: float | None,
    energy_scale: float = 1.0,
    with_total: bool = True,
    with_errors: bool = True,
) -> dict:
    """Evaluate fluxes (and 1σ errors) on a log grid in the model's basis.

    Returns plain numpy arrays only, so results are cache- and pickle-friendly.
    Zeros (below threshold / LIS at low energy) are kept as zeros; the plotting
    layer masks them.
    """
    model.energy_scale = float(energy_scale)
    kw = dict(time_interval=time_interval, rigidity_cutoff=rigidity_cutoff)
    x = np.logspace(float(dmin), float(dmax), int(npts))
    out = {"x": x, "series": {}, "total": None}

    def f_and_e(target):
        f = np.asarray(model.flux(x, target, **kw), float)
        e = None
        if with_errors:
            try:
                e = np.asarray(model.error(x, target, **kw), float)
            except Exception:
                e = np.zeros_like(f)
        return f, e

    for g in groups:
        out["series"][g] = f_and_e(g)
    for z in elements:
        # "D" (GSF2026+ deuterium species) is a native flux/error target.
        name = "D" if z == "D" else ELEMENT_SYMBOLS.get(z, f"Z={z}")
        if name in out["series"]:
            continue  # e.g. element He duplicates the He group series
        out["series"][name] = f_and_e(z if z == "D" else int(z))
    if with_total:
        tf = np.asarray(model.total_flux(x, **kw), float)
        te = None
        if with_errors:
            te = np.asarray(model.total_error(x, **kw), float)
        out["total"] = (tf, te)
    return out


def total_covariance(model, x, time_interval, rigidity_cutoff) -> np.ndarray:
    """Covariance matrix of the all-particle flux on grid ``x``."""
    kw = dict(time_interval=time_interval, rigidity_cutoff=rigidity_cutoff)
    n = len(x)
    cov = np.zeros((n, n))
    for g1 in GROUPS:
        for g2 in GROUPS:
            cov += np.asarray(model.covariance(g1, g2, x, **kw), float)
    return cov


# ---------------------------------------------------------------- figure

def _weighted(x, y, gamma):
    y = np.asarray(y, float)
    w = np.where(y > 0, y * x**gamma, np.nan)
    return w


def _gfmt(v: float) -> str:
    """Format an exponent without trailing zeros (2.7, 3, 2.65)."""
    return f"{v:g}"


def axis_labels(basis: str, gamma: float) -> tuple[str, str]:
    b = BASES[basis]
    xlab = f"{b['label']} [{b['unit']}]"
    u = b["unit"]
    if gamma == 0:
        ylab = rf"$\Phi\;[(\mathrm{{{u}}}\,\mathrm{{m^2\,s\,sr}})^{{-1}}]$"
    else:
        ylab = (
            rf"${b['sym']}^{{{_gfmt(gamma)}}}\,\Phi\;"
            rf"[\mathrm{{{u}}}^{{{_gfmt(gamma - 1)}}}"
            rf"\,\mathrm{{m^{{-2}}\,s^{{-1}}\,sr^{{-1}}}}]$"
        )
    return xlab, ylab


def make_figure(
    result: dict,
    basis: str,
    gamma: float,
    *,
    show_total: bool = True,
    show_bands: bool = True,
    band_alpha: float = 0.18,
    ylog: bool = True,
    grid: bool = False,
    width_in: float = 7.0,
    height_in: float = 4.6,
):
    """Publication-style spectrum figure. Returns a matplotlib Figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = result["x"]
    with matplotlib.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=110)

        if show_total and result["total"] is not None:
            tf, te = result["total"]
            w = _weighted(x, tf, gamma)
            ax.plot(x, w, color=GROUP_COLORS["all"], lw=1.9, label="all-particle",
                    zorder=6)
            if show_bands and te is not None:
                ax.fill_between(
                    x, _weighted(x, tf - te, gamma), _weighted(x, tf + te, gamma),
                    color=GROUP_COLORS["all"], alpha=band_alpha, lw=0, zorder=2)

        ecolors = iter(ELEMENT_COLORS * 4)
        for name, (f, e) in result["series"].items():
            if name in GROUP_COLORS:
                c, ls, lw, z = GROUP_COLORS[name], "-", 1.5, 5
            else:
                c, ls, lw, z = next(ecolors), "--", 1.1, 4
            ax.plot(x, _weighted(x, f, gamma), color=c, ls=ls, lw=lw,
                    label=name, zorder=z)
            if show_bands and e is not None and name in GROUP_COLORS:
                ax.fill_between(
                    x, _weighted(x, f - e, gamma), _weighted(x, f + e, gamma),
                    color=c, alpha=band_alpha, lw=0, zorder=1)

        ax.set_xscale("log")
        # Anchor limits to the MEANS of the solid series (total + groups):
        # matplotlib's autoscale would otherwise include the fill_between
        # bands (sigma reaches ~65x flux at the highest energies) and faint
        # element curves; paper-style tightness.
        solid = [result["total"][0]] if (show_total and result["total"]) else []
        solid += [f for n, (f, _) in result["series"].items()
                  if n in GROUP_COLORS]
        vals = np.concatenate([_weighted(x, f, gamma) for f in solid]) \
            if solid else np.array([np.nan])
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if ylog:
            ax.set_yscale("log")
            if vals.size:
                ax.set_ylim(vals.min() / 2.5, vals.max() * 2.5)
        elif vals.size:
            ax.set_ylim(0, vals.max() * 1.06)
        ax.set_xlim(x[0], x[-1])
        xlab, ylab = axis_labels(basis, gamma)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        if grid:
            ax.grid(alpha=0.25, lw=0.5, which="major")
        nser = len(result["series"]) + (1 if show_total else 0)
        if nser:
            ax.legend(ncol=2 if nser > 4 else 1, handlelength=1.9,
                      labelspacing=0.35, columnspacing=1.3)
        fig.tight_layout(pad=0.4)
    return fig


def figure_svg(fig) -> str:
    """Serialize a figure to responsive SVG (fixed pixel size stripped)."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.04)
    svg = buf.getvalue()
    svg = svg[svg.find("<svg"):]
    svg = re.sub(r'(<svg[^>]*?)\swidth="[^"]*"', r"\1", svg, count=1)
    svg = re.sub(r'(<svg[^>]*?)\sheight="[^"]*"', r"\1", svg, count=1)
    return svg


def figure_bytes(fig, fmt: str, dpi: int = 300) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    return buf.getvalue()


# ---------------------------------------------------------------- export

def modulation_phrase(ti) -> str:
    if ti is None:
        return "Solar Cycle 24 average (Dec 2008 – Dec 2019)"
    if ti == "LIS":
        return "local interstellar spectrum (no solar modulation)"
    a, b = ti
    return f"averaged over {a // 100}-{a % 100:02d} to {b // 100}-{b % 100:02d}"


def caption(version: str, basis: str, gamma: float, ti, elements: list[int],
            show_bands: bool, rigidity_cutoff: float | None) -> str:
    b = BASES[basis]
    name = version if version.startswith("GSF") else f"GSF {version}"
    sym = {"etot": "E", "ekin": "E_kin", "rig": "R",
           "en": "E_N", "ekn": "E_kin,N"}[basis]
    parts = [
        f"{name}, {modulation_phrase(ti)}.",
        f"All-particle and mass-group fluxes versus {b['phrase']}"
        + (f", weighted by {sym}^{_gfmt(gamma)}." if gamma else "."),
    ]
    if show_bands:
        parts.append("Shaded bands: ±1σ model uncertainty from the fit covariance.")
    if elements:
        syms = ", ".join(ELEMENT_SYMBOLS.get(z, str(z)) for z in elements)
        parts.append(f"Dashed: individual elements {syms}.")
    if rigidity_cutoff:
        parts.append(f"Geomagnetic cutoff {rigidity_cutoff:g} GV applied.")
    return " ".join(parts)


def build_csv(model, result: dict, basis: str, version: str, ti,
              rigidity_cutoff, include_cov: bool) -> str:
    """CSV export: provenance header, flux table, optional covariance block."""
    b = BASES[basis]
    x = result["x"]
    cols = ["x"]
    arrs = [x]
    for name, (f, e) in result["series"].items():
        cols.append(f"flux_{name.replace('*', 'star')}")
        arrs.append(f)
        if e is not None:
            cols.append(f"err_{name.replace('*', 'star')}")
            arrs.append(e)
    if result["total"] is not None:
        tf, te = result["total"]
        cols.append("flux_total")
        arrs.append(tf)
        if te is not None:
            cols.append("err_total")
            arrs.append(te)

    lines = [
        f"# Global Spline Fit (GSF) — parameter set {version}",
        f"# abscissa: {b['phrase']} [{b['unit']}], log grid, {len(x)} points",
        f"# flux unit: ({b['unit']} m^2 s sr)^-1; err = 1 sigma from fit covariance",
        f"# solar modulation: {modulation_phrase(ti)}",
    ]
    if rigidity_cutoff:
        lines.append(f"# geomagnetic rigidity cutoff: {rigidity_cutoff:g} GV")
    lines.append("# generated by GSF Explorer (globalsplinefit)")
    lines.append(",".join(cols))
    mat = np.column_stack(arrs)
    lines += [",".join(f"{v:.6e}" for v in row) for row in mat]

    if include_cov:
        cov = total_covariance(model, x, ti, rigidity_cutoff)
        lines.append("#")
        lines.append(f"# covariance of flux_total on the grid above "
                     f"({len(x)}x{len(x)}, row-major)")
        lines += [",".join(f"{v:.4e}" for v in row) for row in cov]
    return "\n".join(lines) + "\n"


__all__ = [
    "BASES", "GROUPS", "GROUP_COLORS", "ELEMENT_SYMBOLS", "VERSION_NOTES",
    "MODEL_VERSIONS", "version_info", "make_model", "phi_year_range",
    "evaluate", "total_covariance", "make_figure", "figure_svg",
    "figure_bytes", "axis_labels", "caption", "modulation_phrase", "build_csv",
]
