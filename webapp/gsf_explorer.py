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
    MODEL_VERSIONS,
    GSFEnergy,
    GSFEnergyPerNucleon,
    GSFKineticEnergy,
    GSFKineticEnergyPerNucleon,
    GSFRigidity,
    version_info,
)

# ---------------------------------------------------------------- registry

GROUPS = ["p", "He", "O*", "Fe*"]
NUCLEON_BASES = frozenset({"en", "ekn"})

QUANTITIES = {
    "nucleus": {"label": "Nucleus flux", "ui_label": "Nucleus Flux", "kind": "flux"},
    "nucleon": {"label": "Nucleon flux", "ui_label": "Nucleon Flux", "kind": "flux"},
    "mean_lna": {
        "label": r"$\langle\ln A\rangle$",
        "ui_label": "⟨ln A⟩",
        "kind": "composition",
    },
    "var_lna": {
        "label": r"$\sigma^2(\ln A)$",
        "ui_label": "σ²(ln A)",
        "kind": "composition",
    },
}

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
    "2026": "default — mixture covering, GMD potential",
    "2026-USO": "Usoskin-2017 potential (modulation systematic)",
    "2026-S23e": "SIBYLL-2.3e only (single-interpretation variant)",
    "2026-EPOS-LHCR": "EPOS-LHC-R only (single-interpretation variant)",
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


def make_model(basis: str, version: str, phi_bins: int | None = 12):
    """Construct the GSF model class for an abscissa basis + parameter set.

    ``phi_bins`` sets the solar-cycle period-average resolution (the model's
    ``solar_cycle_average_bins``). Evaluation cost grows with it, so the app
    keeps the package default of 12 and exposes the knob under Advanced.
    """
    return BASES[basis]["cls"](version=version, solar_cycle_average_bins=phi_bins)


def validate_quantity_basis(quantity: str, basis: str) -> None:
    """Reject unknown observables and physically invalid axis combinations."""
    if quantity not in QUANTITIES:
        raise ValueError(f"unknown plot quantity {quantity!r}")
    if basis not in BASES:
        raise ValueError(f"unknown abscissa {basis!r}")
    if quantity == "nucleon" and basis not in NUCLEON_BASES:
        raise ValueError("nucleon flux requires an energy-per-nucleon abscissa")


def _nucleus_flux_jacobian(model, x, target, time_interval, rigidity_cutoff):
    """Nucleus intensity and Jacobian in any supported coordinate.

    The per-nucleon model classes expose *nucleon* intensity: besides the
    coordinate Jacobian they multiply by the number of nucleons.  For a
    nucleus-flux or composition plot versus E/A we need only
    ``dE_nucleus/d(E/A) = A``.  Other coordinates already expose nucleus
    intensity directly.
    """
    if not isinstance(model, (GSFEnergyPerNucleon, GSFKineticEnergyPerNucleon)):
        kw = dict(time_interval=time_interval, rigidity_cutoff=rigidity_cutoff)
        return (
            np.asarray(model.flux(x, target, **kw), float),
            np.asarray(model.jacobian(x, target, **kw), float),
        )

    ti = model._resolve_time_interval(time_interval)
    cutoff = model._resolve_rigidity_cutoff(rigidity_cutoff)
    zlist, _ = model._resolve_z(target)
    energy_per_nucleon = model._transform_energy_per_nucleon(x, target)
    flux = np.zeros_like(energy_per_nucleon, dtype=float)
    jacobian = 0.0
    for sid in model._target_sids(zlist):
        mass_scale = model.z_to_a[sid]
        energy = energy_per_nucleon * mass_scale
        mask = model._rigidity_cutoff_mask(sid, energy, cutoff)
        flux += model._element_flux(sid, energy, ti) * mass_scale * mask
        jacobian += model._element_flux_jacobian(sid, energy, ti) * (
            mass_scale * mask
        )[:, np.newaxis]
    return flux, np.asarray(jacobian)


def _error_from_jacobian(model, target, jacobian) -> np.ndarray:
    block = model._covariance_block(target, target)
    if block is None:
        return np.zeros(jacobian.shape[0])
    variance = np.einsum("ni,ij,nj->n", jacobian, block, jacobian)
    return np.sqrt(np.clip(variance, 0.0, None))


def _total_from_groups(
    model, x, time_interval, rigidity_cutoff, with_errors, cache=None
):
    """Nucleus all-particle intensity, including cross-group covariance."""
    flux = np.zeros(len(x))
    jacobians = {}
    cache = {} if cache is None else cache
    for group in GROUPS:
        if group not in cache:
            cache[group] = _nucleus_flux_jacobian(
                model, x, group, time_interval, rigidity_cutoff
            )
        group_flux, jac = cache[group]
        flux += group_flux
        jacobians[group] = jac
    if not with_errors:
        return flux, None
    variance = np.zeros(len(x))
    for group1, jac1 in jacobians.items():
        for group2, jac2 in jacobians.items():
            block = model._covariance_block(group1, group2)
            if block is not None:
                variance += np.einsum("ni,ij,nj->n", jac1, block, jac2)
    return flux, np.sqrt(np.clip(variance, 0.0, None))


def _composition(model, x, quantity, time_interval, rigidity_cutoff, with_errors):
    """Evaluate a nucleus-weighted ln(A) moment and its propagated error."""
    fluxes = []
    jacobians = []
    leaders = []
    ln_a = []
    for sid in model.species:
        flux, jac = _nucleus_flux_jacobian(
            model, x, sid, time_interval, rigidity_cutoff
        )
        fluxes.append(flux)
        jacobians.append(jac)
        leaders.append(model.z_ungroup[sid[0]])
        ln_a.append(np.log(model.mass_number[sid]))
    fluxes = np.asarray(fluxes)
    ln_a = np.asarray(ln_a)[:, np.newaxis]
    denominator = np.where(fluxes.sum(axis=0) > 0, fluxes.sum(axis=0), np.nan)
    mean = (fluxes * ln_a).sum(axis=0) / denominator
    second = (fluxes * ln_a**2).sum(axis=0) / denominator
    if quantity == "mean_lna":
        value = mean
        weights = (ln_a - mean) / denominator
    else:
        value = second - mean**2
        weights = ((ln_a**2 - second) - 2.0 * mean * (ln_a - mean)) / denominator
    error = (
        model._derived_error(weights, jacobians, leaders) if with_errors else None
    )
    return value, error


def phi_year_range(model) -> tuple[int, int]:
    """First and last calendar year covered by the modulation-potential data."""
    years = sorted(model.phi.keys())
    return int(years[0]), int(years[-1])


def evaluate(
    model,
    basis: str,
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
    quantity: str = "nucleus",
) -> dict:
    """Evaluate the selected flux or composition observable on a log grid.

    Returns plain numpy arrays only, so results are cache- and pickle-friendly.
    Zeros (below threshold / LIS at low energy) are kept as zeros; the plotting
    layer masks them.
    """
    validate_quantity_basis(quantity, basis)
    energy_scale = float(energy_scale)
    if not np.isfinite(energy_scale) or energy_scale <= 0:
        raise ValueError("energy-scale factor must be finite and positive")
    # The Explorer exposes a multiplicative factor (1 = unchanged), whereas
    # the package property stores a fractional shift (0 = unchanged).
    model.energy_scale = energy_scale - 1.0
    kw = dict(time_interval=time_interval, rigidity_cutoff=rigidity_cutoff)
    x = np.logspace(float(dmin), float(dmax), int(npts))
    out = {"x": x, "series": {}, "total": None, "quantity": quantity}

    if QUANTITIES[quantity]["kind"] == "composition":
        out["series"][quantity] = _composition(
            model, x, quantity, time_interval, rigidity_cutoff, with_errors
        )
        return out

    nucleus_cache = {}

    def f_and_e(target):
        if quantity == "nucleus":
            f, jac = _nucleus_flux_jacobian(
                model, x, target, time_interval, rigidity_cutoff
            )
            nucleus_cache[target] = (f, jac)
        else:
            f = np.asarray(model.flux(x, target, **kw), float)
            jac = None
        e = None
        if with_errors:
            e = (
                _error_from_jacobian(model, target, jac)
                if jac is not None
                else np.asarray(model.error(x, target, **kw), float)
            )
        return f, e

    for g in groups:
        out["series"][g] = f_and_e(g)
    for z in elements:
        # "D" (2026+ deuterium species) is a native flux/error target.
        name = "D" if z == "D" else ELEMENT_SYMBOLS.get(z, f"Z={z}")
        if name in out["series"]:
            continue  # e.g. element He duplicates the He group series
        out["series"][name] = f_and_e(z if z == "D" else int(z))
    if with_total:
        if quantity == "nucleus":
            out["total"] = _total_from_groups(
                model, x, time_interval, rigidity_cutoff, with_errors,
                nucleus_cache,
            )
        else:
            tf = np.asarray(model.total_flux(x, **kw), float)
            te = np.asarray(model.total_error(x, **kw), float) if with_errors else None
            out["total"] = (tf, te)
    return out


def total_covariance(
    model, x, time_interval, rigidity_cutoff, quantity: str = "nucleus"
) -> np.ndarray:
    """Covariance matrix of the total nucleus or nucleon flux on grid ``x``."""
    if quantity not in {"nucleus", "nucleon"}:
        raise ValueError("total covariance is defined only for flux quantities")
    kw = dict(time_interval=time_interval, rigidity_cutoff=rigidity_cutoff)
    n = len(x)
    cov = np.zeros((n, n))
    jacobians = (
        {
            group: _nucleus_flux_jacobian(
                model, x, group, time_interval, rigidity_cutoff
            )[1]
            for group in GROUPS
        }
        if quantity == "nucleus"
        else None
    )
    for g1 in GROUPS:
        for g2 in GROUPS:
            if quantity == "nucleon":
                block = np.asarray(model.covariance(g1, g2, x, **kw), float)
            else:
                jac1 = jacobians[g1]
                jac2 = jacobians[g2]
                parameter_covariance = model._covariance_block(g1, g2)
                block = (
                    model._propagate_cov(jac1, jac2, parameter_covariance)
                    if parameter_covariance is not None
                    else np.zeros((n, n))
                )
            cov += block
    return cov


# ---------------------------------------------------------------- figure

def _weighted(x, y, gamma):
    y = np.asarray(y, float)
    w = np.where(y > 0, y * x**gamma, np.nan)
    return w


def _gfmt(v: float) -> str:
    """Format an exponent without trailing zeros (2.7, 3, 2.65)."""
    return f"{v:g}"


def axis_labels(
    basis: str, gamma: float, quantity: str = "nucleus"
) -> tuple[str, str]:
    b = BASES[basis]
    xlab = f"{b['label']} [{b['unit']}]"
    if quantity == "mean_lna":
        return xlab, r"$\langle\ln A\rangle$"
    if quantity == "var_lna":
        return xlab, r"$\sigma^2(\ln A)$"
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

        quantity = result.get("quantity", "nucleus")
        is_composition = QUANTITIES[quantity]["kind"] == "composition"

        if is_composition:
            value, error = result["series"][quantity]
            label = QUANTITIES[quantity]["label"]
            ax.plot(x, value, color=GROUP_COLORS["all"], lw=1.9, label=label,
                    zorder=6)
            if show_bands and error is not None:
                ax.fill_between(
                    x, value - error, value + error,
                    color=GROUP_COLORS["all"], alpha=band_alpha, lw=0, zorder=2)

        if not is_composition and show_total and result["total"] is not None:
            tf, te = result["total"]
            w = _weighted(x, tf, gamma)
            ax.plot(x, w, color=GROUP_COLORS["all"], lw=1.9, label="all-particle",
                    zorder=6)
            if show_bands and te is not None:
                ax.fill_between(
                    x, _weighted(x, tf - te, gamma), _weighted(x, tf + te, gamma),
                    color=GROUP_COLORS["all"], alpha=band_alpha, lw=0, zorder=2)

        ecolors = iter(ELEMENT_COLORS * 4)
        for name, (f, e) in result["series"].items() if not is_composition else ():
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
        if is_composition:
            vals = np.asarray(result["series"][quantity][0], float)
        else:
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
        xlab, ylab = axis_labels(basis, gamma, quantity)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        if grid:
            ax.grid(alpha=0.25, lw=0.5, which="major")
        nser = len(result["series"]) + (1 if show_total and result["total"] else 0)
        if nser:
            ax.legend(ncol=2 if nser > 4 else 1, handlelength=1.9,
                      labelspacing=0.35, columnspacing=1.3)
        # TEMPORARY pre-publication marker (mirrors PRELIMINARY in chart.js).
        # Delete at the GSF 2026 release.
        ax.text(0.5, 0.5, "PRELIMINARY", transform=ax.transAxes,
                ha="center", va="center", rotation=30, fontsize=34,
                fontweight="bold", color="gray", alpha=0.18, zorder=100)
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
            show_bands: bool, rigidity_cutoff: float | None,
            quantity: str = "nucleus") -> str:
    b = BASES[basis]
    name = version if version.startswith("GSF") else f"GSF {version}"
    sym = {"etot": "E", "ekin": "E_kin", "rig": "R",
           "en": "E_N", "ekn": "E_kin,N"}[basis]
    if QUANTITIES[quantity]["kind"] == "composition":
        observable = (
            "Mean logarithmic mass <ln A>" if quantity == "mean_lna"
            else "Logarithmic-mass variance sigma^2(ln A)"
        )
        parts = [
            f"{name}, {modulation_phrase(ti)}.",
            f"{observable} versus {b['phrase']}.",
        ]
    else:
        flux_name = "All-nucleon" if quantity == "nucleon" else "All-particle"
        parts = [
            f"{name}, {modulation_phrase(ti)}.",
            f"{flux_name} and mass-group fluxes versus {b['phrase']}"
            + (f", weighted by {sym}^{_gfmt(gamma)}." if gamma else "."),
        ]
    if show_bands:
        parts.append("Shaded bands: ±1σ model uncertainty from the fit covariance.")
    if elements and QUANTITIES[quantity]["kind"] == "flux":
        syms = ", ".join(ELEMENT_SYMBOLS.get(z, str(z)) for z in elements)
        parts.append(f"Dashed: individual elements {syms}.")
    if rigidity_cutoff:
        parts.append(f"Geomagnetic cutoff {rigidity_cutoff:g} GV applied.")
    return " ".join(parts)


def build_csv(model, result: dict, basis: str, version: str, ti,
              rigidity_cutoff, include_cov: bool,
              quantity: str = "nucleus") -> str:
    """CSV export: provenance header, flux table, optional covariance block."""
    b = BASES[basis]
    x = result["x"]
    cols = ["x"]
    arrs = [x]
    prefix = {
        "nucleus": "nucleus_flux",
        "nucleon": "nucleon_flux",
        "mean_lna": "mean_lnA",
        "var_lna": "var_lnA",
    }[quantity]
    for name, (f, e) in result["series"].items():
        suffix = "" if quantity in {"mean_lna", "var_lna"} else f"_{name.replace('*', 'star')}"
        cols.append(f"{prefix}{suffix}")
        arrs.append(f)
        if e is not None:
            cols.append(f"err_{prefix}{suffix}")
            arrs.append(e)
    if result["total"] is not None:
        tf, te = result["total"]
        cols.append(f"{prefix}_total")
        arrs.append(tf)
        if te is not None:
            cols.append("err_total")
            arrs.append(te)

    lines = [
        f"# Global Spline Fit (GSF) — parameter set {version}",
        f"# abscissa: {b['phrase']} [{b['unit']}], log grid, {len(x)} points",
        f"# quantity: {QUANTITIES[quantity]['label']}",
        (
            f"# flux unit: ({b['unit']} m^2 s sr)^-1; err = 1 sigma from fit covariance"
            if QUANTITIES[quantity]["kind"] == "flux"
            else "# composition values are dimensionless; err = 1 sigma from fit covariance"
        ),
        f"# solar modulation: {modulation_phrase(ti)}",
    ]
    if ti != "LIS":
        _bins = model.solar_cycle_average_bins
        lines.append(
            "# period average: "
            + ("every month, explicitly" if _bins is None else f"{_bins} phi bins")
        )
    if rigidity_cutoff:
        lines.append(f"# geomagnetic rigidity cutoff: {rigidity_cutoff:g} GV")
    lines.append("# generated by GSF Explorer (globalsplinefit)")
    lines.append(",".join(cols))
    mat = np.column_stack(arrs)
    lines += [",".join(f"{v:.6e}" for v in row) for row in mat]

    if include_cov and QUANTITIES[quantity]["kind"] == "flux":
        cov = total_covariance(model, x, ti, rigidity_cutoff, quantity)
        lines.append("#")
        lines.append(f"# covariance of {prefix}_total on the grid above "
                     f"({len(x)}x{len(x)}, row-major)")
        lines += [",".join(f"{v:.4e}" for v in row) for row in cov]
    return "\n".join(lines) + "\n"


__all__ = [
    "BASES", "QUANTITIES", "NUCLEON_BASES", "GROUPS", "GROUP_COLORS",
    "ELEMENT_SYMBOLS", "VERSION_NOTES",
    "MODEL_VERSIONS", "version_info", "make_model", "phi_year_range",
    "validate_quantity_basis", "evaluate", "total_covariance", "make_figure",
    "figure_svg",
    "figure_bytes", "axis_labels", "caption", "modulation_phrase", "build_csv",
]
