"""Core evaluation and publication-figure code for the GSF Explorer web app.

Pure functions over ``globalsplinefit`` models — no UI imports here, so this
module runs headless (tests, scripts) and in the browser (Pyodide worker)
unchanged. The figure style reproduces the paper's matplotlib conventions:
Nimbus Roman text and math, framed axes with major+minor ticks, the GSF group
colors, no chartjunk.
"""

from __future__ import annotations

import io
import re
from functools import lru_cache
from pathlib import Path

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

GROUPS = ["H*", "He*", "O*", "Fe*"]
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

# Main-paper spectrum palette (as in the paper figures).
GROUP_COLORS = {
    "H*": "r",
    "He*": "y",
    "O*": "g",
    "Fe*": "b",
    "all": "#000000",
}
GROUP_STYLES = dict(zip(GROUPS, ("-", "--", "-.", ":"), strict=True))
# Muted cycle for individual (subleading) elements, dashed
ELEMENT_COLORS = ["#8c564b", "#9467bd", "#7f7f7f", "#bcbd22", "#17becf", "#e377c2"]

ELEMENT_SYMBOLS = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
}

BASES = {
    "etot": {
        "cls": GSFEnergy,
        "unit": "GeV",
        "sym": "E",
        "label": r"Total energy per nucleus $E$",
        "phrase": "total energy per nucleus",
    },
    "ekin": {
        "cls": GSFKineticEnergy,
        "unit": "GeV",
        "sym": r"E_{\mathrm{kin}}",
        "label": r"Kinetic energy per nucleus $E_{\mathrm{kin}}$",
        "phrase": "kinetic energy per nucleus",
    },
    "rig": {
        "cls": GSFRigidity,
        "unit": "GV",
        "sym": "R",
        "label": r"Rigidity $R$",
        "phrase": "rigidity",
    },
    "en": {
        "cls": GSFEnergyPerNucleon,
        "unit": "GeV",
        "sym": r"E_{N}",
        "label": r"Total energy per nucleon $E_{N}$",
        "phrase": "total energy per nucleon",
    },
    "ekn": {
        "cls": GSFKineticEnergyPerNucleon,
        "unit": "GeV",
        "sym": r"E_{\mathrm{kin},N}",
        "label": r"Kinetic energy per nucleon $E_{\mathrm{kin},N}$",
        "phrase": "kinetic energy per nucleon",
    },
}

# One line per model version for the picker. DERIVED from the package registry
# rather than written out per name: a new revision (2026.1, ...) must not need
# an edit here, and the previous revision must not keep calling itself the
# default. Historical conference labels are spelled out.
_NOTE_OVERRIDES = {
    "2025": "(UHECR2024/ICRC2025)",
    "2019": "(ICRC2019)",
    "2017": "(ICRC2017)",
}


def _version_note(name, info):
    if name in _NOTE_OVERRIDES:
        return _NOTE_OVERRIDES[name]
    role = info.get("role", "")
    obsolete = role.startswith("superseded")
    if "single-interpretation" in role:
        return "(single mod. interpretation" + ("; obsolete)" if obsolete else ")")
    elif role == "alternative" or "alternative of" in role:
        note = "USO solar mod."
    elif obsolete:
        note = "first prerelease" if name == "2026.0" else "earlier prerelease"
    elif role == "default":
        note = "default"
    else:
        note = role or "historical release"
    return note + (" (obsolete)" if obsolete else "")


VERSION_NOTES = {
    name: _version_note(name, info) for name, info in MODEL_VERSIONS.items()
}

# The paper figures use Nimbus Roman text and custom mathtext;
# Paper figures use 0.75 pt group lines, 1.2 pt total, and 0.3-opacity bands.
# Bundle the font so desktop matplotlib and Pyodide render the same family.
PAPER_RC = {
    "font.family": "Nimbus Roman",
    "font.cursive": ["STIXGeneral"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Nimbus Roman",
    "mathtext.it": "Nimbus Roman:italic",
    "mathtext.bf": "Nimbus Roman:bold",
    "font.size": 10.0,
    "axes.labelsize": 10.0,
    "axes.linewidth": 0.8,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "pdf.fonttype": 42,
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "axes.grid": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.top": False,
    "ytick.right": False,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "0.8",
    "legend.framealpha": 0.8,
    "legend.fontsize": "small",
    "axes.formatter.use_mathtext": True,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "path",  # text as outlines: no font dependency in exports
}


@lru_cache(maxsize=1)
def _load_paper_fonts():
    from matplotlib import font_manager

    for variant in ("Regular", "Italic", "Bold", "BoldItalic"):
        font_manager.fontManager.addfont(
            str(Path(__file__).with_name("fonts") / f"NimbusRoman-{variant}.ttf")
        )


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
        kw = {"time_interval": time_interval, "rigidity_cutoff": rigidity_cutoff}
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
        jacobian += (
            model._element_flux_jacobian(sid, energy, ti)
            * (mass_scale * mask)[:, np.newaxis]
        )
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
    error = model._derived_error(weights, jacobians, leaders) if with_errors else None
    return value, error


def phi_year_range(model) -> tuple[int, int]:
    """First and last calendar year covered by the modulation-potential data."""
    years = sorted(model.phi.keys())
    return int(years[0]), int(years[-1])


def _sample_delta(model, n_samples: int, seed: int):
    """Shared amplitude draw for pseudo-experiments.

    One draw underlies every displayed curve (groups, elements, total), so
    each draw index is one internally consistent model realization; the seed
    is fixed by the caller so recomputes redraw the same experiments.
    """
    slices, factor = model._amplitude_sample_factor()
    rng = np.random.default_rng(seed)
    return slices, rng.standard_normal((n_samples, factor.shape[0])) @ factor.T


def _flux_samples(
    model,
    x,
    named_targets,
    time_interval,
    rigidity_cutoff,
    quantity,
    n_samples,
    seed,
    nucleus_cache,
    with_total,
):
    """Pseudo-experiment draws per displayed series, plus the total.

    Returns ``{series_name: (n_samples, n_x) array}`` with the all-particle
    (or all-nucleon) sum under ``"total"``. Reuses the jacobians the band
    evaluation already computed for the nucleus quantity.
    """
    slices, delta = _sample_delta(model, n_samples, seed)
    kw = {"time_interval": time_interval, "rigidity_cutoff": rigidity_cutoff}
    draws_by_target = {}

    def draws_for(target):
        key = target if not isinstance(target, list) else tuple(target)
        if key in draws_by_target:
            return draws_by_target[key]
        if quantity == "nucleus":
            if target not in nucleus_cache:
                nucleus_cache[target] = _nucleus_flux_jacobian(
                    model, x, target, time_interval, rigidity_cutoff
                )
            flux, jac = nucleus_cache[target]
        else:
            flux = np.asarray(model.flux(x, target, **kw), float)
            jac = np.asarray(model.jacobian(x, target, **kw), float)
        _zlist, leader = model._resolve_z(target)
        sid = model._leader_by_charge[leader]
        draws_by_target[key] = flux + delta[:, slices[sid]] @ jac.T
        return draws_by_target[key]

    out = {name: draws_for(target) for name, target in named_targets}
    if with_total:
        out["total"] = sum(draws_for(g) for g in GROUPS)
    return out


def _composition_samples(
    model, x, quantity, time_interval, rigidity_cutoff, n_samples, seed
):
    """Pseudo-experiment draws of a ln(A) moment (nonlinear, per draw)."""
    slices, delta = _sample_delta(model, n_samples, seed)
    denominator = np.zeros((n_samples, len(x)))
    first = np.zeros_like(denominator)
    second = np.zeros_like(denominator)
    for sid in model.species:
        flux, jac = _nucleus_flux_jacobian(
            model, x, sid, time_interval, rigidity_cutoff
        )
        leader_sid = model._leader_by_charge[model.z_ungroup[sid[0]]]
        draws = flux + delta[:, slices[leader_sid]] @ jac.T
        ln_a = np.log(model.mass_number[sid])
        denominator += draws
        first += ln_a * draws
        second += ln_a**2 * draws
    denominator = np.where(denominator > 0, denominator, np.nan)
    mean = first / denominator
    if quantity == "mean_lna":
        return mean
    return second / denominator - mean**2


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
    n_samples: int = 0,
    sample_seed: int = 0,
) -> dict:
    """Evaluate the selected flux or composition observable on a log grid.

    Returns plain numpy arrays only, so results are cache- and pickle-friendly.
    Zeros (below threshold / LIS at low energy) are kept as zeros; the plotting
    layer masks them. With ``n_samples > 0`` the result additionally carries
    ``out["samples"]``: pseudo-experiment draws per series (and ``"total"``)
    from one shared, seeded amplitude draw — each draw index is one
    internally consistent model realization.
    """
    validate_quantity_basis(quantity, basis)
    energy_scale = float(energy_scale)
    if not np.isfinite(energy_scale) or energy_scale <= 0:
        raise ValueError("energy-scale factor must be finite and positive")
    # The Explorer exposes a multiplicative factor (1 = unchanged), whereas
    # the package property stores a fractional shift (0 = unchanged).
    model.energy_scale = energy_scale - 1.0
    kw = {"time_interval": time_interval, "rigidity_cutoff": rigidity_cutoff}
    x = np.logspace(float(dmin), float(dmax), int(npts))
    out = {"x": x, "series": {}, "total": None, "quantity": quantity}

    if QUANTITIES[quantity]["kind"] == "composition":
        out["series"][quantity] = _composition(
            model, x, quantity, time_interval, rigidity_cutoff, with_errors
        )
        if n_samples:
            out["samples"] = {
                quantity: _composition_samples(
                    model,
                    x,
                    quantity,
                    time_interval,
                    rigidity_cutoff,
                    int(n_samples),
                    int(sample_seed),
                )
            }
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

    named_targets = []
    for g in groups:
        out["series"][g] = f_and_e(g)
        named_targets.append((g, g))
    for z in elements:
        # "D" (2026+ deuterium species) is a native flux/error target.
        name = "D" if z == "D" else ELEMENT_SYMBOLS.get(z, f"Z={z}")
        if name in out["series"]:
            continue  # e.g. element He duplicates the He group series
        out["series"][name] = f_and_e(z if z == "D" else int(z))
        named_targets.append((name, z if z == "D" else int(z)))
    # series name -> model target, for downstream per-series covariance
    out["targets"] = dict(named_targets)
    if n_samples:
        out["samples"] = _flux_samples(
            model,
            x,
            named_targets,
            time_interval,
            rigidity_cutoff,
            quantity,
            int(n_samples),
            int(sample_seed),
            nucleus_cache,
            with_total,
        )
    if with_total:
        if quantity == "nucleus":
            out["total"] = _total_from_groups(
                model,
                x,
                time_interval,
                rigidity_cutoff,
                with_errors,
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
    kw = {"time_interval": time_interval, "rigidity_cutoff": rigidity_cutoff}
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


def series_covariance(
    model, x, target, time_interval, rigidity_cutoff, quantity: str = "nucleus"
) -> np.ndarray:
    """Covariance matrix of one group's or element's flux on grid ``x``.

    The diagonal is the square of the ``err_`` column for the same series, so
    the pointwise sigma and the full covariance are exported consistently.
    """
    if quantity not in {"nucleus", "nucleon"}:
        raise ValueError("series covariance is defined only for flux quantities")
    if quantity == "nucleon":
        kw = {"time_interval": time_interval, "rigidity_cutoff": rigidity_cutoff}
        return np.asarray(model.covariance(target, target, x, **kw), float)
    _, jac = _nucleus_flux_jacobian(model, x, target, time_interval, rigidity_cutoff)
    block = model._covariance_block(target, target)
    if block is None:
        return np.zeros((len(x), len(x)))
    return model._propagate_cov(jac, jac, block)


def pair_covariance(
    model,
    x,
    target1,
    target2,
    time_interval,
    rigidity_cutoff,
    quantity: str = "nucleus",
) -> np.ndarray:
    """Cross-covariance matrix Cov(A(E1), B(E2)) of two series on grid ``x``.

    The diagonal of the returned matrix is the equal-energy covariance
    Cov(A(E), B(E)) between the two series (e.g. H* and He*).
    """
    if quantity not in {"nucleus", "nucleon"}:
        raise ValueError("pair covariance is defined only for flux quantities")
    if quantity == "nucleon":
        kw = {"time_interval": time_interval, "rigidity_cutoff": rigidity_cutoff}
        return np.asarray(model.covariance(target1, target2, x, **kw), float)
    jac1 = _nucleus_flux_jacobian(model, x, target1, time_interval, rigidity_cutoff)[1]
    jac2 = _nucleus_flux_jacobian(model, x, target2, time_interval, rigidity_cutoff)[1]
    block = model._covariance_block(target1, target2)
    if block is None:
        return np.zeros((len(x), len(x)))
    return model._propagate_cov(jac1, jac2, block)


# ---------------------------------------------------------------- figure


def _weighted(x, y, gamma):
    y = np.asarray(y, float)
    w = np.where(y > 0, y * x**gamma, np.nan)
    return w


def _gfmt(v: float) -> str:
    """Format an exponent without trailing zeros (2.7, 3, 2.65)."""
    return f"{v:g}"


def axis_labels(basis: str, gamma: float, quantity: str = "nucleus") -> tuple[str, str]:
    b = BASES[basis]
    xlab = rf"${b['sym']} / \mathrm{{{b['unit']}}}$"
    if quantity == "mean_lna":
        return xlab, r"$\langle\ln A\rangle$"
    if quantity == "var_lna":
        return xlab, r"$\sigma^2(\ln A)$"
    u = b["unit"]
    flux = r"J_\mathrm{n}" if quantity == "nucleon" else "J"
    ylab = rf"${flux} / (\mathrm{{{u}}}\,\mathrm{{m}}^2\,\mathrm{{s}}\,\mathrm{{sr}})^{{-1}}"
    if gamma:
        ylab += rf"\times ({b['sym']}/\mathrm{{{u}}})^{{{_gfmt(gamma)}}}"
    return xlab, ylab + "$"


MODEL_STYLES = ("-", ":", "-.")
MODEL_STYLE_NAMES = ("solid", "dotted", "dash-dot")


def fraction_result(model, result, ti, cutoff, with_errors=True):
    """Normalize each series with covariance propagation through its total."""
    x = result["x"]
    total, total_error = result["total"]
    denominator = np.where(total > 0, total, np.nan)
    quantity = result.get("quantity", "nucleus")

    def jacobian(target):
        if quantity == "nucleus":
            return _nucleus_flux_jacobian(model, x, target, ti, cutoff)[1]
        return model.jacobian(x, target, time_interval=ti, rigidity_cutoff=cutoff)

    group_jacs = {g: jacobian(g) for g in GROUPS} if with_errors else {}
    series = {}
    for name, (flux, error) in result["series"].items():
        value = flux / denominator
        sigma = None
        if with_errors and error is not None:
            target = result["targets"][name]
            jac = group_jacs[name] if name in group_jacs else jacobian(target)
            cross = np.zeros(len(x))
            for group, group_jac in group_jacs.items():
                block = model._covariance_block(target, group)
                if block is not None:
                    cross += np.einsum("ni,ij,nj->n", jac, block, group_jac)
            variance = error**2 + value**2 * total_error**2 - 2 * value * cross
            sigma = np.sqrt(np.clip(variance, 0, None)) / denominator
        series[name] = value, sigma
    return {**result, "series": series, "total": None, "ratio": True}


def _add_figure_caption(fig, text):
    """Add a measured, wrapped caption below the axes without shrinking them."""
    from matplotlib.font_manager import FontProperties

    symbols = {
        "E_kin,N": r"E_{\mathrm{kin},N}",
        "E_kin": r"E_{\mathrm{kin}}",
        "E_N": "E_N",
        "E": "E",
        "R": "R",
    }
    text = re.sub(
        r"\b(E_kin,N|E_kin|E_N|E|R)\^(-?\d+(?:\.\d+)?)",
        lambda match: f"${symbols[match[1]]}^{{{match[2]}}}$",
        text,
    )
    # The manuscript's math symbols use mathtext, not the text font's Unicode
    # coverage. Keep each expression together while wrapping the caption.
    for plain, math in {
        "⟨ln A⟩": r"$\langle\ln{A}\rangle$",
        "σ²(ln A)": r"$\sigma^2(\ln{A})$",
        "±1σ": r"$\pm1\sigma$",
    }.items():
        text = text.replace(plain, math)
    width, height = fig.get_size_inches()
    margin = 0.12
    fontsize = 8.5
    renderer = fig.canvas.get_renderer()
    font = FontProperties(family=PAPER_RC["font.family"], size=fontsize)
    max_width = (width - 2 * margin) * fig.dpi
    lines, line = [], ""
    for word in ("Figure. " + text).split():
        candidate = f"{line} {word}".strip()
        if (
            line
            and renderer.get_text_width_height_descent(
                candidate, font, "$" in candidate
            )[0]
            > max_width
        ):
            lines.append(line)
            line = word
        else:
            line = candidate
    lines.append(line)
    caption_height = (len(lines) * fontsize * 1.25) / 72 + 0.18
    positions = [ax.get_position().frozen() for ax in fig.axes]
    fig.set_size_inches(width, height + caption_height)
    for ax, pos in zip(fig.axes, positions, strict=True):
        ax.set_position(
            [
                pos.x0,
                (pos.y0 * height + caption_height) / (height + caption_height),
                pos.width,
                pos.height * height / (height + caption_height),
            ]
        )
    fig.text(
        margin / width,
        (caption_height - 0.06) / (height + caption_height),
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=fontsize,
        linespacing=1.25,
        color="black",
    )


def _comparison_legend(ax, handles, labels, versions):
    """One measured, wrapping key for model styles and component colors."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import (
        AnchoredOffsetbox,
        DrawingArea,
        HPacker,
        TextArea,
        VPacker,
    )
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.transforms import ScaledTranslation

    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    font = FontProperties(size=PAPER_RC["legend.fontsize"])
    size = font.get_size_in_points()
    pad, borderpad = 0.4, 0.6
    available = ax.bbox.width - 2 * (pad + borderpad) * size * fig.dpi / 72

    def entry(handle, label):
        sample = DrawingArea(2.4 * size, 0.75 * size)
        for artist in handle if isinstance(handle, tuple) else (handle,):
            if isinstance(artist, Patch):
                sample.add_artist(
                    Rectangle(
                        (0, 0),
                        sample.width,
                        sample.height,
                        facecolor=artist.get_facecolor(),
                        edgecolor="none",
                        alpha=artist.get_alpha(),
                    )
                )
            else:
                sample.add_artist(
                    Line2D(
                        [0, sample.width],
                        [sample.height / 2] * 2,
                        color=artist.get_color(),
                        linestyle=artist.get_linestyle(),
                        linewidth=artist.get_linewidth(),
                    )
                )
        box = HPacker(
            children=[sample, TextArea(label, textprops={"fontproperties": font})],
            align="center",
            pad=0,
            sep=0.3 * size,
        )
        box.set_figure(fig)
        return box

    def rows(entries):
        packed, row, width = [], [], 0
        gap = 0.9 * size * fig.dpi / 72
        for item in entries:
            item_width = item.get_window_extent(renderer).width
            if row and width + gap + item_width > available:
                packed.append(
                    HPacker(children=row, align="center", pad=0, sep=0.9 * size)
                )
                row, width = [], 0
            width += (gap if row else 0) + item_width
            row.append(item)
        if row:
            packed.append(HPacker(children=row, align="center", pad=0, sep=0.9 * size))
        return VPacker(children=packed, align="left", pad=0, sep=0.35 * size)

    model_keys = rows(
        entry(Line2D([], [], color="black", ls=MODEL_STYLES[i], lw=1.2), f"GSF {v}")
        for i, v in enumerate(versions)
    )
    component_keys = rows(entry(handles[name], labels[name]) for name in handles)
    legend = AnchoredOffsetbox(
        loc="lower right",
        child=VPacker(
            children=[model_keys, component_keys], align="left", pad=0, sep=0.6 * size
        ),
        prop=font,
        pad=pad,
        borderpad=0,
        frameon=True,
        bbox_to_anchor=(1, 1),
        bbox_transform=ax.transAxes,
    )
    legend.set_gid("comparison-legend")
    legend.set_zorder(10)
    legend.patch.set(facecolor="white", edgecolor="0.8", alpha=0.8)
    legend.patch.set_boxstyle("round,pad=0,rounding_size=0.2")
    ax.add_artist(legend)

    # Keep the scientific multiplier beside the header when there is room,
    # otherwise reserve a small separate gap above it.
    gap = 4 / 72
    offset = ax.yaxis.get_offset_text()
    if offset.get_visible() and offset.get_text():
        bounds = offset.get_window_extent(renderer)
        if bounds.x1 + 4 * fig.dpi / 72 > legend.get_window_extent(renderer).x0:
            gap += max(0, bounds.y1 - ax.bbox.y1) / fig.dpi
    legend.set_bbox_to_anchor(
        (1, 1),
        transform=ax.transAxes + ScaledTranslation(0, gap, fig.dpi_scale_trans),
    )
    return legend


def make_figure(
    result: dict | list[dict],
    basis: str,
    gamma: float,
    *,
    versions: list[str] | None = None,
    show_total: bool = True,
    show_bands: bool = True,
    overlay_bands: bool = False,
    band_alpha: float = 0.3,
    line_weight: float = 1.0,
    ylog: bool = True,
    y_range=None,
    grid: bool = False,
    width_in: float = 6.0,
    height_in: float = 3.6,
    caption_text: str = "",
):
    """Publication figure with shared component colors and model line styles."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import LogLocator, NullFormatter

    _load_paper_fonts()
    results = [result] if isinstance(result, dict) else result
    if not 1 <= len(results) <= len(MODEL_STYLES):
        raise ValueError("a publication figure needs one to three models")
    if versions is not None and len(versions) != len(results):
        raise ValueError("one version label is required per model")
    quantity = results[0].get("quantity", "nucleus")
    is_composition = QUANTITIES[quantity]["kind"] == "composition"
    ratio = results[0].get("ratio", False)
    gamma = 0 if is_composition or ratio else gamma
    ylog = ylog and not is_composition
    with matplotlib.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=110)
        means, fallback_means, handles, labels = [], [], {}, {}
        for mi, res in enumerate(results):
            x = res["x"]
            curves = dict(res["series"])
            if show_total and res["total"] is not None and not ratio:
                curves = {**curves, "all": res["total"]}
            ecolors = iter(ELEMENT_COLORS * 5)
            for name, (flux, error) in curves.items():
                element = name not in GROUP_COLORS and not is_composition
                color = next(ecolors) if element else GROUP_COLORS.get(name, "black")
                style = "--" if element else MODEL_STYLES[mi]
                if len(results) == 1 and name in GROUP_STYLES:
                    style = GROUP_STYLES[name]
                label = (
                    QUANTITIES[quantity]["label"]
                    if is_composition
                    else ("all-nucleon" if quantity == "nucleon" else "all-particle")
                    if name == "all"
                    else name
                )
                width = (1.2 if name == "all" or is_composition else 0.75) * line_weight
                value = np.asarray(flux) * x**gamma
                if not is_composition:
                    value = np.where(np.asarray(flux) > 0, value, np.nan)
                # Keep below-validity zero fluxes as gaps, as on the canvas.
                value = np.where(value > 0, value, np.nan) if ylog else value
                (line,) = ax.plot(
                    x,
                    value,
                    color=color,
                    ls=style,
                    lw=width,
                    label=label,
                    zorder=4 if element else 6,
                )
                handles.setdefault(name, line)
                labels.setdefault(name, label)
                (fallback_means if element else means).append(value)
                if (
                    show_bands
                    and (mi == 0 or overlay_bands)
                    and not element
                    and error is not None
                ):
                    lo, hi = (flux - error) * x**gamma, (flux + error) * x**gamma
                    if ylog:
                        lo = np.where(
                            hi > 0, np.maximum(lo, np.finfo(float).tiny), np.nan
                        )
                    kw = {"color": color, "alpha": band_alpha, "lw": 0}
                    if mi:
                        kw = {
                            "facecolor": "none",
                            "edgecolor": color,
                            "hatch": "///" if mi == 1 else "\\\\\\",
                            "alpha": min(band_alpha * 2.5, 0.8),
                            "lw": 0,
                        }
                    ax.fill_between(x, lo, hi, zorder=1, **kw)
                    if mi == 0:
                        handles[name] = (
                            Patch(facecolor=color, alpha=band_alpha, edgecolor="none"),
                            Line2D([], [], color=color, ls=style, lw=width),
                        )

        ax.set_xscale("log")
        tick_count = 20 if width_in >= 5 else 8
        ax.xaxis.set_major_locator(LogLocator(numticks=tick_count))
        ax.xaxis.set_minor_locator(LogLocator(subs="auto", numticks=tick_count))
        ax.xaxis.set_minor_formatter(NullFormatter())
        values = np.concatenate(means or fallback_means or [np.array([])])
        values = values[np.isfinite(values) & (values > 0)]
        if ylog:
            ax.set_yscale("log")
            if values.size:
                ax.set_ylim(values.min() / 2.2, values.max() * 2.2)
        elif values.size:
            ax.set_ylim(0, values.max() * 1.06)
        if y_range is not None:
            ax.set_ylim(*y_range)
        ax.set_xlim(results[0]["x"][0], results[0]["x"][-1])
        xlab, ylab = axis_labels(basis, gamma, quantity)
        if ratio:
            ylab = r"$J_i / J_{\mathrm{total}}$"
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        if grid:
            ax.grid(alpha=0.25, lw=0.5, which="major")
        if handles and not (versions and len(results) > 1):
            ax.legend(
                handles=list(handles.values()),
                labels=list(labels.values()),
                handler_map={tuple: HandlerTuple(ndivide=1)},
                ncol=min(len(handles), 5 if width_in >= 5 else 3),
                handlelength=1.9,
                labelspacing=0.35,
                columnspacing=0.7,
                handletextpad=0.3,
            )
        fig.tight_layout(pad=0.4)
        if handles and versions and len(results) > 1:
            fig.canvas.draw()
            _comparison_legend(ax, handles, labels, versions)
            fig.tight_layout(pad=0.4)
        if caption_text:
            _add_figure_caption(fig, caption_text)
    return fig


def figure_svg(fig) -> str:
    """Serialize a figure to responsive SVG (fixed pixel size stripped)."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.04)
    svg = buf.getvalue()
    svg = svg[svg.find("<svg") :]
    svg = re.sub(r'(<svg[^>]*?)\swidth="[^"]*"', r"\1", svg, count=1)
    svg = re.sub(r'(<svg[^>]*?)\sheight="[^"]*"', r"\1", svg, count=1)
    return svg


def figure_bytes(fig, fmt: str, dpi: int = 300) -> bytes:
    buf = io.BytesIO()
    import matplotlib

    with matplotlib.rc_context(PAPER_RC):
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


def caption(
    version: str | list[str],
    basis: str,
    gamma: float,
    ti,
    elements: list[int | str],
    show_bands: bool,
    rigidity_cutoff: float | None,
    quantity: str = "nucleus",
    *,
    groups=None,
    show_total: bool = True,
    overlay_bands: bool = False,
    ratio: bool = False,
    energy_scale: float = 1.0,
) -> str:
    """Describe only the selected observable, curves, and uncertainty display."""
    versions = [version] if isinstance(version, str) else version
    names = [v if v.startswith("GSF") else f"GSF {v}" for v in versions]
    groups = GROUPS if groups is None else groups
    composition = QUANTITIES[quantity]["kind"] == "composition"
    if composition:
        observable = (
            "Mean logarithmic mass ⟨ln A⟩"
            if quantity == "mean_lna"
            else "Variance of the logarithmic mass σ²(ln A)"
        )
    elif ratio:
        observable = (
            "Fractions of the "
            + ("nucleon" if quantity == "nucleon" else "nucleus")
            + " flux"
        )
    else:
        observable = (
            "Cosmic-ray "
            + ("nucleon" if quantity == "nucleon" else "nucleus")
            + " flux"
        )
    parts = [f"{observable} as a function of {BASES[basis]['phrase']}."]
    if len(names) > 1:
        parts.append(
            "Curves show "
            + "; ".join(f"{n} ({MODEL_STYLE_NAMES[i]})" for i, n in enumerate(names))
            + "."
        )
    else:
        parts.append(f"Curves show {names[0]}.")
    if not composition:
        selected = (
            ["all-nucleon" if quantity == "nucleon" else "all-particle"]
            if show_total and not ratio
            else []
        ) + list(groups)
        if selected:
            parts.append("Components: " + ", ".join(selected) + ".")
        if gamma and not ratio:
            sym = {
                "etot": "E",
                "ekin": "E_kin",
                "rig": "R",
                "en": "E_N",
                "ekn": "E_kin,N",
            }[basis]
            parts.append(f"Fluxes are scaled by {sym}^{_gfmt(gamma)}.")
    parts.append(f"Solar modulation: {modulation_phrase(ti)}.")
    if show_bands and (composition or groups or (show_total and not ratio)):
        parts.append(
            "Shaded bands show ±1σ uncertainties propagated from the fit covariance"
            + (", including the correlation with the total flux." if ratio else ".")
        )
        if len(names) > 1:
            parts.append(
                "The primary model has filled bands; compared models have hatched bands."
                if overlay_bands
                else f"Bands are shown for {names[0]} only."
            )
    if elements and not composition:
        syms = ", ".join(ELEMENT_SYMBOLS.get(z, str(z)) for z in elements)
        parts.append(
            f"Dashed curves show individual elements {syms}"
            + (f" from {names[0]}" if len(names) > 1 else "")
            + "; their uncertainty bands are omitted."
        )
    if rigidity_cutoff:
        parts.append(f"A geomagnetic cutoff of {rigidity_cutoff:g} GV is applied.")
    if energy_scale != 1:
        parts.append(f"The energy-scale factor is {energy_scale:g}.")
    return " ".join(parts)


def build_csv(
    model,
    result: dict,
    basis: str,
    version: str,
    ti,
    rigidity_cutoff,
    include_cov: bool,
    quantity: str = "nucleus",
    include_series_cov: bool = False,
    include_cross_cov: bool = False,
) -> str:
    """CSV export: provenance header, flux table, optional covariance blocks.

    ``include_cov`` appends the N x N covariance of the total flux;
    ``include_series_cov`` additionally appends one N x N block per exported
    series (each group and each individual element), so single-element
    uncertainties are available as pointwise sigma AND full covariance.
    ``include_cross_cov`` appends one N x N cross-covariance block per
    unordered pair of mass groups (Cov(A(E1), B(E2)); the diagonal is the
    equal-energy cross-group covariance).
    """
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
        suffix = (
            ""
            if quantity in {"mean_lna", "var_lna"}
            else f"_{name.replace('*', 'star')}"
        )
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

    import globalsplinefit

    lines = [
        f"# Global Spline Fit (GSF) — parameter set {version} "
        "(physical model version incl. revision)",
        f"# code: globalsplinefit {globalsplinefit.__version__}",
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
        lines.append(
            f"# covariance of {prefix}_total on the grid above "
            f"({len(x)}x{len(x)}, row-major)"
        )
        lines += [",".join(f"{v:.4e}" for v in row) for row in cov]
    if include_series_cov and QUANTITIES[quantity]["kind"] == "flux":
        targets = result.get("targets", {})
        for name in result["series"]:
            if name not in targets:
                continue
            cov = series_covariance(
                model, x, targets[name], ti, rigidity_cutoff, quantity
            )
            lines.append("#")
            lines.append(
                f"# covariance of {prefix}_{name.replace('*', 'star')} "
                f"on the grid above ({len(x)}x{len(x)}, row-major)"
            )
            lines += [",".join(f"{v:.4e}" for v in row) for row in cov]
    if include_cross_cov and QUANTITIES[quantity]["kind"] == "flux":
        targets = result.get("targets", {})
        exported = [g for g in GROUPS if g in targets]
        for ia, ga in enumerate(exported):
            for gb in exported[ia + 1 :]:
                cov = pair_covariance(
                    model, x, targets[ga], targets[gb], ti, rigidity_cutoff, quantity
                )
                lines.append("#")
                lines.append(
                    f"# cross-covariance {prefix}_{ga.replace('*', 'star')} x "
                    f"{prefix}_{gb.replace('*', 'star')} on the grid above "
                    f"({len(x)}x{len(x)}, row-major; diagonal = equal-energy "
                    f"cross-group covariance)"
                )
                lines += [",".join(f"{v:.4e}" for v in row) for row in cov]
    return "\n".join(lines) + "\n"


__all__ = [
    "BASES",
    "QUANTITIES",
    "NUCLEON_BASES",
    "GROUPS",
    "GROUP_COLORS",
    "ELEMENT_SYMBOLS",
    "VERSION_NOTES",
    "MODEL_VERSIONS",
    "version_info",
    "make_model",
    "phi_year_range",
    "validate_quantity_basis",
    "evaluate",
    "total_covariance",
    "series_covariance",
    "pair_covariance",
    "make_figure",
    "figure_svg",
    "figure_bytes",
    "axis_labels",
    "caption",
    "modulation_phrase",
    "build_csv",
]
