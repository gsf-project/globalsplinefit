"""Pyodide-side bridge for GSF Explorer.

Thin JSON adapter between the web worker RPC and ``gsf_explorer`` (the pure
model/figure/CSV core shared with headless use). All entry points take and
return JSON strings so no proxy objects cross the JS boundary.
"""

import base64
import json
import math

import gsf_explorer as gx

_models = {}


def _numbers(values):
    """Convert an array to strict-JSON numbers, using null for invalid points."""
    return [float(value) if math.isfinite(value) else None for value in values]


def _draws(matrix):
    """Sample draws as compact JSON rows (4 significant digits — well below
    line width on screen, and it halves the payload of large ensembles)."""
    return [
        [float(f"{v:.4g}") if math.isfinite(v) else None for v in row]
        for row in matrix
    ]


def _phi_bins(v):
    """JSON phi-bin spec -> solar_cycle_average_bins (0/None/"full" -> None)."""
    if v is None or v == "full" or v == 0:
        return None
    return int(v)


def _model(basis, version, phi_bins=12):
    """Cached model for a basis+version, retuned to the requested phi binning.

    The bin count is read per flux call rather than baked in at construction, so
    one instance per (basis, version) serves every setting -- keeping the wheel's
    data out of memory N times over and avoiding a model rebuild (seconds, in
    WASM) when the knob moves.
    """
    key = (basis, version)
    if key not in _models:
        _models[key] = gx.make_model(basis, version, phi_bins)
    model = _models[key]
    model.solar_cycle_average_bins = phi_bins
    return model


def _element_entries(m, basis):  # noqa: ARG001 - basis kept for RPC signature
    """Selectable elements; multi-isotope Z=1 exposes deuterium separately."""
    entries = []
    for z in sorted(int(z) for z in m.z_to_sids):
        entries.append({"z": z, "sym": gx.ELEMENT_SYMBOLS.get(z, str(z))})
        if z == 1 and len(m.z_to_sids[1]) > 1:
            entries.append({"z": "D", "sym": "D"})
    return entries


def _ti(mod):
    """JSON modulation spec -> gsf time_interval argument."""
    if mod is None or mod == "SC24":
        return None
    if mod == "LIS":
        return "LIS"
    return (int(mod[0]), int(mod[1]))


def meta(basis, version=None):
    """App metadata. ``version=None`` (or an unknown name, e.g. after a
    package rename) falls back to the package's first registered version, so
    the page adapts to model renames/additions without code changes."""
    versions = list(gx.MODEL_VERSIONS)
    if version not in versions:
        version = versions[0]
    m = _model(basis, version)
    y0, y1 = gx.phi_year_range(m)
    return json.dumps({
        "default": version,
        "versions": versions,
        "notes": gx.VERSION_NOTES,
        "bases": {k: {"unit": b["unit"], "sym": b["sym"], "phrase": b["phrase"]}
                  for k, b in gx.BASES.items()},
        "quantities": {
            key: {
                "label": value["ui_label"],
                "kind": value["kind"],
                "bases": (
                    [basis for basis in gx.BASES if basis in gx.NUCLEON_BASES]
                    if key == "nucleon"
                    else list(gx.BASES)
                ),
            }
            for key, value in gx.QUANTITIES.items()
        },
        "groups": gx.GROUPS,
        "elements": _element_entries(m, basis),
        "phiYears": [y0, y1],
    })


def _params(p):
    return dict(
        model=_model(p["basis"], p["version"], _phi_bins(p.get("phiBins", 12))),
        basis=p["basis"],
        dmin=p["dmin"], dmax=p["dmax"], npts=p["npts"],
        groups=[g for g in gx.GROUPS if g in p["groups"]],
        elements=[z if z == "D" else int(z) for z in p["elements"]],
        time_interval=_ti(p["mod"]),
        rigidity_cutoff=(float(p["cutoff"]) if p.get("cutoff") else None),
        energy_scale=float(p.get("escale", 1.0)),
        quantity=p.get("quantity", "nucleus"),
        n_samples=int(p.get("samples", 0) or 0),
        sample_seed=int(p.get("sampleSeed", 0) or 0),
    )


def _evaluate(p):
    q = _params(p)
    model = q.pop("model")
    return model, gx.evaluate(model, **q)


def evaluate(params_json):
    p = json.loads(params_json)
    _, res = _evaluate(p)
    samples = res.get("samples") or {}
    series = [
        {"name": name, "flux": _numbers(f),
         "err": (_numbers(e) if e is not None else None),
         "samples": (_draws(samples[name]) if name in samples else None)}
        for name, (f, e) in res["series"].items()
    ]
    total = None
    if res["total"] is not None:
        tf, te = res["total"]
        total = {"flux": _numbers(tf),
                 "err": (_numbers(te) if te is not None else None),
                 "samples": (_draws(samples["total"])
                             if "total" in samples else None)}
    return json.dumps({
        "x": _numbers(res["x"]),
        "series": series,
        "total": total,
        "quantity": p.get("quantity", "nucleus"),
        "quantityLabel": gx.QUANTITIES[p.get("quantity", "nucleus")]["ui_label"],
        "caption": gx.caption(
            p["version"], p["basis"], p.get("gamma", 0), _ti(p["mod"]),
            [z if z == "D" else int(z) for z in p["elements"]], True,
            (float(p["cutoff"]) if p.get("cutoff") else None),
            p.get("quantity", "nucleus")),
        "modulation": gx.modulation_phrase(_ti(p["mod"])),
    })


def csv(params_json, include_cov):
    p = json.loads(params_json)
    p["samples"] = 0  # exports never need pseudo-experiment draws
    model, res = _evaluate(p)
    return gx.build_csv(model, res, p["basis"], p["version"], _ti(p["mod"]),
                        (float(p["cutoff"]) if p.get("cutoff") else None),
                        bool(include_cov), p.get("quantity", "nucleus"))


def figure(params_json, opts_json):
    """Publication-style matplotlib figure (paper rc) -> base64 bytes."""
    p = json.loads(params_json)
    o = json.loads(opts_json)
    p["samples"] = 0  # exports never need pseudo-experiment draws
    _, res = _evaluate(p)
    fig = gx.make_figure(
        res, p["basis"], float(o.get("gamma", 2.7)),
        show_total=bool(o.get("showTotal", True)),
        show_bands=bool(o.get("showBands", True)),
        band_alpha=float(o.get("bandAlpha", 0.18)),
        ylog=bool(o.get("ylog", True)),
        grid=bool(o.get("grid", False)),
        width_in=float(o.get("widthIn", 7.0)),
        height_in=float(o.get("heightIn", 4.5)),
    )
    data = gx.figure_bytes(fig, o.get("fmt", "pdf"), dpi=int(o.get("dpi", 300)))
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(data).decode()


def about(version):
    return json.dumps(gx.version_info(version), default=str)
