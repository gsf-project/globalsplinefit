import marimo

__generated_with = "0.23.15"
app = marimo.App(
    width="full",
    app_title="GSF — Paper Figures",
    css_file="gallery.css",
    html_head_file="gallery_head.html",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GSF 2026 — the paper figures

    The figures of the GSF 2026 paper that the model draws on its own, in
    paper order, at the paper's figure sizes, weightings, axis ranges, tick
    locators and colours — ported from the plotting code of the GSF fitter,
    which is not public.

    **The measurements are left out.** They belong to the experiments that
    published them; the paper is where they are shown together with the fit.
    Everything else is the paper's recipe: the model is drawn as the fitted
    **local interstellar spectrum** ($J_\mathrm{LIS}$, no solar modulation),
    against **kinetic energy** or rigidity as each figure requires.

    Two typographic families are in play, exactly as in the paper: the deck
    figures are set in a Times-like serif, the stand-alone figures in the
    stock sans-serif. Each cell sets its own, so the order the notebook runs
    in cannot change the result.

    Pick a parameter set below — every figure redraws.
    """)
    return


@app.cell
async def _():
    import sys

    NOTEBOOK = "paper_figures"  # used by the download buttons at the end
    IN_WASM = "pyodide" in sys.modules
    SITE = ""

    if IN_WASM:
        # WASM (docs gallery): install the wheel built together with these
        # docs and published under /wheels/ on the same site, so the notebook
        # always runs the documented version (works on any host).
        from urllib.parse import urljoin

        import marimo as _mo
        import micropip

        # notebook_location() resolution depth differs between marimo export
        # layouts — try site-root /wheels/ from every plausible depth, and
        # keep the depth that worked as the site root for other assets.
        _base = str(_mo.notebook_location()) + "/"
        _err = None
        for _up in ("", "../", "../../", "../../../"):
            try:
                await micropip.install(
                    urljoin(
                        _base,
                        _up + "wheels/globalsplinefit-2.0.1-py3-none-any.whl",
                    )
                )
                SITE = urljoin(_base, _up)
                _err = None
                break
            except Exception as _e:  # noqa: BLE001 — 404 lands as generic error
                _err = _e
        if _err is not None:
            raise _err

        # crflux (H3a, GST, poly-gonato, ...) for the comparison figures — a
        # pure python wheel, so micropip takes it straight from PyPI
        await micropip.install("crflux")

    import crflux.models as cm
    import matplotlib.pyplot as plt
    import numpy as np

    from globalsplinefit import (
        MODEL_VERSIONS,
        GSFEnergy,
        GSFEnergyPerNucleon,
        GSFKineticEnergy,
        GSFKineticEnergyPerNucleon,
        GSFRigidity,
    )

    # ------------------------------------------------------------ raster dpi --
    # These figsizes are paper column widths (3.4-3.8 in), so marimo's
    # `figsize x 100` display rule makes them the smallest plots in the
    # gallery; gallery.css scales every figure up to a common width. dpi does
    # not affect the display size (marimo normalises it away) — only the pixel
    # count — so raise it here to keep the upscaled figure sharp: at 150 a
    # 3.4 in figure rasterises 1020 px wide for a 900 px box. Do NOT express
    # the size change as a bigger figsize: that would rescale the fonts
    # against the axes and break the match to the published figures.
    plt.rcParams["figure.dpi"] = 150

    # ---------------------------------------------------------------- fonts --
    # The paper's figures come from two families of scripts and they do NOT
    # share a font. The fitter's figure deck asks for Nimbus Roman (else a Times-like
    # serif); the stand-alone scripts (paper_figures/*, the SM run scripts)
    # never touch the font and get matplotlib's stock DejaVu Sans. Neither
    # Nimbus Roman nor Times exists in pyodide, but matplotlib bundles
    # STIXGeneral — a Times metric clone — so the browser gets the paper's
    # letterforms. Every plot cell opens `plt.rc_context(...)` with the family
    # its figure was drawn in, so cell execution order cannot leak a font.
    RC_SERIF = {
        "font.family": "serif",
        "font.serif": [
            "Nimbus Roman",
            "Times New Roman",
            "STIXGeneral",
            "DejaVu Serif",
            "serif",
        ],
        "mathtext.fontset": "stix",
    }
    RC_SANS = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "dejavusans",
    }
    # the size block shared by the smaller paper and SM figures
    RC_SMALL = {
        "font.size": 8.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
    }

    # ------------------------------------------- the deck's drawing vocabulary
    COLORS = ("r", "y", "g", "b")  # style.colors
    LINES = ("-", "--", "-.", ":")  # style.lines
    LABELS = ("H*", "He*", "O*", "Fe*")  # style.labels
    ELEM_LABELS = ("$p$", "He", "O", "Fe")  # config.prim_label of the leaders
    GROUPS = ("H*", "He", "O*", "Fe*")  # the model's mass-GROUP target keys
    LEADER_Z = (1, 2, 8, 26)
    FIGSIZE_SMALL = (3.8, 3.2)  # config.figsize_small
    FIGSIZE_WIDE = (6, 3.6)  # config.figsize_wide
    LIGHT_COLOR = (1.0, 0.4, 0.0)  # style.light_color
    HEAVY_COLOR = (0.0, 0.7, 0.7)  # style.heavy_color

    SYMBOLS = (
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
    )

    def species_name(sid):
        """Return the deck's species name.

        The element symbol, but 'p' for H and 'D' for the deuteron, which
        carries its own spline under the proton group.
        """
        z, a = int(sid[0]), round(float(sid[1]))
        if z == 1:
            return "D" if a == 2 else "p"
        return SYMBOLS[z - 1]

    def show(fig=None):
        """Return a figure as the cell output.

        marimo renders a cell's last expression; `plt.show()` produces no
        output at all in app view, so every plot cell ends with `show(fig)`.
        """
        fig = plt.gcf() if fig is None else fig
        return fig

    def draw_band(ax, x, val, err, color, line=None, lw=None, alpha=0.3, **kw):
        """Model central line (optional) + translucent ±1σ band."""
        if line is not None:
            ax.plot(x, val, line, color=color, lw=lw)
        return ax.fill_between(
            x, val - err, val + err, color=color, alpha=alpha, linewidth=0, **kw
        )

    def band_line_handle(color, ls="-", lw=1.2, alpha=0.3):
        """ONE legend key showing the band with its line on top."""
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        return (
            Patch(facecolor=color, alpha=alpha, edgecolor="none", zorder=1),
            Line2D([], [], color=color, ls=ls, lw=lw, zorder=2),
        )

    def logx_ticks(ax, numticks=20):
        """Set the dense log-axis tick locators used on every panel."""
        from matplotlib.ticker import LogLocator

        ax.xaxis.set_major_locator(LogLocator(numticks=numticks))
        ax.xaxis.set_minor_locator(LogLocator(subs="auto", numticks=numticks))

    def cornertext(ax, text, loc="upper left", fontsize=None):
        """Anchored corner label (pyik.mplext.cornertext)."""
        from matplotlib.offsetbox import AnchoredText

        codes = {
            "upper left": 2,
            "upper right": 1,
            "lower left": 3,
            "lower right": 4,
        }
        at = AnchoredText(
            text,
            loc=codes.get(loc, 2),
            frameon=False,
            prop={
                "size": fontsize if fontsize is not None else plt.rcParams["font.size"]
            },
        )
        ax.add_artist(at)
        return at

    def pair_error(model, x, a, b):
        """Return the band for the sum of two groups.

        sqrt(sum_ij Cov(g_i, g_j)) over the light (p+He) or heavy (O*+Fe*)
        pair.
        """
        return np.sqrt(
            np.clip(
                sum(
                    np.diag(np.asarray(model.covariance(_a, _b, x), float))
                    for _a in (a, b)
                    for _b in (a, b)
                ),
                0.0,
                None,
            )
        )

    def gsf_label(version):
        """Return the figure label for a parameter-set version.

        The paper writes the current fit as 'GSF2026'; keep that verbatim for
        the released set and fall back to the version string otherwise.
        """
        return "GSF2026" if version == "2026.1" else f"GSF {version}"

    # ------------------------------------------------------- data-free tables --
    # Fig. 3 draws each species over the rigidity interval its DIRECT data
    # covers, not over its knot range — so without the data the extent has to
    # come from somewhere. These are the (min, max) measured rigidities per
    # species in the GSF2026 fit, i.e. the coverage of the published spectra,
    # not the measurements themselves. Read once off the fitter's data zoo
    # (the fitter's plotting code builds the same dict at draw time).
    DIRECT_RANGE_GV = {
        "p": (1.64192, 529880),
        "D": (2.78832, 20.8802),
        "He": (1.86055, 424948),
        "Li": (2.88908, 2071.87),
        "Be": (2.87991, 2071.87),
        "B": (1.51943, 11997),
        "C": (1.49172, 84028.8),
        "N": (1.49135, 2071.87),
        "O": (1.53236, 84028.8),
        "F": (1.5842, 1897.96),
        "Ne": (1.58835, 1897.96),
        "Na": (1.62407, 1897.98),
        "Mg": (1.59205, 1897.96),
        "Al": (1.6169, 1897.98),
        "Si": (1.64413, 1897.96),
        "P": (1.67553, 71.917),
        "S": (1.64304, 1897.96),
        "Cl": (1.68648, 72.6246),
        "Ar": (1.75753, 77.2329),
        "K": (1.74014, 71.6722),
        "Ca": (1.70923, 69.8164),
        "Sc": (1.78756, 74.5279),
        "Ti": (1.80754, 445.819),
        "V": (1.82979, 77.0785),
        "Cr": (1.80225, 443.922),
        "Mn": (1.81993, 76.4821),
        "Fe": (3.53461, 50888.9),
        "Co": (1.90252, 75.9722),
        "Ni": (1.84893, 394.116),
    }

    # Solar-system abundances of Lodders, ApJ 591, 1220 (2003), summed over the
    # isotopes of each element (Si = 10^6 in the source normalisation). SM S2
    # overlays these; they are a published reference table, so they travel with
    # the notebook rather than with the fitter.
    LODDERS_2003 = (
        2.43105e10,
        2.34339e9,
        55.47,
        0.7374,
        17.32,
        7.07942e6,
        1.95014e6,
        1.41335e7,
        841.1,
        2.14759e6,
        5.7510e4,
        1.02000e6,
        8.4100e4,
        1.0e6,
        8373.0,
        4.44866e5,
        5237.0,
        1.02500e5,
        3691.93,
        6.28704e4,
        34.2,
        2422.0,
        288.4,
        1.2860e4,
        9168.0,
        8.38000e5,
        2323.0,
        4.7797e4,
    )

    return (
        COLORS,
        DIRECT_RANGE_GV,
        ELEM_LABELS,
        FIGSIZE_SMALL,
        FIGSIZE_WIDE,
        GROUPS,
        GSFEnergy,
        GSFEnergyPerNucleon,
        GSFKineticEnergy,
        GSFKineticEnergyPerNucleon,
        GSFRigidity,
        HEAVY_COLOR,
        LABELS,
        LEADER_Z,
        LIGHT_COLOR,
        LINES,
        LODDERS_2003,
        MODEL_VERSIONS,
        NOTEBOOK,
        RC_SANS,
        RC_SERIF,
        RC_SMALL,
        SITE,
        SYMBOLS,
        band_line_handle,
        cm,
        cornertext,
        draw_band,
        gsf_label,
        logx_ticks,
        pair_error,
        np,
        plt,
        show,
        species_name,
    )


@app.cell(hide_code=True)
def _(MODEL_VERSIONS, mo):
    version = mo.ui.dropdown(
        options=list(MODEL_VERSIONS), value="2026.1", label="parameter set"
    )
    plot_size = mo.ui.slider(
        start=0.6,
        stop=1.6,
        step=0.1,
        value=1.0,
        label="plot size",
        show_value=True,
    )
    mo.hstack([version, plot_size], justify="start", gap=2)
    return plot_size, version


@app.cell(hide_code=True)
def _(mo, plot_size):
    # Drives --gsf-plot-scale, which gallery.css multiplies into every figure's
    # width. Scaling the rendered raster (rather than the figsize) is what keeps
    # it proportional: fonts, line widths and the aspect ratio all ride along.
    mo.Html(f"<style>:root{{--gsf-plot-scale:{plot_size.value}}}</style>")
    return


@app.cell
def _(
    GSFEnergy,
    GSFEnergyPerNucleon,
    GSFKineticEnergy,
    GSFKineticEnergyPerNucleon,
    GSFRigidity,
    gsf_label,
    version,
):
    # The paper draws EVERY model curve as the fitted LIS (the fitter pins
    # default_time_interval="LIS" on all three model classes), against kinetic energy per nucleus, rigidity, or kinetic
    # energy per nucleon as the figure requires.
    gsf_k = GSFKineticEnergy(version=version.value, default_time_interval="LIS")
    gsf_r = GSFRigidity(version=version.value, default_time_interval="LIS")
    gsf_kn = GSFKineticEnergyPerNucleon(
        version=version.value, default_time_interval="LIS"
    )
    # The comparison figures (9, 10, 12) and the SM covariance figures work in
    # TOTAL energy and construct their models without a time interval, which is
    # the frame the external parametrizations are defined in — kept verbatim.
    gsf_e = GSFEnergy(version=version.value)
    gsf_en = GSFEnergyPerNucleon(version=version.value)
    VLAB = gsf_label(version.value)
    return VLAB, gsf_e, gsf_en, gsf_k, gsf_kn, gsf_r


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fig. 1 — Elements and mass groups

    A group is named after its leading element; groups divide the $\ln A$ axis
    into four roughly equal parts. The area of each circle is proportional to
    the flux ratio of the element to its group leader (values tabulated in the
    SM).
    """)
    return


@app.cell
def _(
    FIGSIZE_SMALL,
    LEADER_Z,
    RC_SERIF,
    gsf_r,
    np,
    plt,
    show,
    species_name,
):
    # fitter plotting code: mass_groups_lnA_vs_Z
    with plt.rc_context(RC_SERIF):
        _fig, _ax = plt.subplots(figsize=FIGSIZE_SMALL)
        for _sid in sorted(gsf_r.species, key=lambda s: (s[0], s[1])):
            _z, _a = _sid
            _lead = gsf_r.flux_ratio[_sid][0]
            if _sid != _lead:  # ratio at the species' last knot, as in the deck
                _rmax = float(np.exp(np.asarray(gsf_r.kx[_sid], float)[-1]))
                _num = float(np.asarray(gsf_r.flux(_rmax, _sid)).ravel()[0])
                _den = float(np.asarray(gsf_r.flux(_rmax, _lead)).ravel()[0])
                _ratio = _num / _den if _den > 0 else 0.0
            else:
                _ratio = 1.0
            _col = "rygb"[LEADER_Z.index(_lead[0])]
            _name = species_name(_sid)
            _ax.plot([np.log(_a)], [_z], "o" + _col, ms=0.3 + 6 * _ratio**0.5)
            if _name == "p":
                _dx, _dy = -0.02, 1.4
            elif _name == "He":
                _dx, _dy = -0.24, 1.0
            elif _name in ("Ne", "Mg"):
                _dx, _dy = -0.31, 0.5
            elif _name in ("C", "O", "Si", "Fe"):
                _dx, _dy = -0.23, 0.5
            elif _name in ("Be", "S", "K", "Ca", "Ti", "Cr", "Ni"):
                _dx, _dy = -0.20, 0.5
            else:
                _dx, _dy = 0.05, -0.6
            _ax.text(
                np.log(_a) + _dx,
                _z + _dy,
                _name,
                va="center",
                fontsize="x-small",
                color=_col,
            )
        _ax.text(0.44, 30.55, "H*", color="r")
        _ax.text(1.38, 30.5, "He*", color="y")
        _ax.text(2.37, 30.5, "O*", color="g")
        _ax.text(3.45, 30.5, "Fe*", color="b")
        _ax.axvspan(0, 1, alpha=0.1, color="r")
        _ax.axvspan(1, 1.93, alpha=0.1, color="y")
        _ax.axvspan(1.93, 3, alpha=0.1, color="g")
        _ax.axvspan(3, 4.1, alpha=0.1, color="b")
        _ax.set_ylabel(r"Z")
        _ax.set_xlabel(r"$\ln A$")
        _ax.set_xlim(-0.1, 4.5)
        _ax.set_ylim(0, 30)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fig. 2 — Groups against their leading elements

    Flux of the oxygen (top) and iron (bottom) groups (thick lines) compared to
    the flux of the leading elements alone (dashed) and of the sub-leading
    elements (thin lines). The offset between group and leading element
    underlines the role of the sub-leading abundances.
    """)
    return


@app.cell
def _(
    FIGSIZE_SMALL,
    RC_SERIF,
    cornertext,
    gsf_k,
    gsf_r,
    logx_ticks,
    np,
    plt,
    show,
):
    # fitter plotting code: flux_from_group_components (one figure per group)
    def group_panel(leader_z, group_key, tag, color):
        with plt.rc_context(RC_SERIF):
            _fig, _ax = plt.subplots(figsize=FIGSIZE_SMALL)
            _fig.subplots_adjust(left=0.14, right=0.95, top=0.97, bottom=0.12)
            _lsid = gsf_r.z_to_sids[leader_z][0]
            _kn = np.exp(np.asarray(gsf_r.kx[_lsid], float))
            _e = np.geomspace(_kn[0], _kn[-1], 1000)
            _sc = _e**3.0
            _ax.plot(
                _e,
                np.asarray(gsf_k.flux(_e, group_key), float) * _sc,
                "-",
                color=color,
                lw=3,
            )
            for _sid in gsf_r.species:
                if gsf_r.flux_ratio[_sid][0] != _lsid:
                    continue
                _is_leader = _sid == _lsid
                _ax.plot(
                    _e,
                    np.asarray(gsf_k.flux(_e, _sid), float) * _sc,
                    "--" if _is_leader else "-",
                    color=color,
                    lw=2 if _is_leader else 1,
                    alpha=1.0 if _is_leader else 0.2,
                )
            _ax.loglog()
            _ax.set_xlabel(r"$E_\mathrm{kin} / \mathrm{GeV}$")
            _ax.set_ylabel(
                r"$J_\mathrm{LIS} / (\mathrm{GeV}\,\mathrm{m}^2\,\mathrm{s}\,\mathrm{sr})^{-1}"
                r"\times (E_\mathrm{kin}/\mathrm{GeV})^{3}$"
            )
            cornertext(_ax, tag, loc="upper right")
            logx_ticks(_ax)
            _ax.set_xlim(5, 10**11.5)
            _ax.set_ylim(1e2, 1e7)
        return _fig

    show(group_panel(8, "O*", "O*", "g"))
    return (group_panel,)


@app.cell
def _(group_panel, show):
    show(group_panel(26, "Fe*", "Fe*", "b"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fig. 3 — The leading elements versus rigidity

    The frame of the direct measurements: fitted LIS rigidity spectra of the
    leading elements ($\pm1\sigma$ band) and of the sub-leading members of the
    oxygen and iron groups (pale); the deuteron, carried as a sub-leading
    species of the proton group, is dimmed. Each curve spans the rigidity
    range its direct measurements cover, as in the paper. In the paper the
    data are drawn on top, each on its fitted energy scale.
    """)
    return


@app.cell
def _(
    COLORS,
    DIRECT_RANGE_GV,
    FIGSIZE_SMALL,
    LEADER_Z,
    RC_SERIF,
    draw_band,
    gsf_r,
    logx_ticks,
    np,
    plt,
    show,
    species_name,
):
    # fitter plotting code: flux_direct (show_data=False)
    def _direct_color(z, leader_z):
        _i = LEADER_Z.index(leader_z)
        if leader_z in (1, 2) or z == leader_z:
            return COLORS[_i]
        return (0.7, 0.9, 0.7) if leader_z == 8 else (0.7, 0.7, 1.0)

    with plt.rc_context(RC_SERIF):
        _fig, _ax = plt.subplots(figsize=FIGSIZE_SMALL)
        _fig.subplots_adjust(left=0.17, bottom=0.12, right=0.95, top=0.85)
        for _sid in sorted(gsf_r.species, key=lambda s: (s[0], s[1])):
            _lsid = gsf_r.flux_ratio[_sid][0]
            _name = species_name(_sid)
            if _name not in DIRECT_RANGE_GV:
                continue
            _a, _b = DIRECT_RANGE_GV[_name]
            _col = _direct_color(_sid[0], _lsid[0])
            _r = np.geomspace(_a, _b, 1000)
            _sc = _r**2.6
            _val = np.asarray(gsf_r.flux(_r, _sid), float) * _sc
            _ax.plot(
                _r,
                _val,
                "-",
                color=_col,
                zorder=11 if _sid == _lsid else 6,
                alpha=0.4 if _name == "D" else 1.0,
            )
            if _sid == _lsid:
                _err = np.asarray(gsf_r.error(_r, _sid), float) * _sc
                draw_band(_ax, _r, _val, _err, _col, alpha=0.1, zorder=0)
        _ax.text(1200, 9000, "$p$", color=COLORS[0], zorder=10)
        _ax.text(1200, 4e2, "He", color=COLORS[1], zorder=10)
        _ax.text(8e2, 61, "O", color=COLORS[2], zorder=10)
        _ax.text(4e2, 10, "Fe", color=COLORS[3], zorder=10)
        _ax.loglog()
        _ax.set_xlabel(r"$R / \mathrm{GV}$")
        _ax.set_ylabel(
            r"$J_\mathrm{LIS} / (\mathrm{GV}\,\mathrm{m}^2\,\mathrm{s}\,\mathrm{sr})^{-1}"
            r"\times (R/\mathrm{GV})^{2.6}$"
        )
        logx_ticks(_ax)
        _ax.set_xlim(1, 1e6)
        _ax.set_ylim(1e-2, 2e4)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fig. 5 — Mass-group fractions

    Fractions of the four mass groups as a function of energy, $\pm1\sigma$.
    In the paper each panel carries the composition-sensitive air-shower
    measurements, adjusted to the common energy scale, and the two lower
    panels a note about summing elemental flux points — both are data
    annotations and are left out here.
    """)
    return


@app.cell
def _(
    COLORS,
    FIGSIZE_WIDE,
    GROUPS,
    LABELS,
    RC_SERIF,
    draw_band,
    gsf_k,
    logx_ticks,
    np,
    plt,
    show,
):
    # fitter plotting code: fractions
    with plt.rc_context(RC_SERIF):
        _fig, _axes = plt.subplots(4, 1, figsize=FIGSIZE_WIDE)
        _fig.subplots_adjust(hspace=0, right=0.97, left=0.1, top=0.91, bottom=0.12)
        _e = np.geomspace(1, 5e11, 1000)
        _all = [np.asarray(gsf_k.flux(_e, _g), float) for _g in GROUPS]
        for _i, _g in enumerate(GROUPS):
            _ax = _axes[_i]
            _val = _all[_i] / np.sum(_all, axis=0)
            _err = np.asarray(gsf_k.fraction_error(_e, _g), float)
            draw_band(_ax, _e, _val, _err, COLORS[_i], line="-", lw=1, zorder=0)
            _ax.set_ylabel(LABELS[_i])
            if _i == 0:
                _ax.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
                _ax.set_yticklabels(("$0$", "$1/4$", "$1/2$", "$3/4$", "$1$"))
            else:
                _ax.set_yticks((0, 0.25, 0.5, 0.75))
                _ax.set_yticklabels(("$0$", "$1/4$", "$1/2$", "$3/4$"))
            _ax.set_ylim(0, 1)
            _ax.semilogx()
            if _i != 3:
                _ax.tick_params(labelbottom=False, bottom=False)
            logx_ticks(_ax)
            _ax.set_xlim(1e0, 1e11)
        _axes[3].set_xlabel(r"$E_\mathrm{kin} / \mathrm{GeV}$")
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fig. 6 (top) — The cosmic-ray flux, ~1 GeV to $10^{11}$ GeV

    All-particle and mass-group fluxes with their $1\sigma$ uncertainties,
    weighted by $E_\mathrm{kin}^{2.6}$, with the elemental oxygen and iron
    spectra over the range their direct data cover and the light ($p+$He)
    component drawn over the range the paper centres it on. In the paper this
    panel carries the direct measurements of the leading elements and the
    air-shower measurements of the all-particle flux and of the mass groups,
    all adjusted to the common energy scale found by the fit and demodulated
    to the LIS — the frame in which the model is drawn.
    """)
    return


@app.cell
def _(
    COLORS,
    ELEM_LABELS,
    FIGSIZE_WIDE,
    GROUPS,
    LABELS,
    LIGHT_COLOR,
    LINES,
    RC_SERIF,
    band_line_handle,
    draw_band,
    gsf_k,
    logx_ticks,
    np,
    pair_error,
    plt,
    show,
):
    # fitter plotting code: flux_log (energy_exponent=2.6, show_composition=4,
    # show_heavy=False), as the deck registry calls it for cr_flux_traditional
    from matplotlib.legend_handler import HandlerTuple as _HandlerTuple1

    with plt.rc_context(RC_SERIF):
        _fig, _ax = plt.subplots(figsize=FIGSIZE_WIDE, facecolor="white")
        _fig.subplots_adjust(left=0.1, bottom=0.15, right=0.97, top=0.97)
        _e = np.geomspace(1, 5e11, 1000)
        _sc = _e**2.6
        _keys = []
        for _i, _g in enumerate(GROUPS):
            _val = np.asarray(gsf_k.flux(_e, _g), float) * _sc
            _err = np.asarray(gsf_k.error(_e, _g), float) * _sc
            draw_band(_ax, _e, _val, _err, COLORS[_i], line=LINES[_i], lw=0.75)
            _keys.append(
                (band_line_handle(COLORS[_i], ls=LINES[_i], lw=0.75), LABELS[_i])
            )
            # the single ELEMENT oxygen / iron spectra over their data range
            if _i in (2, 3):
                _e1 = np.geomspace(1, 10**5.5 if _i == 2 else 10**5.8, 200)
                _ax.plot(
                    _e1,
                    np.asarray(gsf_k.flux(_e1, ELEM_LABELS[_i]), float) * _e1**2.6,
                    LINES[_i],
                    color=COLORS[_i],
                    lw=0.75,
                )
        # light component (p + He), drawn over the range the paper centres it on
        _mask = (_e > 2e3) & (_e < 2e5)
        _light = sum(np.asarray(gsf_k.flux(_e, _g), float) for _g in GROUPS[:2]) * _sc
        _lerr = pair_error(gsf_k, _e, "H*", "He") * _sc
        _ax.plot(_e, np.where(_mask, _light, np.nan), "-", color=LIGHT_COLOR, lw=0.75)
        draw_band(_ax, _e[_mask], _light[_mask], _lerr[_mask], LIGHT_COLOR, zorder=0)

        _tot = np.asarray(gsf_k.total_flux(_e), float) * _sc
        _terr = np.asarray(gsf_k.total_error(_e), float) * _sc
        _ax.plot(_e, _tot, "k-", lw=1.2)
        draw_band(_ax, _e, _tot, _terr, "k", zorder=0)
        _keys.append((band_line_handle("k", lw=1.2), "total"))

        _ax.loglog()
        _ax.set_xlabel(r"$E_\mathrm{kin} / \mathrm{GeV}$")
        _ax.set_ylabel(
            r"$J_\mathrm{LIS} / (\mathrm{GeV}\,\mathrm{m}^2\,\mathrm{s}\,\mathrm{sr})^{-1}"
            r"\times (E_\mathrm{kin}/\mathrm{GeV})^{2.6}$"
        )
        _ax.legend(
            handles=[_h for _h, _ in _keys],
            labels=[_l for _, _l in _keys],
            handler_map={tuple: _HandlerTuple1(ndivide=1)},
            loc="lower left",
            ncol=len(_keys),
            columnspacing=0.7,
            fontsize="small",
            handletextpad=0.3,
            bbox_to_anchor=(0.15, 0.08, 1.0, 0.8),
        )
        logx_ticks(_ax)
        _ax.set_xlim(1e0, 5e11)
        _ax.set_ylim(5e0, 3e5)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fig. 6 (bottom) — The knee-to-suppression region

    The same fluxes, scaled by $E_\mathrm{kin}^{3}$ and drawn with a **linear**
    flux axis, which emphasizes the structures between the knee and the
    suppression.
    """)
    return


@app.cell
def _(
    COLORS,
    FIGSIZE_WIDE,
    GROUPS,
    LINES,
    RC_SERIF,
    draw_band,
    gsf_k,
    logx_ticks,
    np,
    plt,
    show,
):
    # fitter plotting code: flux_linear (energy_exponent=3, linear y axis)
    from matplotlib.ticker import EngFormatter as _EngFormatter1

    with plt.rc_context(RC_SERIF):
        _fig, _ax = plt.subplots(figsize=FIGSIZE_WIDE)
        _fig.subplots_adjust(left=0.1, bottom=0.15, right=0.97, top=0.97)
        _e = np.geomspace(1e1, 5e11, 1000)
        _sc = _e**3.0
        for _i, _g in enumerate(GROUPS):
            _val = np.asarray(gsf_k.flux(_e, _g), float) * _sc
            _err = np.asarray(gsf_k.error(_e, _g), float) * _sc
            draw_band(_ax, _e, _val, _err, COLORS[_i], line=LINES[_i], lw=0.75)
        _tot = np.asarray(gsf_k.total_flux(_e), float) * _sc
        _terr = np.asarray(gsf_k.total_error(_e), float) * _sc
        draw_band(_ax, _e, _tot, _terr, "k", line="-", lw=1.2, zorder=0)
        _ax.semilogx()
        _ax.yaxis.set_major_formatter(_EngFormatter1())
        _ax.set_xlabel(r"$E_\mathrm{kin} / \mathrm{GeV}$")
        _ax.set_ylabel(
            r"$J_\mathrm{LIS} / (\mathrm{GeV}\,\mathrm{m}^2\,\mathrm{s}\,\mathrm{sr})^{-1}"
            r"\times (E_\mathrm{kin}/\mathrm{GeV})^{3}$"
        )
        logx_ticks(_ax)
        _ax.set_xlim(1e3, 10**11.5)
        _ax.set_ylim(0, 6e6)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fig. 8 — Mean logarithmic mass and its variance

    $\langle \ln A \rangle$ (top) and $\sigma^2(\ln A)$ (bottom) computed from
    the model. The dashed lines mark the four group leaders (top) and the
    variance of an equal mix of protons with each of them (bottom). In the
    paper the $\langle \ln A \rangle$ data used in the fit and the Auger
    $X_\mathrm{max}$ moments are overlaid.
    """)
    return


@app.cell
def _(COLORS, RC_SERIF, draw_band, gsf_k, np, plt, show):
    # fitter plotting code: lna_moments (the stacked two-panel figure = the paper figure)
    from matplotlib.ticker import LogLocator as _LogLocator1
    from matplotlib.ticker import NullFormatter as _NullFormatter1

    with plt.rc_context(RC_SERIF):
        _fig, _axes = plt.subplots(2, 1, figsize=(3.4, 4.0), sharex=True)
        _fig.subplots_adjust(hspace=0.1, left=0.15, right=0.85)
        _e = np.geomspace(1, 1e12, 200)
        _lna_main = [
            float(np.log(gsf_k.mass_number[gsf_k.z_to_sids[_z][0]]))
            for _z in (1, 2, 8, 26)
        ]
        _m = np.asarray(gsf_k.mean_lnA(_e), float)
        _me = np.asarray(gsf_k.mean_lnA_error(_e), float)
        _v = np.asarray(gsf_k.var_lnA(_e), float)
        _ve = np.asarray(gsf_k.var_lnA_error(_e), float)

        draw_band(_axes[0], _e, _m, _me, "k", line="-", lw=1.0)
        draw_band(_axes[1], _e, _v, _ve, "k", line="-", lw=1.0)
        for _j, _lna in enumerate(_lna_main):
            _axes[0].axhline(_lna, ls="--", color=COLORS[_j], zorder=0)
            _axes[0].text(3e11, _lna, ("$p$", "He", "O", "Fe")[_j], va="center")
            if _j == 0:
                continue
            _var = 0.5 * _lna**2 - (0.5 * _lna) ** 2
            _axes[1].axhline(_var, ls="--", color=COLORS[_j], zorder=0)
            _axes[1].text(
                3e11,
                _var,
                (
                    "",
                    r"$\frac{1}{2}$$p$,$\frac{1}{2}$He",
                    r"$\frac{1}{2}$$p$,$\frac{1}{2}$O",
                    r"$\frac{1}{2}$$p$,$\frac{1}{2}$Fe",
                )[_j],
                va="center",
            )
        _axes[0].set_ylabel(r"$\langle \ln A \rangle$")
        _axes[0].set_ylim(-0.02, _lna_main[-1] + 0.05)
        _axes[1].set_ylabel(r"$\sigma^2(\ln A)$")
        _axes[1].set_ylim(0, 4.2)
        _axes[1].set_xlabel(r"$E_\mathrm{kin} / \mathrm{GeV}$")
        for _ax in _axes:  # label every second decade
            _ax.set_xscale("log")
            _ax.xaxis.set_major_locator(_LogLocator1(base=100))
            _ax.xaxis.set_minor_locator(_LogLocator1(base=100, subs=(10.0,)))
            _ax.xaxis.set_minor_formatter(_NullFormatter1())
            _ax.set_xlim(5, 2e11)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fig. 9 — Mass groups against other parametrizations

    The GSF 2026 fit (solid, filled $\pm1\sigma$) against the earlier GSF 2019
    fit (dashed, hatched band) and the parametrizations of Gaisser (H3a),
    Gaisser–Stanev–Tilav (GST-4) and Lv et al. (2024) (drawn only above its
    100 GeV all-particle validity floor). Rows: all-particle flux, H\*, He\*,
    O\* and Fe\*, and $\langle\ln A\rangle$. Left: spectra weighted by
    $E^{2.7}$; right: ratio to the GSF 2026. Each external composition is
    summed into the four GSF leading groups using the fit's own charge→leader
    map, and its $\langle\ln A\rangle$ uses the true mass of each component.
    """)
    return


@app.cell
def _(
    RC_SERIF,
    RC_SMALL,
    VLAB,
    cm,
    gsf_e,
    np,
    plt,
    show,
):
    # fitter plotting code, verbatim
    from matplotlib.legend_handler import HandlerTuple as _HandlerTuple2
    from matplotlib.lines import Line2D as _Line2D2
    from matplotlib.patches import Patch as _Patch2

    from globalsplinefit import GSFEnergy as _GSFEnergy2

    _DASHDOTDOT = (0, (3, 1.2, 1, 1.2, 1, 1.2))
    # the palette this figure was drawn with (globalsplinefit.plotting.COLORS)
    _MG = {"H*": "#d62728", "He*": "#ff7f0e", "O*": "#2ca02c", "Fe*": "#1f77b4"}
    _RATIO_YLIM = (0.5, 1.7)
    _RPAD = 0.5 * (_RATIO_YLIM[1] - _RATIO_YLIM[0])

    # Lv et al. (2024), ApJ 979, 225 — a five-component power-law model with
    # rigidity-dependent cutoffs. It ships with the fitter, not with crflux, so
    # the parametrization is carried here as a crflux PrimaryFlux subclass
    # (transcribed from the arXiv source).
    _P, _HE, _C, _N, _O, _NE, _MG12, _SI, _FE, _Z53 = (
        14,
        402,
        1206,
        1407,
        1608,
        2010,
        2412,
        2814,
        5626,
        12753,
    )
    _POP1 = {
        _P: (4.5e-5, 2.90),
        _HE: (4.4e-5, 2.81),
        _C: (1.1e-5, 2.77),
        _O: (2.1e-5, 2.77),
        _NE: (5.0e-6, 2.75),
        _MG12: (8.3e-6, 2.75),
        _SI: (9.4e-6, 2.75),
        _FE: (2.9e-5, 2.75),
    }
    _LS = {
        _P: (2.3e-4, 2.77),
        _HE: (1.5e-4, 2.45),
        _C: (3.2e-5, 2.45),
        _O: (4.5e-5, 2.45),
        _NE: (1.5e-5, 2.45),
        _MG12: (1.7e-5, 2.45),
        _SI: (2.0e-5, 2.45),
        _FE: (1.8e-5, 2.45),
    }
    _POP2 = {
        _P: (2.8e-5, 2.45),
        _HE: (2.1e-5, 2.45),
        _C: (2.4e-6, 2.45),
        _O: (3.1e-6, 2.45),
        _NE: (6.0e-7, 2.45),
        _MG12: (7.8e-7, 2.45),
        _SI: (8.3e-7, 2.45),
        _FE: (1.6e-6, 2.38),
        _Z53: (1.5e-8, 2.10),
    }
    _XGLE = {_P: (8.0e-19, 2.67), _HE: (8.5e-19, 2.67)}
    _XGHE = {_HE: (3.1e-20, 2.0), _N: (1.7e-20, 2.0)}
    _TEV, _100PEV = 1.0e3, 1.0e8
    _EC_POP1, _EC_LS, _EC_POP2 = 3.0e4, 3.6e4, 4.8e6
    _EC_XGLE, _EL_XGLE = 1.0e9, 1.9e7
    _EC_XGHE, _EL_XGHE = 4.0e9, 1.5e9
    _PC, _YR = 3.0857e18, 3.1557e7
    _KSUP = (500.0 * _PC) ** 2 / (4.0 * 5.0e27 * 2.0e5 * _YR)
    _DELTA = 0.33

    class LvEtAl2024(cm.PrimaryFlux):
        """Lv et al. (2024) LHAASO-anchored multi-component flux model."""

        def __init__(self, model_config="default", geomagnetic_cutoff=None):
            super().__init__(geomagnetic_cutoff=geomagnetic_cutoff)
            self.name = "Lv et al. (2024)"
            self.sname = "Lv24"
            self.model_config = model_config
            _ids = set()
            for _tab in (_POP1, _LS, _POP2, _XGLE, _XGHE):
                _ids.update(_tab)
            self.nucleus_ids = sorted(_ids)

        @staticmethod
        def _plaw(E, phi0, gamma, pivot, Z, ec):
            return phi0 * (E / pivot) ** (-gamma) * np.exp(-E / (Z * ec))

        @staticmethod
        def _sech(x):
            return 1.0 / np.cosh(np.minimum(x, 700.0))

        def _nucleus_flux(self, corsika_id, E):
            E = np.atleast_1d(np.asarray(E, dtype=float))
            Z, _ = self.Z_A(corsika_id)
            flux = np.zeros_like(E)
            if corsika_id in _POP1:
                _phi0, _g = _POP1[corsika_id]
                flux += self._plaw(E, _phi0, _g, _TEV, Z, _EC_POP1)
            if corsika_id in _LS:
                _phi0, _g = _LS[corsika_id]
                _sup = np.exp(-_KSUP * ((E / Z) / 4.0) ** (-_DELTA))
                flux += self._plaw(E, _phi0, _g, _TEV, Z, _EC_LS) * _sup
            if corsika_id in _POP2:
                _phi0, _g = _POP2[corsika_id]
                flux += self._plaw(E, _phi0, _g, _TEV, Z, _EC_POP2)
            if corsika_id in _XGLE:
                _phi0, _g = _XGLE[corsika_id]
                flux += self._plaw(E, _phi0, _g, _100PEV, Z, _EC_XGLE) * self._sech(
                    Z * _EL_XGLE / E
                )
            if corsika_id in _XGHE:
                _phi0, _g = _XGHE[corsika_id]
                flux += self._plaw(E, _phi0, _g, _100PEV, Z, _EC_XGHE) * self._sech(
                    Z * _EL_XGHE / E
                )
            return flux

    _E = np.logspace(np.log10(5.0), np.log10(5e11), 500)
    _w = _E**2.7
    _rows = [
        ("all-particle", None, None, "k", "0.55", "flux"),
        ("H*", "H*", 1, _MG["H*"], _MG["H*"], "flux"),
        ("He*", "He", 2, _MG["He*"], _MG["He*"], "flux"),
        ("O*", "O*", 8, _MG["O*"], _MG["O*"], "flux"),
        ("Fe*", "Fe*", 26, _MG["Fe*"], _MG["Fe*"], "flux"),
        (r"$\langle\ln A\rangle$", None, None, "k", "0.55", "lna"),
    ]
    _model_ctors = [
        (lambda: cm.HillasGaisser2012("H3a"), "H3a", ":", None),
        (lambda: cm.GaisserStanevTilav("4-gen"), "GST-4", "-.", None),
        (LvEtAl2024, "Lv et al. 2024", _DASHDOTDOT, 100.0),
    ]
    _hist = _GSFEnergy2(version="2019")
    _zungroup = gsf_e.z_ungroup

    def _safe_ratio(num, den):
        _r = np.full_like(den, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            _ok = den > 0
            _r[_ok] = num[_ok] / den[_ok]
        return _r

    def _clip_ratio(y):
        return np.clip(y, _RATIO_YLIM[0] - _RPAD, _RATIO_YLIM[1] + _RPAD)

    def _model_group_flux(pf, leader):
        _out = np.zeros_like(_E)
        for _cid in pf.nucleus_ids:
            if _zungroup.get(pf.Z_A(_cid)[0], 26) == leader:
                _out += pf.nucleus_flux(_cid, _E)
        return _out

    def _model_mean_lnA(pf):
        _num, _den = np.zeros_like(_E), np.zeros_like(_E)
        for _cid in pf.nucleus_ids:
            _f = pf.nucleus_flux(_cid, _E)
            _num += _f * np.log(pf.Z_A(_cid)[1])
            _den += _f
        return _safe_ratio(_num, _den)

    def _gsf_quantity(model, kind, target):
        if kind == "lna":
            return (
                np.asarray(model.mean_lnA(_E), float),
                np.asarray(model.mean_lnA_error(_E), float),
            )
        if target is None:
            return (
                np.asarray(model.total_flux(_E), float),
                np.asarray(model.total_error(_E), float),
            )
        return (
            np.asarray(model.flux(_E, target), float),
            np.asarray(model.error(_E, target), float),
        )

    def _crflux_quantity(pf, kind, leader, target):
        if kind == "lna":
            return _model_mean_lnA(pf)
        if target is None:
            return np.asarray(pf.total_flux(_E), float)
        return _model_group_flux(pf, leader)

    def _hist_band(ax, x, lo, hi, col):
        ax.fill_between(
            x,
            lo,
            hi,
            facecolor="none",
            edgecolor=col,
            hatch="///",
            linewidth=0.0,
            alpha=0.55,
            zorder=2.4,
        )
        ax.plot(x, lo, color=col, ls="-", lw=0.5, alpha=0.6, zorder=2.5)
        ax.plot(x, hi, color=col, ls="-", lw=0.5, alpha=0.6, zorder=2.5)

    _models = [(ctor(), tag, ls, floor) for ctor, tag, ls, floor in _model_ctors]

    with plt.rc_context({**RC_SERIF, **RC_SMALL}):
        _fig, _axes = plt.subplots(
            6,
            2,
            figsize=(6.9, 7.2),
            sharex=True,
            gridspec_kw={"hspace": 0.09, "wspace": 0.06},
        )
        for _i, (_label, _target, _leader, _col, _bandcol, _kind) in enumerate(_rows):
            _axL, _axR = _axes[_i, 0], _axes[_i, 1]
            _weight = 1.0 if _kind == "lna" else _w
            _c26, _e26 = _gsf_quantity(gsf_e, _kind, _target)
            _axL.fill_between(
                _E,
                _weight * (_c26 - _e26),
                _weight * (_c26 + _e26),
                color=_bandcol,
                alpha=0.30,
                lw=0,
                zorder=1,
            )
            _rel26 = _safe_ratio(_e26, _c26)
            _axR.fill_between(
                _E, 1 - _rel26, 1 + _rel26, color=_bandcol, alpha=0.30, lw=0, zorder=1
            )
            _axR.axhline(1, color="k", lw=0.8, zorder=0)
            _cH, _eH = _gsf_quantity(_hist, _kind, _target)
            _hist_band(_axL, _E, _weight * (_cH - _eH), _weight * (_cH + _eH), _col)
            _hist_band(
                _axR,
                _E,
                _clip_ratio(_safe_ratio(_cH - _eH, _c26)),
                _clip_ratio(_safe_ratio(_cH + _eH, _c26)),
                _col,
            )
            _axL.plot(_E, _weight * _c26, color=_col, ls="-", lw=1.7, zorder=5)
            _axL.plot(_E, _weight * _cH, color=_col, ls="--", lw=1.2, zorder=4)
            _axR.plot(_E, _safe_ratio(_cH, _c26), color=_col, ls="--", lw=1.2, zorder=4)
            for _pf, _tag, _ls, _floor in _models:
                _y = _crflux_quantity(_pf, _kind, _leader, _target)
                _m = np.ones_like(_E, dtype=bool) if _floor is None else (_floor <= _E)
                _axL.plot(
                    _E[_m], (_weight * _y)[_m], color=_col, ls=_ls, lw=1.1, zorder=3
                )
                _axR.plot(
                    _E[_m],
                    _safe_ratio(_y, _c26)[_m],
                    color=_col,
                    ls=_ls,
                    lw=1.1,
                    zorder=3,
                )
            _axL.set_xscale("log")
            if _kind == "flux":
                _axL.set_yscale("log")
            _axL.text(
                0.035,
                0.94,
                _label,
                transform=_axL.transAxes,
                fontsize=10,
                fontweight="bold",
                color=_col,
                va="top",
                ha="left",
            )
            _axR.text(
                0.035,
                0.94,
                _label,
                transform=_axR.transAxes,
                fontsize=9,
                fontweight="bold",
                color=_col,
                va="top",
                ha="left",
            )
            _axR.yaxis.tick_right()
            _axR.yaxis.set_label_position("right")
            _axR.tick_params(labelright=True)
            _axR.set_ylim(*_RATIO_YLIM)
            if _kind == "lna":
                _axL.set_ylim(0.0, 4.2)
            elif _target is None:
                _axL.set_ylim(2e2, 2e5)
            else:
                _axL.set_ylim(1.5e2, 5e4)
        _axes[0, 0].set_xlim(5.0, 5e11)
        _axes[2, 0].set_ylabel(
            r"$E^{2.7} J(E)$  [GeV$^{1.7}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]"
        )
        _axes[5, 0].set_ylabel(r"$\langle\ln A\rangle$")
        _axes[2, 1].set_ylabel(f"model / {VLAB}")
        _axes[5, 0].set_xlabel(r"$E$  [GeV]")
        _axes[5, 1].set_xlabel(r"$E$  [GeV]")
        _handles = [
            (
                _Patch2(facecolor="0.55", alpha=0.5),
                _Line2D2([0], [0], color="k", ls="-", lw=1.6),
            ),
            (
                _Patch2(facecolor="none", edgecolor="k", hatch="///"),
                _Line2D2([0], [0], color="k", ls="--", lw=1.2),
            ),
            _Line2D2([0], [0], color="k", ls=":", lw=1.1),
            _Line2D2([0], [0], color="k", ls="-.", lw=1.1),
            _Line2D2([0], [0], color="k", ls=_DASHDOTDOT, lw=1.1),
        ]
        _fig.legend(
            _handles,
            [VLAB, "GSF2019", "H3a", "GST-4", "Lv et al. 2024"],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=5,
            frameon=False,
            fontsize=8,
            handlelength=2.2,
            columnspacing=1.3,
            handletextpad=0.5,
            handler_map={tuple: _HandlerTuple2(ndivide=1)},
        )
        _fig.subplots_adjust(top=0.955)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fig. 10 — All-particle flux against common parametrizations

    The GSF all-particle flux (black, $\pm1\sigma$) compared with the
    broken-power-law parametrizations in common use, and with the earlier GSF
    2019 fit (hatched). Bottom: ratio to the GSF. Zatsepin–Sokolskaya is
    truncated at $5\times10^{7}$ GeV and Gaisser–Honda at $10^{7}$ GeV, the
    limits of their validity.
    """)
    return


@app.cell
def _(GSFEnergy, RC_SERIF, RC_SMALL, VLAB, cm, gsf_e, np, plt, show):
    # fitter plotting code: _two_panel, all-particle
    from matplotlib.legend_handler import HandlerTuple

    MODEL_CTORS = [
        (lambda: cm.HillasGaisser2012("H3a"), "H3a"),
        (lambda: cm.HillasGaisser2012("H4a"), "H4a"),
        (lambda: cm.GaisserStanevTilav("4-gen"), "GST (4-gen)"),
        (lambda: cm.PolyGonato(), "poly-gonato"),
        (lambda: cm.GaisserHonda(), "Gaisser-Honda"),
        (lambda: cm.ZatsepinSokolskaya("default"), "Zatsepin-Sokolskaya"),
    ]
    TRUNC = {"Zatsepin-Sokolskaya": 5e7, "Gaisser-Honda": 1e7}
    OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]
    HIST_COL = "0.40"

    def two_panel(
        x, gsf_flux, gsf_err, model_curves, xlabel, ylabel, ylim1, ylim2, hist, wexp=2.7
    ):
        _fig, (_ax1, _ax2) = plt.subplots(
            2,
            1,
            figsize=(3.4, 4.2),
            sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.07},
        )
        _w = x**wexp
        _band = _ax1.fill_between(
            x,
            _w * (gsf_flux - gsf_err),
            _w * (gsf_flux + gsf_err),
            color="0.55",
            alpha=0.5,
            lw=0,
        )
        (_line,) = _ax1.loglog(x, _w * gsf_flux, color="k", lw=1.6)
        _handles, _labels = [(_band, _line)], [VLAB]
        _yh, _eh = hist
        _b19 = _ax1.fill_between(
            x,
            _w * (_yh - _eh),
            _w * (_yh + _eh),
            facecolor="none",
            hatch="////",
            edgecolor=HIST_COL,
            lw=0,
            alpha=0.9,
        )
        (_l19,) = _ax1.loglog(x, _w * _yh, color=HIST_COL, lw=1.3, ls="--")
        _ax2.fill_between(
            x,
            (_yh - _eh) / gsf_flux,
            (_yh + _eh) / gsf_flux,
            facecolor="none",
            hatch="////",
            edgecolor=HIST_COL,
            lw=0,
            alpha=0.9,
        )
        _ax2.semilogx(x, _yh / gsf_flux, color=HIST_COL, lw=1.3, ls="--")
        _handles.append((_b19, _l19))
        _labels.append("GSF2019")
        for (_y, _tag), _c in zip(model_curves, OKABE_ITO, strict=False):
            (_ln,) = _ax1.loglog(x, _w * _y, lw=1.2, color=_c)
            _ax2.semilogx(x, _y / gsf_flux, lw=1.2, color=_c)
            _handles.append(_ln)
            _labels.append(_tag)
        _rel = gsf_err / gsf_flux
        _ax2.fill_between(x, 1 - _rel, 1 + _rel, color="0.55", alpha=0.5, lw=0)
        _ax2.axhline(1, color="k", lw=0.9)
        _ax1.set_ylabel(ylabel)
        _ax1.legend(
            _handles,
            _labels,
            fontsize=6.5,
            ncol=2,
            loc="lower left",
            handlelength=1.6,
            columnspacing=1.0,
            handletextpad=0.5,
            labelspacing=0.3,
            borderpad=0.4,
            handler_map={tuple: HandlerTuple(ndivide=1)},
        )
        _ax2.set_ylabel(f"model / {VLAB}")
        _ax2.set_xlabel(xlabel)
        _ax2.set_ylim(*ylim2)
        _ax1.set_ylim(*ylim1)
        return _fig

    with plt.rc_context({**RC_SERIF, **RC_SMALL}):
        _E = np.logspace(1, 11, 400)
        _curves = []
        for _ctor, _tag in MODEL_CTORS:
            _y = np.asarray(_ctor().total_flux(_E), float)
            if _tag in TRUNC:
                _y = np.where(TRUNC[_tag] >= _E, _y, np.nan)
            _curves.append((_y, _tag))
        _g19 = GSFEnergy(version="2019")
        _fig10 = two_panel(
            _E,
            np.asarray(gsf_e.total_flux(_E), float),
            np.asarray(gsf_e.total_error(_E), float),
            _curves,
            xlabel="E (GeV)",
            ylabel=r"$E^{2.7} J(E)$ [GeV$^{1.7}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]",
            ylim1=(8e2, 8e4),
            ylim2=(0.51, 1.49),
            hist=(
                np.asarray(_g19.total_flux(_E), float),
                np.asarray(_g19.total_error(_E), float),
            ),
        )
    show(_fig10)
    return MODEL_CTORS, TRUNC, two_panel


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fig. 11 — Nucleon flux

    Nucleon flux versus energy per nucleon from the GSF (thick line,
    $\pm1\sigma$ band) with the contributions of the four mass groups. The
    nucleon flux is not directly observed: it is an elementary derived
    quantity of the model, with uncertainties propagated through the full
    covariance.
    """)
    return


@app.cell
def _(COLORS, GROUPS, LABELS, LINES, RC_SERIF, gsf_kn, np, plt, show):
    # fitter plotting code: nucleon_flux (energy_exponent=2.7, show_lines=True)
    from matplotlib.ticker import FixedLocator as _FixedLocator1
    from matplotlib.ticker import NullFormatter as _NullFormatter2

    with plt.rc_context(RC_SERIF):
        _fig, _ax = plt.subplots(figsize=(3.4, 2.9))
        _fig.subplots_adjust(left=0.13, right=0.95, top=0.95)
        _e = np.geomspace(1e-1, 1e12, 1000)
        _sc = _e**2.7
        for _i, _g in enumerate(GROUPS):
            _fl = np.asarray(gsf_kn.flux(_e, _g), float)
            _ax.plot(
                _e, _fl * _sc, LINES[_i], color=COLORS[_i], lw=0.75, label=LABELS[_i]
            )
        _tot = np.asarray(gsf_kn.total_flux(_e), float)
        _terr = np.asarray(gsf_kn.total_error(_e), float)
        _ax.plot(_e, _tot * _sc, "-k", lw=1.2, label="Total")
        _ax.fill_between(
            _e, (_tot - _terr) * _sc, (_tot + _terr) * _sc, color="k", alpha=0.3, lw=0
        )
        _ax.loglog()
        _ax.set_xlabel(r"$E_{\mathrm{kin,n}} / \mathrm{GeV}$")
        _ax.set_ylabel(
            r"$J_{\mathrm{LIS,n}} / (\mathrm{GeV}\,\mathrm{m}^2\,\mathrm{s}\,\mathrm{sr})^{-1}"
            r"\times (E_{\mathrm{kin,n}}/\mathrm{GeV})^{2.7}$"
        )
        _ax.legend(
            loc="upper right",
            ncol=5,
            columnspacing=1.0,
            handlelength=1.0,
            handletextpad=0.4,
            fontsize="small",
        )
        # major ticks every second power of ten (labelled), minor at every power
        _ax.xaxis.set_major_locator(
            _FixedLocator1([10.0**_k for _k in range(0, 12, 2)])
        )
        _ax.xaxis.set_minor_locator(_FixedLocator1([10.0**_k for _k in range(0, 12)]))
        _ax.xaxis.set_minor_formatter(_NullFormatter2())
        _ax.set_xlim(1, 10**11.5)
        _ax.set_ylim(1e1, 1.001e5)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fig. 12 — Nucleon flux against the same parametrizations

    Bottom: ratio to the GSF; the grey band is the GSF $\pm1\sigma$.
    """)
    return


@app.cell
def _(
    GSFEnergyPerNucleon,
    MODEL_CTORS,
    RC_SERIF,
    RC_SMALL,
    TRUNC,
    gsf_en,
    np,
    plt,
    show,
    two_panel,
):
    # fitter plotting code: _two_panel, nucleon flux
    with plt.rc_context({**RC_SERIF, **RC_SMALL}):
        _EN = np.logspace(1, 9.5, 400)
        _ncurves = []
        for _ctor, _tag in MODEL_CTORS:
            _, _p, _n = _ctor().p_and_n_flux(_EN)
            _y = np.asarray(_p + _n, float)
            if _tag in TRUNC:
                _y = np.where(TRUNC[_tag] >= _EN, _y, np.nan)
            _ncurves.append((_y, _tag))
        _g19n = GSFEnergyPerNucleon(version="2019")
        _fig12 = two_panel(
            _EN,
            np.asarray(gsf_en.total_flux(_EN), float),
            np.asarray(gsf_en.total_error(_EN), float),
            _ncurves,
            xlabel=r"$E_N$ (GeV)",
            ylabel=r"$E_N^{2.7} J_N$ [GeV$^{1.7}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]",
            ylim1=(2e2, 5e4),
            ylim2=(0.3, 1.9),
            hist=(
                np.asarray(_g19n.total_flux(_EN), float),
                np.asarray(_g19n.total_error(_EN), float),
            ),
        )
    show(_fig12)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SM S1 — The proton-leader spline

    The fitted flux (thick line), the knot positions (dots), and each
    single-knot basis contribution (thin dashed/dotted), which sum to the flux.

    > The published version of this panel shows only the flux and the knots:
    > the deck draws each basis contribution by zeroing every other amplitude
    > and re-evaluating, but the frozen model it renders from ignores the
    > parameter vector, so all 25 "basis" curves come out equal to the full
    > spline and hide under it. Here the contributions are taken from the
    > spline's own Jacobian, so the figure shows what its caption describes.
    """)
    return


@app.cell
def _(
    COLORS,
    ELEM_LABELS,
    FIGSIZE_SMALL,
    RC_SERIF,
    cornertext,
    gsf_r,
    logx_ticks,
    np,
    plt,
    show,
):
    # fitter plotting code: spline_plots (proton panel, rigidity_exponent=3)
    with plt.rc_context(RC_SERIF):
        _fig, _ax = plt.subplots(figsize=FIGSIZE_SMALL)
        _fig.subplots_adjust(left=0.15, bottom=0.12, right=0.95, top=0.98)
        _sid = gsf_r.z_to_sids[1][0]
        _kn = np.exp(np.asarray(gsf_r.kx[_sid], float))
        _r = np.geomspace(_kn[0], _kn[-1], 1000)
        _sc = _r**3.0
        _ax.plot(
            _r,
            np.asarray(gsf_r.flux(_r, _sid), float) * _sc,
            "-",
            color=COLORS[0],
            lw=1.5,
        )
        _knots = np.unique(_kn)
        _ax.plot(
            _knots,
            np.asarray(gsf_r.flux(_knots, _sid), float) * _knots**3.0,
            "o",
            color=COLORS[0],
            markersize=3,
        )
        # each basis contribution = one amplitude x its basis function; the
        # jacobian columns ARE the basis functions, so this is the deck's
        # "all other coefficients zeroed" curve without touching the parameters
        _jac = np.asarray(gsf_r.jacobian(_r, _sid), float)
        _amp = np.asarray(gsf_r.pars[_sid], float)[: _jac.shape[1]]
        for _i in range(_jac.shape[1]):
            _ax.plot(
                _r,
                _jac[:, _i] * _amp[_i] * _sc,
                "--" if _i % 2 == 0 else ":",
                color=COLORS[0],
                lw=0.75,
            )
        _ax.loglog()
        _ax.set_xlabel(r"$R / \mathrm{GV}$")
        _ax.set_ylabel(
            r"$J_\mathrm{LIS} / (\mathrm{GV}\,\mathrm{m}^2\,\mathrm{s}\,\mathrm{sr})^{-1}"
            r"\times (R/\mathrm{GV})^{3}$"
        )
        cornertext(_ax, ELEM_LABELS[0], loc="upper left")
        logx_ticks(_ax, numticks=30)
        _ax.set_xlim(_r[0], _r[-1])
        _ax.set_ylim(1e1, 5e6)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SM S2 — Elemental abundances

    Abundances at fixed kinetic energy per nucleon from the GSF (coloured
    lines, $1$–$10^{9}$ GeV/n), normalized to $\mathrm{Si}\equiv100$, compared
    with the solar-system abundances of Lodders (2003). With the constant-ratio
    extrapolation of previous releases all curves above ~1 TeV/n would
    coincide; the power-law tilts drive the secondary valleys toward the
    (source-like) solar pattern as the energy increases, until the pattern
    freezes at the saturation rigidity $R_\mathrm{sat}=5$ PV.
    """)
    return


@app.cell
def _(LODDERS_2003, RC_SANS, SYMBOLS, gsf_kn, np, plt, show):
    with plt.rc_context(RC_SANS):
        _fig, _ax = plt.subplots(figsize=(5.6, 3.4))
        _zs = np.arange(1, 29)
        _ek = [10.0**_k for _k in range(10)]
        _colors = plt.cm.viridis(np.linspace(0.88, 0.02, len(_ek)))

        def _elabel(e):
            if e < 1e4:
                return f"{e:g} GeV/n"
            return rf"$10^{{{int(np.log10(e))}}}$ GeV/n"

        for _e, _col in zip(_ek, _colors, strict=False):
            _f = np.array(
                [
                    float(np.asarray(gsf_kn.flux(np.array([_e]), int(_z))).ravel()[0])
                    for _z in _zs
                ]
            )
            _si = _f[13]  # Z = 14
            if _si <= 0:
                continue
            _rel = 100.0 * _f / _si
            _ok = _rel > 0
            _ax.plot(
                _zs[_ok],
                _rel[_ok],
                "-",
                color=_col,
                lw=1.1,
                label=_elabel(_e),
                zorder=3,
            )

        _solar = np.asarray(LODDERS_2003, float)
        _solar = _solar * (100.0 / _solar[13])
        _ax.plot(
            _zs,
            _solar,
            "--o",
            color="black",
            lw=1.1,
            ms=2.8,
            mfc="white",
            mew=0.8,
            label="Solar system (Lodders 2003)",
            zorder=2,
        )

        _ax.set_yscale("log")
        _ax.set_xlim(0.5, 28.8)
        _ax.set_xticks(_zs)
        _ax.set_xticklabels([str(_z) for _z in _zs], fontsize=6.5)
        _sec = _ax.secondary_xaxis("top")
        _sec.set_xticks(_zs)
        _sec.set_xticklabels(SYMBOLS, fontsize=6.5)
        _sec.tick_params(length=0, pad=2)
        _ax.set_xlabel("nuclear charge $Z$", fontsize=9)
        _ax.set_ylabel("relative abundance (Si $\\equiv$ 100)", fontsize=9)
        _ax.tick_params(axis="y", labelsize=7)
        _ax.grid(True, which="major", color="#e3e5e8", lw=0.5)
        _ax.set_axisbelow(True)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(True)
        _ax.tick_params(axis="y", which="both", right=True, direction="in")
        _ax.legend(
            fontsize=7,
            ncol=3,
            frameon=False,
            loc="upper right",
            title="kinetic energy per nucleon",
            title_fontsize=7,
            columnspacing=1.2,
            handlelength=1.6,
            borderaxespad=0.3,
        )
        _fig.tight_layout(pad=0.4)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SM S3 — Sub-leading to leader flux ratios

    One panel per sub-leading species: the fitted spline ratio to its group
    leader (black) over the measured range, the slope-fit window (shaded), the
    adopted power-law extrapolation above the matching point with the fitted
    index $s_j$ quoted per panel (blue, saturating to a constant ratio at
    $R_\mathrm{sat}=5$ PV), and the constant-ratio extrapolation of previous
    GSF releases (grey dashed). The paper adds the offset- and
    demodulation-corrected data ratios that enter the likelihood.

    The window is the last decade of the species' direct data below the
    matching point; here it is taken as the decade below the top knot, which
    the fit ties to the last datum (within a few percent of the fitted window
    for every species).
    """)
    return


@app.cell
def _(RC_SANS, gsf_r, np, plt, show, species_name):
    import matplotlib.ticker as _mticker
    from matplotlib.lines import Line2D as _Line2D3
    from matplotlib.patches import Patch as _Patch3

    _C_SPLINE, _C_NEW, _C_OLD = "#30363d", "#2f6fde", "#9aa0a6"
    _X_END, _X_SAT = np.log(1e8), np.log(5.0e6)

    _subs = []
    for _sid in gsf_r.species:
        _lsid, _norm = gsf_r.flux_ratio[_sid]
        if _sid != _lsid:
            _subs.append((_sid, _lsid, _norm, gsf_r.flux_slope[_sid]))
    _subs.sort(key=lambda t: (t[1][0], t[0][0], t[0][1]))

    with plt.rc_context(RC_SANS):
        _ncol, _nrow = 4, 7
        _fig, _axes = plt.subplots(_nrow, _ncol, figsize=(7.08, 8.15), sharex=True)
        for _k, (_sid, _lsid, _norm, _slope) in enumerate(_subs):
            _ax = _axes[_k // _ncol, _k % _ncol]
            _kx = np.asarray(gsf_r.kx[_sid], float)
            _xmin, _xmax = _kx[0], _kx[-1]
            _x0 = max(_xmin, float(np.asarray(gsf_r.kx[_lsid], float)[0])) + 0.02
            _x_lo = np.linspace(_x0, _xmax, 200)
            _R_lo = np.exp(_x_lo)
            with np.errstate(divide="ignore", invalid="ignore"):
                _r_lo = np.asarray(gsf_r.flux(_R_lo, _sid), float) / np.asarray(
                    gsf_r.flux(_R_lo, _lsid), float
                )
            _r_lo[~np.isfinite(_r_lo)] = 0.0
            _x_hi = np.linspace(_xmax, _X_END, 160)
            _r_new = _norm * np.exp(
                _slope * np.clip(_x_hi - _xmax, 0.0, max(_X_SAT - _xmax, 0.0))
            )
            _ax.axvspan(
                np.exp(_xmax) / 10.0, np.exp(_xmax), color=_C_NEW, alpha=0.10, lw=0
            )
            _ok = _r_lo > 0
            _ax.plot(_R_lo[_ok], _r_lo[_ok], color=_C_SPLINE, lw=1.1, zorder=3)
            _ax.plot(
                np.exp(_x_hi),
                np.full_like(_x_hi, _norm),
                color=_C_OLD,
                lw=0.9,
                ls="--",
                zorder=2,
            )
            (_ln_new,) = _ax.plot(np.exp(_x_hi), _r_new, color=_C_NEW, lw=1.4, zorder=4)
            _ax.set_xscale("log")
            _ax.set_yscale("log")
            _ymax = max(np.max(_r_lo[_ok]) if np.any(_ok) else _norm, _norm) * 3
            _ymin = min(np.min(_r_new), _norm) / 5
            _ymin = max(_ymin, _ymax * 1e-6)
            if _ymax / _ymin < 10**1.5:
                _c = np.sqrt(_ymax * _ymin)
                _ymin, _ymax = _c / 10**0.75, _c * 10**0.75
            _ax.set_ylim(_ymin, _ymax)
            _ax.yaxis.set_major_locator(_mticker.LogLocator(base=10.0, numticks=4))
            _ax.yaxis.set_major_formatter(_mticker.LogFormatterMathtext(base=10.0))
            _ax.yaxis.set_minor_locator(
                _mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=99)
            )
            _ax.yaxis.set_minor_formatter(_mticker.NullFormatter())
            _ax.legend(
                [_ln_new],
                [f"{species_name(_sid)}/{species_name(_lsid)}  s={_slope:+.3f}"],
                loc="lower left",
                fontsize=6.3,
                frameon=False,
                handlelength=1.2,
                borderaxespad=0.15,
            )
            _ax.tick_params(
                which="both", direction="in", top=True, right=True, labelsize=6, pad=2
            )
            _ax.grid(True, axis="x", which="major", color="#e3e5e8", lw=0.5)
            _ax.set_axisbelow(True)
        _axes[0, 0].set_xlim(0.8, 1.3e8)
        for _ax in _axes.ravel():
            _ax.xaxis.set_major_locator(_mticker.LogLocator(base=100.0))
            _ax.xaxis.set_major_formatter(_mticker.LogFormatterMathtext(base=10.0))
            _ax.xaxis.set_minor_locator(_mticker.LogLocator(base=10.0, numticks=99))
            _ax.xaxis.set_minor_formatter(_mticker.NullFormatter())
        _leg_ax = _axes[6, 1]
        _leg_ax.axis("off")
        _leg_ax.legend(
            handles=[
                _Line2D3([], [], color=_C_SPLINE, lw=1.1, label="fitted spline ratio"),
                _Patch3(
                    facecolor=_C_NEW,
                    alpha=0.10,
                    label="slope-fit window (last data decade)",
                ),
                _Line2D3(
                    [],
                    [],
                    color=_C_NEW,
                    lw=1.4,
                    label=r"extrapolation norm$\cdot(\min(R,R_{\rm sat})/"
                    r"R_{\rm max})^{s}$, $R_{\rm sat}=5$ PV",
                ),
                _Line2D3(
                    [], [], color=_C_OLD, lw=0.9, ls="--", label="constant ratio (old)"
                ),
            ],
            loc="center left",
            bbox_to_anchor=(0.02, 0.5),
            fontsize=6.8,
            frameon=False,
            handlelength=1.6,
            borderaxespad=0.0,
        )
        for _k in range(len(_subs), _nrow * _ncol):
            if _k != 25:  # 25 = the legend panel
                _axes[_k // _ncol, _k % _ncol].axis("off")
        for _j in range(_ncol):
            _ax = _axes[6, _j] if _axes[6, _j].axison else _axes[5, _j]
            _ax.tick_params(labelbottom=True)
            _ax.set_xlabel(r"$R$ [GV]", fontsize=7, labelpad=1.5)
        for _i in range(_nrow):
            if _axes[_i, 0].axison:
                _axes[_i, 0].set_ylabel("flux ratio", fontsize=7, labelpad=1.5)
        _fig.subplots_adjust(
            left=0.075, right=0.982, top=0.995, bottom=0.05, wspace=0.28, hspace=0.10
        )
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SM S4 — Impact of the solar-modulation potential

    Fitted LIS of the four mass groups relative to the default
    Ghelfi–Maurin–Derome reconstruction (parameter set `2026.1`); the grey band
    is the $1\sigma$ uncertainty of the default. Blue: the Usoskin 2017
    potential with the same window average and $\delta\phi$ nuisances
    (parameter set `2026.1-USO`).

    The paper's third curve — the single-epoch demodulation used by previous
    GSF releases — comes from a third re-fit that is not part of the released
    model, so it cannot be drawn here.
    """)
    return


@app.cell
def _(GROUPS, GSFRigidity, RC_SERIF, np, plt, show):
    import matplotlib.ticker as _mticker2

    _C_USO = "#1f77b4"

    with plt.rc_context({**RC_SERIF, "font.size": 8}):
        _fig, _axes = plt.subplots(4, 1, figsize=(3.4, 5.7), sharex=True, sharey=True)
        _R = np.logspace(np.log10(1.3), np.log10(50), 160)
        _gmd = GSFRigidity(version="2026.1")
        _uso = GSFRigidity(version="2026.1-USO")
        _panel_labels = ("H*", "He*", "O*", "Fe*")
        for _i, (_ax, _g) in enumerate(zip(_axes, GROUPS, strict=False)):
            _jg = np.asarray(_gmd.flux(_R, _g, time_interval="LIS"), float)
            _eg = np.asarray(_gmd.error(_R, _g, time_interval="LIS"), float)
            _ref = np.where(_jg > 0, _jg, 1.0)
            _ax.fill_between(
                _R,
                -100 * _eg / _ref,
                100 * _eg / _ref,
                color="0.55",
                alpha=0.55,
                lw=0,
                label=r"GMD (default) $\pm1\sigma$",
            )
            _ju = np.asarray(_uso.flux(_R, _g, time_interval="LIS"), float)
            _ax.plot(
                _R,
                100 * (_ju / _ref - 1.0),
                color=_C_USO,
                lw=1.5,
                label="Usoskin (USO)",
            )
            _ax.axhline(0, color="k", lw=0.5)
            _ax.set_xscale("log")
            _ax.text(
                0.035,
                0.88,
                _panel_labels[_i],
                transform=_ax.transAxes,
                fontsize=9,
                fontweight="bold",
                va="top",
                ha="left",
            )
            _ax.grid(alpha=0.3)
            _ax.tick_params(labelsize=7.5)
        _axes[-1].set_xticks([2, 5, 10, 20, 50])
        _axes[-1].xaxis.set_major_formatter(_mticker2.ScalarFormatter())
        _axes[-1].xaxis.set_minor_formatter(_mticker2.NullFormatter())
        _axes[-1].set_xlabel(r"rigidity $R$ [GV]")
        _axes[0].set_ylim(-22, 34)
        _axes[0].legend(
            frameon=False,
            fontsize=6.6,
            loc="upper right",
            handlelength=1.6,
            borderaxespad=0.4,
            labelspacing=0.3,
        )
        _fig.supylabel("deviation from GMD default  [%]", fontsize=8, x=0.02)
        _fig.tight_layout(pad=0.3, h_pad=0.4)
        _fig.subplots_adjust(left=0.155)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SM S7 — Correlation matrix of the fitted spline amplitudes

    Correlation of the leading spline amplitudes, grouped into the $p$, He,
    O\* and Fe\* blocks. The energy-scale offsets are not part of the published
    covariance and are not shown.
    """)
    return


@app.cell
def _(ELEM_LABELS, FIGSIZE_SMALL, RC_SERIF, gsf_e, np, plt, show):
    # fitter plotting code: covariance_plot
    with plt.rc_context(RC_SERIF):
        _leaders = [gsf_e.z_to_sids[_z][0] for _z in (1, 2, 8, 26)]
        _sizes = [np.shape(gsf_e.cov[(_l, _l)])[0] for _l in _leaders]
        _n1 = sum(_sizes)
        _cov = np.zeros((_n1, _n1))
        _off = np.cumsum([0, *_sizes])
        for _i, _li in enumerate(_leaders):
            for _j, _lj in enumerate(_leaders):
                _block = gsf_e.cov.get((_li, _lj))
                if _block is not None:
                    _cov[_off[_i] : _off[_i + 1], _off[_j] : _off[_j + 1]] = np.asarray(
                        _block, float
                    )
        _m = np.zeros((_n1, _n1))
        for _i in range(_n1):
            for _j in range(_i + 1, _n1):
                _m[_i, _j] = (
                    _cov[_i, _j] / abs(_cov[_i, _i] * _cov[_j, _j] + 1e-300) ** 0.5
                )
                _m[_j, _i] = _m[_i, _j]
        _fig, _ax = plt.subplots(figsize=FIGSIZE_SMALL)
        _fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.12)
        _pc = _ax.pcolormesh(_m.T, cmap="bwr", vmin=-1, vmax=1)
        _fig.colorbar(_pc, ax=_ax)
        _ax.set_xlim(0, _n1)
        _ax.set_ylim(0, _n1)
        for _i in range(len(_leaders)):
            _ia, _ib = _off[_i], _off[_i + 1]
            _ax.plot([_ia, _ib, _ib, _ia, _ia], [_ia, _ia, _ib, _ib, _ia], "-k")
            _ax.text(0.5 * (_ia + _ib), -2.5, ELEM_LABELS[_i], va="top", ha="center")
            _ax.text(-2.5, 0.5 * (_ia + _ib), ELEM_LABELS[_i], va="center", ha="right")
        _ax.axis("off")
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SM S8 — Pivot components against the full covariance

    Relative $1\sigma$ uncertainty of the total proton and neutron fluxes
    from the full amplitude covariance (solid) and from the 24 pivot
    components (dashed); dots mark the pivot energies. Bottom: ratio of the
    two — within the shaded band the reduced uncertainty stays inside the
    worst-case coverage factor, and it is exact at the pivots.
    """)
    return


@app.cell
def _(RC_SANS, RC_SMALL, gsf_en, np, plt, show):
    from globalsplinefit.reduced import ReducedGSF, _relative_species_system

    _COL = {"p": "#0072B2", "n": "#D55E00"}  # Okabe-Ito blue / vermillion
    _red = ReducedGSF(gsf_en)
    _lo, _hi = _red.pivot_energies[0], _red.pivot_energies[-1]
    _e = np.logspace(np.log10(_lo), np.log10(_hi), 601)
    # exact reference from the construction's own trimmed stacked system, so
    # the ratio panel reproduces the coverage-factor definition verbatim
    _jac_rel, _cov_par, _central, _species = _relative_species_system(
        gsf_en, _e, per_group=False
    )
    _exact = np.sqrt(np.einsum("ij,jk,ik->i", _jac_rel, _cov_par, _jac_rel)).reshape(
        len(_species), len(_e)
    )
    _reduced = _red.error(_e) / _red.flux(_e)
    _ratio = _reduced / _exact
    _worst = float(np.exp(np.max(np.abs(np.log(_ratio)))))
    _piv = _red.error(_red.pivot_energies) / _red.flux(_red.pivot_energies)

    with plt.rc_context({**RC_SANS, **RC_SMALL}):
        _fig, (_ax1, _ax2) = plt.subplots(
            2,
            1,
            figsize=(3.4, 3.6),
            sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.06},
        )
        for _s, _name in enumerate(_species):
            _ax1.loglog(
                _e, 100 * _exact[_s], color=_COL[_name], lw=1.4, label=f"{_name}, full"
            )
            _ax1.loglog(
                _e,
                100 * _reduced[_s],
                color=_COL[_name],
                lw=1.1,
                ls="--",
                label=f"{_name}, reduced",
            )
            _ax1.plot(
                _red.pivot_energies,
                100 * _piv[_s],
                "o",
                color=_COL[_name],
                ms=3,
                mec="none",
                zorder=5,
            )
        _ax1.set_ylabel(r"relative $1\sigma$ uncertainty (%)")
        _ax1.legend(
            loc="upper left", frameon=False, ncol=2, columnspacing=1.2, handlelength=1.8
        )
        _ax1.tick_params(which="both", direction="in", top=True, right=True)
        _ax2.axhspan(1 / _worst, _worst, color="0.90", zorder=0)
        _ax2.axhline(1.0, color="k", lw=0.7)
        for _s, _name in enumerate(_species):
            _ax2.semilogx(
                _e,
                _ratio[_s],
                color=_COL[_name],
                lw=1.1,
                ls="-" if _name == "p" else "--",
            )
            _ax2.plot(
                _red.pivot_energies,
                np.ones_like(_red.pivot_energies),
                "o",
                color=_COL[_name],
                ms=3,
                mec="none",
                zorder=5,
            )
        _ax2.set_xlabel("kinetic energy per nucleon (GeV)")
        _ax2.set_ylabel(r"$\sigma_{\rm red}/\sigma_{\rm full}$")
        _ax2.set_ylim(0.65, 1.45)
        _ax2.set_xlim(_lo, _hi)
        _ax2.tick_params(which="both", direction="in", top=True, right=True)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SM S9 — Sampling check for pseudo-experiments

    Empirical standard deviation of 2000 draws from the native amplitude
    covariance, divided by the exact Jacobian-propagated uncertainty, for the
    four mass groups and the all-particle flux, in total energy per nucleus.
    The residual scatter is the statistical fluctuation of the ensemble.
    """)
    return


@app.cell
def _(GROUPS, RC_SANS, RC_SMALL, gsf_e, np, plt, show):
    _COLG = {"H*": "#D55E00", "He": "#E69F00", "O*": "#009E73", "Fe*": "#0072B2"}
    _E_LO, _E_HI, _N_GRID, _N_DRAWS = 1e2, 1e11, 601, 2000

    _energy = np.logspace(np.log10(_E_LO), np.log10(_E_HI), _N_GRID)
    _central = np.array([np.asarray(gsf_e.flux(_energy, _g), float) for _g in GROUPS])
    _jacs = [np.asarray(gsf_e.jacobian(_energy, _g), float) for _g in GROUPS]
    _widths = [_j.shape[1] for _j in _jacs]
    _sids = [gsf_e.z_to_sids[_z][0] for _z in (1, 2, 8, 26)]

    _P = sum(_widths)
    _C = np.zeros((_P, _P))
    _off = np.cumsum([0, *_widths])
    for _i, _si in enumerate(_sids):
        for _j, _sj in enumerate(_sids):
            _block = gsf_e.cov.get((_si, _sj))
            if _block is not None:
                _C[_off[_i] : _off[_i + 1], _off[_j] : _off[_j + 1]] = _block
    _C = 0.5 * (_C + _C.T)
    _w, _V = np.linalg.eigh(_C)
    _factor = _V * np.sqrt(np.clip(_w, 0.0, None))

    _sig_g = np.array([np.asarray(gsf_e.error(_energy, _g), float) for _g in GROUPS])
    _var_all = np.zeros(_N_GRID)
    for _gi in GROUPS:
        for _gj in GROUPS:
            _var_all += np.diag(np.asarray(gsf_e.covariance(_gi, _gj, _energy), float))
    _sig_all = np.sqrt(_var_all)

    _rng = np.random.default_rng(0)
    _dc = _rng.standard_normal((_N_DRAWS, _P)) @ _factor.T
    _draws = np.empty((_N_DRAWS, len(GROUPS), _N_GRID))
    for _i in range(len(GROUPS)):
        _draws[:, _i] = _central[_i] + _dc[:, _off[_i] : _off[_i + 1]] @ _jacs[_i].T
    _r_g = _draws.std(axis=0) / _sig_g
    _r_all = _draws.sum(axis=1).std(axis=0) / _sig_all

    with plt.rc_context({**RC_SANS, **RC_SMALL}):
        _fig, _ax = plt.subplots(figsize=(3.4, 2.3))
        for _i, _g in enumerate(GROUPS):
            _ax.semilogx(
                _energy,
                _r_g[_i],
                color=_COLG[_g],
                lw=1.0,
                label="H" if _g == "H*" else _g,
            )
        _ax.semilogx(_energy, _r_all, color="k", lw=1.4, label="all-particle")
        _ax.axhline(1, color="k", lw=0.7)
        _ax.set_xlabel("total energy per nucleus (GeV)")
        _ax.set_ylabel(r"$\sigma_{\rm sample}/\sigma_{\rm exact}$")
        _ax.set_xlim(_E_LO, _E_HI)
        _ax.set_ylim(0.85, 1.15)
        _ax.legend(
            loc="upper left", frameon=False, ncol=3, columnspacing=1.0, handlelength=1.4
        )
        _ax.tick_params(which="both", direction="in", top=True, right=True)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SM S14 — Light and heavy fluxes

    Light ($p+$He) and heavy (O\*$+$Fe\*) fluxes, weighted by
    $E_\mathrm{kin}^{3}$, against the all-particle flux (black). In the paper
    both panels carry the experiments that report two-component composition.
    """)
    return


@app.cell
def _(
    FIGSIZE_WIDE,
    HEAVY_COLOR,
    LIGHT_COLOR,
    RC_SERIF,
    draw_band,
    gsf_k,
    logx_ticks,
    np,
    pair_error,
    plt,
    show,
):
    # fitter plotting code: light_heavy
    from matplotlib.ticker import EngFormatter as _EngFormatter2

    with plt.rc_context(RC_SERIF):
        _fig, _axes = plt.subplots(2, 1, figsize=FIGSIZE_WIDE, sharex=True, sharey=True)
        _fig.subplots_adjust(hspace=0, right=0.95, left=0.13, top=0.95, bottom=0.12)
        _e = np.geomspace(1e2, 1e11, 1000)
        _sc = _e**3.0
        _tot = np.asarray(gsf_k.total_flux(_e), float) * _sc
        _pairs = (
            (("H*", "He"), LIGHT_COLOR, r"$p + \mathrm{He}$"),
            (("O*", "Fe*"), HEAVY_COLOR, r"$\mathrm{O*} + \mathrm{Fe*}$"),
        )
        for _i, (_pair, _col, _label) in enumerate(_pairs):
            _val = sum(np.asarray(gsf_k.flux(_e, _g), float) for _g in _pair) * _sc
            _err = pair_error(gsf_k, _e, *_pair) * _sc
            draw_band(_axes[_i], _e, _val, _err, _col, line="-", lw=0.9, zorder=0)
            _axes[_i].plot(_e, _tot, "-", color="k", lw=1.5)
            _axes[_i].set_ylabel(_label)
        _axes[1].yaxis.set_major_formatter(_EngFormatter2())
        _axes[1].set_xscale("log")
        _axes[1].set_xlabel(r"$E_\mathrm{kin} / \mathrm{GeV}$")
        _axes[1].set_yticks(np.arange(0, 4.5e6, 1e6))
        _axes[1].set_ylim(0, 4.8e6)
        _axes[1].set_xlim(1e3, 1e11)
        logx_ticks(_axes[1])
        _fig.text(
            0.010,
            0.5,
            r"$J_\mathrm{LIS} / (\mathrm{GeV}\,\mathrm{m}^2\,\mathrm{s}\,\mathrm{sr})^{-1}"
            r"\times (E_\mathrm{kin}/\mathrm{GeV})^{3}$",
            rotation=90,
            va="center",
        )
    show(_fig)
    return


@app.cell(hide_code=True)
def _(NOTEBOOK, SITE, mo):
    if SITE:  # running in the browser: link the sources published next to it
        _py = f"{SITE}gallery/{NOTEBOOK}/{NOTEBOOK}.py"
        _ipynb = f"{SITE}gallery/{NOTEBOOK}/{NOTEBOOK}.ipynb"
    else:  # running locally: hand over the file on disk
        _py = (mo.notebook_dir() / f"{NOTEBOOK}.py").read_bytes()
        _ipynb = None

    _buttons = [
        mo.download(
            _py,
            filename=f"{NOTEBOOK}.py",
            label="marimo notebook (.py)",
            mimetype="text/x-python",
        )
    ]
    if _ipynb is not None:
        _buttons.append(
            mo.download(
                _ipynb,
                filename=f"{NOTEBOOK}.ipynb",
                label="Jupyter notebook (.ipynb)",
                mimetype="application/x-ipynb+json",
            )
        )
    mo.vstack(
        [
            mo.md(r"""### Take this notebook with you"""),
            mo.hstack(_buttons, justify="start"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
