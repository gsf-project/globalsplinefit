"""Publication exports retain the selected comparison and physical observable."""

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "webapp"))
import bridge  # noqa: E402
import gsf_explorer as gx  # noqa: E402

matplotlib = pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text  # noqa: E402


def comparison_legend(ax):
    legends = ax.findobj(lambda artist: artist.get_gid() == "comparison-legend")
    assert len(legends) == 1
    return legends[0]


@pytest.fixture
def params():
    return {
        "version": "2026.1",
        "basis": "etot",
        "quantity": "nucleus",
        "dmin": 2,
        "dmax": 4,
        "npts": 10,
        "groups": gx.GROUPS,
        "elements": [6],
        "mod": "LIS",
        "cutoff": 0,
        "escale": 1,
        "phiBins": 1,
    }


@pytest.mark.parametrize(
    "quantity,basis,ratio",
    [
        ("nucleus", "etot", False),
        ("nucleon", "en", False),
        ("mean_lna", "etot", False),
        ("var_lna", "rig", False),
        ("nucleus", "en", True),
        ("nucleon", "ekn", True),
    ],
)
@pytest.mark.filterwarnings("error:Glyph.*")
def test_combined_export_contains_all_models_and_selected_view(
    params, monkeypatch, quantity, basis, ratio
):
    params.update(quantity=quantity, basis=basis)
    selections = [
        params,
        dict(params, version="2026.1-USO", elements=[]),
        dict(params, version="2026.1-EPOSLHCR", elements=[]),
    ]
    captured = []
    monkeypatch.setattr(
        gx, "figure_bytes", lambda fig, *_a, **_kw: captured.append(fig) or b"figure"
    )
    opts = {
        "overlayBands": True,
        "ratio": ratio,
        "yRange": [0.01, 8],
        "includeCaption": True,
        "bandAlpha": 0.05,
        "lineWeight": 4,
    }
    assert (
        base64.b64decode(bridge.figure(json.dumps(selections), json.dumps(opts)))
        == b"figure"
    )
    fig = captured[0]
    ax = fig.axes[0]
    composition = quantity in {"mean_lna", "var_lna"}
    assert len(ax.lines) == (3 if composition else 13 if ratio else 16)
    assert {line.get_linestyle() for line in ax.lines} >= {"-", ":", "-."}
    legend_labels = [text.get_text() for text in comparison_legend(ax).findobj(Text)]
    assert legend_labels[:3] == [f"GSF {p['version']}" for p in selections]
    assert len(legend_labels) == 3 + (1 if composition else 5 if ratio else 6)
    assert ax.get_xlim() == (100, 10000)
    assert ax.get_ylim() == (0.01, 8)
    assert ax.get_yscale() == ("linear" if composition else "log")
    assert any(collection.get_hatch() for collection in ax.collections)
    assert len(fig.texts) == 1
    caption = " ".join(fig.texts[0].get_text().split())
    assert "GSF 2026.1-USO" in caption
    assert "hatched bands" in caption
    if composition:
        assert "mass" in caption
        assert "Fluxes are scaled" not in caption
    elif ratio:
        assert "Fractions" in caption
        assert "correlation with the total" in caption
        assert "J_i" in ax.get_ylabel()
    assert ax.lines[0].get_linewidth() == (1.2 if composition else 0.75)
    assert ax.collections[0].get_alpha() == 0.3
    assert not plt.get_fignums(), "bridge leaked an open figure"


@pytest.mark.parametrize("quantity,basis", [("nucleus", "etot"), ("nucleon", "en")])
def test_fraction_errors_include_denominator_covariance(params, quantity, basis):
    params.update(quantity=quantity, basis=basis)
    model, result = bridge._evaluate(params)
    normalized = gx.fraction_result(model, result, "LIS", None)
    np.testing.assert_allclose(sum(normalized["series"][g][0] for g in gx.GROUPS), 1)
    for group in gx.GROUPS:
        np.testing.assert_allclose(
            normalized["series"][group][1],
            model.fraction_error(result["x"], group, time_interval="LIS"),
            rtol=1e-8,
        )


def test_caption_describes_only_visible_curves_and_bands(params):
    params.update(groups=[], elements=["D"], cutoff=2, escale=1.1)
    text = bridge.figure_caption(
        json.dumps(params), json.dumps({"showTotal": False, "showBands": False})
    )
    assert "individual elements D" in text
    assert "all-particle" not in text
    assert "Shaded bands" not in text
    assert "2 GV" in text and "1.1" in text


@pytest.mark.parametrize("width,height", [(3.375, 2.55), (5, 3.55), (7, 4.5)])
def test_caption_stays_below_axes_at_publication_sizes(params, width, height):
    _, result = bridge._evaluate(params)
    text = gx.caption(
        ["2026.1", "2026.1-USO", "2026.1-EPOSLHCR"],
        "etot",
        2.7,
        "LIS",
        [6],
        True,
        None,
        overlay_bands=True,
    )
    fig = gx.make_figure(
        [result] * 3,
        "etot",
        2.7,
        versions=["2026.1", "2026.1-USO", "2026.1-EPOSLHCR"],
        caption_text=text,
        width_in=width,
        height_in=height,
    )
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = fig.texts[0].get_window_extent(renderer)
        xlabel = fig.axes[0].xaxis.label.get_window_extent(renderer)
        assert bounds.y1 < xlabel.y0
        assert bounds.x0 >= 0 and bounds.x1 <= fig.bbox.x1
        assert bounds.y0 >= 0
        assert fig.axes[0].xaxis.label.get_fontname() == "Nimbus Roman"
        assert fig.axes[0].xaxis.label.get_color() == "black"
    finally:
        plt.close(fig)


def test_single_model_export_can_omit_caption(params, monkeypatch):
    captured = []
    monkeypatch.setattr(
        gx, "figure_bytes", lambda fig, *_a, **_kw: captured.append(fig) or b"figure"
    )
    bridge.figure(json.dumps(params), json.dumps({"includeCaption": False}))
    assert captured[0].texts == []


@pytest.mark.parametrize("width,height", [(3.375, 2.55), (5, 3.55), (6, 3.6)])
@pytest.mark.parametrize("ymax", [6e6, 6e-6])
@pytest.mark.parametrize("include_caption", [False, True])
@pytest.mark.parametrize(
    "versions",
    [
        ["2026.1-SIB23e", "2026.1-EPOSLHCR"],
        ["2026.1-SIB23e", "2026.1-USO", "2026.1-EPOSLHCR"],
    ],
)
def test_comparison_header_stays_aligned_and_clear_of_plot_and_exponent(
    params, width, height, ymax, include_caption, versions
):
    _, result = bridge._evaluate(params)
    fig = gx.make_figure(
        [result] * len(versions),
        "etot",
        3,
        versions=versions,
        ylog=False,
        y_range=[0, ymax],
        width_in=width,
        height_in=height,
        caption_text="Comparison of three GSF models." if include_caption else "",
    )
    try:
        with matplotlib.rc_context(gx.PAPER_RC):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            ax = fig.axes[0]
            offset = ax.yaxis.get_offset_text()
            assert offset.get_text(), "The regression requires a scientific multiplier"
            offset_bounds = offset.get_window_extent(renderer)
            legend = comparison_legend(ax)
            legend_bounds = legend.get_window_extent(renderer)
            assert not legend_bounds.overlaps(offset_bounds)
            assert legend_bounds.y0 > ax.bbox.y1
            assert legend_bounds.x1 == pytest.approx(ax.bbox.x1)
            assert legend_bounds.x0 >= ax.bbox.x0
            assert legend_bounds.y1 <= fig.bbox.y1
            texts = legend.findobj(Text)
            assert [text.get_text() for text in texts[: len(versions)]] == [
                f"GSF {version}" for version in versions
            ]
            text_bounds = [text.get_window_extent(renderer) for text in texts]
            for i, bounds in enumerate(text_bounds):
                assert legend_bounds.contains(bounds.x0, bounds.y0)
                assert legend_bounds.contains(bounds.x1, bounds.y1)
                assert all(not bounds.overlaps(other) for other in text_bounds[i + 1 :])
        svg = gx.figure_bytes(fig, "svg")
        assert b"PRELIMINARY" not in svg
        assert all(f"GSF {version}".encode() in svg for version in versions)
    finally:
        plt.close(fig)
