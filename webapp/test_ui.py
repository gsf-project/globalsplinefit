"""GSF Explorer — full UI test suite.

Exercises EVERY control element on the live page (headless chromium against
the real Pyodide runtime) and fails on any error toast, console error, page
error, or missing effect. Checks keep running after a failure; the summary
lists every control with PASS/FAIL.

Usage:
    pip install playwright && playwright install chromium
    python test_ui.py [--fast]     # --fast skips matplotlib exports

Sections: boot · about/citations · model list (add/remove/reorder) ·
plot quantity (nucleus/nucleon/lnA moments) · abscissa constraints (all 5,
incl. deuterium on rigidity) · components (groups + all element
chips + deuterium) · display (gamma, scale, bands, overlay hatches,
opacity) · modulation (SC24/LIS/interval, reversed, edges) · advanced
(escale, cutoff, npts) · plot navigation (box-zoom, pan, wheel-pinch,
wheel-pan, home, double-click) · hover readout · exports (CSV, PDF/SVG/PNG,
snapshot, data table) · theme (toggle, persistence, browser default) ·
responsive (mobile viewport).
"""

import argparse
import pathlib
import re
import subprocess
import sys
import time

from playwright.sync_api import sync_playwright

WEBAPP = pathlib.Path(__file__).resolve().parent
PORT = 8129
URL = f"http://localhost:{PORT}/"

results = []
console_errors = []


def check(name):
    """Decorator-ish context: run fn, record PASS/FAIL, keep going."""
    class _Ctx:
        def __enter__(self):
            self.t0 = time.time()
            self.err0 = len(console_errors)
            return self

        def __exit__(self, exc_type, exc, tb):
            dt = time.time() - self.t0
            new_errs = console_errors[self.err0:]
            if exc is None and not new_errs:
                results.append((name, "PASS", f"{dt:.1f}s"))
            else:
                msg = (f"{type(exc).__name__}: {str(exc)[:120]}" if exc
                       else f"console: {new_errs[0][:120]}")
                results.append((name, "FAIL", msg))
            return True   # swallow, keep testing
    return _Ctx()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="skip matplotlib figure exports")
    args = ap.parse_args()

    srv = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(PORT), "-d", str(WEBAPP)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            ctx = browser.new_context(
                viewport={"width": 1680, "height": 1000},
                permissions=["clipboard-read", "clipboard-write"])
            page = ctx.new_page()
            page.on("console", lambda m: m.type == "error"
                    and console_errors.append(m.text[:200]))
            page.on("pageerror",
                    lambda e: console_errors.append(str(e)[:200]))

            # ---------------------------------------------------- helpers
            def toast():
                return (page.text_content(".toast")[:150]
                        if page.query_selector(".toast") else None)

            def settle(ms=2500):
                page.wait_for_timeout(ms)
                t = toast()
                assert t is None, f"error toast: {t}"
                assert page.query_selector("svg.chart path"), "chart empty"

            def open_panel(title):
                command = {
                    "Components": "Series",
                    "Model": "Series",
                    "Abscissa": "Settings",
                    "Solar modulation": "Settings",
                    "Advanced": "Settings",
                    "Export": "Export",
                }.get(title)
                if command and not page.query_selector(
                        f".panel:has(h2:text-is('{title}'))"):
                    page.click(f".commandbtn:text-is('{command}')")
                    page.wait_for_selector(
                        f".panel:has(h2:text-is('{title}'))", timeout=5_000)
                if title == "Display":
                    return
                sel = f".panel:has(h2:text-is('{title}'))"
                if "closed" in (page.get_attribute(sel, "class") or ""):
                    page.click(f"{sel} > header")
                    page.wait_for_timeout(150)

            def by_label(label):
                return page.locator(f"label.field:has-text('{label}')")

            def el_chip(sym):
                return page.locator(".elgrid button",
                                    has_text=re.compile(f"^{sym}$"))

            def header_tag():
                return page.text_content(".wordmark .tag")

            def yticks():
                return page.eval_on_selector_all(
                    "svg.chart text[text-anchor='end']",
                    "els => els.map(e => e.textContent).join(' ')")

            def plot_frame():
                """Plot-area frame in page coords (mirrors chart.js insets)."""
                bb = page.locator("svg.chart").bounding_box()
                return {"l": bb["x"] + 14 + 74,
                        "r": bb["x"] + bb["width"] - 44,
                        "t": bb["y"] + 72 + 18,
                        "b": bb["y"] + bb["height"] - 140}

            # ------------------------------------------------------- boot
            with check("boot: pyodide + first evaluation"):
                page.goto(URL, timeout=60_000)
                page.wait_for_selector("svg.chart path", timeout=300_000)
                settle(800)

            # -------------------------------------------- about/citations
            with check("about: open via top rail button"):
                page.click(".aboutbtn")
                page.wait_for_selector(".modal", timeout=5_000)
                assert "Global Spline Fit" in page.text_content(".modal")

            with check("about: 3 BibTeX records + TBD + copy works"):
                assert page.locator(".cite").count() == 4
                assert page.locator(".cite button").count() == 3
                for key in ("Dembinski:2017zsh", "Fujisue:2025wnp",
                            "Dembinski:2025nmp"):
                    assert key in page.text_content(".modal"), key
                page.click(".cite button >> nth=0")
                page.wait_for_timeout(250)
                clip = page.evaluate("navigator.clipboard.readText()")
                assert clip.startswith("@article{Dembinski:2017zsh")

            with check("about: close"):
                page.click(".modal header button")
                assert not page.query_selector(".modal")

            # ------------------------------------------------- model list
            with check("model: add overlay (2026-USO)"):
                open_panel("Model")
                page.select_option(
                    "label.field:has-text('Add model') select", "2026-USO")
                settle(3_500)
                assert page.locator(".modelrow").count() == 2
                assert "2026 vs 2026-USO" in header_tag()

            with check("model: add third (2017), cap reached"):
                page.select_option(
                    "label.field:has-text('Add model') select", "2017")
                settle(4_000)
                assert page.locator(".modelrow").count() == 3
                assert not page.query_selector(
                    "label.field:has-text('Add model')")

            with check("model: drag-reorder promotes new primary"):
                page.drag_and_drop(".modelrow >> nth=1", ".modelrow >> nth=0")
                settle(4_000)
                assert header_tag().startswith("2026-USO")
                page.drag_and_drop(".modelrow >> nth=1", ".modelrow >> nth=0")
                settle(4_000)
                assert header_tag().startswith("2026 ")

            with check("model: legend shows model line styles"):
                assert "2026-USO" in page.text_content("svg.chart")
                toggle = page.locator(".legend-toggle")
                assert toggle.get_attribute("aria-expanded") == "true"
                toggle.click()
                assert toggle.get_attribute("aria-expanded") == "false"
                toggle.press("Enter")
                assert toggle.get_attribute("aria-expanded") == "true"

            with check("model: remove overlays"):
                while page.locator(".modelrow .mdel").count() > 0 \
                        and page.locator(".modelrow").count() > 1:
                    page.click(".modelrow .mdel >> nth=-1")
                    page.wait_for_timeout(1_500)
                settle(2_500)
                assert page.locator(".modelrow").count() == 1

            # ---------------------------------------------------- abscissa
            for basis in ("ekin", "rig", "en", "ekn", "etot"):
                with check(f"abscissa: {basis}"):
                    open_panel("Abscissa")
                    page.select_option(
                        ".panel:has(h2:text-is('Abscissa')) "
                        "label.field:has-text('Horizontal axis') select", basis)
                    settle(3_000)
                    open_panel("Components")
                    assert not el_chip("D").is_disabled(), \
                        "D chip should be enabled on " + basis
                    if basis == "rig":
                        # regression: D was gated off non-energy abscissas
                        # before globalsplinefit accepted "D" as a target
                        el_chip("D").click()
                        settle(3_000)
                        assert "D" in page.text_content("svg.chart")
                        el_chip("D").click()
                        settle(2_000)

            # ----------------------------------------- quantity / axis rules
            with check("quantity: nucleon flux restricts abscissa"):
                open_panel("Abscissa")
                quantity = page.locator(
                    ".settings-popover label.field:has-text('Plot') select")
                axis = page.locator(
                    ".panel:has(h2:text-is('Abscissa')) "
                    "label.field:has-text('Horizontal axis') select")
                quantity.select_option("nucleon")
                settle(5_000)
                assert axis.input_value() == "en"
                assert axis.locator("option").evaluate_all(
                    "opts => opts.map(o => o.value)") == ["en", "ekn"]
                assert "Nucleon Flux" in header_tag()

            with check("quantity: lnA moments expose every abscissa"):
                open_panel("Abscissa")
                quantity = page.locator(
                    ".settings-popover label.field:has-text('Plot') select")
                axis = page.locator(
                    ".panel:has(h2:text-is('Abscissa')) "
                    "label.field:has-text('Horizontal axis') select")
                for value, title in (("mean_lna", "⟨ln A⟩"),
                                     ("var_lna", "σ²(ln A)")):
                    quantity.select_option(value)
                    settle(12_000)
                    assert axis.locator("option").count() == 5
                    assert title in page.text_content("svg.chart")
                open_panel("Components")
                assert "include every nucleus" in page.text_content(
                    ".panel:has(h2:text-is('Components'))")

            with check("quantity: nucleus flux restores every abscissa"):
                open_panel("Abscissa")
                page.locator(
                    ".settings-popover label.field:has-text('Plot') select"
                ).select_option("nucleus")
                settle(5_000)
                axis = page.locator(
                    ".panel:has(h2:text-is('Abscissa')) "
                    "label.field:has-text('Horizontal axis') select")
                assert axis.locator("option").count() == 5
                assert "Nucleus Flux" in header_tag()

            # -------------------------------------------------- components
            with check("components: toggle all-particle + each group"):
                open_panel("Components")
                assert page.query_selector(
                    ".series-popover .panel:has(h2:text-is('Model'))")
                for name in ("all-particle", "p", "He", "O*", "Fe*"):
                    row = page.locator(".seriesrow",
                                       has_text=re.compile(f"^{re.escape(name)}$"))
                    row.click(); page.wait_for_timeout(120)
                    row.click(); page.wait_for_timeout(120)
                settle(1_200)

            with check("components: deuterium selectable on E basis"):
                assert not el_chip("D").is_disabled()
                el_chip("D").click()
                settle(3_000)
                assert "D" in page.text_content("svg.chart")
                el_chip("D").click()
                settle(2_000)

            with check("components: every element chip on, then reset"):
                n = page.locator(".elgrid button:not(:disabled)").count()
                for i in range(n):
                    page.locator(".elgrid button:not(:disabled)").nth(i).click()
                    page.wait_for_timeout(40)
                settle(12_000)   # all-elements evaluation is the heaviest
                assert page.locator(".elgrid button.on").count() == n
                page.click(".resetbtn")
                settle(6_000)
                assert page.locator(".elgrid button.on").count() == 0
                assert page.locator(".resetbtn").is_disabled()

            # ------------------------------------------------------ display
            with check("display: gamma slider (0, 3.2, 2.7)"):
                slider = page.locator(".displaydock input[type=range]").first
                for v in ("0", "3.2", "2.7"):
                    slider.fill(v); page.wait_for_timeout(150)
                settle(800)

            with check("display: log/linear"):
                page.click(".displaydock .seg button:text-is('Linear')")
                page.wait_for_timeout(300)
                page.click(".displaydock .seg button:text-is('Log')")
                settle(600)

            with check("display: ratio-to-total view"):
                page.click(".displaydock .seg button:text-is('Fraction')")
                settle(800)
                assert "Φ / Φ(all-particle)" in page.text_content("svg.chart")
                page.click(".displaydock .seg button:text-is('Flux')")
                settle(600)

            with check("nav: hover on/off toolbar toggle"):
                page.mouse.move(900, 420); page.wait_for_timeout(300)
                assert page.query_selector(".hoverbox")
                page.click(".plottools button[title*='Hover']")
                page.mouse.move(880, 420); page.wait_for_timeout(300)
                assert not page.query_selector(".hoverbox"), "hover not disabled"
                page.click(".plottools button[title*='Hover']")
                page.mouse.move(20, 20)

            with check("display: line weight is immediate"):
                line = page.locator("svg.chart path[fill='none']").first
                before = float(line.get_attribute("stroke-width"))
                weight = page.locator(
                    ".lineweight-control input[type=range]")
                weight.fill("1.6")
                page.wait_for_timeout(180)
                after = float(line.get_attribute("stroke-width"))
                assert after > before * 1.5
                assert "busy" not in (
                    page.get_attribute(".statuspill", "class") or "")
                weight.fill("1")
                settle(500)

            with check("display: bands off/on + opacity"):
                bands = page.locator(
                    ".band-control .switchcheck:has-text('Bands')")
                bands.click(); page.wait_for_timeout(200)
                bands.click(); page.wait_for_timeout(200)
                page.locator(".opacity-control input[type=range]").fill("0.4")
                settle(600)

            with check("display: overlay hatch bands (needs 2nd model)"):
                open_panel("Model")
                page.select_option(
                    "label.field:has-text('Add model') select", "2026-USO")
                settle(3_500)
                page.locator(
                    ".band-control .switchcheck:has-text('Compared')").click()
                settle(800)
                assert page.query_selector("svg.chart pattern"), "no hatch"
                page.locator(
                    ".band-control .switchcheck:has-text('Compared')").click()
                open_panel("Model")
                page.click(".modelrow .mdel >> nth=-1")
                settle(2_500)

            # ---------------------------------------------------- modulation
            with check("modulation: LIS / SC24"):
                open_panel("Solar modulation")
                page.click(".settings-popover .seg button:text-is('LIS')")
                settle(3_000)
                page.click(".settings-popover .seg button:text-is('SC24')")
                settle(2_500)

            with check("modulation: custom interval + edges"):
                page.click(".settings-popover .seg button:text-is('Interval')")
                settle(4_000)
                page.fill("input[type=month] >> nth=0", "1975-01")
                settle(4_000)
                page.fill("input[type=month] >> nth=1", "2025-12")
                settle(6_000)

            with check("modulation: reversed interval auto-sorts"):
                page.fill("input[type=month] >> nth=0", "2020-06")
                settle(4_000)
                page.fill("input[type=month] >> nth=1", "2010-01")
                settle(5_000)
                assert "2010/01–2020/06" in header_tag()
                page.click(".settings-popover .seg button:text-is('SC24')")
                settle(2_500)

            # ------------------------------------------------------ advanced
            with check("advanced: energy scale 0.8 / 1.2 / 1.0"):
                open_panel("Advanced")
                inp = by_label("Energy-scale").locator("input")
                for v in ("0.8", "1.2", "1.0"):
                    inp.fill(v); inp.dispatch_event("change")
                    page.wait_for_timeout(400)
                settle(2_500)

            with check("advanced: rigidity cutoff 30 / 0"):
                inp = by_label("Rigidity cutoff").locator("input")
                inp.fill("30"); inp.dispatch_event("change")
                settle(3_000)
                inp.fill("0"); inp.dispatch_event("change")
                settle(2_500)

            with check("advanced: grid points 240 / 960 / 480"):
                for n in ("240", "960", "480"):
                    page.click(f".settings-popover .seg button:text-is('{n}')")
                    settle(3_500)
                    assert f"{n} pts" in page.text_content(".statuspill")

            # ---------------------------------------------- plot navigation
            with check("nav: box-zoom sets window + custom y"):
                page.mouse.move(700, 300); page.mouse.down()
                page.mouse.move(1200, 650, steps=6); page.mouse.up()
                settle(3_000)
                assert "10⁻¹–10¹¹" not in header_tag()
                assert not page.locator(
                    ".plottools button[title*='Reset']").is_disabled()

            with check("nav: home button resets"):
                page.click(".plottools button[title*='Reset']")
                settle(2_500)
                assert "10⁻¹–10¹¹" in header_tag()

            with check("nav: x-axis drag zooms x only (y frozen)"):
                f = plot_frame()
                t0 = yticks()
                yax = f["b"] + 25          # on the x-axis tick strip
                page.mouse.move(f["l"] + 0.45 * (f["r"] - f["l"]), yax)
                page.mouse.down()
                page.mouse.move(f["l"] + 0.75 * (f["r"] - f["l"]), yax, steps=5)
                page.mouse.up()
                settle(3_000)
                assert "10⁻¹–10¹¹" not in header_tag(), "x window unchanged"
                assert yticks() == t0, "y scale moved on an x-axis drag"
                page.click(".plottools button[title*='Reset']")
                settle(2_500)

            with check("nav: y-axis drag zooms y only (x frozen)"):
                f = plot_frame()
                t0 = yticks()
                xax = f["l"] - 30          # on the y-axis tick strip
                page.mouse.move(xax, f["t"] + 0.30 * (f["b"] - f["t"]))
                page.mouse.down()
                page.mouse.move(xax, f["t"] + 0.65 * (f["b"] - f["t"]), steps=5)
                page.mouse.up()
                settle(3_000)
                assert "10⁻¹–10¹¹" in header_tag(), "x window moved on a y-axis drag"
                assert yticks() != t0, "y scale unchanged"
                page.click(".plottools button[title*='Reset']")
                settle(2_500)

            with check("nav: auto y-scale fits the visible x-window"):
                # zoom into the high-energy end, then move gamma -> y resets
                # to AUTO; the auto range must come from the means INSIDE the
                # window, so it differs from the full-domain auto range at
                # the same gamma
                page.mouse.move(900, 300); page.mouse.down()
                page.mouse.move(1250, 700, steps=6); page.mouse.up()
                settle(3_000)
                gamma = page.locator(".displaydock input[type=range]").first
                gamma.fill("2")
                settle(2_500)
                zoomed = yticks()
                page.click(".plottools button[title*='Reset']")
                settle(2_500)
                assert zoomed != yticks(), "auto y-scale ignored the x-window"
                gamma.fill("2.7")
                settle(1_500)

            with check("nav: pan drags the window"):
                page.click(".plottools button[title*='Pan']")
                page.mouse.move(1000, 500); page.mouse.down()
                page.mouse.move(850, 540, steps=5); page.mouse.up()
                settle(3_000)
                assert not page.locator(
                    ".plottools button[title*='Reset']").is_disabled()
                page.click(".plottools button[title*='Zoom'], .plottools button[title*='Box']")

            with check("nav: double-click = home"):
                page.dblclick("svg.chart", position={"x": 900, "y": 500})
                settle(2_500)
                assert "10⁻¹–10¹¹" in header_tag()

            with check("nav: wheel pinch-zoom (trackpad)"):
                page.mouse.move(1000, 500)
                page.keyboard.down("Control")
                page.mouse.wheel(0, -240)
                page.keyboard.up("Control")
                settle(3_000)
                assert "10⁻¹–10¹¹" not in header_tag()

            with check("nav: wheel two-finger pan + home"):
                page.mouse.wheel(0, 120)
                settle(3_000)
                page.click(".plottools button[title*='Reset']")
                settle(2_500)
                assert "10⁻¹–10¹¹" in header_tag()

            # --------------------------------------------------------- hover
            with check("hover: crosshair readout with series values"):
                page.mouse.move(900, 420)
                page.wait_for_timeout(400)
                assert page.query_selector(".hoverbox")
                assert page.locator(".hoverbox .hrow").count() >= 4
                page.mouse.move(20, 20)

            # ------------------------------------------------------- exports
            with check("export: data table modal + its CSV download"):
                open_panel("Export")
                page.click("button:text-is('Data table')")
                page.wait_for_selector(".modal table.data", timeout=10_000)
                assert page.locator(".modal table.data tbody tr").count() >= 100
                with page.expect_download(timeout=30_000) as dl:
                    page.click(".modal header button:has-text('Download CSV')")
                assert dl.value.suggested_filename.startswith("gsf_view")
                page.click(".modal header button >> nth=-1")

            with check("export: CSV per model (2 active -> 2 files)"):
                open_panel("Model")
                page.select_option(
                    "label.field:has-text('Add model') select", "2026-USO")
                settle(4_500)
                open_panel("Export")
                with page.expect_download(timeout=90_000) as dl:
                    page.click("button:text-is('CSV')")
                f1 = dl.value.suggested_filename
                dl2 = page.wait_for_event("download", timeout=90_000)
                names = {f1, dl2.suggested_filename}
                assert any("uso" in n for n in names), names
                open_panel("Model")
                page.click(".modelrow .mdel >> nth=-1")
                settle(3_000)

            with check("export: live view snapshot SVG"):
                open_panel("Export")
                with page.expect_download(timeout=30_000) as dl:
                    page.click("button:has-text('View → SVG')")
                assert dl.value.suggested_filename.endswith(".svg")

            if not args.fast:
                for fmt in ("pdf", "svg", "png"):
                    with check(f"export: publication figure {fmt}"):
                        page.click(f".export-popover .seg button:text-is('{fmt}')")
                        with page.expect_download(timeout=300_000) as dl:
                            page.click("button:text-is('Download')")
                        assert dl.value.suggested_filename.endswith(f".{fmt}")

            # --------------------------------------------------------- theme
            with check("theme: toggle + persistence"):
                before = page.evaluate("document.documentElement.dataset.theme")
                page.click(".iconbtn[title='Switch theme']")
                page.wait_for_timeout(400)
                after = page.evaluate("document.documentElement.dataset.theme")
                assert after != before
                assert page.evaluate("localStorage.getItem('gsfTheme')") == after
                page.click(".iconbtn[title='Switch theme']")

            with check("theme: fresh browser follows prefers-color-scheme"):
                c2 = browser.new_context(color_scheme="light",
                                         viewport={"width": 1200, "height": 800})
                p2 = c2.new_page()
                p2.goto(URL, timeout=60_000)
                p2.wait_for_timeout(1_000)
                assert p2.evaluate(
                    "document.documentElement.dataset.theme") == "light"
                c2.close()

            # ---------------------------------------------------- responsive
            with check("responsive: 420px mobile renders"):
                c3 = browser.new_context(viewport={"width": 420, "height": 560})
                p3 = c3.new_page()
                p3.goto(URL, timeout=60_000)
                p3.wait_for_selector("svg.chart path", timeout=300_000)
                assert p3.locator(".commandbtn").count() == 3
                legend = p3.locator(".legend-toggle")
                assert legend.get_attribute("aria-expanded") == "false"
                legend.click()
                assert legend.get_attribute("aria-expanded") == "true"
                for command, selector in (
                        ("Series", ".series-popover"),
                        ("Settings", ".settings-popover"),
                        ("Export", ".export-popover")):
                    p3.click(f".commandbtn:text-is('{command}')")
                    pane = p3.locator(selector)
                    scroller = pane.locator(".panescroll")
                    assert scroller.evaluate(
                        "e => getComputedStyle(e).overflowY") == "auto"
                    dims = scroller.evaluate(
                        "e => ({h:e.clientHeight, sh:e.scrollHeight})")
                    assert dims["sh"] > dims["h"], (command, dims)
                    assert pane.locator(".scrollrail.active").count() == 1
                    scroller.evaluate("e => { e.scrollTop = e.scrollHeight; }")
                    assert scroller.evaluate("e => e.scrollTop") > 0
                    p3.click(f"{selector} .closebtn")
                p3.click(".commandbtn:text-is('Settings')")
                sheet = p3.locator(".overlay-surface")
                assert sheet.is_visible()
                assert sheet.bounding_box()["y"] > 100
                c3.close()

            # -------------------------------------------------------- touch
            # iPad-class device (short viewport + touch): chart gestures and
            # pane fit — regression guard for the 2026-08 touch support.
            c4 = browser.new_context(viewport={"width": 1024, "height": 700},
                                     has_touch=True, is_mobile=True)
            p4 = c4.new_page()

            def p4_window():
                return p4.evaluate(
                    "() => document.querySelector('header').innerText")

            with check("touch: boots on iPad-size viewport"):
                p4.goto(URL, timeout=60_000)
                p4.wait_for_selector("svg.chart path", timeout=300_000)
                p4.wait_for_timeout(1500)

            tbox = p4.locator("svg.chart").bounding_box() or {}
            tx = tbox.get("x", 0) + tbox.get("width", 800) * 0.55
            ty = tbox.get("y", 0) + tbox.get("height", 500) * 0.45

            with check("touch: tap shows hover readout"):
                p4.touchscreen.tap(tx, ty)
                p4.wait_for_timeout(600)
                assert p4.evaluate(
                    "() => document.body.innerText.includes('all-particle')")

            with check("touch: pinch zooms the x-window"):
                w0 = p4_window()
                cdp = c4.new_cdp_session(p4)
                cdp.send("Input.synthesizePinchGesture",
                         {"x": tx, "y": ty, "scaleFactor": 2.5,
                          "relativeSpeed": 400,
                          "gestureSourceType": "touch"})
                p4.wait_for_timeout(1000)
                assert p4_window() != w0, "window unchanged after pinch"

            with check("touch: double-tap homes the view"):
                w1 = p4_window()
                p4.touchscreen.tap(tx, ty)
                p4.wait_for_timeout(120)
                p4.touchscreen.tap(tx + 3, ty + 2)
                p4.wait_for_timeout(1000)
                assert p4_window() != w1, "window unchanged after double-tap"

            with check("touch: panes fit and scroll to the bottom"):
                for command, selector in (
                        ("Series", ".series-popover"),
                        ("Settings", ".settings-popover"),
                        ("Export", ".export-popover")):
                    p4.click(f".commandbtn:text-is('{command}')")
                    p4.wait_for_timeout(300)
                    state = p4.locator(selector).evaluate("""pane => {
                        const r = pane.getBoundingClientRect();
                        const sc = pane.querySelector('.panescroll');
                        sc.scrollTop = sc.scrollHeight;
                        return {fits: r.bottom <= innerHeight + 1,
                                end: sc.scrollTop + sc.clientHeight
                                     >= sc.scrollHeight - 2};
                    }""")
                    assert state["fits"], (command, "pane clipped")
                    assert state["end"], (command, "cannot reach bottom")
                    p4.click(f"{selector} .closebtn")
                    p4.wait_for_timeout(200)
            c4.close()

            # ----------------------------------------------- panel collapse
            with check("panels: every header expands and collapses"):
                for t in ("Model", "Components", "Abscissa",
                          "Solar modulation", "Advanced", "Export"):
                    open_panel(t)
                    sel = f".panel:has(h2:text-is('{t}'))"
                    page.click(f"{sel} > header")
                    page.wait_for_timeout(100)
                    page.click(f"{sel} > header")
                    page.wait_for_timeout(100)

            browser.close()
    finally:
        srv.terminate()

    width = max(len(n) for n, _, _ in results)
    fails = 0
    for name, status, info in results:
        fails += status == "FAIL"
        print(f"{'✓' if status == 'PASS' else '✗'} {name:<{width}}  {info}")
    print(f"\n{len(results) - fails}/{len(results)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
