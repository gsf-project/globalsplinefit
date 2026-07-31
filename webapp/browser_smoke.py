"""In-browser smoke test for GSF Explorer (Preact + Pyodide).

Boots the real page in headless chromium and fails on any console/page error
or broken interaction. This is the acceptance gate: the desktop Python is not
the runtime — Pyodide ships its own package versions, so only in-browser
tests count.

Usage:
    pip install playwright && playwright install chromium
    python browser_smoke.py [--shots DIR]

Needs internet on first run (pyodide + preact CDN, ~25 MB, then cached).
"""

import argparse
import pathlib
import subprocess
import sys

from playwright.sync_api import sync_playwright

WEBAPP = pathlib.Path(__file__).resolve().parent
PORT = 8123


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", metavar="DIR",
                    help="also save desktop + mobile screenshots into DIR")
    args = ap.parse_args()

    srv = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(PORT), "-d", str(WEBAPP)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    errors = []
    try:
        with sync_playwright() as p:
            b = p.chromium.launch()
            page = b.new_page(viewport={"width": 1680, "height": 1000})
            page.on("console", lambda m: m.type == "error"
                    and errors.append(f"[console] {m.text[:300]}"))
            page.on("pageerror",
                    lambda e: errors.append(f"[page] {str(e)[:300]}"))
            page.goto(f"http://localhost:{PORT}/", timeout=60_000)
            page.wait_for_selector("svg.chart path", timeout=300_000)
            page.wait_for_timeout(1_000)
            print("boot: chart rendered")

            # instant display dock controls (no worker round-trip)
            page.locator(".displaydock input[type=range]").first.fill("1.5")
            page.click(".displaydock .seg button:has-text('Linear')")
            page.click(".displaydock .seg button:has-text('Log')")

            # worker round-trips in task-specific command surfaces
            page.click(".commandbtn:text-is('Series')")
            page.click(".elgrid button:has-text('Fe')")
            page.click(".commandbtn:text-is('Settings')")
            page.click(".settings-popover .seg button:has-text('LIS')")
            page.click(".aboutbtn")
            page.wait_for_timeout(600)
            page.click(".modal header button")
            page.wait_for_timeout(3_000)

            # second model overlay
            page.click(".commandbtn:text-is('Series')")
            page.select_option(
                "label.field:has-text('Add model') select", value="2026-USO")
            page.wait_for_timeout(3_000)

            # toolbar: box zoom, pan, home
            page.mouse.move(760, 300); page.mouse.down()
            page.mouse.move(1200, 650, steps=6); page.mouse.up()
            page.wait_for_timeout(2_000)
            page.click(".plottools button[title*='Pan']")
            page.mouse.move(1000, 500); page.mouse.down()
            page.mouse.move(900, 540, steps=4); page.mouse.up()
            page.wait_for_timeout(2_000)
            page.click(".plottools button[title*='Reset']")
            page.wait_for_timeout(2_000)

            # theme toggle persists
            page.click(".iconbtn[title='Switch theme']")
            page.wait_for_timeout(400)
            assert page.evaluate("localStorage.getItem('gsfTheme')") in (
                "light", "dark"), "theme not persisted"
            print("interactions: ok")

            # exports: CSV (worker), publication PDF (lazy matplotlib), SVG
            page.click(".commandbtn:text-is('Export')")
            for label, timeout in (("CSV", 60_000), ("Download", 300_000),
                                   ("View → SVG", 30_000)):
                with page.expect_download(timeout=timeout) as dl:
                    page.click(f"button:has-text('{label}')")
                print(f"export {label!r}: {dl.value.suggested_filename}")

            if args.shots:
                out = pathlib.Path(args.shots)
                out.mkdir(parents=True, exist_ok=True)
                if page.query_selector(".export-popover"):
                    page.click(".commandbtn:text-is('Export')")
                page.screenshot(path=out / "smoke_desktop.png")
                page.click(".iconbtn[title='Switch theme']")
                page.wait_for_timeout(250)
                page.screenshot(path=out / "smoke_desktop_alt.png")
                page.set_viewport_size({"width": 760, "height": 800})
                page.wait_for_timeout(250)
                page.screenshot(path=out / "smoke_compact.png")
                page.set_viewport_size({"width": 520, "height": 760})
                page.wait_for_timeout(250)
                page.screenshot(path=out / "smoke_autocollapsed.png")
                m = b.new_page(viewport={"width": 390, "height": 844})
                m.goto(f"http://localhost:{PORT}/", timeout=60_000)
                m.wait_for_selector("svg.chart path", timeout=300_000)
                m.wait_for_timeout(1_200)
                m.screenshot(path=out / "smoke_mobile.png")
                m.click(".commandbtn:text-is('Series')")
                m.wait_for_timeout(250)
                m.screenshot(path=out / "smoke_mobile_series.png")
                m.click(".series-popover .closebtn")
                m.click(".commandbtn:text-is('Settings')")
                m.wait_for_timeout(250)
                m.screenshot(path=out / "smoke_mobile_sheet.png")
            b.close()
    finally:
        srv.terminate()

    for e in errors:
        print(e)
    print("PASS" if not errors else "FAIL")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
