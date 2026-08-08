"""Gallery acceptance test — every tutorial page, in a headless browser.

The gallery is published twice per notebook (app view at /gallery/<name>/,
editor at /gallery/<name>/edit/). This asserts what a reader actually gets:

  * the app page lands WITHOUT code, whatever URL it was entered by
    (gallery_head.html defaults marimo's show-code parameter to false),
  * figures are on screen at first paint (the --execute snapshot) — a page
    that only renders after pyodide boots is a regression,
  * the controls are there and the figures react to them,
  * both download buttons resolve to a real file,
  * the editor variant runs and shows only the file panel in its sidebar,
  * no console errors anywhere.

Usage:
    pip install playwright && playwright install chromium
    python .github/scripts/test_gallery.py [--fast] [notebook ...]

Run it against a built site (`mkdocs build && .github/scripts/build_gallery.sh`).
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
PORT = 8241
NOTEBOOKS = sorted(p.stem for p in (ROOT / "examples").glob("*.py"))

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")


def wait_for(fn, limit):
    t0 = time.time()
    while time.time() - t0 < limit:
        if fn():
            return round(time.time() - t0, 1)
        time.sleep(0.5)
    return None


def check_notebook(browser, base, name, fast):
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors = []
    page.on(
        "console", lambda m: errors.append(m.text[:200]) if m.type == "error" else None
    )
    page.on("pageerror", lambda e: errors.append("pageerror: " + str(e)[:200]))

    # --- app view, entered with no query parameter at all
    page.goto(f"{base}/gallery/{name}/", wait_until="domcontentloaded")
    paint = wait_for(lambda: page.locator("img").count() > 0, 120)
    check(f"{name}: figures at first paint", paint is not None, f"{paint} s")
    check(
        f"{name}: lands in app view",
        "show-code=false" in page.url,
        page.url.split("/")[-1],
    )
    check(f"{name}: no code editors visible", page.locator(".cm-editor").count() == 0)

    n_figs = page.locator("img").count()
    check(f"{name}: figures rendered", n_figs > 0, f"{n_figs} figures")

    controls = page.locator("select, input[type=range], [role=slider]").count()
    check(f"{name}: controls present", controls > 0, f"{controls} controls")

    # download buttons must point at files that exist
    # A download control is in the pre-rendered snapshot, but that snapshot is
    # of a LOCAL run: SITE is empty when the export executes, so the cell hands
    # the file over from disk as a single data-URL button. Once the kernel is
    # live in the browser the cell re-runs with SITE set and swaps in the two
    # published URLs — that is what the reader ends up with, so the link check
    # belongs after the boot wait.
    check(
        f"{name}: download control at first paint",
        page.get_by_text("notebook (.").count() > 0,
    )

    if not fast:
        # Wait for the live kernel to take over from the snapshot: it has to
        # boot pyodide and micropip-install the wheel before the download cell
        # knows the site root, so poll instead of guessing a sleep.
        def _links():
            return page.eval_on_selector_all(
                "a[href$='.py'], a[href$='.ipynb']", "els => els.map(e => e.href)"
            )

        took = wait_for(lambda: len(_links()) == 2, 180)
        live = _links()
        check(
            f"{name}: two download buttons once live",
            len(live) == 2,
            f"{len(live)} after {took} s",
        )
        for href in live:
            try:
                with urllib.request.urlopen(href, timeout=15) as r:
                    ok, detail = r.status == 200, str(r.status)
            except Exception as exc:  # noqa: BLE001
                ok, detail = False, str(exc)[:60]
            check(f"{name}: {href.rsplit('/', 1)[-1]} downloads", ok, detail)

        # Compare EVERY figure, not just the first: on the deck the first
        # figure is the mass-group schematic, which the scaling slider does
        # not touch.
        def _srcs():
            return page.eval_on_selector_all(
                "img", "els => els.map(e => e.src.slice(-64))"
            )

        before = _srcs()
        slider = page.locator("input[type=range], [role=slider]")
        if slider.count() and before:
            slider.first.click()
            for _ in range(4):
                page.keyboard.press("ArrowRight")
            changed = wait_for(lambda: _srcs() != before, 120)
            check(
                f"{name}: a figure follows the slider",
                changed is not None,
                f"{changed} s",
            )

    # a failed cell (the wheel 404ing, say) renders as an error box, not a
    # console error — catch it explicitly
    cell_errors = page.evaluate(
        "() => document.querySelectorAll('.marimo-error, [data-testid=cell-error]').length"
    )
    check(f"{name}: no cell errors", cell_errors == 0, str(cell_errors))
    check(f"{name}: no console errors (app)", not errors, "; ".join(errors[:2]))
    page.close()

    # --- editable variant
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors2 = []
    page.on(
        "console", lambda m: errors2.append(m.text[:200]) if m.type == "error" else None
    )
    page.goto(f"{base}/gallery/{name}/edit/", wait_until="domcontentloaded")
    ready = wait_for(lambda: page.locator(".cm-editor").count() > 0, 120)
    check(f"{name}: editor loads", ready is not None, f"{ready} s")
    ran = wait_for(lambda: page.locator("img").count() > 0, 180)
    check(f"{name}: editor auto-runs", ran is not None, f"{ran} s")
    panels = page.evaluate("""() => {
        const sb = document.querySelector('[data-testid=chrome-sidebar]');
        return sb ? [...sb.querySelectorAll('[role=option]')]
                      .filter(e => e.offsetParent !== null)
                      .map(e => e.getAttribute('data-key')) : null;
    }""")
    check(f"{name}: sidebar trimmed to files", panels == ["files"], json.dumps(panels))
    check(f"{name}: no console errors (edit)", not errors2, "; ".join(errors2[:2]))
    page.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fast", action="store_true", help="skip the slower interaction checks"
    )
    ap.add_argument("notebooks", nargs="*", default=None)
    args = ap.parse_args()
    names = args.notebooks or NOTEBOOKS

    if not (SITE / "gallery").is_dir():
        sys.exit(
            "site/gallery not built — run mkdocs build && .github/scripts/build_gallery.sh"
        )

    srv = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(PORT), "-d", str(SITE)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.5)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for name in names:
                print(f"\n=== {name}")
                check_notebook(browser, f"http://localhost:{PORT}", name, args.fast)
            browser.close()
    finally:
        srv.terminate()

    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
