# Getting Started

Four ways to use GSF, roughly in order of commitment:

1. **[GSF Explorer](explorer.md)** —
   interactive model exploration in the browser, nothing to install.
2. **[Tutorial gallery](gallery.md)** — notebooks in the browser.
3. **Run a tutorial locally** — download one notebook, run it on your machine.
4. **Use as a package** — `import globalsplinefit` in your own code.

## Install

=== "pip"

    ```bash
    pip install globalsplinefit
    ```

=== "uv"

    ```bash
    uv add globalsplinefit          # into a project
    uv pip install globalsplinefit  # into the active environment
    ```

    New to [uv](https://docs.astral.sh/uv/)? It is a fast drop-in replacement
    for pip and virtualenv —
    [installation](https://docs.astral.sh/uv/getting-started/installation/).
    Everything below works with plain pip too; pick the tab you prefer.

!!! note
    Until the first PyPI release lands, install from GitHub instead:

    === "pip"

        ```bash
        pip install "globalsplinefit @ git+https://github.com/gsf-project/globalsplinefit"
        ```

    === "uv"

        ```bash
        uv pip install "globalsplinefit @ git+https://github.com/gsf-project/globalsplinefit"
        ```

### What gets installed

The base package depends only on **numpy** and **scipy** — that is all you
need to evaluate the model. Everything else is an optional extra:

| Install | Adds | For |
|---|---|---|
| `globalsplinefit` | numpy, scipy | using the model in your own code |
| `globalsplinefit[examples]` | matplotlib, marimo, nbformat, crflux | running the tutorial notebooks |
| `globalsplinefit[docs]` | mkdocs + the examples extra | building this documentation |
| `globalsplinefit[dev]` | test, docs and examples extras | contributing to the package |

Quick check:

```python
import numpy as np
from globalsplinefit import GSFEnergy

gsf = GSFEnergy()  # default: newest 2026 revision
print(gsf.flux(np.logspace(3, 5, 3), "p"))
```

See the [User Guide](user_guide.md) for model classes, versions, and units.

## Run a tutorial locally

Every page in the [tutorial gallery](gallery.md) ends with download buttons
for the notebook — as a [marimo](https://marimo.io) notebook (`.py`) or as a
Jupyter notebook (`.ipynb`). No clone required:

=== "pip"

    ```bash
    pip install "globalsplinefit[examples]"
    # download e.g. cosmic_ray_flux.py from the gallery page, then:
    marimo edit cosmic_ray_flux.py     # interactive editor
    python cosmic_ray_flux.py          # run straight through
    ```

=== "uv"

    ```bash
    uv pip install "globalsplinefit[examples]"
    # download e.g. cosmic_ray_flux.py from the gallery page, then:
    uv run marimo edit cosmic_ray_flux.py
    uv run python cosmic_ray_flux.py
    ```

Prefer Jupyter? Download the `.ipynb` from the same page and open it with
`jupyter lab`.

The notebooks run unchanged locally: the cell that fetches the package inside
the browser is skipped outside it, so your installed version is used.

## Run the GSF Explorer locally

The Explorer is a static web app (Pyodide — Python in the browser). From a
clone of the repository:

```bash
python webapp/serve.py          # http://127.0.0.1:8123, default port 8123
```

If you changed the package source, rebuild the wheel the Explorer loads
first: `webapp/update_wheel.sh`.

## Build from source

For contributing to the package itself. Development uses
[uv](https://docs.astral.sh/uv/); `uv.lock` is committed.

```bash
git clone https://github.com/gsf-project/globalsplinefit.git
cd globalsplinefit
uv sync --all-extras        # create/refresh .venv with all extras
uv run pytest tests/        # run the test suite
uv build                    # build sdist + wheel into dist/
```

Build and preview this documentation site:

```bash
uv sync --extra docs
uv run mkdocs serve         # core docs at http://127.0.0.1:8000
```

The browser tutorials under `/gallery/` are produced in CI by
`marimo export html-wasm`. To build the whole gallery locally with the same
script CI uses:

```bash
uv run mkdocs build
.github/scripts/build_gallery.sh                    # ~1 min for all notebooks
```

Each notebook is published twice: the app view at `/gallery/<name>/` (figures
first, code behind the "Show code" menu item) and a full in-browser editor at
`/gallery/<name>/edit/`. CI additionally runs
`.github/scripts/test_gallery.py`, which loads every published page in a
headless browser and fails the build if a tutorial renders without figures —
a contributor check, not something you need in order to read the docs.
