# Getting Started

Four ways to use GSF, roughly in order of commitment:

1. **[GSF Explorer](explorer.md)** —
   interactive model exploration in the browser, nothing to install.
2. **[Tutorial gallery](gallery.md)** — notebooks running in the browser,
   editable and downloadable.
3. **Run notebooks locally** — marimo or Jupyter, full filesystem access
   (e.g. for custom fit directories).
4. **Use as a package** — `import globalsplinefit` in your own code.

## Install

```bash
pip install globalsplinefit
```

!!! note
    Until the first PyPI release lands, install from GitHub instead:
    ```bash
    pip install "globalsplinefit @ git+https://github.com/gsf-project/globalsplinefit"
    ```

The core package depends only on `numpy` and `scipy`. Quick check:

```python
import numpy as np
from globalsplinefit import GSFEnergy

gsf = GSFEnergy()  # default: version "2026"
print(gsf.flux(np.logspace(3, 5, 3), "p"))
```

See the [User Guide](user_guide.md) for model classes, versions, and units.

## Run the tutorials locally

The tutorials in [`examples/`](https://github.com/gsf-project/globalsplinefit/tree/main/examples)
are [marimo](https://marimo.io) notebooks — plain Python files. From a clone:

```bash
git clone https://github.com/gsf-project/globalsplinefit.git
cd globalsplinefit
uv run marimo edit examples/cosmic_ray_flux.py   # interactive editor
uv run python examples/cosmic_ray_flux.py        # run straight through
```

Prefer Jupyter? Convert any notebook:

```bash
uvx marimo export ipynb examples/cosmic_ray_flux.py -o cosmic_ray_flux.ipynb
jupyter lab cosmic_ray_flux.ipynb
```

## Run the GSF Explorer locally

The Explorer is a static web app (Pyodide — Python in the browser); the
local server is only needed to serve the files:

```bash
python webapp/serve.py          # http://127.0.0.1:8123, default port 8123
```

In **Settings → Abscissa**, choose nucleus flux, nucleon flux, ⟨ln A⟩, or
σ²(ln A). Nucleus flux and the composition moments support all five axes;
nucleon flux is restricted to total or kinetic energy per nucleon.

If you changed the package source, rebuild the wheel the Explorer loads
first: `webapp/update_wheel.sh`.

## Build from source

Development uses [uv](https://docs.astral.sh/uv/); `uv.lock` is committed.

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
`marimo export html-wasm`; to build one locally:

```bash
uv run marimo export html-wasm examples/cosmic_ray_flux.py -o site/gallery/cosmic_ray_flux --mode run --show-code
```
