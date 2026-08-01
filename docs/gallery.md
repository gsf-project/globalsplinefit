# Tutorial gallery

The tutorials are [marimo](https://marimo.io) notebooks compiled to
WebAssembly — they run **entirely in your browser** (first load takes about a
minute while Python boots and packages install; the plots then appear
below each cell). Notebooks with controls (model version, parameters) are
live — change a value and the plots update. To modify the code itself, grab
the source from `examples/` and
[run it locally](getting_started.md#run-the-tutorials-locally) with marimo or
Jupyter.

<div class="grid cards" markdown>

-   **Model Deck**

    ---

    The one-look model summary: flux spectrum, composition fractions, ln A
    moments, relative uncertainty, nucleon flux, flux correlation — plus a
    back-to-back comparison between any two model versions.

    [:octicons-arrow-right-24: Open](model_deck/)

-   **Cosmic Ray Flux**

    ---

    Differential fluxes for the four GSF mass groups with 1-sigma error
    bands propagated from the parameter covariance.

    [:octicons-arrow-right-24: Open](cosmic_ray_flux/)

-   **Nucleon Flux**

    ---

    Proton and neutron fluxes per energy-per-nucleon — the input for
    atmospheric shower and lepton-flux calculations.

    [:octicons-arrow-right-24: Open](nucleon_flux/)

-   **Solar Modulation**

    ---

    Time-dependent solar modulation of the low-energy flux; local
    interstellar spectrum vs. modulated periods.

    [:octicons-arrow-right-24: Open](solar_modulation/)

-   **Rigidity Cutoff**

    ---

    Geomagnetic rigidity cutoff effects on the flux at Earth.

    [:octicons-arrow-right-24: Open](rigidity_cutoff/)

-   **Model Comparison 2017 vs 2025**

    ---

    What changed between the published GSF versions.

    [:octicons-arrow-right-24: Open](model_comparison_2017_vs_2025/)

-   **Reduced Model**

    ---

    `ReducedGSF`: pivot-based flux nuisance parameters with an exact
    covariance penalty, for use in downstream fits (daemonflux-style).

    [:octicons-arrow-right-24: Open](reduced_model/)

-   **Hybrid PCA**

    ---

    `HybridPCA`: low-rank representation of the flux covariance —
    validation, components, MC sampling.

    [:octicons-arrow-right-24: Open](hybrid_pca/)

</div>

The notebook sources live in
[`examples/`](https://github.com/gsf-project/globalsplinefit/tree/main/examples)
— plain Python files, exported to WebAssembly by the docs build. To get a
Jupyter `.ipynb`, run `uvx marimo export ipynb <notebook>.py -o <notebook>.ipynb`
on a downloaded source file.
