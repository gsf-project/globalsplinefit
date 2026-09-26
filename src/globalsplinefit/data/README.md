# Model data

One directory per registered model version (see `MODEL_VERSIONS` in
`data_management.py`). Every file except the solar-modulation tables is output
of the GSF fit: spline knots (`knots.dat`), amplitudes (`parameters.dat`), their
covariance (`covariance.dat`), the species list (`nuclei.dat`), the pivot grid of
the reduced representation (`reduced_pivots.dat`), and for the 2026 sets the
sub-leading extrapolation (`subleading.dat`) and a machine-readable record of the
fit (`fit_result.json`). They are distributed under the package license.

## Solar-modulation potentials (third-party)

The monthly force-field potentials are reconstructions published by their
authors and redistributed here with attribution. Please cite them when you use
the solar-modulated fluxes.

| File | Source | Used by |
|---|---|---|
| `solar_modulation.dat` (this directory) | Usoskin et al., [InspireHEP 1600902](https://inspirehep.net/literature/1600902) (the file header carries the full references) | `2026.1-USO`, `2025`, `2019`, `2017` |
| `2026.1*/solar_modulation.dat` | Ghelfi, Maurin, Derome et al., [InspireHEP 1474379](https://inspirehep.net/literature/1474379) (GMD φ time series) | `2026.1`, `2026.1-SIB23e`, `2026.1-EPOSLHCR` |

The `2026.1-USO` set also carries its own copy of the Usoskin table under
`2026.1-USO/solar_modulation.dat`.
