# User Guide

Installation and first steps: [Getting Started](getting_started.md).

## Model Classes

One class per input variable, sharing the same interface
(full signatures in the [API Reference](api_reference.md)):

### GSFEnergy

[`GSFEnergy`][globalsplinefit.GSFEnergy] —
**Input**: Total energy per nucleus [GeV]
**Output**: Differential flux [particles/(m² s sr GeV)]

### GSFEnergyPerNucleon

[`GSFEnergyPerNucleon`][globalsplinefit.GSFEnergyPerNucleon] —
**Input**: Total energy per nucleon [GeV/nucleon] (use
[`GSFKineticEnergyPerNucleon`][globalsplinefit.GSFKineticEnergyPerNucleon]
for kinetic energy per nucleon)
**Output**: Nucleon flux [nucleons/(m² s sr GeV)]

### GSFRigidity

[`GSFRigidity`][globalsplinefit.GSFRigidity] —
**Input**: Magnetic rigidity [GV]
**Output**: Modulated flux [particles/(m² s sr GV)]

## Model Versions

All model classes accept a `version` parameter selecting the fitted parameter set:

```python
gsf = GSFEnergy()                           # default 2026 fit
gsf_uso = GSFEnergy(version="2026-USO")     # Usoskin modulation variant
gsf_s23e = GSFEnergy(version="2026.0-SIB23e")
gsf_epos = GSFEnergy(version="2026.0-EPOSLHCR")
gsf_2025 = GSFEnergy(version="2025")        # historical release
```

Physical model versions are named `<line>.<revision>[-<physics classifier>]`:
`2026.0` is the first revision of the 2026 line, `2026.0-USO` its
Usoskin-potential variant, `2026.0-SIB23e` its SIBYLL-2.3e-only variant. A data or fit patch is published as a new revision (`2026.1`, ...).
An unrevisioned name (`"2026"`, `"2026-USO"`) resolves to the newest
registered revision of that line and variant — use it to follow patches
automatically, or pass the revisioned name to pin one
(`resolve_version("2026")` shows the resolution). Bare names `2025`, `2019`,
`2017` select the earlier published sets.

| Version | Status | Covering | Modulation | Description |
|---------|--------|----------|------------|-------------|
| `"2026.0"` | **current -- default** | SIBYLL-2.3e / EPOS-LHC-R mixture | Ghelfi--Maurin--Derome (**baseline**) | Default 2026 fit |
| `"2026.0-USO"` | current -- alternative | SIBYLL-2.3e / EPOS-LHC-R mixture | Usoskin 2017 (alternative) | The same fit with the other potential; use it to gauge the solar-modulation systematic |
| `"2026.0-SIB23e"` | current -- single interpretation | Auger FD-2026 SIBYLL-2.3e only | Ghelfi--Maurin--Derome | For applications needing a definite hadronic model |
| `"2026.0-EPOSLHCR"` | current -- variant | Auger FD-2026 EPOS-LHC-R only | Ghelfi--Maurin--Derome | The other single-interpretation half of the mixture |
| `"2025"` | historical | PDG scale factors | Usoskin | GSF 2025 published release |
| `"2019"` | historical | PDG scale factors | Usoskin | GSF 2019 published release |
| `"2017"` | historical | PDG scale factors | Usoskin | Original GSF release (Dembinski et al. 2017) |

`get_available_versions(include_historical=False)` returns just the current
sets, and `version_info(version)` reports any version's status, covering and
modulation potential. A loaded model also carries its own provenance:

```python
gsf.version                      # -> "2026.0", the revision actually loaded
gsf.params.provenance["covering"]                  # the air-shower interpretation
gsf.params.provenance["solar_modulation_source"]   # GMD or USO
```

### What "covering" means, and why both current sets are a mixture

Above ~10^8 GeV the mass composition inferred from air-shower data depends on the
hadronic interaction model used to interpret it. Both current sets use an
equal-weight **parameter-level mixture** of the Auger FD-2026 SIBYLL-2.3e and
EPOS-LHC-R interpretations: parameters are the mean of the two fits, and the
covariance carries an additional rank-one between-model term. The published band
therefore spans both interpretations where they diverge and collapses to the
ordinary fit covariance where they agree (below ~2x10^8 GeV, where the two
coincide). `2026.0-SIB23e` and `2026.0-EPOSLHCR` are the two halves of the mixture.

The 2026 sets are fitted in isotope format (deuterium and the ³He/⁴He
split are carried explicitly). The choice of solar modulation potential
affects only the local interstellar spectrum below ~10 GV: relative to
2026, the Usoskin potential yields a 10--14% lower interstellar
spectrum below 2 GV, with the two converging above ~10 GV. All
higher-energy results are identical.

Each set is re-modulated with the potential it was demodulated with: every
current set includes its own monthly `phi(t)` table
(`<version>/solar_modulation.dat` — Ghelfi--Maurin--Derome for all but
2026-USO, which carries the Usoskin table), while the historical sets use the
bundled Usoskin table at the package data root. Because each LIS is paired
with its own potential, the fluxes at Earth agree far better than the LIS do.

## Particle Groups

All models support these cosmic ray groups:

| Group | Description | Atomic Numbers |
|-------|-------------|----------------|
| `"H"` | Hydrogen group (p and D when available) | Z = 1 |
| `"He"` | Helium isotopes | Z = 2 |
| `"O*"` or `"CNO"` | Light/intermediate group | Z = 3--9 |
| `"Fe*"` or `"heavy"` | Heavy group | Z = 10--28 |

Use `"p"`, `"O"`, or `"Fe"` for the individual proton, oxygen, or iron
species rather than the complete group.

## Sub-leading Elements and High-Energy Extrapolation

Within each mass group only the *leading* element carries its own B-spline; the
remaining *sub-leading* species are tied to their group leader through a
rigidity-dependent flux ratio. Inside the range covered by direct data this
ratio follows the fitted splines. **Above each species' top knot**
$R_{\text{max}}$ (the highest rigidity at which that element has data), the
ratio is extrapolated as a power law in rigidity that saturates to a constant:

$$
\frac{J_j(R)}{J_L(R)} = w_j \left(\frac{\min(R,\,R_{\text{sat}})}{R_{\text{max},j}}\right)^{s_j},
\qquad R_{\text{sat}} = 5\ \text{PV},
$$

where $J_L$ is the group-leader flux. The normalization $w_j$ and the slope
$s_j$ are **anchored to the data** — obtained from an error-weighted power-law
fit to the measured member-to-leader ratio over the last decade of the
element's direct data. The saturation rigidity $R_{\text{sat}} = 5$ PV sits
near the proton knee, motivated by the common galactic origin and the assumed
similarity of transport effects above it.

For the `2026` and `2026-USO` sets these per-element parameters are in
`data/<version>/subleading.dat` (columns `Z A norm slope`). Without the file
(`2017`, `2019`, `2025`), or with a zero slope, the ratio is constant above
$R_{\text{max}}$.

!!! note
    The norm and slope are best-fit point estimates; their uncertainty is not
    propagated (`covariance.dat` stores the four group leaders only). The
    saturation constant is exposed as
    `globalsplinefit.model.SUBLEADING_SAT_LNR`.

## Uncertainty Quantification

Errors and covariances:

```python
flux = gsf.flux(energy, "p")
error = gsf.error(energy, "p")

# Relative uncertainty
rel_error = error / flux

# Covariance matrix
cov_matrix = gsf.covariance("p", "He", energy)
```

## Solar Modulation

All models support time-dependent solar modulation via the `time_interval` parameter:

```python
# Local interstellar spectrum (no modulation)
flux_lis = gsf.flux(energy, "p", time_interval="LIS")

# Specific time period (YYYYMM format)
flux_2009 = gsf.flux(energy, "p", time_interval=(200901, 201001))  # end EXCLUSIVE: calendar year 2009

# Default: Solar Cycle 24 average (Dec 2008 - Dec 2019)
flux_default = gsf.flux(energy, "p")
```

See the [Solar Modulation tutorial](../gallery/solar_modulation/) for detailed examples.

## Geomagnetic Rigidity Cutoff

Apply a geomagnetic cutoff to suppress low-rigidity cosmic rays:

```python
# Cutoff at 20 GV (smooth sigmoid transition by default, cutoff_width=1 GV;
# construct the model with cutoff_width=0.0 for a sharp Heaviside cutoff)
flux_cut = gsf.flux(energy, "p", rigidity_cutoff=20.0)
```

See the [Rigidity Cutoff tutorial](../gallery/rigidity_cutoff/) for more details.

## Jacobian Access

For sensitivity studies and error propagation:

```python
jacobian = gsf.jacobian(energy, "p")
print(f"Jacobian shape: {jacobian.shape}")
```

## Best Practices

1. **Vectorize calculations**: Pass arrays instead of loops; repeated
   Jacobians use a byte-bounded in-memory cache
2. **Handle uncertainties**: Always consider flux uncertainties in your analysis
3. **Energy ranges**: Stay within the fitted energy range (~1 GeV -- 10^11 GeV
   per nucleus); the flux is zero below the first knot and extrapolated above
   the last
4. **Memory**: Consider chunking for very large arrays (>10^6 points)
5. **Solar modulation**: Use time intervals for time-dependent studies

## Tutorials

See the [tutorial gallery](gallery.md) for detailed, worked examples.
