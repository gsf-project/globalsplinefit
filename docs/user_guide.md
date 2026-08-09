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
gsf_2025 = GSFEnergy(version="2025")        # earlier conference update
```

Physical model versions are named `<line>.<revision>[-<physics classifier>]`:
`2026.0` is the first revision of the 2026 line, `2026.0-USO` its
Usoskin-potential variant, `2026.0-SIB23e` its SIBYLL-2.3e-only variant. A data or fit patch is published as a new revision (`2026.1`, ...).
An unrevisioned name (`"2026"`, `"2026-USO"`) resolves to the newest
registered revision of that line and variant — use it to follow patches
automatically, or pass the revisioned name to pin one
(`resolve_version("2026")` shows the resolution). Bare names `2025`, `2019`,
`2017` select the earlier sets.

| Version | Status | Covering | Solar modulation | Description |
|---------|--------|----------|------------------|-------------|
| `"2026.0"` | **current -- default** | SIBYLL-2.3e / EPOS-LHC-R mixture | self-consistent, Ghelfi--Maurin--Derome (**baseline**) | Default 2026 fit |
| `"2026.0-USO"` | current -- alternative | SIBYLL-2.3e / EPOS-LHC-R mixture | self-consistent, Usoskin 2017 | The same fit with the other potential; use it to gauge the solar-modulation systematic |
| `"2026.0-SIB23e"` | current -- single interpretation | Auger FD-2026 SIBYLL-2.3e only | self-consistent, Ghelfi--Maurin--Derome | For applications needing a definite hadronic model |
| `"2026.0-EPOSLHCR"` | current -- variant | Auger FD-2026 EPOS-LHC-R only | self-consistent, Ghelfi--Maurin--Derome | The other single-interpretation half of the mixture |
| `"2025"` | conference update (ICRC 2025 / UHECR 2024) | PDG scale factors | data demodulated with a single effective potential; forward-modulated here with the bundled Usoskin table | GSF 2025 |
| `"2019"` | conference update | PDG scale factors | as 2025 | GSF 2019 |
| `"2017"` | original release | PDG scale factors | see the ICRC 2017 proceedings — the treatment is not recorded in this package; forward-modulated here with the bundled Usoskin table | Original GSF (Dembinski et al. 2017) |

`get_available_versions(include_historical=False)` returns just the current
sets, and `version_info(version)` reports any version's status, covering and
modulation potential. A loaded model also carries its own provenance:

```python
gsf.version                      # -> "2026.0", the revision actually loaded
gsf.params.provenance["covering"]                  # the air-shower interpretation
gsf.params.provenance["solar_modulation_source"]   # GMD or USO
```

### How to cite each version

<!-- citations:versions -->

Full records, with BibTeX to copy:

<!-- citations:all -->

### What "covering" means, and why both current sets are a mixture

Above ~10^8 GeV the mass composition inferred from air-shower data depends on the
hadronic interaction model used to interpret it. Both current sets use an
equal-weight **parameter-level mixture** of the Auger FD-2026 SIBYLL-2.3e and
EPOS-LHC-R interpretations: parameters are the mean of the two fits, and the
covariance carries an additional rank-one between-model term. The published band
therefore spans both interpretations where they diverge and collapses to the
ordinary fit covariance where they agree (below ~2x10^8 GeV, where the two
coincide). `2026.0-SIB23e` and `2026.0-EPOSLHCR` are the two halves of the mixture.

The 2026 sets are fitted in isotope format, with deuterium carried explicitly
as its own spline under the H\* group.

## Particle Groups

All models support these cosmic ray groups:

| Group | Description | Atomic Numbers |
|-------|-------------|----------------|
| `"H*"` | Hydrogen group (p and D when available) | Z = 1 |
| `"He"` | Helium group (written He\* in the papers) | Z = 2 |
| `"O*"` or `"CNO"` | Light/intermediate group | Z = 3--9 |
| `"Fe*"` or `"heavy"` | Heavy group | Z = 10--28 |

The four groups are written **H\*, He\*, O\*, Fe\*** in the papers — the star
marks a group that carries neighbouring elements scaled from its leader. As
target names the model still takes `"H*"` and `"He"` (`"He*"` is not an
accepted key).

A bare element symbol selects that **single element** instead of its group:

```python
gsf.flux(energy, "O*")   # the whole oxygen group, Z = 3-9
gsf.flux(energy, "O")    # oxygen alone (Z = 8), the group leader
gsf.flux(energy, 8)      # the same, addressed by charge number
```

`"p"`, `"O"` and `"Fe"` are the individual proton, oxygen and iron species;
any element can be addressed by its charge number `Z`.

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

### Working with sub-leading elements

```python
gsf.z_group                  # {leader Z: (member Z, ...)} for all four groups
gsf.z_group[8]               # -> (3, 4, 5, 6, 7, 8, 9): the O* group members

# flux of one sub-leading species (carbon, Z = 6)
carbon = gsf.flux(energy, 6)

# its uncertainty: error() propagates the LEADER's spline covariance through
# the species' own kinematics, so the relative error differs from the
# leader's at the same energy per nucleus
carbon_err = gsf.error(energy, 6)

# the fitted ratio to the group leader, per species
leader, ratio = gsf.flux_ratio[(6, 12.011)]   # -> ((8, 15.999), 1.1487)
```

For the current sets these per-element parameters are in
`data/<version>/subleading.dat` (columns `Z A norm slope`). Without the file
(`2017`, `2019`, `2025`), or with a zero slope, the ratio is constant above
$R_{\text{max}}$.

!!! note
    The norm and slope are best-fit point estimates, and uncertainty
    propagation uses the four group-leader blocks alone, so a sub-leading
    flux carries the leader's uncertainty and nothing extra. The 2026 sets
    do ship covariance blocks for all 28 charges — each species' own short
    spline over its direct-data range, mostly pinned — which the model loads
    but never propagates. The saturation constant is exposed as
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

### Jacobian Access

The flux is **linear in the fitted spline amplitudes**, so a single Jacobian
carries the whole error propagation:
$J_{ij} = \partial\,\text{flux}(E_i)\,/\,\partial\,a_j$ for the amplitudes
$a_j$ of the group that target belongs to. `error()` is exactly
$\sqrt{\mathrm{diag}(J\,C\,J^\mathsf{T})}$ with $C$ the fitted amplitude
covariance — the Jacobian is what you need when you want something else:
a correlated band across energies, a derived quantity, or a covariance
between two groups.

```python
jac = gsf.jacobian(energy, "p")       # shape (len(energy), n_amplitudes)
cov = gsf.covariance("p", "p", energy)  # flux covariance across energies

# uncertainty of a p + He sum, correlations included
import numpy as np

f_sum = gsf.flux(energy, "p") + gsf.flux(energy, "He")
var = (
    np.diag(gsf.covariance("p", "p", energy))
    + np.diag(gsf.covariance("He", "He", energy))
    + 2 * np.diag(gsf.covariance("p", "He", energy))
)
err_sum = np.sqrt(var)
```

Adding the two errors in quadrature instead would ignore the p--He
correlation, which the fit constrains.

## Solar Modulation

The model is fitted as a **local interstellar spectrum** (LIS) and modulated
to the top of the atmosphere with a force-field potential $\phi(t)$.

```python
# Local interstellar spectrum (no modulation)
flux_lis = gsf.flux(energy, "p", time_interval="LIS")

# Specific time period (YYYYMM format)
flux_2009 = gsf.flux(energy, "p", time_interval=(200901, 201001))  # end EXCLUSIVE: calendar year 2009

# Default: Solar Cycle 24 average (Dec 2008 - Dec 2019), NOT the LIS
flux_default = gsf.flux(energy, "p")
```

### The potentials, and what "self-consistent" means

<!-- citations:potentials -->

Every current set is **self-consistent**: the model is modulated with exactly
the same potential model that was used to demodulate the data during the fit.
Each current set therefore ships its own monthly $\phi(t)$ table
(`data/<version>/solar_modulation.dat`), while the historical sets use the
bundled Usoskin table at the package data root.

This is why the choice of potential matters less than it appears. Relative to
the default, the Usoskin potential yields a 10--14% lower interstellar
spectrum below 2 GV, with the two converging above ~10 GV — but because each
LIS is paired with the potential it was derived with, the **fluxes at Earth
are almost identical** even where the interstellar spectra differ. All
higher-energy results are identical.

See the [Solar Modulation tutorial](../gallery/solar_modulation/) for worked
examples.

## Geomagnetic Rigidity Cutoff

**No cutoff is applied by default** (`rigidity_cutoff=None`, i.e. 0 GV).
Pass a cutoff to suppress low-rigidity cosmic rays:

```python
# Cutoff at 20 GV (smooth sigmoid transition by default, cutoff_width=1 GV;
# construct the model with cutoff_width=0.0 for a sharp Heaviside cutoff)
flux_cut = gsf.flux(energy, "p", rigidity_cutoff=20.0)
```

See the [Rigidity Cutoff tutorial](../gallery/rigidity_cutoff/) for more details.

## Practical Notes

- **Pass arrays, not loops.** Every method is vectorized over the energy
  argument; calling it once with an array of 1000 energies is far cheaper than
  1000 scalar calls. That is the only vectorization gain — there is nothing to
  batch over targets or versions.
- **Memory: the Jacobian cache.** Jacobians are cached per model instance and
  per energy grid. One model with a dense grid is fine, but many models
  (~10 or more) each holding a dense grid can add up to a substantial
  footprint. If that bites, change the access pattern rather than the grid:
  reuse one model instance instead of constructing many, and keep the set of
  distinct energy grids small.
- **Energy range.** Stay within the fitted range (~1 GeV -- 10^11 GeV per
  nucleus); the flux is zero below the first knot and extrapolated above the
  last.
- **The default is modulated.** `flux(energy, target)` returns the
  Solar-Cycle-24 average at Earth, not the interstellar spectrum — pass
  `time_interval="LIS"` when you want the LIS.

## Tutorials

See the [tutorial gallery](gallery.md) for detailed, worked examples.
