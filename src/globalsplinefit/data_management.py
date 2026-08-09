import json
import warnings
from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy.interpolate import splev

# Physical/mathematical constants shared with model.py
SPLINE_DEGREE = 3
"""Degree of B-spline interpolation (cubic)."""

# Solar Cycle 24 constants
SOLAR_CYCLE_24_START = (200812, "December 2008")  # (YYYYMM, description)
SOLAR_CYCLE_24_END = (201912, "December 2019")  # (YYYYMM, description)
SOLAR_CYCLE_24_DURATION_YEARS = 11


def _collect_phi_values(
    phi_dict: Mapping[int, np.ndarray], start_yyyymm: int, end_yyyymm: int
) -> np.ndarray:
    """Collect monthly phi values between start and end (YYYYMM format).

    Parameters
    ----------
    phi_dict : dict
        Dictionary mapping year to array of 12 monthly phi values.
    start_yyyymm : int
        Start date in YYYYMM format (inclusive).
    end_yyyymm : int
        End date in YYYYMM format (exclusive).

    Returns
    -------
    np.ndarray
        Concatenated array of monthly phi values.
    """

    def _parse(value: int, name: str) -> tuple[int, int]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer in YYYYMM form")
        year, month = divmod(int(value), 100)
        if year < 1 or not 1 <= month <= 12:
            raise ValueError(f"{name} must be a valid YYYYMM value, got {value!r}")
        return year, month

    start_year, start_month = _parse(start_yyyymm, "start")
    end_year, end_month = _parse(end_yyyymm, "end")
    if start_yyyymm >= end_yyyymm:
        raise ValueError("time interval start must be earlier than end")

    values = []
    year, month = start_year, start_month
    while (year, month) < (end_year, end_month):
        monthly = phi_dict.get(year)
        if monthly is None or len(monthly) != 12:
            raise ValueError(
                f"solar-modulation data are missing for {year:04d}-{month:02d}"
            )
        value = float(monthly[month - 1])
        if not np.isfinite(value):
            raise ValueError(
                f"solar-modulation value is invalid for {year:04d}-{month:02d}"
            )
        values.append(value)
        month += 1
        if month == 13:
            year += 1
            month = 1
    return np.asarray(values, dtype=float)


#: Default model version used by model constructors.
DEFAULT_VERSION = "2026.0"

#: Registry of the distributable model versions.
#:
#: Model names are ``<line>.<revision>[-<physics classifier>]``: ``2026.0``
#: is the first revision of the 2026 line, ``2026.0-USO`` its
#: Usoskin-potential variant, ``2026.0-SIB23e`` its SIBYLL-2.3e-only
#: variant. A data or fit patch is published as a new
#: revision (``2026.1``, ...). Unrevisioned names resolve via
#: :func:`resolve_version`. Historical releases (``2025``, ``2019``,
#: ``2017``) use bare names.
#:
#: ``status`` is one of:
#:   ``"current"``     the default 2026 fit and its variants;
#:   ``"historical"``  a published GSF release, kept for reproducing older
#:                     work.
#:
#: The mixture covering is an equal-weight parameter-level combination of
#: the Auger FD-2026 SIBYLL-2.3e and EPOS-LHC-R interpretations, whose
#: covariance carries a rank-one between-model term, so the band spans both
#: hadronic interpretations.
#:
#: Allow-list: only directories named here are offered as model versions.
MODEL_VERSIONS: dict[str, dict[str, str]] = {
    "2026.0": {
        "status": "current",
        "role": "default",
        "covering": "mixture: equal-weight Auger FD-2026 SIBYLL-2.3e + EPOS-LHC-R",
        "solar_modulation": "GMD (Ghelfi-Maurin-Derome, Ghelfi et al. 2017)",
        "description": (
            "Default 2026 mixture fit with the Ghelfi-Maurin-Derome potential, "
            "which the "
            "data mildly prefer."
        ),
    },
    "2026.0-USO": {
        "status": "current",
        "role": "alternative",
        "covering": "mixture: equal-weight Auger FD-2026 SIBYLL-2.3e + EPOS-LHC-R",
        "solar_modulation": "USO (Usoskin et al. 2017)",
        "description": (
            "The same mixture fit with the Usoskin 2017 potential, "
            "which runs about 65 MV lower "
            "and yields a 10-14% lower interstellar spectrum below 2 GV. Use it "
            "to gauge the solar-modulation systematic."
        ),
    },
    "2026.0-SIB23e": {
        "status": "current",
        "role": "single-interpretation variant",
        "covering": "Auger FD-2026 SIBYLL-2.3e (single interpretation)",
        "solar_modulation": "GMD (Ghelfi-Maurin-Derome, Ghelfi et al. 2017)",
        "description": (
            "The 2026 fit under the SIBYLL-2.3e interpretation of the Auger "
            "FD-2026 composition alone, for applications that need a single "
            "consistent hadronic-interaction model."
        ),
    },
    "2026.0-EPOSLHCR": {
        "status": "current",
        "role": "single-interpretation variant",
        "covering": "Auger FD-2026 EPOS-LHC-R (single interpretation)",
        "solar_modulation": "GMD (Ghelfi-Maurin-Derome, Ghelfi et al. 2017)",
        "description": (
            "The 2026 fit under the EPOS-LHC-R interpretation of the Auger "
            "FD-2026 composition alone -- the other half of the mixture, for "
            "applications that need a single consistent hadronic-interaction "
            "model."
        ),
    },
    "2025": {
        "status": "historical",
        "role": "conference update",
        "covering": "see the GSF 2025 proceedings",
        "solar_modulation": (
            "data demodulated with a single effective potential; "
            "forward-modulated with the shared Usoskin table"
        ),
        "description": (
            "GSF 2025, presented at UHECR 2024 and ICRC 2025 "
            "(Fujisue:2025wnp, Dembinski:2025nmp)."
        ),
    },
    "2019": {
        "status": "historical",
        "role": "conference update",
        "covering": "see the GSF 2019 proceedings",
        "solar_modulation": (
            "data demodulated with a single effective potential; "
            "forward-modulated with the shared Usoskin table"
        ),
        "description": "GSF 2019 conference update.",
    },
    "2017": {
        "status": "historical",
        "role": "original release",
        "covering": "see Dembinski et al. (2017)",
        "solar_modulation": (
            "treatment not recorded here -- see the ICRC 2017 proceedings; "
            "forward-modulated with the shared Usoskin table"
        ),
        "description": "Original GSF release (Dembinski:2017zsh).",
    },
}


def resolve_version(version: str | None = None) -> str:
    """Resolve a version request to a registered ``<line>.<revision>`` name.

    An exact registered name (``"2026.0"``, ``"2025"``) is returned as is.
    An unrevisioned name -- ``"2026"``, ``"2026-USO"`` -- resolves to the
    NEWEST registered revision of that line and variant, so callers that do
    not pin a revision follow data patches automatically. ``None`` resolves
    :data:`DEFAULT_VERSION`.
    """
    name = DEFAULT_VERSION if version is None else str(version).strip()
    if name in MODEL_VERSIONS:
        return name
    line, _, variant = name.partition("-")
    revisions = []
    for registered in MODEL_VERSIONS:
        rline, _, rvariant = registered.partition("-")
        rbase, dot, rev = rline.partition(".")
        if dot and rbase == line and rvariant == variant and rev.isdigit():
            revisions.append((int(rev), registered))
    if revisions:
        return max(revisions)[1]
    raise ValueError(
        f"Version {name!r} not found. Available versions: {get_available_versions()}"
    )


_REQUIRED_FILES = ("knots.dat", "nuclei.dat", "parameters.dat", "covariance.dat")


def get_available_versions(include_historical: bool = True) -> list[str]:
    """Get the list of available GSF data versions.

    Only registered versions whose required data files are present are returned.

    Parameters
    ----------
    include_historical
        When True (default), include historical releases. Pass False for
        the current 2026 family.

    Returns
    -------
    list[str]
        Available version names, usable as the ``version=`` argument.
    """
    data_dir = Path(__file__).parent / "data"
    if not data_dir.exists():
        return []

    present = {
        p.name
        for p in data_dir.iterdir()
        if p.is_dir()
        and not p.name.startswith((".", "__"))
        and all((p / f).exists() for f in _REQUIRED_FILES)
    }

    unregistered = sorted(present - MODEL_VERSIONS.keys())
    if unregistered:
        warnings.warn(
            "ignoring unregistered model data director"
            + ("ies" if len(unregistered) > 1 else "y")
            + f" under {data_dir}: {', '.join(unregistered)}. Only versions in "
            "globalsplinefit.data_management.MODEL_VERSIONS are distributable; "
            "add an entry there if one of these is meant to be a release.",
            UserWarning,
            stacklevel=2,
        )

    return sorted(
        name
        for name in MODEL_VERSIONS
        if name in present
        and (include_historical or MODEL_VERSIONS[name]["status"] == "current")
    )


def version_info(version: str | None = None) -> dict[str, str]:
    """Describe a model version: its status, covering and modulation potential.

    Parameters
    ----------
    version
        Version name. Defaults to :data:`DEFAULT_VERSION`.

    Returns
    -------
    dict[str, str]
        The :data:`MODEL_VERSIONS` entry for that version.
    """
    name = resolve_version(version)
    return {"name": name, **MODEL_VERSIONS[name]}


class Parameters:
    """Immutable bundle of loaded GSF model data.

    Holds knots, spline parameters, covariance, nuclei and the
    solar-modulation table.

    Parameters
    ----------
    data_path : str or Path, optional
        Path to directory containing GSF data files. If None, uses
        :data:`DEFAULT_VERSION`.
    version : str, optional
        Registered model version. Defaults to :data:`DEFAULT_VERSION`.
        See :data:`MODEL_VERSIONS` for provenance and available variants.
    """

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("Parameters instances are immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        data_path: str | Path | None = None,
        version: str | None = None,
    ):
        self.data_path = self._setup_data_path(data_path, version)
        self._load_all_data()
        self._provenance = self._read_provenance()
        self._validate_loaded_data()
        self._freeze()

    @classmethod
    def for_model(
        cls,
        data_path: str | Path | None = None,
        version: str | None = None,
    ) -> "Parameters":
        """Return a shared immutable parameter bundle for model instances."""
        if data_path is not None:
            return cls(data_path=data_path, version=version)
        return cls._cached(resolve_version(version))

    @staticmethod
    @lru_cache(maxsize=32)
    def _cached(version: str) -> "Parameters":
        return Parameters(version=version)

    def _setup_data_path(
        self, data_path: str | Path | None, version: str | None
    ) -> Path:
        """Set up the data path, recording which version was resolved."""
        #: Resolved version name; None when loading an arbitrary data_path.
        self.version = None
        if version is not None and data_path is not None:
            raise ValueError("pass either version or data_path, not both")
        if version is not None or data_path is None:
            version_str = resolve_version(version)
            data_dir = Path(__file__).parent / "data" / version_str
            if not data_dir.exists():
                raise ValueError(
                    f"Version '{version_str}' not found. Available versions: {get_available_versions()}"
                )
            self.version = version_str
            return data_dir
        path = Path(data_path)
        if not path.exists():
            raise OSError(f"Data path does not exist: {path}")
        # An explicit path may still point at a packaged version directory.
        if (
            path.resolve().parent == (Path(__file__).parent / "data").resolve()
            and path.name in MODEL_VERSIONS
        ):
            self.version = path.name
        return path

    @property
    def provenance(self) -> dict:
        """Return a copy of the fit provenance metadata."""
        return deepcopy(self._provenance)

    def _read_provenance(self) -> dict:
        """Read and validate ``fit_result.json`` when present."""
        info: dict = {}
        meta_file = self.data_path / "fit_result.json"
        if meta_file.exists():
            try:
                loaded = json.loads(meta_file.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError(f"invalid provenance file: {meta_file}") from exc
            if not isinstance(loaded, dict):
                raise ValueError(f"invalid provenance file: {meta_file}")
            meta = loaded.get("metadata", loaded)
            if not isinstance(meta, dict):
                raise ValueError(f"invalid provenance metadata: {meta_file}")
            for key in (
                "covering",
                "solar_modulation_source",
                "mixture",
                "mixture_components",
                "mixture_weights",
                "mixture_chi2_note",
                "slope_freeze",
                "norm_penalty",
                "written",
            ):
                if key in meta:
                    info[key] = meta[key]
        if self.version is not None:
            info["registry"] = dict(MODEL_VERSIONS[self.version])
            info["version"] = self.version
        return info

    def _load_all_data(self):
        """Load all required GSF data files."""
        self._load_nuclei_data()  # first: defines the species (Z, A) + groups
        self._load_knots()
        self._load_parameters()
        self._load_covariance()
        self._load_solar_modulation()
        self._load_subleading()
        self._load_reduced_pivots()
        self._calculate_flux_ratios()

    def _load_reduced_pivots(self):
        """Load the optional per-bundle ``reduced_pivots.dat`` grid.

        An absent file leaves ``reduced_pivots = None``.
        """
        self.reduced_pivots = None
        pivots_file = self.data_path / "reduced_pivots.dat"
        if not pivots_file.exists():
            return
        pivots = np.atleast_1d(np.loadtxt(pivots_file, dtype=float))
        if (
            pivots.ndim != 1
            or len(pivots) < 2
            or not np.all(np.isfinite(pivots))
            or np.any(pivots <= 0)
            or np.any(np.diff(pivots) <= 0)
        ):
            raise ValueError(
                f"invalid pivot table {pivots_file}: need at least two positive, "
                "finite, strictly increasing energies"
            )
        pivots.setflags(write=False)
        self.reduced_pivots = pivots

    def _validate_loaded_data(self) -> None:
        """Validate cross-file invariants before a model can use the bundle."""
        species = set(self.species)
        if not species:
            raise ValueError(f"no species found in {self.data_path / 'nuclei.dat'}")
        knot_species = {key for key in self.kx if isinstance(key, tuple)}
        parameter_species = {key for key in self.pars if isinstance(key, tuple)}
        if knot_species != species or parameter_species != species:
            missing_knots = sorted(species - knot_species)
            missing_parameters = sorted(species - parameter_species)
            raise ValueError(
                "incomplete model bundle: "
                f"missing knots for {missing_knots}, parameters for {missing_parameters}"
            )
        for sid in species:
            if sid[0] <= 0 or not np.isfinite(sid[1]) or sid[1] <= 0:
                raise ValueError(f"invalid species identifier {sid!r}")
            if self.z_ungroup[sid[0]] not in self.leader_sid:
                raise ValueError(f"species {sid!r} refers to an undefined group leader")
            knots = self.kx[sid][3:-3]
            if not np.all(np.isfinite(knots)) or np.any(np.diff(knots) <= 0):
                raise ValueError(f"knots for species {sid!r} are not finite/increasing")
            if len(self.pars[sid]) != self.npar[sid] + 4:
                raise ValueError(f"wrong parameter count for species {sid!r}")
            if not np.all(np.isfinite(self.pars[sid])):
                raise ValueError(f"non-finite parameters for species {sid!r}")
        for (sid1, sid2), block in self.cov.items():
            if not isinstance(sid1, tuple) or not isinstance(sid2, tuple):
                continue
            expected = (self.npar[sid1], self.npar[sid2])
            if block.shape != expected or not np.all(np.isfinite(block)):
                raise ValueError(
                    f"invalid covariance block {(sid1, sid2)!r}: "
                    f"expected {expected}, got {block.shape}"
                )
            reverse = self.cov.get((sid2, sid1))
            if reverse is None or not np.allclose(
                block, reverse.T, rtol=1e-12, atol=1e-12
            ):
                raise ValueError(f"asymmetric covariance blocks for {sid1!r}, {sid2!r}")

    def _freeze(self) -> None:
        """Make loaded numerical state safe to share between model instances."""
        array_maps = (self.kx, self.pars, self.cov, self.phi)
        for mapping in array_maps:
            for value in mapping.values():
                value.setflags(write=False)
        self.species = tuple(self.species)
        self._leaders = frozenset(self._leaders)
        self.z_to_sids = MappingProxyType(
            {charge: tuple(sids) for charge, sids in self.z_to_sids.items()}
        )
        self.z_group = MappingProxyType(
            {leader: tuple(charges) for leader, charges in self.z_group.items()}
        )
        for name in (
            "kx",
            "npar",
            "pars",
            "offset",
            "z_to_a",
            "mass_number",
            "z_ungroup",
            "leader_sid",
            "cov",
            "phi",
            "flux_ratio",
            "flux_slope",
        ):
            setattr(self, name, MappingProxyType(dict(getattr(self, name))))
        object.__setattr__(self, "_frozen", True)

    @staticmethod
    def _ncols(path) -> int:
        """Count whitespace-delimited fields in the first data line.

        Used to auto-detect the v1 vs v2 .dat column layout.
        """
        for line in Path(path).read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                return len(line.replace(":", " ").split())
        return 0

    def _sid_for_z(self, z: int):
        """Return the unique species ID for a charge.

        Version 1 files contain one species per charge and use charge-only keys.
        """
        return self.z_to_sids[z][0]

    def _load_nuclei_data(self):
        """Load nuclear species and group membership.

        Species use ``(Z, A)`` IDs, allowing isotopes at the same charge. The
        lightest isotope at a leader charge is the group leader.
        """
        nuclei_file = self.data_path / "nuclei.dat"
        if not nuclei_file.exists():
            raise FileNotFoundError(f"Nuclei file not found: {nuclei_file}")

        data_array = np.atleast_1d(
            np.loadtxt(nuclei_file, dtype=[("z", int), ("a", float), ("l", int)])
        )
        rows = [(int(r["z"]), float(r["a"]), int(r["l"])) for r in data_array]
        species_rows = [(z, a) for z, a, _leader in rows]
        if len(species_rows) != len(set(species_rows)):
            raise ValueError(f"duplicate species in {nuclei_file}")
        if any(z <= 0 or not np.isfinite(a) or round(a) < z for z, a, _ in rows):
            raise ValueError(f"invalid charge or mass in {nuclei_file}")

        self.species = sorted({(z, a) for z, a, _ in rows})  # all species ids
        self.z_to_a = {(z, a): a for z, a, _ in rows}  # sid -> A (charge
        self.mass_number = {(z, a): int(round(a)) for z, a, _ in rows}
        #   aliases added by _add_charge_aliases; iterate self.species, not this)
        self.z_to_sids = {}  # charge -> [sids]
        for z, a, _ in rows:
            self.z_to_sids.setdefault(z, []).append((z, a))
        # Leader sid per group charge: lightest-A species at that charge.
        self.leader_sid = {}  # charge -> leader sid
        for z, a, leader in rows:
            if z == leader and (
                leader not in self.leader_sid or a < self.leader_sid[leader][1]
            ):
                self.leader_sid[leader] = (z, a)
        self._leaders = set(self.leader_sid.values())  # leader sids
        # Public group structure stays CHARGE-based (one entry per element charge)
        # so charge-indexed callers/tests are unchanged; isotopes are expanded
        # from a charge to its species ids (z_to_sids) inside the flux loops.
        self.z_group = {}  # int leader charge -> [member element charges]
        self.z_ungroup = {}  # element charge -> leader charge
        for z, _a, leader in rows:
            if z not in self.z_group.get(leader, []):
                self.z_group.setdefault(leader, []).append(z)
            self.z_ungroup[z] = leader

    def _load_knots(self):
        """Load spline knots.

        V2 lines are ``Z A: …``; v1 lines ``Z: …`` (keyed by
        the unique sid at that charge). Keys are species ids ``(Z, A)``.
        """
        knots_file = self.data_path / "knots.dat"
        if not knots_file.exists():
            raise FileNotFoundError(f"Knots file not found: {knots_file}")

        self.kx = {}
        self.npar = {}

        with open(knots_file) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                head, k = line.split(":")
                toks = head.split()
                z = int(toks[0])
                sid = (z, float(toks[1])) if len(toks) >= 2 else self._sid_for_z(z)
                x = np.asarray([float(v) * np.log(10.0) for v in k.split()])
                self.npar[sid] = len(x) + 2
                # splev requires extended knot vector
                x = np.append((x[0], x[0], x[0]), x)
                x = np.append(x, (x[-1], x[-1], x[-1]))
                self.kx[sid] = x

    def _load_parameters(self):
        """Load spline parameters. v2: ``i Z A val``; v1: ``i Z val``."""
        params_file = self.data_path / "parameters.dat"
        if not params_file.exists():
            raise FileNotFoundError(f"Parameters file not found: {params_file}")

        self.pars = {}
        self.offset = {}
        v2 = self._ncols(params_file) >= 4
        dt = (
            [("i", int), ("z", int), ("a", float), ("val", float)]
            if v2
            else [("i", int), ("z", int), ("val", float)]
        )
        data_array = np.atleast_1d(np.loadtxt(params_file, dtype=dt))
        seen: dict[tuple[int, float], set[int]] = {}

        for r in data_array:
            i, z, val = int(r["i"]), int(r["z"]), float(r["val"])
            sid = (z, float(r["a"])) if v2 else self._sid_for_z(z)
            if sid not in self.pars:
                # 4 extra zeros at the end are needed by splev
                self.pars[sid] = np.zeros(self.npar[sid] + 4)
                self.offset[sid] = i
                seen[sid] = set()
            local_index = i - self.offset[sid]
            if local_index in seen[sid] or not 0 <= local_index < self.npar[sid]:
                raise ValueError(
                    f"invalid or duplicate parameter index {i} for {sid!r}"
                )
            seen[sid].add(local_index)
            self.pars[sid][local_index] = val

        for sid in self.species:
            if seen.get(sid) != set(range(self.npar[sid])):
                raise ValueError(f"incomplete parameter vector for species {sid!r}")

    def _load_covariance(self):
        """Load the parameter covariance.

        V2 rows contain ``i j Z1 A1 Z2 A2 value``; v1 rows contain
        ``i j Z1 Z2 value``.

        Keyed by species-id pairs ``((Z1,A1), (Z2,A2))``.
        """
        cov_file = self.data_path / "covariance.dat"
        if not cov_file.exists():
            raise FileNotFoundError(f"Covariance file not found: {cov_file}")

        self.cov = {}
        v2 = self._ncols(cov_file) >= 7
        dt = (
            [
                ("i", int),
                ("j", int),
                ("z1", int),
                ("a1", float),
                ("z2", int),
                ("a2", float),
                ("val", float),
            ]
            if v2
            else [("i", int), ("j", int), ("z1", int), ("z2", int), ("val", float)]
        )
        data_array = np.atleast_1d(np.loadtxt(cov_file, dtype=dt))

        for r in data_array:
            i, j, val = int(r["i"]), int(r["j"]), float(r["val"])
            if v2:
                s1 = (int(r["z1"]), float(r["a1"]))
                s2 = (int(r["z2"]), float(r["a2"]))
            else:
                s1, s2 = self._sid_for_z(int(r["z1"])), self._sid_for_z(int(r["z2"]))
            if s1 not in self.npar or s2 not in self.npar:
                raise ValueError(
                    f"covariance references unknown species {s1!r}, {s2!r}"
                )
            local_i = i - self.offset[s1]
            local_j = j - self.offset[s2]
            if not 0 <= local_i < self.npar[s1] or not 0 <= local_j < self.npar[s2]:
                raise ValueError(
                    f"covariance index {(i, j)!r} is outside the parameter blocks "
                    f"for {s1!r}, {s2!r}"
                )
            if (s1, s2) not in self.cov:
                self.cov[(s1, s2)] = np.zeros((self.npar[s1], self.npar[s2]))
            if (s2, s1) not in self.cov:
                self.cov[(s2, s1)] = np.zeros((self.npar[s2], self.npar[s1]))
            self.cov[(s1, s2)][local_i, local_j] = val
            self.cov[(s2, s1)][local_j, local_i] = val

    def _load_solar_modulation(self):
        """Load the monthly solar-modulation potential table (phi, MV).

        Precedence: a VERSION-LOCAL ``<data_path>/solar_modulation.dat``,
        else the shared Usoskin table at the package data root. The LIS is
        demodulated with a specific phi(t), so the fitted LIS must be
        re-modulated by the SAME potential to recover a flux at Earth: the
        GMD sets (2026 mixture and single-interpretation variants) store
        their Ghelfi-Maurin-Derome table alongside the parameters, while
        2017/2019/2025 and 2026-USO use the shared Usoskin table.
        """
        local = self.data_path / "solar_modulation.dat"
        shared = Path(__file__).parent / "data" / "solar_modulation.dat"
        solar_file = local if local.exists() else shared
        if not solar_file.exists():
            raise FileNotFoundError(f"Solar modulation file not found: {solar_file}")

        # Detect encoding from the byte-order mark (shared file is UTF-16-LE
        # with BOM; version-local files are plain UTF-8).
        with open(solar_file, "rb") as fh:
            encoding = "utf-16" if fh.read(2) == b"\xff\xfe" else "utf-8"

        # '#'-commented header of arbitrary length; data rows are numeric.
        data_array = np.loadtxt(solar_file, comments="#", encoding=encoding)

        self.phi = {}
        for row in data_array:
            months = row[1:13]  # Year, Jan..Dec[, Annual] -> keep the 12 months
            if np.isnan(months).any():  # drop incomplete years (e.g. Jan 1951)
                continue
            self.phi[int(row[0])] = months * 1e-3  # MV -> GV

    def _load_subleading(self):
        """Load optional subleading-species extrapolation parameters."""
        self._stored_sub = {}
        sub_file = self.data_path / "subleading.dat"
        if not sub_file.exists():
            return
        dt = [("z", int), ("a", float), ("norm", float), ("slope", float)]
        for r in np.atleast_1d(np.loadtxt(sub_file, dtype=dt)):
            sid = (int(r["z"]), float(r["a"]))
            values = (float(r["norm"]), float(r["slope"]))
            if sid not in self.species or sid in self._stored_sub:
                raise ValueError(f"invalid or duplicate species {sid!r} in {sub_file}")
            if not np.all(np.isfinite(values)) or values[0] < 0:
                raise ValueError(f"invalid extrapolation values for {sid!r}")
            self._stored_sub[sid] = values

    def _calculate_flux_ratios(self):
        """Calculate subleading-species flux ratios and slopes.

        Stored values take precedence. Otherwise the ratio is evaluated at the
        species' top knot and the slope is zero.
        """
        self.flux_ratio = {}
        self.flux_slope = {}

        for sid in self.species:
            leader_sid = self.leader_sid[self.z_ungroup[sid[0]]]
            xmax = self.kx[sid][-1]
            if sid != leader_sid:
                if sid in self._stored_sub:
                    ratio, slope = self._stored_sub[sid]
                else:
                    # Subleading species - ratio to its group leader
                    ratio = splev(
                        xmax, (self.kx[sid], self.pars[sid], SPLINE_DEGREE)
                    ) / splev(
                        xmax,
                        (self.kx[leader_sid], self.pars[leader_sid], SPLINE_DEGREE),
                    )
                    slope = 0.0
            else:
                ratio, slope = 1.0, 0.0
            self.flux_ratio[sid] = (leader_sid, ratio)
            self.flux_slope[sid] = slope
        self._add_charge_aliases()

    def _add_charge_aliases(self):
        """Add integer aliases for charges with exactly one species.

        Multi-isotope charges require explicit ``(Z, A)`` keys.
        """
        for z, sids in self.z_to_sids.items():
            if len(sids) != 1:
                continue
            s = sids[0]
            self.kx[z] = self.kx[s]
            self.npar[z] = self.npar[s]
            self.pars[z] = self.pars[s]
            self.offset[z] = self.offset[s]
            self.z_to_a[z] = self.z_to_a[s]
            self.mass_number[z] = self.mass_number[s]
            self.flux_ratio[z] = self.flux_ratio[s]
            self.flux_slope[z] = self.flux_slope[s]
        for s1, s2 in list(self.cov):
            z1, z2 = s1[0], s2[0]
            if len(self.z_to_sids[z1]) == 1 and len(self.z_to_sids[z2]) == 1:
                self.cov[(z1, z2)] = self.cov[(s1, s2)]

    def get_solar_cycle_24_interval(self) -> tuple[int, int]:
        """Get the time interval for Solar Cycle 24.

        Returns
        -------
        tuple[int, int]
            Time interval (start, end) in YYYYMM format.
        """
        return SOLAR_CYCLE_24_START[0], SOLAR_CYCLE_24_END[0]

    def get_solar_cycle_24_phi_average(self) -> float:
        """Mean of the monthly modulation potentials over Solar Cycle 24, in GV.

        Returns
        -------
        float
            Average solar modulation potential in GV.
        """
        start, end = self.get_solar_cycle_24_interval()
        phis = _collect_phi_values(self.phi, start, end)
        return float(np.mean(phis))


def list_versions(return_paths: bool = False) -> None | list:
    """List available GSF data versions and their validity.

    Returns
    -------
    list or None
        If ``return_paths`` is True, the list of valid data paths, usable
        to initialize :class:`Parameters`. Otherwise prints a table and
        returns None.
    """
    current_dir = Path(__file__).parent
    data_dir = current_dir / "data"

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return [] if return_paths else None

    results = []
    valid_paths = []

    available_versions = get_available_versions()

    if not available_versions:
        print("No valid GSF data versions found in the data directory.")
        return [] if return_paths else None

    for version_name in available_versions:
        version_path = data_dir / version_name
        try:
            _ = Parameters(data_path=version_path)
            results.append([version_name, "Valid", ""])
            valid_paths.append(version_path)

        except Exception as e:
            error_reason = str(e)
            if len(error_reason) > 60:
                error_reason = error_reason[:57] + "..."
            results.append([version_name, "Invalid", error_reason])

    if return_paths:
        return valid_paths

    if results:
        widths = [
            max(len(header), *(len(str(row[index])) for row in results))
            for index, header in enumerate(("Version", "Status", "Error"))
        ]
        print(f"{'Version':<{widths[0]}}  {'Status':<{widths[1]}}  Error")
        for version_name, status, reason in results:
            print(f"{version_name:<{widths[0]}}  {status:<{widths[1]}}  {reason}")
    else:
        print("No GSF data versions found in the data directory.")
