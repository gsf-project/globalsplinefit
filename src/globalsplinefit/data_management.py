import json
import warnings
from pathlib import Path

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
    phi_dict: dict[int, np.ndarray], start_yyyymm: int, end_yyyymm: int
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
    yr_a, im_a = divmod(start_yyyymm, 100)
    yr_b, im_b = divmod(end_yyyymm, 100)
    im_a -= 1  # Convert to 0-indexed months
    im_b -= 1

    chunks = []
    if yr_a == yr_b:
        chunks.append(phi_dict[yr_a][im_a:im_b])
    else:
        chunks.append(phi_dict[yr_a][im_a:])
        for yr in range(yr_a + 1, yr_b):
            if yr in phi_dict:
                chunks.append(phi_dict[yr])
        if yr_b in phi_dict:
            chunks.append(phi_dict[yr_b][:im_b])
    return np.concatenate(chunks)


#: The promoted default model version: a bare ``GSFEnergy()`` resolves to this.
DEFAULT_VERSION = "GSF2026"

#: Registry of the distributable model versions.
#:
#: ``status`` is one of:
#:   ``"current"``     the promoted GSF2026 fit -- the default and its one
#:                     sanctioned alternative, differing only in the solar
#:                     modulation potential;
#:   ``"historical"``  a previously published GSF release, kept so older work can
#:                     be reproduced. NOT an alternative to the current fit.
#:
#: Both current sets use the same **mixture** covering: an equal-weight
#: parameter-level combination of the Auger FD-2026 SIBYLL-2.3e and EPOS-LHC-R
#: interpretations, whose covariance carries a rank-one between-model term so the
#: band spans both hadronic interpretations instead of committing to one.
#:
#: This registry is an allow-list. Only directories named here are offered as
#: model versions, so an intermediate or transient fit exported into ``data/``
#: cannot become distributable by accident.
MODEL_VERSIONS: dict[str, dict[str, str]] = {
    "GSF2026": {
        "status": "current",
        "role": "default",
        "covering": "mixture: equal-weight Auger FD-2026 SIBYLL-2.3e + EPOS-LHC-R",
        "solar_modulation": "GMD (Ghelfi-Maurin-Derome, Ghelfi et al. 2017)",
        "description": (
            "Promoted default. Mixture covering with the Ghelfi-Maurin-Derome "
            "modulation potential, which the data mildly prefer."
        ),
    },
    "GSF2026-USO": {
        "status": "current",
        "role": "alternative",
        "covering": "mixture: equal-weight Auger FD-2026 SIBYLL-2.3e + EPOS-LHC-R",
        "solar_modulation": "USO (Usoskin et al. 2017)",
        "description": (
            "The one sanctioned alternative: the same mixture fit with the "
            "Usoskin 2017 potential, which runs about 65 MV lower and yields a "
            "10-14% lower interstellar spectrum below 2 GV. Use it to gauge the "
            "solar-modulation systematic."
        ),
    },
    "2025": {
        "status": "historical",
        "role": "superseded release",
        "covering": "see the GSF 2025 release notes",
        "solar_modulation": "USO (shared Usoskin table)",
        "description": "Previous published release. Superseded by GSF2026.",
    },
    "2019": {
        "status": "historical",
        "role": "superseded release",
        "covering": "see the GSF 2019 release notes",
        "solar_modulation": "USO (shared Usoskin table)",
        "description": "Legacy published release. Superseded by GSF2026.",
    },
    "2017": {
        "status": "historical",
        "role": "superseded release",
        "covering": "see Dembinski et al. (2017)",
        "solar_modulation": "USO (shared Usoskin table)",
        "description": (
            "Original GSF release (Dembinski et al. 2017). Superseded by GSF2026."
        ),
    },
}

_REQUIRED_FILES = ("knots.dat", "nuclei.dat", "parameters.dat", "covariance.dat")


def get_available_versions(include_historical: bool = True) -> list[str]:
    """Get the list of available GSF data versions.

    Only versions in the :data:`MODEL_VERSIONS` allow-list are returned, and only
    when their data files are actually present. A directory under ``data/`` that
    is not registered is never offered as a version -- it warns instead, so a
    transient or intermediate fit exported there cannot silently become
    distributable.

    Parameters
    ----------
    include_historical
        When True (default) previously published releases are included alongside
        the current fit. Pass False for just the current default and its
        sanctioned alternative.

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
    name = DEFAULT_VERSION if version is None else str(version).strip()
    if name not in MODEL_VERSIONS:
        raise ValueError(
            f"Unknown model version {name!r}. Available versions: "
            f"{get_available_versions()}"
        )
    return dict(MODEL_VERSIONS[name])


class Parameters:
    """Parameter loading and validation class for GSF model data.

    This class handles loading and validation of all GSF model parameters
    including knots, spline parameters, covariance matrices, nuclear data,
    and solar modulation data.

    Parameters
    ----------
    data_path : str or Path, optional
        Path to directory containing GSF data files. If None, uses
        :data:`DEFAULT_VERSION`.
    version : str, optional
        Model version to use. "GSF2026" (the default) is the promoted fit: the
        mixture covering -- an equal-weight combination of the Auger FD-2026
        SIBYLL-2.3e and EPOS-LHC-R interpretations -- with the
        Ghelfi-Maurin-Derome modulation potential. "GSF2026-USO" is the one
        sanctioned alternative: the same mixture fit with the Usoskin 2017
        potential (about 65 MV lower, giving a 10-14% lower interstellar
        spectrum below 2 GV). "2025", "2019" and "2017" are superseded
        historical releases, kept only so older work can be reproduced. See
        :data:`MODEL_VERSIONS`. If specified, overrides data_path and uses the
        corresponding package data directory.
    use_approximate_solar_cycle_average : bool, optional
        When True (default), solar cycle averages are calculated approximately
        from the average of monthly phi values. When False, averages are
        calculated explicitly by averaging monthly fluxes over the solar cycle.
    """

    def __init__(
        self,
        data_path: str | Path | None = None,
        version: str | None = None,
        use_approximate_solar_cycle_average: bool = True,
    ):
        """Initialize GSF parameters."""
        self.use_approximate_solar_cycle_average = use_approximate_solar_cycle_average
        self.data_path = self._setup_data_path(data_path, version)
        self._load_all_data()

    def _setup_data_path(
        self, data_path: str | Path | None, version: str | None
    ) -> Path:
        """Set up the data path, recording which version was resolved."""
        #: Resolved version name, or None when loading an arbitrary data_path.
        #: Unlike the constructor argument this is filled in for the default, so
        #: a model built with no arguments still reports what it loaded.
        self.version = None
        if version is not None:
            version_str = str(version).strip()
            if not version_str:
                raise ValueError(
                    f"Version '' not found. Available versions: {get_available_versions()}"
                )
            data_dir = Path(__file__).parent / "data" / version_str
            if not data_dir.exists():
                raise ValueError(
                    f"Version '{version_str}' not found. Available versions: {get_available_versions()}"
                )
            self.version = version_str
            return data_dir
        if data_path is None:
            data_dir = Path(__file__).parent / "data" / DEFAULT_VERSION
            if not data_dir.exists():
                raise OSError(f"Default data directory not found: {data_dir}")
            self.version = DEFAULT_VERSION
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
        """What this parameter set actually is, read from its ``fit_result.json``.

        Returns the recorded fit provenance -- most importantly ``covering`` (the
        air-shower interpretation, e.g. the SIBYLL/EPOS mixture) and
        ``solar_modulation_source`` (GMD or USO), the two axes that distinguish
        the fits. For a registered version the :data:`MODEL_VERSIONS` entry is
        merged in under ``registry``.

        Returns an empty dict for legacy sets, which predate the metadata file.
        """
        info: dict = {}
        meta_file = self.data_path / "fit_result.json"
        if meta_file.exists():
            try:
                loaded = json.loads(meta_file.read_text())
            except (OSError, ValueError):
                loaded = {}
            meta = loaded.get("metadata", loaded)
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
        self._load_nuclei_data()      # first: defines the species (Z, A) + groups
        self._load_knots()
        self._load_parameters()
        self._load_covariance()
        self._load_solar_modulation()
        self._load_subleading()
        self._calculate_flux_ratios()

    @staticmethod
    def _ncols(path) -> int:
        """Whitespace-token count of the first data (non-#) line (``:`` -> space).
        Used to auto-detect the v1 vs v2 .dat column layout."""
        for line in Path(path).read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                return len(line.replace(":", " ").split())
        return 0

    def _sid_for_z(self, z: int):
        """The unique species id ``(Z, A)`` at charge ``z`` (v1: one species per
        charge). Used to normalise v1 files, which key by charge only."""
        return self.z_to_sids[z][0]

    def _load_nuclei_data(self):
        """Load nuclear data from nuclei.dat. Species are identified by ``(Z, A)``
        so ISOTOPES (e.g. deuteron D at Z=1 alongside p) coexist. The leader of a
        charge group is the lightest-A species at the leader charge (p for the
        proton group; the single species otherwise)."""
        nuclei_file = self.data_path / "nuclei.dat"
        if not nuclei_file.exists():
            raise FileNotFoundError(f"Nuclei file not found: {nuclei_file}")

        data_array = np.atleast_1d(
            np.loadtxt(nuclei_file, dtype=[("z", int), ("a", float), ("l", int)]))
        rows = [(int(r["z"]), float(r["a"]), int(r["l"])) for r in data_array]

        self.species = sorted({(z, a) for z, a, _ in rows})  # all species ids
        self.z_to_a = {(z, a): a for z, a, _ in rows}        # sid -> A (charge
        #   aliases added by _add_charge_aliases; iterate self.species, not this)
        self.z_to_sids = {}                                  # charge -> [sids]
        for z, a, _ in rows:
            self.z_to_sids.setdefault(z, []).append((z, a))
        # leader sid per group charge l: the lightest-A species with z == l
        self.leader_sid = {}                                 # charge -> leader sid
        for z, a, l in rows:
            if z == l and (l not in self.leader_sid or a < self.leader_sid[l][1]):
                self.leader_sid[l] = (z, a)
        self._leaders = set(self.leader_sid.values())        # leader sids
        # Public group structure stays CHARGE-based (one entry per element charge)
        # so charge-indexed callers/tests are unchanged; isotopes are expanded
        # from a charge to its species ids (z_to_sids) inside the flux loops.
        self.z_group = {}      # int leader charge -> [member element charges]
        self.z_ungroup = {}    # element charge -> leader charge
        for z, a, l in rows:
            if z not in self.z_group.get(l, []):
                self.z_group.setdefault(l, []).append(z)
            self.z_ungroup[z] = l

    def _load_knots(self):
        """Load knot data. v2 lines are ``Z A: …``; v1 lines ``Z: …`` (keyed by
        the unique sid at that charge). Keys are species ids ``(Z, A)``."""
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
                x = np.log([10 ** float(v) for v in k.split()])
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
        dt = ([("i", int), ("z", int), ("a", float), ("val", float)] if v2
              else [("i", int), ("z", int), ("val", float)])
        data_array = np.atleast_1d(np.loadtxt(params_file, dtype=dt))

        for r in data_array:
            i, z, val = int(r["i"]), int(r["z"]), float(r["val"])
            sid = (z, float(r["a"])) if v2 else self._sid_for_z(z)
            if sid not in self.pars:
                # 4 extra zeros at the end are needed by splev
                self.pars[sid] = np.zeros(self.npar[sid] + 4)
                self.offset[sid] = i
            self.pars[sid][i - self.offset[sid]] = val

    def _load_covariance(self):
        """Load covariance. v2: ``i j Z1 A1 Z2 A2 val``; v1: ``i j Z1 Z2 val``.
        Keyed by species-id pairs ``((Z1,A1), (Z2,A2))``."""
        cov_file = self.data_path / "covariance.dat"
        if not cov_file.exists():
            raise FileNotFoundError(f"Covariance file not found: {cov_file}")

        self.cov = {}
        v2 = self._ncols(cov_file) >= 7
        dt = ([("i", int), ("j", int), ("z1", int), ("a1", float),
               ("z2", int), ("a2", float), ("val", float)] if v2
              else [("i", int), ("j", int), ("z1", int), ("z2", int), ("val", float)])
        data_array = np.atleast_1d(np.loadtxt(cov_file, dtype=dt))

        for r in data_array:
            i, j, val = int(r["i"]), int(r["j"]), float(r["val"])
            if v2:
                s1 = (int(r["z1"]), float(r["a1"]))
                s2 = (int(r["z2"]), float(r["a2"]))
            else:
                s1, s2 = self._sid_for_z(int(r["z1"])), self._sid_for_z(int(r["z2"]))
            if (s1, s2) not in self.cov:
                self.cov[(s1, s2)] = np.zeros((self.npar[s1], self.npar[s2]))
            if (s2, s1) not in self.cov:
                self.cov[(s2, s1)] = np.zeros((self.npar[s2], self.npar[s1]))
            self.cov[(s1, s2)][i - self.offset[s1], j - self.offset[s2]] = val
            self.cov[(s2, s1)][j - self.offset[s2], i - self.offset[s1]] = val

    def _load_solar_modulation(self):
        """Load the monthly solar-modulation potential table (phi, MV).

        Precedence: a VERSION-LOCAL ``<data_path>/solar_modulation.dat`` if the
        parameter set ships its own potential, else the shared table at the
        package data root. This matters because the LIS is demodulated with a
        specific phi(t): the fitted LIS must be re-modulated by the SAME
        potential to recover a flux at Earth. Sets whose LIS was demodulated
        with a non-default potential (e.g. GSF2026, Ghelfi-Maurin-Derome) ship
        their table alongside the parameters; legacy sets (2017/2019/2025) and
        the Usoskin variant (GSF2026-USO) fall back to the shared Usoskin table.

        Robust to both the shared file (UTF-16-BOM, Usoskin) and version-local
        UTF-8 files: encoding is detected from the byte-order mark, header lines
        are '#'-commented, and the columns are Year followed by the 12 monthly
        values (a trailing Annual column, present in the Usoskin file, is
        ignored). Rows with any NaN month (e.g. Jan 1951 in the Usoskin table)
        are dropped.
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
        """Load the OPTIONAL subleading.dat extrapolation table: per sub-leading
        species ``(Z, A, norm, slope)`` with ratio(R > Rmax) = norm *
        (R/Rmax)**slope above the species' top knot Rmax. Absent in pre-slope
        bundles (2017/2019/2025) -> empty table, and the constant-ratio
        extrapolation is recomputed from the splines exactly as before."""
        self._stored_sub = {}
        sub_file = self.data_path / "subleading.dat"
        if not sub_file.exists():
            return
        dt = [("z", int), ("a", float), ("norm", float), ("slope", float)]
        for r in np.atleast_1d(np.loadtxt(sub_file, dtype=dt)):
            self._stored_sub[(int(r["z"]), float(r["a"]))] = (
                float(r["norm"]), float(r["slope"]))

    def _calculate_flux_ratios(self):
        """Calculate flux ratios (and extrapolation slopes) for subleading
        species (keyed by species id). When subleading.dat supplied stored
        (norm, slope) values, those are used verbatim — they are the values the
        fit itself used; otherwise the ratio is recomputed from the splines at
        the species' top knot and the slope defaults to 0 (the historical
        constant-ratio extrapolation)."""
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
                        xmax, (self.kx[leader_sid], self.pars[leader_sid], SPLINE_DEGREE)
                    )
                    slope = 0.0
            else:
                ratio, slope = 1.0, 0.0
            self.flux_ratio[sid] = (leader_sid, ratio)
            self.flux_slope[sid] = slope
        self._add_charge_aliases()

    def _add_charge_aliases(self):
        """Expose the per-species dicts under a bare integer charge as well, for
        every charge with a SINGLE species (all charges in a v1 model). Lets
        charge-indexed access (existing callers/tests) coexist with (Z, A) keys.
        ``species`` (not ``z_to_a``) is the species iterator, so aliasing z_to_a
        is safe. Multi-species charges (e.g. Z=1 with p+D) get no int alias."""
        for z, sids in self.z_to_sids.items():
            if len(sids) != 1:
                continue
            s = sids[0]
            self.kx[z] = self.kx[s]
            self.npar[z] = self.npar[s]
            self.pars[z] = self.pars[s]
            self.offset[z] = self.offset[s]
            self.z_to_a[z] = self.z_to_a[s]
            self.flux_ratio[z] = self.flux_ratio[s]
            self.flux_slope[z] = self.flux_slope[s]
        for (s1, s2) in list(self.cov):
            z1, z2 = s1[0], s2[0]
            if len(self.z_to_sids[z1]) == 1 and len(self.z_to_sids[z2]) == 1:
                self.cov[(z1, z2)] = self.cov[(s1, s2)]

    def get_solar_cycle_24_interval(self) -> tuple[int, int]:
        """Get the time interval for Solar Cycle 24.

        Returns
        -------
        tuple[int, int]
            Time interval tuple (start, end) in YYYYMM format for Solar Cycle 24
            (December 2008 to December 2019)
        """
        return SOLAR_CYCLE_24_START[0], SOLAR_CYCLE_24_END[0]

    def get_solar_cycle_24_phi_average(self) -> float:
        """Get the average solar modulation potential for Solar Cycle 24.

        This method implements the approximate averaging by calculating the
        mean of all monthly phi values during Solar Cycle 24.

        Returns
        -------
        float
            Average solar modulation potential in GV for Solar Cycle 24
        """
        start, end = self.get_solar_cycle_24_interval()
        phis = _collect_phi_values(self.phi, start, end)
        return float(np.mean(phis))


def list_versions(return_paths: bool = False) -> None | list:
    """List available GSF data versions and their validity.

    This function scans the data directory for available GSF data versions,
    attempts to load parameters from each, and displays a table showing
    which versions are valid and which have errors.

    Parameters
    ----------
    return_paths : bool, optional
        If True, returns a list of valid data paths that can be used
        to initialize Parameters objects. If False (default), prints
        a table and returns None.

    Returns
    -------
    list or None
        If return_paths is True, returns list of valid Path objects.
        Otherwise prints table and returns None.
    """
    try:
        from tabulate import tabulate
    except ImportError:
        print("tabulate package not available. Install with: pip install tabulate")
        return [] if return_paths else None

    # Get the data directory from the package
    current_dir = Path(__file__).parent
    data_dir = current_dir / "data"

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return [] if return_paths else None

    results = []
    valid_paths = []

    # Get available versions using the helper function
    available_versions = get_available_versions()

    if not available_versions:
        print("No valid GSF data versions found in the data directory.")
        return [] if return_paths else None

    # Test each version by trying to initialize Parameters
    for version_name in available_versions:
        version_path = data_dir / version_name
        try:
            # Try to initialize Parameters with this data path
            _ = Parameters(data_path=version_path)
            results.append([version_name, "Valid", ""])
            valid_paths.append(version_path)

        except Exception as e:
            # Capture the error reason
            error_reason = str(e)
            # Truncate long error messages
            if len(error_reason) > 60:
                error_reason = error_reason[:57] + "..."
            results.append([version_name, "Invalid", error_reason])

    if return_paths:
        return valid_paths

    # Display results as a table
    if results:
        headers = ["Version", "Status", "Error Reason"]
        print(tabulate(results, headers=headers, tablefmt="grid"))
    else:
        print("No GSF data versions found in the data directory.")
