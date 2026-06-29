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


def get_available_versions() -> list[str]:
    """Get list of available GSF data versions.

    Returns
    -------
    list[str]
        List of available version names that can be used to initialize models.
    """
    current_dir = Path(__file__).parent
    data_dir = current_dir / "data"

    if not data_dir.exists():
        return []

    versions = []
    for version_path in data_dir.iterdir():
        if (
            version_path.is_dir()
            and not version_path.name.startswith(".")
            and not version_path.name.startswith("__")
        ):
            # Try to validate that this is a proper version by checking for required files
            required_files = [
                "knots.dat",
                "nuclei.dat",
                "parameters.dat",
                "covariance.dat",
            ]
            if all((version_path / file).exists() for file in required_files):
                versions.append(version_path.name)

    return sorted(versions)


class Parameters:
    """Parameter loading and validation class for GSF model data.

    This class handles loading and validation of all GSF model parameters
    including knots, spline parameters, covariance matrices, nuclear data,
    and solar modulation data.

    Parameters
    ----------
    data_path : str or Path, optional
        Path to directory containing GSF data files. If None, uses the
        default 2017 data version from the package.
    version : str, optional
        Model version to use ("2017", "2019", "2025"). If specified,
        overrides data_path and uses the corresponding package data directory.
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
        """Set up the data path."""
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
            return data_dir
        if data_path is None:
            data_dir = Path(__file__).parent / "data" / "2017"
            if not data_dir.exists():
                raise OSError(f"Default data directory not found: {data_dir}")
            return data_dir
        path = Path(data_path)
        if not path.exists():
            raise OSError(f"Data path does not exist: {path}")
        return path

    def _load_all_data(self):
        """Load all required GSF data files."""
        self._load_nuclei_data()      # first: defines the species (Z, A) + groups
        self._load_knots()
        self._load_parameters()
        self._load_covariance()
        self._load_solar_modulation()
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
        """Load solar modulation data from solar_modulation.dat.

        The solar modulation file is observational data shared across all
        model versions. It always lives at the package's data root, not
        inside any version directory, so it's looked up independently of
        ``self.data_path``.
        """
        solar_file = Path(__file__).parent / "data" / "solar_modulation.dat"
        if not solar_file.exists():
            raise FileNotFoundError(f"Solar modulation file not found: {solar_file}")

        self.phi = {}
        # File is UTF-16 with BOM (provenance: external observational source).
        data_array = np.loadtxt(solar_file, skiprows=13, encoding="utf-16")

        for row in data_array:
            self.phi[int(row[0])] = row[1:] * 1e-3  # Convert to GV

    def _calculate_flux_ratios(self):
        """Calculate flux ratios for subleading species (keyed by species id)."""
        self.flux_ratio = {}

        for sid in self.species:
            leader_sid = self.leader_sid[self.z_ungroup[sid[0]]]
            xmax = self.kx[sid][-1]
            if sid != leader_sid:
                # Subleading species - ratio to its group leader
                ratio = splev(
                    xmax, (self.kx[sid], self.pars[sid], SPLINE_DEGREE)
                ) / splev(
                    xmax, (self.kx[leader_sid], self.pars[leader_sid], SPLINE_DEGREE)
                )
            else:
                ratio = 1.0
            self.flux_ratio[sid] = (leader_sid, ratio)
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
