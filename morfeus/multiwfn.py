"""Multiwfn (http://sobereva.com/multiwfn/) interface code using pexpect for interactive control."""

from __future__ import annotations

from dataclasses import dataclass
import functools
from pathlib import Path
import re
import shutil
from typing import Any, Iterable

import pexpect

from morfeus.utils import requires_executable

REAL_SPACE_FUNCTIONS = {
    "rho": "1 Electron density",
    "rho_grad_norm": "2 Gradient norm of rho",
    "rho_lapl": "3 Laplacian of rho",
    # "orbital_wfn": "4 Value of orbital wavefunction",
    # "orbital_prop": "44 Orbital probability density",
    "spin_density": "5 Electron spin density",
    "Kr": "6 Hamiltonian kinetic energy density",
    "Gr": "7 Lagrangian kinetic energy density",
    "ESP_nuclear": "8 Electrostatic potential from nuclear charges",
    "ELF": "9 Electron localization function",
    "LOL": "10 Localized orbital locator",
    "LIE": "11 Local information entropy",
    "ESP_total": "12 Total electrostatic potential",
    "RDG": "13 Reduced density gradient",  # only for grid
    "RDG_pro": "14 RDG with promolecular approximation",
    "sign": "15 Sign",  # (lambda2)*rho",
    "sign_pro": "16 Sign",  # (lambda2)*rho with promolecular approximation",
    "correlation_alpha": "17 Correlation hole for alpha",
    "ALIE": "18 Average local ionization energy",
    # "source":"19 Source function, mode: 1",
    # "EDR": "20 Electron delocal. range func.",
    # "Dr": "21 Orbital overlap dist. func.",
    "deltag_pro": "22 Delta-g",
    "deltag_hirsh": "23 Delta-g",
    "IRI": "24 Interaction region indicator",
    "vdw": "25 van der Waals potential",
}


@dataclass
class CommandStep:
    """Command step with optional flag for conditional execution.

    Args:
        cmd: Command to send
        expect: Pattern to expect before sending
        optional: If True, only execute if expect pattern is found in output;
                  if False, always execute regardless
        wait: If True, wait for any progress bar to complete before sending
    """

    cmd: str | None = None
    expect: str | None = None
    optional: bool = False
    wait: bool = False


@dataclass
class MultiwfnRunResult:
    """Result from a single Multiwfn run."""

    stdout: str
    workdir: Path


@dataclass
class MultiwfnResults:
    """Cached Multiwfn results."""

    charges: dict[str, dict[int, float]] | None = None
    surfaces: dict[str, dict[str, Any]] | None = None
    atomic_descriptors: dict[int, float] | None = None
    grid_descriptors: dict[int, float] | None = None
    electric_moments: dict[str, float] | None = None
    citations: set[str] | None = None


class _PexpectSession:
    """Pexpect session manager for Multiwfn."""

    def __init__(
        self,
        base_command: str,
        workdir: Path,
        debug: bool,
    ) -> None:
        self._child = pexpect.spawn(
            base_command,
            cwd=str(workdir),
            encoding="utf-8",
        )
        self._timeout = 10
        self._expect_timeout = 3
        self._debug = debug
        self._transcript: list[str] = []
        self._last_command_pos = 0

    def read_available(self) -> None:
        """Read available output without blocking."""
        try:
            chunk = self._child.read_nonblocking(size=4096, timeout=0.1)
            if chunk:
                self._transcript.append(chunk)
        except (pexpect.TIMEOUT, pexpect.EOF):
            pass

    def wait_for_progress(self) -> None:
        """Wait for any ongoing progress bar to complete."""
        max_retries = 100
        retry_count = 0

        while retry_count < max_retries:
            # Read new output with a short wait
            try:
                chunk = self._child.read_nonblocking(size=4096, timeout=0.5)
                if chunk:
                    self._transcript.append(chunk)
                    # Check only the newly read chunk for progress bar
                    has_progress, current_progress = self._has_progress_bar(chunk)
                    if has_progress:
                        retry_count += 1
                        if self._debug:
                            progress_info = (
                                f" ({current_progress}%)"
                                if current_progress is not None
                                else ""
                            )
                            print(f"[DEBUG] Progress bar detected{progress_info}")
                        continue
                    # New output without progress bar - done
                    return
            except pexpect.TIMEOUT:
                # No new output available - done waiting
                return
            except pexpect.EOF:
                return

    def send(self, cmd: str, wait: bool = False) -> None:
        """Send command to process.

        Args:
            cmd: Command to send
            wait: If True, wait for any progress bar to complete first
        """
        if wait:
            self.wait_for_progress()
        self._child.sendline(cmd)
        if self._debug:
            print(f"[DEBUG] Sent: {cmd!r}")
        self._last_command_pos = len(self._transcript)

    def _has_progress_bar(self, text: str) -> tuple[bool, float | None]:
        """Check if text contains a progress bar and extract percentage.

        Returns:
            Tuple of (has_progress_bar, percentage)
        """
        # Match patterns like: "Progress: [###---] 14.3 %" or "14.3 %"
        progress_pattern = r"Progress:\s*\[[#-]+\]\s+([\d.]+)\s*%"
        match = re.search(progress_pattern, text)
        if match:
            try:
                percentage = float(match.group(1))
                return True, percentage
            except ValueError:
                return True, None
        return False, None

    def expect(self, pattern: str) -> bool:
        """Wait for pattern. Returns True on match, False on timeout."""
        if self._debug:
            print(f"[DEBUG] Expecting: {pattern!r}")

        try:
            self._child.expect(pattern, timeout=self._expect_timeout)
            self._transcript.append(self._child.before or "")
            self._transcript.append(self._child.after or "")
            if self._debug:
                print("[DEBUG] Got match")
            return True
        except pexpect.TIMEOUT:
            self.read_available()
            recent_output = self.get_output_since_last_command()
            print(f"[WARN] Timeout ({self._expect_timeout}s) waiting for: {pattern!r}")
            print(f"Recent output: {recent_output[-500:]}")
            return False

    def try_expect(self, pattern: str, timeout: float = 1.0) -> bool:
        """Try to match pattern with short timeout. Silent on timeout.

        Does not extend timeout for progress bars (use expect() for that).
        """
        if self._debug:
            print(f"[DEBUG] Try expecting: {pattern!r} (timeout={timeout}s)")
        try:
            self._child.expect(pattern, timeout=timeout)
            self._transcript.append(self._child.before or "")
            self._transcript.append(self._child.after or "")
            if self._debug:
                print("[DEBUG] Got match")
            return True
        except pexpect.TIMEOUT:
            self.read_available()
            if self._debug:
                print("[DEBUG] Pattern not found (optional)")
            return False

    def wait_for_exit(self) -> None:
        """Wait for process to exit."""
        try:
            self._child.expect(pexpect.EOF, timeout=self._timeout)
        except pexpect.TIMEOUT:
            print("[WARN] Timeout waiting for EOF")
        self._transcript.append(self._child.before or "")
        if self._child.isalive():
            self._child.close()

    def get_recent_output(self, n_chunks: int = 3, max_chars: int = 1000) -> str:
        """Get recent output for debugging."""
        recent = (
            self._transcript[-n_chunks:]
            if len(self._transcript) >= n_chunks
            else self._transcript
        )
        output = "".join(recent)
        return output[-max_chars:] if len(output) > max_chars else output

    def get_output_since_last_command(self) -> str:
        """Get output since last command."""
        if self._last_command_pos < len(self._transcript):
            return "".join(self._transcript[self._last_command_pos :])
        return ""

    @property
    def stdout(self) -> str:
        """Full session transcript."""
        return "".join(self._transcript)

    def _execute_command(
        self, cmd: str, expect: str | None, use_robust: bool, wait: bool = False
    ) -> None:
        """Execute command with optional expect."""
        if use_robust and expect:
            self.expect(expect)
        self.send(cmd, wait=wait)
        if not use_robust:
            self.read_available()

    def _process_command(self, item: str | CommandStep, use_robust: bool) -> None:
        """Process command step."""
        if isinstance(item, str):
            self._execute_command(item, None, use_robust)
            return

        if item.optional:
            # For optional commands, try to match the pattern with a short timeout
            # If the pattern is found, execute the command; otherwise skip it
            if item.expect:
                pattern_found = self.try_expect(item.expect, timeout=1.0)
                if self._debug:
                    print(
                        f"[DEBUG] Optional pattern {'found' if pattern_found else 'not found'}: {item.expect!r}"
                    )
                if not pattern_found:
                    return
            if item.cmd:
                self.send(item.cmd, wait=item.wait)
                if not use_robust:
                    self.read_available()
        else:
            if item.cmd:
                self._execute_command(item.cmd, item.expect, use_robust, wait=item.wait)


class Multiwfn:
    """Multiwfn interface using pexpect for interactive control.

    Args:
        file_path: Path to molden/wfn file from xtb or other QM program.
        run_path: Directory for output files. Defaults to temporary directory.
        robust_mode: If True, wait for expect patterns between commands (slower but safer).
                     If False, send commands without waiting (faster but may desync).
        debug: If True, print debug info including sent commands and expect patterns.
    """

    def __init__(
        self,
        file_path: str | Path,
        run_path: str | Path | None = None,
        robust_mode: bool = True,
        debug: bool = False,
    ) -> None:
        self._file_path = Path(file_path).resolve()
        if not self._file_path.exists():
            raise FileNotFoundError(f"File not found: {self._file_path}")

        self._run_path = Path(run_path).resolve() if run_path else None
        self._robust_mode = robust_mode
        self._debug = debug
        self._results = MultiwfnResults()

        self._results.citations = {
            # Both following papers ***MUST BE CITED IN MAIN TEXT*** if Multiwfn is used:
            # See "How to cite Multiwfn.pdf" in Multiwfn binary package for more information
            "Tian Lu, Feiwu Chen, J. Comput. Chem., 33, 580 (2012) DOI: 10.1002/jcc.22885",
            "Tian Lu, J. Chem. Phys., 161, 082503 (2024) DOI: 10.1063/5.0216272",
        }

    def load_settingsini(self, file_path: str | Path) -> None:
        """Load custom settings.ini file.
        Overwriting morfeus/config/settings.ini.

        Args:
            file_path: Path to custom settings.ini file.
        """
        settings_path = Path(file_path).resolve()
        if not settings_path.exists() or settings_path.name != "settings.ini":
            raise FileNotFoundError(f"Settings file not found: {settings_path}")
        self._setup_settings_ini(settings_path)

    def _setup_settings_ini(
        self, target_path: str | Path = None, file_path: str | Path = None
    ) -> None:
        """Copy settings.ini to run_path."""
        if target_path is None:
            target_path = self._run_path / "settings.ini"

        target = (
            Path(target_path) / "settings.ini"
            if Path(target_path).suffix != ".ini"
            else Path(target_path)
        )

        if target.exists():
            return

        # Use package default
        if file_path is None:
            source = Path(__file__).parent / "config" / "settings.ini"
        else:
            source = Path(file_path).resolve()

        if source.exists():
            shutil.copy2(source, target)

    def _commands_change_isuerfunc(self, function: int = -1) -> None:
        """Adjust the iuserfunc setting.
        Refer to for a complete list (ctrl F: !Below functions can be selected by "iuserfunc" parameter):
        https://github.com/foxtran/MultiWFN/blob/master/src/function.f90
        """
        return [
            CommandStep("1000", expect="300 Other functions (Part 3)"),
            CommandStep("2", expect="2 Set iuserfunc"),
            CommandStep(
                str(function), expect="Input index of the user-defined function"
            ),
        ]

    def _setup_workdir(self, subdir: str | None) -> Path:
        """Get or create working directory."""
        workdir = self._run_path / subdir if subdir else self._run_path
        workdir.mkdir(parents=True, exist_ok=True)

        # Copy the settings.ini file from either the run_path or package default
        self._setup_settings_ini(workdir)
        return workdir

    @requires_executable(["Multiwfn"])
    def run_commands(
        self,
        commands: Iterable[str | CommandStep],
        subdir: str | None = None,
    ) -> MultiwfnRunResult:
        """Run Multiwfn with a custom command sequence.

        Args:
            commands: Sequence of commands (strings or CommandStep objects).
            subdir: Subdirectory within run_path to store results.
        Returns:
            MultiwfnRunResult with stdout, workdir, and list of generated files.
        """

        workdir = self._setup_workdir(subdir)

        session = _PexpectSession("Multiwfn", workdir, self._debug)
        session.send(str(self._file_path))
        session.read_available()

        for item in commands:
            session._process_command(item, self._robust_mode)

        session.wait_for_exit()

        return MultiwfnRunResult(
            stdout=session.stdout,
            workdir=workdir,
        )

    def get_charges(self, model: str = "ADCH") -> dict[int, float]:
        """Calculate atomic charges.

        Args:
            model: Charge model to use.
                - 'Hirshfeld': Standard Hirshfeld charges
                - 'ADCH': Atomic dipole moment corrected Hirshfeld charges (more accurate)
                - 'VDD': Voronoi deformation density
                - 'Mulliken': Mulliken charges
                - 'CM5': CM5 charges
                - '12CM5': 1.2*CM5 charges

        Raises:
            ValueError: If given charge model is not available.

        Returns:
            Atomic charges indexed by atom number (1-indexed)
        """
        if self._results.charges is not None and model in self._results.charges:
            return self._results.charges[model]

        models = {
            "Hirshfeld": "1 Hirshfeld atomic",
            "VDD": "2 Voronoi deformation density",
            "Mulliken": "5 Mulliken atom & basis function",
            "ADCH": "11 Atomic dipole corrected Hirshfeld",
            "CM5": "16 CM5 atomic charge",
            "12CM5": "-16 Generate 1.2*CM5",
        }
        citations = {
            "Hirshfeld": "Theor. Chim. Acta. (Berl), 44, 129-138 (1977)",
            "VDD": "J. Comput. Chem., 25, 189.",
            "ADCH": "Tian Lu, Feiwu Chen, Atomic dipole moment corrected Hirshfeld population method, J. Theor. Comput. Chem., 11, 163 (2012)",
        }

        if model not in models:
            raise ValueError(
                f"Charge model {model!r} not supported. Choose between {', '.join(models.keys())}."
            )

        menu_pattern = models[model]
        menu_cmd = menu_pattern.split(" ")[0]

        commands = [
            CommandStep("7", expect="7 Population analysis"),
            CommandStep(menu_cmd, expect=menu_pattern),
            CommandStep(
                "1",
                expect=(
                    "1 Use build-in sphericalized atomic"
                    if model != "Mulliken"
                    else " 1 Output Mulliken population and atomic"
                ),
            ),
            CommandStep("y", expect="charges to molden.chg"),
            CommandStep("0", expect="Mulliken population analysis", optional=True),
            CommandStep("0", expect="0 Return"),
            CommandStep("q", expect="gracefully"),
        ]

        citation = citations.get(model, None)
        if citation is not None:
            self._results.citations.add(citation)

        result = self.run_commands(commands, subdir=model)

        chg_files = sorted(result.workdir.glob("*.chg"))
        if not chg_files:
            charges = {}
        else:
            charges = self._parse_chg_file(chg_files[0])

        if self._results.charges is None:
            self._results.charges = {}
        self._results.charges[model] = charges

        return charges

    @requires_executable(["Multiwfn"])
    def get_surface(self, model: str = "ESP") -> dict[int, dict[str, float]]:
        """Calculate molecular surface properties.

        Args:
            model: Surface analysis model to use.
                - 'ESP': Electrostatic Potential
                - 'ALIE': Average Local Ionization Energy
                - 'LEA': Local Electron Affinity
                - 'LEAE': Local Electron Attachment Energy
                - 'Electron': Electron Density
                - 'Sign': Sign(lambda2)*rho

        Raises:
            ValueError: If given model is not available.

        Returns:
            Dictionary with 'atomic' and 'global' keys:
                - 'atomic': Dict mapping atom index (1-based) to atomic properties
                    - area_total, area_positive, area_negative: Surface areas (Ų)
                    - min_value, max_value: Min/max function values
                    - avg_all, avg_positive, avg_negative: Average values
                    - var_all, var_positive, var_negative: Variance values
                - 'global': Dict with global surface statistics
                    - surface_area, volume, average, variance, minimum, maximum
        """
        if self._results.surfaces is not None and model in self._results.surfaces:
            return self._results.surfaces[model]

        models = {
            "ESP": "1 Electrostatic potential",
            "ALIE": "2 Average local ionization energy",
            "LEA": "4 Local electron affinity",
            "LEAE": "-4 Local electron attachment energy",
            "Electron": "11 Electron density",
            "Sign": "12 Sign",
        }

        if model not in models:
            raise ValueError(
                f"Surface model {model!r} not supported. Choose between {', '.join(models.keys())}."
            )
        menu_pattern = models[model]
        menu_cmd = menu_pattern.split(" ")[0]

        self._results.citations.add(
            "Tian Lu, Feiwu Chen, Quantitative analysis of molecular surface based on improved Marching Tetrahedra algorithm, J. Mol. Graph. Model., 38, 314-323 (2012)"
        )

        commands = [
            CommandStep("12", expect="12 Quantitative analysis of molecular surface"),
            CommandStep("2", expect="2 Select mapped function"),
            CommandStep(menu_cmd, expect=menu_pattern),
            CommandStep("0", expect="0 Start analysis", wait=True),
            CommandStep("1", expect="1 Export surface extrema as surfanalysis.txt"),
            CommandStep("11", expect="11 Output surface properties"),
            CommandStep("n", expect="surface facets to locsurf.pqr"),
            CommandStep("-1", expect="-1 Return to upper level"),
            CommandStep("-1", expect="-1 Return to main menu"),
            CommandStep("q", expect="Exit program gracefully"),
        ]

        result = self.run_commands(commands, subdir=model)

        atomic_props = self._parse_atomic_surface_properties(result.stdout)
        surfanalysis_file = result.workdir / "surfanalysis.txt"
        global_props = self.parse_surfanalysis(surfanalysis_file).get("statistics", {})
        surface_result = {"atomic": atomic_props, "global": global_props}

        if self._results.surfaces is None:
            self._results.surfaces = {}
        self._results.surfaces[model] = surface_result

        return surface_result

    def get_bond_order(self, model: str) -> dict[tuple[int, int], float]:
        raise NotImplementedError()

    @requires_executable(["Multiwfn"])
    def grid_to_descriptors(
        self, grid_path, userfunction: int = -1
    ) -> dict[int, float]:
        """Calculate atomic densities from integration in fuzzy atomic spaces.

        Args:
            grid_path: Path to a grid file in .cub or .grd
            userfunction: User-defined function index for integration.

        Returns:
            Dictionary mapping atom index (1-based) to integrated density value.

        Raises:
            ValueError: If input file is not a cube/grid file.
        """
        if grid_path.suffix.lower() not in {".cub", ".grd"}:
            raise ValueError("Density analysis requires a .cub or .grd file as input.")

        if self._results.grid_descriptors is not None:
            return self._results.grid_descriptors

        commands = self._commands_change_isuerfunc(function=userfunction) + [
            CommandStep("15", expect="15 Fuzzy atomic space analysis"),
            CommandStep("1", expect="1 Perform integration in fuzzy atomic spaces"),
            CommandStep("100", expect="100 User-defined function", wait=True),
            CommandStep("0", expect="0 Return"),
            CommandStep("q", expect="Exit program gracefully"),
        ]

        result = self.run_commands(commands, subdir="grid")
        grid_descriptors = self._parse_atomic_values(result.stdout)

        self._results.grid_descriptors = grid_descriptors

        return grid_descriptors

    def molden_to_grid(
        self, descriptor: str, grid_quality: str, grid_file_name: str = None
    ) -> str:
        """Generate a grid (.cub) file for a real space function.
        Args:
            descriptor: Descriptor to calculate on grid.
            grid_quality: Grid quality, one of 'low', 'medium', 'high'.
            grid_file_name: Optional name for output cube file.
        """
        if descriptor not in REAL_SPACE_FUNCTIONS:
            raise ValueError(
                f"Descriptor {descriptor!r} not supported. Choose between {', '.join(REAL_SPACE_FUNCTIONS.keys())}."
            )

        menu_pattern = REAL_SPACE_FUNCTIONS[descriptor]
        menu_cmd = menu_pattern.split(" ")[0]

        grids = {
            "low": "1 Low quality grid",
            "medium": "2 Medium quality grid",
            "high": "3 High quality grid",
        }
        grid_pattern = grids[grid_quality]
        grid_cmd = grid_pattern.split(" ")[0]

        commands = [
            CommandStep(
                "5",
                expect="5 Output and plot specific property within a spatial region",
            ),
            CommandStep(menu_cmd, expect=menu_pattern),
            CommandStep(grid_cmd, expect=grid_pattern),
            CommandStep(
                "2", expect="2 Export data to a Gaussian-type cube file", wait=True
            ),
            CommandStep("0", expect="0 Return to main menu"),
            CommandStep("q", expect="gracefully"),
        ]
        result = self.run_commands(commands, subdir=descriptor)

        cub_file_name = next(f for f in result.workdir.iterdir() if f.suffix == ".cub")
        if grid_file_name is not None:  # Rename the file
            new_path = result.workdir / grid_file_name
            cub_file_name.rename(new_path)
            return new_path

        return result.workdir / cub_file_name

    def get_descriptor(self, descriptor: str) -> dict[int, float]:
        """Calculate atomic densities from integration in fuzzy atomic spaces.

        Args:
            descriptor: Descriptor to calculate on grid.
        """
        if descriptor not in REAL_SPACE_FUNCTIONS:
            raise ValueError(
                f"Descriptor {descriptor!r} not supported. Choose between {', '.join(REAL_SPACE_FUNCTIONS.keys())}."
            )

        if (
            self._results.atomic_descriptors is not None
            and descriptor in self._results.atomic_descriptors
        ):
            return self._results.atomic_descriptors[descriptor]

        menu_pattern = REAL_SPACE_FUNCTIONS[descriptor]
        menu_cmd = menu_pattern.split(" ")[0]

        if menu_cmd in [13]:
            raise ValueError(f"Descriptor {descriptor!r} requires grid data!")

        commands = [
            CommandStep("15", expect="15 Fuzzy atomic space analysis"),
            CommandStep("1", expect="1 Perform integration in fuzzy atomic spaces"),
            CommandStep(menu_cmd, expect=menu_pattern),
            CommandStep("0", expect="0 Return"),
            CommandStep("q", expect="gracefully"),
        ]
        result = self.run_commands(commands, subdir=descriptor)
        descriptor_result = self._parse_atomic_values(result.stdout)

        if self._results.atomic_descriptors is None:
            self._results.atomic_descriptors = {}
        self._results.atomic_descriptors[descriptor] = descriptor_result

        return descriptor_result

    """
    def molden_to_grid_descriptors(self, descriptor, grid_parameters) -> dict:
        grid_path = self.molden_to_grid(descriptor, grid_parameters)
        grid_descriptors = self.grid_to_descriptors(grid_path)
        return grid_descriptors
    """

    def get_electric_moments(self) -> dict:
        """Calculate electric dipole/multipole moments."""
        commands = [
            CommandStep("300", expect="300 Other functions (Part 3)"),
            CommandStep("5", expect="5 Calculate electric dipole/multipole moments"),
            CommandStep("0", expect="0 Return"),
            CommandStep("q", expect="gracefully"),
        ]
        result = self.run_commands(commands, subdir="electric_moments")
        moments = self._parse_electric_moments(result.stdout)

        if self._results.electric_moments is None:
            self._results.electric_moments = {}
        self._results.electric_moments.update(moments)

        return moments

    def _parse_electric_moments(self, stdout: str) -> dict:
        """Parse electric moments from stdout."""
        patterns = {
            "dipole_magnitude_au": r"Magnitude of dipole moment:\s*([-\d.]+)\s+a\.u\.",
            "quadrupole_traceless_magnitude": r"Magnitude of the traceless quadrupole moment tensor:\s*([-\d.]+)",
            "quadrupole_spherical_magnitude": r"Magnitude: \|Q_2\|=\s*([-\d.]+)",
            "octopole_spherical_magnitude": r"Magnitude: \|Q_3\|=\s*([-\d.]+)",
            "electronic_spatial_extent": r"Electronic spatial extent <r\^2>:\s*([-\d.]+)",
        }

        moments = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, stdout)
            if match:
                moments[key] = float(match.group(1))

        return moments

    def _parse_chg_file(self, chg_path: Path) -> dict[int, float]:
        """Parse .chg file to {atom_idx: charge}."""
        charges = {}
        with chg_path.open("r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f, start=1):
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    charges[idx] = float(parts[4])
                except ValueError:
                    continue
        return charges

    def _parse_atomic_values(self, stdout: str) -> dict[int, float]:
        """Parse fuzzy atomic space integration values from stdout."""
        lines = stdout.split("\n")
        atomic_values = {}

        for i, line in enumerate(lines):
            if "Atomic space" in line and "Value" in line:
                for j in range(i + 1, len(lines)):
                    data_line = lines[j]
                    if "Summing up" in data_line or not data_line.strip():
                        break
                    # Parse format: "    1(C )            0.00607663"
                    match = re.match(
                        r"\s*(\d+)\([A-Z][a-z]?\s*\)\s+([\d.-]+)", data_line
                    )
                    if match:
                        atom_idx = int(match.group(1))
                        value = float(match.group(2))
                        atomic_values[atom_idx] = value
                break

        return atomic_values

    def _parse_atomic_areas(
        self, lines: list[str], start_idx: int
    ) -> tuple[dict[int, dict[str, float]], int]:
        """Parse atomic surface areas section from stdout."""
        atomic_props = {}
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if "Atom#   All/Positive/Negative average" in line:
                return atomic_props, i
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 6 and parts[0].isdigit():
                try:
                    atom_idx = int(parts[0])
                    atomic_props[atom_idx] = {
                        "area_total": float(parts[1]),
                        "area_positive": float(parts[2]),
                        "area_negative": float(parts[3]),
                        "min_value": float(parts[4]),
                        "max_value": float(parts[5]),
                    }
                except ValueError:
                    continue
        return atomic_props, len(lines)

    def _parse_atomic_averages(
        self, lines: list[str], atomic_props: dict, start_idx: int
    ) -> dict[int, dict[str, float]]:
        """Parse atomic surface averages section from stdout."""
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if not line.strip():
                break
            parts = line.split()
            if len(parts) >= 7 and parts[0].isdigit():
                try:
                    atom_idx = int(parts[0])
                    if atom_idx in atomic_props:
                        atomic_props[atom_idx].update(
                            {
                                "avg_all": float(parts[1]),
                                "avg_positive": float(parts[2]),
                                "avg_negative": float(parts[3]),
                                "var_all": float(parts[4]),
                                "var_positive": float(parts[5]),
                                "var_negative": float(parts[6]),
                            }
                        )
                except ValueError:
                    continue
        return atomic_props

    def _parse_atomic_surface_properties(
        self, stdout: str
    ) -> dict[int, dict[str, float]]:
        """Parse atomic surface properties from stdout."""
        lines = stdout.split("\n")
        atomic_props = {}

        for i, line in enumerate(lines):
            if "Atom#" in line:
                atomic_props, avg_idx = self._parse_atomic_areas(lines, i)
                self._parse_atomic_averages(lines, atomic_props, avg_idx)
                break

        return atomic_props

    def parse_surfanalysis(self, filepath: Path) -> dict[str, Any]:
        """Parse surfanalysis.txt for global surface statistics."""
        data: dict[str, Any] = {"raw": "", "statistics": {}}

        if not filepath.exists():
            return data

        with filepath.open("r", encoding="utf-8", errors="replace") as f:
            content = f.read()
            data["raw"] = content

            minima_match = re.search(r"Number of surface minima:\s*(\d+)", content)
            maxima_match = re.search(r"Number of surface maxima:\s*(\d+)", content)

            if minima_match:
                data["statistics"]["num_minima"] = int(minima_match.group(1))
            if maxima_match:
                data["statistics"]["num_maxima"] = int(maxima_match.group(1))

            extrema_pattern = r"[\*\s]+\d+\s+([\d.]+)\s+[\d.-]+\s+[\d.-]+"
            matches = re.findall(extrema_pattern, content)

            if matches:
                extrema_values = [float(m) for m in matches]
                data["statistics"]["num_extrema"] = len(extrema_values)
                data["statistics"]["extrema_min"] = min(extrema_values)
                data["statistics"]["extrema_max"] = max(extrema_values)
                data["statistics"]["extrema_average"] = sum(extrema_values) / len(
                    extrema_values
                )

        return data


def cli(file: str) -> Any:
    """CLI entry point for Multiwfn."""
    partial_func = functools.partial(Multiwfn, file)
    return partial_func


def molden2aim(
    path: str,
    molden_name: str = "xtb.molden",
    molden2aim_executable_path: str = "molden2aim.exe",
) -> str:
    """
    Utility function to convert normalize molden file using molden2aim (https://github.com/zorkzou/Molden2AIM/tree/main).
    Args:
        path (str): Path to working directory in which a molden2aim executable is located.
        molden_name (str): Name of the molden file to convert.
    """
    session = _PexpectSession(
        path,
        base_command=f"./{molden2aim_executable_path} -i {molden_name}",
    )
    commands = ["Yes", "No", "No", "No"]
    for item in commands:
        session.send(item)
    session.wait_for_exit()

    return molden_name.split(".molden")[0] + "_new.molden"
