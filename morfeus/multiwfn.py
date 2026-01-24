"""Multiwfn (http://sobereva.com/multiwfn/) interface code.

Uses pexpect for interactive control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import functools
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, cast, Iterable

import pexpect

from morfeus.utils import build_execution_env, requires_executable

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
    """

    cmd: str | None = None
    expect: str | None = None
    optional: bool = False


@dataclass
class WaitProgress:
    """Wait for a progress bar to complete."""

    max_wait: float = 30.0


@dataclass
class ProgressState:
    """Track progress bar waiting state."""

    last_activity: float
    saw_progress: bool
    saw_please_wait: bool
    idle_timeout: float


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
    atomic_descriptors: dict[str, dict[int, float]] | None = None
    grid_descriptors: dict[int, float] | None = None
    electric_moments: dict[str, float] | None = None
    bond_orders: dict[str, dict[tuple[int, int], float]] | None = None
    citations: set[str] = field(default_factory=set)


class _PexpectSession:
    """Pexpect session manager for Multiwfn."""

    def __init__(
        self,
        base_command: str,
        workdir: Path,
        debug: bool,
        env: dict[str, str] | None = None,
    ) -> None:
        self._child = pexpect.spawn(
            base_command,
            cwd=str(workdir),
            encoding="utf-8",
            env=env,
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

    def wait_for_progress(self, max_wait: float = 30.0) -> None:
        """Wait for a progress bar to appear and complete."""
        state = self._init_progress_state(max_wait)
        if self._debug:
            print("[DEBUG] Waiting for progress bar " f"(max_idle_wait={max_wait}s)")
        while True:
            chunk = self._read_progress_chunk()
            if chunk is None:
                if self._should_stop_on_timeout(state):
                    return
                continue
            if chunk == "EOF":
                return
            if chunk == "":
                continue
            if self._handle_progress_chunk(state, chunk):
                return

    def send(self, cmd: str) -> None:
        """Send command to process.

        Args:
            cmd: Command to send
        """
        self._child.sendline(cmd)
        if self._debug:
            print(f"[DEBUG] Sent: {cmd!r}")
        self._last_command_pos = len(self._transcript)

    def _has_progress_bar(self, text: str) -> tuple[bool, float | None]:
        """Check if text contains a progress bar and extract percentage.

        Args:
            text: Text chunk to inspect.

        Returns:
            Tuple of (has_progress_bar, percentage).
        """
        bar_match = re.search(r"\[([#=\-]+)\]", text)
        if bar_match:
            bar = bar_match.group(1)
            if "-" not in bar:
                return True, 100.0
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
        """Try to match a pattern with short timeout. Silent on timeout.

        Does not extend timeout for progress bars (use expect() for that).

        Args:
            pattern: Regular expression to match.
            timeout: Timeout in seconds.

        Returns:
            True if the pattern is matched.
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

    def _execute_command(self, cmd: str, expect: str | None, use_robust: bool) -> None:
        """Execute command with optional expect."""
        if use_robust and expect:
            self.expect(expect)
        self.send(cmd)
        if not use_robust:
            self.read_available()

    def _process_command(
        self, item: str | CommandStep | WaitProgress, use_robust: bool
    ) -> None:
        """Process command step."""
        if isinstance(item, str):
            self._execute_command(item, None, use_robust)
            return
        if isinstance(item, WaitProgress):
            if self._debug:
                print(f"[DEBUG] Executing WaitProgress (max_wait={item.max_wait}s)")
            self.wait_for_progress(max_wait=item.max_wait)
            return

        if item.optional and item.expect:
            pattern_found = self.try_expect(item.expect, timeout=1.0)
            if self._debug:
                status = "found" if pattern_found else "not found"
                print(f"[DEBUG] Optional pattern {status}: {item.expect!r}")
            if not pattern_found:
                return

        if item.cmd:
            self._execute_command(item.cmd, item.expect, use_robust)

    def _init_progress_state(self, max_wait: float) -> ProgressState:
        """Initialize progress waiting state.

        Args:
            max_wait: Maximum idle time in seconds before timing out.

        Returns:
            Initialized progress waiting state.
        """
        now = time.monotonic()
        return ProgressState(
            last_activity=now,
            saw_progress=False,
            saw_please_wait=False,
            idle_timeout=max_wait,
        )

    def _read_progress_chunk(self) -> str | None:
        """Read output chunk while waiting for progress.

        Returns:
            Chunk string, "EOF" on end-of-file, or None on timeout.
        """
        try:
            chunk = self._child.read_nonblocking(size=4096, timeout=0.5)
            return cast(str, chunk)
        except pexpect.TIMEOUT:
            if self._debug:
                print("[DEBUG] No new output while waiting for progress")
            return None
        except pexpect.EOF:
            if self._debug:
                print("[DEBUG] EOF reached while waiting for progress")
            return "EOF"

    def _should_stop_on_timeout(self, state: ProgressState) -> bool:
        """Check if progress waiting should stop due to inactivity.

        Args:
            state: Current progress waiting state.

        Returns:
            True if waiting should stop.
        """
        timeout = (
            state.idle_timeout if state.saw_progress else min(1.0, state.idle_timeout)
        )
        if (time.monotonic() - state.last_activity) >= timeout:
            if self._debug:
                if state.saw_progress:
                    print("[DEBUG] No progress updates within idle timeout; done")
                else:
                    print("[DEBUG] No progress observed within timeout; done")
            return True
        return False

    def _handle_progress_chunk(self, state: ProgressState, chunk: str) -> bool:
        """Handle a progress chunk and return True when waiting is done.

        Args:
            state: Current progress waiting state.
            chunk: Output chunk from the process.

        Returns:
            True when waiting should stop.
        """
        self._transcript.append(chunk)
        if self._debug:
            print(f"[DEBUG] Read chunk while waiting: {chunk!r}")
        state.last_activity = time.monotonic()
        if "please wait" in chunk.lower():
            state.saw_please_wait = True
            if self._debug:
                print("[DEBUG] 'Please wait' seen while waiting")
        has_progress, current_progress = self._has_progress_bar(chunk)
        if not has_progress:
            if state.saw_progress:
                if self._debug:
                    print("[DEBUG] Progress bar no longer detected; done")
                return True
            return False
        state.saw_progress = True
        if self._debug:
            progress_info = (
                f" ({current_progress}%)" if current_progress is not None else ""
            )
            print(f"[DEBUG] Progress bar detected{progress_info}")
        if current_progress is not None and current_progress >= 100.0:
            if self._debug:
                print("[DEBUG] Progress bar completed (>=100%)")
            return True
        return False


class Multiwfn:
    """Multiwfn interface using pexpect for interactive control.

    Args:
        file_path: Path to molden/wfn file from xtb or other QM program.
        run_path: Directory for output files. Defaults to temporary directory
            that is cleaned up when the instance is closed or garbage-collected.
        robust_mode: If True, wait for expect patterns between commands (slower but
            safer).
            If False, send commands without waiting (faster but may desync).
        debug: If True, print debug info including sent commands and expect patterns.
        env_variables: Environment variables to use for the Multiwfn process.
    """

    def __init__(
        self,
        file_path: str | Path,
        run_path: str | Path | None = None,
        robust_mode: bool = True,
        debug: bool = False,
        env_variables: dict[str, str] | None = None,
    ) -> None:
        self._file_path = Path(file_path).resolve()
        if not self._file_path.exists():
            raise FileNotFoundError(f"File not found: {self._file_path}")

        if run_path:
            self._run_path = Path(run_path).resolve()
            self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="morfeus_multiwfn_")
            self._run_path = Path(self._temp_dir.name).resolve()
        self._robust_mode = robust_mode
        self._debug = debug
        self._env_variables = env_variables
        self._results = MultiwfnResults()
        self._settings_ini_path: Path | None = None

        self._results.citations = {
            # Both following papers ***MUST BE CITED IN MAIN TEXT*** if Multiwfn is used:
            # See "How to cite Multiwfn.pdf" in Multiwfn binary package for more
            # information.
            (
                "Tian Lu, Feiwu Chen, J. Comput. Chem., 33, 580 (2012) DOI: "
                "10.1002/jcc.22885"
            ),
            "Tian Lu, J. Chem. Phys., 161, 082503 (2024) DOI: 10.1063/5.0216272",
        }

    def load_settingini(self, file_path: str | Path) -> None:
        """Load custom settings.ini file for all future runs.

        Args:
            file_path: Path to custom settings.ini file.

        Raises:
            FileNotFoundError: If settings.ini does not exist.
        """
        settings_path = Path(file_path).resolve()
        if not settings_path.exists() or settings_path.name != "settings.ini":
            raise FileNotFoundError(f"Settings file not found: {settings_path}")
        self._settings_ini_path = settings_path
        self._copy_settings_ini(self._run_path)

    def load_settingsini(self, file_path: str | Path) -> None:
        """Backward-compatible alias for load_settingini()."""
        self.load_settingini(file_path)

    def close(self) -> None:
        """Clean up temporary working directory, if created."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _copy_settings_ini(self, workdir: str | Path) -> None:
        """Copy configured settings.ini into a working directory."""
        if self._settings_ini_path is None:
            return

        target_dir = Path(workdir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "settings.ini"
        shutil.copy2(self._settings_ini_path, target)

    def _commands_change_isuerfunc(self, function: int = -1) -> list[CommandStep]:
        """Adjust the iuserfunc setting.

        Refer to a complete list (ctrl F: !Below functions can be selected by
        "iuserfunc" parameter):
        https://github.com/foxtran/MultiWFN/blob/master/src/function.f90

        Args:
            function: User-defined function index.

        Returns:
            Command steps to apply iuserfunc.
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

        # Copy a custom settings.ini file from run_path
        self._copy_settings_ini(workdir)
        return workdir

    def _set_env(self) -> dict[str, str]:
        """Set environment variables for Multiwfn execution."""
        return build_execution_env(env_variables=self._env_variables)

    @requires_executable(["Multiwfn"])
    def run_commands(
        self,
        commands: Iterable[str | CommandStep | WaitProgress],
        subdir: str | None = None,
    ) -> MultiwfnRunResult:
        """Run Multiwfn with a custom command sequence.

        Args:
            commands: Sequence of commands (strings or CommandStep objects).
            subdir: Subdirectory within run_path to store results.

        Returns:
            MultiwfnRunResult with stdout, workdir
        """
        workdir = self._setup_workdir(subdir)

        env = self._set_env()
        session = _PexpectSession("Multiwfn", workdir, self._debug, env=env)
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
            "ADCH": (
                "Tian Lu, Feiwu Chen, Atomic dipole moment corrected Hirshfeld "
                "population method, J. Theor. Comput. Chem., 11, 163 (2012)"
            ),
        }

        if model not in models:
            choices = ", ".join(models.keys())
            raise ValueError(
                f"Charge model {model!r} not supported. Choose between {choices}."
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
    def get_surface(self, model: str = "ESP") -> dict[str, Any]:
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
            choices = ", ".join(models.keys())
            raise ValueError(
                f"Surface model {model!r} not supported. Choose between {choices}."
            )
        menu_pattern = models[model]
        menu_cmd = menu_pattern.split(" ")[0]

        self._results.citations.add(
            "Tian Lu, Feiwu Chen, Quantitative analysis of molecular surface based "
            "on improved Marching Tetrahedra algorithm, J. Mol. Graph. Model., 38, "
            "314-323 (2012)"
        )

        commands = [
            CommandStep("12", expect="12 Quantitative analysis of molecular surface"),
            CommandStep("2", expect="2 Select mapped function"),
            CommandStep(menu_cmd, expect=menu_pattern),
            CommandStep("0", expect="0 Start analysis"),
            WaitProgress(),
            CommandStep("11", expect="11 Output surface properties"),
            CommandStep("n", expect="surface facets to locsurf.pqr"),
            CommandStep("-1", expect="-1 Return to upper level"),
            CommandStep("-1", expect="-1 Return to main menu"),
            CommandStep("q", expect="Exit program gracefully"),
        ]

        result = self.run_commands(commands, subdir=model)

        atomic_props = self._parse_atomic_surface_properties(result.stdout)
        global_props = self._parse_global_surface_properties(result.stdout)
        surface_result = {"atomic": atomic_props, "global": global_props}

        if self._results.surfaces is None:
            self._results.surfaces = {}
        self._results.surfaces[model] = surface_result

        return surface_result

    def get_bond_order(self, model: str) -> dict[tuple[int, int], float]:
        """Calculate bond orders.

        Args:
            model: Bond order model to use.

        Returns:
            Bond order matrix as {(i, j): bond_order}.

        Raises:
            ValueError: If given bond order model is not available.
        """
        if self._results.bond_orders is not None and model in self._results.bond_orders:
            return self._results.bond_orders[model]

        bond_orders = {
            "mayer": "1 Mayer bond order analysis",
            "wiberg": "3 Wiberg bond order analysis in Lowdin orthogonalized basis",
            "wilberg": "3 Wiberg bond order analysis in Lowdin orthogonalized basis",
            "mulliken": "4 Mulliken bond order (Mulliken overlap population) analysis",
            "fuzzy": "7 Fuzzy bond order analysis (FBO)",
            "laplacian": "8 Laplacian bond order (LBO)",
        }

        if model not in bond_orders:
            choices = ", ".join(sorted(bond_orders.keys()))
            raise ValueError(
                "Bond order type " f"{model!r} not supported. Choose between {choices}."
            )

        menu_pattern = bond_orders[model]
        menu_cmd = menu_pattern.split(" ")[0]

        if model == "laplacian":
            self._results.citations.add(
                "Tian Lu and Feiwu Chen, Bond Order Analysis Based on the Laplacian of "
                "Electron Density in Fuzzy Overlap Space, J. Phys. Chem. A, 117, "
                "3100-3108 (2013)"
            )
        commands = [
            CommandStep("9", expect="9 Bond order analysis"),
            CommandStep(menu_cmd, expect=menu_pattern),
            WaitProgress(),
            CommandStep("n", expect="outputting bond order matrix"),
            CommandStep("0", expect="0 Return"),
            CommandStep("q", expect="gracefully"),
        ]

        result = self.run_commands(commands, subdir=model)
        bond_order_matrix = self._parse_bond_orders(result.stdout)

        if self._results.bond_orders is None:
            self._results.bond_orders = {}
        self._results.bond_orders[model] = bond_order_matrix

        return bond_order_matrix

    @requires_executable(["Multiwfn"])
    def grid_to_descriptors(
        self, grid_path: str | Path, userfunction: int = -1
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
        grid_path = Path(grid_path)
        if grid_path.suffix.lower() not in {".cub", ".grd"}:
            raise ValueError("Density analysis requires a .cub or .grd file as input.")

        if self._results.grid_descriptors is not None:
            return self._results.grid_descriptors

        commands = self._commands_change_isuerfunc(function=userfunction) + [
            CommandStep("15", expect="15 Fuzzy atomic space analysis"),
            CommandStep("1", expect="1 Perform integration in fuzzy atomic spaces"),
            CommandStep("100", expect="100 User-defined function"),
            WaitProgress(),
            CommandStep("0", expect="0 Return"),
            CommandStep("q", expect="Exit program gracefully"),
        ]

        result = self.run_commands(commands, subdir="grid")
        grid_descriptors = self._parse_atomic_values(result.stdout)

        self._results.grid_descriptors = grid_descriptors

        return grid_descriptors

    def molden_to_grid(
        self,
        descriptor: str,
        grid_quality: str,
        grid_file_name: str | None = None,
    ) -> Path:
        """Generate a grid (.cub) file for a real space function.

        Args:
            descriptor: Descriptor to calculate on grid.
            grid_quality: Grid quality, one of 'low', 'medium', 'high'.
            grid_file_name: Optional name for output cube file.

        Returns:
            Path to the generated cube file.

        Raises:
            ValueError: If the descriptor is not supported.
        """
        if descriptor not in REAL_SPACE_FUNCTIONS:
            choices = ", ".join(REAL_SPACE_FUNCTIONS.keys())
            raise ValueError(
                f"Descriptor {descriptor!r} not supported. Choose between {choices}."
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
            CommandStep("2", expect="2 Export data to a Gaussian-type cube file"),
            WaitProgress(),
            CommandStep("0", expect="0 Return to main menu"),
            CommandStep("q", expect="gracefully"),
        ]
        result = self.run_commands(commands, subdir=descriptor)

        cub_file_path = next(f for f in result.workdir.iterdir() if f.suffix == ".cub")
        cub_file_path = cast(Path, cub_file_path)
        if grid_file_name is not None:  # Rename the file
            new_path = result.workdir / grid_file_name
            new_path = cast(Path, new_path)
            cub_file_path.rename(new_path)
            return new_path

        return cub_file_path

    def get_descriptor(self, descriptor: str) -> dict[int, float]:
        """Calculate atomic densities from integration in fuzzy atomic spaces.

        Args:
            descriptor: Descriptor to calculate on grid.

        Returns:
            Dictionary mapping atom index (1-based) to integrated density value.

        Raises:
            ValueError: If the descriptor is not supported or requires grid data.
        """
        if descriptor not in REAL_SPACE_FUNCTIONS:
            choices = ", ".join(REAL_SPACE_FUNCTIONS.keys())
            raise ValueError(
                f"Descriptor {descriptor!r} not supported. Choose between {choices}."
            )

        if (
            self._results.atomic_descriptors is not None
            and descriptor in self._results.atomic_descriptors
        ):
            return self._results.atomic_descriptors[descriptor]

        menu_pattern = REAL_SPACE_FUNCTIONS[descriptor]
        menu_cmd = menu_pattern.split(" ")[0]

        if menu_cmd in {"13"}:
            raise ValueError(f"Descriptor {descriptor!r} requires grid data!")

        commands = [
            CommandStep("15", expect="15 Fuzzy atomic space analysis"),
            CommandStep("1", expect="1 Perform integration in fuzzy atomic spaces"),
            CommandStep(menu_cmd, expect=menu_pattern),
            WaitProgress(),
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
            "dipole_magnitude_au": (
                r"Magnitude of dipole moment:\s*([-\d.]+)\s+a\.u\."
            ),
            "quadrupole_traceless_magnitude": (
                r"Magnitude of the traceless quadrupole moment tensor:\s*([-\d.]+)"
            ),
            "quadrupole_spherical_magnitude": r"Magnitude: \|Q_2\|=\s*([-\d.]+)",
            "octopole_spherical_magnitude": r"Magnitude: \|Q_3\|=\s*([-\d.]+)",
            "electronic_spatial_extent": (
                r"Electronic spatial extent <r\^2>:\s*([-\d.]+)"
            ),
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
        atomic_values: dict[int, float] = {}

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

    def _parse_bond_orders(self, stdout: str) -> dict[tuple[int, int], float]:
        """Parse bond order matrix from Multiwfn output."""
        bond_orders: dict[tuple[int, int], float] = {}
        atom_pattern = re.compile(r"(\d+)\([A-Z][a-z]?\s*\)")
        float_pattern = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")

        for line in stdout.splitlines():
            atom_matches = list(atom_pattern.finditer(line))
            if len(atom_matches) < 2:
                continue
            atom_i = int(atom_matches[0].group(1))
            atom_j = int(atom_matches[1].group(1))
            float_matches = float_pattern.findall(line)
            if not float_matches:
                continue
            bond_orders[(atom_i, atom_j)] = float(float_matches[-1])

        if bond_orders:
            return bond_orders

        for line in stdout.splitlines():
            if "(" not in line or ")" not in line:
                continue
            tokens = line.replace(":", " ").split()
            atom_tokens = [t for t in tokens if "(" in t and ")" in t]
            if len(atom_tokens) < 2:
                continue
            try:
                atom_i = int(atom_tokens[0].split("(")[0])
                atom_j = int(atom_tokens[1].split("(")[0])
                bond_orders[(atom_i, atom_j)] = float(tokens[-1])
            except (ValueError, IndexError):
                continue

        return bond_orders

    def _parse_atomic_areas(
        self, lines: list[str], start_idx: int
    ) -> tuple[dict[int, dict[str, float]], int]:
        """Parse atomic surface areas section from stdout."""
        atomic_props: dict[int, dict[str, float]] = {}
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
                        "area_total": self._parse_float(parts[1]),
                        "area_positive": self._parse_float(parts[2]),
                        "area_negative": self._parse_float(parts[3]),
                        "min_value": self._parse_float(parts[4]),
                        "max_value": self._parse_float(parts[5]),
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
                                "avg_all": self._parse_float(parts[1]),
                                "avg_positive": self._parse_float(parts[2]),
                                "avg_negative": self._parse_float(parts[3]),
                                "var_all": self._parse_float(parts[4]),
                                "var_positive": self._parse_float(parts[5]),
                                "var_negative": self._parse_float(parts[6]),
                            }
                        )
                except ValueError:
                    continue
        return atomic_props

    def _parse_atomic_charge_separation(
        self, lines: list[str], atomic_props: dict, start_idx: int
    ) -> dict[int, dict[str, float]]:
        """Parse atomic internal charge separation section from stdout."""
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if not line.strip():
                break
            parts = line.split()
            if len(parts) >= 4 and parts[0].isdigit():
                atom_idx = int(parts[0])
                if atom_idx in atomic_props:
                    atomic_props[atom_idx].update(
                        {
                            "pi": self._parse_float(parts[1]),
                            "nu": self._parse_float(parts[2]),
                            "nu_sigma2": self._parse_float(parts[3]),
                        }
                    )
        return atomic_props

    def _parse_atomic_surface_properties(
        self, stdout: str
    ) -> dict[int, dict[str, float]]:
        """Parse atomic surface properties from stdout."""
        lines = stdout.split("\n")
        atomic_props: dict[int, dict[str, float]] = {}

        for i, line in enumerate(lines):
            if "All/Positive/Negative area" in line:
                atomic_props, avg_idx = self._parse_atomic_areas(lines, i)
                self._parse_atomic_averages(lines, atomic_props, avg_idx)
                for j in range(avg_idx + 1, len(lines)):
                    if "Atom#           Pi" in lines[j]:
                        self._parse_atomic_charge_separation(lines, atomic_props, j)
                        break
                break
            if "Atom#      Area(Ang^2)" in line:
                atomic_props = self._parse_atomic_simple_table(lines, i)
                break

        return atomic_props

    def _parse_atomic_simple_table(
        self, lines: list[str], start_idx: int
    ) -> dict[int, dict[str, float]]:
        """Parse atomic surface properties from single-table format."""
        atomic_props: dict[int, dict[str, float]] = {}
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if not line.strip():
                break
            parts = line.split()
            if len(parts) >= 6 and parts[0].isdigit():
                atom_idx = int(parts[0])
                atomic_props[atom_idx] = {
                    "area_total": self._parse_float(parts[1]),
                    "min_value": self._parse_float(parts[2]),
                    "max_value": self._parse_float(parts[3]),
                    "avg_all": self._parse_float(parts[4]),
                    "var_all": self._parse_float(parts[5]),
                }
        return atomic_props

    def _parse_global_surface_properties(self, stdout: str) -> dict[str, float]:
        """Parse global surface properties from stdout."""
        lines = stdout.split("\n")
        props: dict[str, float] = {}
        in_summary = False
        for line in lines:
            if "================= Summary of surface analysis" in line:
                in_summary = True
                continue
            if not in_summary:
                continue
            if "Surface analysis finished!" in line:
                break
            if not line.strip() or ":" not in line:
                continue

            handled = self._parse_summary_minmax(line, props)
            if handled:
                continue
            self._parse_summary_key_value(line, props)
        return props

    def _parse_summary_minmax(self, line: str, props: dict[str, float]) -> bool:
        """Parse a summary line containing both minimal and maximal values."""
        if "Minimal value:" not in line or "Maximal value:" not in line:
            return False
        for label, value in re.findall(
            r"(Minimal value|Maximal value)\s*:\s*([-\d.+Ee]+)", line
        ):
            key = re.sub(r"\s+", "_", label.strip().lower())
            props[key] = self._parse_float(value)
        return True

    def _parse_summary_key_value(self, line: str, props: dict[str, float]) -> None:
        """Parse a single summary key/value line."""
        key_raw, rest = line.split(":", 1)
        key = re.sub(r"[()\[\]=]", "", key_raw.strip().lower())
        key = re.sub(r"\s+", "_", key)
        value = self._parse_first_number(rest)
        if value is not None:
            props[key] = value
        elif re.search(r"\bnan\b", rest, flags=re.IGNORECASE):
            props[key] = float("nan")

    @staticmethod
    def _parse_first_number(text: str) -> float | None:
        match = re.search(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", text)
        if match:
            return float(match.group(0))
        return None

    @staticmethod
    def _parse_float(token: str) -> float:
        if token.lower() == "nan":
            return float("nan")
        return float(token)

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
    """Normalize molden file using molden2aim.

    Args:
        path: Path to working directory containing the molden2aim executable.
        molden_name: Name of the molden file to convert.
        molden2aim_executable_path: Name of the molden2aim executable.

    Returns:
        Name of the converted molden file.
    """
    session = _PexpectSession(
        base_command=f"./{molden2aim_executable_path} -i {molden_name}",
        workdir=Path(path),
        debug=False,
        env=build_execution_env(),
    )
    commands = ["Yes", "No", "No", "No"]
    for item in commands:
        session.send(item)
    session.wait_for_exit()

    return molden_name.split(".molden")[0] + "_new.molden"
