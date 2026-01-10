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
class MultiwfnRunResult:
    """Result from a single Multiwfn run."""

    stdout: str
    workdir: Path
    generated_files: list[Path]


@dataclass
class MultiwfnResults:
    """Cached Multiwfn results."""

    charges: dict[str, dict[int, float]] | None = None
    surfaces: dict[str, dict[str, Any]] | None = None
    densities: dict[int, float] | None = None
    density_cube: Path | None = None
    spin_density_cube: Path | None = None


class _PexpectSession:
    """Pexpect session manager for Multiwfn."""

    def __init__(
        self, workdir: Path, timeout: int, expect_timeout: int, debug: bool
    ) -> None:
        self._child = pexpect.spawn(
            "Multiwfn",
            cwd=str(workdir),
            encoding="utf-8",
            timeout=timeout,
        )
        self._timeout = timeout
        self._expect_timeout = expect_timeout
        self._debug = debug
        self._transcript: list[str] = []
        self._last_command_pos = 0

    def read_available(self) -> None:
        """Read available output without blocking."""
        try:
            chunk = self._child.read_nonblocking(size=4096, timeout=30)
            if chunk:
                self._transcript.append(chunk)
        except (pexpect.TIMEOUT, pexpect.EOF):
            pass

    def send(self, cmd: str) -> None:
        """Send command to process."""
        self._child.sendline(cmd)
        if self._debug:
            print(f"[DEBUG] Sent: {cmd!r}")
        self._last_command_pos = len(self._transcript)

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
            print(f"[WARN] Timeout ({self._expect_timeout}s) waiting for: {pattern!r}")
            print(f"Recent output: {''.join(self._transcript[-2:])[-500:]}")
            return False

    def try_expect(self, pattern: str, timeout: float = 2.0) -> bool:
        """Try to match pattern with short timeout. Silent on timeout."""
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


class Multiwfn:
    """Multiwfn interface using pexpect for interactive control.

    Args:
        file_path: Path to molden/wfn file from xtb or other QM program.
        output_dir: Directory for output files. Defaults to current directory.
        timeout: Global timeout in seconds for process operations.
        expect_timeout: Timeout in seconds for expect pattern matching.
        robust_mode: If True, wait for expect patterns between commands (slower but safer).
                     If False, send commands without waiting (faster but may desync).
        debug: If True, print debug info including sent commands and expect patterns.
    """

    def __init__(
        self,
        file_path: str | Path,
        output_dir: str | Path | None = None,
        timeout: int = 1800,
        expect_timeout: int = 30,
        robust_mode: bool = True,
        debug: bool = False,
    ) -> None:
        self._file_path = Path(file_path).resolve()
        if not self._file_path.exists():
            raise FileNotFoundError(f"File not found: {self._file_path}")

        self._output_dir = Path(output_dir).resolve() if output_dir else Path.cwd()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout
        self._expect_timeout = expect_timeout
        self._robust_mode = robust_mode
        self._debug = debug
        self._results = MultiwfnResults()
        self._custom_settings_path: Path | None = None
        self._citations = {
            # Both following papers ***MUST BE CITED IN MAIN TEXT*** if Multiwfn is used:
            # See "How to cite Multiwfn.pdf" in Multiwfn binary package for more information
            "Tian Lu, Feiwu Chen, J. Comput. Chem., 33, 580 (2012) DOI: 10.1002/jcc.22885",
            "Tian Lu, J. Chem. Phys., 161, 082503 (2024) DOI: 10.1063/5.0216272",
        }

    @property
    def file_path(self) -> Path:
        """Path to the input molden/wfn file."""
        return self._file_path

    @property
    def output_dir(self) -> Path:
        """Directory for output files."""
        return self._output_dir

    def load_settingsini(self, file_path: str | Path) -> None:
        """Load custom settings.ini file."""
        settings_path = Path(file_path).resolve()
        if not settings_path.exists():
            raise FileNotFoundError(f"Settings file not found: {settings_path}")
        self._custom_settings_path = settings_path

    def _setup_settings_ini(self, workdir: Path) -> None:
        """Copy settings.ini to working directory."""
        target = workdir / "settings.ini"
        if target.exists():
            return

        # Use custom settings if provided, otherwise use default from config
        if self._custom_settings_path:
            source = self._custom_settings_path
        else:
            # Use package default
            source = Path(__file__).parent / "config" / "settings.ini"

        if source.exists():
            shutil.copy2(source, target)

    def _get_workdir(self, subdir: str | None) -> Path:
        """Get or create working directory."""
        workdir = self._output_dir / subdir if subdir else self._output_dir
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir

    def _check_pattern_in_output(self, session: _PexpectSession, pattern: str) -> bool:
        """Check if pattern exists in recent output."""
        session.read_available()
        output_since_last_cmd = session.get_output_since_last_command()
        return bool(re.search(pattern, output_since_last_cmd, re.IGNORECASE))

    def _execute_command(
        self, session: _PexpectSession, cmd: str, expect: str | None, use_robust: bool
    ) -> None:
        """Execute command with optional expect."""
        if use_robust and expect:
            session.expect(expect)
        session.send(cmd)
        if not use_robust:
            session.read_available()

    def _process_command(
        self, session: _PexpectSession, item: str | CommandStep, use_robust: bool
    ) -> None:
        """Process command step, handling optional flag."""
        if isinstance(item, str):
            self._execute_command(session, item, None, use_robust)
            return

        if item.optional:
            session.read_available()
            output_since_last = session.get_output_since_last_command()
            if item.expect and not re.search(
                item.expect, output_since_last, re.IGNORECASE
            ):
                return
            if item.cmd:
                session.send(item.cmd)
                if not use_robust:
                    session.read_available()
        else:
            if item.cmd:
                self._execute_command(session, item.cmd, item.expect, use_robust)

    @requires_executable(["Multiwfn"])
    def run_commands(
        self,
        commands: Iterable[str | CommandStep],
        subdir: str | None = None,
        robust_mode: bool | None = None,
    ) -> MultiwfnRunResult:
        """Run Multiwfn with a custom command sequence.

        Args:
            commands: Sequence of commands (strings or CommandStep objects).
            subdir: Subdirectory within output_dir to store results.
            robust_mode: Override instance robust_mode for this call.

        Returns:
            MultiwfnRunResult with stdout, workdir, and list of generated files.
        """

        use_robust = robust_mode if robust_mode is not None else self._robust_mode
        workdir = self._get_workdir(subdir)
        before = {p.resolve() for p in workdir.glob("*")}

        session = _PexpectSession(
            workdir, self._timeout, self._expect_timeout, self._debug
        )
        session.send(str(self._file_path))
        session.read_available()

        for item in commands:
            self._process_command(session, item, use_robust)

        session.wait_for_exit()

        after = {p.resolve() for p in workdir.glob("*")}
        new_files = sorted(after - before)

        return MultiwfnRunResult(
            stdout=session.stdout,
            workdir=workdir,
            generated_files=new_files,
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
            "Hirshfeld": (
                "1",
                "1 Hirshfeld atomic",
                "Theor. Chim. Acta. (Berl), 44, 129-138 (1977)",
            ),
            "VDD": ("2", "2 Voronoi deformation density", "J. Comput. Chem., 25, 189."),
            "Mulliken": ("5", "5 Mulliken atom & basis function", None),
            "ADCH": (
                "11",
                "11 Atomic dipole corrected Hirshfeld",
                "Tian Lu, Feiwu Chen, Atomic dipole moment corrected Hirshfeld population method, J. Theor. Comput. Chem., 11, 163 (2012)",
            ),
            "CM5": ("16", "16 CM5 atomic charge", None),
            "12CM5": ("-16", "-16 Generate 1.2*CM5", None),
        }

        if model not in models:
            raise ValueError(
                f"Charge model {model!r} not supported. Choose between {', '.join(models.keys())}."
            )

        menu_cmd, menu_pattern, citation = models[model]
        if citation:
            self._citations.add(citation)

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
        # Check if already calculated
        if self._results.surfaces is not None and model in self._results.surfaces:
            return self._results.surfaces[model]

        models = {
            "ESP": (
                "1",
                "1 Electrostatic potential",
            ),
            "ALIE": (
                "2",
                "2 Average local ionization energy",
            ),
            "LEA": (
                "4",
                "4 Local electron affinity",
            ),
            "LEAE": (
                "-4",
                "-4 Local electron attachment energy",
            ),
            "Electron": (
                "11",
                "11 Electron density",
            ),
            "Sign": (
                "12",
                "12 Sign",
            ),
        }

        if model not in models:
            raise ValueError(
                f"Surface model {model!r} not supported. Choose between {', '.join(models.keys())}."
            )

        menu_cmd, menu_pattern = models[model]

        self._citations.add(
            "Tian Lu, Feiwu Chen, Quantitative analysis of molecular surface based on improved Marching Tetrahedra algorithm, J. Mol. Graph. Model., 38, 314-323 (2012)"
        )

        commands = [
            CommandStep("12", expect="12 Quantitative analysis of molecular surface"),
            CommandStep("2", expect="2 Select mapped function"),
            CommandStep(menu_cmd, expect=menu_pattern),
            CommandStep("0", expect="0 Start analysis"),
            CommandStep("1", expect="1 Export surface extrema as surfanalysis.txt"),
            CommandStep("11", expect="11 Output surface properties"),
            CommandStep("n", expect="surface facets to locsurf.pqr"),
            CommandStep("-1", expect="-1 Return to upper level"),
            CommandStep("-1", expect=" -1 Return to main menu"),
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

    @requires_executable(["Multiwfn"])
    def get_densities(self) -> dict[int, float]:
        """Calculate atomic densities from integration in fuzzy atomic spaces.

        Note: This requires a cube/grid file as input.

        Returns:
            Dictionary mapping atom index (1-based) to integrated density value.

        Raises:
            ValueError: If input file is not a cube/grid file.
        """
        if self.file_path.suffix.lower() not in {".cub", ".grd"}:
            raise ValueError("Density analysis requires a .cub or .grd file as input.")

        if self._results.densities is not None:
            return self._results.densities

        workdir = self._get_workdir("density")
        self._setup_settings_ini(workdir)

        commands = [
            CommandStep("15", expect="15 Fuzzy atomic space analysis"),
            CommandStep("1", expect="1 Perform integration in fuzzy atomic spaces"),
            CommandStep("100", expect="100 User-defined function"),
            CommandStep("0", expect="0 Return"),
            CommandStep("q", expect="Exit program gracefully"),
        ]

        result = self.run_commands(commands, subdir="density")
        densities = self._parse_atomic_values(result.stdout)

        self._results.densities = densities

        return densities

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
            if "Atom#    All/Positive/Negative area" in line:
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
