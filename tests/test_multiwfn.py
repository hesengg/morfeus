"""Test Multiwfn code."""

import math
from pathlib import Path
import time
from typing import Any

import pexpect
import pytest

import morfeus.config as morfeus_config
from morfeus.multiwfn import (
    _PexpectSession,
    CommandStep,
    Multiwfn,
    MultiwfnRunResult,
    ProgressState,
    WaitProgress,
)
from morfeus.utils import build_execution_env

DATA_DIR = Path(__file__).parent / "data" / "multiwfn"


class _DummyChild:
    """Dummy pexpect child process for unit-testing session behavior."""

    before: str
    after: str
    sent: list[str]
    read_responses: list[str | BaseException]
    expect_responses: list[dict[str, str] | BaseException]
    alive: bool
    closed: bool

    def __init__(self) -> None:
        self.before = ""
        self.after = ""
        self.sent = []
        self.read_responses = []
        self.expect_responses = []
        self.alive = True
        self.closed = False

    def read_nonblocking(self, size: int, timeout: float) -> str:
        """Read queued response or raise timeout."""
        del size, timeout
        if not self.read_responses:
            raise pexpect.TIMEOUT("timeout")
        response = self.read_responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def sendline(self, cmd: str) -> None:
        """Record sent commands."""
        self.sent.append(cmd)

    def expect(self, pattern: Any, timeout: float) -> int:
        """Consume queued expect result or succeed by default."""
        del pattern, timeout
        if self.expect_responses:
            response = self.expect_responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            self.before = response.get("before", "")
            self.after = response.get("after", "")
        return 0

    def isalive(self) -> bool:
        """Return process liveness."""
        return self.alive

    def close(self) -> None:
        """Mark process closed."""
        self.closed = True
        self.alive = False


def _make_run_result(
    tmp_path: Path, subdir: str | None, stdout: str = ""
) -> MultiwfnRunResult:
    """Create a run result with a temporary workdir."""
    workdir = tmp_path / (subdir or "root")
    workdir.mkdir(parents=True, exist_ok=True)
    return MultiwfnRunResult(stdout=stdout, workdir=workdir)


@pytest.fixture
def molden_file() -> Path:
    """Path to molden file used by tests."""
    return DATA_DIR / "example_xtb" / "molden.input"


@pytest.fixture
def cube_file() -> Path:
    """Path to cube file used by tests."""
    return DATA_DIR / "example_xtb_cub" / "spindensity.cub"


@pytest.fixture
def fake_session() -> _PexpectSession:
    """Construct a session object without spawning a real process."""
    session = _PexpectSession.__new__(_PexpectSession)
    session._child = _DummyChild()
    session._timeout = 1
    session._expect_timeout = 1
    session._debug = False
    session._transcript = []
    session._last_command_pos = 0
    return session


@pytest.mark.multiwfn
class TestMultiwfnCharges:
    """Test charge calculation methods."""

    @pytest.fixture
    def mwfn(self):
        """Create Multiwfn instance with molden file."""
        molden_file = DATA_DIR / "example_xtb" / "molden.input"
        return Multiwfn(molden_file, run_path=DATA_DIR / "test_output")

    def test_get_charges_adch(self, mwfn):
        """Test ADCH charge calculation."""
        charges = mwfn.get_charges(model="adch")
        assert isinstance(charges, dict)
        assert len(charges) > 0
        # Check that keys are integers (atom indices)
        assert all(isinstance(k, int) for k in charges.keys())
        # Check that values are floats (charges)
        assert all(isinstance(v, float) for v in charges.values())
        # Check charge conservation (sum should be close to total charge)
        total_charge = sum(charges.values())
        assert abs(total_charge) < 1e-3  # Should sum to ~0 for neutral molecule

    def test_get_charges_hirshfeld(self, mwfn):
        """Test Hirshfeld charge calculation."""
        charges = mwfn.get_charges(model="hirshfeld")
        assert isinstance(charges, dict)
        assert len(charges) > 0
        assert all(isinstance(k, int) for k in charges.keys())
        assert all(isinstance(v, float) for v in charges.values())
        assert abs(sum(charges.values())) < 1e-3
        abs_charge = sum(abs(v) for v in charges.values())
        assert abs_charge < 10.0
        assert abs_charge > 0.0

    def test_get_charges_vdd(self, mwfn):
        """Test VDD charge calculation."""
        charges = mwfn.get_charges(model="vdd")
        assert isinstance(charges, dict)
        assert len(charges) > 0
        assert abs(sum(charges.values())) < 1e-3
        abs_charge = sum(abs(v) for v in charges.values())
        assert abs_charge < 10.0
        assert abs_charge > 0.0

    def test_get_charges_mulliken(self, mwfn):
        """Test Mulliken charge calculation."""
        charges = mwfn.get_charges(model="mulliken")
        assert isinstance(charges, dict)
        assert len(charges) > 0
        assert abs(sum(charges.values())) < 1e-3
        abs_charge = sum(abs(v) for v in charges.values())
        assert abs_charge < 10.0
        assert abs_charge > 0.0

    def test_get_charges_cm5(self, mwfn):
        """Test CM5 charge calculation."""
        charges = mwfn.get_charges(model="cm5")
        assert isinstance(charges, dict)
        assert len(charges) > 0
        assert abs(sum(charges.values())) < 1e-3
        abs_charge = sum(abs(v) for v in charges.values())
        assert abs_charge < 10.0
        assert abs_charge > 0.0

    def test_get_charges_12cm5(self, mwfn):
        """Test 1.2*CM5 charge calculation."""
        charges = mwfn.get_charges(model="12cm5")
        assert isinstance(charges, dict)
        assert len(charges) > 0
        assert abs(sum(charges.values())) < 1e-3
        abs_charge = sum(abs(v) for v in charges.values())
        assert abs_charge < 10.0
        assert abs_charge > 0.0

    def test_get_charges_invalid_model(self, mwfn):
        """Test that invalid charge model raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            mwfn.get_charges(model="InvalidModel")

    def test_get_charges_caching(self, mwfn):
        """Test that charges are cached after first calculation."""
        charges1 = mwfn.get_charges(model="adch")
        charges2 = mwfn.get_charges(model="adch")
        # Should return the same object (cached)
        assert charges1 is charges2


@pytest.mark.multiwfn
class TestMultiwfnSurface:
    """Test surface analysis methods."""

    @pytest.fixture
    def mwfn(self):
        """Create Multiwfn instance with molden file."""
        molden_file = DATA_DIR / "example_xtb" / "molden.input"
        return Multiwfn(molden_file, run_path=DATA_DIR / "test_output")

    def test_get_surface_esp(self, mwfn):
        """Test ESP surface calculation."""
        surface = mwfn.get_surface(model="esp")
        assert isinstance(surface, dict)
        assert "atomic" in surface
        assert "global" in surface

        # Check atomic properties
        atomic = surface["atomic"]
        assert len(atomic) > 0
        for atom_idx, props in atomic.items():
            assert isinstance(atom_idx, int)
            assert isinstance(props, dict)
            # Check expected properties
            expected_keys = [
                "area_total",
                "area_positive",
                "area_negative",
                "min_value",
                "max_value",
                "avg_all",
                "avg_positive",
                "avg_negative",
                "var_all",
                "var_positive",
                "var_negative",
                "pi",
                "nu",
                "nu_sigma2",
            ]
            # Some properties might be missing for certain atoms
            assert any(key in props for key in expected_keys)

        # Check global properties
        global_props = surface["global"]
        assert isinstance(global_props, dict)
        if global_props:  # May be empty depending on output
            expected_global_keys = [
                "volume",
                "overall_surface_area",
                "positive_surface_area",
                "negative_surface_area",
                "minimal_value",
                "maximal_value",
            ]
            assert any(key in global_props for key in expected_global_keys)

    def test_get_surface_alie(self, mwfn):
        """Test ALIE surface calculation."""
        surface = mwfn.get_surface(model="alie")
        assert isinstance(surface, dict)
        assert "atomic" in surface
        assert "global" in surface

    def test_get_surface_lea(self, mwfn):
        """Test LEA surface calculation."""
        surface = mwfn.get_surface(model="lea")
        assert isinstance(surface, dict)
        assert "atomic" in surface
        assert "global" in surface

    def test_get_surface_leae(self, mwfn):
        """Test LEAE surface calculation."""
        surface = mwfn.get_surface(model="leae")
        assert isinstance(surface, dict)
        assert "atomic" in surface
        assert "global" in surface

    def test_get_surface_electron(self, mwfn):
        """Test Electron Density surface calculation."""
        surface = mwfn.get_surface(model="electron")
        assert isinstance(surface, dict)
        assert "atomic" in surface
        assert "global" in surface

    def test_get_surface_sign(self, mwfn):
        """Test Sign(lambda2)*rho surface calculation."""
        surface = mwfn.get_surface(model="sign")
        assert isinstance(surface, dict)
        assert "atomic" in surface
        assert "global" in surface

    def test_get_surface_invalid_model(self, mwfn):
        """Test that invalid surface model raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            mwfn.get_surface(model="InvalidModel")

    def test_get_surface_caching(self, mwfn):
        """Test that surface results are cached."""
        surface1 = mwfn.get_surface(model="esp")
        surface2 = mwfn.get_surface(model="esp")
        assert surface1 is surface2


@pytest.mark.multiwfn
class TestMultiwfnDensities:
    """Test density calculation from cube files."""

    cube_file = DATA_DIR / "example_xtb_cub" / "spindensity.cub"

    @pytest.fixture
    def mwfn_cube(self):
        """Create Multiwfn instance with cube file."""
        return Multiwfn(self.cube_file, run_path=DATA_DIR / "test_output_cube")

    def test_get_densities(self, mwfn_cube):
        """Test density calculation from cube file."""
        densities = mwfn_cube.grid_to_descriptors(self.cube_file)
        assert isinstance(densities, dict)
        assert len(densities) > 0
        # Check that keys are integers (atom indices)
        assert all(isinstance(k, int) for k in densities.keys())
        # Check that values are floats (density values)
        assert all(isinstance(v, float) for v in densities.values())

    def test_get_densities_wrong_file_type(self):
        """Test that non-cube file raises ValueError."""
        molden_file = DATA_DIR / "example_xtb" / "molden.input"
        mwfn = Multiwfn(molden_file)
        with pytest.raises(ValueError, match="requires a .cub or .grd file"):
            mwfn.grid_to_descriptors(molden_file)

    def test_get_densities_caching(self, mwfn_cube):
        """Test that densities are cached."""
        densities1 = mwfn_cube.grid_to_descriptors(self.cube_file)
        densities2 = mwfn_cube.grid_to_descriptors(self.cube_file)
        assert densities1 is densities2


@pytest.mark.multiwfn
class TestMultiwfnParsing:
    """Test parsing methods."""

    def test_parse_bond_orders(self):
        """Test parsing bond order matrix from stdout."""
        molden_file = DATA_DIR / "example_xtb" / "molden.input"
        mwfn = Multiwfn(molden_file)
        stdout = "\n".join(
            [
                " 1(C ) 2(O )    0.1234",
                " 2(O ) 3(H )    0.5678",
            ]
        )
        result = mwfn._parse_bond_orders(stdout)
        assert result[(1, 2)] == pytest.approx(0.1234)
        assert result[(2, 3)] == pytest.approx(0.5678)

    def test_parse_bond_orders_fallback(self):
        """Test parsing bond orders via fallback format."""
        molden_file = DATA_DIR / "example_xtb" / "molden.input"
        mwfn = Multiwfn(molden_file)
        stdout = "\n".join(
            [
                "1(c) 2(o) 0.42",
                "2(o) 3(h) 0.77",
            ]
        )
        result = mwfn._parse_bond_orders(stdout)
        assert result[(1, 2)] == pytest.approx(0.42)
        assert result[(2, 3)] == pytest.approx(0.77)

    def test_parse_surfanalysis(self):
        """Test parsing of surfanalysis.txt file."""
        # Create a test surfanalysis.txt file
        test_content = """Number of surface minima:    12
   #       a.u.         eV      kcal/mol           X/Y/Z coordinate(Angstrom)
     1  0.48025989   13.068536  301.367885      -2.821749  -1.268933  -1.450532
     2  0.48025779   13.068479  301.366565      -2.815790   1.167369   1.616685

 Number of surface maxima:    13
   #       a.u.         eV      kcal/mol           X/Y/Z coordinate(Angstrom)
     1  0.51741287   14.079520  324.681749      -3.664456   1.168678  -0.808267
     2  0.49694130   13.522460  311.835634      -2.902645  -0.506156   0.468219
"""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(test_content)
            temp_path = Path(f.name)

        try:
            molden_file = DATA_DIR / "example_xtb" / "molden.input"
            mwfn = Multiwfn(molden_file)
            result = mwfn.parse_surfanalysis(temp_path)

            assert "statistics" in result
            assert "raw" in result
            stats = result["statistics"]
            assert stats["num_minima"] == 12
            assert stats["num_maxima"] == 13
            assert "extrema_min" in stats
            assert "extrema_max" in stats
            assert "extrema_average" in stats
        finally:
            temp_path.unlink()

    def test_parse_surfanalysis_nonexistent(self):
        """Test parsing of nonexistent file returns empty dict."""
        molden_file = DATA_DIR / "example_xtb" / "molden.input"
        mwfn = Multiwfn(molden_file)
        result = mwfn.parse_surfanalysis(Path("nonexistent.txt"))
        assert result["statistics"] == {}
        assert result["raw"] == ""


@pytest.mark.multiwfn
class TestMultiwfnInitialization:
    """Test Multiwfn initialization."""

    def test_init_valid_file(self):
        """Test initialization with valid molden file."""
        molden_file = DATA_DIR / "example_xtb" / "molden.input"
        mwfn = Multiwfn(molden_file, run_path=DATA_DIR / "test_output")
        assert mwfn._file_path == molden_file.resolve()
        assert mwfn._run_path.exists()

    def test_init_nonexistent_file(self):
        """Test initialization with nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            Multiwfn("nonexistent_file.input")

    def test_init_custom_output_dir(self):
        """Test initialization with custom output directory."""
        molden_file = DATA_DIR / "example_xtb" / "molden.input"
        run_path = DATA_DIR / "custom_output"
        mwfn = Multiwfn(molden_file, run_path=run_path)
        assert mwfn._run_path == run_path.resolve()

    def test_context_manager(self):
        """Test context manager support."""
        molden_file = DATA_DIR / "example_xtb" / "molden.input"
        with Multiwfn(molden_file) as mwfn:
            assert mwfn._file_path.exists()

    def test_init_cube_does_not_detect_spin(self):
        """Test that spin detection is not attempted for cube inputs."""
        cube_file = DATA_DIR / "example_xtb_cub" / "spindensity.cub"
        mwfn = Multiwfn(cube_file, run_path=DATA_DIR / "test_output_cube")
        assert mwfn._has_spin is None


@pytest.mark.multiwfn
class TestMultiwfnResults:
    """Test results caching and access."""

    def test_results_caching_multiple_models(self):
        """Test that multiple charge models are cached separately."""
        molden_file = DATA_DIR / "example_xtb" / "molden.input"
        mwfn = Multiwfn(molden_file, run_path=DATA_DIR / "test_output")

        charges_adch = mwfn.get_charges(model="adch")
        charges_hirsh = mwfn.get_charges(model="hirshfeld")

        # Should be different results
        assert charges_adch is not charges_hirsh

        # Both should be cached
        assert mwfn._results.charges is not None
        assert "adch" in mwfn._results.charges
        assert "hirshfeld" in mwfn._results.charges

    def test_results_multiple_surfaces(self):
        """Test that multiple surface models are cached separately."""
        molden_file = DATA_DIR / "example_xtb" / "molden.input"
        mwfn = Multiwfn(molden_file, run_path=DATA_DIR / "test_output")

        surface_esp = mwfn.get_surface(model="esp")
        surface_alie = mwfn.get_surface(model="alie")

        # Should be different results
        assert surface_esp is not surface_alie

        # Both should be cached
        assert mwfn._results.surfaces is not None
        assert "esp" in mwfn._results.surfaces
        assert "alie" in mwfn._results.surfaces

    def test_get_citations(self):
        """Test citation collection helper."""
        molden_file = DATA_DIR / "example_xtb" / "molden.input"
        mwfn = Multiwfn(molden_file, run_path=DATA_DIR / "test_output")
        citations = mwfn.get_citations()
        assert isinstance(citations, list)
        assert any(
            "J. Comput. Chem., 33, 580 (2012)" in citation for citation in citations
        )

    def test_list_options(self):
        """Test listing supported analysis options."""
        molden_file = DATA_DIR / "example_xtb" / "molden.input"
        mwfn = Multiwfn(molden_file, run_path=DATA_DIR / "test_output")
        options = mwfn.list_options()
        assert "charges" in options
        assert "surface" in options
        assert "bond_order" in options
        assert "grid_quality" in options
        assert "descriptors" in options
        assert "descriptors_fast" in options
        assert "adch" in options["charges"]
        assert "rho" in options["descriptors_fast"]


@pytest.mark.multiwfn
class TestMultiwfnBatchBehavior:
    """Test batch API behavior on invalid entries."""

    @pytest.fixture
    def mwfn(self):
        """Create Multiwfn instance with molden file."""
        molden_file = DATA_DIR / "example_xtb" / "molden.input"
        return Multiwfn(molden_file, run_path=DATA_DIR / "test_output")

    def test_get_descriptors_raises_on_invalid(self, mwfn):
        """Test that get_descriptors raises on invalid descriptor."""
        with pytest.raises(ValueError, match="not supported"):
            mwfn.get_descriptors(["invalid_descriptor"])

    def test_get_grids_raises_on_invalid(self, mwfn):
        """Test that get_grids raises on invalid descriptor."""
        with pytest.raises(ValueError, match="not supported"):
            mwfn.get_grids(["invalid_descriptor"], grid_quality="low")


@pytest.mark.multiwfn
class TestMultiwfnBondOrders:
    """Test bond order calculation methods."""

    @pytest.fixture
    def mwfn(self):
        """Create Multiwfn instance with molden file."""
        molden_file = DATA_DIR / "example_xtb" / "molden.input"
        return Multiwfn(molden_file, run_path=DATA_DIR / "test_output")

    def test_get_bond_order_invalid_model(self, mwfn):
        """Test that invalid bond order model raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            mwfn.get_bond_order(model="InvalidModel")

    def test_get_bond_order_wilberg_rejected(self, mwfn):
        """Test that misspelled model name 'wilberg' is rejected."""
        with pytest.raises(ValueError, match="not supported"):
            mwfn.get_bond_order(model="wilberg")


@pytest.mark.multiwfn
class TestMultiwfnSettingsAndHelpers:
    """Test helper/configuration methods and utility behavior."""

    def test_build_execution_env_config_fallbacks(self, monkeypatch):
        """Test environment construction fallback for missing config attributes."""
        monkeypatch.delattr(morfeus_config, "OMP_NUM_THREADS", raising=False)
        monkeypatch.delattr(morfeus_config, "OMP_STACKSIZE", raising=False)
        monkeypatch.delattr(morfeus_config, "OMP_MAX_ACTIVE_LEVELS", raising=False)

        env = build_execution_env()

        assert "OMP_NUM_THREADS" in env
        assert "MKL_NUM_THREADS" in env
        assert "OMP_STACKSIZE" in env
        assert "OMP_MAX_ACTIVE_LEVELS" in env

        omp_threads, _, omp_levels = env["OMP_NUM_THREADS"].partition(",")
        assert omp_threads
        assert omp_threads.isdigit()
        assert omp_levels
        assert omp_levels.isdigit()
        assert env["MKL_NUM_THREADS"] == omp_threads
        assert env["OMP_STACKSIZE"]
        assert env["OMP_MAX_ACTIVE_LEVELS"].isdigit()

    def test_load_settingini_and_alias(self, molden_file, tmp_path):
        """Test loading settings.ini via both method names."""
        settings_file = DATA_DIR / "test_output" / "ESP" / "settings.ini"
        mwfn = Multiwfn(molden_file, run_path=tmp_path, has_spin=False)
        mwfn.load_settingini(settings_file)
        assert mwfn._settings_ini_path == settings_file.resolve()
        assert (tmp_path / "settings.ini").exists()

        alias_path = tmp_path / "alias"
        mwfn_alias = Multiwfn(molden_file, run_path=alias_path, has_spin=False)
        mwfn_alias.load_settingsini(settings_file)
        assert (alias_path / "settings.ini").exists()

    def test_load_settingini_invalid_file(self, molden_file, tmp_path):
        """Test that non-settings filename is rejected."""
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("dummy")
        mwfn = Multiwfn(molden_file, run_path=tmp_path / "run", has_spin=False)
        with pytest.raises(FileNotFoundError):
            mwfn.load_settingini(invalid_file)

    def test_close_cleans_temp_directory(self, molden_file):
        """Test that close removes temporary workdir."""
        mwfn = Multiwfn(molden_file, has_spin=False)
        run_path = mwfn._run_path
        assert run_path.exists()
        mwfn.close()
        assert not run_path.exists()

    def test_option_helpers(self, molden_file, tmp_path):
        """Test option normalization and option listing."""
        mwfn = Multiwfn(molden_file, run_path=tmp_path, has_spin=False)
        assert mwfn._normalize_option("  ADCH ") == "adch"
        options = mwfn.list_options()
        assert set(options) == {
            "charges",
            "surface",
            "bond_order",
            "grid_quality",
            "descriptors",
            "descriptors_fast",
        }
        for key in ("charges", "surface", "bond_order", "grid_quality"):
            model_names = options[key]
            assert all(name == name.lower() for name in model_names)
        assert "rho" in options["descriptors"]
        assert set(options["descriptors_fast"]).issubset(set(options["descriptors"]))
        assert "electron_esp_Vele" not in options["descriptors_fast"]

    def test_citation_sorting(self, molden_file, tmp_path):
        """Test that get_citations returns sorted values."""
        mwfn = Multiwfn(molden_file, run_path=tmp_path, has_spin=False)
        citations_before = set(mwfn.get_citations())
        mwfn._results.citations.update({"z", "a", "m"})
        assert mwfn.get_citations() == sorted(citations_before | {"z", "a", "m"})

    def test_internal_command_builders(self, molden_file, tmp_path):
        """Test helper command builders and parser helpers."""
        mwfn = Multiwfn(molden_file, run_path=tmp_path, has_spin=False)
        assert mwfn._parse_user_function(7) == (7, None)
        assert mwfn._parse_user_function((8, "citation")) == (8, "citation")

        commands = mwfn._build_fuzzy_integration_commands(
            "1",
            "1 Electron density",
            prefix_commands=[CommandStep("99", expect="foo")],
        )
        assert isinstance(commands[0], CommandStep)
        assert any(isinstance(item, WaitProgress) for item in commands)

        citations_before = set(mwfn._results.citations)
        vector_functions = mwfn._parse_vector_functions("magnetic_dipole_moment")
        assert vector_functions == (75, 76, 77)
        assert len(mwfn._results.citations) == len(citations_before) + 1


@pytest.mark.multiwfn
class TestMultiwfnDescriptorBranches:
    """Test descriptor selection and spin-dependent branches."""

    def test_detect_spin_state(self, molden_file, tmp_path, monkeypatch):
        """Test multiplicity parser for open- and closed-shell outputs."""
        mwfn = Multiwfn(molden_file, run_path=tmp_path, has_spin=False)

        monkeypatch.setattr(
            mwfn,
            "run_commands",
            lambda commands, subdir=None: _make_run_result(
                tmp_path, subdir, "Expected multiplicity: 1"
            ),
        )
        assert mwfn._detect_spin_state() is False

        monkeypatch.setattr(
            mwfn,
            "run_commands",
            lambda commands, subdir=None: _make_run_result(
                tmp_path, subdir, "Expected multiplicity: 2"
            ),
        )
        assert mwfn._detect_spin_state() is True

    def test_detect_spin_state_missing_pattern(
        self, molden_file, tmp_path, monkeypatch
    ):
        """Test multiplicity parser failure path."""
        mwfn = Multiwfn(molden_file, run_path=tmp_path, has_spin=False)
        monkeypatch.setattr(
            mwfn,
            "run_commands",
            lambda commands, subdir=None: _make_run_result(
                tmp_path, subdir, "No match"
            ),
        )
        with pytest.raises(RuntimeError, match="Could not detect multiplicity"):
            mwfn._detect_spin_state()

    def test_check_function_selection_branches(self, molden_file, tmp_path):
        """Test function filtering for unknown/open/closed spin states."""
        mwfn = Multiwfn(molden_file, run_path=tmp_path, has_spin=False)

        mwfn._has_spin = None
        with pytest.raises(ValueError, match="requires known spin state"):
            mwfn._check_function_selection(24, "dft_linear_response_kernel")
        mwfn._check_function_selection(11, "electron_energy")

        mwfn._has_spin = True
        with pytest.raises(ValueError, match="open-shell systems"):
            mwfn._check_function_selection(24, "dft_linear_response_kernel")
        mwfn._check_function_selection(1, "alpha_density")

        mwfn._has_spin = False
        with pytest.raises(ValueError, match="closed-shell systems"):
            mwfn._check_function_selection(1, "alpha_density")
        mwfn._check_function_selection(24, "dft_linear_response_kernel")

    def test_get_descriptor_function_paths(self, molden_file, tmp_path):
        """Test descriptor function resolution for real/user/vector cases."""
        mwfn = Multiwfn(molden_file, run_path=tmp_path, has_spin=False)
        menu_cmd, menu_pattern, commands = mwfn._get_descriptor_function("rho")
        assert menu_cmd == "1"
        assert "Electron density" in menu_pattern
        assert commands == []

        menu_cmd, menu_pattern, commands = mwfn._get_descriptor_function("fplus")
        assert menu_cmd == "100"
        assert menu_pattern == "100 User-defined function"
        assert len(commands) > 0

        with pytest.raises(ValueError, match="vector-valued"):
            mwfn._get_descriptor_function("coordinates")

    def test_get_descriptor_and_get_descriptors(
        self, molden_file, tmp_path, monkeypatch
    ):
        """Test scalar descriptor methods with a stubbed integration backend."""
        mwfn = Multiwfn(molden_file, run_path=tmp_path, has_spin=False)

        def fake_run(menu_cmd, menu_pattern, commands, descriptor):
            del menu_cmd, menu_pattern, commands
            return {1: float(len(descriptor))}

        monkeypatch.setattr(mwfn, "_run_fuzzy_integration", fake_run)
        result = mwfn.get_descriptor("rho")
        assert result[1] == 3.0
        assert mwfn.get_descriptor("rho") is result

        many = mwfn.get_descriptors(["rho", "fplus"])
        assert set(many) == {"rho", "fplus"}

    def test_get_vector(self, molden_file, tmp_path, monkeypatch):
        """Test vector descriptor assembly and caching."""
        mwfn = Multiwfn(molden_file, run_path=tmp_path, has_spin=False)
        calls = []

        def fake_run(menu_cmd, menu_pattern, commands, descriptor):
            del menu_cmd, menu_pattern, commands
            calls.append(descriptor)
            lookup = {
                "coordinates_x": {1: 1.0},
                "coordinates_y": {1: 2.0},
                "coordinates_z": {1: 3.0},
            }
            return lookup[descriptor]

        monkeypatch.setattr(mwfn, "_run_fuzzy_integration", fake_run)
        vector = mwfn.get_vector("coordinates")
        assert vector[1] == (1.0, 2.0, 3.0)
        assert mwfn.get_vector("coordinates") is vector
        assert calls == ["coordinates_x", "coordinates_y", "coordinates_z"]

        with pytest.raises(ValueError, match="not supported"):
            mwfn.get_vector("invalid")


@pytest.mark.multiwfn
class TestMultiwfnRunStubs:
    """Test methods that invoke Multiwfn by stubbing run_commands."""

    @pytest.fixture
    def mwfn(self, molden_file, tmp_path):
        """Create a controllable Multiwfn instance."""
        return Multiwfn(molden_file, run_path=tmp_path / "run", has_spin=False)

    @pytest.mark.parametrize(
        "model",
        ["hirshfeld", "vdd", "mulliken", "adch", "cm5", "12cm5"],
    )
    def test_get_charges_all_models(self, mwfn, model, monkeypatch, tmp_path):
        """Test all charge models through a stubbed run."""

        def fake_run(commands, subdir=None):
            del commands
            result = _make_run_result(tmp_path, subdir)
            (result.workdir / "molden.chg").write_text("a b c d 0.1\n" "a b c d -0.1\n")
            return result

        monkeypatch.setattr(mwfn, "run_commands", fake_run)
        charges = mwfn.get_charges(model=model.upper())
        assert charges == {1: 0.1, 2: -0.1}

    @pytest.mark.parametrize(
        "model",
        ["mayer", "wiberg", "mulliken", "fuzzy", "laplacian"],
    )
    def test_get_bond_order_all_models(self, mwfn, model, monkeypatch, tmp_path):
        """Test all bond-order models through a stubbed run."""

        def fake_run(commands, subdir=None):
            del commands
            stdout = " 1(C ) 2(O )    0.1234\n"
            return _make_run_result(tmp_path, subdir, stdout)

        monkeypatch.setattr(mwfn, "run_commands", fake_run)
        matrix = mwfn.get_bond_order(model=model.upper())
        assert matrix[(1, 2)] == pytest.approx(0.1234)

    @pytest.mark.parametrize(
        "model",
        ["esp", "alie", "lea", "leae", "electron", "sign"],
    )
    def test_get_surface_all_models(self, mwfn, model, monkeypatch, tmp_path):
        """Test all surface models through a stubbed run."""

        def fake_run(commands, subdir=None):
            del commands
            stdout = (
                "All/Positive/Negative area\n"
                "1 10.0 6.0 4.0 -0.2 0.3\n"
                "Atom#   All/Positive/Negative average\n"
                "1 0.1 0.2 0.3 0.4 0.5 0.6\n"
                "Atom#           Pi\n"
                "1 1.1 2.2 3.3\n"
                "================= Summary of surface analysis\n"
                "Minimal value: -1.0 Maximal value: 2.0\n"
                "Overall surface area: 25.5\n"
                "Surface analysis finished!\n"
            )
            return _make_run_result(tmp_path, subdir, stdout)

        monkeypatch.setattr(mwfn, "run_commands", fake_run)
        surface = mwfn.get_surface(model=model.upper())
        assert "atomic" in surface
        assert "global" in surface
        assert surface["atomic"][1]["area_total"] == pytest.approx(10.0)

    def test_get_grid_and_get_grids_all_qualities(self, mwfn, monkeypatch, tmp_path):
        """Test grid generation for multiple quality settings."""
        counter = {"n": 0}

        def fake_run(commands, subdir=None):
            del commands
            result = _make_run_result(tmp_path, subdir, "")
            counter["n"] += 1
            (result.workdir / f"generated_{counter['n']}.cub").write_text("cube")
            return result

        monkeypatch.setattr(mwfn, "run_commands", fake_run)

        for quality in ("low", "medium", "high", "LOW"):
            grid_file = mwfn.get_grid("rho", quality)
            assert grid_file.suffix == ".cub"
            assert grid_file.exists()

        renamed = mwfn.get_grid("rho", "low", grid_file_name="renamed.cub")
        assert renamed.name == "renamed.cub"
        assert renamed.exists()

        many = mwfn.get_grids(["rho", "rho_lapl"], "medium")
        assert set(many) == {"rho", "rho_lapl"}

        with pytest.raises(ValueError, match="Grid quality"):
            mwfn.get_grid("rho", "ultra")

    def test_grid_to_descriptors_per_file_cache(self, cube_file, monkeypatch, tmp_path):
        """Test descriptor caching is keyed by active grid file path."""
        second_cube = tmp_path / "second.cub"
        second_cube.write_text("cube")
        mwfn = Multiwfn(cube_file, run_path=tmp_path / "grid", has_spin=None)
        calls = []

        def fake_run(commands, subdir=None):
            del commands, subdir
            calls.append(str(mwfn._file_path))
            value = "1.0" if mwfn._file_path == cube_file.resolve() else "2.0"
            stdout = (
                "Atomic space            Value\n"
                f"    1(C )            {value}\n"
                "Summing up\n"
            )
            return _make_run_result(tmp_path, "grid", stdout)

        monkeypatch.setattr(mwfn, "run_commands", fake_run)
        first = mwfn.grid_to_descriptors(cube_file)
        again = mwfn.grid_to_descriptors(cube_file)
        second = mwfn.grid_to_descriptors(second_cube)
        assert first is again
        assert first[1] == pytest.approx(1.0)
        assert second[1] == pytest.approx(2.0)
        assert len(calls) == 2

    def test_get_fukui_and_superdelocalizabilities(self, mwfn, monkeypatch, tmp_path):
        """Test conceptual-DFT methods for success and spin errors."""
        mwfn._has_spin = None
        with pytest.raises(ValueError, match="known spin state"):
            mwfn.get_fukui()
        with pytest.raises(ValueError, match="known spin state"):
            mwfn.get_superdelocalizabilities()

        mwfn._has_spin = True
        with pytest.raises(ValueError, match="closed-shell"):
            mwfn.get_fukui()
        with pytest.raises(ValueError, match="closed-shell"):
            mwfn.get_superdelocalizabilities()

        def fake_run(commands, subdir=None):
            del commands
            if subdir == "fukui":
                stdout = "Atom index      OW f+\n" "   1(C    0.10 0.20 0.30 0.40\n"
            else:
                stdout = (
                    "Atom      D_N      D_E\n"
                    "   1(C    1.10 2.20 3.30 4.40\n"
                    "Sum of\n"
                )
            return _make_run_result(tmp_path, subdir, stdout)

        mwfn._has_spin = False
        monkeypatch.setattr(mwfn, "run_commands", fake_run)
        fukui = mwfn.get_fukui()
        superdeloc = mwfn.get_superdelocalizabilities()
        assert fukui["f_plus"][1] == pytest.approx(0.10)
        assert superdeloc["d_n"][1] == pytest.approx(1.10)

    def test_get_electric_moments(self, mwfn, monkeypatch, tmp_path):
        """Test electric-moment extraction from stdout."""

        def fake_run(commands, subdir=None):
            del commands
            stdout = (
                "Magnitude of dipole moment: 1.23 a.u.\n"
                "Magnitude of the traceless quadrupole moment tensor: 2.34\n"
                "Magnitude: |Q_2|= 3.45\n"
                "Magnitude: |Q_3|= 4.56\n"
                "Electronic spatial extent <r^2>: 5.67\n"
            )
            return _make_run_result(tmp_path, subdir, stdout)

        monkeypatch.setattr(mwfn, "run_commands", fake_run)
        moments = mwfn.get_electric_moments()
        assert moments["dipole_magnitude_au"] == pytest.approx(1.23)
        assert moments["quadrupole_traceless_magnitude"] == pytest.approx(2.34)
        assert moments["quadrupole_spherical_magnitude"] == pytest.approx(3.45)
        assert moments["octopole_spherical_magnitude"] == pytest.approx(4.56)
        assert moments["electronic_spatial_extent"] == pytest.approx(5.67)


@pytest.mark.multiwfn
class TestMultiwfnParserHelpers:
    """Test parser helper methods with synthetic output data."""

    @pytest.fixture
    def mwfn(self, molden_file, tmp_path):
        """Create parser-test Multiwfn instance."""
        return Multiwfn(molden_file, run_path=tmp_path / "parser", has_spin=False)

    def test_parse_atomic_table_helpers(self, mwfn):
        """Test atomic table parsing and wrapper methods."""
        fukui_stdout = "Atom index      OW f+\n" "   1(C    0.10 0.20 0.30 0.40\n"
        parsed_fukui = mwfn._parse_fukui(fukui_stdout)
        assert parsed_fukui["f_plus"][1] == pytest.approx(0.10)
        assert parsed_fukui["dd"][1] == pytest.approx(0.40)

        super_stdout = (
            "Atom      D_N      D_E\n"
            "   1(C    1.10 2.20 3.30 4.40\n"
            "Sum of D_N and D_E\n"
        )
        parsed_super = mwfn._parse_superdelocalizabilities(super_stdout)
        assert parsed_super["d_n"][1] == pytest.approx(1.10)
        assert parsed_super["d_e_0"][1] == pytest.approx(4.40)

    def test_parse_atomic_values_and_chg(self, mwfn, tmp_path):
        """Test scalar atomic-value parser and chg-file parser."""
        stdout = (
            "Atomic space           Value\n"
            "    1(C )            0.00607663\n"
            "Summing up\n"
        )
        values = mwfn._parse_atomic_values(stdout)
        assert values == {1: pytest.approx(0.00607663)}
        assert mwfn._parse_atomic_values("no matching table") == {}

        chg_file = tmp_path / "test.chg"
        chg_file.write_text("x y z q 0.123\n" "broken line\n" "x y z q -0.456\n")
        charges = mwfn._parse_chg_file(chg_file)
        assert charges == {1: pytest.approx(0.123), 3: pytest.approx(-0.456)}

    def test_parse_surface_tables(self, mwfn):
        """Test detailed and simple atomic surface property table parsers."""
        detailed = (
            "All/Positive/Negative area\n"
            "1 10.0 6.0 4.0 -0.2 0.3\n"
            "Atom#   All/Positive/Negative average\n"
            "1 0.1 0.2 0.3 0.4 0.5 0.6\n"
            "Atom#           Pi\n"
            "1 1.1 2.2 3.3\n"
        )
        parsed_detailed = mwfn._parse_atomic_surface_properties(detailed)
        assert parsed_detailed[1]["avg_all"] == pytest.approx(0.1)
        assert parsed_detailed[1]["pi"] == pytest.approx(1.1)

        simple = "Atom#      Area(Ang^2)\n" "1 5.0 -0.2 0.2 0.0 0.1\n"
        parsed_simple = mwfn._parse_atomic_surface_properties(simple)
        assert parsed_simple[1]["area_total"] == pytest.approx(5.0)
        assert parsed_simple[1]["var_all"] == pytest.approx(0.1)

    def test_parse_global_summary_helpers(self, mwfn):
        """Test summary parsing and static number/float helpers."""
        summary = (
            "================= Summary of surface analysis\n"
            "Minimal value: -1.0 Maximal value: 2.0\n"
            "Overall surface area: 25.5\n"
            "Average value: nan\n"
            "Surface analysis finished!\n"
        )
        props = mwfn._parse_global_surface_properties(summary)
        assert props["minimal_value"] == pytest.approx(-1.0)
        assert props["maximal_value"] == pytest.approx(2.0)
        assert props["overall_surface_area"] == pytest.approx(25.5)
        assert math.isnan(props["average_value"])

        helper_props: dict[str, float] = {}
        assert mwfn._parse_summary_minmax(
            "Minimal value: -2.0 Maximal value: 3.0", helper_props
        )
        mwfn._parse_summary_key_value("Value(x): 1.25", helper_props)
        assert helper_props["valuex"] == pytest.approx(1.25)

        assert mwfn._parse_first_number("foo -1.234 bar") == pytest.approx(-1.234)
        assert mwfn._parse_first_number("no number") is None
        assert math.isnan(mwfn._parse_float("NaN"))
        assert mwfn._parse_float("1.5") == pytest.approx(1.5)

    def test_parse_electric_moments_partial(self, mwfn):
        """Test partial electric-moment extraction with missing fields."""
        parsed = mwfn._parse_electric_moments("Magnitude of dipole moment: 3.0 a.u.")
        assert parsed == {"dipole_magnitude_au": pytest.approx(3.0)}


@pytest.mark.multiwfn
class TestPexpectSessionHelpers:
    """Test pexpect-session helper methods without running external executables."""

    def test_has_progress_bar(self, fake_session):
        """Test progress-bar detection patterns."""
        has_bar, value = fake_session._has_progress_bar("Progress: [###---] 14.3 %")
        assert has_bar
        assert value == pytest.approx(14.3)

        has_bar, value = fake_session._has_progress_bar("Some text [#####]")
        assert has_bar
        assert value == pytest.approx(100.0)

        has_bar, value = fake_session._has_progress_bar("No progress here")
        assert not has_bar
        assert value is None

    def test_transcript_helpers(self, fake_session):
        """Test transcript inspection helpers."""
        fake_session._transcript = ["abc", "def", "ghi"]
        fake_session._last_command_pos = 1
        assert fake_session.get_output_since_last_command() == "defghi"
        assert fake_session.get_recent_output(n_chunks=2, max_chars=4) == "fghi"
        assert fake_session.stdout == "abcdefghi"

    def test_expect_and_try_expect(self, fake_session):
        """Test expect/try_expect success and timeout handling."""
        fake_session._child.expect_responses = [{"before": "a", "after": "b"}]
        assert fake_session.expect("pattern")
        assert fake_session._transcript[-2:] == ["a", "b"]

        fake_session._child.expect_responses = [pexpect.TIMEOUT("timeout")]
        assert not fake_session.try_expect("pattern", timeout=0.1)

    def test_read_helpers_and_exit(self, fake_session):
        """Test read_nonblocking wrappers and wait_for_exit behavior."""
        fake_session._child.read_responses = ["chunk"]
        fake_session.read_available()
        assert fake_session._transcript[-1] == "chunk"

        fake_session._child.read_responses = [
            "data",
            pexpect.TIMEOUT("x"),
            pexpect.EOF("y"),
        ]
        assert fake_session._read_progress_chunk() == "data"
        assert fake_session._read_progress_chunk() is None
        assert fake_session._read_progress_chunk() == "EOF"

        fake_session._child.before = "tail"
        fake_session._child.expect_responses = [pexpect.TIMEOUT("timeout")]
        fake_session.wait_for_exit()
        assert fake_session._child.closed
        assert fake_session._transcript[-1] == "tail"

    def test_progress_state_helpers(self, fake_session):
        """Test timeout and chunk handling logic for progress waiting."""
        state = ProgressState(
            last_activity=time.monotonic() - 2,
            saw_progress=False,
            saw_please_wait=False,
            idle_timeout=1.0,
        )
        assert fake_session._should_stop_on_timeout(state)

        state = fake_session._init_progress_state(max_wait=2.0)
        done = fake_session._handle_progress_chunk(state, "Progress: [##---] 30 %")
        assert not done
        assert state.saw_progress
        done = fake_session._handle_progress_chunk(state, "some text [#####]")
        assert done

    def test_execute_and_process_command_paths(self, fake_session, monkeypatch):
        """Test command dispatch logic for different command-step types."""
        events: list[tuple[str, Any]] = []

        def expect_ok(pattern: str) -> bool:
            events.append(("expect", pattern))
            return True

        def send_record(cmd: str) -> None:
            events.append(("send", cmd))

        def read_record() -> None:
            events.append(("read", None))

        def wait_record(max_wait: float = 30.0) -> None:
            events.append(("wait", max_wait))

        def try_expect_false(pattern: str, timeout: float = 1.0) -> bool:
            del pattern, timeout
            return False

        def try_expect_true(pattern: str, timeout: float = 1.0) -> bool:
            del pattern, timeout
            return True

        monkeypatch.setattr(
            fake_session,
            "expect",
            expect_ok,
        )
        monkeypatch.setattr(fake_session, "send", send_record)
        monkeypatch.setattr(fake_session, "read_available", read_record)
        monkeypatch.setattr(fake_session, "wait_for_progress", wait_record)

        fake_session._execute_command("cmd", "pattern", use_robust=True)
        fake_session._execute_command("cmd2", None, use_robust=False)

        fake_session._process_command("raw", use_robust=False)
        fake_session._process_command(WaitProgress(5.0), use_robust=True)

        monkeypatch.setattr(fake_session, "try_expect", try_expect_false)
        fake_session._process_command(
            CommandStep("skip", expect="optional", optional=True), use_robust=True
        )

        monkeypatch.setattr(fake_session, "try_expect", try_expect_true)
        fake_session._process_command(
            CommandStep("run", expect="needed", optional=True), use_robust=True
        )

        assert ("send", "cmd") in events
        assert ("send", "cmd2") in events
        assert ("wait", 5.0) in events
        assert ("send", "run") in events
        assert ("send", "skip") not in events

    def test_wait_for_progress_loop_timeout(self, fake_session, monkeypatch):
        """Test wait_for_progress early termination when idle timeout is reached."""
        monkeypatch.setattr(fake_session, "_read_progress_chunk", lambda: None)
        monkeypatch.setattr(fake_session, "_should_stop_on_timeout", lambda state: True)
        fake_session.wait_for_progress(max_wait=1.0)
