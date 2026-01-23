"""Test Multiwfn code."""

from pathlib import Path

import pytest

from morfeus.multiwfn import Multiwfn

DATA_DIR = Path(__file__).parent / "data" / "multiwfn"


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
        charges = mwfn.get_charges(model="ADCH")
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
        charges = mwfn.get_charges(model="Hirshfeld")
        assert isinstance(charges, dict)
        assert len(charges) > 0
        assert all(isinstance(k, int) for k in charges.keys())
        assert all(isinstance(v, float) for v in charges.values())

    def test_get_charges_vdd(self, mwfn):
        """Test VDD charge calculation."""
        charges = mwfn.get_charges(model="VDD")
        assert isinstance(charges, dict)
        assert len(charges) > 0

    def test_get_charges_mulliken(self, mwfn):
        """Test Mulliken charge calculation."""
        charges = mwfn.get_charges(model="Mulliken")
        assert isinstance(charges, dict)
        assert len(charges) > 0

    def test_get_charges_cm5(self, mwfn):
        """Test CM5 charge calculation."""
        charges = mwfn.get_charges(model="CM5")
        assert isinstance(charges, dict)
        assert len(charges) > 0

    def test_get_charges_12cm5(self, mwfn):
        """Test 1.2*CM5 charge calculation."""
        charges = mwfn.get_charges(model="12CM5")
        assert isinstance(charges, dict)
        assert len(charges) > 0

    def test_get_charges_invalid_model(self, mwfn):
        """Test that invalid charge model raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            mwfn.get_charges(model="InvalidModel")

    def test_get_charges_caching(self, mwfn):
        """Test that charges are cached after first calculation."""
        charges1 = mwfn.get_charges(model="ADCH")
        charges2 = mwfn.get_charges(model="ADCH")
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
        surface = mwfn.get_surface(model="ESP")
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
        surface = mwfn.get_surface(model="ALIE")
        assert isinstance(surface, dict)
        assert "atomic" in surface
        assert "global" in surface

    def test_get_surface_lea(self, mwfn):
        """Test LEA surface calculation."""
        surface = mwfn.get_surface(model="LEA")
        assert isinstance(surface, dict)
        assert "atomic" in surface
        assert "global" in surface

    def test_get_surface_leae(self, mwfn):
        """Test LEAE surface calculation."""
        surface = mwfn.get_surface(model="LEAE")
        assert isinstance(surface, dict)
        assert "atomic" in surface
        assert "global" in surface

    def test_get_surface_electron(self, mwfn):
        """Test Electron Density surface calculation."""
        surface = mwfn.get_surface(model="Electron")
        assert isinstance(surface, dict)
        assert "atomic" in surface
        assert "global" in surface

    def test_get_surface_sign(self, mwfn):
        """Test Sign(lambda2)*rho surface calculation."""
        surface = mwfn.get_surface(model="Sign")
        assert isinstance(surface, dict)
        assert "atomic" in surface
        assert "global" in surface

    def test_get_surface_invalid_model(self, mwfn):
        """Test that invalid surface model raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            mwfn.get_surface(model="InvalidModel")

    def test_get_surface_caching(self, mwfn):
        """Test that surface results are cached."""
        surface1 = mwfn.get_surface(model="ESP")
        surface2 = mwfn.get_surface(model="ESP")
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

    def test_init_debug_mode(self):
        """Test initialization with debug mode enabled."""
        molden_file = DATA_DIR / "example_xtb" / "molden.input"
        mwfn = Multiwfn(molden_file, debug=True)
        assert mwfn._file_path.exists()


@pytest.mark.multiwfn
class TestMultiwfnResults:
    """Test results caching and access."""

    def test_results_caching_multiple_models(self):
        """Test that multiple charge models are cached separately."""
        molden_file = DATA_DIR / "example_xtb" / "molden.input"
        mwfn = Multiwfn(molden_file, run_path=DATA_DIR / "test_output")

        charges_adch = mwfn.get_charges(model="ADCH")
        charges_hirsh = mwfn.get_charges(model="Hirshfeld")

        # Should be different results
        assert charges_adch is not charges_hirsh

        # Both should be cached
        assert mwfn._results.charges is not None
        assert "ADCH" in mwfn._results.charges
        assert "Hirshfeld" in mwfn._results.charges

    def test_results_multiple_surfaces(self):
        """Test that multiple surface models are cached separately."""
        molden_file = DATA_DIR / "example_xtb" / "molden.input"
        mwfn = Multiwfn(molden_file, run_path=DATA_DIR / "test_output")

        surface_esp = mwfn.get_surface(model="ESP")
        surface_alie = mwfn.get_surface(model="ALIE")

        # Should be different results
        assert surface_esp is not surface_alie

        # Both should be cached
        assert mwfn._results.surfaces is not None
        assert "ESP" in mwfn._results.surfaces
        assert "ALIE" in mwfn._results.surfaces


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
