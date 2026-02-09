"""Test XTB code."""

from pathlib import Path

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from morfeus import read_xyz, XTB

DATA_DIR = Path(__file__).parent / "data" / "xtb"


@pytest.mark.parametrize(
    ("kwargs", "expected_fragments"),
    [
        ({}, ["--json", "--chrg 0", "--gfn 2"]),
        ({"method": 1}, ["--gfn 1"]),
        ({"method": "ptb"}, ["--ptb"]),
        (
            {
                "charge": -1,
                "solvent": "water",
                "n_unpaired": 2,
                "electronic_temperature": 500,
            },
            ["--chrg -1", "--alpb water", "--uhf 2", "--etemp 500"],
        ),
    ],
)
def test_default_xtb_command_options(tmp_path, kwargs, expected_fragments):
    """Test basic xTB command options in minimal examples."""
    xtb = XTB(["H"], [[0.0, 0.0, 0.0]], run_path=tmp_path, **kwargs)
    command = xtb._default_xtb_command

    for fragment in expected_fragments:
        assert fragment in command


def test_make_xtb_inp_keeps_default_command_immutable(tmp_path):
    """Test that preparing xtb input files does not mutate default command."""
    xtb = XTB(["H"], [[0.0, 0.0, 0.0]], solvent="water", run_path=tmp_path)
    default_command = xtb._default_xtb_command

    sp_run = tmp_path / "sp"
    sp_run.mkdir()
    assert xtb._make_xtb_inp(sp_run, "sp")
    assert xtb._default_xtb_command == default_command
    sp_input = (sp_run / XTB._xtb_input_file).read_text()
    assert "gbsa=true" in sp_input

    density_run = tmp_path / "density"
    density_run.mkdir()
    assert xtb._make_xtb_inp(density_run, "density")
    assert xtb._default_xtb_command == default_command
    density_input = (density_run / XTB._xtb_input_file).read_text()
    assert "density=true" in density_input


def test_make_xtb_inp_spin_density_writes_expected_flags(tmp_path):
    """Test spin-density option in input file generation."""
    xtb = XTB(["H"], [[0.0, 0.0, 0.0]], solvent="water", run_path=tmp_path)

    spin_run = tmp_path / "spin-density"
    spin_run.mkdir()
    assert xtb._make_xtb_inp(spin_run, "spin density")
    spin_input = (spin_run / XTB._xtb_input_file).read_text()
    assert "gbsa=true" in spin_input
    assert "spin density=true" in spin_input


@pytest.mark.xtb
def test_fukui():
    """Test Fukui coefficients."""
    elements, coordinates = read_xyz(DATA_DIR / "1-penten-3-one.xyz")
    xtb = XTB(elements, coordinates)

    ref_data = np.genfromtxt(DATA_DIR / "fukui.csv", delimiter=",", names=True)
    f_nuc = list(xtb.get_fukui(variety="nucleophilicity").values())
    assert_array_almost_equal(f_nuc, ref_data["f_nuc"], decimal=3)
    f_elec = list(xtb.get_fukui(variety="electrophilicity").values())
    assert_array_almost_equal(f_elec, ref_data["f_elec"], decimal=3)
    f_rad = list(xtb.get_fukui(variety="radical").values())
    assert_array_almost_equal(f_rad, ref_data["f_rad"], decimal=3)
    f_dual = list(xtb.get_fukui(variety="dual").values())
    assert_array_almost_equal(f_dual, ref_data["f_dual"], decimal=3)
    f_loc_nuc = list(xtb.get_fukui(variety="local_nucleophilicity").values())
    assert_array_almost_equal(f_loc_nuc, ref_data["f_loc_nuc"], decimal=3)
    f_loc_elec = list(
        xtb.get_fukui(variety="local_electrophilicity", corrected=False).values()
    )
    assert_array_almost_equal(f_loc_elec, ref_data["f_loc_elec"], decimal=3)
