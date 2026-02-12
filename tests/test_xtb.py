"""Test XTB code."""

from pathlib import Path

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from morfeus import read_xyz, XTB
from morfeus.data import AU_TO_DEBYE, HARTREE_TO_EV, HARTREE_TO_KCAL, HARTREE_TO_KJ

DATA_DIR = Path(__file__).parent / "data" / "xtb"


@pytest.fixture
def pentenone():
    """Load xTB test molecule."""
    return read_xyz(DATA_DIR / "1-penten-3-one.xyz")


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


@pytest.mark.xtb
def test_sp_descriptors_shapes_and_consistency(tmp_path, pentenone):
    """Test basic GFN2-xTB single-point descriptors."""
    elements, coordinates = pentenone
    n_atoms = len(elements)
    xtb = XTB(elements, coordinates, run_path=tmp_path)

    bond_orders = xtb.get_bond_orders()
    assert len(bond_orders) > 0
    (i, j), bond_order = next(iter(bond_orders.items()))
    assert bond_order > 0
    assert xtb.get_bond_order(i, j) == bond_order
    assert xtb.get_bond_order(j, i) == bond_order

    missing_pair = next(
        (a, b)
        for a in range(1, n_atoms + 1)
        for b in range(a + 1, n_atoms + 1)
        if (a, b) not in bond_orders and (b, a) not in bond_orders
    )
    with pytest.raises(ValueError, match="No bond order calculated"):
        xtb.get_bond_order(*missing_pair)

    charges = xtb.get_charges(model="Mulliken")
    assert len(charges) == n_atoms
    assert np.isclose(sum(charges.values()), 0.0, atol=1e-3)

    energy = xtb.get_energy()
    assert np.isfinite(energy)

    homo_eh = xtb.get_homo(unit="Eh")
    homo_ev = xtb.get_homo(unit="eV")
    lumo_eh = xtb.get_lumo(unit="Eh")
    lumo_ev = xtb.get_lumo(unit="eV")
    gap_ev = xtb.get_homo_lumo_gap(unit="eV")
    assert np.isclose(homo_ev, homo_eh * HARTREE_TO_EV, atol=1e-3)
    assert np.isclose(lumo_ev, lumo_eh * HARTREE_TO_EV, atol=1e-3)
    assert np.isclose(gap_ev, lumo_ev - homo_ev, atol=5e-3)
    assert np.isclose(
        xtb.get_homo_lumo_gap(unit="kcal/mol"),
        xtb.get_homo_lumo_gap(unit="Eh") * HARTREE_TO_KCAL,
        atol=1e-3,
    )
    assert np.isclose(
        xtb.get_homo_lumo_gap(unit="kJ/mol"),
        xtb.get_homo_lumo_gap(unit="Eh") * HARTREE_TO_KJ,
        atol=1e-3,
    )

    fermi_level = xtb.get_fermi_level()
    assert np.isfinite(fermi_level)

    dipole_vect = xtb.get_dipole()
    assert dipole_vect.shape == (3,)
    dipole_au = xtb.get_dipole_moment(unit="au")
    dipole_debye = xtb.get_dipole_moment(unit="debye")
    assert np.isclose(dipole_debye, dipole_au * AU_TO_DEBYE, atol=2e-3)
    assert np.isclose(np.linalg.norm(dipole_vect), dipole_au, atol=2e-3)

    atom_dipoles = xtb.get_atom_dipoles()
    assert len(atom_dipoles) == n_atoms
    assert all(v.shape == (3,) for v in atom_dipoles.values())

    atom_dipole_moments = xtb.get_atom_dipole_moments(unit="au")
    atom_dipole_moments_debye = xtb.get_atom_dipole_moments(unit="debye")
    assert len(atom_dipole_moments) == n_atoms
    assert len(atom_dipole_moments_debye) == n_atoms
    for idx in (1, n_atoms):
        assert np.isclose(
            atom_dipole_moments[idx],
            np.linalg.norm(atom_dipoles[idx]),
            atol=1e-8,
        )
        assert np.isclose(
            atom_dipole_moments_debye[idx],
            atom_dipole_moments[idx] * AU_TO_DEBYE,
            atol=1e-8,
        )

    atom_polarizabilities = xtb.get_atom_polarizabilities()
    assert len(atom_polarizabilities) == n_atoms
    assert all(np.isfinite(value) for value in atom_polarizabilities.values())

    molecular_polarizability = xtb.get_molecular_polarizability()
    assert np.isfinite(molecular_polarizability)

    covcn = xtb.get_covcn()
    assert len(covcn) == n_atoms


@pytest.mark.xtb
def test_ipea_and_global_descriptors_consistency(tmp_path, pentenone):
    """Test IPEA-derived descriptors and global descriptor definitions."""
    elements, coordinates = pentenone
    xtb = XTB(elements, coordinates, run_path=tmp_path)

    ip = xtb.get_ip()
    ea = xtb.get_ea()
    assert np.isfinite(ip)
    assert np.isfinite(ea)

    chem_pot = xtb.get_chemical_potential()
    assert np.isclose(chem_pot, -(ip + ea) / 2, atol=1e-8)
    assert np.isclose(xtb.get_electronegativity(), -chem_pot, atol=1e-8)

    hardness = xtb.get_hardness()
    assert hardness == round(ip - ea, 4)
    assert xtb.get_softness() == round(1 / hardness, 4)

    assert np.isclose(
        xtb.get_global_descriptor("electrophilicity"),
        (ip + ea) ** 2 / (8 * (ip - ea)),
        atol=1e-8,
    )
    assert np.isclose(xtb.get_global_descriptor("nucleophilicity"), -ip, atol=1e-8)
    assert np.isclose(
        xtb.get_global_descriptor("electrofugality"),
        (3 * ip - ea) ** 2 / (8 * (ip - ea)),
        atol=1e-8,
    )
    assert np.isclose(
        xtb.get_global_descriptor("nucleofugality"),
        (ip - 3 * ea) ** 2 / (8 * (ip - ea)),
        atol=1e-8,
    )

    ip_uncorrected = xtb.get_ip(corrected=False)
    ea_uncorrected = xtb.get_ea(corrected=False)
    assert np.isfinite(ip_uncorrected)
    assert np.isfinite(ea_uncorrected)

    with pytest.raises(ValueError, match="does not exist"):
        xtb.get_global_descriptor("invalid")


@pytest.mark.xtb
def test_gfn1_descriptors_and_method_restrictions(tmp_path, pentenone):
    """Test GFN1-xTB specific outputs and guarded functionality."""
    elements, coordinates = pentenone
    n_atoms = len(elements)
    xtb = XTB(elements, coordinates, method=1, run_path=tmp_path / "gfn1")

    mulliken = xtb.get_charges(model="Mulliken")
    cm5 = xtb.get_charges(model="CM5")
    assert len(mulliken) == n_atoms
    assert len(cm5) == n_atoms
    assert np.isclose(sum(cm5.values()), 0.0, atol=1e-3)

    assert len(xtb.get_s_pop()) == n_atoms
    assert len(xtb.get_p_pop()) == n_atoms
    assert len(xtb.get_d_pop()) == n_atoms

    with pytest.raises(ValueError, match="Atomic dipoles are not available"):
        xtb.get_atom_dipoles()
    with pytest.raises(ValueError, match="Polarizability is only available"):
        xtb.get_atom_polarizabilities()
    with pytest.raises(
        ValueError, match="Covalent coordination number is only available"
    ):
        xtb.get_covcn()

    xtb_gfn2 = XTB(elements, coordinates, run_path=tmp_path / "gfn2")
    with pytest.raises(ValueError, match="CM5 charge model is only available"):
        xtb_gfn2.get_charges(model="CM5")


@pytest.mark.xtb
def test_fod_population_and_nfod_consistency(tmp_path, pentenone):
    """Test FOD outputs and NFOD aggregation."""
    elements, coordinates = pentenone
    n_atoms = len(elements)
    xtb = XTB(elements, coordinates, run_path=tmp_path)

    fod_population = xtb.get_fod_population()
    assert len(fod_population) == n_atoms
    assert all(np.isfinite(value) for value in fod_population.values())
    assert np.isclose(xtb.get_nfod(), sum(fod_population.values()), atol=1e-12)


@pytest.mark.xtb
def test_solvation_outputs_and_unit_conversions(tmp_path, pentenone):
    """Test solvation descriptors with a polar solvent."""
    elements, coordinates = pentenone
    n_atoms = len(elements)
    xtb = XTB(elements, coordinates, solvent="water", run_path=tmp_path / "water")

    g_solv_eh = xtb.get_solvation_energy(unit="Eh")
    assert np.isfinite(g_solv_eh)
    assert np.isclose(xtb.get_solvation_energy(unit="eV"), g_solv_eh * HARTREE_TO_EV)
    assert np.isclose(
        xtb.get_solvation_energy(unit="kcal/mol"),
        g_solv_eh * HARTREE_TO_KCAL,
    )
    assert np.isclose(
        xtb.get_solvation_energy(unit="kJ/mol"),
        g_solv_eh * HARTREE_TO_KJ,
    )

    g_hbond_eh = xtb.get_solvation_h_bond_correction(unit="Eh")
    assert np.isfinite(g_hbond_eh)
    assert np.isclose(
        xtb.get_solvation_h_bond_correction(unit="eV"),
        g_hbond_eh * HARTREE_TO_EV,
    )
    assert np.isclose(
        xtb.get_solvation_h_bond_correction(unit="kcal/mol"),
        g_hbond_eh * HARTREE_TO_KCAL,
    )
    assert np.isclose(
        xtb.get_solvation_h_bond_correction(unit="kJ/mol"),
        g_hbond_eh * HARTREE_TO_KJ,
    )

    atom_hbond = xtb.get_atomic_h_bond_corrections(unit="Eh")
    assert len(atom_hbond) == n_atoms
    assert all(np.isfinite(value) for value in atom_hbond.values())

    no_solvent_xtb = XTB(elements, coordinates, run_path=tmp_path / "no-solvent")
    with pytest.raises(ValueError, match="only available with solvent"):
        no_solvent_xtb.get_solvation_energy()
    with pytest.raises(ValueError, match="only available with solvent"):
        no_solvent_xtb.get_solvation_h_bond_correction()
    with pytest.raises(ValueError, match="Atomic hydrogen bonding corrections"):
        no_solvent_xtb.get_atomic_h_bond_corrections()


@pytest.mark.xtb
def test_generated_output_files(tmp_path, pentenone):
    """Test generation of molden, density and spin-density files."""
    elements, coordinates = pentenone

    xtb = XTB(elements, coordinates, run_path=tmp_path / "files")
    molden_file = xtb.get_molden()
    density_file = xtb.get_density()
    assert molden_file.exists()
    assert molden_file.name == XTB._xtb_molden_file
    assert molden_file.stat().st_size > 0
    assert density_file.exists()
    assert density_file.name == XTB._xtb_density_cube_file
    assert density_file.stat().st_size > 0

    xtb_spin = XTB(
        elements,
        coordinates,
        charge=1,
        n_unpaired=1,
        run_path=tmp_path / "spin",
    )
    spin_density_file = xtb_spin.get_spin_density()
    assert spin_density_file.exists()
    assert spin_density_file.name == XTB._xtb_spin_density_cube_file
    assert spin_density_file.stat().st_size > 0

    xtb_closed_shell = XTB(
        elements,
        coordinates,
        n_unpaired=0,
        run_path=tmp_path / "closed-shell",
    )
    with pytest.raises(ValueError, match="requires unpaired electrons"):
        xtb_closed_shell.get_spin_density()
