"""Chemistry-focused integration tests for Multiwfn."""

from __future__ import annotations

from pathlib import Path

import pytest

from morfeus.multiwfn import Multiwfn

EXAMPLE_MOLDEN_DIR = Path(__file__).parent / "data" / "multiwfn" / "example_molden"
SINGLET_MOLDEN_FILES = sorted(EXAMPLE_MOLDEN_DIR.glob("*singlet*.molden"))
TRIPLET_MOLDEN_FILES = sorted(EXAMPLE_MOLDEN_DIR.glob("*triplet*.molden"))
ALL_MOLDEN_FILES = SINGLET_MOLDEN_FILES + TRIPLET_MOLDEN_FILES
MOLDEN_FILE_BY_NAME = {file_path.name: file_path for file_path in ALL_MOLDEN_FILES}

BOND_ORDER_PARTITION_MODELS = ("mayer", "wiberg", "fuzzy")
DOUBLE_BOND_RANGE = (1.5, 2.5)
SINGLE_BOND_RANGE = (0.75, 1.3)
DOUBLE_BOND_PAIR = (3, 4)
SINGLE_BOND_PAIRS = ((1, 2), (1, 7), (1, 8), (1, 9))
OXYGEN_INDEX = 4
HYDROGEN_INDICES = (7, 8, 9, 10, 11, 12, 13, 14)
SINGLET_TRIPLET_EQUIVALENT_FILES = (
    ("xtb_singlet.molden", "xtb_triplet.molden"),
    (
        "orca_dft_singlet_molden2aim.molden",
        "orca_dft_triplet_uks_molden2aim.molden",
    ),
    ("pyscf_dft_singlet_rks.molden", "pyscf_dft_triplet_uks.molden"),
    (
        "pyscf_casscf_singlet_rhf_natorb.molden",
        "pyscf_casscf_triplet_rohf_natorb_molden2aim.molden",
    ),
)
MULTIPOLE_KEYS = ("quadrupole_spherical_magnitude", "quadrupole_traceless_magnitude")


@pytest.mark.multiwfn
@pytest.mark.parametrize("file_path", SINGLET_MOLDEN_FILES, ids=lambda path: path.name)
def test_singlet_bond_orders_follow_expected_chemistry(
    file_path: Path, tmp_path: Path
) -> None:
    """Assert expected single- and double-bond patterns for singlet Molden inputs."""
    mwfn = Multiwfn(file_path, run_path=tmp_path / file_path.stem, has_spin=False)

    for model in BOND_ORDER_PARTITION_MODELS:
        bond_orders = mwfn.get_bond_order(model=model)

        double_bond_value = _get_bond_order(bond_orders, *DOUBLE_BOND_PAIR)
        assert DOUBLE_BOND_RANGE[0] <= double_bond_value <= DOUBLE_BOND_RANGE[1], (
            f"{file_path.name} ({model}): expected bond {DOUBLE_BOND_PAIR[0]}-"
            f"{DOUBLE_BOND_PAIR[1]} in {DOUBLE_BOND_RANGE}, got {double_bond_value:.3f}"
        )

        for atom_i, atom_j in SINGLE_BOND_PAIRS:
            single_bond_value = _get_bond_order(bond_orders, atom_i, atom_j)
            assert SINGLE_BOND_RANGE[0] <= single_bond_value <= SINGLE_BOND_RANGE[1], (
                f"{file_path.name} ({model}): expected bond {atom_i}-{atom_j} in "
                f"{SINGLE_BOND_RANGE}, got {single_bond_value:.3f}"
            )


@pytest.mark.multiwfn
@pytest.mark.parametrize("file_path", SINGLET_MOLDEN_FILES, ids=lambda path: path.name)
def test_singlet_hirshfeld_charges_follow_expected_sign_pattern(
    file_path: Path, tmp_path: Path
) -> None:
    """Assert expected charge signs for oxygen and hydrogens in singlet files."""
    mwfn = Multiwfn(file_path, run_path=tmp_path / file_path.stem, has_spin=False)
    charges = mwfn.get_charges(model="hirshfeld")

    assert min(charges, key=lambda atom_index: charges[atom_index]) == OXYGEN_INDEX
    for hydrogen_index in HYDROGEN_INDICES:
        assert charges[hydrogen_index] > 0.0


@pytest.mark.multiwfn
@pytest.mark.parametrize("file_path", ALL_MOLDEN_FILES, ids=lambda path: path.name)
def test_rho_descriptor_is_highest_on_oxygen_for_all_spin_states(
    file_path: Path, tmp_path: Path
) -> None:
    """Assert oxygen has the largest integrated electron density contribution."""
    has_spin = "triplet" in file_path.stem.lower()
    mwfn = Multiwfn(file_path, run_path=tmp_path / file_path.stem, has_spin=has_spin)
    rho = mwfn.get_descriptor("rho")

    assert max(rho, key=lambda atom_index: rho[atom_index]) == OXYGEN_INDEX


@pytest.mark.multiwfn
@pytest.mark.parametrize("file_path", ALL_MOLDEN_FILES, ids=lambda path: path.name)
def test_all_spin_state_files_have_nonzero_dipole_and_multipole(
    file_path: Path, tmp_path: Path
) -> None:
    """Assert all molecules show nonzero dipole and multipole magnitudes."""
    has_spin = "triplet" in file_path.stem.lower()
    mwfn = Multiwfn(file_path, run_path=tmp_path / file_path.stem, has_spin=has_spin)
    moments = mwfn.get_electric_moments()

    assert abs(moments["dipole_magnitude_au"]) > 0.0
    multipoles = [moments[key] for key in MULTIPOLE_KEYS if key in moments]
    assert multipoles, f"{file_path.name}: no multipole value was parsed."
    assert any(abs(value) > 0.0 for value in multipoles)


@pytest.mark.multiwfn
@pytest.mark.parametrize("model", BOND_ORDER_PARTITION_MODELS)
@pytest.mark.parametrize(
    "singlet_name,triplet_name",
    SINGLET_TRIPLET_EQUIVALENT_FILES,
)
def test_triplet_co_bond_order_is_lower_than_equivalent_singlet(
    singlet_name: str, triplet_name: str, model: str, tmp_path: Path
) -> None:
    """Assert C=O bond order decreases going from singlet to equivalent triplet."""
    singlet_file = MOLDEN_FILE_BY_NAME[singlet_name]
    triplet_file = MOLDEN_FILE_BY_NAME[triplet_name]

    singlet_mwfn = Multiwfn(
        singlet_file,
        run_path=tmp_path / singlet_file.stem,
        has_spin=False,
    )
    triplet_mwfn = Multiwfn(
        triplet_file,
        run_path=tmp_path / triplet_file.stem,
        has_spin=True,
    )

    singlet_bond_orders = singlet_mwfn.get_bond_order(model=model)
    triplet_bond_orders = triplet_mwfn.get_bond_order(model=model)

    singlet_co_bond_order = _get_bond_order(singlet_bond_orders, *DOUBLE_BOND_PAIR)
    triplet_co_bond_order = _get_bond_order(triplet_bond_orders, *DOUBLE_BOND_PAIR)

    assert triplet_co_bond_order < singlet_co_bond_order, (
        f"{model}: expected triplet C-O bond order to be lower than singlet for "
        f"{triplet_name} vs {singlet_name}, but got "
        f"{triplet_co_bond_order:.3f} >= {singlet_co_bond_order:.3f}"
    )


def _get_bond_order(
    bond_orders: dict[tuple[int, int], float], atom_i: int, atom_j: int
) -> float:
    """Return bond order value for a pair, independent of index order."""
    if (atom_i, atom_j) in bond_orders:
        return bond_orders[(atom_i, atom_j)]
    if (atom_j, atom_i) in bond_orders:
        return bond_orders[(atom_j, atom_i)]

    raise AssertionError(f"Bond order for atoms {atom_i}-{atom_j} was not reported.")
