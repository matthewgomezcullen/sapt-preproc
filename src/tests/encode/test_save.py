"""
Artefacts of a complex, for utils/save.

save.py reads and writes plain records; PrepareComplex, EncodeProtein and SolveLigand decide what goes
into them. Everything one complex produces sits in the directory its caller hands over,
out/<job>/<complex>, and the SCF of each of its poses in a pose_scf directory within it.
"""

import os
from collections import Counter

import numpy as np
import pytest
from rdkit import Chem

from utils import save

NAME = "ABC_123"

STAGES = ["prepared", "solved", "encoded", "casci"]

ARTEFACTS = STAGES + ["scf"]

WRITTEN = ["prepared", "scf", "dice_log", "solved", "encoded", "casci"]

# One heavy atom and one hydrogen, enough for RDKit to read back and for a coordinate to be checked.
POSE = """ABC_123
     RDKit          3D

  2  1  0  0  0  0  0  0  0  0999 V2000
    1.2500   -0.5000    3.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.1000   -0.5000    3.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
M  END
"""

RECORD = {
    "charge": -1,
    "electrons": 1188,
    "heavy_atoms": 157,
    "cutout": "ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N",
    "poses": np.array([POSE, POSE.replace("1.2500", "4.7500")]),
    "orbitals": np.arange(9.0).reshape(3, 3),
}

# What PySCF's own chkfile holds of a converged SCF.
SCF = {
    "e_tot": -1.5,
    "mo_energy": np.arange(3.0),
    "mo_occ": np.array([2.0, 0.0, 0.0]),
    "mo_coeff": np.eye(3),
}


def stage(name):
    return getattr(save, f"save_{name}"), getattr(save, f"load_{name}")


def write_every_artefact(directory):
    save.save_prepared(RECORD, NAME, directory)
    save.save_scf(SCF, NAME, directory)
    with open(save.dice_log_path(NAME, directory), "w") as file:
        file.write("Dice's own log")
    save.save_solved(RECORD, NAME, directory)
    save.save_encoded(RECORD, NAME, directory)
    save.save_casci(RECORD, NAME, directory)


def on_disk(directory):
    return [
        artefact
        for artefact in WRITTEN
        if os.path.isfile(getattr(save, f"{artefact}_path")(NAME, directory))
    ]


def test_every_artefact_sits_in_the_complex_directory(tmp_path):
    directory = str(tmp_path)
    expected = {
        save.prepared_path: f"{NAME}_prepared.npz",
        save.solved_path: f"{NAME}_solved.npz",
        save.encoded_path: f"{NAME}_encoded.npz",
        save.casci_path: f"{NAME}_casci.npz",
        save.dice_log_path: f"{NAME}.dice.out",
        save.scf_path: f"{NAME}_rhf.chk",
    }

    for path, filename in expected.items():
        assert path(NAME, directory) == os.path.join(directory, filename)


@pytest.mark.parametrize("name", STAGES)
def test_a_record_survives_a_round_trip(tmp_path, name):
    store, fetch = stage(name)

    store(RECORD, NAME, str(tmp_path))
    loaded = fetch(NAME, str(tmp_path))

    assert loaded.keys() == RECORD.keys()
    for key, value in RECORD.items():
        np.testing.assert_array_equal(loaded[key], value)


def test_scalars_come_back_as_scalars(tmp_path):
    """
    np.load hands a scalar back as a zero-dimensional array.
    """
    save.save_prepared(RECORD, NAME, str(tmp_path))

    loaded = save.load_prepared(NAME, str(tmp_path))

    assert isinstance(loaded["charge"], int)
    assert isinstance(loaded["cutout"], str)


def test_poses_come_back_as_the_molecules_that_were_stored(tmp_path):
    save.save_prepared(RECORD, NAME, str(tmp_path))

    loaded = save.load_prepared(NAME, str(tmp_path))

    for stored, block in zip(RECORD["poses"], loaded["poses"]):
        was = Chem.MolFromMolBlock(stored, removeHs=False) # pyright: ignore[reportAttributeAccessIssue]
        now = Chem.MolFromMolBlock(block, removeHs=False) # pyright: ignore[reportAttributeAccessIssue]
        assert now.GetNumAtoms() == was.GetNumAtoms() == 2
        assert np.array_equal(
            now.GetConformer().GetPositions(), was.GetConformer().GetPositions()
        )


@pytest.mark.parametrize("name", ARTEFACTS)
def test_an_absent_artefact_loads_as_none(tmp_path, name):
    _, fetch = stage(name)

    assert fetch(NAME, str(tmp_path)) is None


@pytest.mark.parametrize("name", ARTEFACTS)
def test_an_unreadable_artefact_loads_as_none(tmp_path, name):
    """
    An interrupted job leaves a truncated file behind. It is treated as absent.
    """
    _, fetch = stage(name)
    with open(getattr(save, f"{name}_path")(NAME, str(tmp_path)), "wb") as file:
        file.write(b"PK\x03\x04 cut off mid-write")

    assert fetch(NAME, str(tmp_path)) is None


def test_saving_creates_the_complex_directory(tmp_path):
    directory = str(tmp_path / "out" / "filter" / NAME)

    save.save_prepared(RECORD, NAME, directory)

    assert os.path.isfile(save.prepared_path(NAME, directory))


def test_the_stages_are_separate_files(tmp_path):
    directory = str(tmp_path)

    save.save_scf(SCF, NAME, directory)
    save.save_solved(RECORD, NAME, directory)

    assert save.load_encoded(NAME, directory) is None
    save.save_encoded({"e_core": -1.5}, NAME, directory)
    assert save.load_solved(NAME, directory).keys() == RECORD.keys()
    assert save.load_scf(NAME, directory)["e_tot"] == SCF["e_tot"]


@pytest.mark.parametrize(
    "name, kept",
    [
        ("prepared", ["prepared"]),
        ("scf", ["prepared", "scf"]),
        ("solved", ["prepared", "scf", "dice_log", "solved"]),
        ("encoded", ["prepared", "scf", "dice_log", "solved", "encoded"]),
        ("casci", ["prepared", "scf", "dice_log", "solved", "encoded", "casci"]),
    ],
)
def test_saving_an_artefact_discards_the_ones_built_on_it(tmp_path, name, kept):
    directory = str(tmp_path)
    write_every_artefact(directory)

    store, _ = stage(name)
    store(SCF if name == "scf" else RECORD, NAME, directory)

    assert on_disk(directory) == kept


def test_a_record_that_needs_pickling_is_refused_on_save(tmp_path):
    """
    Records are read with pickling off, so one that needs it could never be read back, and loading
        treats unreadable as absent. Refusing it here is what stops the stage re-running forever.
    """
    with pytest.raises(ValueError):
        save.save_prepared({"failed": Counter({"bond_lengths": 3})}, NAME, str(tmp_path))


def test_each_pose_scf_sits_in_the_pose_scf_directory(tmp_path):
    directory = str(tmp_path)

    assert save.pose_scf_path(NAME, directory, 3) == os.path.join(
        directory, "pose_scf", f"{NAME}_pose3_rhf.chk"
    )


def test_a_pose_scf_survives_a_round_trip_apart_from_every_other_scf(tmp_path):
    directory = str(tmp_path)

    save.save_pose_scf(SCF, NAME, directory, 1)

    loaded = save.load_pose_scf(NAME, directory, 1)
    assert loaded.keys() == SCF.keys()
    for key, value in SCF.items():
        np.testing.assert_array_equal(loaded[key], value)
    assert save.load_pose_scf(NAME, directory, 0) is None
    assert save.load_scf(NAME, directory) is None


def test_an_unreadable_pose_scf_loads_as_none_and_is_written_over(tmp_path):
    directory = str(tmp_path)
    save.save_pose_scf(SCF, NAME, directory, 0)
    path = save.pose_scf_path(NAME, directory, 0)
    with open(path, "rb") as file:
        whole = file.read()
    with open(path, "wb") as file:
        file.write(whole[:len(whole) // 2])

    assert save.load_pose_scf(NAME, directory, 0) is None
    save.save_pose_scf(SCF, NAME, directory, 0)
    assert save.load_pose_scf(NAME, directory, 0)["e_tot"] == SCF["e_tot"]


def test_a_new_preparation_discards_every_pose_scf(tmp_path):
    directory = str(tmp_path)
    for index in range(3):
        save.save_pose_scf(SCF, NAME, directory, index)

    save.save_prepared(RECORD, NAME, directory)

    assert not any(os.path.isfile(save.pose_scf_path(NAME, directory, index)) for index in range(3))


@pytest.mark.parametrize("name", ["scf", "solved", "encoded", "casci"])
def test_the_proteins_own_artefacts_leave_the_pose_scfs_alone(tmp_path, name):
    directory = str(tmp_path)
    save.save_pose_scf(SCF, NAME, directory, 0)
    store, _ = stage(name)

    store(SCF if name == "scf" else RECORD, NAME, directory)

    assert os.path.isfile(save.pose_scf_path(NAME, directory, 0))


def test_saving_a_pose_scf_discards_no_other_artefact(tmp_path):
    directory = str(tmp_path)
    write_every_artefact(directory)
    save.save_pose_scf(SCF, NAME, directory, 0)
    save.save_pose_scf(SCF, NAME, directory, 1)

    save.save_pose_scf(SCF, NAME, directory, 0)

    assert on_disk(directory) == WRITTEN
    assert os.path.isfile(save.pose_scf_path(NAME, directory, 1))
