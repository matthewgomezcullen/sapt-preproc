"""
Artefacts of a complex, for utils/save.

save.py reads and writes plain records; PrepareComplex and EncodeProtein decide what goes into them.
Everything one complex produces sits in the directory its caller hands over, out/<job>/<complex>.
"""

import os
from collections import Counter

import numpy as np
import pytest
from rdkit import Chem

from utils import save

NAME = "ABC_123"

STAGES = ["prepared", "solved", "encoded"]

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


def test_every_artefact_sits_in_the_complex_directory(tmp_path):
    directory = str(tmp_path)
    expected = {
        save.prepared_path: f"{NAME}_prepared.npz",
        save.solved_path: f"{NAME}_solved.npz",
        save.encoded_path: f"{NAME}_encoded.npz",
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


@pytest.mark.parametrize("name", STAGES)
def test_an_absent_artefact_loads_as_none(tmp_path, name):
    _, fetch = stage(name)

    assert fetch(NAME, str(tmp_path)) is None


@pytest.mark.parametrize("name", STAGES)
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


def test_a_record_that_needs_pickling_is_refused_on_save(tmp_path):
    """
    Records are read with pickling off, so one that needs it could never be read back, and loading
        treats unreadable as absent. Refusing it here is what stops the stage re-running forever.
    """
    with pytest.raises(ValueError):
        save.save_prepared({"failed": Counter({"bond_lengths": 3})}, NAME, str(tmp_path))
