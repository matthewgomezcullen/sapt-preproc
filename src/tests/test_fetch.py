import os

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from conftest import paths
from prepare import PrepareComplex, PrepareError

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
COMPLEX = os.path.join(DATA, "5S8I_2LY")
PROTEIN = os.path.join(COMPLEX, "5S8I_2LY_protein.pdb")
POSES_DIR = os.path.join(COMPLEX, "poses")


@pytest.fixture
def poses_paths():
    return sorted(os.path.join(POSES_DIR, f) for f in os.listdir(POSES_DIR) if f.endswith(".sdf"))


def test_fetch_loads_protein_and_every_pose(poses_paths):
    """
    _fetch populates the protein model and one sanitised, heavy-atom-only RDKit mol per pose,
        each carrying a 3D conformer in the same coordinate frame as the PDB.
    """
    prepared = PrepareComplex(PROTEIN, poses_paths)
    prepared._fetch()

    assert prepared.whole is not None
    assert sum(len(chain) for chain in prepared.whole) > 0

    assert len(prepared.poses) == len(poses_paths)
    assert all(pose is not None for pose in prepared.poses)

    for pose in prepared.poses:
        assert pose.GetNumConformers() == 1
        assert pose.GetConformer().Is3D()
        # DiffDock writes heavy atoms only; hydrogens are added downstream, not by _fetch.
        assert not any(atom.GetAtomicNum() == 1 for atom in pose.GetAtoms())
        # Sanitisation succeeded, so valences and aromaticity are perceived.
        Chem.SanitizeMol(Chem.Mol(pose))

    # The pose ensemble is a set of conformers of one ligand.
    formulas = {rdMolDescriptors.CalcMolFormula(pose) for pose in prepared.poses}
    assert len(formulas) == 1

    # Poses share the PDB's frame, so they sit inside the binding site rather than at the origin.
    protein_positions = [
        (atom.pos.x, atom.pos.y, atom.pos.z)
        for chain in prepared.whole
        for residue in chain
        for atom in residue
    ]
    pose_positions = prepared.poses[0].GetConformer().GetPositions()
    nearest = min(
        (px - ax) ** 2 + (py - ay) ** 2 + (pz - az) ** 2
        for px, py, pz in pose_positions
        for ax, ay, az in protein_positions
    ) ** 0.5
    assert nearest < 4.5


def test_fetch_raises_on_unparseable_pose(tmp_path, poses_paths):
    """
    An SDF RDKit cannot parse must raise, not leave None in self.poses for later steps to trip on.
    """
    broken = tmp_path / "rank1_confidence-0.00.sdf"
    broken.write_text("not an sdf\n")

    prepared = PrepareComplex(PROTEIN, [poses_paths[0], str(broken)])

    with pytest.raises(PrepareError):
        prepared._fetch()


@pytest.mark.parametrize("name", ["7OPG_06N", "7R9N_F97", "7UQ3_O2U", "6M73_FNR", "7SCW_GSP"])
def test_positive_confidence_poses_are_not_dropped(name):
    """
    DiffDock writes its confidence unsigned when it is positive, so a pattern that assumes a leading
        minus silently discards the highest-confidence poses, and every pose of these three.
    """
    _, poses = paths(name)

    assert poses
    assert any("confidence-" not in os.path.basename(pose) for pose in poses)
    assert not any(pose.endswith("confidence-1000.00.sdf") for pose in poses)
