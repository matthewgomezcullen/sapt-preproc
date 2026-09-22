"""
Pose hydrogens and pose minimisation, for PrepareComplex._protonate and ._minimise.

_protonate adds the protein's hydrogens with Modeller and the poses' with RDKit, in one method.
_minimise relaxes each pose in the protonated protein, protein fixed and ligand free.

    5S8I_2LY    the cheapest structure to carry through the pipeline
    6ZCY_QF8    27 of its 40 poses are rejected before minimisation
"""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from conftest import paths
from prepare import PrepareComplex
from utils import mm

pytestmark = pytest.mark.prepare_long

SMALL = "5S8I_2LY"
CLASHING = "6ZCY_QF8"

# PoseBusters calls a heavy-atom pair a clash below 0.75 of the sum of its van der Waals radii.
CLASH = 0.75

VDW = {"C": 1.7, "N": 1.6, "O": 1.55, "S": 1.8, "F": 1.47, "P": 1.8, "Cl": 1.75}


def protonated(name):
    prepared = PrepareComplex(*paths(name))
    prepared._fetch()
    prepared._verify()
    prepared._fix()
    prepared._clean()
    prepared._protonate()
    return prepared


def get_heavy(pose):
    positions = pose.GetConformer().GetPositions()
    indices = [a.GetIdx() for a in pose.GetAtoms() if a.GetAtomicNum() > 1]
    return positions[indices], [pose.GetAtomWithIdx(i).GetSymbol() for i in indices]


def protein_heavy(model):
    return np.array([
        (atom.pos.x, atom.pos.y, atom.pos.z)
        for chain in model
        for residue in chain
        for atom in residue
        if not atom.element.is_hydrogen
    ]), [
        atom.element.name
        for chain in model
        for residue in chain
        for atom in residue
        if not atom.element.is_hydrogen
    ]


def closest(pose, model):
    """
    The smallest ligand-protein heavy-atom separation, relative to the sum of the pair's radii.
    """
    ligand, ligand_elements = get_heavy(pose)
    protein, protein_elements = protein_heavy(model)
    distances = np.linalg.norm(ligand[:, None, :] - protein[None, :, :], axis=-1)
    radii = np.array([VDW.get(e, 1.7) for e in ligand_elements])[:, None] + np.array(
        [VDW.get(e, 1.7) for e in protein_elements]
    )[None, :]
    return (distances / radii).min()


def test_protonate_gives_every_pose_explicit_hydrogens():
    prepared = protonated(SMALL)

    for pose in prepared.poses:
        assert any(atom.GetAtomicNum() == 1 for atom in pose.GetAtoms())
        # AddHs satisfies the valences the SDF's bond graph implies; nothing is left implicit.
        assert all(atom.GetNumImplicitHs() == 0 for atom in pose.GetAtoms())


def test_protonate_leaves_pose_heavy_atoms_where_they_were():
    before = PrepareComplex(*paths(SMALL))
    before._fetch()
    original = [get_heavy(pose)[0] for pose in before.poses]

    prepared = protonated(SMALL)

    for was, pose in zip(original, prepared.poses):
        assert np.allclose(was, get_heavy(pose)[0])


def test_minimise_moves_pose_heavy_atoms():
    prepared = protonated(CLASHING)
    before = [get_heavy(pose)[0] for pose in prepared.poses]

    prepared._minimise()

    moved = [
        np.linalg.norm(was - get_heavy(pose)[0], axis=-1).max()
        for was, pose in zip(before, prepared.poses)
    ]
    assert max(moved) > 0.1
    # A relaxation, not a relocation
    assert max(moved) < 5.0


def test_minimise_preserves_the_molecule():
    prepared = protonated(CLASHING)
    before = [
        (rdMolDescriptors.CalcMolFormula(pose), Chem.MolToSmiles(Chem.RemoveHs(pose)))
        for pose in prepared.poses
    ]

    prepared._minimise()

    for (formula, smiles), pose in zip(before, prepared.poses):
        assert rdMolDescriptors.CalcMolFormula(pose) == formula
        assert Chem.MolToSmiles(Chem.RemoveHs(pose)) == smiles


def test_minimise_relieves_clashes():
    prepared = protonated(CLASHING)
    before = [closest(pose, prepared.whole) for pose in prepared.poses]

    prepared._minimise()
    after = [closest(pose, prepared.whole) for pose in prepared.poses]

    assert sum(1 for r in after if r < CLASH) < sum(1 for r in before if r < CLASH)
    for was, now in zip(before, after):
        if was < CLASH:
            assert now > was


def test_minimise_leaves_the_protein_fixed():
    prepared = protonated(CLASHING)
    before, _ = protein_heavy(prepared.whole)

    prepared._minimise()

    assert np.allclose(before, protein_heavy(prepared.whole)[0])


def test_minimise_keeps_the_poses_in_the_protein_frame():
    prepared = protonated(CLASHING)
    before = [get_heavy(pose)[0].mean(axis=0) for pose in prepared.poses]

    prepared._minimise()

    for was, pose in zip(before, prepared.poses):
        assert np.linalg.norm(was - get_heavy(pose)[0].mean(axis=0)) < 5.0


def test_charges_are_assigned_once_for_the_whole_ensemble(monkeypatch):
    """
    Every pose is one molecule in another conformation, and AM1-BCC is the expensive part.
    """
    prepared = protonated(CLASHING)
    assert len(prepared.poses) > 1
    calls = []
    parameterise = mm._parameterise
    monkeypatch.setattr(
        mm, "_parameterise", lambda pose: calls.append(pose) or parameterise(pose)
    )

    prepared._minimise()

    assert len(calls) == 1
