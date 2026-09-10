"""
PoseBusters screening, for PrepareComplex._bust.

    5S8I_2LY    the cheapest structure to carry through the pipeline
    6ZCY_QF8    27 of its 40 poses are rejected before minimisation
"""

import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from conftest import paths
from prepare import PrepareComplex, OutOfScopeError, OutOfScopeErrorType
from utils import bust, verify

pytestmark = pytest.mark.long_protonate

SMALL = "5S8I_2LY"
CLASHING = "6ZCY_QF8"

# Far enough that no protein atom is within reach of the displaced pose.
AWAY = 100.0


def minimised(name):
    prepared = PrepareComplex(*paths(name))
    prepared._fetch()
    prepared._verify()
    prepared._fix()
    prepared._clean()
    prepared._protonate()
    prepared._minimise()
    return prepared


def broken(pose):
    """
    A pose with two atoms on top of each other, which no geometry check can pass.
    """
    copy = Chem.Mol(pose) # pyright: ignore[reportAttributeAccessIssue]
    conformer = copy.GetConformer()
    conformer.SetAtomPosition(1, conformer.GetAtomPosition(0))
    return copy


def displaced(pose, distance):
    copy = Chem.Mol(pose) # pyright: ignore[reportAttributeAccessIssue]
    conformer = copy.GetConformer()
    for index in range(copy.GetNumAtoms()):
        position = conformer.GetAtomPosition(index)
        conformer.SetAtomPosition(
            index, Point3D(position.x + distance, position.y, position.z)
        )
    return copy


def residues(model):
    return {
        (chain.name, residue.seqid.num, residue.seqid.icode)
        for chain in model
        for residue in chain
        if residue.name not in {"ACE", "NME"}
    }


def test_bust_keeps_the_poses_posebusters_accepts():
    prepared = minimised(CLASHING)
    before = list(prepared.poses)

    prepared._bust()

    assert prepared.poses
    assert len(prepared.poses) <= len(before)
    # Narrowed in place, in the order they came in, so a pose still answers to its rank.
    assert [pose for pose in before if pose in prepared.poses] == list(prepared.poses)


def test_bust_records_how_many_it_excluded():
    """
    filter.py writes this into the screen.
    """
    prepared = minimised(CLASHING)
    before = len(prepared.poses)

    prepared._bust()

    assert prepared.excluded == before - len(prepared.poses)


def test_bust_rejects_a_complex_with_no_valid_pose():
    """
    An ensemble PoseBusters empties is out of scope.
    """
    prepared = minimised(SMALL)
    prepared.poses = [broken(prepared.poses[0])]

    with pytest.raises(OutOfScopeError) as rejection:
        prepared._bust()

    assert rejection.value.error_type is OutOfScopeErrorType.INVALID_POSES


def test_bust_drops_a_pose_that_fails_any_check():
    prepared = minimised(SMALL)
    sound = prepared.poses[0]

    kept, failed = bust.valid(prepared.whole, [sound, broken(sound)])

    assert kept == [sound]
    assert failed


def test_bust_holds_a_pose_against_the_protein_only():
    """
    _clean has already deleted every heterogen, so the cofactor and water checks are vacuous.
    """
    prepared = minimised(CLASHING)

    _, failed = bust.valid(prepared.whole, prepared.poses)

    assert not [check for check in failed if "cofactor" in check or "water" in check]


def test_bust_does_not_hold_distance_from_the_protein_against_a_pose():
    prepared = minimised(CLASHING)
    away = displaced(prepared.poses[0], AWAY)

    kept, failed = bust.valid(prepared.whole, [away])

    assert kept == [away]
    assert not [check for check in failed if "maximum_distance" in check]


def test_reduce_takes_the_cutout_over_the_surviving_poses():
    """
    The union is over what will be scored.
    """
    prepared = minimised(CLASHING)
    prepared._bust()
    surviving = {
        verify.identifier(chain, residue)
        for chain, residue, _, _ in verify.cutout(
            prepared.whole, prepared._pose_coordinates(), prepared.cutoff
        )
    }

    prepared._reduce()

    assert residues(prepared.reduced) <= surviving
