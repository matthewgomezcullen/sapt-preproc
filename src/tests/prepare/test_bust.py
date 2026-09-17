"""
PoseBusters screening, for PrepareComplex._bust.

Most of these carry a complex through repair, protonation and minimisation and are marked for it.
    The bookkeeping _bust does around PoseBusters' verdict is not, so it stands in front of a rerun
    of filter.py rather than behind a flag.

    5S8I_2LY    the cheapest structure to carry through the pipeline
    6ZCY_QF8    27 of its 40 poses are rejected before minimisation
"""

from collections import Counter
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from conftest import paths
from prepare import PrepareComplex, OutOfScopeError, OutOfScopeErrorType
from utils import bust, verify

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


def fetched(name, count=3):
    """
    A complex read off disk, which is all the bookkeeping tests need.
    """
    protein, poses = paths(name)
    prepared = PrepareComplex(protein, sorted(poses)[:count])
    prepared._fetch()
    return prepared


def keeping(indices, failed=None):

    return lambda model, poses: (list(indices), Counter(failed or {}))


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


def swallow_single_residues(model, keep):
    """
    Swallow single residues between two kept ones.
    """
    widened = set(keep)
    for chain in model:
        identifiers = [verify.identifier(chain, residue) for residue in chain]
        for before, gap, after in zip(identifiers, identifiers[1:], identifiers[2:]):
            if before in keep and after in keep:
                widened.add(gap)
    return widened


def residues(model):
    return {
        (chain.name, residue.seqid.num, residue.seqid.icode)
        for chain in model
        for residue in chain
        if residue.name not in {"ACE", "NME"}
    }


def test_bust_narrows_the_sources_with_the_poses(monkeypatch):
    """
    Dropping the middle pose must drop the middle name.
    """
    prepared = fetched(SMALL)
    before = list(prepared.source)
    assert len(before) == 3
    monkeypatch.setattr(bust, "valid_idxs", keeping([0, 2]))

    prepared._bust()

    assert prepared.source == [before[0], before[2]]
    assert len(prepared.source) == len(prepared.poses)


def test_bust_counts_what_it_dropped_against_the_sources_it_kept(monkeypatch):
    prepared = fetched(SMALL)
    monkeypatch.setattr(bust, "valid_idxs", keeping([1]))

    prepared._bust()

    assert prepared.excluded == 2
    assert len(prepared.source) == 1


def test_bust_rejects_a_complex_posebusters_empties(monkeypatch):
    prepared = fetched(SMALL)
    monkeypatch.setattr(bust, "valid_idxs", keeping([]))

    with pytest.raises(OutOfScopeError) as rejection:
        prepared._bust()

    assert rejection.value.error_type is OutOfScopeErrorType.INVALID_POSES


@pytest.mark.prepare_long
def test_bust_keeps_the_poses_posebusters_accepts():
    prepared = minimised(CLASHING)
    before = list(prepared.poses)

    prepared._bust()

    assert prepared.poses
    assert len(prepared.poses) <= len(before)
    # Narrowed in place, in the order they came in, so a pose still answers to its rank.
    assert [pose for pose in before if pose in prepared.poses] == list(prepared.poses)


@pytest.mark.prepare_long
def test_bust_records_how_many_it_excluded():
    """
    filter.py writes this into the screen.
    """
    prepared = minimised(CLASHING)
    before = len(prepared.poses)

    prepared._bust()

    assert prepared.excluded == before - len(prepared.poses)


@pytest.mark.prepare_long
def test_bust_rejects_a_complex_with_no_valid_pose():
    """
    An ensemble PoseBusters empties is out of scope.
    """
    prepared = minimised(SMALL)
    prepared.poses = [broken(prepared.poses[0])]
    prepared.source = [prepared.source[0]]

    with pytest.raises(OutOfScopeError) as rejection:
        prepared._bust()

    assert rejection.value.error_type is OutOfScopeErrorType.INVALID_POSES


@pytest.mark.prepare_long
def test_bust_drops_a_pose_that_fails_any_check():
    """
    The positions it hands back are what _bust narrows both of its lists by.
    """
    prepared = minimised(SMALL)
    sound = prepared.poses[0]

    kept, failed = bust.valid_idxs(prepared.whole, [sound, broken(sound)])

    assert kept == [0]
    assert failed


@pytest.mark.prepare_long
def test_bust_holds_a_pose_against_the_protein_only():
    """
    _clean has already deleted every heterogen, so the cofactor and water checks are vacuous.
    """
    prepared = minimised(CLASHING)

    _, failed = bust.valid_idxs(prepared.whole, prepared.poses)

    assert not [check for check in failed if "cofactor" in check or "water" in check]


@pytest.mark.prepare_long
def test_bust_does_not_hold_distance_from_the_protein_against_a_pose():
    prepared = minimised(CLASHING)
    away = displaced(prepared.poses[0], AWAY)

    kept, failed = bust.valid_idxs(prepared.whole, [away])

    assert kept == [0]
    assert not [check for check in failed if "maximum_distance" in check]


@pytest.mark.prepare_long
def test_reduce_takes_the_cutout_over_the_surviving_poses(monkeypatch):
    prepared = minimised(CLASHING)
    prepared._bust()
    surviving = {
        verify.identifier(chain, residue)
        for chain, residue, _, _ in verify.cutout(
            prepared.whole, prepared._pose_coordinates(), prepared.cutoff
        )
    }
    monkeypatch.setattr(verify, "incomplete_residues", lambda protein_path: set())

    prepared._reduce()

    assert residues(prepared.reduced) <= swallow_single_residues(prepared.whole, surviving)
