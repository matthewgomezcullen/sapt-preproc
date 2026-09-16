"""
Stages `prepare` runs, for PrepareComplex.prepare and its `mm` switch.

    5S8I_2LY    the cheapest structure to carry through the pipeline
"""

from collections import Counter

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from conftest import paths
from prepare import PrepareComplex
from utils import bust, mm

SMALL = "5S8I_2LY"

# Every stage `prepare` runs, in the order it runs them.
STAGES = [
    "_fetch", "_verify", "_fix", "_clean", "_protonate", "_minimise", "_bust", "_reduce",
    "_reverify", "_calculate_charge", "_verify_num_electrons", "save",
]


def stub_stages(prepared, monkeypatch):
    ran = []
    for stage in STAGES:
        monkeypatch.setattr(prepared, stage, lambda stage=stage: ran.append(stage))
    return ran


def get_heavy_atoms(pose):
    positions = pose.GetConformer().GetPositions()
    return positions[[atom.GetIdx() for atom in pose.GetAtoms() if atom.GetAtomicNum() > 1]]


def keep_all_poses(model, poses):
    return list(range(len(poses))), Counter()


def test_prepare_minimises_by_default(monkeypatch):
    prepared = PrepareComplex(*paths(SMALL))
    ran = stub_stages(prepared, monkeypatch)

    prepared.prepare()

    assert ran == STAGES


def test_without_mm_prepare_drops_the_minimisation_and_nothing_else(monkeypatch):
    prepared = PrepareComplex(*paths(SMALL), mm=False)
    ran = stub_stages(prepared, monkeypatch)

    prepared.prepare()

    assert ran == [stage for stage in STAGES if stage != "_minimise"]


@pytest.mark.long_protonate
def test_without_mm_the_prepared_poses_are_the_docked_ones(monkeypatch):
    docked = PrepareComplex(*paths(SMALL))
    docked._fetch()
    original = dict(zip(docked.source, (get_heavy_atoms(pose) for pose in docked.poses)))

    prepared = PrepareComplex(*paths(SMALL), mm=False)
    monkeypatch.setattr(bust, "valid_idxs", keep_all_poses)
    prepared.prepare()

    assert prepared.source
    for source, pose in zip(prepared.source, prepared.poses):
        assert np.allclose(original[source], get_heavy_atoms(pose))

TETHER = 7.5

SHIFT = 0.25


def shift(poses, distance=SHIFT):
    moved = []
    for pose in poses:
        copy = Chem.Mol(pose)
        conformer = copy.GetConformer()
        for index, (x, y, z) in enumerate(conformer.GetPositions()):
            conformer.SetAtomPosition(index, Point3D(x + distance, y, z))
        moved.append(copy)
    return moved


def keep(indices):
    return lambda model, poses: (list(indices), Counter())


def fetch(name=SMALL, count=3):
    protein, poses = paths(name)
    prepared = PrepareComplex(protein, sorted(poses)[:count])
    prepared._fetch()
    return prepared


def test_a_complex_is_minimised_free_by_default(monkeypatch):
    prepared = fetch()
    seen = []
    monkeypatch.setattr(mm, "minimise", lambda whole, poses, tether: seen.append(tether) or poses)

    prepared._minimise()

    assert seen == [None]


def test_minimise_records_how_far_each_pose_moved(monkeypatch):
    prepared = fetch()
    monkeypatch.setattr(mm, "minimise", lambda whole, poses, tether: shift(poses))

    prepared._minimise()

    assert prepared.displacement == pytest.approx([SHIFT] * len(prepared.poses))


def test_a_complex_that_is_not_minimised_records_no_displacement():
    prepared = PrepareComplex(*paths(SMALL), mm=False)

    assert prepared.displacement == []


def test_busting_keeps_the_displacements_parallel_to_the_poses(monkeypatch):
    prepared = fetch()
    prepared.displacement = [float(index) for index in range(len(prepared.poses))]
    sources = list(prepared.source)
    monkeypatch.setattr(bust, "valid_idxs", keep([0, 2]))

    prepared._bust()

    assert prepared.source == [sources[0], sources[2]]
    assert prepared.displacement == pytest.approx([0.0, 2.0])
