"""
Stages `prepare` runs, for PrepareComplex.prepare and its `mm` switch.

    5S8I_2LY    the cheapest structure to carry through the pipeline
"""

from collections import Counter

import numpy as np
import pytest

from conftest import paths
from prepare import PrepareComplex
from utils import bust

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
    monkeypatch.setattr(bust, "valid", keep_all_poses)
    prepared.prepare()

    assert prepared.source
    for source, pose in zip(prepared.source, prepared.poses):
        assert np.allclose(original[source], get_heavy_atoms(pose))
