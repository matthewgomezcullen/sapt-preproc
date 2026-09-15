"""
What a prepared complex keeps and reads back, for PrepareComplex.save and ._load.
"""

import os
from collections import Counter

import gemmi
import numpy as np
import pytest
from rdkit import Chem

from conftest import paths
from prepare import PrepareComplex

NAME = "5S8I_2LY"

FAILED = Counter({"bond_lengths": 2, "internal_steric_clash": 1})

# How far minimisation moved each of the three poses, one entry a pose.
DISPLACEMENT = [0.12, 0.4, 1.03]


def dummy_prepared(out, count=3):
    protein, poses = paths(NAME)
    prepared = PrepareComplex(protein, sorted(poses)[:count], out)
    prepared._fetch()
    prepared.reduced = gemmi.read_pdb(protein)[0] # pyright: ignore[reportAttributeAccessIssue]
    prepared.charge = -1
    prepared.electrons = 1188
    prepared.heavy_atoms = 157
    prepared.excluded = 4
    prepared.failed = Counter(FAILED)
    prepared.displacement = list(DISPLACEMENT)
    prepared.save()
    return prepared


def reloaded(out, count=3):
    protein, poses = paths(NAME)
    return PrepareComplex(protein, sorted(poses)[:count], out)


def test_the_artefact_carries_the_file_every_pose_came_from(tmp_path):
    out = str(tmp_path / NAME)
    before = dummy_prepared(out)

    after = reloaded(out)

    assert after.source == before.source
    assert after.source == [os.path.basename(path) for path in before.poses_paths]


def test_a_reloaded_complex_keeps_its_poses_and_sources_parallel(tmp_path):
    """
    A reload that reorders or drops one silently mislabels every pose after it.
    """
    out = str(tmp_path / NAME)
    before = dummy_prepared(out)

    after = reloaded(out)

    assert len(after.source) == len(after.poses) == len(before.poses)
    for was, now in zip(before.poses, after.poses):
        assert np.allclose(
            was.GetConformer().GetPositions(), now.GetConformer().GetPositions()
        )
        assert Chem.MolToSmiles(was) == Chem.MolToSmiles(now) # pyright: ignore[reportAttributeAccessIssue]


def test_a_reloaded_complex_keeps_the_numbers_the_screen_reports(tmp_path):
    out = str(tmp_path / NAME)
    before = dummy_prepared(out)

    after = reloaded(out)

    assert after.charge == before.charge
    assert after.electrons == before.electrons
    assert after.heavy_atoms == before.heavy_atoms
    assert after.excluded == before.excluded
    assert after.failed == FAILED
    assert after.prepared()


def test_a_reloaded_complex_keeps_how_far_each_pose_moved(tmp_path):
    out = str(tmp_path / NAME)
    before = dummy_prepared(out)

    after = reloaded(out)

    assert after.displacement == pytest.approx(before.displacement)
    assert len(after.displacement) == len(after.poses)
