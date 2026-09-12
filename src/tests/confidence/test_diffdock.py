"""
The scoring pass itself, for utils/diffdock.py.

It runs in the interpreter DiffDock was installed into, so these are skipped unless DIFFDOCK_PYTHON
    and DIFFDOCK_MODELS say where that interpreter and the released weights are. The first run also
    downloads ESM-2 650M, which is gigabytes; after that a complex is a minute or so on a CPU.

    5S8I_2LY    twenty poses, the cheapest structure in the set
"""

import os
import random

import pytest
from rdkit import Chem

import confidence
from conftest import paths

pytestmark = pytest.mark.diffdock

NAME = "5S8I_2LY"

# The filenames carry two decimals, so agreement can only be asserted to about that.
ROUNDING = 0.05


def published(sources):
    return {source: confidence.docked(source)[1] for source in sources}


def untouched(directory, shuffle=False):
    os.makedirs(directory, exist_ok=True)
    _, poses = paths(NAME)
    sources = [os.path.basename(path) for path in sorted(poses)]
    molecules = [
        Chem.MolFromMolFile(path, removeHs=False) # pyright: ignore[reportAttributeAccessIssue]
        for path in sorted(poses)
    ]
    pairs = list(zip(sources, molecules))
    if shuffle:
        random.Random(0).shuffle(pairs)
    path = os.path.join(directory, f"{NAME}_poses.sdf")
    confidence.write_poses([molecule for _, molecule in pairs],
                           [source for source, _ in pairs], path)
    return path, [source for source, _ in pairs]


def run_scoring(directory, poses):
    return confidence.score(
        [(NAME, paths(NAME)[0], poses)],
        manifest=os.path.join(directory, "manifest.csv"),
        scores=os.path.join(directory, "scores.csv"),
    )


def test_the_confidence_model_reproduces_the_scores_diffdock_published(tmp_path):
    poses, sources = untouched(str(tmp_path))

    scored = run_scoring(str(tmp_path), poses)

    assert set(scored) == {(NAME, source) for source in sources}
    diffdock = published(sources)
    for source in sources:
        assert scored[(NAME, source)] == pytest.approx(diffdock[source], abs=ROUNDING)


def test_the_most_confident_pose_is_the_one_diffdock_ranked_first(tmp_path):
    poses, sources = untouched(str(tmp_path))

    scored = run_scoring(str(tmp_path), poses)

    best = max(sources, key=lambda source: scored[(NAME, source)])
    assert confidence.docked(best)[0] == 1


def test_the_scores_do_not_depend_on_the_order_of_the_poses(tmp_path):
    ordered, sources = untouched(str(tmp_path / "ordered"))
    shuffled, reordered = untouched(str(tmp_path / "shuffled"), shuffle=True)
    assert reordered != sources

    first = run_scoring(str(tmp_path / "ordered"), ordered)
    second = run_scoring(str(tmp_path / "shuffled"), shuffled)

    for source in sources:
        assert first[(NAME, source)] == pytest.approx(second[(NAME, source)], abs=1e-4)
