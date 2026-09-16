"""
The scoring pass itself, for utils/diffdock.py.

It runs in the interpreter DiffDock was installed into, so these are skipped unless DIFFDOCK_PYTHON
    says where that is. Three sets of weights have to be on disk already, under data/diffdock_models:
    the confidence model, its model_parameters.yml, and esm2_t33_650M_UR50D.pt.

We check that a pose handed to the model as coordinates is the pose DiffDock would have built, 
    proven by its ordering rather than its scores. The scores cannot match because these poses are 
    PoseBench's, docked into an ESMFold structure aligned onto the crystal binding site, and they 
    are read here in the deposited crystal instead. 

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

# Spearman correlation threshold on ordering.
ORDERING = 0.95

# The two pockets agree to 0.03 on the pose DiffDock ranked first.
TOP = 0.1


def published(sources):
    return {source: confidence.docked_rank_and_score(source)[1] for source in sources}


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


def test_the_confidence_model_orders_the_poses_as_diffdock_ordered_them(tmp_path):
    from scipy.stats import spearmanr

    poses, sources = untouched(str(tmp_path))

    scored = run_scoring(str(tmp_path), poses)

    assert set(scored) == {(NAME, source) for source in sources}
    diffdock = published(sources)
    ordering = spearmanr(
        [scored[(NAME, source)] for source in sources], [diffdock[source] for source in sources]
    ).correlation
    assert ordering > ORDERING
    best = min(sources, key=lambda source: confidence.docked_rank_and_score(source)[0])
    assert scored[(NAME, best)] == pytest.approx(diffdock[best], abs=TOP)


def test_the_most_confident_pose_is_the_one_diffdock_ranked_first(tmp_path):
    poses, sources = untouched(str(tmp_path))

    scored = run_scoring(str(tmp_path), poses)

    best = max(sources, key=lambda source: scored[(NAME, source)])
    assert confidence.docked_rank_and_score(best)[0] == 1


def test_the_scores_do_not_depend_on_the_order_of_the_poses(tmp_path):
    ordered, sources = untouched(str(tmp_path / "ordered"))
    shuffled, reordered = untouched(str(tmp_path / "shuffled"), shuffle=True)
    assert reordered != sources

    first = run_scoring(str(tmp_path / "ordered"), ordered)
    second = run_scoring(str(tmp_path / "shuffled"), shuffled)

    for source in sources:
        assert first[(NAME, source)] == pytest.approx(second[(NAME, source)], abs=1e-4)
