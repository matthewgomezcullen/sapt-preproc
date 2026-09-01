import os
import re
import shutil
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))

# src/, so the tests can import the modules under test. Then this directory, because the tests
# sit a level below it now and pytest only puts their own directory on the path.
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

DATA = os.path.join(HERE, "data")
POSEBUSTERS = os.path.join(DATA, "posebusters")
DIFFDOCK = os.path.join(DATA, "diffdock")

POSE_PATTERN = re.compile(r"rank\d+_confidence-?[\d.]+\.sdf")
DROPPED_CONFIDENCE = "confidence-1000.00.sdf"


def paths(name):
    """
    Resolve a complex to its protein PDB and its candidate poses, dropping the failed-confidence
        poses that README excludes.
    """
    standalone = os.path.join(DATA, name)
    if os.path.isdir(os.path.join(standalone, "poses")):
        protein = os.path.join(standalone, f"{name}_protein.pdb")
        poses_dir = os.path.join(standalone, "poses")
    else:
        protein = os.path.join(POSEBUSTERS, name, f"{name}_protein.pdb")
        poses_dir = os.path.join(DIFFDOCK, name)
    poses = sorted(
        os.path.join(poses_dir, f)
        for f in os.listdir(poses_dir)
        if POSE_PATTERN.fullmatch(f) and not f.endswith(DROPPED_CONFIDENCE)
    )
    return protein, poses


def pytest_addoption(parser):
    parser.addoption(
        "--fast",
        action="store_true",
        help="Skip the tests marked slow, which are the ones that carry several complexes through "
             "the repair and protonation pipelines.",
    )
    parser.addoption(
        "--hpc",
        action="store_true",
        help="Also run the tests marked hpc, which solve a real SCF over a subset cutout. One Fock "
             "build of the smallest is around four minutes on twelve cores, so these are hours "
             "each and are left out of every ordinary run.",
    )
    parser.addoption(
        "--encode",
        action="store_true",
        help="Run only the encoding tests, for a change that touches nothing before them.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: carries complexes through the preparation pipeline; skipped by --fast"
    )
    config.addinivalue_line(
        "markers", "hpc: solves a full SCF over a subset cutout; run only under --hpc"
    )
    config.addinivalue_line(
        "markers", "dice: needs the Dice executable; skipped wherever it is not on the PATH"
    )


# The encoding tests are a directory now, so the flag selects on that rather than a filename.
ENCODING = "encode"


def pytest_collection_modifyitems(config, items):
    """
    `slow` is opt-out and `hpc` is opt-in.

    A slow test is a minute of preparation, an hpc test is hours of SCF.

    Dice is an external program that only builds on Linux

    `--encode` narrows to the encoding tests, which are deselected rather than skipped so that the
        run reports only what it was asked for.
    """
    if config.getoption("--encode"):
        selected = [item for item in items if ENCODING in item.path.parts]
        deselected = [item for item in items if ENCODING not in item.path.parts]
        if deselected:
            config.hook.pytest_deselected(items=deselected)
        items[:] = selected

    if shutil.which("Dice") is None:
        skipped = pytest.mark.skip(reason="Dice is not on the PATH")
        for item in items:
            if "dice" in item.keywords:
                item.add_marker(skipped)

    if not config.getoption("--hpc"):
        skipped = pytest.mark.skip(reason="needs --hpc")
        for item in items:
            if "hpc" in item.keywords:
                item.add_marker(skipped)

    if not config.getoption("--fast"):
        return
    skipped = pytest.mark.skip(reason="skipped by --fast")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skipped)
