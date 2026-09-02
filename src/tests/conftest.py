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
        "--encode",
        action="store_true",
        help="Run only the encoding tests, for a change that touches nothing before them.",
    )
    parser.addoption(
        "--long-protonate",
        action="store_true",
        help="Also run the tests that carry complexes through the repair and protonation "
             "pipelines. Minutes each, because protonation is seeded onto a reference platform "
             "for determinism.",
    )
    parser.addoption(
        "--hpc",
        action="store_true",
        help="Also run the tests that solve a real SCF over a subset cutout and correlate it. "
             "About an hour a cutout cold, seconds against a checkpoint.",
    )
    parser.addoption(
        "--hpc-long-stab",
        action="store_true",
        help="Also run the stability analysis over a subset cutout's converged SCF. Several hours "
             "each, and true once per cutout rather than once per run.",
    )
    parser.addoption(
        "--hpc-long-dice",
        action="store_true",
        help="Also run Dice over the fifty orbitals MP2 leaves on a subset cutout.",
    )
    parser.addoption(
        "--hpc-long-run",
        action="store_true",
        help="Also run the driver end to end over a subset cutout, which repeats AVAS, MP2 and "
             "Dice rather than sharing the cached ones.",
    )


OPTIONAL = {
    "long_protonate": "--long-protonate",
    "hpc": "--hpc",
    "hpc_long_stab": "--hpc-long-stab",
    "hpc_long_dice": "--hpc-long-dice",
    "hpc_long_run": "--hpc-long-run",
}


def pytest_configure(config):
    for marker, flag in OPTIONAL.items():
        config.addinivalue_line("markers", f"{marker}: long; run only under {flag}")
    config.addinivalue_line(
        "markers", "dice: needs the Dice executable; skipped wherever it is not on the PATH"
    )


# The encoding tests are a directory now, so the flag selects on that rather than a filename.
ENCODING = "encode"


def pytest_collection_modifyitems(config, items):
    """
    Every long-running mark is opt-in, so a bare run is the quick one and nothing has to be
        remembered to keep it that way.

    `--encode` It narrows to the encoding tests, which are deselected rather than skipped.

    For Dice, the tests that need it are skipped wherever it is absent and run wherever it is
        there, without a flag either way.
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

    for marker, flag in OPTIONAL.items():
        if config.getoption(flag):
            continue
        skipped = pytest.mark.skip(reason=f"needs {flag}")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skipped)
