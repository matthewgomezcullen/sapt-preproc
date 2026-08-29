import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
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


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: carries complexes through the preparation pipeline; skipped by --fast"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--fast"):
        return
    skipped = pytest.mark.skip(reason="skipped by --fast")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skipped)
