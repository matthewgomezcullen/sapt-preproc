"""
How well DiffDock's confidence model ranks the poses the screen kept.

Every kept pose was minimised, which moves it, so the confidence DiffDock published no longer
    describes the pose on disk. Each is scored again as it now stands, ranked on that score, and the
    top-ranked one held against the deposited ligand. out/confidence.csv is a row a pose;
    out/confidence_summary.csv is a row a complex, carrying both the top-1 and the share of the
    ensemble that is near-native, which is the rate a random pick off it would manage.

The scoring runs as a subprocess in the interpreter DiffDock was installed into, because DiffDock's
    `utils` package shadows ours and its pins are a different python. DIFFDOCK_PYTHON says where that
    interpreter is. The weights are expected under data/diffdock_models and are never downloaded.

    python confidence.py --complexes 7USH_82V 7R9N_F97
"""

import argparse
import csv
import itertools
import os
import re
import statistics
import subprocess

from rdkit import Chem

import filter
from utils import save

ROOT = os.path.dirname(os.path.abspath(__file__))

# Only ever run, never imported
SCRIPT = os.path.join(ROOT, "utils", "diffdock.py")

# The complexes the screen prepared, and the two tables this script produces.
JOB = filter.job_dir()
TABLE = os.path.join(filter.OUT, "confidence.csv")
SUMMARY = os.path.join(filter.OUT, "confidence_summary.csv")

# Input and output. Output is left on disk for `--reuse`
MANIFEST = os.path.join(filter.OUT, "confidence_manifest.csv")
SCORES = os.path.join(filter.OUT, "confidence_scores.csv")

PYTHON = os.environ.get("DIFFDOCK_PYTHON")
MODELS = os.environ.get("DIFFDOCK_MODELS", os.path.join(filter.DATA, "diffdock_models"))
ESM = os.environ.get("DIFFDOCK_ESM", os.path.join(MODELS, "esm2_t33_650M_UR50D.pt"))

POSES = "_poses.sdf"

FIELDS = [
    "name", "source", "rank_docked", "confidence_docked", "rank_minimised",
    "confidence_minimised", "rmsd",
]

SUMMARY_FIELDS = ["name", "poses", "near_native", "fraction", "top1_minimised", "top1_docked"]

SCORE_FIELDS = ["name", "source", "confidence"]

MANIFEST_FIELDS = ["name", "protein", "sdf"]

# csv hands every column back as a string.
INTEGERS = ["rank_docked", "rank_minimised", "poses", "near_native"]
DECIMALS = ["confidence_docked", "confidence_minimised", "rmsd", "fraction", "confidence"]
BOOLEANS = ["top1_minimised", "top1_docked"]

# DiffDock names a scored pose rank<N>_confidence<X>.sdf.
SCORED = re.compile(r"^rank(\d+)_confidence(-?\d+\.\d+)\.sdf$")


def complex_dir(name):
    return os.path.join(JOB, name)


def poses_path(name):
    return os.path.join(complex_dir(name), f"{name}{POSES}")


def docked_rank_and_score(source):
    scored = SCORED.match(source)
    if scored is None:
        raise ValueError(f"{source} is not a pose DiffDock scored")
    return int(scored.group(1)), float(scored.group(2))


def _prepared(name):
    """
    The poses a complex's artefact holds and the source file.
    """
    record = save.load_prepared(name, complex_dir(name))
    if record is None:
        raise FileNotFoundError(f"{name} has no readable preparation in {complex_dir(name)}")
    return (
        [
            Chem.MolFromMolBlock(str(block), removeHs=False) # pyright: ignore[reportAttributeAccessIssue]
            for block in record["poses"]
        ],
        [str(source) for source in record["source"]],
    )


def write_poses(molecules, sources, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with Chem.SDWriter(path) as writer: # pyright: ignore[reportAttributeAccessIssue]
        for molecule, source in zip(molecules, sources):
            titled = Chem.Mol(molecule) # pyright: ignore[reportAttributeAccessIssue]
            titled.SetProp("_Name", source)
            writer.write(titled)
    return path


def export(name):
    molecules, sources = _prepared(name)
    return write_poses(molecules, sources, poses_path(name))


def initial_rows(name, native):
    molecules, sources = _prepared(name)
    return [
        dict(zip(FIELDS, (name, source, *docked_rank_and_score(source), None, None, rmsd)))
        for source, rmsd in zip(sources, filter.calc_rmsds(molecules, native))
    ]


def score(work, python=None, models=None, esm=None, manifest=MANIFEST, scores=SCORES):
    """
    Every complex's poses scored by the confidence model in one pass over `work`, which is a
        (complex, deposited PDB, exported SDF) triple each.

    Keyed by (complex, source) on the way back.
    """
    python, models, esm = python or PYTHON, models or MODELS, esm or ESM
    if not python:
        raise SystemExit(
            "DIFFDOCK_PYTHON is not set. It is the interpreter DiffDock is installed into, which "
            "cannot be this one: DiffDock's `utils` package shadows ours."
        )
    write_table([dict(zip(MANIFEST_FIELDS, one)) for one in work], manifest, MANIFEST_FIELDS)
    subprocess.run(
        [
            python, SCRIPT,
            "--manifest", manifest,
            "--scores", scores,
            "--models", models,
            "--esm", esm,
        ],
        check=True,
    )
    return read_scores(scores)


def read_scores(path):
    return {(row["name"], row["source"]): row["confidence"] for row in read_table(path)}


def join(rows, scores):
    """
    Each pose given the confidence. A pose left unscored is dropped and reported.
    """
    joined, missing = [], []
    for row in rows:
        pose = (row["name"], row["source"])
        if pose in scores:
            joined.append({**row, "confidence_minimised": scores[pose]})
        else:
            missing.append(pose)
    return joined, missing


def rank(rows):
    """
    A tie goes to the pose DiffDock ranked better.
    """
    ordered = sorted(
        rows, key=lambda row: (row["name"], -row["confidence_minimised"], row["rank_docked"])
    )
    ranked = []
    for _, group in itertools.groupby(ordered, key=lambda row: row["name"]):
        for position, row in enumerate(group, start=1):
            ranked.append({**row, "rank_minimised": position})
    return ranked


def _is_near_native(row, threshold):
    return None if row["rmsd"] is None else row["rmsd"] <= threshold


def summarise(rows, threshold=filter.NEAR_NATIVE):
    """
    A row a complex: how much of its ensemble is near-native, and top-ranked pose correctness.

    `top1_docked` asks the same of the best-ranked pose DiffDock left in the ensemble, which is what
        the re-scoring is measured against. Either is None where that pose's RMSD is missing.
    """
    summary = []
    for name, group in itertools.groupby(
        sorted(rows, key=lambda row: row["name"]), key=lambda row: row["name"]
    ):
        theirs = list(group)
        near = [row for row in theirs if _is_near_native(row, threshold)]
        summary.append({
            "name": name,
            "poses": len(theirs),
            "near_native": len(near),
            "fraction": len(near) / len(theirs),
            "top1_minimised": _is_near_native(min(theirs, key=lambda row: row["rank_minimised"]), threshold),
            "top1_docked": _is_near_native(min(theirs, key=lambda row: row["rank_docked"]), threshold),
        })
    return summary


def write_table(rows, path, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_table(path):
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for field, value in row.items():
            if value == "":
                row[field] = None
            elif field in INTEGERS:
                row[field] = int(value)
            elif field in DECIMALS:
                row[field] = float(value)
            elif field in BOOLEANS:
                row[field] = value == "True"
    return rows


def _report(summary, skipped=(), missing=()):
    """
    Top-1 against the rate a random pick off the ensemble would manage.
    """
    print('Complexes', len(summary))
    if skipped:
        print('  no preparation', len(skipped), sorted(skipped))
    if missing:
        print('  poses left unscored', len(missing))

    answered = [row for row in summary if row["top1_minimised"] is not None]
    if not answered:
        return
    rescored = sum(1 for row in answered if row["top1_minimised"])
    published = sum(1 for row in answered if row["top1_docked"])
    chance = statistics.mean(row["fraction"] for row in answered)
    print(f'\nTop-1 over {len(answered)} complexes\n')
    print(f'  {rescored:4d}  re-scored, {rescored / len(answered):.1%}')
    print(f'  {published:4d}  as DiffDock ranked them, {published / len(answered):.1%}')
    print(f'        a random pick off the ensemble, {chance:.1%}')


def run(complexes=None, python=None, models=None, esm=None, reuse=False):
    """
    Every prepared complex exported, scored, ranked and summarised into the two tables.

    `reuse` reads back the last pass's scores instead of running it again.
    """
    rows, work, skipped = [], [], []
    for name, protein, _, native in filter.inventory(complexes)[0]:
        if save.load_prepared(name, complex_dir(name)) is None:
            skipped.append(name)
            continue
        work.append((name, protein, export(name)))
        rows += initial_rows(name, native)
    if not work:
        raise SystemExit(f"No prepared complex under {JOB}; run filter.py first")

    scores = read_scores(SCORES) if reuse else score(work, python, models, esm)
    rows, missing = join(rows, scores)
    rows = rank(rows)
    summary = summarise(rows)
    write_table(rows, TABLE, FIELDS)
    write_table(summary, SUMMARY, SUMMARY_FIELDS)
    _report(summary, skipped, missing)
    return rows, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--complexes",
        nargs="+",
        default=None,
        help="The complexes to score, by name. Every prepared one by default.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help=f"Rank and summarise the scores already in {os.path.relpath(SCORES, ROOT)} instead of "
             "scoring every pose again.",
    )
    parser.add_argument(
        "--python",
        default=PYTHON,
        help="The interpreter DiffDock is installed into. DIFFDOCK_PYTHON by default.",
    )
    parser.add_argument(
        "--models",
        default=MODELS,
        help="The directory the released weights were unpacked into.",
    )
    parser.add_argument(
        "--esm",
        default=ESM,
        help="The ESM-2 650M weights.",
    )
    arguments = parser.parse_args()

    run(arguments.complexes, arguments.python, arguments.models, arguments.esm, arguments.reuse)
