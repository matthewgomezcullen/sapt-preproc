"""
How well DiffDock's confidence model ranks the poses the screen kept.

Every kept pose was minimised, which moves it, so the confidence DiffDock published no longer
    describes the pose on disk. Each is scored again as it now stands, ranked on that score, and the
    top-ranked one held against the deposited ligand. confidence.csv is a row a pose;
    confidence_summary.csv is a row a complex, carrying both the top-1 and the share of the
    ensemble that is near-native, which is the rate a random pick off it would manage. Both are
    written into the directory of the screen they score, out/filter or out/filter_<name>.

The scoring runs as a subprocess in the interpreter DiffDock was installed into, because DiffDock's
    `utils` package shadows ours and its pins are a different python. DIFFDOCK_PYTHON says where that
    interpreter is. The weights are expected under data/diffdock_models and are never downloaded.

    python confidence.py --complexes 7USH_82V 7R9N_F97
"""

import argparse
import itertools
import os
import subprocess

from rdkit import Chem

import filter
import motivation
from utils import report, save

ROOT = os.path.dirname(os.path.abspath(__file__))

# Only ever run, never imported
SCRIPT = os.path.join(ROOT, "utils", "diffdock.py")

# The screen scored when none is named, and the directory its preparations are kept in.
NAME = filter.NAME
JOB = filter.job_dir()

# The two tables this script produces, written into the directory of the screen it scores.
TABLE_NAME = "confidence.csv"
SUMMARY_NAME = "confidence_summary.csv"

# Input and output, kept there too. Output is left on disk for `--reuse`
MANIFEST_NAME = "confidence_manifest.csv"
SCORES_NAME = "confidence_scores.csv"

PYTHON = os.environ.get("DIFFDOCK_PYTHON")
MODELS = os.environ.get("DIFFDOCK_MODELS", os.path.join(filter.DATA, "diffdock_models"))
ESM = os.environ.get("DIFFDOCK_ESM", os.path.join(MODELS, "esm2_t33_650M_UR50D.pt"))

POSES = "_poses.sdf"

FIELDS = [
    "name", "source", "rank_docked", "confidence_docked", "rank_minimised",
    "confidence_minimised", "rmsd",
]

SUMMARY_FIELDS = [
    "name", "poses", "near_native", "fraction", "top1_minimised", "top1_docked",
    "discrimination_minimised", "discrimination_docked",
]

RANKINGS = [
    ("top1_minimised", "discrimination_minimised", "re-scored by DiffDock's confidence model"),
    ("top1_docked", "discrimination_docked", "as DiffDock ranked them"),
]

SCORE_FIELDS = ["name", "source", "confidence"]

MANIFEST_FIELDS = ["name", "protein", "sdf"]


def complex_dir(name, job=None):
    return os.path.join(job or JOB, name)


def poses_path(name, job=None):
    return os.path.join(complex_dir(name, job), f"{name}{POSES}")


def _prepared(name, job=None):
    """
    The poses a complex's artefact holds and the source file.
    """
    directory = complex_dir(name, job)
    record = save.load_prepared(name, directory)
    if record is None:
        raise FileNotFoundError(f"{name} has no readable preparation in {directory}")
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


def export(name, job=None):
    molecules, sources = _prepared(name, job)
    return write_poses(molecules, sources, poses_path(name, job))


def initial_rows(name, native, job=None):
    molecules, sources = _prepared(name, job)
    return [
        dict(zip(FIELDS, (name, source, *report.docked_rank_and_score(source), None, None, rmsd)))
        for source, rmsd in zip(sources, filter.calc_rmsds(molecules, native))
    ]


def score(work, manifest, scores, python=None, models=None, esm=None):
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
    report.write_table([dict(zip(MANIFEST_FIELDS, one)) for one in work], manifest, MANIFEST_FIELDS)
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
    return {(row["name"], row["source"]): row["confidence"] for row in report.read_table(path)}


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
    return number(
        rows, lambda row: (-row["confidence_minimised"], row["rank_docked"]), "rank_minimised"
    )


def number(rows, order, field):
    """
    Each complex's rows numbered from 1 as `field`, in the order the key `order` gives.
    """
    ordered = sorted(rows, key=lambda row: (row["name"], order(row)))
    numbered = []
    for _, group in itertools.groupby(ordered, key=lambda row: row["name"]):
        for position, row in enumerate(group, start=1):
            numbered.append({**row, field: position})
    return numbered


def is_near_native(row, threshold):
    return None if row["rmsd"] is None else row["rmsd"] <= threshold


def summarise(rows, threshold=filter.NEAR_NATIVE):
    """
    A row a complex: how much of its ensemble is near-native, top-ranked pose correctness, and how
        much of the ensemble each ranking successfully orders.

    `top1_docked` asks the same of the best-ranked pose DiffDock left in the ensemble, which is what
        the re-scoring is measured against. Either is None where that pose's RMSD is missing, as is
        either rate where the complex holds no pair to order.
    """
    summary = []
    for name, group in itertools.groupby(
        sorted(rows, key=lambda row: row["name"]), key=lambda row: row["name"]
    ):
        theirs = list(group)
        near = [is_near_native(row, threshold) for row in theirs]
        found = sum(1 for one in near if one)
        summary.append({
            "name": name,
            "poses": len(theirs),
            "near_native": found,
            "fraction": found / len(theirs),
            "top1_minimised": is_near_native(min(theirs, key=lambda row: row["rank_minimised"]), threshold),
            "top1_docked": is_near_native(min(theirs, key=lambda row: row["rank_docked"]), threshold),
            "discrimination_minimised": report.pairwise_discriminate(
                [row["confidence_minimised"] for row in theirs], near
            ),
            "discrimination_docked": report.pairwise_discriminate(
                [row["confidence_docked"] for row in theirs], near
            ),
        })
    return summary


def _report(summary, skipped=(), missing=()):
    """
    Each ranking's top-1 and pairwise discrimination against the rate a random pick off the
        ensemble would manage.
    """
    print('Complexes', len(summary))
    if skipped:
        print('  no preparation', len(skipped), sorted(skipped))
    if missing:
        print('  poses left unscored', len(missing))
    report.rankings(summary, RANKINGS)


def run(complexes=None, python=None, models=None, esm=None, reuse=False, name=NAME, plot=True):
    """
    Every complex the screen prepared exported, scored, ranked and summarised into the two tables,
        which are written into the screen's directory beside its preparations.

    `reuse` reads back the last pass's scores instead of running it again. `plot` draws the re-scored
        top-1 beside them, as motivation.py's figure.
    """
    rows, work, skipped = [], [], []
    job = filter.job_dir(name)
    for complex, protein, _, native in filter.inventory(complexes)[0]:
        if save.load_prepared(complex, complex_dir(complex, job)) is None:
            skipped.append(complex)
            continue
        work.append((complex, protein, export(complex, job)))
        rows += initial_rows(complex, native, job)
    if not work:
        raise SystemExit(f"No prepared complex under {job}; run filter.py first")

    scores = read_scores(os.path.join(job, SCORES_NAME)) if reuse else score(
        work,
        os.path.join(job, MANIFEST_NAME),
        os.path.join(job, SCORES_NAME),
        python,
        models,
        esm
    )
    rows, missing = join(rows, scores)
    rows = rank(rows)
    summary = summarise(rows)
    report.write_table(rows, os.path.join(job, TABLE_NAME), FIELDS)
    report.write_table(summary, os.path.join(job, SUMMARY_NAME), SUMMARY_FIELDS)
    _report(summary, skipped, missing)
    if plot:
        answered = [
            {**row, "top1": row["top1_minimised"]}
            for row in summary
            if row["top1_minimised"] is not None
        ]
        path, _ = motivation.plot(answered, name, job=True)
        print("\nWrote", os.path.relpath(path, ROOT))
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
        help="Rank and summarise the scores the last pass left in the screen's directory instead "
             "of scoring every pose again.",
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
    parser.add_argument(
        "--name",
        default=NAME,
        help="The screen to score, as it was named for filter.py. Its preparations are read from, "
             "and the tables written into, out/filter_<name>, or out/filter without a name.",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot the confidence using motivation.py's plot function."
    )
    arguments = parser.parse_args()

    run(arguments.complexes,
        arguments.python,
        arguments.models,
        arguments.esm,
        arguments.reuse,
        arguments.name,
        arguments.plot
    )
