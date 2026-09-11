"""
Screen the benchmark set, then bin eligible cutouts by the size and charge of its cutout.

The per-complex numbers are written to out/filter.csv. `--reuse` reads that file back and
    reprints the tables without screening again.

Each complex's preparation is kept in out/filter/<complex> and read back by the next screen.
    `--force` prepares every complex again.
"""

import argparse
import bisect
import csv
import os
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from posebusters import check_rmsd
from rdkit import Chem
from tqdm import tqdm

from prepare import PrepareComplex, OutOfScopeError, PrepareError
from run import FAIL, POSE

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
DIFFDOCK = os.path.join(DATA, "diffdock")
POSEBUSTERS = os.path.join(DATA, "posebusters")
OUT = os.path.join(ROOT, "out")
TABLE = os.path.join(OUT, "filter.csv")
JOB = os.path.join(OUT, "filter")

NEAR_NATIVE = 2.0

# Minimisation moves a pose a little, so nothing outside this can come back inside NEAR_NATIVE.
LOOSE = 3.5

GENERATOR = "no near-native pose generated"

FIELDS = [
    "name", "status", "heavy_atoms", "charge", "electrons", "poses", "excluded", "near_native",
    "rejection",
]

# The numeric columns, which csv hands back as strings.
COUNTS = ["heavy_atoms", "charge", "electrons", "poses", "excluded", "near_native"]


def _protein(name):
    """
    The single deposited structure of a PoseBusters complex.
    """
    directory = os.path.join(POSEBUSTERS, name)
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.endswith(".pdb")
    ]


def _poses(name):
    """
    The candidate poses DiffDock produced for a complex, one per rank.
    """
    directory = os.path.join(DIFFDOCK, name)
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if POSE.match(f) and FAIL not in f
    ]


def _native(name):
    """
    The native ligand pose.
    """
    directory = os.path.join(POSEBUSTERS, name)
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.endswith("_ligand.sdf")
    ]


def inventory():
    """
    Return complexes on disk (name, protein, poses, native)
    """
    for directory in (POSEBUSTERS, DIFFDOCK):
        if not os.path.isdir(directory):
            raise SystemExit(
                f"{directory} is missing. The benchmark set is not tracked; see README.md for "
                "where to download it."
            )

    names = sorted(
        name
        for name in set(os.listdir(POSEBUSTERS)) & set(os.listdir(DIFFDOCK))
        if os.path.isdir(os.path.join(POSEBUSTERS, name))
        and os.path.isdir(os.path.join(DIFFDOCK, name))
    )

    complexes = []
    incomplete = []
    for name in names:
        proteins, poses, natives = _protein(name), _poses(name), _native(name)
        if len(proteins) != 1 or not poses or len(natives) != 1:
            incomplete.append(name)
            continue
        complexes.append((name, proteins[0], poses, natives[0]))
    return complexes, incomplete


def near_native(poses, native, threshold=NEAR_NATIVE):
    """
    The poses sitting within `threshold` of the deposited ligand, given as paths or as molecules.

    RMSD is symmetry-corrected and over heavy atoms.
    """
    crystal = Chem.MolFromMolFile(native) # pyright: ignore[reportAttributeAccessIssue]
    if crystal is None:
        return []

    near = []
    for pose in poses:
        docked = (
            Chem.MolFromMolFile(pose) # pyright: ignore[reportAttributeAccessIssue]
            if isinstance(pose, str)
            else pose
        )
        if docked is None:
            continue
        if check_rmsd(docked, crystal, rmsd_threshold=threshold)["results"][
            "rmsd_within_threshold"
        ]:
            near.append(pose)
    return near


def sweep_for_near_native(complexes):
    kept, incorrect = [], []
    for one in tqdm(complexes, desc="Near-Native", unit="complex"):
        name, _, poses, native = one
        if near_native(poses, native, LOOSE):
            kept.append(one)
        else:
            incorrect.append((name, GENERATOR))
    return kept, incorrect


def _row(name, status, prepared, near, rejection=""):
    """
    One complex's screening result.
    """
    return {
        "name": name,
        "status": status,
        "heavy_atoms": "" if prepared.heavy_atoms is None else prepared.heavy_atoms,
        "charge": "" if prepared.charge is None else prepared.charge,
        "electrons": "" if prepared.electrons is None else prepared.electrons,
        "poses": len(prepared.poses) if prepared.poses else "",
        "excluded": "" if prepared.excluded is None else prepared.excluded,
        "near_native": near,
        "rejection": rejection,
    }


def _prepare(one, force=False):
    """
    One complex prepared, or read back, and then checked for near-native existence.

    Can run in its own process.
    """
    name, protein, poses, native = one
    prepared = PrepareComplex(protein, poses, os.path.join(JOB, name))
    try:
        if force or not prepared.prepared():
            prepared.prepare()
    except OutOfScopeError as error:
        return _row(name, "rejected", prepared, 0, error.error_type.value), prepared.failed
    except PrepareError as error:
        return _row(name, "failed", prepared, 0, str(error)), prepared.failed

    near = len(near_native(prepared.poses, native))
    if not near:
        return _row(name, "unusable", prepared, near, GENERATOR), prepared.failed
    return _row(name, "eligible", prepared, near), prepared.failed


def screen(complexes, force=False):
    """
    Prepare every complex.

    An OutOfScopeError means the complex is outside the method; a PrepareError means it could not
        be read or prepared; `unusable` means the ensemble holds no near-native pose.
    """
    rows, checks = [], Counter()
    prepared = tqdm(
        (_prepare(complex, force=force) for complex in complexes),
        total=len(complexes),
        desc="Screening",
        unit="complex",
    )
    for row, failed in prepared:
        rows.append(row)
        checks.update(failed)
    return rows, checks



def screen_parallel(complexes, workers=None, force=False):
    """
    Parallelised. `map` keeps the rows in the order the complexes came in.
    """
    rows, checks = [], Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        prepared = tqdm(
            pool.map(partial(_prepare, force=force), complexes),
            total=len(complexes),
            desc="Screening",
            unit="complex",
        )
        for row, failed in prepared:
            rows.append(row)
            checks.update(failed)
    return rows, checks


def write(rows, path=TABLE):
    """
    Store the screening results, creating the output directory if it is not there yet.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read(path=TABLE):
    """
    A stored screen, with the numeric columns back as ints and an absent value as None.
    """
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for field in COUNTS:
            row[field] = int(row[field]) if row[field] else None
    return rows


def _summarise(rows, incomplete=(), incorrect=(), checks=()):
    counted = Counter(row["status"] for row in rows)
    print('Screened', len(rows))
    print('Eligible', counted["eligible"])
    print('Rejected', counted["rejected"])
    rejections = Counter(row["rejection"] for row in rows if row["status"] == "rejected")
    for rejection, count in rejections.most_common():
        print(f'  {count:4d}  {rejection}')
    print('Failed', counted["failed"])
    for row in rows:
        if row["status"] == "failed":
            print(f'  {row["name"]}: {row["rejection"]}')
    unusable = counted["unusable"] + len(incorrect)
    if unusable:
        print('Unusable', unusable, f'({GENERATOR})')
    if incomplete:
        print('Incomplete', len(incomplete), sorted(incomplete))

    eligible = [row for row in rows if row["status"] == "eligible"]
    counted = [row for row in eligible if row["poses"] is not None]
    if counted:
        kept = sum(row["poses"] for row in counted)
        excluded = sum(row["excluded"] for row in counted)
        near = sum(row["near_native"] for row in counted)
        print(f'\nPoses {kept + excluded} over {len(counted)} eligible complexes')
        print(f'  {excluded:4d}  excluded by PoseBusters')
        print(f'  {kept:4d}  kept, {near} of them near-native')

    if checks:
        print('\nChecks failed, by pose\n')
        for check, count in Counter(checks).most_common():
            print(f'  {count:5d}  {check}')


def quartiles(sizes):
    """
    The three cut points splitting `sizes` into quarters, each boundary inclusive at the top.
    """
    return statistics.quantiles(sizes, n=4, method="inclusive")


def _band(charge):
    """
    Which magnitude band a charge falls in. Neutral, mildly charged, and the cases where the SCF is
        expected to be hard.
    """
    magnitude = abs(charge)
    if not magnitude:
        return 0
    return 1 if magnitude <= 2 else 2


def report(rows):
    """
    How the eligible cutouts distribute over size, against charge.
    """
    eligible = [row for row in rows if row["status"] == "eligible"]
    if len(eligible) < 4:
        print(f'\nToo few eligible complexes ({len(eligible)}) to bin')
        return

    cuts = quartiles([row["heavy_atoms"] for row in eligible])
    bins = [[] for _ in range(4)]
    for row in eligible:
        bins[bisect.bisect_left(cuts, row["heavy_atoms"])].append(row)

    print(f'\nSize quartiles of the {len(eligible)} eligible cutouts, by heavy atoms in the '
          f'capped cutout\n')
    print(f'  {"bin":5s}{"heavy atoms":>14s}{"n":>6s}{"q=0":>6s}{"q!=0":>6s}')
    for label, binned in list(zip(["Q1", "Q2", "Q3", "Q4"], bins)) + [("all", eligible)]:
        sizes = [row["heavy_atoms"] for row in binned]
        neutral = sum(1 for row in binned if row["charge"] == 0)
        span = f'{min(sizes)} - {max(sizes)}'
        print(f'  {label:5s}{span:>14s}{len(binned):6d}{neutral:6d}{len(binned) - neutral:6d}')

    print('\nCharge magnitude\n')
    print(f'  {"bin":5s}{"|q|=0":>8s}{"|q|=1-2":>10s}{"|q|>=3":>9s}')
    for label, binned in list(zip(["Q1", "Q2", "Q3", "Q4"], bins)) + [("all", eligible)]:
        banded = Counter(_band(row["charge"]) for row in binned)
        print(f'  {label:5s}{banded[0]:8d}{banded[1]:10d}{banded[2]:9d}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse",
        action="store_true",
        help=f"Bin a stored screen from {os.path.relpath(TABLE, ROOT)} instead of preparing every "
             "complex again.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Processes to prepare the complexes with. `-1` defaults to the machine's cores, which "
             "under a scheduler is the node's rather than what the job was given. None avoids "
             "parallelisation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"Prepare every complex again, even one kept in {os.path.relpath(JOB, ROOT)}.",
    )
    arguments = parser.parse_args()

    if arguments.reuse:
        rows = read()
        _summarise(rows)
    else:
        complexes, incomplete = inventory()
        complexes = complexes[:10]
        complexes, incorrect = sweep_for_near_native(complexes)
        if arguments.workers is None:
            rows, checks = screen(complexes, force=arguments.force)
        else:
            # ProcessPoolExecutor reads None, not -1, as every core.
            workers = None if arguments.workers == -1 else arguments.workers
            rows, checks = screen_parallel(complexes, workers=workers, force=arguments.force)
        write(rows)
        _summarise(rows, incomplete, incorrect, checks)
    report(rows)
