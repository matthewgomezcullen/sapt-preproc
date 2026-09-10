"""
Screen the benchmark set, then bin eligible cutouts by the size and charge of its cutout.

The per-complex numbers are written to out/filter.csv. `--reuse` reads that file back and
    reprints the tables without screening again.
"""

import argparse
import bisect
import csv
import os
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from posebusters import PoseBusters, check_rmsd
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

VALIDITY = "dock"

FAR = "protein-ligand_maximum_distance"
NEAR_NATIVE = 2.0

# Why an ensemble leaves the screen before its complex is ever prepared.
GENERATOR = "no near-native pose generated"
VALIDITY_FAILURE = "every near-native pose is physically invalid"

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


def _near_native(poses, native, threshold=NEAR_NATIVE):
    """
    The poses sitting within `threshold` of the deposited ligand.

    RMSD is symmetry-corrected and over heavy atoms.
    """
    crystal = Chem.MolFromMolFile(native) # pyright: ignore[reportAttributeAccessIssue]
    if crystal is None:
        return set()

    near = set()
    for pose in poses:
        docked = Chem.MolFromMolFile(pose) # pyright: ignore[reportAttributeAccessIssue]
        if docked is None:
            continue
        results = check_rmsd(docked, crystal, rmsd_threshold=threshold)["results"]
        if results["rmsd_within_threshold"]:
            near.add(pose)
    return near


def _valid(poses, protein):
    """
    The poses PoseBusters finds physically plausible, and the checks the rest of them failed.

    FAR is dropped. We do not exclude based on the distance to the native pose.
    """
    table = PoseBusters(VALIDITY, max_workers=0).bust(poses, None, protein)
    table = table.drop(columns=FAR, errors="ignore")
    passed = {file for (file, _, _), ok in table.all(axis=1).items() if ok}
    failed = Counter({
        check: int(count) for check, count in (~table).sum().items() if count
    })
    return [pose for pose in poses if pose in passed], failed


def _bust(complex):
    """
    One ensemble, reviewed.

    Runs in a process of its own, so it takes and returns paths and counts.
    """
    name, protein, poses, native = complex
    near = _near_native(poses, native)
    if not near:
        return None, (name, GENERATOR), Counter()

    valid, failed = _valid(poses, protein)
    usable = near.intersection(valid)
    if not usable:
        return None, (name, VALIDITY_FAILURE), failed
    kept = (name, protein, valid, native, len(poses) - len(valid), len(usable))
    return kept, None, failed


def bust_poses(complexes, workers=None):
    """
    Reviews complex poses with PoseBusters.

    Physically implausible poses are excluded and recorded. Sets with no near-native poses are
        rejected.

    Ensembles busted in parallel.

    Returns complexes as (name, protein, poses, native, excluded, near_native), the ensembles
        dropped as (name, why), and the checks every excluded pose failed.
    """
    kept, incorrect, checks = [], [], Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        reviewed = tqdm(
            pool.map(_bust, complexes),
            total=len(complexes),
            desc="Busting",
            unit="complex",
        )
        for one, dropped, failed in reviewed:
            checks.update(failed)
            if dropped:
                incorrect.append(dropped)
            else:
                kept.append(one)
    return kept, incorrect, checks


def _row(name, status, prepared, ensemble, rejection=""):
    """
    One complex's and PoseBusters' screening result.
    """
    poses, excluded, near_native = ensemble
    return {
        "name": name,
        "status": status,
        "heavy_atoms": "" if prepared.heavy_atoms is None else prepared.heavy_atoms,
        "charge": "" if prepared.charge is None else prepared.charge,
        "electrons": "" if prepared.electrons is None else prepared.electrons,
        "poses": poses,
        "excluded": excluded,
        "near_native": near_native,
        "rejection": rejection,
    }


def screen(complexes):
    """
    Prepare every complex.

    An OutOfScopeError means the complex is outside the method; a PrepareError means it could not
        be read or prepared.

    TODO: drop `eligiblel`
    """
    rows = []
    eligible = []
    for name, protein, poses, _, excluded, near_native in tqdm(
        complexes, desc="Screening", unit="complex"
    ):
        ensemble = (len(poses), excluded, near_native)
        prepared = PrepareComplex(protein, poses)
        try:
            prepared.prepare()
        except OutOfScopeError as error:
            rows.append(_row(name, "rejected", prepared, ensemble, error.error_type.value))
        except PrepareError as error:
            rows.append(_row(name, "failed", prepared, ensemble, str(error)))
        else:
            rows.append(_row(name, "eligible", prepared, ensemble))
            eligible.append((name, prepared))
    return rows, eligible


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
    if incorrect:
        print('Busted', len(incorrect))
        for why, count in Counter(why for _, why in incorrect).most_common():
            print(f'  {count:4d}  {why}')
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
        print(f'\nChecks failed, by pose. {FAR} is not among them; it is not held against a pose\n')
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
        help="Processes to review the ensembles with. Defaults to the machine's cores, which under "
             "a scheduler is the node's rather than what the job was given.",
    )
    arguments = parser.parse_args()

    if arguments.reuse:
        rows = read()
        eligible = []
        _summarise(rows)
    else:
        complexes, incomplete = inventory()
        complexes = complexes[:10]
        complexes, incorrect, checks = bust_poses(complexes, workers=arguments.workers)
        rows, eligible = screen(complexes)
        write(rows)
        _summarise(rows, incomplete, incorrect, checks)
    report(rows)
