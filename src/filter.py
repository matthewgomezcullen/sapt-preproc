"""
Screen the benchmark set, then bin eligible cutouts by the size and charge of its cutout.

The per-complex numbers are written to out/filter.csv. `--reuse` reads that file back and
    reprints the tables without screening again.
"""

import argparse
import bisect
import csv
import os
import re
import statistics
from collections import Counter

from prepare import PrepareComplex, OutOfScopeError, PrepareError

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
DIFFDOCK = os.path.join(DATA, "diffdock")
POSEBUSTERS = os.path.join(DATA, "posebusters")
OUT = os.path.join(ROOT, "out")
TABLE = os.path.join(OUT, "filter.csv")

# DiffDock names each pose it kept rank<N>_confidence<X>.sdf. Alongside those it writes a bare
# rank1.sdf copy of the top-ranked pose and, for some complexes, an energy-minimised
# rank<N>_confidence<X>_ensemble_relaxed.sdf.
POSE = re.compile(r"^rank\d+_confidence-?\d+\.\d+\.sdf$")

FAIL = "confidence-1000"

FIELDS = ["name", "status", "heavy_atoms", "charge", "electrons", "rejection"]

# The numeric columns, which csv hands back as strings.
COUNTS = ["heavy_atoms", "charge", "electrons"]


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


def inventory():
    """
    The complexes on disk, as (name, protein, poses), and the names of those missing a half.

    A complex is screened only where both halves are present: DIFFDOCK also holds bust_results.csv,
        and POSEBUSTERS holds complexes DiffDock was never run on.
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
        proteins, poses = _protein(name), _poses(name)
        if len(proteins) != 1 or not poses:
            incomplete.append(name)
            continue
        complexes.append((name, proteins[0], poses))
    return complexes, incomplete


def _row(name, status, prepared, rejection=""):
    """
    One complex's screening result.
    """
    return {
        "name": name,
        "status": status,
        "heavy_atoms": "" if prepared.heavy_atoms is None else prepared.heavy_atoms,
        "charge": "" if prepared.charge is None else prepared.charge,
        "electrons": "" if prepared.electrons is None else prepared.electrons,
        "rejection": rejection,
    }


def screen(complexes):
    """
    Prepare every complex.

    An OutOfScopeError means the complex is outside the method; a PrepareError means it could not
        be read or prepared.
    """
    rows = []
    eligible = []
    for i, (name, protein, poses) in enumerate(complexes, start=1):
        prepared = PrepareComplex(protein, poses)
        try:
            prepared.prepare()
        except OutOfScopeError as error:
            rows.append(_row(name, "rejected", prepared, error.error_type.value))
        except PrepareError as error:
            rows.append(_row(name, "failed", prepared, str(error)))
        else:
            rows.append(_row(name, "eligible", prepared))
            eligible.append((name, prepared))
        if i % 25 == 0:
            print(f'File {i} screened')
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


def _summarise(rows, incomplete=()):
    """
    How many complexes were looked at, and what became of each.
    """
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
    if incomplete:
        print('Incomplete', len(incomplete), sorted(incomplete))


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
    arguments = parser.parse_args()

    if arguments.reuse:
        # A stored screen carries the numbers but not the objects, which is all the tables need.
        rows = read()
        eligible = []
        _summarise(rows)
    else:
        complexes, incomplete = inventory()
        rows, eligible = screen(complexes)
        write(rows)
        _summarise(rows, incomplete)
    report(rows)
