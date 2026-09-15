"""
Evaluate DiffDock's original ranking across all complexes and poses.

out/motivation.csv is a row a complex; out/motivation.png is the figure.

    python motivation.py
"""

import argparse
import bisect
import csv
import os
import re
import statistics
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

import filter

ROOT = filter.ROOT

TABLE = os.path.join(filter.OUT, "motivation.csv")
FIGURE = os.path.join(filter.OUT, "motivation.png")

# DiffDock names a scored pose rank<N>_confidence<X>.sdf
RANKED = re.compile(r"^rank(\d+)_confidence(-?\d+\.\d+)\.sdf$")

FIELDS = ["name", "poses", "near_native", "fraction", "top1", "rank_top1", "rmsd_top1"]

# csv columns
INTEGERS = ["poses", "near_native", "rank_top1"]
DECIMALS = ["fraction", "rmsd_top1"]
BOOLEANS = ["top1"]

# The five bands.
EDGES = [0.25, 0.50, 0.75]
LABELS = ["0%", "0-25%", "25-50%", "50-75%", "75-100%"]


def rank_of(path):
    matched = RANKED.match(os.path.basename(path))
    if matched is None:
        raise ValueError(f"{path} is not a pose DiffDock scored")
    return int(matched.group(1))


def _row(one):
    name, _, paths, native = one
    paths = sorted(paths, key=rank_of)
    measured = filter.rmsds(paths, native)
    near = [rmsd is not None and rmsd <= filter.NEAR_NATIVE for rmsd in measured]
    return {
        "name": name,
        "poses": len(paths),
        "near_native": sum(near),
        "fraction": sum(near) / len(paths),
        "top1": near[0],
        "rank_top1": rank_of(paths[0]),
        "rmsd_top1": measured[0],
    }


def band_of(fraction):
    return 0 if not fraction else bisect.bisect_left(EDGES, fraction) + 1


def bin(rows):
    bins = [[] for _ in LABELS]
    for row in rows:
        bins[band_of(row["fraction"])].append(row)
    return bins


def rates(bins):
    def mean(group, key):
        return statistics.mean(row[key] for row in group) if group else float("nan")

    return (
        [mean(group, "top1") for group in bins],
        [mean(group, "fraction") for group in bins],
        [len(group) for group in bins],
    )


def plot(rows, path=FIGURE):
    """
    Two plots over the same five bands: top-1 success against what a random pick would manage, and
        how many complexes each band holds, which is what says how much the top panel is worth.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    success, chance, counts = rates(bin(rows))
    positions = range(len(LABELS))

    figure, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(7, 6), height_ratios=[2, 1]
    )

    top.bar(positions, success, color="#4C72B0", label="DiffDock's top-ranked pose")
    top.plot(positions, chance, "o--", color="#C44E52", label="a random pick off the ensemble")
    top.set_ylim(0, 1)
    top.set_ylabel("near-native top-1")
    top.yaxis.set_major_formatter(PercentFormatter(1.0))
    top.legend(frameon=False)

    bottom.bar(positions, counts, color="#BBBBBB")
    bottom.set_ylabel("complexes")
    bottom.set_xlabel(
        f"share of the ensemble within {filter.NEAR_NATIVE} Å of the deposited ligand"
    )
    bottom.set_xticks(list(positions), LABELS)

    figure.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, dpi=200)
    return path


def write(rows, path=TABLE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read(path=TABLE):
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


def _report(rows, incomplete=()):
    print("Complexes", len(rows))
    if incomplete:
        print("  incomplete", len(incomplete), sorted(incomplete))

    poses = sum(row["poses"] for row in rows)
    near = sum(row["near_native"] for row in rows)
    print(f"Poses {poses}, {near} near-native ({near / poses:.1%})")

    success, chance, counts = rates(bin(rows))
    print("\nTop-1 by how much of the ensemble is near-native\n")
    print(f'  {"band":>8s}{"n":>6s}{"top-1":>9s}{"chance":>9s}')
    for label, count, hit, random_pick in zip(LABELS, counts, success, chance):
        print(f"  {label:>8s}{count:6d}{hit:9.1%}{random_pick:9.1%}")
    print(f'  {"all":>8s}{len(rows):6d}'
          f'{statistics.mean(row["top1"] for row in rows):9.1%}'
          f'{statistics.mean(row["fraction"] for row in rows):9.1%}')


def run(complexes=None, reuse=False):
    if reuse:
        rows, incomplete = read(), ()
    else:
        found, incomplete = filter.inventory(complexes)
        if not found:
            raise SystemExit("No complex has both an ensemble and a deposited ligand")
        rows = [_row(_found) for _found in found]
        write(rows)
    _report(rows, incomplete)
    print("\nWrote", os.path.relpath(plot(rows), ROOT))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--complexes",
        nargs="+",
        default=None,
        help="The complexes to measure, by name. Every one by default.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help=f"Plot the measurements already in {os.path.relpath(TABLE, ROOT)} instead of measuring "
             "every pose again.",
    )
    arguments = parser.parse_args()

    run(arguments.complexes, arguments.reuse)
