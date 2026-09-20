"""
Evaluate DiffDock's original ranking across all complexes and poses.

out/motivation.csv is a row a complex; out/motivation.png is the figure.

    python motivation.py
"""

import argparse
import bisect
import os
import statistics
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

import filter
from utils import report

ROOT = filter.ROOT

NAME_PREFIX = "motivation"

FIELDS = [
    "name", "poses", "near_native", "fraction", "top1", "rank_top1", "rmsd_top1", "discrimination",
]

# The five/six bands.
# EDGES = [0.19, 0.39, 0.59, 0.79]
# LABELS = ["0%", "20%", "40%", "60%", "80%", "100%"]
EDGES = [0.25, 0.50, 0.75]
LABELS = ["0%", "25%", "50%", "75%", "100%"]


def motivation_path(name, job=False):
    out = filter.job_dir(name) if job else filter.OUT
    file_base = f"{NAME_PREFIX}_{name}" if name else NAME_PREFIX
    return os.path.join(out, file_base)


def _published(path):
    return report.docked_rank_and_score(os.path.basename(path))


def rank_of(path):
    return _published(path)[0]


def _row(one):
    name, _, paths, native = one
    paths = sorted(paths, key=rank_of)
    measured = filter.calc_rmsds(paths, native)
    labelled = [None if rmsd is None else rmsd <= filter.NEAR_NATIVE for rmsd in measured]
    near = [found is True for found in labelled]
    return {
        "name": name,
        "poses": len(paths),
        "near_native": sum(near),
        "fraction": sum(near) / len(paths),
        "top1": near[0],
        "rank_top1": rank_of(paths[0]),
        "rmsd_top1": measured[0],
        "discrimination": report.pairwise_discriminate(
            [_published(path)[1] for path in paths], labelled
        ),
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


def plot(rows, name=None, job=False):
    """
    Two plots over the same five bands: top-1 success against what a random pick would manage, and
        how many complexes each band holds, which is what says how much the top panel is worth.
    """
    path = f"{motivation_path(name, job)}.png"
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


def write(rows, name=None):
    report.write_table(rows, f"{motivation_path(name)}.csv", FIELDS)


def read(name=None):
    return report.read_table(f"{motivation_path(name)}.csv")


def _report(rows, incomplete=()):
    print("Complexes", len(rows))
    if incomplete:
        print("  incomplete", len(incomplete), sorted(incomplete))

    poses = sum(row["poses"] for row in rows)
    near = sum(row["near_native"] for row in rows)
    print(f"Poses {poses}, {near} near-native ({near / poses:.1%})")

    bins = bin(rows)
    success, chance, counts = rates(bins)
    print("\nTop-1 and pairwise discrimination by how much of the ensemble is near-native\n")
    print(f'  {"band":>8s}{"n":>6s}{"top-1":>9s}{"chance":>9s}{"D":>9s}')
    for label, count, hit, random_pick, group in zip(LABELS, counts, success, chance, bins):
        print(f"  {label:>8s}{count:6d}{hit:9.1%}{random_pick:9.1%}"
              f'{report.rate(report.average(group, "discrimination")):>9s}')
    print(f'  {"all":>8s}{len(rows):6d}'
          f'{statistics.mean(row["top1"] for row in rows):9.1%}'
          f'{statistics.mean(row["fraction"] for row in rows):9.1%}'
          f'{report.rate(report.average(rows, "discrimination")):>9s}')


def run(complexes=None, reuse=False, name=None):
    if reuse:
        rows, incomplete = read(name=name), ()
    else:
        found, incomplete = filter.inventory(complexes)
        if not found:
            raise SystemExit("No complex has both an ensemble and a deposited ligand")
        rows = [_row(_found) for _found in found]
        write(rows, name=name)
    _report(rows, incomplete)
    print("\nWrote", os.path.relpath(plot(rows, name), ROOT))
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
        help=f"Plot the measurements stored under the name of measuring every pose again.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help=f"Plot the measurements under a different name."
    )
    arguments = parser.parse_args()

    run(arguments.complexes, arguments.reuse, arguments.name)
