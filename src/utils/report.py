"""
Common functions for ranking poses and writing atables.

Top-1 and pairwise discrimination are reported against a random picker.
"""

import csv
import os
import re
import statistics

from scipy.stats import mannwhitneyu, spearmanr

NUM_POSES_FOR_SPEARMAN = 3

# DiffDock names a scored pose rank<N>_confidence<X>.sdf.
POSE = re.compile(r"^rank(\d+)_confidence(-?\d+\.\d+)\.sdf$")

INTEGERS = [
    "poses", "near_native", "rank_docked", "rank_minimised", "rank_sapt", "rank_rhf",
    "rank_separable", "moved", "moved_rhf", "rank_top1",
]
DECIMALS = [
    "confidence", "confidence_docked", "confidence_minimised", "rmsd", "rmsd_top1", "fraction",
    "elst", "exch", "cumulant", "interaction", "elst_rhf", "exch_rhf", "interaction_rhf",
    "interaction_separable",
    "discrimination", "discrimination_docked", "discrimination_minimised", "discrimination_sapt",
    "discrimination_rhf", "discrimination_separable", "spearman_separable", "spearman_rhf",
    "retention",
]
BOOLEANS = [
    "top1", "top1_docked", "top1_minimised", "top1_sapt", "top1_rhf", "top1_separable",
]


def docked_rank_and_score(source):
    """
    The rank and confidence DiffDock published, read from the file names.
    """
    scored = POSE.match(source)
    if scored is None:
        raise ValueError(f"{source} is not a pose DiffDock scored")
    return int(scored.group(1)), float(scored.group(2))


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


def pairwise_discriminate(scores, near):
    """
    `scores` is one a pose, higher the better, so an energy must aarrive negated

    Mann-Whitney's U counts exactly the pairs, halving the ties, so the rate is U over the pairs
        the two groups make.
    """
    first = [score for score, is_near in zip(scores, near) if is_near]
    second = [score for score, is_near in zip(scores, near) if is_near is not None and not is_near]
    if not first or not second:
        return None
    return float(mannwhitneyu(first, second).statistic) / (len(first) * len(second))


def spearman_correlate(first, second):
    """
    None where there are too few of them to rank.
    """
    if len(first) < NUM_POSES_FOR_SPEARMAN:
        return None
    return float(spearmanr(first, second).statistic)


def average(rows, field):
    answered = [row[field] for row in rows if row.get(field) is not None]
    return statistics.mean(answered) if answered else None


def rate(value):
    return "" if value is None else f"{value:.3f}"


def rankings(summary, ranked, chance="fraction"):
    """
    A line per ranking: how often its first pose is near-native, and how often it orders a pair 
        correctly, against what a random pick off the ensemble would manage.

    `ranked` is (top-1 column, discrimination column, name) for each ranking.
    """
    answered = [row for row in summary if row[ranked[0][0]] is not None]
    if not answered:
        return
    rated = [row for row in answered if row[ranked[0][1]] is not None]
    print(f"\nOver {len(answered)} complexes\n")
    _line("", "", "top-1", "D")
    for top1, discriminated, name in ranked:
        right = sum(1 for row in answered if row[top1])
        _line(name, str(right), f"{right / len(answered):.1%}", rate(average(rated, discriminated)))
    _line(
        "a random pick off the ensemble",
        "",
        f"{statistics.mean(row[chance] for row in answered):.1%}",
        rate(0.5 if rated else None),
    )
    if len(rated) < len(answered):
        print(f"\n  D is over the {len(rated)} of {len(answered)} complexes that hold a pair to "
              "order.")


def _line(name, count, top1, discriminated):
    print(f"  {name:42s}{count:>4s}{top1:>8s}{discriminated:>9s}".rstrip())
