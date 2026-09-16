"""
Carry complexes of the benchmark set through preparation and encoding, keeping what each stage 
    produces in <out>/<name>/<complex>.

A stage whose artefact is already kept is read back rather than run again.

    python run.py bin --complexes 7USH_82V 7R9N_F97
"""

import argparse
import os
import re

from prepare import PrepareComplex
from encode import EncodeProtein, SolveLigand


ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
DATA = os.path.join(ROOT, "data")

# DiffDock names each pose it kept rank<N>_confidence<X>.sdf. Alongside those it writes a bare
# rank1.sdf copy of the top-ranked pose and, for some complexes, an energy-minimised
# rank<N>_confidence<X>_ensemble_relaxed.sdf.
POSE = re.compile(r"^rank\d+_confidence-?\d+\.\d+\.sdf$")

FAIL = "confidence-1000"


def _poses(directory):
    return sorted(
        os.path.join(directory, entry)
        for entry in os.listdir(directory)
        if POSE.match(entry) and FAIL not in entry
    )


def _load(data, out, complexes) -> list[PrepareComplex]:
    """
    The complexes under `data`, or those of them `complexes` names, each kept in a directory of its
        own under `out`.
    """
    proteins, poses = os.path.join(data, "posebusters"), os.path.join(data, "diffdock")
    if complexes is None:
        complexes = sorted(
            name
            for name in set(os.listdir(proteins)) & set(os.listdir(poses))
            if os.path.isdir(os.path.join(poses, name))
        )
    return [
        PrepareComplex(
            os.path.join(proteins, complex, f"{complex}_protein.pdb"),
            _poses(os.path.join(poses, complex)),
            os.path.join(out, complex),
        )
        for complex in complexes
    ]


def run(name, data, out, complexes=None, force=False, prepare_only=False):
    out = os.path.join(out, name)
    complexes = _load(data, out, complexes)
    for complex in complexes:
        if force or not complex.prepared():
            complex.prepare()
        if prepare_only:
            continue
        # Before the protein, so a pose that cannot be solved shows first.
        ligand = SolveLigand(complex, complex.out)
        if force or not ligand.solved():
            ligand.RHF()
        protein = EncodeProtein(complex, complex.out)
        if force or not protein.solved():
            protein.solve()
        if force or not protein.encoded():
            protein.encode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="The job, which names its directory under --out.")
    parser.add_argument(
        "--data",
        default=DATA,
        help="The benchmark set, split into posebusters/ and diffdock/.",
    )
    parser.add_argument("--out", default=OUT, help="Where jobs are kept.")
    parser.add_argument(
        "--complexes",
        nargs="+",
        default=None,
        help="The complexes to run, by name. Every one in --data by default.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run every stage again, even one already kept.",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Prepare only. Don't encode.")
    arguments = parser.parse_args()
    run(
        arguments.name,
        arguments.data,
        arguments.out,
        arguments.complexes,
        arguments.force, 
        arguments.prepare_only
    )
