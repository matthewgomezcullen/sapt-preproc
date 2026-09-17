"""
Carry complexes of the benchmark set through preparation, encoding and CASCI, keeping what each
    stage produces in <out>/<name>/<complex>.

A stage whose artefact is already kept is read back rather than run again. A complex whose
    preparation is kept needs none of its inputs, so a job can start from a screen's preparations
    with no benchmark set on disk.

    python run.py filter_v1_1_mm_unsize --complexes 7LOE_Y84 7F5D_EUO
"""

import argparse
import os

import filter
from prepare import PrepareComplex
from encode import EncodeProtein, SolveLigand


ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")


def _load(out, complexes=None, force=False) -> list[PrepareComplex]:
    """
    The complexes `complexes` names, or every one in the benchmark set.

    A kept preparation is read back without its inputs, unless `force` is going to prepare it again.
    """
    loaded = {}
    if complexes is not None and not force:
        for name in complexes:
            kept = PrepareComplex("", [], os.path.join(out, name))
            if kept.prepared():
                loaded[name] = kept

    wanted = None if complexes is None else [name for name in complexes if name not in loaded]
    # An empty list would ask filter.inventory for every complex.
    if wanted is None or wanted:
        for name, protein, poses, _ in filter.inventory(wanted)[0]:
            loaded[name] = PrepareComplex(protein, poses, os.path.join(out, name))

    if complexes is None:
        return list(loaded.values())
    missing = [name for name in complexes if name not in loaded]
    if missing:
        raise SystemExit(
            f"{', '.join(missing)} kept no preparation under {out}, and the benchmark set holds no "
            "complete inputs to prepare them from"
        )
    return [loaded[name] for name in complexes]


def run(name, out=OUT, complexes=None, force=False, prepare_only=False):
    out = os.path.join(out, name)
    for complex in _load(out, complexes, force):
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
            # Dice solved the whole window. The Hamiltonian is over the paper's, which `rewindow`
            # takes by default, and the solved space is kept as Dice left it.
            protein.rewindow()
            protein.encode()
        if force or not protein.correlated():
            protein.CASCI()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="The job, which names its directory under --out.")
    parser.add_argument("--out", default=OUT, help="Where jobs are kept.")
    parser.add_argument(
        "--complexes",
        nargs="+",
        default=None,
        help="The complexes to run, by name. Every one in the benchmark set by default.",
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
        arguments.out,
        arguments.complexes,
        arguments.force,
        arguments.prepare_only
    )
