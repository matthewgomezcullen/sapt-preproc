"""
Carry complexes of the benchmark set through preparation, encoding and CASCI, keeping what each
    stage produces in <out>/<name>/<complex>.

A stage whose artefact is already kept is read back rather than run again. A complex whose
    preparation is kept needs none of its inputs, so a job can start from a screen's preparations
    with no benchmark set on disk.

    python run.py filter_v1_1_mm_unsize --complexes 7LOE_Y84 7F5D_EUO

    python run.py filter_v1_1_mm_unsize --rewindow --orbitals 14
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


def _window(given):
    if given is None:
        return None
    if len(given) not in (0, 2):
        raise SystemExit("--rewindow takes both thresholds, lo and hi, or neither")
    return tuple(given)


def run(name, out=OUT, complexes=None, force=False, prepare_only=False, window=None, orbitals=None):
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
        if orbitals is not None:
            protein.ncas_limit = orbitals
        if force or not protein.solved():
            protein.solve()
        if force or window is not None or not protein.encoded():
            protein.rewindow(*(window or ()))
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
    parser.add_argument(
        "--rewindow",
        nargs="*",
        type=float,
        default=None,
        metavar=("LO", "HI"),
        help="Cut the window again over a complex already encoded, at the thresholds LO and HI, or "
             "at the one --orbitals leaves.",
    )
    parser.add_argument(
        "--orbitals",
        type=int,
        default=None,
        help="The most orbitals a window with no thresholds leaves. Fourteen by default.",
    )
    arguments = parser.parse_args()
    run(
        arguments.name,
        arguments.out,
        arguments.complexes,
        arguments.force,
        arguments.prepare_only,
        _window(arguments.rewindow),
        arguments.orbitals,
    )
