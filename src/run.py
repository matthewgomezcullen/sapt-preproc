"""
Run the encoding pipeline over one complex and keep what it produces.

RHF over a cutout of the bin is eight to twelve hours and nothing above it is free either, so a
    finished run writes its active space to <out>/<name>.npz and a run that finds one already
    there does nothing. The SCF underneath is checkpointed separately by `utils.encode`, so a job
    killed part way through still leaves the expensive half behind.

What is stored is the whole window Dice returned, untruncated. Choosing an occupation window is
    then arithmetic on those numbers rather than another solve, which is what makes it safe to run
    this before the window has been settled.

    python run.py --complex 7USH_82V
"""

import argparse
import os

import numpy as np

from encode import EncodeProtein, EncodingError
from filter import _poses, _protein
from prepare import PrepareComplex

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")

# What `$DATA` is joined with, and what `out/` is joined with off the cluster.
SPACES = "spaces"

# The window the driver solves under. See the module docstring.
EVERYTHING = (0.0, 2.0)


def spaces():
    """
    Where the finished active spaces go.
    """
    data = os.environ.get("DATA")
    return os.path.join(data, SPACES) if data else os.path.join(OUT, SPACES)


def path(name, out):
    """
    The file this complex's active space takes in `out`.
    """
    return os.path.join(out, f"{name}.npz")


def save(encoded, name, destination):
    """
    Everything a later experiment needs from a finished run.
    """
    if encoded.energy_cas is None:
        raise EncodingError(
            f"{name} has not been through SHCI, so there is no active space to save"
        )
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    np.savez(
        destination,
        name=name,
        ncas=encoded.ncas,
        nelecas=encoded.nelecas,
        orbitals=encoded.orbitals,
        occupations=encoded.occupations,
        energy=encoded.energy,
        energy_cas=encoded.energy_cas,
        correlation=encoded.correlation,
        window=np.array(EVERYTHING),
    )


def load(source):
    """
    A stored run, with its scalars as scalars rather than as zero-dimensional arrays.
    """
    with np.load(source, allow_pickle=False) as stored:
        result = {key: stored[key] for key in stored.files}
    return {
        key: value.item() if value.ndim == 0 else value for key, value in result.items()
    }


def done(name, out):
    """
    Whether this complex already has a result worth keeping.

    A file cut off mid-write is not one, and says so by failing to read.
    """
    try:
        load(path(name, out))
    except (OSError, ValueError, KeyError):
        return False
    return True


def _complex(name):
    """
    A complex of the benchmark set, resolved to the halves `PrepareComplex` wants.
    """
    proteins, poses = _protein(name), _poses(name)
    if len(proteins) != 1 or not poses:
        raise EncodingError(
            f"{name} has {len(proteins)} deposited structures and {len(poses)} poses; it needs "
            "exactly one structure and at least one pose"
        )
    return PrepareComplex(proteins[0], poses)


def run(name, out, prepared=None, targets=None, nmax=None, eps1=1e-4, force=False):
    """
    Carry one complex the whole way and keep what comes out, or nothing if it is already there.

    Returns the finished encoder, or None where there was nothing to do.

    `targets` is handed to AVAS. None lets it work them out from the poses, which is the pipeline;
        a callable is given the built molecule, which is how a cutout with no pose near its
        contact can still be run.
    """
    if not force and done(name, out):
        return None

    encoded = EncodeProtein(prepared if prepared is not None else _complex(name))
    if prepared is None:
        encoded.prepared.prepare()
    if nmax is not None:
        encoded.nmax = nmax

    encoded.RHF()
    encoded.AVAS(targets(encoded.mol) if callable(targets) else targets)
    encoded.MP2()
    encoded.SHCI(eps1=eps1, lo=EVERYTHING[0], hi=EVERYTHING[1])

    save(encoded, name, path(name, out))
    return encoded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complex", required=True, help="The complex to run, by name.")
    parser.add_argument(
        "--out",
        default=None,
        help=f"Where to write the active space. Defaults to $DATA/{SPACES} on the cluster and "
             f"{os.path.relpath(os.path.join(OUT, SPACES), ROOT)} without it.",
    )
    parser.add_argument(
        "--nmax", type=int, default=None, help="Natural orbitals the MP2 cap keeps."
    )
    parser.add_argument(
        "--eps1",
        type=float,
        default=1e-4,
        help="Dice's selection threshold. Larger selects fewer determinants and costs less.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even where a result is already stored, overwriting it.",
    )
    arguments = parser.parse_args()

    out = arguments.out or spaces()
    encoded = run(
        arguments.complex,
        out,
        nmax=arguments.nmax,
        eps1=arguments.eps1,
        force=arguments.force,
    )
    if encoded is None:
        print(f"{arguments.complex} is already in {out}; nothing to do")
    else:
        print(
            f"{arguments.complex}: ({encoded.nelecas}e, {encoded.ncas}o) "
            f"E = {encoded.energy_cas:.9f} written to {path(arguments.complex, out)}"
        )
