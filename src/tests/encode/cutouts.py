"""
The cutouts the encoding tests are built on.

Each is read back from the artefact filter.py saved rather than prepared again, so a change to
    preparation reaches these tests only once the artefacts are copied again. Solving them is not
    cheap, so everything is cached and shared across the modules that import it.
"""

import functools
import os

import gemmi
import numpy as np

from conftest import PREPARED
from encode import EncodeProtein
from prepare import PrepareComplex

# 7BJJ_TVW is charged
# 7LOE_Y84 has a sulfur within the cutoff of a pose
SUBSET = ["7BJJ_TVW", "7LOE_Y84"]

# Heavy atoms, net charge and electrons, from the screen in out/filter_v1_1_mm_unsize.csv.
EXPECTED = {
    "7BJJ_TVW": (96, -1, 756),
    "7LOE_Y84": (147, 0, 1144),
}

# ACE-VAL-NME: VAL A21 of 7BJJ_TVW's cutout, which the cut left capped on either side.
FRAGMENT = "7BJJ_TVW"
FRAGMENT_RESIDUE = ("A", "VAL", 21)
FRAGMENT_RESIDUES = ["ACE", "VAL", "NME"]
FRAGMENT_ATOMS = 28
FRAGMENT_ELECTRONS = 94


def read(name):
    prepared = PrepareComplex("", [], os.path.join(PREPARED, name))
    if not prepared.prepared():
        raise FileNotFoundError(f"No prepared artefact for {name} under {PREPARED}")
    return prepared


@functools.lru_cache(maxsize=None)
def prepare(name):
    """
    A complex carried through the whole pipeline, read once and shared.
    """
    return read(name)


def slice_out(model, chain, name, number):
    """
    A residue and its neighbours either side in its chain, lifted out of a cutout.
    """
    residues = list(model[chain])
    index = next(
        index
        for index, residue in enumerate(residues)
        if residue.name == name and residue.seqid.num == number
    )
    structure = gemmi.Structure() # pyright: ignore[reportAttributeAccessIssue]
    sliced = gemmi.Model("1") # pyright: ignore[reportAttributeAccessIssue]
    kept = gemmi.Chain(chain) # pyright: ignore[reportAttributeAccessIssue]
    for residue in residues[index - 1:index + 2]:
        kept.add_residue(residue)
    sliced.add_chain(kept)
    structure.add_model(sliced)
    structure.setup_entities()
    return structure[0]


@functools.lru_cache(maxsize=None)
def fragment():
    """
    ACE-VAL-NME, prepared and charged, standing in for a cutout small enough to solve.
    """
    prepared = read(FRAGMENT)
    prepared.reduced = slice_out(prepared.reduced, *FRAGMENT_RESIDUE)
    prepared._calculate_charge()
    prepared._verify_num_electrons()
    prepared.heavy_atoms = sum(
        1
        for chain in prepared.reduced
        for residue in chain
        for atom in residue
        if not atom.element.is_hydrogen
    )
    return prepared


@functools.lru_cache(maxsize=None)
def solved(prepared):
    """
    A prepared cutout carried through RHF.

    Keyed on the prepared complex itself, which works because `prepare` and `fragment` are cached
        too and so hand back the same object every time. A test that means to solve twice, or that
        changes a setting first, builds its own encoder; what comes back from here is shared, and
        read-only by convention.
    """
    encoded = EncodeProtein(prepared)
    encoded.RHF()
    return encoded


def elements(model):
    return [
        atom.element.name
        for chain in model
        for residue in chain
        for atom in residue
    ]


def all_carbons(mol):
    """
    Every carbon's 2p shell: a target set large enough that the cap has to cut.
    """
    return [
        f"{index} C 2p"
        for index in range(mol.natm)
        if mol.atom_symbol(index) == "C"
    ]


def window(encoded):
    """
    The active columns of the orbital set, located by the electron bookkeeping.
    """
    core = (encoded.mol.nelectron - encoded.active_electrons) // 2
    return encoded.orbital_initial[:, core:core + encoded.active_space_size]


def contact_weight(mol, vectors, targets):
    """
    Weight of each orbital on the span of the target AOs, in the overlap metric. One per column,
        each within [0, 1].
    """
    overlap = mol.intor("int1e_ovlp")
    indices = np.unique(np.concatenate([mol.search_ao_label(t) for t in targets]))
    projected = overlap[indices] @ vectors
    return np.einsum(
        "ti,ti->i", np.linalg.solve(overlap[np.ix_(indices, indices)], projected), projected
    )
