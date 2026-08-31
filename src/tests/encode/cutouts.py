"""
The cutouts the encoding tests are built on, and what it costs to get them.

The bin under test is Q1 of the size quartiles with |q_A| <= 1: the smallest eligible cutouts
    carrying at most one net charge, so the cheapest SCF in the set and the least distorted by the
    cut. `filter.py --reuse` names all twelve; three are tracked as fixtures.

Nothing here is cheap, and the SCFs are not cheap by three orders of magnitude, so everything is
    cached and shared across the modules that import it.
"""

import functools

import gemmi
import numpy as np

from conftest import paths
from encode import EncodeProtein
from prepare import PrepareComplex

# Q1, |q_A| <= 1, restricted to the complexes tracked as fixtures. The nine left out are 7BJJ_TVW,
# 6YQW_82I, 7R59_I5F, 7LOE_Y84, 7W05_GMP, 7PJQ_OWH, 7XI7_4RI, 7EBG_J0L and 7Z1Q_NIO.
SUBSET = ["7USH_82V", "7W06_ITN", "7R9N_F97"]

# Heavy atoms, net charge and electrons, from the screen in out/filter.csv.
EXPECTED = {
    "7USH_82V": (157, -1, 1188),
    "7W06_ITN": (168, +1, 1286),
    "7R9N_F97": (169, -1, 1314),
}

# A capped run lifted whole out of 5S8I_2LY's cutout, which is one chain of ten of them. Small
# enough to solve here, and real: the coordinates are the ones the pipeline produced.
FRAGMENT = "5S8I_2LY"
FRAGMENT_SLICE = (4, 7)
FRAGMENT_RESIDUES = ["ACE", "VAL", "NME"]
FRAGMENT_ATOMS = 28
FRAGMENT_ELECTRONS = 94


@functools.lru_cache(maxsize=None)
def prepare(name):
    """
    A complex carried through the whole pipeline.
    """
    prepared = PrepareComplex(*paths(name))
    prepared.prepare()
    return prepared


def slice_out(model, start, stop):
    """
    A run of residues lifted out of a cutout as a model of its own.
    """
    structure = gemmi.Structure() # pyright: ignore[reportAttributeAccessIssue]
    sliced = gemmi.Model("1") # pyright: ignore[reportAttributeAccessIssue]
    chain = gemmi.Chain(model[0].name) # pyright: ignore[reportAttributeAccessIssue]
    for residue in list(model[0])[start:stop]:
        chain.add_residue(residue)
    sliced.add_chain(chain)
    structure.add_model(sliced)
    structure.setup_entities()
    return structure[0]


@functools.lru_cache(maxsize=None)
def fragment():
    """
    ACE-VAL-NME, prepared and charged, standing in for a cutout small enough to solve.
    """
    prepared = PrepareComplex(*paths(FRAGMENT))
    prepared.prepare()
    prepared.reduced = slice_out(prepared.reduced, *FRAGMENT_SLICE)
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
    core = (encoded.mol.nelectron - encoded.nelecas) // 2
    return encoded.orbitals[:, core:core + encoded.ncas]


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
