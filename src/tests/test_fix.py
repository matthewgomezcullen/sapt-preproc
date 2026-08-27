"""
Structure repair for EncodeProtein._fix.

All three cases are exercised on 5S8I_2LY, which is missing side-chain atoms (LYS A1323 lacks CE
    and NZ), a terminal atom (ARG A1434 lacks OXT), and four single-residue chain breaks.
"""

import pytest

from conftest import paths
from encode import EncodeProtein

COMPLEX = "5S8I_2LY"

INCOMPLETE_RESIDUE = ("A", 1323)                # LYS, modelled without CE and NZ
MISSING_SIDE_CHAIN_ATOMS = {"CE", "NZ"}

TERMINAL_RESIDUE = ("A", 1434)                  # ARG, the C-terminus, modelled without OXT
MISSING_TERMINAL_ATOM = "OXT"

# Residues absent from the model, each leaving a C-N break of 3.2-4.0 A rather than a peptide bond.
MISSING_RESIDUES = [("A", 1335), ("A", 1382), ("A", 1405), ("A", 1419)]


def atom_names(model, chain_name, seqid):
    """
    Atom names of one residue, or an empty set if the model does not contain it.
    """
    for chain in model:
        if chain.name != chain_name:
            continue
        for residue in chain:
            if residue.seqid.num == seqid:
                return {atom.name for atom in residue}
    return set()


def heavy_atom_coordinates(model):
    """
    Every heavy atom keyed by (chain, sequence number, atom name), so repaired structures can be
        compared against the input residue by residue.
    """
    return {
        (chain.name, residue.seqid.num, atom.name): (atom.pos.x, atom.pos.y, atom.pos.z)
        for chain in model
        for residue in chain
        for atom in residue
        if not atom.element.is_hydrogen
    }


@pytest.fixture
def encoding():
    encode = EncodeProtein(*paths(COMPLEX))
    encode._fetch()
    return encode


def test_fix_adds_missing_side_chain_atoms(encoding):
    """
    Heavy atoms absent from a modelled residue are built back onto it.
    """
    chain, seqid = INCOMPLETE_RESIDUE
    assert not atom_names(encoding.whole, chain, seqid) & MISSING_SIDE_CHAIN_ATOMS

    encoding._fix()

    assert MISSING_SIDE_CHAIN_ATOMS <= atom_names(encoding.whole, chain, seqid)


def test_fix_adds_missing_terminal_atoms(encoding):
    """
    A chain terminus missing its OXT is completed.
    """
    chain, seqid = TERMINAL_RESIDUE
    assert MISSING_TERMINAL_ATOM not in atom_names(encoding.whole, chain, seqid)

    encoding._fix()

    assert MISSING_TERMINAL_ATOM in atom_names(encoding.whole, chain, seqid)


def test_fix_preserves_existing_heavy_atom_coordinates(encoding):
    """
    Repair only adds atoms. Every heavy atom already present keeps its input coordinates, so the
        cutout taken later is the one the poses were docked against.
    """
    before = heavy_atom_coordinates(encoding.whole)

    encoding._fix()

    after = heavy_atom_coordinates(encoding.whole)
    assert before.keys() <= after.keys()
    for key, position in before.items():
        assert after[key] == pytest.approx(position, abs=1e-3)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "TODO: PDBFixer.findMissingResidues needs SEQRES records to know the full sequence, and "
        "no PoseBusters PDB carries them, so chain breaks go undetected and unrepaired."
    ),
)
def test_fix_adds_missing_residues(encoding):
    """
    Residues absent from the model entirely are rebuilt, closing the chain breaks.
    """
    for chain, seqid in MISSING_RESIDUES:
        assert not atom_names(encoding.whole, chain, seqid)

    encoding._fix()

    for chain, seqid in MISSING_RESIDUES:
        assert atom_names(encoding.whole, chain, seqid), f"chain break at {chain}{seqid} unrepaired"
