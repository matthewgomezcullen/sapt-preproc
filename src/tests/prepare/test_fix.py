"""
Structure repair for PrepareComplex._fix.

All three cases are exercised on 5S8I_2LY, which is missing side-chain atoms (LYS A1323 lacks CE
    and NZ), a terminal atom (ARG A1434 lacks OXT), and four single-residue chain breaks.
"""

import pytest

from conftest import paths
from prepare import PrepareComplex
from utils import fix

COMPLEX = "5S8I_2LY"

# PDBFixer rebuilds the ring of PHE A382 into a pocket it does not fit, and falls back to Langevin
# dynamics to push it clear. That path amplifies any wobble in the forces into whole angstroms.
CHAOTIC = "6ZCY_QF8"

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
def prepared():
    _prepare = PrepareComplex(*paths(COMPLEX))
    _prepare._fetch()
    return _prepare


def test_fix_adds_missing_side_chain_atoms(prepared):
    """
    Heavy atoms absent from a modelled residue are built back onto it.
    """
    chain, seqid = INCOMPLETE_RESIDUE
    assert not atom_names(prepared.whole, chain, seqid) & MISSING_SIDE_CHAIN_ATOMS

    prepared._fix()

    assert MISSING_SIDE_CHAIN_ATOMS <= atom_names(prepared.whole, chain, seqid)


def test_fix_adds_missing_terminal_atoms(prepared):
    """
    A chain terminus missing its OXT is completed.
    """
    chain, seqid = TERMINAL_RESIDUE
    assert MISSING_TERMINAL_ATOM not in atom_names(prepared.whole, chain, seqid)

    prepared._fix()

    assert MISSING_TERMINAL_ATOM in atom_names(prepared.whole, chain, seqid)


def test_fix_preserves_existing_heavy_atom_coordinates(prepared):
    """
    Repair only adds atoms. Every heavy atom already present keeps its input coordinates, so the
        cutout taken later is the one the poses were docked against.
    """
    before = heavy_atom_coordinates(prepared.whole)

    prepared._fix()

    after = heavy_atom_coordinates(prepared.whole)
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
def test_fix_adds_missing_residues(prepared):
    """
    Residues absent from the model entirely are rebuilt, closing the chain breaks.
    """
    for chain, seqid in MISSING_RESIDUES:
        assert not atom_names(prepared.whole, chain, seqid)

    prepared._fix()

    for chain, seqid in MISSING_RESIDUES:
        assert atom_names(prepared.whole, chain, seqid), f"chain break at {chain}{seqid} unrepaired"


MODIFIED = "7W06_ITN"
SELENOMETHIONINES = [("A", 161), ("A", 167), ("A", 168), ("A", 171), ("A", 239), ("A", 248)]


def test_fix_replaces_modified_residues_with_their_standard_form():
    """
    Selenomethionine is substituted for methionine, taking its selenium with it.

    Leaving MSE in place breaks two later steps. Selenium has no 6-31G basis, so a cutout reaching
        one would be rejected outright; and Modeller has no hydrogen definitions for MSE, so it
        would come out of _protonate either bare or filled in from a definition for the free amino
        acid, which carries a second backbone amide hydrogen a mid-chain residue must not have.
    """
    prepared = PrepareComplex(*paths(MODIFIED))
    prepared._fetch()
    assert all(atom_names(prepared.whole, chain, seqid) for chain, seqid in SELENOMETHIONINES)
    assert any(
        atom.element.name == "Se"
        for chain in prepared.whole
        for residue in chain
        for atom in residue
    )

    prepared._fix()

    for chain_name, seqid in SELENOMETHIONINES:
        names = atom_names(prepared.whole, chain_name, seqid)
        assert "SD" in names, f"{chain_name}{seqid} kept its selenium rather than becoming MET"
        assert "SE" not in names
    assert not any(
        atom.element.name == "Se"
        for chain in prepared.whole
        for residue in chain
        for atom in residue
    )


def test_repair_places_rebuilt_atoms_in_the_same_position_every_run():
    """
    Two repairs of one structure agree exactly.

    `addMissingAtoms` minimises each rebuilt atom, and where the minimum still leaves atoms on top
        of each other it runs Langevin dynamics to separate them. Unpinned, the integrator draws its
        own seed and the forces are summed in whatever order the CPU platform's threads finish, and
        the side chain of PHE A382 lands up to 6 Å apart between two runs of the same input. That is
        enough to carry it across the 4.5 Å cutoff on one run and not the next, which decides both
        whether the residue is in the cutout and whether the complex is rejected for holding it.
    """
    prepared = PrepareComplex(*paths(CHAOTIC))

    first = fix.repair(prepared.protein_path, prepared.seed)[0]
    second = fix.repair(prepared.protein_path, prepared.seed)[0]

    assert heavy_atom_coordinates(first) == heavy_atom_coordinates(second)
