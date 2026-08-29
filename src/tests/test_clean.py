"""
Deletion of out-of-scope molecules for PrepareComplex._clean.

_clean runs after _fix and before _protonate. It drops anything that protonation, the cutout, or RHF
     should not see. Everything it deletes has already been cleared by _verify as outside the cutout.

Deletion must not delete too much, so many tests pin what must survive.

Three real complexes cover the cases, all of them accepted by _verify:

    6TW5_9M2    crystallisation additives (MYA, DMS, CL, SO4) and three loose Mg ions
    7NFB_GEN    4317 deposited hydrogens, plus an ACE cap that must not be mistaken for a heterogen
    7WUX_6OI    two crystallographic copies of the very ligand being docked
"""

import pytest

from conftest import paths
from prepare import PrepareComplex, _is_amino_acid, _is_water
from utils import verify

# Acetyl and amide caps terminate a polypeptide chain. They are not amino acids, so a name test 
#   alone reads them as heterogens, but deleting one strips a chain terminus of its cap.
PEPTIDE_CAPS = frozenset({"ACE", "NME", "NH2"})

ADDITIVES = "6TW5_9M2"
ADDITIVE_HETEROGENS = {"MYA": 3, "DMS": 3, "CL": 3, "SO4": 3, "MG": 3}
ADDITIVE_METALS = {"Mg": 3}

HYDROGENATED = "7NFB_GEN"
DEPOSITED_HYDROGENS = 4317
CAP_RESIDUE = "ACE"

CRYSTALLOGRAPHIC_LIGAND = "7WUX_6OI"
NATIVE_LIGAND = "6OI"
NATIVE_LIGAND_COPIES = 2

COMPLEXES = [ADDITIVES, HYDROGENATED, CRYSTALLOGRAPHIC_LIGAND]


def prepare(name):
    """
    A complex carried up to the point _clean is called.
    """
    _prepare = PrepareComplex(*paths(name))
    _prepare._fetch()
    _prepare._verify()
    _prepare._fix()
    return _prepare


def residue_names(model):
    """
    How many copies of each residue name the model holds.
    """
    counts = {}
    for chain in model:
        for residue in chain:
            counts[residue.name] = counts.get(residue.name, 0) + 1
    return counts


def heterogens(model):
    """
    Residue names that are neither an amino acid, a water, nor a peptide cap.
    """
    return {
        residue.name
        for chain in model
        for residue in chain
        if not _is_amino_acid(residue.name)
        and not _is_water(residue.name)
        and residue.name not in PEPTIDE_CAPS
    }


def amino_acids(model):
    """
    Every amino-acid heavy atom as (chain, sequence number, insertion code, residue, atom,
        position), in model order, so a cleaned structure can be compared against the repaired one
        atom by atom. Hydrogens are excluded because _clean deletes them.

    The insertion code is part of the key because two residues in this dataset can share a chain
        and a sequence number.
    """
    return [
        (
            chain.name,
            residue.seqid.num,
            residue.seqid.icode,
            residue.name,
            atom.name,
            (atom.pos.x, atom.pos.y, atom.pos.z),
        )
        for chain in model
        for residue in chain
        if _is_amino_acid(residue.name)
        for atom in residue
        if not atom.element.is_hydrogen
    ]


def cutout_residues(prepared, amino_acids_only=False):
    """
    The residues the 4.5 Å cutout selects, keyed so they survive deletion elsewhere in the model.
    """
    return {
        (chain.name, residue.seqid.num, residue.seqid.icode, residue.name)
        for chain, residue, _, _ in verify.cutout(
            prepared.whole, prepared._pose_coordinates(), prepared.cutoff
        )
        if _is_amino_acid(residue.name) or not amino_acids_only
    }


@pytest.mark.parametrize("name", COMPLEXES)
def test_clean_deletes_heterogens(name):
    """
    No heterogen survives cleaning. Crystallisation additives, buffer molecules, and cryoprotectants
        carry no interaction energy with the poses and would otherwise have to be protonated and
        charge-assigned as non-amino-acid molecules.
    """
    prepared = prepare(name)
    assert heterogens(prepared.whole)

    prepared._clean()

    assert not heterogens(prepared.whole)


def test_clean_deletes_named_additives():
    """
    Every additive in a structure is deleted.
    """
    prepared = prepare(ADDITIVES)
    before = residue_names(prepared.whole)
    assert {name: before.get(name) for name in ADDITIVE_HETEROGENS} == ADDITIVE_HETEROGENS

    prepared._clean()

    after = residue_names(prepared.whole)
    assert not [name for name in ADDITIVE_HETEROGENS if name in after]


@pytest.mark.parametrize("name", COMPLEXES)
def test_clean_deletes_metals(name):
    """
    Loose ions are deleted along with everything else outside the cutout.
    """
    prepared = prepare(name)
    assert prepared.whole
    assert any(
        atom.element.is_metal for chain in prepared.whole for residue in chain for atom in residue
    )

    prepared._clean()

    assert not any(
        atom.element.is_metal for chain in prepared.whole for residue in chain for atom in residue
    )


def test_clean_deletes_the_crystallographic_ligand():
    """
    A deposited copy of the ligand being docked is also a heterogen. 7WUX_6OI keeps two copies of 
        6OI in a second binding site. Leaving them in would put the answer inside the structure.
    """
    prepared = prepare(CRYSTALLOGRAPHIC_LIGAND)
    assert residue_names(prepared.whole).get(NATIVE_LIGAND) == NATIVE_LIGAND_COPIES

    prepared._clean()

    assert NATIVE_LIGAND not in residue_names(prepared.whole)


def test_clean_keeps_peptide_caps():
    """
    An ACE or NME already capping a chain terminus is part of the polypeptide, not a heterogen.
    """
    prepared = prepare(HYDROGENATED)
    assert residue_names(prepared.whole).get(CAP_RESIDUE)

    prepared._clean()

    assert residue_names(prepared.whole).get(CAP_RESIDUE)


def test_clean_strips_deposited_hydrogens():
    """
    Hydrogens that came with the deposited structure are removed. _protonate owns every hydrogen.
    """
    prepared = prepare(HYDROGENATED)
    hydrogens = lambda model: sum(
        atom.element.is_hydrogen for chain in model for residue in chain for atom in residue
    )
    assert hydrogens(prepared.whole) == DEPOSITED_HYDROGENS

    prepared._clean()

    assert hydrogens(prepared.whole) == 0


@pytest.mark.parametrize("name", COMPLEXES)
def test_clean_preserves_every_amino_acid(name):
    """
    Cleaning only deletes. Every amino-acid atom stays constant.
    """
    prepared = prepare(name)
    before = amino_acids(prepared.whole)

    prepared._clean()

    after = amino_acids(prepared.whole)
    assert [entry[:5] for entry in after] == [entry[:5] for entry in before]
    for cleaned, repaired in zip(after, before):
        assert cleaned[5] == pytest.approx(repaired[5], abs=1e-3)


@pytest.mark.parametrize("name", COMPLEXES)
def test_clean_preserves_the_cutout(name):
    """
    Every protein residue _verify accepted is preserved.

    This only applies to amino acids because crystallisation additives are allowed into
        the cutout and deleted here.
    """
    prepared = prepare(name)
    before = cutout_residues(prepared, amino_acids_only=True)

    prepared._clean()

    assert cutout_residues(prepared, amino_acids_only=True) == before


def test_clean_leaves_no_empty_chains():
    """
    Repair splits heterogens into chains that reuse the polymer chain names, so deleting them leaves
        duplicate empty chains behind. The cutout keys residues by chain name, causing ambiguity.
    """
    prepared = prepare(ADDITIVES)

    prepared._clean()

    assert prepared.whole
    chains = [chain.name for chain in prepared.whole]
    assert all(len(chain) for chain in prepared.whole)
    assert len(chains) == len(set(chains))


@pytest.mark.parametrize("name", COMPLEXES)
def test_clean_is_idempotent(name):
    """
    Idempotency.
    """
    prepared = prepare(name)
    prepared._clean()
    once = amino_acids(prepared.whole)

    prepared._clean()

    assert amino_acids(prepared.whole) == once


ADDITIVE_IN_CUTOUT = "7USH_82V"
CUTOUT_ADDITIVE = ("A", 504, "EDO")
CUTOUT_PROTEIN_RESIDUES = 12


def test_clean_deletes_an_additive_from_inside_the_cutout():
    """
    _clean removes crystallisation additives without disturbing the protein around it.

    7USH_82V retains an ethylene glycol at A504 among twelve amino acids. After cleaning the twelve
        are still there and the EDO is not.
    """
    prepared = prepare(ADDITIVE_IN_CUTOUT)
    chain_name, seqid, name = CUTOUT_ADDITIVE
    before = cutout_residues(prepared)
    assert (chain_name, seqid, " ", name) in before
    assert len(cutout_residues(prepared, amino_acids_only=True)) == CUTOUT_PROTEIN_RESIDUES

    prepared._clean()

    after = cutout_residues(prepared)
    assert (chain_name, seqid, " ", name) not in after
    assert after == cutout_residues(prepared, amino_acids_only=True)
    assert len(after) == CUTOUT_PROTEIN_RESIDUES
