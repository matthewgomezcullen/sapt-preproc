"""
Deletion of out-of-scope molecules for EncodeProtein._clean.

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
from encode import EncodeProtein, _is_amino_acid, _is_water
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


def encoding(name):
    """
    A complex carried up to the point _clean is called.
    """
    encode = EncodeProtein(*paths(name))
    encode._fetch()
    encode._verify()
    encode._fix()
    return encode


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


def cutout_residues(encode):
    """
    The residues the 4.5 Å cutout selects, keyed so they survive deletion elsewhere in the model.
    """
    return {
        (chain.name, residue.seqid.num, residue.seqid.icode, residue.name)
        for chain, residue, _, _ in verify.cutout(
            encode.whole, encode._pose_coordinates(), encode.cutoff
        )
    }


@pytest.mark.parametrize("name", COMPLEXES)
def test_clean_deletes_heterogens(name):
    """
    No heterogen survives cleaning. Crystallisation additives, buffer molecules, and cryoprotectants
        carry no interaction energy with the poses and would otherwise have to be protonated and
        charge-assigned as non-amino-acid molecules.
    """
    encode = encoding(name)
    assert heterogens(encode.whole)

    encode._clean()

    assert not heterogens(encode.whole)


def test_clean_deletes_named_additives():
    """
    Every additive in a structure is deleted.
    """
    encode = encoding(ADDITIVES)
    before = residue_names(encode.whole)
    assert {name: before.get(name) for name in ADDITIVE_HETEROGENS} == ADDITIVE_HETEROGENS

    encode._clean()

    after = residue_names(encode.whole)
    assert not [name for name in ADDITIVE_HETEROGENS if name in after]


@pytest.mark.parametrize("name", COMPLEXES)
def test_clean_deletes_metals(name):
    """
    Loose ions are deleted along with everything else outside the cutout.
    """
    encode = encoding(name)
    assert encode.whole
    assert any(
        atom.element.is_metal for chain in encode.whole for residue in chain for atom in residue
    )

    encode._clean()

    assert not any(
        atom.element.is_metal for chain in encode.whole for residue in chain for atom in residue
    )


def test_clean_deletes_the_crystallographic_ligand():
    """
    A deposited copy of the ligand being docked is also a heterogen. 7WUX_6OI keeps two copies of 
        6OI in a second binding site. Leaving them in would put the answer inside the structure.
    """
    encode = encoding(CRYSTALLOGRAPHIC_LIGAND)
    assert residue_names(encode.whole).get(NATIVE_LIGAND) == NATIVE_LIGAND_COPIES

    encode._clean()

    assert NATIVE_LIGAND not in residue_names(encode.whole)


def test_clean_keeps_peptide_caps():
    """
    An ACE or NME already capping a chain terminus is part of the polypeptide, not a heterogen.
    """
    encode = encoding(HYDROGENATED)
    assert residue_names(encode.whole).get(CAP_RESIDUE)

    encode._clean()

    assert residue_names(encode.whole).get(CAP_RESIDUE)


def test_clean_strips_deposited_hydrogens():
    """
    Hydrogens that came with the deposited structure are removed. _protonate owns every hydrogen.
    """
    encode = encoding(HYDROGENATED)
    hydrogens = lambda model: sum(
        atom.element.is_hydrogen for chain in model for residue in chain for atom in residue
    )
    assert hydrogens(encode.whole) == DEPOSITED_HYDROGENS

    encode._clean()

    assert hydrogens(encode.whole) == 0


@pytest.mark.parametrize("name", COMPLEXES)
def test_clean_preserves_every_amino_acid(name):
    """
    Cleaning only deletes. Every amino-acid atom stays constant.
    """
    encode = encoding(name)
    before = amino_acids(encode.whole)

    encode._clean()

    after = amino_acids(encode.whole)
    assert [entry[:5] for entry in after] == [entry[:5] for entry in before]
    for cleaned, repaired in zip(after, before):
        assert cleaned[5] == pytest.approx(repaired[5], abs=1e-3)


@pytest.mark.parametrize("name", COMPLEXES)
def test_clean_preserves_the_cutout(name):
    """
    The cutout _verify accepted is the cutout _reduce later takes.
    """
    encode = encoding(name)
    before = cutout_residues(encode)

    encode._clean()

    assert cutout_residues(encode) == before


def test_clean_leaves_no_empty_chains():
    """
    Repair splits heterogens into chains that reuse the polymer chain names, so deleting them leaves
        duplicate empty chains behind. The cutout keys residues by chain name, causing ambiguity.
    """
    encode = encoding(ADDITIVES)

    encode._clean()

    assert encode.whole
    chains = [chain.name for chain in encode.whole]
    assert all(len(chain) for chain in encode.whole)
    assert len(chains) == len(set(chains))


@pytest.mark.parametrize("name", COMPLEXES)
def test_clean_is_idempotent(name):
    """
    Idempotency.
    """
    encode = encoding(name)
    encode._clean()
    once = amino_acids(encode.whole)

    encode._clean()

    assert amino_acids(encode.whole) == once
