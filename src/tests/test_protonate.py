"""
Hydrogen assignment for PrepareComplex._protonate.

_protonate runs on the whole cleaned protein rather than on the cutout

It must only be additive.

Every assertion below reads the hydrogens themselves.

Three real complexes cover the cases, all accepted by _verify:

    5S8I_2LY    115 residues, the cheapest structure to protonate twice for the pH comparison
    6YT6_PKE    CYS A456-A459 at an SG-SG distance of 2.05 Å, against six free cysteines
    6TW5_9M2    21 histidines and an N-terminal ASP A27
"""

import pytest

from conftest import paths
from prepare import PrepareComplex
from utils import verify

SMALL = "5S8I_2LY"
DISULFIDE = "6YT6_PKE"
# Every test here runs the preparation pipeline over several complexes. --fast drops them.
pytestmark = pytest.mark.slow

TITRATABLE = "6TW5_9M2"

COMPLEXES = [SMALL, DISULFIDE, TITRATABLE]

# CYS A456 and A459 are bonded to each other at 2.05 Å, so neither keeps a thiol hydrogen. The
# other four cysteines in chain A are free, their nearest partner being 7.1 Å away or more.
BONDED_CYSTEINES = [("A", 456), ("A", 459), ("B", 456), ("B", 459)]
FREE_CYSTEINES = [("A", 427), ("A", 502), ("A", 559), ("A", 611)]

# Aspartate is deprotonated above its pKa and picks up HD2 below it; histidine gains a second ring
# hydrogen and becomes HIP. Both are how the tests tell 7.4 apart from an ignored pH.
ACIDIC_pH = 1.0
ASPARTATE_PROTON = "HD2"
LYSINE_PROTONS = {"HZ1", "HZ2", "HZ3"}
HISTIDINE_PROTONS = ("HD1", "HE2")


def prepare(name):
    """
    A complex carried up to the point _protonate is called.
    """
    _prepare = PrepareComplex(*paths(name))
    _prepare._fetch()
    _prepare._verify()
    _prepare._fix()
    _prepare._clean()
    return _prepare


def hydrogens(model, chain_name, seqid):
    """
    Hydrogen names on one residue, or an empty set if the model does not contain it.
    """
    for chain in model:
        if chain.name != chain_name:
            continue
        for residue in chain:
            if residue.seqid.num == seqid:
                return {atom.name for atom in residue if atom.element.is_hydrogen}
    return set()


def heavy_atoms(model):
    """
    Every heavy atom keyed by residue and atom name, in model order, so a protonated structure can
        be compared against the cleaned one atom by atom.
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
        for atom in residue
        if not atom.element.is_hydrogen
    ]


def count_hydrogens(model):
    return sum(
        atom.element.is_hydrogen for chain in model for residue in chain for atom in residue
    )


def cutout_residues(prepared):
    return {
        verify.identifier(chain, residue)
        for chain, residue, _, _ in verify.cutout(
            prepared.whole, prepared._pose_coordinates(), prepared.cutoff
        )
    }


@pytest.mark.parametrize("name", COMPLEXES)
def test_protonate_adds_hydrogens(name):
    """
    Protonation adds hydrogens.
    """
    prepared = prepare(name)
    assert count_hydrogens(prepared.whole) == 0

    prepared._protonate()

    assert count_hydrogens(prepared.whole) > 0


@pytest.mark.parametrize("name", COMPLEXES)
def test_protonate_preserves_heavy_atoms(name):
    """
    Protonation only adds hydrogens.
    """
    prepared = prepare(name)
    before = heavy_atoms(prepared.whole)

    prepared._protonate()

    after = heavy_atoms(prepared.whole)
    assert [entry[:5] for entry in after] == [entry[:5] for entry in before]
    for protonated, cleaned in zip(after, before):
        assert protonated[5] == pytest.approx(cleaned[5], abs=1e-3)


@pytest.mark.parametrize("name", COMPLEXES)
def test_protonate_preserves_the_cutout(name):
    """
    Adding hydrogens must not change which residues the cutout selects.
    """
    prepared = prepare(name)
    before = cutout_residues(prepared)

    prepared._protonate()

    assert cutout_residues(prepared) == before


@pytest.mark.parametrize("name", COMPLEXES)
def test_protonate_leaves_no_retained_residue_bare(name):
    """
    Every residue in the cutout receives at least one hydrogen.
    """
    prepared = prepare(name)

    prepared._protonate()

    for chain, residue, _, _ in verify.cutout(
        prepared.whole, prepared._pose_coordinates(), prepared.cutoff
    ):
        assert any(atom.element.is_hydrogen for atom in residue), (
            f"{residue.name} {verify.identifier(chain, residue)} received no hydrogens"
        )


def test_protonate_leaves_disulfide_cysteines_without_a_thiol_hydrogen():
    """
    A cysteine in a disulfide has no SG-H to add; giving it one invents a fifth bond on the sulfur
        and a spurious electron pair for RHF.

    _verify already rejects a cutout that splits a disulfide, so a bonded pair either sits wholly
        inside the retained region or wholly outside it. Both halves must still be assigned
        correctly, because the pair inside is part of the QM region.
    """
    prepared = prepare(DISULFIDE)

    prepared._protonate()

    for chain, seqid in BONDED_CYSTEINES:
        assert "HG" not in hydrogens(prepared.whole, chain, seqid), f"CYS {chain}{seqid} is bonded"
    for chain, seqid in FREE_CYSTEINES:
        assert "HG" in hydrogens(prepared.whole, chain, seqid), f"CYS {chain}{seqid} is free"


def test_protonate_charges_acidic_and_basic_side_chains():
    """
    At pH 7.4 aspartate and glutamate are deprotonated and lysine and arginine are protonated.
    """
    prepared = prepare(TITRATABLE)
    assert prepared.whole

    prepared._protonate()

    for chain in prepared.whole:
        for residue in chain:
            names = {atom.name for atom in residue if atom.element.is_hydrogen}
            if residue.name == "ASP":
                assert ASPARTATE_PROTON not in names
            if residue.name == "GLU":
                assert "HE2" not in names
            if residue.name == "LYS":
                assert LYSINE_PROTONS <= names


@pytest.mark.parametrize("name", COMPLEXES)
def test_protonate_leaves_histidine_neutral(name):
    """
    Histidine is the one residue with pKa near 7.4, and no histidine in the dataset comes out
        doubly protonated at that pH. A HIS carrying both HD1 and HE2 is HIP, a +1 that would show
        up in q_A, so the tautomer choice must stay between HID and HIE.
    """
    prepared = prepare(name)
    assert prepared.whole

    prepared._protonate()

    for chain in prepared.whole:
        for residue in chain:
            if residue.name != "HIS":
                continue
            names = {atom.name for atom in residue if atom.element.is_hydrogen}
            assert not set(HISTIDINE_PROTONS) <= names, (
                f"HIS {verify.identifier(chain, residue)} is doubly protonated"
            )
            assert names & set(HISTIDINE_PROTONS), (
                f"HIS {verify.identifier(chain, residue)} has neither ring hydrogen"
            )


def test_protonate_uses_the_configured_pH():
    """
    PDBFixer defaults to 7.0, so a run that ignored the field would be indistinguishable from one 
        at 7.4 unless the two are compared at a pH far enough apart to flip a residue.

    Below aspartate's pKa the side chain takes HD2 and histidine becomes doubly protonated.
    """
    physiological = prepare(SMALL)
    physiological._protonate()

    acidic = prepare(SMALL)
    acidic.pH = ACIDIC_pH
    acidic._protonate()

    assert count_hydrogens(acidic.whole) > count_hydrogens(physiological.whole)

    def aspartates(prepared):
        return [
            {atom.name for atom in residue if atom.element.is_hydrogen}
            for chain in prepared.whole
            for residue in chain
            if residue.name == "ASP"
        ]

    assert aspartates(physiological), "fixture must contain an aspartate to compare"
    assert all(ASPARTATE_PROTON not in names for names in aspartates(physiological))
    assert all(ASPARTATE_PROTON in names for names in aspartates(acidic))
