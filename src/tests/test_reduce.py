"""
Truncation and capping for EncodeProtein._reduce

The median complex breaks into many runs of residues, so most need a cap on both sides. A run 
    separated from the next by exactly one residue would have that residue's backbone claimed by
    an NME on one side and an ACE on the other, placing the same atom twice at the same coordinates.

Caps are heavy atoms that affect the complexes tractability for RHF, so the size limit must be 
    checked after capping.

_fix rebuilds side chains, and a rebuilt side chain can reach a pose that the deposited one did
    not. The residue then enters the cutout carrying invented coordinates, having escaped the
    incomplete-residue rule because it was outside the provisional cutout when _verify ran.

Five real complexes cover these, all accepted by _verify:

    5S8I_2LY    12 residues in 9 runs and no single-residue gap, the clean baseline
    7WPW_F15    eight single-residue gaps, the most of any complex that stays inside the cap
    6TW5_9M2    chain C is retained to residue 410, where the chain ends and there is nothing to
                    build a cap out of
    8FLV_ZB9    GLN A412 is deposited without the end of its side chain; rebuilt, it closes from
                    4.52 Å of a pose to 3.34 Å
    7VBU_6I4    47 residues holding 422 heavy atoms, which 42 caps push to 527
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree # pyright: ignore[reportAttributeAccessIssue]

from conftest import paths
from encode import EncodeProtein, OutOfScopeError, OutOfScopeErrorType
from utils import verify

SIMPLE = "5S8I_2LY"
GAPPED = "7WPW_F15"
TERMINUS = "6TW5_9M2"
DRIFTED = "8FLV_ZB9"
OVERSIZED = "7VBU_6I4"

COMPLEXES = [SIMPLE, GAPPED, TERMINUS]

CAPS = frozenset({"ACE", "NME"})
ACE_ATOMS = {"C", "O", "CH3"}
# OpenMM names an NME's methyl carbon "C", not "CH3", and places its hydrogens off that name.
NME_ATOMS = {"N", "C"}

# Every residue that sits alone between two retained runs in 7WPW_F15. Each is one peptide bond from
# retained residues on both sides, so capping around it would duplicate its backbone.
BRIDGED = [
    ("A", 24), ("A", 39), ("A", 54), ("A", 56),
    ("A", 77), ("A", 105), ("A", 116), ("A", 127),
]

# 6TW5_9M2 chain C ends at residue 410 and the cutout reaches it. Protonation has given it the OXT
# of a free C-terminus and left the carboxylate charged.
CHAIN_TERMINUS = ("C", 410)
TERMINAL_ATOMS = {"OXT"}

# GLN A412 of 8FLV_ZB9 is deposited as backbone plus CB. PDBFixer rebuilds the rest of the side chain.
REBUILT_RESIDUE = ("A", 412)
REBUILT_ATOMS = {"CG", "CD", "OE1", "NE2"}

# No two atoms in a real structure sit this close. A shared backbone would put two at zero.
CLASH = 0.5


def encoding(name):
    """
    A complex carried up to the point _reduce is called.
    """
    encode = EncodeProtein(*paths(name))
    encode._fetch()
    encode._verify()
    encode._fix()
    encode._clean()
    encode._protonate()
    return encode


def within_cutoff(encode):
    """
    The residues the distance rule selects, before any bridging or capping.
    """
    return {
        verify.identifier(chain, residue)
        for chain, residue, _, _ in verify.cutout(
            encode.whole, encode._pose_coordinates(), encode.cutoff
        )
    }


def sequence_index(model):
    """
    Where each residue sits in its chain, to indicate if two residues are peptide-bonded
        neighbours rather than consecutive in a truncated selection.
    """
    return {
        verify.identifier(chain, residue): (chain.name, position)
        for chain in model
        for position, residue in enumerate(chain)
    }


def protein_residues(model):
    return [
        (chain, residue)
        for chain in model
        for residue in chain
        if residue.name not in CAPS
    ]


def positions(model):
    return np.array([
        (atom.pos.x, atom.pos.y, atom.pos.z)
        for chain in model
        for residue in chain
        for atom in residue
    ])


def atoms_by_residue(model):
    return {
        verify.identifier(chain, residue): {atom.name for atom in residue}
        for chain in model
        for residue in chain
    }


@pytest.mark.parametrize("name", COMPLEXES)
def test_reduce_keeps_every_residue_within_the_cutoff(name):
    """
    Nothing the distance rule selects is dropped.
    """
    encode = encoding(name)
    expected = within_cutoff(encode)

    encode._reduce()

    assert expected <= {
        verify.identifier(chain, residue) for chain, residue in protein_residues(encode.reduced)
    }


@pytest.mark.parametrize("name", COMPLEXES)
def test_reduce_keeps_nothing_but_the_site(name):
    """
    Every protein residue kept is either within the cutoff itself or bridges two that are.
    """
    encode = encoding(name)
    selected = within_cutoff(encode)
    order = sequence_index(encode.whole)

    encode._reduce()

    for chain, residue in protein_residues(encode.reduced):
        key = verify.identifier(chain, residue)
        if key in selected:
            continue
        name_, position = order[key]
        neighbours = {
            other
            for other in selected
            if order[other][0] == name_ and abs(order[other][1] - position) == 1
        }
        assert len(neighbours) == 2, f"{key} is neither in the cutout nor bridging it"


@pytest.mark.parametrize("name", COMPLEXES)
def test_reduce_preserves_heavy_atom_coordinates(name):
    """
    Truncation moves nothing.
    """
    encode = encoding(name)
    before = {
        (verify.identifier(chain, residue), atom.name): (atom.pos.x, atom.pos.y, atom.pos.z)
        for chain in encode.whole
        for residue in chain
        for atom in residue
        if not atom.element.is_hydrogen
    }

    encode._reduce()

    for chain, residue in protein_residues(encode.reduced):
        for atom in residue:
            if atom.element.is_hydrogen:
                continue
            key = (verify.identifier(chain, residue), atom.name)
            assert key in before, f"{key} was not in the protonated structure"
            assert (atom.pos.x, atom.pos.y, atom.pos.z) == pytest.approx(before[key], abs=1e-3)


@pytest.mark.parametrize("name", COMPLEXES)
def test_reduce_preserves_protonation(name):
    """
    The hydrogens _protonate placed are unchanged.
    """
    encode = encoding(name)
    before = atoms_by_residue(encode.whole)

    encode._reduce()

    for chain, residue in protein_residues(encode.reduced):
        key = verify.identifier(chain, residue)
        assert {atom.name for atom in residue} == before[key]


def test_reduce_bridges_a_single_residue_gap():
    """
    A residue alone between two retained runs is kept whole rather than capped around.

    Capping both sides would take its N and CA into an ACE and its C, O and CA into an NME, placing
        CA twice at one position.
    """
    encode = encoding(GAPPED)

    encode._reduce()

    kept = {
        (chain.name, residue.seqid.num)
        for chain, residue in protein_residues(encode.reduced)
    }
    for key in BRIDGED:
        assert key in kept, f"{key} sits alone between two retained runs and must be bridged"


@pytest.mark.parametrize("name", COMPLEXES)
def test_reduce_places_no_duplicate_atoms(name):
    """
    No two atoms occupy the same point.
    """
    encode = encoding(name)

    encode._reduce()

    coordinates = positions(encode.reduced)
    pairs = cKDTree(coordinates).query_pairs(CLASH)
    assert not pairs, f"{len(pairs)} atom pair(s) closer than {CLASH} Å"


@pytest.mark.parametrize("name", COMPLEXES)
def test_reduce_caps_every_cut(name):
    """
    Wherever the truncation breaks a peptide bond, a cap stands in for the residue that was removed.
        An uncapped backbone N or C is a radical, which no closed-shell singlet reference describes.

    A chain terminus is not a break: nothing was removed there, so there is nothing to cap and no
        neighbouring backbone to build a cap out of.
    """
    encode = encoding(name)
    order = sequence_index(encode.whole)
    length = {chain.name: len(chain) for chain in encode.whole}

    encode._reduce()

    for chain in encode.reduced:
        residues = list(chain)
        for position, residue in enumerate(residues):
            if residue.name in CAPS:
                continue
            key = verify.identifier(chain, residue)
            name_, index = order[key]
            for step, cap in ((-1, "ACE"), (1, "NME")):
                if not 0 <= index + step < length[name_]:
                    continue
                neighbour = residues[position + step] if 0 <= position + step < len(residues) else None
                if neighbour is not None and neighbour.name not in CAPS:
                    other = order[verify.identifier(chain, neighbour)]
                    if other == (name_, index + step):
                        continue
                assert neighbour is not None and neighbour.name == cap, (
                    f"{residue.name} {key} is cut on the {'N' if step < 0 else 'C'} side "
                    f"without {cap}"
                )


@pytest.mark.parametrize("name", COMPLEXES)
def test_reduce_builds_complete_caps(name):
    """
    A cap is a whole acetyl or N-methyl group, hydrogens included. A cap missing its methyl
        hydrogens induces the same open valence the cap should close.
    """
    encode = encoding(name)

    encode._reduce()

    caps = [
        residue
        for chain in encode.reduced
        for residue in chain
        if residue.name in CAPS
    ]
    assert caps, "a fragmented cutout cannot need no caps"
    for cap in caps:
        names = {atom.name for atom in cap}
        expected = ACE_ATOMS if cap.name == "ACE" else NME_ATOMS
        assert expected <= names, f"{cap.name} lacks {sorted(expected - names)}"
        assert any(atom.element.is_hydrogen for atom in cap), f"{cap.name} has no hydrogens"


def test_reduce_leaves_a_chain_terminus_uncapped():
    """
    Where a chain ends, the terminus protonation gave it is kept.

    A cap replaces a residue the truncation removed, taking its backbone coordinates from the
        structure. At a chain end there is no such residue and no such coordinates, so capping would
        mean inventing an acetyl group.

    The cost is a charge, and the deposited model stopping short is indistinguishable from the chain
        really ending. Separating the cases needs the SEQRES records.
    """
    encode = encoding(TERMINUS)

    encode._reduce()

    chain_name, seqid = CHAIN_TERMINUS
    for chain in encode.reduced:
        residues = list(chain)
        for position, residue in enumerate(residues):
            if (chain.name, residue.seqid.num) != (chain_name, seqid):
                continue
            names = {atom.name for atom in residue}
            assert position == len(residues) - 1, "the chain end was capped"
            assert TERMINAL_ATOMS <= names
            assert "HXT" not in names, "the terminal carboxylate should carry the charge"
            return
    pytest.fail(f"{chain_name}{seqid} was not retained")


@pytest.mark.parametrize("name", COMPLEXES)
def test_reduce_keeps_the_capped_cutout_within_the_size_cap(name):
    """
    Ensure the capped cutout stays in the size cap.
    """
    encode = encoding(name)

    encode._reduce()

    heavy = [
        atom
        for chain in encode.reduced
        for residue in chain
        for atom in residue
        if not atom.element.is_hydrogen
    ]
    assert len(heavy) <= encode.size_cap


def test_reduce_rejects_a_cutout_the_caps_push_over_the_size_cap():
    """
    Reject a cutout that grows beyond the size limit from capping.
    """
    encode = encoding(OVERSIZED)

    with pytest.raises(OutOfScopeError) as rejected:
        encode._reduce()
    assert rejected.value.error_type is OutOfScopeErrorType.SIZE_CAP


def test_reduce_rejects_a_residue_repaired_into_the_cutout():
    """
    A residue whose rebuilt side chain reaches a pose is out of scope.
    """
    encode = encoding(DRIFTED)
    chain_name, seqid = REBUILT_RESIDUE
    assert REBUILT_ATOMS <= atoms_by_residue(encode.whole)[(chain_name, seqid, " ")]

    with pytest.raises(OutOfScopeError) as rejected:
        encode._reduce()
    assert rejected.value.error_type is OutOfScopeErrorType.INCOMPLETE_RESIDUE
