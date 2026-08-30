"""
Net charge and electron count for PrepareComplex.

Neither _calculate_charge nor _verify_num_electrons writes anything back into the cutout.

The charge is determined by the set of hydrogens the residue holds.

Three complexes cover the cases, all accepted by _reduce:

    5S8I_2LY    one charged side chain and no free terminus, the smallest cutout of the three
    7WPW_F15    four arginines and a lysine against two aspartates, a cutout that is net positive
    6TW5_9M2    chain C is kept to residue 410, an uncapped C-terminus carrying a fourth charge
"""

import functools

import pytest

from conftest import paths
from prepare import PrepareComplex, PrepareError

SIMPLE = "5S8I_2LY"
CHARGED = "7WPW_F15"
TERMINUS = "6TW5_9M2"

COMPLEXES = [SIMPLE, CHARGED, TERMINUS]

# What each cutout comes to once every residue is counted. 6TW5_9M2 is -1 from its side chains and
# -1 again from the terminus.
NET_CHARGE = {SIMPLE: -1, CHARGED: 3, TERMINUS: -2}

CAPS = frozenset({"ACE", "NME"})

# 6TW5_9M2 keeps LEU C410, where chain C ends. _reduce leaves a chain end uncapped, so protonation's
# OXT and the missing HXT stay, and the carboxylate keeps its charge.
FREE_TERMINUS = ("C", 410)


@functools.lru_cache(maxsize=None)
def prepare(name):
    """
    A complex carried to the end of _reduce.

    Cached because the pipeline is far slower than the three methods under test, and none of them
        changes the structure they read.
    """
    _prepare = PrepareComplex(*paths(name))
    _prepare._fetch()
    _prepare._verify()
    _prepare._fix()
    _prepare._clean()
    _prepare._protonate()
    _prepare._reduce()
    return _prepare


def formal_charge(model, termini=True):
    """
    The charge of a cutout read off the hydrogens each residue holds.

    An independent reading of the same chemistry the implementation is asked for: arginine is
        charged whatever the pH, lysine is charged while it keeps all three hydrogens on NZ,
        aspartate and glutamate are charged while they lack the one on the carboxylate, and
        histidine runs from -1 to +1 on the two it can carry. A cap is neutral. A chain end that
        kept an OXT without an HXT carries the charge protonation left on it.
    """
    total = 0
    for chain in model:
        residues = list(chain)
        for position, residue in enumerate(residues):
            if residue.name in CAPS:
                continue
            names = {atom.name for atom in residue}
            if residue.name == "ARG":
                total += 1
            elif residue.name == "LYS":
                total += {"HZ1", "HZ2", "HZ3"} <= names
            elif residue.name == "ASP":
                total -= "HD2" not in names
            elif residue.name == "GLU":
                total -= "HE2" not in names
            elif residue.name == "HIS":
                total += ("HD1" in names) + ("HE2" in names) - 1
            if not termini:
                continue
            if position == 0 and {"H2", "H3"} <= names:
                total += 1
            if position == len(residues) - 1 and "OXT" in names and "HXT" not in names:
                total -= 1
    return total


def nuclear_charge(model):
    return sum(
        atom.element.atomic_number
        for chain in model
        for residue in chain
        for atom in residue
    )


def atoms(model):
    return [
        (atom.element.name, (atom.pos.x, atom.pos.y, atom.pos.z))
        for chain in model
        for residue in chain
        for atom in residue
    ]


@pytest.mark.parametrize("name", COMPLEXES)
def test_calculate_charge_counts_every_charged_residue(name):
    """
    q_A is the sum of the formal charges of what the cutout holds.
    """
    prepared = prepare(name)

    prepared._calculate_charge()

    assert prepared.charge == formal_charge(prepared.reduced)
    assert prepared.charge == NET_CHARGE[name]


def test_calculate_charge_counts_a_free_terminus():
    """
    An uncapped chain end brings its charge into q_A.
    """
    prepared = prepare(TERMINUS)
    chain_name, seqid = FREE_TERMINUS
    terminus = [
        residue
        for chain in prepared.reduced
        for residue in chain
        if (chain.name, residue.seqid.num) == (chain_name, seqid)
    ]
    assert terminus and "OXT" in {atom.name for atom in terminus[0]}

    prepared._calculate_charge()

    assert formal_charge(prepared.reduced, termini=False) == -1
    assert prepared.charge == -2


@pytest.mark.parametrize("name", COMPLEXES)
def test_verify_num_electrons_accepts_an_even_count(name):
    """
    N_e = sum_I Z_I - q_A, over every atom of the cutout, hydrogens included.
    """
    prepared = prepare(name)
    prepared._calculate_charge()

    prepared._verify_num_electrons()

    assert prepared.electrons == nuclear_charge(prepared.reduced) - prepared.charge
    assert not prepared.electrons % 2


def test_verify_num_electrons_rejects_an_odd_count():
    """
    An odd count means the preparation is wrong.
    """
    prepared = prepare(SIMPLE)
    prepared._calculate_charge()
    prepared.charge += 1

    with pytest.raises(PrepareError):
        prepared._verify_num_electrons()
