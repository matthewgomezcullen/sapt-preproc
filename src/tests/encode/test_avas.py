"""
Choosing the active space with AVAS.

AVAS projects the converged occupied and virtual spaces onto a set of target atomic orbitals and
    keeps whatever has weight on them. The targets are the the method's judgement, so we test the 
    rule that picks them

The default rule, from README:

    for every non-cap heavy atom of the cutout, take its shortest distance to any heavy atom of any
    candidate pose; keep C, N, O, S and P within the cutoff; target the valence p shell, 2p for
    C/N/O and 3p for S/P; address each by its zero-based PySCF atom index.

PySCF matches AO labels by pattern, so an index-prefixed label has to resolve to that atom

The size of what comes back is set by the number of targets. Capping it to what SHCI can run
    is MP2's job, tested in test_mp2.py.

We run the fragment, ACE-VAL-NME, lifted out of a cutout. The bin is marked hpc.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree # pyright: ignore[reportAttributeAccessIssue]

from cutouts import SUBSET, fragment, prepare, solved
from encode import EncodeProtein, EncodingError
from prepare import PrepareComplex, PrepareError
from conftest import paths

# A cap stands in for a residue the truncation removed. It is an artefact of the cut, so its
# orbitals are not part of the chemistry the active space is meant to describe.
CAPS = frozenset({"ACE", "NME"})

# The valence p shell of each element the rule keeps.
VALENCE = {"C": "2p", "N": "2p", "O": "2p", "S": "3p", "P": "3p"}

# A p shell is three atomic orbitals.
PER_SHELL = 3


def targeted(prepared, cutoff):
    """
    The atoms the rule should pick.

    A direct reading of README against the cutout and the poses.
    """
    poses = cKDTree(prepared._pose_coordinates())
    picked = []
    for index, (_, residue, atom) in enumerate(prepared.atoms()):
        if residue.name in CAPS or atom.element.name not in VALENCE:
            continue
        distance, _ = poses.query([[atom.pos.x, atom.pos.y, atom.pos.z]])
        if distance[0] <= cutoff:
            picked.append((index, atom.element.name))
    return picked


# --------------------------------------------------------------------------------------------
# The rule that picks the targets. No SCF, so the whole bin is in reach.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", SUBSET)
def test_target_orbitals_name_the_atoms_the_rule_selects(name):
    """
    The targets are the non-cap C, N, O, S and P within the cutoff of a pose.
    """
    prepared = prepare(name)
    encoded = EncodeProtein(prepared)
    encoded._molecule()

    targets = encoded._generate_target_orbitals()

    expected = targeted(prepared, encoded.cutoff)
    assert [int(target.split()[0]) for target in targets] == [index for index, _ in expected]
    assert [target.split()[1] for target in targets] == [symbol for _, symbol in expected]


@pytest.mark.parametrize("name", SUBSET)
def test_target_orbitals_leave_out_the_caps(name):
    """
    No cap is targeted.
    """
    prepared = prepare(name)
    encoded = EncodeProtein(prepared)
    encoded._molecule()

    targets = encoded._generate_target_orbitals()

    atoms = prepared.atoms()
    for target in targets:
        _, residue, _ = atoms[int(target.split()[0])]
        assert residue.name not in CAPS


@pytest.mark.parametrize("name", SUBSET)
def test_target_orbitals_are_the_valence_p_shell(name):
    """
    Each target names its element's valence p shell: 2p for C, N and O, 3p for S and P.

    Hydrogen has none and is not targeted at all.
    """
    prepared = prepare(name)
    encoded = EncodeProtein(prepared)
    encoded._molecule()

    targets = encoded._generate_target_orbitals()

    assert targets
    for target in targets:
        _, symbol, shell = target.split()
        assert symbol in VALENCE
        assert shell == VALENCE[symbol]


@pytest.mark.parametrize("name", SUBSET)
def test_every_target_orbital_addresses_exactly_its_own_atom(name):
    """
    A target resolves to that atom's three p orbitals and to nothing else.
    """
    prepared = prepare(name)
    encoded = EncodeProtein(prepared)
    encoded._molecule()

    targets = encoded._generate_target_orbitals()

    labels = encoded.mol.ao_labels()
    for target in targets:
        found = encoded.mol.search_ao_label(target)
        assert len(found) == PER_SHELL
        assert {int(labels[index].split()[0]) for index in found} == {int(target.split()[0])}


@pytest.mark.parametrize("name", SUBSET)
def test_target_orbitals_are_ordered_and_unique(name):
    """
    The targets come back once each, in the order PySCF numbers the atoms.
    """
    prepared = prepare(name)
    encoded = EncodeProtein(prepared)
    encoded._molecule()

    targets = encoded._generate_target_orbitals()

    indices = [int(target.split()[0]) for target in targets]
    assert len(set(targets)) == len(targets)
    assert indices == sorted(indices)


@pytest.mark.parametrize("name", SUBSET)
def test_target_orbitals_are_deterministic(name):
    """
    The same cutout gives the same targets.
    """
    prepared = prepare(name)
    encoded = EncodeProtein(prepared)
    encoded._molecule()

    assert encoded._generate_target_orbitals() == encoded._generate_target_orbitals()


def test_target_orbitals_refuse_a_complex_with_no_molecule():
    """
    There is nothing to address before the molecule is built.
    """
    encoded = EncodeProtein(PrepareComplex(*paths("5S8I_2LY")))

    with pytest.raises(PrepareError):
        encoded._generate_target_orbitals()


# --------------------------------------------------------------------------------------------
# AVAS itself, on the fragment. Two of its carbons fall within the cutoff of a pose.
# --------------------------------------------------------------------------------------------


def test_avas_returns_a_closed_shell_active_space():
    """
    The active space is one a restricted reference can describe.
    """
    encoded = solved(fragment())

    encoded.AVAS()

    assert encoded.ncas >= 1
    assert encoded.nelecas >= 0
    assert not encoded.nelecas % 2
    assert encoded.nelecas <= 2 * encoded.ncas


def test_avas_keeps_every_orbital_of_the_molecule():
    """
    AVAS rotates the molecular orbitals, it does not discard any.
    """
    encoded = solved(fragment())

    encoded.AVAS()

    orbitals = encoded.orbitals
    assert orbitals.shape == (encoded.mol.nao, encoded.mol.nao)
    overlap = encoded.mol.intor("int1e_ovlp")
    assert np.allclose(orbitals.T @ overlap @ orbitals, np.eye(encoded.mol.nao), atol=1e-8)


def test_avas_active_space_is_smaller_than_the_molecule():
    """
    The active space is a reduction.
    """
    encoded = solved(fragment())

    encoded.AVAS()

    assert encoded.ncas < encoded.mol.nao
    assert encoded.nelecas < encoded.mol.nelectron


def test_avas_grows_with_the_targets_it_is_given():
    """
    More target orbitals means a larger active space.
    """
    encoded = solved(fragment())
    carbons = [
        index
        for index in range(encoded.mol.natm)
        if encoded.mol.atom_symbol(index) == "C"
    ]

    sizes = []
    for count in (1, 2, 4):
        encoded.AVAS(targets=[f"{index} C 2p" for index in carbons[:count]])
        sizes.append(encoded.ncas)

    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_avas_is_reproducible():
    """
    The same converged cutout gives the same active space twice.
    """
    encoded = solved(fragment())

    encoded.AVAS()
    first = (encoded.ncas, encoded.nelecas, encoded.orbitals.copy())
    encoded.AVAS()

    assert (encoded.ncas, encoded.nelecas) == first[:2]
    assert np.allclose(encoded.orbitals, first[2])


def test_avas_refuses_before_the_scf_has_run():
    """
    AVAS projects the converged orbitals. There are none before RHF.
    """
    encoded = EncodeProtein(fragment())

    with pytest.raises(EncodingError):
        encoded.AVAS()


# --------------------------------------------------------------------------------------------
# The bin. Needs the SCF first, so hours each, for the cluster.
# --------------------------------------------------------------------------------------------


@pytest.mark.hpc
@pytest.mark.parametrize("name", SUBSET)
def test_avas_builds_an_active_space_over_the_subset(name):
    """
    AVAS runs to completion on a real cutout and returns a consistent closed-shell active space.
    """
    encoded = solved(prepare(name))

    encoded.AVAS()

    assert encoded.ncas >= 1
    assert not encoded.nelecas % 2
    assert encoded.nelecas <= 2 * encoded.ncas
    assert encoded.ncas < encoded.mol.nao
