"""
The molecule the SCF is solved over, and the contract between preparation and encoding.
"""

import numpy as np
import pytest
from pyscf import gto, scf

from cutouts import (
    EXPECTED,
    FRAGMENT_ATOMS,
    FRAGMENT_ELECTRONS,
    FRAGMENT_RESIDUES,
    SUBSET,
    elements,
    fragment,
    prepare,
)
from encode import EncodeProtein

# 6-31G puts about 11.1 functions on a heavy atom of a cutout of this composition, so every member
# of the bin lands between roughly 1000 and 2300. Wide enough to be a sanity check, not a pin.
FUNCTIONS_PER_HEAVY_ATOM = (9.0, 13.0)

# Canonical orthogonalisation drops any AO the overlap says is redundant, which would renumber
# every atom after it and silently move the orbitals AVAS was asked for. 6-31G is far from that;
# the measured minimum over the bin is 7e-4.
CONDITIONING = 1e-6


def test_the_fragment_is_a_capped_run():
    """
    The fragment the solvable tests rely on is a real capped peptide, not a slice through one.
    """
    prepared = fragment()

    assert [residue.name for chain in prepared.reduced for residue in chain] == FRAGMENT_RESIDUES
    assert len(elements(prepared.reduced)) == FRAGMENT_ATOMS
    assert prepared.charge == 0
    assert prepared.electrons == FRAGMENT_ELECTRONS


@pytest.mark.parametrize("name", SUBSET)
def test_the_subset_is_the_bin_the_screen_found(name):
    """
    Each member of the bin is the size and charge the screen recorded.
    """
    prepared = prepare(name)
    heavy, net, electrons = EXPECTED[name]

    assert prepared.heavy_atoms == heavy
    assert prepared.charge == net
    assert abs(prepared.charge) <= 1
    assert prepared.electrons == electrons


@pytest.mark.parametrize("name", SUBSET)
def test_molecule_is_handed_the_charge_the_preparation_settled(name):
    """
    The molecule is the cutout at q_A.
    """
    prepared = prepare(name)
    encoded = EncodeProtein(prepared)

    encoded._molecule()

    assert encoded.mol.charge == prepared.charge
    assert encoded.mol.spin == prepared.spin == 0
    assert encoded.mol.nelectron == prepared.electrons
    # RHF needs N_alpha = N_beta, which is what an even count and spin 0 buy.
    assert not encoded.mol.nelectron % 2


@pytest.mark.parametrize("name", SUBSET)
def test_molecule_keeps_the_atom_order_avas_addresses_by(name):
    """
    PySCF numbers atoms in the order it is given them, and AVAS asks for orbitals by that number.
    """
    prepared = prepare(name)
    encoded = EncodeProtein(prepared)

    encoded._molecule()

    expected = elements(prepared.reduced)
    assert encoded.mol.natm == len(expected)
    assert [encoded.mol.atom_symbol(i) for i in range(encoded.mol.natm)] == expected


@pytest.mark.parametrize("name", SUBSET)
def test_molecule_uses_the_basis_the_scope_verified(name):
    """
    The basis is the one _verify checked every element against.
    """
    prepared = prepare(name)
    encoded = EncodeProtein(prepared)

    encoded._molecule()

    assert encoded.mol.basis == prepared.basis
    low, high = FUNCTIONS_PER_HEAVY_ATOM
    assert low < encoded.mol.nao / prepared.heavy_atoms < high


@pytest.mark.parametrize("name", SUBSET)
def test_molecule_is_too_large_to_hold_its_integrals(name):
    """
    A cutout this size has to be solved integral-direct.

    Held in memory the two-electron integrals of the smallest member run to nine terabytes.
    """
    prepared = prepare(name)
    encoded = EncodeProtein(prepared)

    encoded._molecule()

    incore = encoded.mol.nao ** 4 * 8
    assert incore > encoded.mol.max_memory * 1e6
    # Building the mean field settles this; solving it is what costs.
    assert scf.RHF(encoded.mol).direct_scf


@pytest.mark.parametrize("name", SUBSET)
def test_molecule_carries_no_redundant_basis_functions(name):
    """
    The AO basis of each cutout is well enough conditioned that nothing is dropped.

    PySCF discards AOs the overlap matrix reports as redundant. That renumbers everything after
        them, and AVAS addresses orbitals by number.
    """
    prepared = prepare(name)
    encoded = EncodeProtein(prepared)

    encoded._molecule()

    overlap = np.linalg.eigvalsh(encoded.mol.intor("int1e_ovlp"))
    assert overlap.min() > CONDITIONING


def test_molecule_carries_the_cutout_geometry_unchanged():
    """
    The molecule holds the cutout's own coordinates, in the cutout's own units.
    """
    prepared = fragment()
    encoded = EncodeProtein(prepared)

    encoded._molecule()
    written = gto.M(
        atom=[
            (atom.element.name, (atom.pos.x, atom.pos.y, atom.pos.z))
            for chain in prepared.reduced
            for residue in chain
            for atom in residue
        ],
        charge=prepared.charge,
        spin=prepared.spin,
        basis=prepared.basis,
    )

    assert encoded.mol.nao == written.nao
    assert np.allclose(encoded.mol.atom_coords(), written.atom_coords())
