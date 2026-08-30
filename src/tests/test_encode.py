"""
RHF over the first bin of the eligible set.

The bin is Q1 of the size quartiles with |q_A| <= 1, which are the cheapest SCF in the set and the least
    distorted by the cut.

None of these can be solved here. One Fock build of the smallest, 7USH_82V at 1733 basis functions,
    is 227 s on twelve cores, so a converged SCF is one to two hours. The tests are in three tiers:

    - preparation and the encoding carry the same charge, spin, electrons, and atom order.
    - a full SCF over ACE-VAL-NME, one capped run lifted out of a prepared cutout.
    - the subset itself, marked hpc and skipped unless asked for, for the cluster.
"""

import functools

import gemmi
import numpy as np
import pytest
from pyscf import gto, scf

from conftest import paths
from encode import EncodeProtein, EncodingError
from prepare import PrepareComplex, PrepareError

# Q1, |q_A| <= 1. The nine left out are 7BJJ_TVW,
# 6YQW_82I, 7R59_I5F, 7LOE_Y84, 7W05_GMP, 7PJQ_OWH, 7XI7_4RI, 7EBG_J0L and 7Z1Q_NIO.
SUBSET = ["7USH_82V", "7W06_ITN", "7R9N_F97"]

# Heavy atoms, net charge and electrons, from the screen in out/filter.csv.
EXPECTED = {
    "7USH_82V": (157, -1, 1188),
    "7W06_ITN": (168, +1, 1286),
    "7R9N_F97": (169, -1, 1314),
}

# 6-31G puts about 11.1 functions on a heavy atom of a cutout of this composition.
FUNCTIONS_PER_HEAVY_ATOM = (9.0, 13.0)

# A capped run lifted whole out of 5S8I_2LY's cutout, which is one chain of ten of them.
FRAGMENT = "5S8I_2LY"
FRAGMENT_SLICE = (4, 7)
FRAGMENT_RESIDUES = ["ACE", "VAL", "NME"]
FRAGMENT_ATOMS = 28
FRAGMENT_ELECTRONS = 94

# The SCF is a stationary point of a smooth functional, so two runs of the same input agree.
REPRODUCIBLE = 1e-9

# Canonical orthogonalisation drops any AO the overlap says is redundant, which would renumber
# every atom after it and silently move the orbitals AVAS was asked for. 6-31G is far from that;
# the measured minimum over the bin is 7e-4.
CONDITIONING = 1e-6


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

# --------------------------------------------------------------------------------------------
# Contract between Preperation and Encoding.
# --------------------------------------------------------------------------------------------

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
    assert prepared.charge
    heavy, net, electrons = EXPECTED[name]

    assert prepared.heavy_atoms == heavy
    assert prepared.charge == net
    assert abs(prepared.charge) <= 1
    assert prepared.electrons == electrons


@pytest.mark.parametrize("name", SUBSET)
def test_rhf_is_handed_the_charge_the_preparation_settled(name):
    """
    The SCF solves the cutout at q_A, not the neutral molecule of the same geometry.
    """
    prepared = prepare(name)
    encoded = EncodeProtein(prepared)

    encoded.molecule()

    assert encoded.mol.charge == prepared.charge
    assert encoded.mol.spin == prepared.spin == 0
    assert encoded.mol.nelectron == prepared.electrons
    # RHF needs N_alpha = N_beta, which is what an even count and spin 0 buy.
    assert not encoded.mol.nelectron % 2


@pytest.mark.parametrize("name", SUBSET)
def test_rhf_keeps_the_atom_order_avas_addresses_by(name):
    """
    PySCF numbers atoms in the order it is given them, and AVAS asks for orbitals by that number.
    """
    prepared = prepare(name)
    encoded = EncodeProtein(prepared)

    encoded.molecule()

    expected = elements(prepared.reduced)
    assert encoded.mol.natm == len(expected)
    assert [encoded.mol.atom_symbol(i) for i in range(encoded.mol.natm)] == expected


@pytest.mark.parametrize("name", SUBSET)
def test_rhf_never_holds_the_two_electron_integrals_in_memory(name):
    """
    A cutout this size has to be solved integral-direct.

    Held in memory the two-electron integrals of the smallest member run to nine terabytes.
    """
    prepared = prepare(name)
    encoded = EncodeProtein(prepared)

    encoded.molecule()

    incore = encoded.mol.nao ** 4 * 8
    assert incore > encoded.mol.max_memory * 1e6
    # Building the mean field settles this; solving it is what costs.
    assert scf.RHF(encoded.mol).direct_scf


@pytest.mark.parametrize("name", SUBSET)
def test_subset_cutouts_carry_no_redundant_basis_functions(name):
    """
    The AO basis of each cutout is well enough conditioned that nothing is dropped.

    PySCF discards AOs the overlap matrix reports as redundant. That renumbers everything after
        them, and AVAS addresses orbitals by number.
    """
    prepared = prepare(name)
    encoded = EncodeProtein(prepared)

    encoded.molecule()

    overlap = np.linalg.eigvalsh(encoded.mol.intor("int1e_ovlp"))
    assert overlap.min() > CONDITIONING


def test_rhf_refuses_a_complex_that_was_never_prepared():
    """
    There is no cutout to solve before the pipeline has run.
    """
    encoded = EncodeProtein(PrepareComplex(*paths(FRAGMENT)))

    with pytest.raises(PrepareError):
        encoded.RHF()


# --------------------------------------------------------------------------------------------
# A real SCF, on a system small enough to solve here.
# --------------------------------------------------------------------------------------------


def test_rhf_converges_on_a_capped_fragment():
    """
    The whole path runs and reaches a converged closed-shell solution.
    """
    prepared = fragment()

    encoded = solved(prepared)

    assert encoded.mean_field.converged
    assert encoded.energy == pytest.approx(encoded.mean_field.e_tot)
    assert encoded.energy < 0
    assert set(np.unique(encoded.mean_field.mo_occ)) <= {0.0, 2.0}
    assert int(encoded.mean_field.mo_occ.sum()) == prepared.electrons


def test_rhf_puts_the_electrons_where_the_charge_says():
    """
    The converged density integrates to the electron count the preparation derived.
    """
    prepared = fragment()

    encoded = solved(prepared)

    density = encoded.mean_field.make_rdm1()
    overlap = encoded.mol.intor("int1e_ovlp")
    assert np.trace(density @ overlap) == pytest.approx(prepared.electrons, abs=1e-6)


def test_rhf_reaches_a_minimum_rather_than_a_saddle_point():
    """
    The converged solution is stable, so the orbitals AVAS is given are the ground state's. An 
        internally unstable one has a lower solution it did not find.
    """
    encoded = solved(fragment())

    internal, _ = encoded.mean_field.stability()

    assert np.allclose(internal, encoded.mean_field.mo_coeff)


def test_rhf_is_reproducible():
    """
    The same cutout gives the same energy twice.
    """
    prepared = fragment()

    first = EncodeProtein(prepared)
    first.RHF()
    second = EncodeProtein(prepared)
    second.RHF()

    assert first.energy == pytest.approx(second.energy, abs=REPRODUCIBLE)
    assert np.allclose(first.mean_field.mo_energy, second.mean_field.mo_energy, atol=REPRODUCIBLE)


def test_molecule_carries_the_cutout_geometry_unchanged():
    """
    The molecule holds the cutout's own coordinates, in the cutout's own units.

    gemmi keeps angstroms and PySCF reads angstroms unless told otherwise, so nothing has to be
        converted. That is worth pinning rather than assuming: a silent unit change would scale
        every bond length and still converge.
    """
    prepared = fragment()
    encoded = EncodeProtein(prepared)

    encoded.molecule()
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


def test_rhf_rejects_an_scf_that_did_not_converge():
    """
    An unconverged SCF is not a result.
    """
    prepared = fragment()
    encoded = EncodeProtein(prepared)
    encoded.max_cycle = 1

    with pytest.raises(EncodingError):
        encoded.RHF()


# --------------------------------------------------------------------------------------------
# The bin itself. Hours each, for the cluster.
# --------------------------------------------------------------------------------------------


@pytest.mark.hpc
@pytest.mark.parametrize("name", SUBSET)
def test_rhf_converges_over_the_subset(name):
    """
    A converged, stable, closed-shell SCF over a cutout of the bin.
    """
    prepared = prepare(name)

    encoded = solved(prepared)

    assert encoded.mean_field.converged
    assert set(np.unique(encoded.mean_field.mo_occ)) <= {0.0, 2.0}
    assert int(encoded.mean_field.mo_occ.sum()) == prepared.electrons

    internal, _ = encoded.mean_field.stability()
    assert np.allclose(internal, encoded.mean_field.mo_coeff)


@pytest.mark.hpc
@pytest.mark.parametrize("name", SUBSET)
def test_subset_orbitals_leave_a_gap_to_correlate_across(name):
    """
    The converged solution has a real gap between the highest occupied and lowest virtual orbital.

    A truncated, charged cutout can close it, and a gap near zero means the closed-shell single
        determinant is the wrong starting point. The SCF may still converge, but the active space
        AVAS builds on it will not describe what is happening.
    """
    # The solve this shares with the test above is the whole of the job's cost.
    encoded = solved(prepare(name))

    energies, occupations = encoded.mean_field.mo_energy, encoded.mean_field.mo_occ
    gap = energies[occupations == 0].min() - energies[occupations > 0].max()
    assert gap > 0
