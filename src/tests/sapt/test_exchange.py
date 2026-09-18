"""
The first order exchange energy in the S^2 approximation, Eq. (14).

A two-particle density matrix sepeartes into its one-particle density matrix plus a cumulant, which 
    a determinant does not have. Eq. (14) is linear in each, so exchange splits in two. Over the 
    separable parts it requires just the two AO densities, and none of its contractions uses 
    D S D = 2D, so the correlated protein's density goes through them as the ligand's does. Only 
    the protein has a cumulant, and it lives in the active space, so what it adds costs O(N_act^4).

Tables III and IV publish the exchange of the paper's water dimer at SAPT(RHF) and at SAPT(CASCI).
    Eq. (14) written out over each monomer's occupied orbitals reproduces both, and is the oracle to
    machine precision.
"""

import itertools

import numpy as np
import pytest
from pyscf import gto

import monomers
import reference
from utils import sapt

# The tables print six decimal places, so half of the last one is the closest a comparison gets.
PUBLISHED = 1e-6  # Hartree

EXACT = 1e-12

# A cumulant element this far from zero is correlation, not rounding.
FRACTIONAL = 1e-3

POSES = 3

DISTINCT_POSES = 1e-5


def restricted_dimer():
    mol_a, mol_b = monomers.water_dimer()
    return (
        mol_a, monomers.solve_water(monomers.EQUILIBRIUM).make_rdm1(),
        mol_b, monomers.solve_water(monomers.COMPRESSED).make_rdm1(),
    )


def correlated_dimer():
    """
    The compressed monomer named first, as the protein is, since only A carries a cumulant.
    """
    return (
        monomers.build_water(monomers.COMPRESSED),
        monomers.correlated_water_density(monomers.COMPRESSED),
        monomers.build_water(monomers.EQUILIBRIUM),
        monomers.solve_water(monomers.EQUILIBRIUM).make_rdm1(),
    )


def water_cumulant():
    return sapt.cumulant(*monomers.correlate_water(monomers.COMPRESSED))


def cumulant_exchange():
    """
    What the compressed monomer's cumulant adds to the exchange of the correlated dimer.
    """
    mol_a, _, mol_b, density_b = correlated_dimer()
    return sapt.cumulant_exchange(
        mol_a, monomers.active_orbitals(monomers.COMPRESSED), water_cumulant(), mol_b, density_b
    )


def reference_dimer():
    return (
        monomers.occupied(monomers.COMPRESSED, correlated=True),
        monomers.occupied(monomers.EQUILIBRIUM, correlated=False),
    )


def test_the_exchange_between_restricted_monomers_is_the_one_the_paper_publishes():
    energy = sapt.exchange(*restricted_dimer())

    assert energy == pytest.approx(monomers.EXCH_RESTRICTED, abs=PUBLISHED)


def test_the_exchange_with_a_correlated_monomer_is_the_one_the_paper_publishes():
    energy = sapt.exchange(*correlated_dimer()) + cumulant_exchange()

    assert energy == pytest.approx(monomers.EXCH_CORRELATED, abs=PUBLISHED)


def test_the_exchange_of_the_densities_is_eq_14_with_every_cumulant_taken_away():
    """
    The AO contractions still give Eq. (14)'s exchange of its separable part precisely. None of 
        them lean on D S D = 2D.
    """
    correlated, restricted = reference_dimer()

    energy = sapt.exchange(*correlated_dimer())

    assert energy == pytest.approx(
        reference.exchange(reference.separable(correlated), restricted), abs=EXACT
    )


def test_the_exchange_does_not_depend_on_which_monomer_is_named_first():
    mol_a, density_a, mol_b, density_b = correlated_dimer()

    assert sapt.exchange(mol_a, density_a, mol_b, density_b) == pytest.approx(
        sapt.exchange(mol_b, density_b, mol_a, density_a), abs=EXACT
    )


def test_the_exchange_falls_away_once_the_monomers_are_apart():
    mol_a, density_a, _, _ = correlated_dimer()
    apart = monomers.build_water(monomers.SEPARATED)
    density_apart = monomers.solve_water(monomers.SEPARATED).make_rdm1()

    separable = sapt.exchange(mol_a, density_a, apart, density_apart)
    cumulant = sapt.cumulant_exchange(
        mol_a, monomers.active_orbitals(monomers.COMPRESSED), water_cumulant(), apart, density_apart
    )

    assert abs(separable) < EXACT
    assert abs(cumulant) < EXACT


def test_the_exchange_is_built_without_the_dimers_whole_tensor(monkeypatch):
    mol_a, density_a, mol_b, density_b = correlated_dimer()
    active, cumulant = monomers.active_orbitals(monomers.COMPRESSED), water_cumulant()
    monkeypatch.setattr(gto.Mole, "intor", reference.refusing("int2e", gto.Mole.intor))

    sapt.exchange(mol_a, density_a, mol_b, density_b)
    sapt.cumulant_exchange(mol_a, active, cumulant, mol_b, density_b)


@pytest.mark.parametrize("ncas, nelecas", [(10, 10), (6, 6), (4, 4), (8, 2)])
def test_a_determinant_has_no_cumulant(ncas, nelecas):
    built = sapt.cumulant(*monomers.restricted_rdm12(ncas, nelecas))

    assert built.shape == (ncas,) * 4
    assert np.allclose(built, 0.0, atol=EXACT)


def test_the_cumulant_traces_to_what_the_density_lacks_of_idempotency():
    """
    The cumulant traces to how much idempotency fails; the correlation.
    """
    rdm1, rdm2 = monomers.correlate_water(monomers.COMPRESSED)

    built = sapt.cumulant(rdm1, rdm2)

    assert np.abs(built).max() > FRACTIONAL
    assert np.allclose(np.einsum("pqrr->pq", built), (rdm1 @ rdm1 - 2 * rdm1) / 2, atol=EXACT)


def test_the_cumulant_carries_what_eq_14_adds_to_the_separable_part():
    correlated, restricted = reference_dimer()

    assert cumulant_exchange() == pytest.approx(
        reference.exchange(correlated, restricted)
        - reference.exchange(reference.separable(correlated), restricted),
        abs=EXACT,
    )


def test_the_cumulant_moves_the_exchange_further_than_the_comparison_tolerates():
    assert abs(cumulant_exchange()) > PUBLISHED


@pytest.mark.parametrize("ncas, nelecas", [(10, 10), (6, 6), (4, 4), (8, 2)])
def test_a_doubly_occupied_active_space_gives_back_the_restricted_exchange(ncas, nelecas):
    """
    With no cumulant the correlated path matches SAPT(RHF).
    """
    mean_field = monomers.solve_water(monomers.COMPRESSED)
    orbitals, electrons = mean_field.mo_coeff, monomers.WATER_ELECTRONS
    _, _, mol_b, density_b = correlated_dimer()
    rdm1, rdm2 = monomers.restricted_rdm12(ncas, nelecas)

    built = sapt.exchange(
        mean_field.mol, sapt.density(orbitals, rdm1, ncas, nelecas, electrons), mol_b, density_b
    ) + sapt.cumulant_exchange(
        mean_field.mol,
        sapt.active(orbitals, ncas, nelecas, electrons),
        sapt.cumulant(rdm1, rdm2),
        mol_b,
        density_b,
    )

    assert built == pytest.approx(
        sapt.exchange(mean_field.mol, mean_field.make_rdm1(), mol_b, density_b), abs=EXACT
    )


@pytest.mark.sapt_long
def test_the_encoded_cutouts_cumulant_traces_to_what_its_density_lacks_of_idempotency():
    cutout = monomers.read_encoded_cutout()

    built = monomers.cutout_cumulant()

    assert built.shape == (cutout.active_space_size,) * 4
    assert np.abs(built).max() > FRACTIONAL
    assert np.allclose(
        np.einsum("pqrr->pq", built), (cutout.rdm1 @ cutout.rdm1 - 2 * cutout.rdm1) / 2, atol=EXACT
    )


@pytest.mark.sapt_long
def test_each_pose_of_the_cutout_is_repelled_by_its_exchange():
    cutout = monomers.read_encoded_cutout()
    poses = monomers.read_poses()
    density_a = monomers.cutout_density()
    active, cumulant = monomers.cutout_active(), monomers.cutout_cumulant()

    energies = []
    for mol, mean_field in zip(poses.mols[:POSES], poses.mean_fields[:POSES]):
        density_b = mean_field.make_rdm1()
        energies.append(
            sapt.exchange(cutout.mol, density_a, mol, density_b)
            + sapt.cumulant_exchange(cutout.mol, active, cumulant, mol, density_b)
        )

    assert all(energy > 0 for energy in energies)
    for first, second in itertools.combinations(energies, 2):
        assert abs(first - second) > DISTINCT_POSES
