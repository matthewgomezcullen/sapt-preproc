"""
The first order electrostatic energy, Eq. (8): one monomer's AO density contracted against the
    generalized Coulomb matrix of the other's.

Tables III and IV score the paper's own water dimer at SAPT(RHF) and at SAPT(CASCI) over a (6e, 6o)
    active space on the compressed monomer, in the same monomer-centred 6-31g basis. Those two
    numbers are the oracle, and the second of them is the arrangement the pipeline is in: one
    monomer correlated over an active space, the other at RHF.
"""

import itertools

import numpy as np
import pytest

import monomers
from utils import sapt

# The tables print six decimal places, so half of the last one is the closest a comparison gets.
PUBLISHED = 1e-6  # Hartree

EXACT = 1e-12

# Two neutral monomers 100 A apart interact through their dipoles, which is 1e-7 Hartree here.
NEGLIGIBLE = 1e-6

POSES = 3

DISTINCT_POSES = 1e-5


def restricted_dimer():
    mol_a, mol_b = monomers.water_dimer()
    return (
        mol_a, monomers.solve_water(monomers.EQUILIBRIUM).make_rdm1(),
        mol_b, monomers.solve_water(monomers.COMPRESSED).make_rdm1(),
    )


def correlated_dimer():
    mol_a, mol_b = monomers.water_dimer()
    return (
        mol_a, monomers.solve_water(monomers.EQUILIBRIUM).make_rdm1(),
        mol_b, monomers.correlated_water_density(monomers.COMPRESSED),
    )


def test_the_energy_over_restricted_monomers_is_the_one_the_paper_publishes():
    energy = sapt.electrostatics(*restricted_dimer())

    assert energy == pytest.approx(monomers.ELST_RESTRICTED, abs=PUBLISHED)


def test_the_energy_over_a_correlated_monomer_is_the_one_the_paper_publishes():
    energy = sapt.electrostatics(*correlated_dimer())

    assert energy == pytest.approx(monomers.ELST_CORRELATED, abs=PUBLISHED)


def test_correlating_a_monomer_moves_the_energy_further_than_the_comparison_tolerates():
    restricted = sapt.electrostatics(*restricted_dimer())
    correlated = sapt.electrostatics(*correlated_dimer())

    assert abs(correlated - restricted) > PUBLISHED


def test_the_energy_does_not_depend_on_which_monomer_is_named_first():
    mol_a, density_a, mol_b, density_b = correlated_dimer()

    assert sapt.electrostatics(mol_a, density_a, mol_b, density_b) == pytest.approx(
        sapt.electrostatics(mol_b, density_b, mol_a, density_a), abs=EXACT
    )


def test_the_energy_is_the_classical_interaction_of_the_two_charge_distributions():
    """
    Four terms: the electron clouds against each other, each monomer's electrons against the other's
        nuclei, and the nuclei against the nuclei.
    """
    mol_a, density_a, mol_b, density_b = correlated_dimer()

    classical = (
        np.einsum("uv,uv->", density_a, sapt.coulomb(mol_a, mol_b, density_b))
        + np.einsum("uv,uv->", density_a, sapt.potential(mol_a, mol_b))
        + np.einsum("kl,kl->", density_b, sapt.potential(mol_b, mol_a))
        + sapt.repulsion(mol_a, mol_b)
    )

    assert sapt.electrostatics(mol_a, density_a, mol_b, density_b) == pytest.approx(
        classical, abs=EXACT
    )


def test_the_energy_falls_away_once_the_monomers_are_apart():
    mol_a, density_a, _, _ = restricted_dimer()
    apart = monomers.build_water(monomers.SEPARATED)

    energy = sapt.electrostatics(
        mol_a, density_a, apart, monomers.solve_water(monomers.SEPARATED).make_rdm1()
    )

    assert abs(energy) < NEGLIGIBLE < abs(monomers.ELST_RESTRICTED)


@pytest.mark.sapt_long
def test_each_pose_of_the_cutout_has_its_own_electrostatic_energy():
    """
    What the experiment ranks by. 
    """
    cutout = monomers.read_encoded_cutout()
    poses = monomers.read_poses()
    density_a = monomers.cutout_density()

    energies = [
        sapt.electrostatics(cutout.mol, density_a, mol, mean_field.make_rdm1())
        for mol, mean_field in zip(poses.mols[:POSES], poses.mean_fields[:POSES])
    ]

    assert all(energy < 0 for energy in energies)
    for first, second in itertools.combinations(energies, 2):
        assert abs(first - second) > DISTINCT_POSES
