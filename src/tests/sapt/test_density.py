"""
The monomer densities in the AO basis.

The protein's is correlated. A pose's is the restricted one PySCF builds.

The active block is (N - N_act) / 2 columns in. The electron count says where it starts.
"""

import numpy as np
import pytest
import scipy.linalg

import monomers
from utils import sapt

EXACT = 1e-10

# An occupation this far from 0 and 2 is fractional.
FRACTIONAL = 1e-3

# The trace accumulates error over a large system.
NUM_ELEC_ACCURACY = 1e-8


def occupations(mol, density):
    overlap = mol.intor("int1e_ovlp")
    return scipy.linalg.eigh(density, np.linalg.inv(overlap), eigvals_only=True)


@pytest.mark.parametrize("ncas, nelecas", [(10, 10), (6, 6), (4, 4), (8, 2)])
def test_a_doubly_occupied_active_space_gives_back_the_restricted_density(ncas, nelecas):
    mean_field = monomers.solve_water(monomers.EQUILIBRIUM)

    built = sapt.density(
        mean_field.mo_coeff,
        monomers.restricted_rdm1(ncas, nelecas),
        ncas,
        nelecas,
        monomers.WATER_ELECTRONS,
    )

    assert np.allclose(built, mean_field.make_rdm1(), atol=EXACT)


@pytest.mark.parametrize("ncas, nelecas", [(10, 10), (6, 6), (4, 4), (8, 2)])
def test_the_active_orbitals_start_where_the_core_ends(ncas, nelecas):
    orbitals = monomers.solve_water(monomers.EQUILIBRIUM).mo_coeff
    core = (monomers.WATER_ELECTRONS - nelecas) // 2

    built = sapt.active(orbitals, ncas, nelecas, monomers.WATER_ELECTRONS)

    assert np.array_equal(built, orbitals[:, core:core + ncas])


def test_the_density_holds_every_electron_of_the_monomer():
    mol = monomers.build_water(monomers.COMPRESSED)

    built = monomers.correlated_water_density(monomers.COMPRESSED)

    assert built.shape == (monomers.WATER_FUNCTIONS,) * 2
    assert np.trace(built @ mol.intor("int1e_ovlp")) == pytest.approx(
        monomers.WATER_ELECTRONS, abs=EXACT
    )


def test_the_density_is_symmetric():
    built = monomers.correlated_water_density(monomers.COMPRESSED)

    assert np.allclose(built, built.T, atol=EXACT)


def test_the_active_orbitals_are_the_only_ones_the_density_matrix_moves():
    mol = monomers.build_water(monomers.COMPRESSED)
    orbitals = monomers.solve_water(monomers.COMPRESSED).mo_coeff
    overlap = mol.intor("int1e_ovlp")
    ncas, nelecas = monomers.WATER_NCAS, monomers.WATER_NELECAS
    core = (monomers.WATER_ELECTRONS - nelecas) // 2

    difference = monomers.correlated_water_density(monomers.COMPRESSED) - sapt.density(
        orbitals, monomers.restricted_rdm1(ncas, nelecas), ncas, nelecas,
        monomers.WATER_ELECTRONS,
    )
    projected = orbitals.T @ overlap @ difference @ overlap @ orbitals
    assert np.abs(projected[core:core + ncas, core:core + ncas]).max() > FRACTIONAL
    projected[core:core + ncas, core:core + ncas] = 0.0

    assert np.allclose(projected, 0.0, atol=EXACT)


def test_the_correlated_density_is_not_idempotent_where_the_restricted_one_is():
    """
    D S D = 2D for a single determinant and not for a correlated one. The exchange terms are written
        for a general active density because of it, so this is why the paper's block expressions 
        cannot be simplified.
    """
    mol = monomers.build_water(monomers.COMPRESSED)
    overlap = mol.intor("int1e_ovlp")
    restricted = monomers.solve_water(monomers.COMPRESSED).make_rdm1()
    correlated = monomers.correlated_water_density(monomers.COMPRESSED)

    assert np.allclose(restricted @ overlap @ restricted, 2 * restricted, atol=EXACT)
    assert not np.allclose(correlated @ overlap @ correlated, 2 * correlated, atol=FRACTIONAL)


def test_the_natural_occupations_lie_between_zero_and_two_and_some_clear_of_both():
    mol = monomers.build_water(monomers.COMPRESSED)

    natural = occupations(mol, monomers.correlated_water_density(monomers.COMPRESSED))

    assert np.all(natural >= -EXACT)
    assert np.all(natural <= 2 + EXACT)
    assert np.any((natural > FRACTIONAL) & (natural < 2 - FRACTIONAL))


@pytest.mark.sapt_long
def test_the_encoded_cutouts_density_holds_every_electron_of_the_cutout():
    cutout = monomers.read_encoded_cutout()

    built = monomers.cutout_density()

    assert built.shape == (monomers.CUTOUT_FUNCTIONS,) * 2
    assert np.allclose(built, built.T, atol=EXACT)
    assert np.trace(built @ cutout.mol.intor("int1e_ovlp")) == pytest.approx(
        monomers.CUTOUT_ELECTRONS, abs=NUM_ELEC_ACCURACY
    )


@pytest.mark.sapt_long
def test_the_encoded_cutouts_density_is_correlated_rather_than_restricted():
    cutout = monomers.read_encoded_cutout()

    natural = occupations(cutout.mol, monomers.cutout_density())

    assert np.all(natural >= -NUM_ELEC_ACCURACY)
    assert np.all(natural <= 2 + NUM_ELEC_ACCURACY)
    assert ((natural > FRACTIONAL) & (natural < 2 - FRACTIONAL)).sum() == cutout.active_space_size


@pytest.mark.sapt_long
def test_every_poses_density_holds_its_own_electrons():
    poses = monomers.read_poses()

    assert poses.solved()
    assert len(poses.mean_fields) == monomers.POSES
    for mol, mean_field in zip(poses.mols, poses.mean_fields):
        built = mean_field.make_rdm1()
        overlap = mol.intor("int1e_ovlp")
        assert built.shape == (monomers.POSE_FUNCTIONS,) * 2
        assert np.trace(built @ overlap) == pytest.approx(mol.nelectron, abs=EXACT)
        assert np.allclose(built @ overlap @ built, 2 * built, atol=EXACT)
