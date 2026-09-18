"""
The integrals spanning both monomers.

None is held as a four-index tensor. The Coulomb matrix is built through the direct path. Over the 
    water dimer's 26 they do fit, so the tests build them outright and compare.

The generalized integrals of Eq. (10) fold each monomer's electron-nucleus potential and their
    nuclear repulsion into the two-electron ones, so that one contraction returns a whole
    interaction energy.
"""

import numpy as np
import pytest
from pyscf import gto

import monomers
from utils import sapt

EXACT = 1e-10

# The overlap between two monomers in contact, against the same block once they are apart.
CONTACT = 1e-3


def generalized_integrals(mol_a, mol_b):
    """
    Eq. (10) written out as the four-index tensor it defines, over both AO bases.
    """
    nao = mol_a.nao
    overlap_a, overlap_b = mol_a.intor("int1e_ovlp"), mol_b.intor("int1e_ovlp")
    return (
        sapt.dimer(mol_a, mol_b).intor("int2e")[:nao, :nao, nao:, nao:]
        + np.einsum("kl,uv->uvkl", sapt.potential(mol_b, mol_a), overlap_a) / mol_a.nelectron
        + np.einsum("uv,kl->uvkl", sapt.potential(mol_a, mol_b), overlap_b) / mol_b.nelectron
        + sapt.repulsion(mol_a, mol_b)
        * np.einsum("uv,kl->uvkl", overlap_a, overlap_b)
        / (mol_a.nelectron * mol_b.nelectron)
    )


def refusing(name, intor):
    """
    `Mole.intor` with one integral taken away, so a test can say which one must not be built.
    """
    def guarded(self, intor_name, *args, **kwargs):
        if intor_name.startswith(name):
            raise AssertionError(f"{intor_name} was built over the dimer")
        return intor(self, intor_name, *args, **kwargs)
    return guarded


def test_the_dimer_basis_is_the_first_monomers_functions_then_the_seconds():
    mol_a, mol_b = monomers.water_dimer()

    dimer = sapt.dimer(mol_a, mol_b)

    assert dimer.nao == mol_a.nao + mol_b.nao == 2 * monomers.WATER_FUNCTIONS
    assert dimer.natm == mol_a.natm + mol_b.natm
    assert dimer.nelectron == mol_a.nelectron + mol_b.nelectron
    overlap = dimer.intor("int1e_ovlp")
    assert np.allclose(overlap[:mol_a.nao, :mol_a.nao], mol_a.intor("int1e_ovlp"), atol=EXACT)
    assert np.allclose(overlap[mol_a.nao:, mol_a.nao:], mol_b.intor("int1e_ovlp"), atol=EXACT)


def test_the_intermonomer_overlap_is_the_off_diagonal_block_of_the_dimers():
    mol_a, mol_b = monomers.water_dimer()

    built = sapt.overlap(mol_a, mol_b)

    assert built.shape == (mol_a.nao, mol_b.nao)
    assert np.allclose(
        built, sapt.dimer(mol_a, mol_b).intor("int1e_ovlp")[:mol_a.nao, mol_a.nao:], atol=EXACT
    )


def test_the_intermonomer_overlap_vanishes_once_the_monomers_are_apart():
    mol_a, mol_b = monomers.water_dimer()
    apart = monomers.build_water(monomers.SEPARATED)

    assert np.abs(sapt.overlap(mol_a, mol_b)).max() > CONTACT
    assert np.abs(sapt.overlap(mol_a, apart)).max() < EXACT


def test_a_monomers_own_potential_is_its_nuclear_attraction_matrix():
    mol_a, _ = monomers.water_dimer()

    assert np.allclose(sapt.potential(mol_a, mol_a), mol_a.intor("int1e_nuc"), atol=EXACT)


def test_the_two_monomers_potentials_add_up_to_the_dimers():
    mol_a, mol_b = monomers.water_dimer()
    nao = mol_a.nao

    built = sapt.potential(mol_a, mol_b)

    assert built.shape == (nao, nao)
    assert np.allclose(
        built + mol_a.intor("int1e_nuc"),
        sapt.dimer(mol_a, mol_b).intor("int1e_nuc")[:nao, :nao],
        atol=EXACT,
    )


def test_the_nuclear_repulsion_splits_into_each_monomers_and_the_cross_term():
    mol_a, mol_b = monomers.water_dimer()

    built = sapt.repulsion(mol_a, mol_b)

    assert built > 0
    assert mol_a.energy_nuc() + mol_b.energy_nuc() + built == pytest.approx(
        sapt.dimer(mol_a, mol_b).energy_nuc(), abs=EXACT
    )


def test_the_coulomb_matrix_is_the_cross_block_of_the_dimers_repulsion():
    mol_a, mol_b = monomers.water_dimer()
    density_b = monomers.solve_water(monomers.COMPRESSED).make_rdm1()
    nao = mol_a.nao
    cross = sapt.dimer(mol_a, mol_b).intor("int2e")[:nao, :nao, nao:, nao:]

    built = sapt.coulomb(mol_a, mol_b, density_b)

    assert built.shape == (nao, nao)
    assert np.allclose(built, np.einsum("uvkl,kl->uv", cross, density_b), atol=EXACT)


def test_the_generalized_coulomb_is_the_generalized_integrals_contracted_with_the_other_monomer():
    """
    Against Eq. (10) as written. The implementation never forms that tensor: its weights of 1/N_A
        and 1/N_B cancel against the traces of the densities, which is what leaves the one-electron
        terms standing on their own.
    """
    mol_a, mol_b = monomers.water_dimer()
    density_b = monomers.correlated_water_density(monomers.COMPRESSED)

    built = sapt.generalized_coulomb(mol_a, mol_b, density_b)

    assert np.allclose(
        built,
        np.einsum("uvkl,kl->uv", generalized_integrals(mol_a, mol_b), density_b),
        atol=EXACT,
    )


def test_the_coulomb_matrix_is_built_without_the_dimers_whole_tensor(monkeypatch):
    mol_a, mol_b = monomers.water_dimer()
    density_b = monomers.solve_water(monomers.COMPRESSED).make_rdm1()
    monkeypatch.setattr(gto.Mole, "intor", refusing("int2e", gto.Mole.intor))

    sapt.coulomb(mol_a, mol_b, density_b)
    sapt.generalized_coulomb(mol_a, mol_b, density_b)


@pytest.mark.sapt_long
def test_the_coulomb_matrix_over_a_real_dimer_comes_back_over_the_cutouts_functions():
    cutout = monomers.read_encoded_cutout()
    poses = monomers.read_poses()
    mol_b, mean_field = poses.mols[0], poses.mean_fields[0]

    built = sapt.coulomb(cutout.mol, mol_b, mean_field.make_rdm1())

    assert built.shape == (monomers.CUTOUT_FUNCTIONS,) * 2
    assert np.isfinite(built).all()
    assert np.allclose(built, built.T, atol=EXACT)
