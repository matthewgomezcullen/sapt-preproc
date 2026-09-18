"""
The integrals spanning both monomers.

None is held as a four-index tensor. The Coulomb and exchange matrices are built through the direct
    path. Over the water dimer's 26 they do fit, so the tests build them outright and compare.

The generalized integrals of Eq. (10) fold each monomer's electron-nucleus potential and their
    nuclear repulsion into the two-electron ones, so that one contraction returns a whole
    interaction energy. Exchange contracts them over pairs of one function from each monomer, which
    needs the nuclei's attraction between the two monomers' functions too.
"""

import numpy as np
import pytest
from pyscf import gto

import monomers
import reference
from utils import sapt

EXACT = 1e-10

# The overlap between two monomers in contact, against the same block once they are apart.
CONTACT = 1e-3


def cross_block(mol_a, mol_b):
    """
    Eq. (10) with A's functions on A's electron and B's on B's, which is all electrostatics reads.
    """
    nao = mol_a.nao
    return reference.generalized_integrals(mol_a, mol_b)[:nao, :nao, nao:, nao:]


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


def test_the_attraction_between_the_monomers_functions_splits_into_each_monomers_nuclei():
    mol_a, mol_b = monomers.water_dimer()
    nao = mol_a.nao

    of_a = sapt.attraction(mol_a, mol_b, mol_a)
    of_b = sapt.attraction(mol_a, mol_b, mol_b)

    assert of_a.shape == of_b.shape == (mol_a.nao, mol_b.nao)
    assert np.allclose(
        of_a + of_b, sapt.dimer(mol_a, mol_b).intor("int1e_nuc")[:nao, nao:], atol=EXACT
    )


def test_a_monomers_potential_is_the_attraction_between_its_own_functions():
    mol_a, mol_b = monomers.water_dimer()

    assert np.allclose(
        sapt.potential(mol_a, mol_b), sapt.attraction(mol_a, mol_a, mol_b), atol=EXACT
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
    Against Eq. (10) as written. The implementation never forms that tensor.
    """
    mol_a, mol_b = monomers.water_dimer()
    density_b = monomers.correlated_water_density(monomers.COMPRESSED)

    built = sapt.generalized_coulomb(mol_a, mol_b, density_b)

    assert np.allclose(
        built, np.einsum("uvkl,kl->uv", cross_block(mol_a, mol_b), density_b), atol=EXACT
    )


def test_the_generalized_coulomb_keeps_its_weights_for_a_matrix_of_any_trace():
    """
    Exchange contracts Eq. (10) with matrices such as D_B S D_A S D_B, whose trace is not electron
        count, so the 1/N_B weight cannot be cancelled against the trace of what it is given.
    """
    mol_a, mol_b = monomers.water_dimer()
    halved = monomers.correlated_water_density(monomers.COMPRESSED) / 2

    built = sapt.generalized_coulomb(mol_a, mol_b, halved)

    assert np.allclose(
        built, np.einsum("uvkl,kl->uv", cross_block(mol_a, mol_b), halved), atol=EXACT
    )


@pytest.mark.parametrize(
    "blocks, script",
    [
        ("abba", "ijkl,jk->il"),  # K over A's functions, of a matrix over B's
        ("abbb", "ijkl,lk->ij"),  # J between A's functions and B's, of a matrix on B's electron
        ("abbb", "ijkl,kj->il"),  # K between them
        ("aaab", "ijkl,ji->kl"),  # J between them, of a matrix on A's electron
        ("aaab", "ijkl,ik->jl"),  # K between them
        ("aabb", "ijkl,lk->ij"),  # J over A's functions, as electrostatics has it
        ("aabb", "ijkl,jk->il"),  # K of a matrix spanning both
    ],
)
def test_the_generalized_integrals_contract_over_any_block_of_the_dimers_functions(blocks, script):
    """
    Eq. (10) contracted as `jk.get_jk` contracts the plain integrals, over whichever monomer's
        functions each index runs. Any matrix will do.
    """
    mol_a, mol_b = monomers.water_dimer()
    integrals = reference.generalized_integrals(mol_a, mol_b)
    functions = {"a": slice(0, mol_a.nao), "b": slice(mol_a.nao, integrals.shape[0])}
    block = integrals[tuple(functions[monomer] for monomer in blocks)]
    contracted = script.split(",")[1].split("->")[0]
    matrix = np.random.default_rng(0).standard_normal(
        [block.shape["ijkl".index(index)] for index in contracted]
    )

    built = sapt.generalized(mol_a, mol_b, blocks, matrix, script)

    assert np.allclose(built, np.einsum(script, block, matrix), atol=EXACT)


def test_the_coulomb_matrix_is_built_without_the_dimers_whole_tensor(monkeypatch):
    mol_a, mol_b = monomers.water_dimer()
    density_b = monomers.solve_water(monomers.COMPRESSED).make_rdm1()
    monkeypatch.setattr(gto.Mole, "intor", reference.refusing("int2e", gto.Mole.intor))

    sapt.coulomb(mol_a, mol_b, density_b)
    sapt.generalized_coulomb(mol_a, mol_b, density_b)
    sapt.generalized(mol_a, mol_b, "abba", density_b, "ijkl,jk->il")


@pytest.mark.sapt_long
def test_the_coulomb_matrix_over_a_real_dimer_comes_back_over_the_cutouts_functions():
    cutout = monomers.read_encoded_cutout()
    poses = monomers.read_poses()
    mol_b, mean_field = poses.mols[0], poses.mean_fields[0]

    built = sapt.coulomb(cutout.mol, mol_b, mean_field.make_rdm1())

    assert built.shape == (monomers.CUTOUT_FUNCTIONS,) * 2
    assert np.isfinite(built).all()
    assert np.allclose(built, built.T, atol=EXACT)
