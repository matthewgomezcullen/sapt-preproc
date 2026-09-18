"""
First order SAPT between a monomer correlated over an active space and one at RHF, each in its own
    monomer-centred basis.

Every integral spanning both is PySCF's, and none is held as a four-index tensor.
"""

import numpy as np
from pyscf import gto
from pyscf.scf import jk


def density(orbitals, rdm1, ncas, nelecas, nelectron):
    """
    A monomer's AO density: its core doubly occupied, and its active orbitals weighted by the
        spin-summed one-particle density matrix over them.

    The active block starts (nelectron - nelecas) / 2 columns into `orbitals`.
    """
    ncore = (nelectron - nelecas) // 2
    core = orbitals[:, :ncore]
    active = orbitals[:, ncore:ncore + ncas]
    return 2 * core @ core.T + active @ rdm1 @ active.T


def dimer(mol_a, mol_b):
    """
    Both monomers as one molecule, `mol_a`'s functions first.
    """
    return gto.conc_mol(mol_a, mol_b)


def overlap(mol_a, mol_b):
    """
    The overlap between `mol_a`'s functions and `mol_b`'s.
    """
    return gto.intor_cross("int1e_ovlp", mol_a, mol_b)


def potential(mol_a, mol_b):
    """
    The attraction of `mol_b`'s nuclei over `mol_a`'s functions.
    """
    attraction = np.zeros((mol_a.nao, mol_a.nao))
    for charge, coords in zip(mol_b.atom_charges(), mol_b.atom_coords()):
        with mol_a.with_rinv_origin(coords):
            attraction -= charge * mol_a.intor("int1e_rinv")
    return attraction


def repulsion(mol_a, mol_b):
    """
    The repulsion between `mol_a`'s nuclei and `mol_b`'s.
    """
    distances = np.linalg.norm(
        mol_a.atom_coords()[:, None, :] - mol_b.atom_coords()[None, :, :], axis=-1
    )
    return np.einsum("a,b,ab->", mol_a.atom_charges(), mol_b.atom_charges(), 1 / distances)


def coulomb(mol_a, mol_b, density_b):
    """
    The Coulomb matrix `density_b` exerts over `mol_a`'s functions, through the direct path.
    """
    return jk.get_jk(
        (mol_a, mol_a, mol_b, mol_b), density_b, "ijkl,lk->ij", intor=mol_a._add_suffix("int2e"),
        aosym="s4",
    )


def generalized_coulomb(mol_a, mol_b, density_b):
    """
    Eq. (10)'s integrals contracted with `density_b`. Its 1/N_B weight cancels against the trace of
        `density_b`, and its 1/N_A weight is left on the overlap of `mol_a`.
    """
    return (
        coulomb(mol_a, mol_b, density_b)
        + potential(mol_a, mol_b)
        + mol_a.intor("int1e_ovlp")
        * (np.einsum("kl,kl->", density_b, potential(mol_b, mol_a)) + repulsion(mol_a, mol_b))
        / mol_a.nelectron
    )


def electrostatics(mol_a, density_a, mol_b, density_b):
    """
    The first order electrostatic energy, Eq. (8).
    """
    return np.einsum("uv,uv->", density_a, generalized_coulomb(mol_a, mol_b, density_b))
