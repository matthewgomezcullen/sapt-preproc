"""
What the SAPT tests compare against, written out as the four-index tensors. They fit over the water 
    dimer's 26 functions.

Eq. (10) over every block of the dimer's functions, and Eq. (14) from each monomer's one- and
    two-particle density matrices over its occupied orbitals. Eq. (14) is written from the
    definition, E_exch(S^2) = -<V P> + E_elst <P> over the single exchanges P, and not from the
    paper's block expressions. Over the water monomers it reproduces Tables III and IV.
"""

import numpy as np
import scipy.linalg
from pyscf import gto


def attraction(dimer, atoms):
    """
    The attraction of `atoms`' nuclei over every function of `dimer`.
    """
    attracted = np.zeros((dimer.nao, dimer.nao))
    for atom in atoms:
        with dimer.with_rinv_origin(dimer.atom_coord(atom)):
            attracted -= dimer.atom_charge(atom) * dimer.intor("int1e_rinv")
    return attracted


def generalized_integrals(mol_a, mol_b):
    """
    Eq. (10) over the dimer's functions, `mol_a`'s first. The first pair is A's electron, which B's
        nuclei attract, and the second pair is B's electron.
    """
    dimer = gto.conc_mol(mol_a, mol_b)
    overlap = dimer.intor("int1e_ovlp")
    of_a = attraction(dimer, range(mol_a.natm))
    of_b = attraction(dimer, range(mol_a.natm, dimer.natm))
    repulsion = dimer.energy_nuc() - mol_a.energy_nuc() - mol_b.energy_nuc()
    electrons_a, electrons_b = mol_a.nelectron, mol_b.nelectron
    return (
        dimer.intor("int2e")
        + np.einsum("ij,kl->ijkl", of_b, overlap) / electrons_b
        + np.einsum("ij,kl->ijkl", overlap, of_a) / electrons_a
        + repulsion * np.einsum("ij,kl->ijkl", overlap, overlap) / (electrons_a * electrons_b)
    )


def exchange(monomer_a, monomer_b):
    """
    E^(1)_exch(S^2), Eq. (14), of two singlets.

    A monomer is its molecule, its occupied orbitals, and its spin-summed one- and two-particle
        density matrices over them in PySCF's order, rdm2[p, q, r, s] = <p+ r+ s q>. Summing each
        spin-orbital contraction over spins leaves a half on every term.
    """
    mol_a, orbitals_a, rdm1_a, rdm2_a = monomer_a
    mol_b, orbitals_b, rdm1_b, rdm2_b = monomer_b
    orbitals = scipy.linalg.block_diag(orbitals_a, orbitals_b)
    v = np.einsum(
        "pqrs,pa,qb,rc,sd->abcd", generalized_integrals(mol_a, mol_b), *[orbitals] * 4,
        optimize=True,
    )
    a = slice(0, orbitals_a.shape[1])
    b = slice(orbitals_a.shape[1], orbitals.shape[1])
    s = (orbitals.T @ gto.conc_mol(mol_a, mol_b).intor("int1e_ovlp") @ orbitals)[a, b]

    electrostatics = np.einsum("pP,qQ,pPqQ->", rdm1_a, rdm1_b, v[a, a, b, b])
    t1 = np.einsum("pP,qQ,PqQp->", rdm1_a, rdm1_b, v[a, b, b, a])
    t2 = np.einsum("pP,RqTU,pR,PqTU->", rdm1_a, rdm2_b, s, v[a, b, b, b])
    t3 = np.einsum("qQ,RpTU,Rq,TUpQ->", rdm1_b, rdm2_a, s, v[a, a, a, b])
    t4 = np.einsum(
        "RpTU,rqtu,Rq,pr,TUtu->", rdm2_a, rdm2_b, s, s, v[a, a, b, b], optimize=True
    )
    exchanged = np.einsum("pP,qQ,Pq,pQ->", rdm1_a, rdm1_b, s, s)
    return -0.5 * (t1 + t2 + t3 + t4 - electrostatics * exchanged)


def separable(monomer):
    """
    Minus the cumulant from the 2-RDM, keeping only the 1-RDM parts.
    """
    mol, orbitals, rdm1, _ = monomer
    return (
        mol,
        orbitals,
        rdm1,
        np.einsum("pq,rs->pqrs", rdm1, rdm1) - 0.5 * np.einsum("ps,rq->pqrs", rdm1, rdm1),
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
