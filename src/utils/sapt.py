"""
First order SAPT between a monomer correlated over an active space and one at RHF, each in its own
    monomer-centred basis.

Every integral spanning both is PySCF's, and none is held as a four-index tensor.

A two-particle density matrix is the part its one-particle one makes plus a cumulant. Exchange is
    linear in each, so it comes in two parts: `exchange` over the two AO densities, which is all of
    it for determinants, and `cumulant_exchange` for what the correlated monomer's cumulant adds.
"""

import numpy as np
from pyscf import gto
from pyscf.scf import jk

# The contractions `jk.get_jk` has kernels for when it uses a pair's symmetry. Any other runs
# without it.
SYMMETRIC = {"ji->kl", "jk->il", "li->kj", "lk->ij"}


def active(orbitals, ncas, nelecas, nelectron):
    """
    The `ncas` active orbitals, which start (nelectron - nelecas) / 2 columns into `orbitals`.
    """
    ncore = (nelectron - nelecas) // 2
    return orbitals[:, ncore:ncore + ncas]


def density(orbitals, rdm1, ncas, nelecas, nelectron):
    """
    A monomer's AO density: its core doubly occupied, and its active orbitals weighted by the
        spin-summed one-particle density matrix over them.
    """
    core = orbitals[:, :(nelectron - nelecas) // 2]
    correlated = active(orbitals, ncas, nelecas, nelectron)
    return 2 * core @ core.T + correlated @ rdm1 @ correlated.T


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


def attraction(bra, ket, nuclei):
    """
    The attraction of `nuclei`'s nuclei between `bra`'s functions and `ket`'s.
    """
    attracted = np.zeros((bra.nao, ket.nao))
    for charge, coords in zip(nuclei.atom_charges(), nuclei.atom_coords()):
        with bra.with_rinv_origin(coords):
            attracted -= charge * gto.intor_cross("int1e_rinv", bra, ket)
    return attracted


def potential(mol_a, mol_b):
    """
    The attraction of `mol_b`'s nuclei over `mol_a`'s functions.
    """
    return attraction(mol_a, mol_a, mol_b)


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


def generalized(mol_a, mol_b, blocks, density, script):
    """
    Eq. (10)'s integrals contracted with `density` as `jk.get_jk` contracts the plain ones, by the
        einsum `script` over ijkl. `blocks` names whose functions each of i, j, k and l run over,
        "a" or "b", and the pair ij is A's electron.

    Each one-electron term is a product of a matrix over ij and one over kl, so it contracts with a
        matrix as two matrix products and no block of Eq. (10) is held.
    """
    monomers = {"a": mol_a, "b": mol_b}
    i, j, k, l = (monomers[block] for block in blocks)
    # A list of matrices is not stacked, which would copy every one of them.
    single = isinstance(density, np.ndarray) and density.ndim == 2
    matrices = [density] if single else list(density)
    scripts = [script] * len(matrices) if isinstance(script, str) else list(script)

    two = jk.get_jk(
        (i, j, k, l), matrices, scripts, intor=mol_a._add_suffix("int2e"),
        aosym=_symmetry(blocks, scripts),
    )

    electrons_a, electrons_b = mol_a.nelectron, mol_b.nelectron
    overlap_ij, overlap_kl = overlap(i, j), overlap(k, l)
    # B's nuclei attract A's electron, and A's nuclei B's. The nuclei's repulsion relies on ij.
    on_a = (
        attraction(i, j, mol_b) / electrons_b
        + repulsion(mol_a, mol_b) * overlap_ij / (electrons_a * electrons_b)
    )
    on_b = attraction(k, l, mol_a) / electrons_a
    built = []
    for contracted, matrix, script in zip(two, matrices, scripts):
        pairs, rest = script.split(",")
        factorised = f"{pairs[:2]},{pairs[2:]},{rest}"
        built.append(
            contracted
            + np.einsum(factorised, on_a, overlap_kl, matrix, optimize=True)
            + np.einsum(factorised, overlap_ij, on_b, matrix, optimize=True)
        )
    return built[0] if single else built


def _symmetry(blocks, scripts):
    """
    The permutational symmetry `jk.get_jk` may use: a pair over one monomer's functions is
        symmetric, if every script is one it has a kernel for.
    """
    if any(script.split(",")[1] not in SYMMETRIC for script in scripts):
        return "s1"
    if blocks[0] == blocks[1] and blocks[2] == blocks[3]:
        return "s4"
    if blocks[0] == blocks[1]:
        return "s2ij"
    if blocks[2] == blocks[3]:
        return "s2kl"
    return "s1"


def generalized_coulomb(mol_a, mol_b, density_b):
    """
    Eq. (10)'s integrals contracted with `density_b`, over `mol_a`'s functions.
    """
    return generalized(mol_a, mol_b, "aabb", density_b, "ijkl,lk->ij")


def electrostatics(mol_a, density_a, mol_b, density_b):
    """
    The first order electrostatic energy, Eq. (8).
    """
    return np.einsum("uv,uv->", density_a, generalized_coulomb(mol_a, mol_b, density_b))


def exchange(mol_a, density_a, mol_b, density_b):
    """
    The first order exchange energy in the S^2 approximation, Eq. (14), with each monomer's
        two-particle density matrix the part its one-particle one makes. Over two determinants this
        is all of it; a correlated monomer's cumulant adds `cumulant_exchange`.

    T_5 cancels the part of T_4 that factorises into the electrostatic energy, and six
        contractions are left. None uses D S D = 2D, so a correlated density goes through them as a
        determinant's does. With W = D_A S D_B,

        -1/2 D_A.K[D_B] - 1/2 W.(J - K/2)[D_B] - 1/2 W.(J - K/2)[D_A]
            + 1/4 D_A.J[D_B S^T W] + 1/4 (W S^T D_A).J[D_B] - 1/8 W.K[W]

        where J and K are Eq. (10)'s, each over the block its partner spans.
    """
    s = overlap(mol_a, mol_b)
    w = density_a @ s @ density_b
    k_b = generalized(mol_a, mol_b, "abba", density_b, "ijkl,jk->il")
    # The exchange-type contractions are kj->il and ik->jl. D_B is symmetric, and so is the pair ij
    # over A's functions, so each is jk->il, which keeps the pair's symmetry.
    j_b, kb_mixed = generalized(
        mol_a, mol_b, "abbb", [density_b, density_b], ["ijkl,lk->ij", "ijkl,jk->il"]
    )
    j_a, ka_mixed = generalized(
        mol_a, mol_b, "aaab", [density_a, density_a], ["ijkl,ji->kl", "ijkl,jk->il"]
    )
    j_r, j_d, k_w = generalized(
        mol_a, mol_b, "aabb",
        [density_b @ s.T @ w, density_b, w],
        ["ijkl,lk->ij", "ijkl,lk->ij", "ijkl,jk->il"],
    )
    return (
        -np.vdot(density_a, k_b) / 2
        - np.vdot(w, j_b - kb_mixed / 2) / 2
        - np.vdot(w, j_a - ka_mixed / 2) / 2
        + np.vdot(density_a, j_r) / 4
        + np.vdot(w @ s.T @ density_a, j_d) / 4
        - np.vdot(w, k_w) / 8
    )


def cumulant(rdm1, rdm2):
    """
    What `rdm2` holds beyond the `rdm1`s, gamma_pq gamma_rs - gamma_ps gamma_rq / 2, in PySCF's 
        order, rdm2[p, q, r, s] = <p+ r+ s q>. A determinant has none.
    """
    return (
        rdm2
        - np.einsum("pq,rs->pqrs", rdm1, rdm1)
        + np.einsum("ps,rq->pqrs", rdm1, rdm1) / 2
    )


def cumulant_exchange(mol_a, active, cumulant, mol_b, density_b):
    """
    What `mol_a`'s cumulant over its `active` orbitals adds to `exchange`. Only T_3 and T_4 hold A's
        two-particle density matrix, so only they carry it:

        -1/2 sum_tuvw cumulant_tuvw [ v(vw|u x_t) + M_tu J[D_B]_vw - v(vw|x_t x_u) / 2 ]

        where x_t, the t-th column of D_B S^T C, is a function over B's functions, and
        M = C^T S D_B S^T C. v and J are Eq. (10)'s.

    Both integrals come from J builds of the active pair densities over the AB and BB blocks. The
        pair vw is symmetric, so one density serves both orders.
    """
    s = overlap(mol_a, mol_b)
    x = density_b @ s.T @ active
    first, second = np.triu_indices(active.shape[1])
    pairs = [
        (np.outer(active[:, v], active[:, w]) + np.outer(active[:, w], active[:, v])) / 2
        for v, w in zip(first, second)
    ]
    # The cumulant over each unordered pair, both orders summed.
    folded = cumulant[:, :, first, second] + cumulant[:, :, second, first]
    folded[:, :, first == second] /= 2

    with_a = np.array(generalized(mol_a, mol_b, "aaab", pairs, "ijkl,ji->kl"))
    with_b = np.array(generalized(mol_a, mol_b, "aabb", pairs, "ijkl,ji->kl"))
    coulomb_active = active.T @ generalized_coulomb(mol_a, mol_b, density_b) @ active
    return -(
        np.einsum("tup,pkl,ku,lt->", folded, with_a, active, x, optimize=True)
        + np.einsum("tuvw,tu,vw->", cumulant, active.T @ s @ x, coulomb_active)
        - np.einsum("tup,pkl,kt,lu->", folded, with_b, x, x, optimize=True) / 2
    ) / 2
