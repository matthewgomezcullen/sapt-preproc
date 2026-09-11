import numpy as np
from prepare import PrepareComplex
from scipy.spatial import cKDTree # pyright: ignore[reportAttributeAccessIssue]

from pyscf import ao2mo, gto, mcscf, mp, scf

def molecule(prepared: PrepareComplex, verbose):
    """
    Build the PySCF molecule the SCF is solved over.
    """
    mol = gto.M(
        atom=[
            (atom.element.name, (atom.pos.x, atom.pos.y, atom.pos.z))
            for _, _, atom in prepared.atoms()
        ],
        charge=prepared.charge,
        spin=prepared.spin,
        basis=prepared.basis,
        verbose=verbose,
    )
    return mol

def rhf(mol, max_cycle, density_fit=False):
    """
    Run RHF with PySCF, in memory.

    `density_fit` fits the two-electron integrals rather than computing them, which is the lever
        for a cutout whose exact SCF will not finish inside a wall clock. It answers differently.
    """
    mean_field = _mean_field(mol, density_fit)
    mean_field.max_cycle = max_cycle
    mean_field.chkfile = None
    mean_field.kernel()
    return mean_field

def restore(mol, record=None, density_fit=False):
    """
    An RHF read back off disk. With the stored SCF, copy its record. Without it, return an RHF 
        object that has never run. `_solved.npz` contains the required values for `H()` anyway.
    """
    mean_field = _mean_field(mol, density_fit)
    if record is not None:
        # PySCF's own way back from a chkfile's `scf` group.
        mean_field.__dict__.update(record)
        mean_field.converged = True
    return mean_field

def _mean_field(mol, density_fit):
    mean_field = scf.RHF(mol)
    return mean_field.density_fit() if density_fit else mean_field

def generate_target_orbitals(prepared, cutoff, exclude, valence):
    """
    Generate AVAS target orbitals based on which atoms are chemically relevant.
    """
    atoms = prepared.atoms()
    poses = cKDTree(prepared._pose_coordinates())
    distances, _ = poses.query(
        [(atom.pos.x, atom.pos.y, atom.pos.z) for _, _, atom in atoms]
    )

    return [
        f"{index} {atom.element.name} {valence[atom.element.name]}"
        for index, ((_, residue, atom), distance) in enumerate(zip(atoms, distances))
        if residue.name not in exclude
        and atom.element.name in valence
        and distance <= cutoff
    ]

def mp2(mean_field, orbitals, ncas, nelecas, density_fit=False, verbose=0):
    """
    Correlate the active space with MP2 and return its one-particle density.

    AVAS returns semicanonical orbitals but not their energies, and the mean field still holds the
        canonical ones. MP2 divides by orbital energies, so they are recomputed from the Fock
        matrix here and the mean field is put back afterwards.
    """
    nao = mean_field.mol.nao
    core = (mean_field.mol.nelectron - nelecas) // 2

    # While the mean field still describes the converged density, before any orbital swap.
    fock = mean_field.get_fock()
    energies = np.diag(orbitals.T @ fock @ orbitals)

    frozen = list(range(core)) + list(range(core + ncas, nao))
    saved = mean_field.mo_coeff, mean_field.mo_energy
    try:
        mean_field.mo_coeff = orbitals
        mean_field.mo_energy = energies
        correlated = mp.MP2(mean_field, frozen=frozen or None)
        if density_fit:
            correlated = correlated.density_fit()
        correlated.verbose = verbose
        correlation = correlated.kernel()[0]
        density = correlated.make_rdm1()[core:core + ncas, core:core + ncas]
    finally:
        mean_field.mo_coeff, mean_field.mo_energy = saved
    return correlation, density

def cap(orbitals, density, ncas, nelecas, core, nmax):
    """
    Returns the space from an MP2 cap.
    """
    occupied = nelecas // 2
    filled, rotate_filled = np.linalg.eigh(density[:occupied, :occupied])
    empty, rotate_empty = np.linalg.eigh(density[occupied:, occupied:])
    filled, rotate_filled = filled[::-1], rotate_filled[:, ::-1]
    empty, rotate_empty = empty[::-1], rotate_empty[:, ::-1]
    natural_occ = orbitals[:, core:core + occupied] @ rotate_filled
    natural_vir = orbitals[:, core + occupied:core + ncas] @ rotate_empty

    ranked = sorted(
        [("occupied", index, min(n, 2 - n)) for index, n in enumerate(filled)]
        + [("virtual", index, min(n, 2 - n)) for index, n in enumerate(empty)],
        key=lambda entry: entry[2],
        reverse=True,
    )[:min(nmax, ncas)]
    kept_occ = sorted(index for block, index, _ in ranked if block == "occupied")
    kept_vir = sorted(index for block, index, _ in ranked if block == "virtual")
    lost_occ = [index for index in range(occupied) if index not in set(kept_occ)]
    lost_vir = [index for index in range(ncas - occupied) if index not in set(kept_vir)]

    return (
        len(ranked),
        2 * len(kept_occ),
        np.hstack([
            orbitals[:, :core],
            natural_occ[:, lost_occ],
            natural_occ[:, kept_occ],
            natural_vir[:, kept_vir],
            natural_vir[:, lost_vir],
            orbitals[:, core + ncas:],
        ]),
        np.concatenate([filled[kept_occ], empty[kept_vir]]),
    )

def dice(mol, executable, mpi, scratch, eps1, verbose=0):
    """
    Dice, configured as a solver CASCI can drive.

    The interface will not import until it has been told where Dice is, and `scratchDirectory` has
        to be an absolute path. The schedule starts coarse and tightens onto eps1.

    The perturbative correction is left off. It is not variational, so it would put the energy
        below full CI, and nothing downstream reads the energy.
    """
    from pyscf import __config__

    __config__.shci_SHCIEXE = executable
    __config__.shci_SHCISCRATCHDIR = scratch
    # Aliased because this module has an `shci` of its own. The ImportError is the caller's.
    from pyscf.shciscf import shci as interface

    solver = interface.SHCI(mol)
    solver.executable = executable
    solver.mpiprefix = mpi
    solver.scratchDirectory = scratch
    solver.runtimeDir = scratch
    solver.sweep_iter, solver.sweep_epsilon = [0, 3], [10 * eps1, eps1]
    solver.nPTiter = 0
    solver.verbose = verbose
    return solver

def shci(mean_field, orbitals, ncas, nelecas, solver, verbose=0):
    """
    Solve the active space with Dice, returning its energy and one-particle density.

    `cas_natorb` cannot rotate what Dice returns, which is a set of RDM files rather than a CI
        vector, so the density is handed back for the caller to diagonalise.
    """
    correlated = mcscf.CASCI(mean_field, ncas, nelecas)
    correlated.fcisolver = solver
    correlated.verbose = verbose
    correlated.kernel(orbitals)
    return correlated.e_tot, correlated.fcisolver.make_rdm1(correlated.ci, ncas, nelecas)

def window(orbitals, density, ncas, nelecas, core, lo, hi):
    """
    The natural orbitals of a CI density whose occupation is inside the window, and the space they
        leave.

    Descending, so the orbitals above the window are the first columns and those below it the last,
        and the window stays contiguous once the core has grown by the ones above it.
    """
    occupations, rotation = np.linalg.eigh(density)
    # Bounded: an occupation that comes back at -1e-17 is noise about zero, and a window of the
    # whole interval has to keep everything.
    occupations, rotation = np.clip(occupations[::-1], 0.0, 2.0), rotation[:, ::-1]
    natural = orbitals[:, core:core + ncas] @ rotation

    return select(
        np.hstack([orbitals[:, :core], natural, orbitals[:, core + ncas:]]),
        occupations,
        nelecas,
        lo,
        hi,
    )

def select(orbitals, occupations, nelecas, lo, hi):
    """
    Selects orbitals from a window and returns the resulting space.
    """
    kept = (occupations >= lo) & (occupations <= hi)
    return (
        int(kept.sum()),
        nelecas - 2 * int((occupations > hi).sum()),
        orbitals,
        occupations[kept],
    )

def integrals(mean_field, orbitals, ncas, nelecas):
    """
    The active-space integrals, as CASCI builds them.

    Returns the core energy, which holds the nuclear repulsion and the frozen electrons; the
        one-electron integrals with the core's Coulomb and exchange folded in; and the two-electron
        integrals over the active orbitals.
    """
    correlated = mcscf.CASCI(mean_field, ncas, nelecas)
    correlated.mo_coeff = orbitals
    h1, e_core = correlated.get_h1eff()
    return e_core, h1, ao2mo.restore(1, correlated.get_h2eff(), ncas)

def qubits(e_core, h1, h2, mapping="jordan_wigner"):
    """
    The active-space Hamiltonian as a qubit operator.

    RHF is closed shell, so the alpha integrals answers both spins and the operator carries two
        qubits per orbital.

    The core energy is not part of the second-quantised operator, so it is added as the identity.
        Without it the energies do not compare between poses.
    """
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy

    mapper = _mapper(mapping)
    operator = mapper.map(ElectronicEnergy.from_raw_integrals(h1, h2).second_q_op())
    identity = SparsePauliOp("I" * operator.num_qubits, e_core)
    return (operator + identity).simplify()

def _mapper(mapping):
    """
    One of the fermion-to-qubit mappings, by name.
    """
    from qiskit_nature.second_q.mappers import (
        BravyiKitaevMapper,
        JordanWignerMapper,
        ParityMapper,
    )

    mappers = {
        "jordan_wigner": JordanWignerMapper,
        "parity": ParityMapper,
        "bravyi_kitaev": BravyiKitaevMapper,
    }
    if mapping not in mappers:
        raise ValueError(
            f"{mapping} is not a mapping; it is one of {', '.join(sorted(mappers))}"
        )
    return mappers[mapping]()
