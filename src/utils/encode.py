import hashlib
import json
import os

import numpy as np
from prepare import PrepareComplex
from scipy.spatial import cKDTree # pyright: ignore[reportAttributeAccessIssue]

from pyscf import gto, mcscf, mp, scf

# What `$DATA` is joined with by default
CHECKPOINTS = "scf"

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

def store():
    """
    The directory SCF checkpoints go in, or None if there is nowhere to put them.
    """
    override = os.environ.get("SCF_CHECKPOINTS")
    if override:
        return override
    data = os.environ.get("DATA")
    return os.path.join(data, CHECKPOINTS) if data else None

def _method(mean_field):
    """
    What produced the orbitals, as far as the digest is concerned.

    Taken from the object rather than a flag, so a mean field this module has never heard of still
        keys apart from the ones it has.
    """
    name = type(mean_field).__name__
    auxbasis = getattr(getattr(mean_field, "with_df", None), "auxbasis", None)
    return f"{name}/{auxbasis}" if auxbasis else name

def checkpoint(mol, store, mean_field=None):
    """
    The path this molecule's SCF takes in `store`.

    `mean_field` of None is the exact RHF the pipeline has always run. Fitting the two-electron
        integrals moves the energy by a few times 1e-6 Ha, which is small, systematic, and exactly
        the sort of difference nothing would notice if the two shared a file.
    """
    if not mol._built:
        mol.build()
    payload = json.dumps(
        {
            "atom": mol._atom,
            "basis": mol._basis,
            "ecp": mol._ecp,
            "charge": mol.charge,
            "spin": mol.spin,
            "cart": mol.cart,
            # `test_the_default_path_is_the_exact_solve` holds this to `_method(scf.RHF(mol))`.
            "method": _method(mean_field) if mean_field is not None else "RHF",
        },
        sort_keys=True,
    )
    return os.path.join(store, f"{hashlib.sha256(payload.encode()).hexdigest()}.chk")

def rhf(mol, max_cycle, store=None, density_fit=False):
    """
    Run RHF with PySCF, keeping a checkpoint if there is somewhere to keep one.

    PySCF writes the checkpoint every cycle.

    `density_fit` fits the two-electron integrals rather than computing them, which is the lever
        for a cutout whose exact SCF will not finish inside a wall clock. It answers differently,
        and the checkpoint knows.
    """
    mean_field = scf.RHF(mol)
    if density_fit:
        mean_field = mean_field.density_fit()
    mean_field.max_cycle = max_cycle
    if store is None:
        mean_field.kernel()
        return mean_field

    os.makedirs(store, exist_ok=True)
    mean_field.chkfile = checkpoint(mol, store, mean_field)

    record = _checkpointed(mean_field.chkfile, mol)
    if record is not None:
        if record.get("converged"):
            return _restore(mean_field, record)
        mean_field.init_guess = "chkfile"

    mean_field.kernel()
    # Convergence flag
    scf.chkfile.save(mean_field.chkfile, "scf/converged", bool(mean_field.converged))
    return mean_field

def _checkpointed(path, mol):
    """
    What is on disk for this molecule.
    """
    if not os.path.exists(path):
        return None
    try:
        record = scf.chkfile.load(path, "scf")
    except Exception: # h5py raises OSError for a file cut off mid-write, among others.
        record = None
    if record is not None and "mo_coeff" in record:
        if np.shape(record["mo_coeff"])[0] == mol.nao:
            return record
    os.remove(path)
    return None

def _restore(mean_field, record):
    """
    A converged solve read back off disk.
    """
    mean_field.mo_coeff = np.asarray(record["mo_coeff"])
    mean_field.mo_energy = np.asarray(record["mo_energy"])
    mean_field.mo_occ = np.asarray(record["mo_occ"])
    # A solve puts a numpy scalar here, and a read should be indistinguishable.
    mean_field.e_tot = record["e_tot"]
    mean_field.converged = True
    return mean_field

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
            # Freezing shrinks the MO indices but not the transformation underneath, which still
            # contracts over every AO and spools the half-transformed integrals to disk. Fitting
            # them replaces that with an auxiliary-basis contraction: on a cutout of the bin it is
            # the difference between a hundred gigabytes of scratch and under one.
            correlated = correlated.density_fit()
        correlated.verbose = verbose
        correlation = correlated.kernel()[0]
        density = correlated.make_rdm1()[core:core + ncas, core:core + ncas]
    finally:
        mean_field.mo_coeff, mean_field.mo_energy = saved
    return correlation, density

def cap(orbitals, density, ncas, nelecas, core, nmax):
    """
    The nmax most fractional natural orbitals of an MP2 density, and the space they leave.
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

    kept = (occupations >= lo) & (occupations <= hi)
    return (
        int(kept.sum()),
        nelecas - 2 * int((occupations > hi).sum()),
        np.hstack([orbitals[:, :core], natural, orbitals[:, core + ncas:]]),
        occupations[kept],
    )
