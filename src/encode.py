import os
import shutil
import tempfile
from subprocess import CalledProcessError

import numpy as np
from pyscf import gto, mcscf, mp, scf
from pyscf.mcscf import avas
from scipy.spatial import cKDTree # pyright: ignore[reportAttributeAccessIssue]

from prepare import PrepareComplex, PrepareError
from utils.reduce import CAPS


# The valence p shell README targets, per element it keeps. Hydrogen has no p shell and is not
# named; anything else in the cutout was rejected by `_verify` for having no 6-31G basis.
VALENCE = {"C": "2p", "N": "2p", "O": "2p", "S": "3p", "P": "3p"}


class EncodingError(RuntimeError):
    pass


class EncodeProtein:
    """
    EncodeProtein takes a PreparedComplex, solves RHF then encodes a tractable active space for
        SAPT(VQE) corrections.
    """

    def __init__(
        self,
        prepared: PrepareComplex,
    ):
        self.prepared = prepared
        self.mol = None
        self.mean_field = None
        self.energy = None

        self.max_cycle = 50 # RHF maximum number of cycles for convergence.
        self.verbose = 0 # PySCF prints its SCF table. Silent by default. Set = 4 for logging.
        self.ncas = None # active-space-size
        self.nelecas = None # active-electrons
        self.orbitals = None # orbital-initial-guess-for-CASCI/CASSCF
        self.occupations = None # natural occupations of the active window
        self.correlation = None # MP2 correlation energy of the AVAS space
        self.energy_cas = None # selected CI total energy of the space MP2 capped
        self.cutoff = 4.5 # Cutoff for chemically relevant atoms.
        self.threshold = 0.2 # AVAS threshold. PySCF's own default.
        self.nmax = 50 # Number of natural orbitals the MP2 caps

        # Dice
        self.dice = shutil.which("Dice") # `setup.sh` builds Dice into the environment's own bin.
        self.mpi = os.environ.get("MPIPREFIX", "") # empty runs on one rank. A cluster wants "srun" 
        # under SLURM, or "mpirun -np <ranks>"
        self.scratch = None # where to write integrals, wavefunction and RDMs.

    def RHF(self):
        """
        Solves RHF over the prepared protein for initial molecular orbitals. Computationally
            expensive and the driver behind the atom size cap.

        Restricted, so every electron is in a doubly occupied spatial orbital, which is what the
            even electron count `_verify_num_electrons` insisted on buys.

        The two-electron integrals are never held: a cutout carries many basis functions, whose 
            integrals run to petabytes, so PySCF builds them on the fly.

        An unconverged SCF is rejected.
        """
        self._molecule()

        mean_field = scf.RHF(self.mol)
        mean_field.max_cycle = self.max_cycle
        mean_field.kernel()
        if not mean_field.converged:
            raise EncodingError(
                f"RHF did not converge in {self.max_cycle} cycles"
            )

        self.mean_field = mean_field
        self.energy = mean_field.e_tot
        return mean_field

    def _molecule(self):
        """
        Build the PySCF molecule the SCF is solved over.

        The geometry is taken in `prepared.atoms()` order, which is the order AVAS addresses its
            targets by. PySCF assumes a molecule it is given no charge by default.
        """
        if self.prepared.charge is None:
            raise PrepareError("Cannot build the molecule before the charge is known")

        self.mol = gto.M(
            atom=[
                (atom.element.name, (atom.pos.x, atom.pos.y, atom.pos.z))
                for _, _, atom in self.prepared.atoms()
            ],
            charge=self.prepared.charge,
            spin=self.prepared.spin,
            basis=self.prepared.basis,
            verbose=self.verbose,
        )
        return self.mol

    def AVAS(self, targets=None):
        """
        Generates a tractable active space with target active orbitals.

        AVAS projects the converged occupied and virtual spaces onto the target atomic orbitals and
            keeps whatever carries weight above `threshold`.

        Returns the size of the active space, the electrons in it, and the full set of molecular
            orbitals rotated so that the active ones are contiguous.
        """
        if self.mean_field is None:
            raise EncodingError("Cannot choose an active space before RHF has been solved")

        if targets is None:
            targets = self._generate_target_orbitals()

        self.ncas, self.nelecas, self.orbitals = avas.avas(
            self.mean_field, targets, threshold=self.threshold
        )
        return self.ncas, self.nelecas, self.orbitals

    def _generate_target_orbitals(self) -> list[str]:
        """
        Generate AVAS target orbitals based on which atoms are "chemically relevant".

        See README.md for how "chemically relevant" is defined.

        Caps are left out.

        Each target is addressed by its zero-based PySCF atom index, which is the position of the
            atom in `prepared.atoms()` because that is the order `molecule` hands the geometry over
            in. PySCF anchors an index-prefixed label.
        """
        if self.mol is None:
            raise PrepareError("Cannot address target orbitals before the molecule is built")

        atoms = self.prepared.atoms()
        poses = cKDTree(self.prepared._pose_coordinates())
        distances, _ = poses.query(
            [(atom.pos.x, atom.pos.y, atom.pos.z) for _, _, atom in atoms]
        )

        return [
            f"{index} {atom.element.name} {VALENCE[atom.element.name]}"
            for index, ((_, residue, atom), distance) in enumerate(zip(atoms, distances))
            if residue.name not in CAPS
            and atom.element.name in VALENCE
            and distance <= self.cutoff
        ]

    def MP2(self):
        """
        Cap the active space at the nmax most correlated natural orbitals.

        MP2 correlates the whole AVAS space, its one-particle density is diagonalised into natural 
            orbitals, and the nmax most fractional by min(n, 2 - n) are kept.

        AVAS returns semicanonical orbitals but not their energies, and the mean field still holds
            the canonical ones, wrong for these orbitals. MP2 divides by orbital energies, and fed 
            the stale set it returns wrong answers. The energies are recomputed from the Fock 
            matrix, and the mean field is restored afterwards

        The occupied-virtual block of the unrelaxed MP2 density is zero, so diagonalising the two
            blocks separately loses nothing and keeps each orbital's provenance. That provenance is
            the electron count: a discarded occupied-derived orbital retires its pair to the core,
            a discarded virtual-derived one stays empty among the virtuals, and the window that
            remains returns the survivors in descending occupation.
        """
        if self.ncas is None:
            raise EncodingError("Cannot cap the active space before AVAS has chosen one")

        nao = self.mol.nao
        core = (self.mol.nelectron - self.nelecas) // 2
        occupied = self.nelecas // 2

        # While the mean field still describes the converged density, before any orbital swap.
        fock = self.mean_field.get_fock()
        energies = np.diag(self.orbitals.T @ fock @ self.orbitals)

        frozen = list(range(core)) + list(range(core + self.ncas, nao))
        saved = self.mean_field.mo_coeff, self.mean_field.mo_energy
        try:
            self.mean_field.mo_coeff = self.orbitals
            self.mean_field.mo_energy = energies
            correlated = mp.MP2(self.mean_field, frozen=frozen or None)
            correlated.verbose = self.verbose
            self.correlation = correlated.kernel()[0]
            density = correlated.make_rdm1()[core:core + self.ncas, core:core + self.ncas]
        finally:
            self.mean_field.mo_coeff, self.mean_field.mo_energy = saved

        filled, rotate_filled = np.linalg.eigh(density[:occupied, :occupied])
        empty, rotate_empty = np.linalg.eigh(density[occupied:, occupied:])
        filled, rotate_filled = filled[::-1], rotate_filled[:, ::-1]
        empty, rotate_empty = empty[::-1], rotate_empty[:, ::-1]
        natural_occ = self.orbitals[:, core:core + occupied] @ rotate_filled
        natural_vir = self.orbitals[:, core + occupied:core + self.ncas] @ rotate_empty

        ranked = sorted(
            [("occupied", index, min(n, 2 - n)) for index, n in enumerate(filled)]
            + [("virtual", index, min(n, 2 - n)) for index, n in enumerate(empty)],
            key=lambda entry: entry[2],
            reverse=True,
        )[:min(self.nmax, self.ncas)]
        kept_occ = sorted(index for block, index, _ in ranked if block == "occupied")
        kept_vir = sorted(index for block, index, _ in ranked if block == "virtual")
        lost_occ = [index for index in range(occupied) if index not in set(kept_occ)]
        lost_vir = [index for index in range(self.ncas - occupied) if index not in set(kept_vir)]

        self.orbitals = np.hstack([
            self.orbitals[:, :core],
            natural_occ[:, lost_occ],
            natural_occ[:, kept_occ],
            natural_vir[:, kept_vir],
            natural_vir[:, lost_vir],
            self.orbitals[:, core + self.ncas:],
        ])
        self.occupations = np.concatenate([filled[kept_occ], empty[kept_vir]])
        self.ncas = len(ranked)
        self.nelecas = 2 * len(kept_occ)
        return self.ncas, self.nelecas, self.orbitals

    def SHCI(self, eps1: float = 1e-4, lo: float = 0.01, hi: float = 1.99):
        """
        Truncate the active space to the correlated orbitals.

        Semistochastic Heat-bath Configuration Interaction (SHCI) solves the space MP2 capped. Its
            one-particle density is diagonalised into natural orbitals. An occupation near two or
            near zero is one a single determinant already describes, so only lo <= n_i <= hi is
            kept: an orbital above the window is doubly occupied and retires its pair to the core,
            one below it is empty and joins the virtuals.

        `eps1` is the selection threshold, below which a determinant is left out of the variational 
            space. Smaller is nearer exact and costs more.

        The window may deviate from the paper's 0.02 <= n_i <= 1.97, due to a lack of correlation.

        The window can leave a space with no excitation in it, whose correction to SAPT is exactly 
            zero. Rejected.

        Dice is an external program, so this leaves the process.
        """
        if self.ncas is None:
            raise EncodingError("Cannot solve the active space before AVAS has chosen one")
        if not self.dice:
            raise EncodingError(
                "Dice was not found. `setup.sh` builds it into the environment, or set `dice` to "
                "the executable"
            )

        scratch = os.path.abspath(self.scratch or tempfile.mkdtemp(prefix="dice-"))
        try:
            correlated = mcscf.CASCI(self.mean_field, self.ncas, self.nelecas)
            correlated.fcisolver = self._dice(eps1, scratch)
            correlated.verbose = self.verbose
            correlated.kernel(self.orbitals)
            self.energy_cas = correlated.e_tot
            # `cas_natorb` cannot rotate what Dice returns, which is a set of RDM files rather than
            # a CI vector, so the density is diagonalised here. Its eigenvalues are the natural
            # occupations and its eigenvectors rotate the window onto them.
            density = correlated.fcisolver.make_rdm1(correlated.ci, self.ncas, self.nelecas)
        except CalledProcessError as error:
            raise EncodingError(
                f"Dice failed over {self.ncas} orbitals; what it wrote is in {scratch}"
            ) from error
        else:
            if self.scratch is None:
                shutil.rmtree(scratch, ignore_errors=True)

        occupations, rotation = np.linalg.eigh(density)
        # Descending, and bounded: an occupation that comes back at -1e-17 is noise about zero, and
        # a window of the whole interval has to keep everything.
        occupations, rotation = np.clip(occupations[::-1], 0.0, 2.0), rotation[:, ::-1]

        core = (self.mol.nelectron - self.nelecas) // 2
        natural = self.orbitals[:, core:core + self.ncas] @ rotation

        kept = (occupations >= lo) & (occupations <= hi)
        ncas = int(kept.sum())
        nelecas = self.nelecas - 2 * int((occupations > hi).sum())
        if ncas == 0 or nelecas == 0 or nelecas == 2 * ncas:
            raise EncodingError(
                f"The window {lo} <= n <= {hi} leaves ({nelecas}e, {ncas}o) of "
                f"({self.nelecas}e, {self.ncas}o), which has no excitation in it to correct"
            )

        self.orbitals = np.hstack([
            self.orbitals[:, :core], natural, self.orbitals[:, core + self.ncas:]
        ])
        self.occupations = occupations[kept]
        self.ncas, self.nelecas = ncas, nelecas
        return self.ncas, self.nelecas, self.orbitals

    def _dice(self, eps1, scratch):
        """
        Dice, configured as a solver CASCI can drive.

        The interface will not import until it has been told where Dice is.

        `scratchDirectory` has to be an absolute path.

        The schedule starts coarse and tightens onto eps1. The run is given six more iterations
            after its last step to converge in.

        The perturbative correction is left off. It is not variational, so it would put the energy
            below full CI, and nothing downstream reads the energy.
        """
        from pyscf import __config__

        __config__.shci_SHCIEXE = self.dice
        __config__.shci_SHCISCRATCHDIR = scratch
        try:
            from pyscf.shciscf import shci
        except ImportError as error:
            raise EncodingError(
                "The Dice interface is not installed; `setup.sh` installs it beside Dice"
            ) from error

        solver = shci.SHCI(self.mol)
        solver.executable = self.dice
        solver.mpiprefix = self.mpi
        solver.scratchDirectory = scratch
        solver.runtimeDir = scratch
        solver.sweep_iter, solver.sweep_epsilon = [0, 3], [10 * eps1, eps1]
        solver.nPTiter = 0
        solver.verbose = self.verbose
        return solver

    def H_fermionic(self):
        """
        Construct the active space fermionic Hamiltonian.
        """
        ...

    def H_JW(self):
        """
        Maps fermionic electronic-structure Hamiltonian into a qubit Hamiltonian following the
            Jordan-Wigner transformation.
        """
        ...

