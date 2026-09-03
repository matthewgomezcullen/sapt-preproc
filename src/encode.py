import os
import shutil
import tempfile
from subprocess import CalledProcessError

from pyscf.mcscf import avas

from prepare import PrepareComplex, PrepareError
from utils.reduce import CAPS
from utils import encode


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

        # RHF
        self.rhf_max_cycle = 50 # RHF maximum number of cycles for convergence.
        self.verbose = 0 # PySCF prints its SCF table. Silent by default. Set = 4 for logging.

        # AVAS
        self.active_space_size = None # active-space-size
        self.active_electrons = None # active-electrons
        self.orbital_initial = None # orbital-initial-guess-for-CASCI/CASSCF
        self.occupations = None # natural occupations of the active window
        self.correlation = None # MP2 correlation energy of the AVAS space
        self.shci_energy = None # SHCI total energy of the space MP2 capped
        self.cutoff = 4.5 # Cutoff for chemically relevant atoms.
        self.avas_threshold = 0.2 # AVAS threshold. PySCF's own default.
        
        # MP2
        self.density_fit = True # Density fit MP2
        self.nmax = 50 # Number of natural orbitals the MP2 caps

        # Where a converged SCF is kept so that the next run reads it instead of solving again.
        # The environment decides; None is no checkpointing, which is the default off the cluster.
        self.checkpoints = encode.store()

        # Dice
        self.dice = shutil.which("Dice") # `setup.sh` builds Dice into the environment's own bin.
        self.mpi = os.environ.get("MPIPREFIX", "") # empty runs on one rank. A cluster wants "srun" 
        # under SLURM, or "mpirun -np <ranks>"
        self.scratch = None # where to write integrals, wavefunction and RDMs.

        # Hamiltonian
        self.e_core = None # nuclear repulsion and the energy of the frozen electrons
        self.hamiltonian = None # the active space as a qubit operator

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
        mean_field = encode.rhf(self.mol, self.rhf_max_cycle, self.checkpoints)
        if not mean_field.converged:
            raise EncodingError(
                f"RHF did not converge in {self.rhf_max_cycle} cycles"
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
        self.mol = encode.molecule(self.prepared, self.verbose)

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
        self.active_space_size, self.active_electrons, self.orbital_initial = avas.avas(
            self.mean_field, targets, threshold=self.avas_threshold
        )
        return self.active_space_size, self.active_electrons, self.orbital_initial

    def _generate_target_orbitals(self) -> list[str]:
        """
        Generate AVAS target orbitals based on which atoms are "chemically relevant".

        See README.md for how "chemically relevant" is defined.

        Caps are left out.

        Each target is addressed by its zero-based PySCF atom index
        """
        if self.mol is None:
            raise PrepareError("Cannot address target orbitals before the molecule is built")
        return encode.generate_target_orbitals(
            self.prepared,
            self.cutoff,
            CAPS,
            VALENCE
        )

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
        if self.active_space_size is None:
            raise EncodingError("Cannot cap the active space before AVAS has chosen one")

        core = (self.mol.nelectron - self.active_electrons) // 2
        self.correlation, density = encode.mp2(
            self.mean_field,
            self.orbital_initial,
            self.active_space_size,
            self.active_electrons,
            self.density_fit,
            verbose=self.verbose,
        )
        self.active_space_size, self.active_electrons, self.orbital_initial, self.occupations = encode.cap(
            self.orbital_initial, density, self.active_space_size, self.active_electrons, core, self.nmax
        )
        return self.active_space_size, self.active_electrons, self.orbital_initial

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
        if self.active_space_size is None:
            raise EncodingError("Cannot solve the active space before AVAS has chosen one")
        if not self.dice:
            raise EncodingError(
                "Dice was not found. `setup.sh` builds it into the environment, or set `dice` to "
                "the executable"
            )

        scratch = os.path.abspath(self.scratch or tempfile.mkdtemp(prefix="dice-"))
        core = (self.mol.nelectron - self.active_electrons) // 2
        try:
            solver = encode.dice(self.mol, self.dice, self.mpi, scratch, eps1, self.verbose)
            self.shci_energy, density = encode.shci(
                self.mean_field, self.orbital_initial, self.active_space_size, self.active_electrons, solver, self.verbose
            )
        except ImportError as error:
            raise EncodingError(
                "The Dice interface is not installed; `setup.sh` installs it beside Dice"
            ) from error
        except CalledProcessError as error:
            raise EncodingError(
                f"Dice failed over {self.active_space_size} orbitals; what it wrote is in {scratch}"
            ) from error
        else:
            if self.scratch is None:
                shutil.rmtree(scratch, ignore_errors=True)

        ncas, nelecas, orbitals, occupations = encode.window(
            self.orbital_initial, density, self.active_space_size, self.active_electrons, core, lo, hi
        )
        if ncas == 0 or nelecas == 0 or nelecas == 2 * ncas:
            raise EncodingError(
                f"The window {lo} <= n <= {hi} leaves ({nelecas}e, {ncas}o) of "
                f"({self.active_electrons}e, {self.active_space_size}o), which has no excitation in it to correct"
            )

        self.orbital_initial, self.occupations = orbitals, occupations
        self.active_space_size, self.active_electrons = ncas, nelecas
        return self.active_space_size, self.active_electrons, self.orbital_initial

    def rewindow(self, lo: float, hi: float):
        """
        Choose another occupation window over a space already solved.
        """
        if self.shci_energy is None:
            raise EncodingError("Cannot rewindow before SHCI has solved the space")

        ncas, nelecas, orbitals, occupations = encode.select(
            self.orbital_initial, self.occupations, self.active_electrons, lo, hi
        )
        if ncas == 0 or nelecas == 0 or nelecas == 2 * ncas:
            raise EncodingError(
                f"The window {lo} <= n <= {hi} leaves ({nelecas}e, {ncas}o) of "
                f"({self.active_electrons}e, {self.active_space_size}o), which has no excitation in it to correct"
            )

        self.orbital_initial, self.occupations = orbitals, occupations
        self.active_space_size, self.active_electrons = ncas, nelecas
        return self.active_space_size, self.active_electrons, self.orbital_initial

    def H(self, mapping: str = "jordan_wigner"):
        """
        Map the active space onto qubits.

        The integrals CASCI builds over the window are handed to `mapping`, which is Jordan-Wigner
            by default. Two qubits per orbital, and the core energy carried as the identity so the
            eigenvalues are total energies.
        """
        if self.active_space_size is None:
            raise EncodingError("Cannot build the Hamiltonian before an active space is chosen")

        self.e_core, h1, h2 = encode.integrals(
            self.mean_field,
            self.orbital_initial,
            self.active_space_size,
            self.active_electrons,
        )
        try:
            self.hamiltonian = encode.qubits(self.e_core, h1, h2, mapping)
        except ValueError as error:
            raise EncodingError(str(error)) from error
        return self.hamiltonian

