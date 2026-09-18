import os
import shutil
import tempfile
from subprocess import CalledProcessError

from pyscf.mcscf import avas
from rdkit import Chem

from prepare import PrepareComplex, PrepareError
from utils.reduce import CAPS
from utils import encode, save


# The valence p shell per element. Hydrogen has no p shell; anything else in the cutout was 
# rejected by `_verify` for having no 6-31G basis.
VALENCE = {"C": "2p", "N": "2p", "O": "2p", "S": "3p", "P": "3p"}

SCF = ("e_tot", "mo_energy", "mo_occ", "mo_coeff")

# The Hamiltonian carries the window it was built over; the orbitals are the space's, which a 
# window does not change.
SOLVED = (
    "energy", "correlation", "shci_energy", "active_space_size", "active_electrons",
    "orbital_initial", "occupations",
)
ENCODED = ("e_core", "h1", "h2", "active_space_size", "active_electrons", "occupations")
# The ground state of the encoded space.
CORRELATED = ("casci_energy", "rdm1", "rdm2")

# <S^2> of a converged singlet is close to zero, and a triplet's is two.
SINGLET = 1e-6

DICE_LOG = "output.dat"


class EncodingError(RuntimeError):
    pass


class EncodeProtein:
    """
    EncodeProtein takes a PreparedComplex, solves RHF then encodes a tractable active space for
        SAPT(VQE) corrections. CASCI solves that space for the density matrices SAPT reads, standing
        in for VQE.

    `out` is the complex's directory. Whatever a previous run left there is read back, which needs
        `prepared` prepared or read back first.
    """

    def __init__(
        self,
        prepared: PrepareComplex,
        out=None
    ):
        self.prepared = prepared
        self.mol = None
        self.mean_field = None
        self.energy = None

        # RHF
        self.rhf_max_cycle = 50 # RHF maximum number of cycles for convergence.
        self.verbose = 0 # PySCF prints its SCF table. Silent by default. Set = 4 for logging.

        # AVAS
        self.active_space_size = None # active-space-size
        self.active_electrons = None # active-electrons
        self.orbital_initial = None # orbital-initial-guess-for-CASCI/CASSCF
        self.occupations = None # natural occupations of the active window
        self.correlation = None # MP2 correlation energy of the AVAS space
        self.shci_energy = None # SHCI total energy of the space MP2 capped
        self.solved_space = None # the space Dice solved
        self.cutoff = 4.5 # Cutoff for chemically relevant atoms.
        self.avas_threshold = 0.2 # AVAS threshold. PySCF's own default.
        
        # MP2
        self.density_fit = True # Density fit MP2
        self.nmax = 50 # Number of natural orbitals the MP2 caps

        # Dice
        self.dice = shutil.which("Dice") # `setup.sh` builds Dice into the environment's own bin.
        self.mpi = os.environ.get("MPIPREFIX", "") # empty runs on one rank. A cluster wants "srun" 
        # under SLURM, or "mpirun -np <ranks>"
        self.scratch = None # where to write integrals, wavefunction and RDMs.

        # Window
        self.ncas_limit = 14 # the most active orbitals a window permits

        # Hamiltonian
        self.e_core = None # nuclear repulsion and the energy of the frozen electrons
        self.h1 = None # one-electron integrals, the core's Coulomb and exchange folded in
        self.h2 = None # two-electron integrals over the active orbitals
        self.hamiltonian = None # the active space as a qubit operator

        # CASCI
        self.AS_limit = 16 # the most active orbitals CASCI solves
        self.casci_max_cycle = 100 # CASCI maximum number of iterations. PySCF's own default.
        self.casci_energy = None # CASCI total energy of the encoded space
        self.rdm1 = None # spin-summed one-particle density matrix over the active orbitals
        self.rdm2 = None # spin-summed two-particle density matrix over the active orbitals

        # Load
        self.out = out
        if out:
            self._load()


    def solve(self):
        self.e_core = self.h1 = self.h2 = self.hamiltonian = None
        self.casci_energy = self.rdm1 = self.rdm2 = None
        self.RHF()
        self.AVAS()
        self.MP2()
        self.SHCI()
        self.save()


    def encode(self):
        self.H()
        self.save()


    def RHF(self):
        """
        Solves RHF over the prepared protein for initial molecular orbitals. Computationally
            expensive and the driver behind the atom size cap.

        Restricted, so every electron is in a doubly occupied spatial orbital, which is what the
            even electron count `_verify_num_electrons` insisted on buys.

        The two-electron integrals are never held: a cutout carries many basis functions, whose 
            integrals run to petabytes, so PySCF builds them on the fly.

        An unconverged SCF is rejected. A converged one is kept in `out`.
        """
        if self.mean_field is not None and self.mean_field.converged:
            return self.mean_field
        self._molecule()
        mean_field = encode.rhf(self.mol, self.rhf_max_cycle)
        if not mean_field.converged:
            raise EncodingError(
                f"RHF did not converge in {self.rhf_max_cycle} cycles"
            )
        self.mean_field = mean_field
        self.energy = mean_field.e_tot
        if self.out:
            save.save_scf(
                {key: getattr(mean_field, key) for key in SCF}, self._name(), self.out
            )
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

        AVAS returns semicanonical orbitals but not their energies. The energies are recomputed 
            from the Fock matrix, and the mean field is restored afterwards.

        The occupied-virtual block of the unrelaxed MP2 density is zero, so diagonalising the two
            blocks separately loses nothing.
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

    def SHCI(self, eps1: float = 1e-4, lo: float = 0, hi: float = 2):
        """
        Truncate the active space to the correlated orbitals.

        Semistochastic Heat-bath Configuration Interaction (SHCI) solves the space MP2 capped. Its
            one-particle density is diagonalised into natural orbitals. An occupation near two or
            near zero is described by a single determinant, so only lo <= n_i <= hi is kept: an 
            orbital above the window is doubly occupied and core, one below it is empty and virtual.

        `eps1` is the selection threshold, below which a determinant is left out of the variational 
            space. Smaller is nearer exact and costs more.

        Window defaults to everything, which can be rewindowed after.

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
        finally:
            log = os.path.join(scratch, DICE_LOG)
            if self.out and os.path.isfile(log):
                os.makedirs(self.out, exist_ok=True)
                shutil.copyfile(log, save.dice_log_path(self._name(), self.out))
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
        self.solved_space = {key: getattr(self, key) for key in SOLVED}
        return self.active_space_size, self.active_electrons, self.orbital_initial

    def rewindow(self, lo: float | None = None, hi: float | None = None):
        """
        Choose another occupation window over a solved space.

        The original paper cut at fixed thresholds, 0.02 <= n <= 1.97. A window with no thresholds 
        derives them instead: the `ncas_limit` orbitals a single determinant describes worst.
        """
        if self.solved_space is None:
            raise EncodingError("Cannot rewindow before SHCI has solved the space")
        if (lo is None) != (hi is None):
            raise EncodingError(
                "A window is both thresholds or neither; neither derives one from `ncas_limit`"
            )

        solved, occupations = self.solved_space, self.solved_space["occupations"]
        if lo is None:
            lo, hi = encode.thresholds(occupations, self.ncas_limit)
        ncas, nelecas, orbitals, occupations = encode.select(
            solved["orbital_initial"], occupations, solved["active_electrons"], lo, hi
        )
        if ncas == 0 or nelecas == 0 or nelecas == 2 * ncas:
            raise EncodingError(
                f"The window {lo} <= n <= {hi} leaves ({nelecas}e, {ncas}o) of "
                f"({solved['active_electrons']}e, {solved['active_space_size']}o), which has no "
                "excitation in it to correct"
            )

        self.e_core = self.h1 = self.h2 = self.hamiltonian = None
        self.casci_energy = self.rdm1 = self.rdm2 = None
        self.orbital_initial, self.occupations = orbitals, occupations
        self.active_space_size, self.active_electrons = ncas, nelecas
        return self.active_space_size, self.active_electrons, self.orbital_initial

    def H(self, mapping: str = "jordan_wigner"):
        """
        Map the active space onto qubits.

        The integrals CASCI builds over the window are handed to `mapping`, which is Jordan-Wigner
            by default. Two qubits per orbital, and the core energy carried as the identity so the
            eigenvalues are total energies.

        The integrals are kept as well as the operator. They are what the driver writes, because
            they rebuild the operator under any mapping and are much smaller than it.

        A CASCI solution of the last Hamiltonian is discarded.
        """
        if self.active_space_size is None:
            raise EncodingError("Cannot build the Hamiltonian before an active space is chosen")
        self.casci_energy = self.rdm1 = self.rdm2 = None

        self.e_core, self.h1, self.h2 = encode.integrals(
            self.mean_field,
            self.orbital_initial,
            self.active_space_size,
            self.active_electrons,
        )
        try:
            self.hamiltonian = encode.qubits(self.e_core, self.h1, self.h2, mapping)
        except ValueError as error:
            raise EncodingError(str(error)) from error
        return self.hamiltonian

    def CASCI(self):
        """
        Solve the encoded space exactly, for the density matrices SAPT reads the protein through.

        Complete Active Space Configuration Interaction (CASCI) diagonalises the Hamiltonian H() 
            kept. It stands in for VQE, as the reference the paper benchmarked VQE against.

        The energy is kept, and the ground state's spin-summed one- and two-particle density
            matrices over the active orbitals, in PySCF's convention: rdm1[p, q] = <q+ p> and
            rdm2[p, q, r, s] = <p+ r+ s q>.

        A space of more than `AS_limit` orbitals is refused for computational constraints.
        
        RHF assumed a singlet, and SAPT's density matrices are written for one, so a ground state of
            any other spin is refused.
        """
        if self.e_core is None or self.h1 is None or self.h2 is None:
            raise EncodingError("Cannot solve the active space before its Hamiltonian is built")
        if self.active_space_size > self.AS_limit:
            raise EncodingError(
                f"({self.active_electrons}e, {self.active_space_size}o) is wider than the "
                f"{self.AS_limit} orbitals CASCI solves"
            )

        converged, energy, spin_square, rdm1, rdm2 = encode.casci(
            self.e_core,
            self.h1,
            self.h2,
            self.active_space_size,
            self.active_electrons,
            self.casci_max_cycle,
            self.verbose,
        )
        if not converged:
            raise EncodingError(f"CASCI did not converge in {self.casci_max_cycle} cycles")
        if abs(spin_square) > SINGLET:
            raise EncodingError(
                f"The ground state of ({self.active_electrons}e, {self.active_space_size}o) is not a "
                f"singlet: <S^2> = {spin_square:.3f}"
            )

        self.casci_energy, self.rdm1, self.rdm2 = energy, rdm1, rdm2
        self.save()
        return self.casci_energy, self.rdm1, self.rdm2


    def solved(self):
        """
        Whether SHCI has solved the space.
        """
        return self.shci_energy is not None


    def encoded(self):
        """
        Whether H() has run over the solved space.
        """
        return self.solved() and self.e_core is not None


    def correlated(self):
        """
        Whether CASCI has solved the encoded space.
        """
        return self.encoded() and self.casci_energy is not None


    def _load(self):
        """
        Read back the SCF, the solved space, the Hamiltonian and its CASCI solution, whichever of them
            `out` holds.
        """
        name = self._name()
        stored = save.load_scf(name, self.out)
        solved = save.load_solved(name, self.out)
        if stored is None and solved is None:
            return

        self._molecule()
        self.mean_field = encode.restore(self.mol, stored)
        if stored is not None:
            self.energy = self.mean_field.e_tot
        if solved is None:
            return
        for key in SOLVED:
            setattr(self, key, solved[key])
        self.solved_space = solved

        encoded = save.load_encoded(name, self.out)
        if encoded is None:
            return
        for key in ENCODED:
            setattr(self, key, encoded[key])

        correlated = save.load_casci(name, self.out)
        if correlated is not None:
            for key in CORRELATED:
                setattr(self, key, correlated[key])


    def save(self):
        """
        Write the latest stage: the space SHCI solved, the integrals H() built over it and their
            window, or the state CASCI solved them to.
        """
        if not self.out:
            return
        if self.casci_energy is not None:
            save.save_casci(
                {key: getattr(self, key) for key in CORRELATED}, self._name(), self.out
            )
        elif self.e_core is not None:
            save.save_encoded(
                {key: getattr(self, key) for key in ENCODED}, self._name(), self.out
            )
        elif self.shci_energy is not None:
            save.save_solved(
                self.solved_space or {key: getattr(self, key) for key in SOLVED},
                self._name(),
                self.out,
            )


    def _name(self):
        """
        The complex, which names the directory its artefacts are kept in.
        """
        return os.path.basename(os.path.normpath(self.out))


class SolveLigand:
    """
    SolveLigand takes a PreparedComplex and solves RHF over each of its poses.

    A job resumes from the poses it solved.

    `out` is the complex's directory. Whatever a previous run left there is read back, which needs
        `prepared` prepared or read back first.
    """

    def __init__(
        self,
        prepared: PrepareComplex,
        out=None
    ):
        self.prepared = prepared
        self.mols = None
        self.mean_fields = None
        self.energies = None # the RHF energy per pose

        # RHF
        self.rhf_max_cycle = 50
        self.verbose = 0 # PySCF prints its SCF table. Silent by default. Set = 4 for logging.

        # Load
        self.out = out
        if out:
            self._load()


    def RHF(self):
        """
        Solves RHF over every pose not yet solved, in order.

        Only the solution is kept. What is kept of a pose is its solution. PySCF holds a molecule's two-electron integrals on its
            mean field and an entire ensemble's worth would not fit in memory.

        An unconverged pose is rejected, and the poses solved before it are kept.
        """
        if self.mols is None:
            self._molecules()
        if self.mean_fields is None:
            self.mean_fields = [None] * len(self.mols)
            self.energies = [None] * len(self.mols)
        for index, mol in enumerate(self.mols):
            if self.mean_fields[index] is not None:
                continue
            mean_field = encode.rhf(mol, self.rhf_max_cycle)
            if not mean_field.converged:
                raise EncodingError(
                    f"RHF did not converge in {self.rhf_max_cycle} cycles over pose "
                    f"{self.prepared.source[index]}"
                )
            record = {key: getattr(mean_field, key) for key in SCF}
            if self.out:
                save.save_pose_scf(record, self._name(), self.out, index)
            self.mean_fields[index] = encode.restore(mol, record)
            self.energies[index] = mean_field.e_tot
        return self.mean_fields

    def _molecules(self):
        """
        Build the PySCF molecule each pose is solved as.

        N_B = \\sum_I Z_I - q_B. Preparation leaves every pose a closed shell, so an odd N_B means the
            preparation is wrong, and the pose is refused before PySCF is handed it.
        """
        if not self.prepared.prepared():
            raise PrepareError("Cannot build the poses before the complex is prepared")
        for pose, source in zip(self.prepared.poses, self.prepared.source):
            electrons = (
                sum(atom.GetAtomicNum() for atom in pose.GetAtoms())
                - Chem.GetFormalCharge(pose) # pyright: ignore[reportAttributeAccessIssue]
            )
            if electrons % 2:
                raise PrepareError(
                    f"pose {source} holds {electrons} electrons, which no closed-shell singlet can hold"
                )
        self.mols = [
            encode.pose_molecule(pose, self.prepared.spin, self.prepared.basis, self.verbose)
            for pose in self.prepared.poses
        ]
        return self.mols


    def solved(self):
        """
        Whether RHF has solved every pose.
        """
        return self.mean_fields is not None and all(
            mean_field is not None for mean_field in self.mean_fields
        )


    def _load(self):
        """
        Read back the SCF for all poses in `out`.
        """
        name = self._name()
        stored = [
            save.load_pose_scf(name, self.out, index)
            for index in range(len(self.prepared.poses or []))
        ]
        if all(record is None for record in stored):
            return

        self._molecules()
        self.mean_fields = [
            None if record is None else encode.restore(mol, record)
            for mol, record in zip(self.mols, stored)
        ]
        self.energies = [
            None if mean_field is None else mean_field.e_tot for mean_field in self.mean_fields
        ]


    def _name(self):
        """
        The complex, which names the directory its artefacts are kept in.
        """
        return os.path.basename(os.path.normpath(self.out))
