from pyscf import gto, scf
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

        # Enough for a well-behaved cutout. A cutout that reaches this has failed.
        self.max_cycle = 50

        # PySCF prints its SCF table at its own default. Silent by default. Turn it up for a run on the
        #   cluster
        self.verbose = 0

        self.ncas = None # active-space-size
        self.nelecas = None # active-electrons
        self.orbitals = None # orbital-initial-guess-for-CASCI/CASSCF

        # How close to a pose an atom has to be for its valence p shell to be targeted. Distinct
        # from `prepared.cutoff`, which decides what the cutout holds at all. The two may differ.
        self.cutoff = 4.5

        # Projection weight above which AVAS keeps an orbital. PySCF's own default, recorded here
        #   because it sets the size of the active space and so has to be reported with a result.
        self.threshold = 0.2

    def molecule(self):
        """
        Build the PySCF molecule the SCF is solved over.

        The geometry is taken in `prepared.atoms()` order, which is the order AVAS addresses its
            targets by. PySCF assumes a molecule it is given no charge for is neutral, so q_A is
            passed explicitly.
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

    def RHF(self):
        """
        Solves RHF over the prepared protein for initial molecular orbitals. Computationally
            expensive and the main driver behind the atom size cap.

        Restricted, so every electron is in a doubly occupied spatial orbital, which is what the
            even electron count `_verify_num_electrons` insisted on buys.

        The two-electron integrals are never held: a cutout carries many basis functions, whose 
            integrals run to petabytes, so PySCF builds them on the fly.

        An unconverged SCF is rejected.
        """
        self.molecule()

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

    def SHCI(self, eps1: float = 1e-4, lo: float = 0.02, hi: float = 1.97):
        """
        Run Semistochastic Heat-bath Configuration Interaction to reduce the number of active 
            orbitals.
        """
        ...

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

