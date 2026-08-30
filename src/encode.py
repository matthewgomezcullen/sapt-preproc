from pyscf import gto, scf

from prepare import PrepareComplex, PrepareError

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

    def molecule(self):
        """
        Build the PySCF molecule the SCF is solved over.

        The geometry is taken in `prepared.atoms()` order, which is the order AVAS addresses its
            targets by. PySCF assumes a molecule it is given no charge for is neutral, so q_A is
            passed explicitly: the neutral molecule of the same geometry converges just as happily
            and answers a different question.
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

    def AVAS(self):
        """
        Generates a tractable active space with target active orbitals.
        """
        target_aos = self._generate_target_orbitals()
        ...

    def _generate_target_orbitals(self) -> list[str]:
        """
        Generate AVAS target orbitals based on which atoms are "chemically relevant".

        See README.md for how "chemically relevant" is defined.
        """
        ...

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

