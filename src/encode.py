from prepare import PrepareComplex

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

        # Scope assumptions
        self.basis = "6-31g"

    def RHF(self):
        """
        Solves RHF over the prepared protein for initial molecular orbitals. Computationally 
            expensive and the main driver behind the atom size cap.
        """
        ...

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

