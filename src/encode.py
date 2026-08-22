class EncodeProtein:
    # EncodeProtein takes a holo-protein structure (.pdb) and candidate poses (.sdf) and encodes a 
    #   reduced protein structure with poses for SAPT(VQE). Aims to generalise the SAPT(VQE) method 
    #   to any protein-ligand complex.

    def __init__(self, protein_path: str, poses_paths: list[str], pH: float=7.4, cutoff: float=4.5):
        self.protein_path = protein_path    # A
        self.poses_paths = poses_paths      # {B}
        self.reduced = None                 # A^{\cup}
        self.protonation = None             # {site_id : {pka, state, tautomer}}
        self.charge = None                  # q_{A^{\cup}}
        self.num_electrons= None            # N_e
        # Scope assumptions
        self.spin = 0
        self.multiplicity = 1 
        self.pH = pH
        self.cutoff = cutoff

    def encode(self):
        self._protonate()
        self._reduce()
        self._calculate_charge()
        self._verify_num_electrons()

    def _reduce(self):
        # Takes union of complete residues with at least one heavy atom within 4.5 Å of the 
        #   nearest pose heavy atom, then caps the truncated protein with ACE/NME.
        if self.protonation is None:
            raise RuntimeError("Must protonate protein before reducing the complex.")
        ...

    def _protonate(self):
        # Protonate the full receptor once with PROPKA. The result maps sites (pK_a) to their 
        # protonation state.
        ...

    def _calculate_charge(self):
        # Calculates total charge. Requires protonation.
        ...

    def _verify_num_electrons(self):
        # N_e = \sum_I Z_I - q_A. Ensure N_e is even.
        ...

    def xyz(self, path: str):
        # Store element and coordinates in a .xyz file
        ...

class CompressProtein:
    # CompressProtein takes an encoded holo-protein structure and reduces it to a tractable 
    #   active space for VQE/CASCI experiments.
    
    def __init__(self, encoding: EncodeProtein, name: str, basis: str="6-31g"):
        self.encoding = encoding
        self.encoding.xyz(f"{name}.xyz")
        self.basis = basis
        self.molecular_orbitals = None
        self.ferm_hamiltonian = None        # H_f
        self.qubit_hamiltonian = None       # H_JW
        
    def RHF(self):
        # Solve Hartree-Fock equations with PySCF, populating the molecular orbits.
        ...

    def AVAS(self, target_aos: list[str] | None=None):
        # Run Atomic Valence Active Space over the MOs and a chosen set of atomic valence orbitals. 
        # If a chosen set is not provided, deterministically generate our own.
        if target_aos is None:
            target_aos = self._generate_target_orbitals()
        ...

    def _generate_target_orbitals(self):
        # **NOVEL WORK**: Generate AVAS target set based on which atoms count as chemically 
        #   relevant.
        ...

    def SHCI(self, eps=1e-4, lo=0.02, hi=1.97):
        # Run Semistochastic Heat-Bath Configuration Interaction. Use Dice as the CI solver in the 
        # AVAS CASCI SPACE, then keep only orbitals satisfying lo \le n_i \le hi.
        ...

    def H_fermionic(self):
        # Construct and retain the active-space fermionic Hamiltonian with PySCF.
        ...

    def H_JW(self):
        # Maps fermionic electronic-structure Hamiltonian into a qubit Hamiltonian following the 
        #   Jordan-Wigner transformation.
        ...
