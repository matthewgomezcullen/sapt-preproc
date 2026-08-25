from pdbfixer import PDBFixer
import gemmi


class EncodingError(RuntimeError):
    pass


class CompressionError(RuntimeError):
    pass


class EncodeProtein:
    # EncodeProtein takes a holo-protein structure (.pdb) and candidate poses (.sdf) and encodes a 
    #   reduced protein structure with poses for SAPT(VQE). Aims to generalise the SAPT(VQE) method 
    #   to any protein-ligand complex.

    def __init__(
        self,
        protein_path: str,
        poses_paths: list[str],
        pH: float = 7.4,
        cutoff: float = 4.5,
    ):
        self.protein_path = protein_path
        self.poses_paths = poses_paths
        self.pH = pH
        self.cutoff = cutoff
        self.whole = None
        self.reduced = None

        # Scope assumptions
        self.spin = 0                           # PySCF: N_alpha - N_beta = 2S
        self.multiplicity = 1

    def encode(self):
        self._fetch()
        self._protonate()
        self._reduce()
        self._calculate_charge()
        self._verify_num_electrons()

    def _fetch(self):
        proteins = gemmi.read_pdb(self.protein_path)
        if not len(proteins):
            raise EncodingError(f"No model found in {self.protein_path}")
        protein = proteins[0]

        protein.remove_alternative_conformations()
        protein.remove_empty_strings().make_pdb_string()

        fixer = PDBFixer()
        self.whole = protein

    def _protonate(self):
        # Protonate the full receptor once with PROPKA. The result maps sites (pK_a) to their 
        # protonation state.
        ...

    def _reduce(self):
        # Takes union of complete residues with at least one heavy atom within 4.5 Å of the 
        #   nearest pose heavy atom, then caps the truncated protein with ACE/NME.
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
    
    def __init__(
        self,
        encoding: EncodeProtein,
        name: str | None = None,
        basis: str = "6-31g",
        avas_threshold: float = 0.2,
        avas_minao: str = "minao",
        avas_with_iao: bool = False,
        avas_canonicalize: bool = True,
    ):
        self.encoding = encoding
        self.name = name
        self.basis = basis

        # RHF/PySCF state. Construction must not write files or run a calculation.
        self.mol = None
        self.mf = None
        self.molecular_orbitals = None
        self.mo_occupations = None
        self.mo_energies = None

        # AVAS inputs and outputs.
        self.avas_threshold = avas_threshold
        self.avas_minao = avas_minao
        self.avas_with_iao = avas_with_iao
        self.avas_canonicalize = avas_canonicalize
        self.target_aos = None
        self.avas_num_electrons = None
        self.avas_num_orbitals = None
        self.avas_molecular_orbitals = None

        # SHCI natural-orbital selection state.
        self.shci_solver = None
        self.shci_result = None
        self.shci_eps1 = None
        self.occupation_window = None
        self.one_rdm = None
        self.natural_orbital_occupations = None
        self.natural_orbital_coefficients = None
        self.active_orbital_indices = None
        self.active_num_electrons = None
        self.active_num_orbitals = None
        self.core_orbital_indices = None
        self.virtual_orbital_indices = None

        # Active-space Hamiltonian state.
        self.core_energy = None
        self.one_body_integrals = None
        self.two_body_integrals = None
        self.ferm_hamiltonian = None        # H_f
        self.qubit_hamiltonian = None       # H_JW
        
    def RHF(self):
        # Solve Hartree-Fock equations with PySCF, populating the molecular orbits.
        ...

    def AVAS(self, target_aos: list[str] | None = None):
        # Run Atomic Valence Active Space over the MOs and a chosen set of atomic valence orbitals. 
        # If a chosen set is not provided, deterministically generate our own.
        if target_aos is None:
            target_aos = self._generate_target_orbitals()
        self.target_aos = target_aos
        ...

    def _generate_target_orbitals(self) -> list[str]:
        # **NOVEL WORK**: Generate AVAS target set based on which atoms count as chemically 
        #   relevant.
        ...

    def SHCI(self, eps1: float = 1e-4, lo: float = 0.02, hi: float = 1.97):
        # Run Semistochastic Heat-Bath Configuration Interaction. Use Dice as the CI solver in the 
        # AVAS CASCI SPACE, then keep only orbitals satisfying lo \le n_i \le hi.
        self.shci_eps1 = eps1
        self.occupation_window = (lo, hi)
        ...

    def H_fermionic(self):
        # Construct and retain the active-space fermionic Hamiltonian with PySCF.
        ...

    def H_JW(self):
        # Maps fermionic electronic-structure Hamiltonian into a qubit Hamiltonian following the 
        #   Jordan-Wigner transformation.
        ...
