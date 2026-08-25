from pdbfixer import PDBFixer
import gemmi
from rdkit import Chem
from enum import Enum


class OutOfScopeErrorType(Enum):
    """
    One member per eligibility rule   
    """
    METAL = "metal in the retained region"
    COFACTOR = "biological cofactor within the cutoff of a pose"
    HETEROGEN = "non-cofactor heterogen within the cutoff of a pose"
    UNSUPPORTED_ELEMENT = "element with no 6-31G basis"
    SPLIT_METAL_COORDINATION = "metal coordination sphere split by the cutout"
    CHARGED_LIGAND = "ligand with a non-zero formal charge"
    SIZE_CAP = "cutout exceeds the heavy-atom cap"
    INCOMPLETE_RESIDUE = "incomplete residue in the cutout"
    SPLIT_DISULFIDE = "disulfide split by the cutout"


class OutOfScopeError(RuntimeError):
    """
    Reject protein because it is outside of scope.
    """

    def __init__(self, error_type: OutOfScopeErrorType, message: str | None = None):
        super().__init__(message or error_type.value)
        self.error_type = error_type


class EncodingError(RuntimeError):
    pass


class CompressionError(RuntimeError):
    pass


class EncodeProtein:
    """
    EncodeProtein takes a holo-protein structure (.pdb) and candidate poses (.sdf) and encodes a 
        reduced protein structure with poses for SAPT(VQE). Aims to generalise the SAPT(VQE) method 
        to any protein-ligand complex.
    """

    def __init__(
        self,
        protein_path: str,
        poses_paths: list[str],
    ):
        self.protein_path = protein_path
        self.poses_paths = poses_paths
        self.whole = None
        self.reduced = None
        self.poses = None

        # Scope assumptions
        self.pH = 7.4
        self.cutoff = 4.5
        self.spin = 0
        self.multiplicity = 1

    def encode(self):
        self._fetch()
        self._verify()
        self._protonate()
        self._reduce()
        self._calculate_charge()
        self._verify_num_electrons()

    def _fetch(self):
        """
        Fetch the protein and candidate poses. Does not fix the protein, as the protein msut be 
            verified first.
        """
        proteins = gemmi.read_pdb(self.protein_path) # pyright: ignore[reportAttributeAccessIssue]
        if not len(proteins):
            raise EncodingError(f"No model found in {self.protein_path}")
        protein = proteins[0]
        poses = []
        for path in self.poses_paths:
            pose = Chem.MolFromMolFile(path, sanitize=True, removeHs=False) # pyright: ignore[reportAttributeAccessIssue]
            if pose is None:
                raise EncodingError(f"Could not parse pose {path}")
            poses.append(pose)
        self.whole = protein
        self.poses = poses

    def _verify(self):
        """
        Take a provisional cutout, reject/adjust the complex according to the scope, then fix 
            missing residues, atoms, terminals, etc.

        `PDBFixer.findMissingResidues` may not work due to lack of SEQRES records.
        """
        ...

    def _protonate(self):
        """
        Protonates the entire protein.
        """
        ...

    def _reduce(self):
        """
        Takes union of complete residues with at least one heavy atom within 4.5 Å of the nearest 
            pose heavy atom, then caps the truncated protein with ACE/NME.
        """
        ...

    def _calculate_charge(self):
        """
        Calculates total charge. Requires protonation.
        """
        ...

    def _verify_num_electrons(self):
        """
        N_e = \\sum_I Z_I - q_A. Ensure N_e is even.
        """
        ...

    def xyz(self, path: str):
        """
        Store element and coordinates in a .xyz file
        """
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
