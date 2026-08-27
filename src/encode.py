import gemmi
import numpy as np
from rdkit import Chem
from pyscf.gto.basis import load as load_basis
from pyscf.lib.exceptions import BasisNotFoundError
from scipy.spatial import cKDTree # pyright: ignore[reportAttributeAccessIssue]
from enum import Enum
from utils import clean, fix, verify


# PDB chemical component IDs for the biological cofactors. Decides which rejection is reported. 
# Any heterogen in the shell is out of scope either way.
COFACTORS = frozenset({
    "FAD", "FMN",                                   # flavins
    "NAD", "NAI", "NAJ", "NAP", "NDP",              # NAD(P)(H)
    "SAM", "SAH",                                   # S-adenosyl methionine/homocysteine
    "ATP", "ADP", "AMP", "UDP", "UTP", "UMP",       # nucleotides
    "PLP",                                          # pyridoxal phosphate
    "ACO", "COA",                                   # acetyl-CoA and coenzyme A
})


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
    ZERO_OCCUPANCY = "zero-occupancy heavy atom in the cutout"
    CHAIN_BREAK = "chain break in the cutout"
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


def _is_amino_acid(name):
    info = gemmi.find_tabulated_residue(name)
    return bool(info) and info.is_amino_acid()


def _is_water(name):
    info = gemmi.find_tabulated_residue(name)
    return bool(info) and info.is_water()


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

        # Eligibility thresholds
        self.basis = "6-31g"
        self.metal_coordination_cutoff = 2.8
        self.disulfide_cutoff = 2.5
        self.size_cap = 400

    def encode(self):
        self._fetch()
        self._verify()
        self._fix()
        self._clean()
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
        Take a provisional cutout and reject the complex according to the scope.

        Repairing the structure is deferred; `PDBFixer.findMissingResidues` may not work due to
            lack of SEQRES records, so chain breaks are not detected here.
        """
        retained = verify.cutout(self.whole, self._pose_coordinates(), self.cutoff)
        residues = [(chain, residue) for chain, residue, _, _ in retained]

        metals = verify.metals(self.whole)
        for _, residue, heavy, _ in retained:
            for atom in heavy:
                if atom.element.is_metal:
                    raise OutOfScopeError(
                        OutOfScopeErrorType.METAL,
                        f"{atom.element.name} in retained residue {residue.name}",
                    )

        elements = {atom.element.name for _, _, heavy, _ in retained for atom in heavy}
        elements |= {atom.GetSymbol() for pose in self.poses for atom in pose.GetAtoms()}
        for element in sorted(elements):
            try:
                load_basis(self.basis, element)
            except BasisNotFoundError:
                raise OutOfScopeError(
                    OutOfScopeErrorType.UNSUPPORTED_ELEMENT,
                    f"{element} has no {self.basis} basis",
                )

        heterogens = {
            residue.name
            for _, residue in residues
            if not _is_amino_acid(residue.name) and not _is_water(residue.name)
        }
        cofactors = heterogens & COFACTORS
        if cofactors:
            raise OutOfScopeError(
                OutOfScopeErrorType.COFACTOR,
                f"cofactor(s) {sorted(cofactors)} within {self.cutoff} A of a pose",
            )
        if heterogens:
            raise OutOfScopeError(
                OutOfScopeErrorType.HETEROGEN,
                f"heterogen(s) {sorted(heterogens)} within {self.cutoff} A of a pose",
            )

        if metals:
            coordinates = np.vstack([c for _, _, _, c in retained])
            distances, _ = cKDTree(coordinates).query([position for _, position in metals])
            if distances.min() < self.metal_coordination_cutoff:
                raise OutOfScopeError(
                    OutOfScopeErrorType.SPLIT_METAL_COORDINATION,
                    f"metal {distances.min():.2f} A from a retained residue",
                )

        for path, pose in zip(self.poses_paths, self.poses):
            charge = Chem.GetFormalCharge(pose) # pyright: ignore[reportAttributeAccessIssue]
            if charge:
                raise OutOfScopeError(
                    OutOfScopeErrorType.CHARGED_LIGAND,
                    f"pose {path} carries formal charge {charge:+d}",
                )

        incomplete = verify.incomplete_residues(self.protein_path) & {
            verify.identifier(chain, residue) for chain, residue in residues
        }
        if incomplete:
            raise OutOfScopeError(
                OutOfScopeErrorType.INCOMPLETE_RESIDUE,
                f"{len(incomplete)} incomplete residue(s) in the cutout, e.g. {min(incomplete)}",
            )

        # Hydrogens are exempt because _clean deletes every deposited one before _protonate assigns
        # its own, so a zero-occupancy hydrogen never reaches the QM region.
        unoccupied = [
            (verify.identifier(chain, residue), atom.name)
            for chain, residue, heavy, _ in retained
            for atom in heavy
            if not atom.occ
        ]
        if unoccupied:
            raise OutOfScopeError(
                OutOfScopeErrorType.ZERO_OCCUPANCY,
                f"{len(unoccupied)} zero-occupancy heavy atom(s) in the cutout, "
                f"e.g. {unoccupied[0][1]} of {unoccupied[0][0]}",
            )

        half = verify.split_disulfide(self.whole, residues, self.disulfide_cutoff)
        if half:
            raise OutOfScopeError(
                OutOfScopeErrorType.SPLIT_DISULFIDE,
                f"cutout retains CYS {half} without its disulfide partner",
            )

        heavy_atoms = sum(len(heavy) for _, _, heavy, _ in retained)
        if heavy_atoms > self.size_cap:
            raise OutOfScopeError(
                OutOfScopeErrorType.SIZE_CAP,
                f"cutout holds {heavy_atoms} heavy atoms, over the cap of {self.size_cap}",
            )

    def _fix(self):
        """
        Fix missing atoms, residues, and terminal atoms.
        """
        repaired = fix.repair(self.protein_path)
        if not len(repaired):
            raise EncodingError(f"Repairing {self.protein_path} left no model")
        self.whole = repaired[0]

    def _clean(self):
        """
        Delete out-of-scope molecules.
        """
        self.whole = clean.strip(self.whole)

    def _pose_coordinates(self):
        """
        Heavy-atom coordinates of every candidate pose stacked into one array.
        """
        assert self.poses
        return np.vstack([
            pose.GetConformer().GetPositions()[
                [atom.GetIdx() for atom in pose.GetAtoms() if atom.GetAtomicNum() > 1]
            ]
            for pose in self.poses
        ])

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
