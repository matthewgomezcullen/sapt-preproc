import gemmi
import numpy as np
from rdkit import Chem
from pyscf.gto.basis import load as load_basis
from pyscf.lib.exceptions import BasisNotFoundError
from scipy.spatial import cKDTree # pyright: ignore[reportAttributeAccessIssue]
from enum import Enum
from utils import charge, clean, fix, protonate, reduce, verify


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


# PDB chemical component IDs for crystallisation additives: cryoprotectants, precipitants, buffers,
# and simple non-metal ions. None of these is part of the biology of the site, and nothing is bonded
# to them, so one inside the shell is deleted by _clean rather than rejected. Metal-ion additives are
# absent because a metal in the retained region is out of scope whatever put it there.
ADDITIVES = frozenset({
    "GOL", "EDO", "MPD",                            # cryoprotectants
    "PEG", "PG4", "PGE", "P6G", "1PE", "2PE",       # polyethylene glycols
    "SO4", "PO4", "NO3", "SCN", "IOD", "CL", "BR",  # simple non-metal ions
    "ACT", "FMT", "ACY", "LAC", "OXL",              # short-chain carboxylates
    "CIT", "FLC", "MLI", "TLA",                     # di- and tricarboxylates
    "TRS", "MES", "EPE", "IMD", "BTB", "B3P",       # buffers
    "DMS", "BME",                                   # solvent and reducing agent
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


class PrepareError(RuntimeError):
    pass


class CompressionError(RuntimeError):
    pass


def _is_amino_acid(name):
    info = gemmi.find_tabulated_residue(name) # pyright: ignore[reportAttributeAccessIssue]
    return bool(info) and info.is_amino_acid()


def _is_water(name):
    info = gemmi.find_tabulated_residue(name) # pyright: ignore[reportAttributeAccessIssue]
    return bool(info) and info.is_water()


class PrepareComplex:
    """
    PrepareComplex takes a holo-protein structure (.pdb) and candidate poses (.sdf) and prepares a 
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
        self.protonation = None
        self.charge = None
        self.electrons = None

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

        # Seeds the minimisations that place rebuilt atoms and new hydrogens, so that both land in
        # the same place every run. Not zero: OpenMM reads a zero seed as a request for a random one.
        self.seed = 1

    def prepare(self):
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
            raise PrepareError(f"No model found in {self.protein_path}")
        protein = proteins[0]
        poses = []
        for path in self.poses_paths:
            pose = Chem.MolFromMolFile(path, sanitize=True, removeHs=False) # pyright: ignore[reportAttributeAccessIssue]
            if pose is None:
                raise PrepareError(f"Could not parse pose {path}")
            poses.append(pose)
        self.whole = protein
        self.poses = poses

    def _verify(self):
        """
        Take a provisional cutout and reject the complex according to the scope.

        Repairing the structure is deferred; `PDBFixer.findMissingResidues` may not work due to
            lack of SEQRES records, so chain breaks are not detected here.
        """
        if self.poses is None:
            raise PrepareError("Cannot verify the complex without poses")
        retained = verify.cutout(self.whole, self._pose_coordinates(), self.cutoff)
        residues = [(chain, residue) for chain, residue, _, _ in retained]

        prepared = [entry for entry in retained if entry[1].name not in ADDITIVES]

        metals = verify.metals(self.whole)
        for _, residue, heavy, _ in retained:
            for atom in heavy:
                if atom.element.is_metal:
                    raise OutOfScopeError(
                        OutOfScopeErrorType.METAL,
                        f"{atom.element.name} in retained residue {residue.name}",
                    )

        elements = {atom.element.name for _, _, heavy, _ in prepared for atom in heavy}
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
        heterogens -= ADDITIVES
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
            coordinates = np.vstack([c for _, _, _, c in prepared])
            distances, _ = cKDTree(coordinates).query([position for _, position in metals])
            if distances.min() < self.metal_coordination_cutoff:
                raise OutOfScopeError(
                    OutOfScopeErrorType.SPLIT_METAL_COORDINATION,
                    f"metal {distances.min():.2f} A from a retained residue",
                )

        for path, pose in zip(self.poses_paths, self.poses):
            formal = Chem.GetFormalCharge(pose) # pyright: ignore[reportAttributeAccessIssue]
            if formal:
                raise OutOfScopeError(
                    OutOfScopeErrorType.CHARGED_LIGAND,
                    f"pose {path} carries formal charge {formal:+d}",
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
            for chain, residue, heavy, _ in prepared
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

        heavy_atoms = sum(len(heavy) for _, _, heavy, _ in prepared)
        if heavy_atoms > self.size_cap:
            raise OutOfScopeError(
                OutOfScopeErrorType.SIZE_CAP,
                f"cutout holds {heavy_atoms} heavy atoms, over the cap of {self.size_cap}",
            )

    def _fix(self):
        """
        Fix missing atoms, residues, and terminal atoms.
        """
        repaired = fix.repair(self.protein_path, self.seed)
        if not len(repaired):
            raise PrepareError(f"Repairing {self.protein_path} left no model")
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
        Protonates the entire protein, recording the state chosen for each residue.
        """
        self.whole, self.protonation = protonate.hydrogens(self.whole, self.pH, self.seed)

    def _reduce(self):
        """
        Takes union of complete residues with at least one heavy atom within 4.5 Å of the nearest
            pose heavy atom, then caps the truncated protein with ACE/NME.

        Runs of retained residues separated by a single residue are bridged rather than capped
            around: capping both sides would take that residue's backbone into an ACE on one side
            and an NME on the other, placing the same atoms twice.

        A cut is capped; a chain end is not. A cap stands in for a residue the truncation removed
            and takes its backbone coordinates from the structure, and at a chain end there is no
            such residue to take them from. However, that costs a charge.

        TODO: Separate the cases with the SEQRES records

        A residue whose side chain _fix rebuilt into the cutout is rejected here. The size cap is 
            applied again
        """
        keep = {
            verify.identifier(chain, residue)
            for chain, residue, _, _ in verify.cutout(
                self.whole, self._pose_coordinates(), self.cutoff
            )
        }

        repaired = verify.incomplete_residues(self.protein_path) & keep
        if repaired:
            raise OutOfScopeError(
                OutOfScopeErrorType.INCOMPLETE_RESIDUE,
                f"{len(repaired)} residue(s) repaired into the cutout, e.g. {min(repaired)}",
            )

        self.reduced = reduce.truncate(
            self.whole, keep, self.protonation, self.pH, self.seed
        )

        heavy_atoms = sum(
            1
            for chain in self.reduced
            for residue in chain
            for atom in residue
            if not atom.element.is_hydrogen
        )
        if heavy_atoms > self.size_cap:
            raise OutOfScopeError(
                OutOfScopeErrorType.SIZE_CAP,
                f"capped cutout holds {heavy_atoms} heavy atoms, over the cap of {self.size_cap}",
            )

    def _calculate_charge(self):
        """
        Calculates total charge.
        """
        self.charge = charge.net(self.reduced)

    def _verify_num_electrons(self):
        """
        N_e = \\sum_I Z_I - q_A. Ensure N_e is even.

        Every residue and cap the pipeline keeps is closed-shell, so the parity of \\sum_I Z_I fixes
            the parity q_A must have and an odd N_e means the preparation is wrong rather than the
            complex being out of scope. It checks if q_A is out by one and rejects an RHF that
            cannot be solved as a closed-shell singlet.
        """
        if self.reduced is None or self.charge is None:
            raise PrepareError("Cannot verify number of electrons before the protein is prepared")
        nuclear = sum(
            atom.element.atomic_number
            for chain in self.reduced
            for residue in chain
            for atom in residue
        )
        self.electrons = nuclear - self.charge
        if self.electrons % 2:
            raise PrepareError(
                f"cutout of {nuclear} nuclear charge at q_A = {self.charge:+d} holds "
                f"{self.electrons} electrons, which no closed-shell singlet can hold"
            )

    def xyz(self, path: str):
        """
        Store element and coordinates in a .xyz file, returning the atoms in the order written.

        PySCF numbers its atoms in file order and the AVAS addresses target orbitals by that 
            number. The file itself keeps no record of which residue an atom came from. The returned
             (chain, residue, atom) triples record this.

        An .xyz carries no charge and PySCF assumes neutral, so the comment line records the charge
            and multiplicity.
        """
        if self.reduced is None or self.charge is None:
            raise PrepareError(f"Cannot write {path} before the protein is prepared")

        atoms = [
            (chain, residue, atom)
            for chain in self.reduced
            for residue in chain
            for atom in residue
        ]
        with open(path, "w") as file:
            file.write(f"{len(atoms)}\n")
            file.write(f"charge={self.charge} multiplicity={self.multiplicity}\n")
            for _, _, atom in atoms:
                file.write(
                    f"{atom.element.name:<2}"
                    f"{atom.pos.x:14.6f}{atom.pos.y:14.6f}{atom.pos.z:14.6f}\n"
                )
        return atoms

