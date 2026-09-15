import os
from collections import Counter

import gemmi
import numpy as np
from rdkit import Chem
from pyscf.gto.basis import load as load_basis
from pyscf.lib.exceptions import BasisNotFoundError
from scipy.spatial import cKDTree # pyright: ignore[reportAttributeAccessIssue]
from enum import Enum
from utils import bust, charge, clean, fix, mm, protonate, reduce, save, verify


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


# Groups whose pKa lies far enough below 7.4 that the SDF's neutral depiction is wrong
ACIDIC = {
    "carboxylic acid": Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1,OX1-]"), # pyright: ignore[reportAttributeAccessIssue]
    "phosphate": Chem.MolFromSmarts("[PX4](=[OX1])[OX2H1,OX1-]"), # pyright: ignore[reportAttributeAccessIssue]
    "sulfonate": Chem.MolFromSmarts("[SX4](=[OX1])(=[OX1])[OX2H1,OX1-]"), # pyright: ignore[reportAttributeAccessIssue]
}


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
    ACIDIC_LIGAND = "ligand carrying a group ionised at pH 7.4"
    INVALID_POSES = "no physically valid pose"


class OutOfScopeError(RuntimeError):
    """
    Reject protein because it is outside of scope.
    """

    def __init__(self, error_type: OutOfScopeErrorType, message: str | None = None):
        super().__init__(message or error_type.value)
        self.error_type = error_type


class PrepareError(RuntimeError):
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
        out=None,
        mm=True,
    ):
        self.protein_path = protein_path
        self.poses_paths = poses_paths
        self.whole = None
        self.reduced = None
        self.poses = None
        self.source = None
        self.protonation = None
        self.poses_protonation = None # TODO
        self.charge = None
        self.electrons = None
        self.heavy_atoms = None
        self.excluded = None
        self.failed = Counter()

        # What _verify records for _reverify
        self.deleted = []
        self.unoccupied = set()

        # Scope assumptions
        self.pH = 7.4
        self.cutoff = 4.5
        self.spin = 0

        # Eligibility thresholds
        self.basis = "6-31g"
        self.metal_coordination_cutoff = 2.8
        self.disulfide_cutoff = 2.5
        self.size_cap = 400

        # Seeds the minimisations that place rebuilt atoms and new hydrogens, so that both land in
        # the same place every run. Not zero: OpenMM reads a zero seed as a request for a random one.
        self.seed = 1

        # Whether each pose is relaxed in the protonated protein before it is screened. Off, every
        # pose reaches PoseBusters and then SAPT as DiffDock placed it, which is the ranking the
        # confidence model produced. The hydrogens _protonate gives a pose are RDKit's rather than
        # any mechanics, so a pose carries them either way.
        self.mm = mm

        # Load
        self.out = out
        if out:
            self._load()

    def prepare(self):
        self._fetch()
        self._verify()
        self._fix()
        self._clean()
        self._protonate()
        if self.mm:
            self._minimise()
        self._bust()
        self._reduce()
        self._reverify()
        self._calculate_charge()
        self._verify_num_electrons()
        self.save()

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
        self.source = [os.path.basename(path) for path in self.poses_paths]

    def _verify(self):
        """
        Take a provisional cutout and reject the complex according to the scope.

        Repairing the structure is deferred; `PDBFixer.findMissingResidues` may not work due to
            lack of SEQRES records, so chain breaks are not detected here.
        """
        if self.poses is None:
            raise PrepareError("Cannot verify the complex without poses")
        self.deleted = [
            (
                residue.name,
                np.array([atom.pos.x, atom.pos.y, atom.pos.z]),
                atom.element.is_metal,
            )
            for chain in self.whole
            for residue in chain
            for atom in residue
            if not atom.element.is_hydrogen
            and not _is_amino_acid(residue.name)
            and not _is_water(residue.name)
        ]
        self.unoccupied = {
            verify.identifier(chain, residue)
            for chain in self.whole
            for residue in chain
            for atom in residue
            if not atom.element.is_hydrogen and not atom.occ
        }

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

        for path, pose in zip(self.poses_paths, self.poses):
            for group, pattern in ACIDIC.items():
                if pose.HasSubstructMatch(pattern):
                    raise OutOfScopeError(
                        OutOfScopeErrorType.ACIDIC_LIGAND,
                        f"pose {path} carries a {group}, drawn neutral but ionised at pH {self.pH}",
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
        Protonates the entire protein, recording the state chosen for each residue, and assigns
            every pose explicit hydrogens .

        Modeller has no template for an arbitrary ligand, so use RDKit.
        """
        self.whole, self.protonation = protonate.hydrogens(self.whole, self.pH, self.seed)
        self.poses = mm.hydrogens(self.poses)

    def _minimise(self):
        """
        Relaxes every pose in the field of the protonated protein, protein fixed and ligand free.
        """
        self.poses = mm.minimise(self.whole, self.poses)

    def _bust(self):
        """
        Drops the poses PoseBusters finds physically implausible.

        _clean has deleted every heterogen, so a pose is only ever held against the polymer.
        """
        kept, self.failed = bust.valid(self.whole, self.poses)
        self.excluded = len(self.poses) - len(kept)
        if not kept:
            raise OutOfScopeError(
                OutOfScopeErrorType.INVALID_POSES,
                f"PoseBusters rejected all {self.excluded} pose(s)",
            )
        self.poses = [self.poses[index] for index in kept]
        self.source = [self.source[index] for index in kept]

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

        TODO (v2): Separate the cases with the SEQRES records

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

        # Recorded before the check, so a complex rejected here still carries the size that
        # rejected it.
        self.heavy_atoms = sum(
            1
            for chain in self.reduced
            for residue in chain
            for atom in residue
            if not atom.element.is_hydrogen
        )
        if self.heavy_atoms > self.size_cap:
            raise OutOfScopeError(
                OutOfScopeErrorType.SIZE_CAP,
                f"capped cutout holds {self.heavy_atoms} heavy atoms, over the cap of {self.size_cap}",
            )

    def _reverify(self):
        """
        The shell rules again, against the poses minimisation left.

        The size cap and repaired residues are _reduce's and are not repeated here.
        """
        if self.reduced is None:
            raise PrepareError("Cannot reverify before the protein is reduced")

        poses = cKDTree(self._pose_coordinates())
        shell = [
            (name, metal)
            for name, position, metal in self.deleted
            if poses.query_ball_point(position, self.cutoff, return_length=True)
        ]
        for name, metal in shell:
            if metal:
                raise OutOfScopeError(
                    OutOfScopeErrorType.METAL,
                    f"{name} within {self.cutoff} A of a minimised pose",
                )

        heterogens = {name for name, _ in shell} - ADDITIVES
        cofactors = heterogens & COFACTORS
        if cofactors:
            raise OutOfScopeError(
                OutOfScopeErrorType.COFACTOR,
                f"cofactor(s) {sorted(cofactors)} within {self.cutoff} A of a minimised pose",
            )
        if heterogens:
            raise OutOfScopeError(
                OutOfScopeErrorType.HETEROGEN,
                f"heterogen(s) {sorted(heterogens)} within {self.cutoff} A of a minimised pose",
            )

        retained = {
            (chain.name, residue.seqid.num, residue.seqid.icode)
            for chain in self.reduced
            for residue in chain
        }
        unoccupied = self.unoccupied & retained
        if unoccupied:
            raise OutOfScopeError(
                OutOfScopeErrorType.ZERO_OCCUPANCY,
                f"{len(unoccupied)} zero-occupancy residue(s) in the cutout, "
                f"e.g. {min(unoccupied)}",
            )

        metals = [position for _, position, metal in self.deleted if metal]
        if metals:
            coordinates = np.array([
                (atom.pos.x, atom.pos.y, atom.pos.z)
                for chain in self.reduced
                for residue in chain
                for atom in residue
                if not atom.element.is_hydrogen
            ])
            distances, _ = cKDTree(coordinates).query(metals)
            if distances.min() < self.metal_coordination_cutoff:
                raise OutOfScopeError(
                    OutOfScopeErrorType.SPLIT_METAL_COORDINATION,
                    f"metal {distances.min():.2f} A from a retained residue",
                )

        half = verify.split_disulfide(
            self.whole,
            [
                (chain, residue)
                for chain, residue, _, _ in verify.cutout(
                    self.whole, self._pose_coordinates(), self.cutoff
                )
            ],
            self.disulfide_cutoff,
        )
        if half:
            raise OutOfScopeError(
                OutOfScopeErrorType.SPLIT_DISULFIDE,
                f"cutout retains CYS {half} without its disulfide partner",
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

    def atoms(self):
        """
        Every atom of the cutout as (chain, residue, atom), in the order the cutout iterates.

        PySCF numbers its atoms in the order it is handed them and AVAS addresses target orbitals by
            that number, so this order is what ties an orbital back to the residue it came from. The
            atom alone does not carry that; the (chain, residue, atom) triple does.
        """
        if self.reduced is None:
            raise PrepareError("Cannot list the atoms before the protein is prepared")
        return [
            (chain, residue, atom)
            for chain in self.reduced
            for residue in chain
            for atom in residue
        ]


    def prepared(self):
        return self.electrons is not None and self.electrons % 2 == 0


    def _load(self):
        """
        Read back a preparation saved to `out`.
        """
        record = save.load_prepared(self._name(), self.out)
        if record is None:
            return
        self.reduced = gemmi.read_pdb_string(record["cutout"])[0] # pyright: ignore[reportAttributeAccessIssue]
        self.poses = [
            Chem.MolFromMolBlock(str(block), removeHs=False) # pyright: ignore[reportAttributeAccessIssue]
            for block in record["poses"]
        ]
        self.source = [str(name) for name in record["source"]]
        self.charge = record["charge"]
        self.electrons = record["electrons"]
        self.heavy_atoms = record["heavy_atoms"]
        self.excluded = record["excluded"]
        self.failed = Counter(record["failed"].tolist())


    def save(self):
        """
        The capped cutout, the poses post-protonation/minimisation/busting and the file each came
            from, and filter.py numbers.
        """
        if not self.out:
            return
        structure = gemmi.Structure() # pyright: ignore[reportAttributeAccessIssue]
        structure.add_model(self.reduced)
        structure.setup_entities()
        save.save_prepared(
            {
                "cutout": structure.make_pdb_string(),
                "poses": [Chem.MolToMolBlock(pose) for pose in self.poses], # pyright: ignore[reportAttributeAccessIssue]
                "source": self.source,
                "charge": self.charge,
                "electrons": self.electrons,
                "heavy_atoms": self.heavy_atoms,
                "excluded": self.excluded,
                # A Counter would need pickling. Its elements, each repeated, count back into one.
                "failed": list(self.failed.elements()),
            },
            self._name(),
            self.out,
        )


    def _name(self):
        """
        The complex, which names the directory its artefacts are kept in.
        """
        return os.path.basename(os.path.normpath(self.out))
