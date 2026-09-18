import confidence
import filter
from encode import EncodeProtein, EncodingError, SolveLigand
from utils import sapt

# confidence.csv's columns, and what SAPT adds to each pose
FIELDS = confidence.FIELDS + ["elst", "exch", "cumulant", "interaction", "rank_sapt"]
SUMMARY_FIELDS = confidence.SUMMARY_FIELDS + ["top1_sapt"]


class SAPT:
    """
    SAPT takes a correlated EncodeProtein and a solved SolveLigand of the same complex, and scores
        each pose against the protein at first order.

    The protein is monomer A, the same for every pose, so its density is built once. Each pose is
        monomer B, at RHF in its own basis.
    """

    def __init__(
        self,
        protein: EncodeProtein,
        ligand: SolveLigand,
        out=None
    ):
        self.protein = protein
        self.ligand = ligand
        self.density = None # the protein's AO density, core plus active
        self.electrostatics = None # E^(1)_elst per pose
        self.exchanges = None # E^(1)_exch per pose
        self.cumulants = None # the share of E^(1)_exch the protein's cumulant carries, per pose
        self.int_energies = None # E_int per pose

        # Load
        self.out = out
        if out:
            self._load()


    def densities(self):
        """
        The protein's AO density, from the orbitals the window was cut over and CASCI's rdm1 over
            its active block.
        """
        if not self.protein.correlated():
            raise EncodingError("Cannot build the protein's density before CASCI has solved it")
        self.density = sapt.density(
            self.protein.orbital_initial,
            self.protein.rdm1,
            self.protein.active_space_size,
            self.protein.active_electrons,
            self.protein.mol.nelectron,
        )
        return self.density


    def elst(self):
        """
        E^(1)_elst of every pose against the protein.
        """
        if not self.ligand.solved():
            raise EncodingError("Cannot score the poses before RHF has solved every one")
        if self.density is None:
            self.densities()
        self.electrostatics = [
            sapt.electrostatics(self.protein.mol, self.density, mol, mean_field.make_rdm1())
            for mol, mean_field in zip(self.ligand.mols, self.ligand.mean_fields)
        ]
        return self.electrostatics


    def exch(self):
        """
        E^(1)_exch(S^2) of every pose against the protein: the exchange of the two densities, and
            what the protein's cumulant adds, which is kept apart too.
        """
        if not self.ligand.solved():
            raise EncodingError("Cannot score the poses before RHF has solved every one")
        if self.density is None:
            self.densities()
        active = sapt.active(
            self.protein.orbital_initial,
            self.protein.active_space_size,
            self.protein.active_electrons,
            self.protein.mol.nelectron,
        )
        cumulant = sapt.cumulant(self.protein.rdm1, self.protein.rdm2)
        self.exchanges, self.cumulants = [], []
        for mol, mean_field in zip(self.ligand.mols, self.ligand.mean_fields):
            density = mean_field.make_rdm1()
            share = sapt.cumulant_exchange(self.protein.mol, active, cumulant, mol, density)
            self.exchanges.append(
                sapt.exchange(self.protein.mol, self.density, mol, density) + share
            )
            self.cumulants.append(share)
        return self.exchanges


    def interaction(self):
        """
        E^(1)_int = E^(1)_elst + E^(1)_exch(S^2) of every pose against the protein.
        """
        if self.electrostatics is None:
            self.elst()
        if self.exchanges is None:
            self.exch()
        self.int_energies = [
            electrostatic + exchange
            for electrostatic, exchange in zip(self.electrostatics, self.exchanges)
        ]
        return self.int_energies



    def save(self):
        pass


    def _load(self):
        pass


def join(rows, energies):
    """
    Each of confidence.csv's rows given its pose's energies, keyed by (complex, source) 
    
    A pose without energies is dropped and reported.
    """
    joined, missing = [], []
    for row in rows:
        pose = (row["name"], row["source"])
        if pose not in energies:
            missing.append(pose)
            continue
        scored = energies[pose]
        joined.append({
            **row,
            "elst": scored["elst"],
            "exch": scored["exch"],
            "cumulant": scored["cumulant"],
            "interaction": scored["elst"] + scored["exch"],
        })
    return joined, missing


def rank(rows):
    return confidence.number(
        rows, lambda row: (row["interaction"], row["rank_docked"]), "rank_sapt"
    )


def summarise(rows, threshold=filter.NEAR_NATIVE):
    summary = confidence.summarise(rows, threshold)
    for entry in summary:
        theirs = [row for row in rows if row["name"] == entry["name"]]
        entry["top1_sapt"] = confidence.is_near_native(
            min(theirs, key=lambda row: row["rank_sapt"]), threshold
        )
    return summary
