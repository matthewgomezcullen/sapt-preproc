from encode import EncodeProtein, EncodingError, SolveLigand
from utils import sapt


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
    ):
        self.protein = protein
        self.ligand = ligand
        self.density = None # the protein's AO density, core plus active
        self.electrostatics = None # E^(1)_elst per pose


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
