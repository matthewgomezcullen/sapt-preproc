class EncodeComplex:
    # EncodeComplex takes a holo-protein structure (.pdb) and candidaet poses (.sdf) and encodes a 
    #   reduced protein structure with poses for SAPT(VQE). Aims to generalise the SAPT(VQE) method 
    #   to any protein-ligand complex.

    def __init__(self, protein_path: str, poses_paths: list[str]):
        self.protein_path = protein_path    # A
        self.poses_paths = poses_paths      # {B}
        self.reduced = None                 # A^{\cup'}
        self.protonation = None             # {pK_A : state}
        self.charge = None                  # q_{A^{\cup'}}
        self.spin = None                    # N_e
        self.multiplicity = 1               # M_{A^{\cup'}}

    def reduce(self):
        # Takes union of complete residues with at least one atom within 4.5 Å of the nearest 
        #   pose atom, then caps the truncated protein with ACE/NME.
        # *Possibly covered by SparcleQC. Likely requires proprietary code with MDAnalysis.
        ...

    def protonate(self):
        # Protonate thefull receptor once with PROPKA. The result maps sites (pK_a) to their 
        # protonation state.
        ...

    def calculate_charge(self):
        # Calculates total charge. Requires protonation.
        ...

    def calculate_spin(self):
        # Calculates total spin. N_e = \sum_I Z_I - q_A
        ...
    
    def xyz(self, path: str):
        # Store element and coordinates in a .xyz file
        ...

class ActiveSpaceSelection:
    # ActiveSpaceSelection takes an encoded holo-protein structure and reduces it to a tractable 
    #   active space for VQE/CASCI experiments.
    
    def __init__(self, encoding: EncodeComplex, name: str, basis: str="6-31g"):
        self.encoding = encoding
        self.encoding.xyz(f"{name}.xyz")
        self.basis = basis
        self.molecular_orbits = None
        self.hamiltonian = None             # H^{act}_A
        
    def RHF(self):
        # Solve Hartree-Fock equations with PySCF, populating the molecular orbits.
        ...

    def AVAS(self):
        # Run Atomic Valence Active Space over the MOs with PySCF.
        ...

    def SHCI(self, lo=0.02, hi=1.98):
        # Run Semistochastic Heat-Bath Configuration Interaction with Dice through PySCF, then keep 
        #   only orbitals satisfying lo \le n_i \le hi.
        ...

    def H(self):
        # Maps fermionic electronic-structure Hamiltonian into a qubit Hamiltonian following the 
        #   Jordan-Wigner transformation.
        ...
