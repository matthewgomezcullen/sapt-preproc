from prepare import PrepareComplex
from scipy.spatial import cKDTree # pyright: ignore[reportAttributeAccessIssue]

from pyscf import gto, scf

def molecule(prepared: PrepareComplex, verbose):
    """
    Build the PySCF molecule the SCF is solved over.
    """
    mol = gto.M(
        atom=[
            (atom.element.name, (atom.pos.x, atom.pos.y, atom.pos.z))
            for _, _, atom in prepared.atoms()
        ],
        charge=prepared.charge,
        spin=prepared.spin,
        basis=prepared.basis,
        verbose=verbose,
    )
    return mol

def rhf(mol, max_cycle):
    """
    Run RHF with PySCF.
    """
    mean_field = scf.RHF(mol)
    mean_field.max_cycle = max_cycle
    mean_field.kernel()
    return mean_field

def generate_target_orbitals(prepared, cutoff, exclude, valence):
    """
    Generate AVAS target orbitals based on which atoms are chemically relevant.
    """
    atoms = prepared.atoms()
    poses = cKDTree(prepared._pose_coordinates())
    distances, _ = poses.query(
        [(atom.pos.x, atom.pos.y, atom.pos.z) for _, _, atom in atoms]
    )

    return [
        f"{index} {atom.element.name} {valence[atom.element.name]}"
        for index, ((_, residue, atom), distance) in enumerate(zip(atoms, distances))
        if residue.name not in exclude
        and atom.element.name in valence
        and distance <= cutoff
    ]

