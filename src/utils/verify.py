import numpy as np
from pdbfixer import PDBFixer
from scipy.spatial import cKDTree # pyright: ignore[reportAttributeAccessIssue]

def identifier(chain, residue):
    """
    A residue's full PDB identity.
    """
    return (chain.name, residue.seqid.num, residue.seqid.icode)


def cutout(protein, pose_coordinates, cutoff):
    """
    Every residue holding at least one heavy atom within `cutoff` of a heavy atom of any 
        candidate pose. Returns (chain, residue, heavy atoms, coordinates).
    """
    poses = cKDTree(pose_coordinates)
    retained = []
    for chain in protein:
        for residue in chain:
            heavy = [atom for atom in residue if not atom.element.is_hydrogen]
            if not heavy:
                continue
            coordinates = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in heavy])
            if poses.query_ball_point(coordinates, cutoff, return_length=True).any():
                retained.append((chain, residue, heavy, coordinates))
    return retained

def metals(protein):
    """
    Every metal atom in the whole structure. A metal outside the cutout can still coordinate a 
        retained residue.
    """
    return [
        (atom, np.array([atom.pos.x, atom.pos.y, atom.pos.z]))
        for chain in protein
        for residue in chain
        for atom in residue
        if atom.element.is_metal
    ]

def incomplete_residues(protein_path):
    """
    Residues PDBFixer reports as missing heavy atoms.

    Missing terminal atoms are ignored; the cutout is capped with ACE/NME regardless.

    TODO: `fixer.findMissingResidues` does not work without SEQRES records. Fix.
    """
    fixer = PDBFixer(filename=protein_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    return {
        (residue.chain.id, int(residue.id), residue.insertionCode)
        for residue in fixer.missingAtoms
    }

def split_disulfide(protein, residues, cutoff):
    """
    Return the retained half of a disulfide whose partner falls outside the cutout, if any.

    Bonding is taken from SG-SG distance rather than SSBOND records, which the PoseBusters
        structures do not reliably provide.
    """
    sulfurs = []
    for chain in protein:
        for residue in chain:
            if residue.name != "CYS":
                continue
            for atom in residue:
                if atom.name == "SG":
                    sulfurs.append((
                        identifier(chain, residue),
                        np.array([atom.pos.x, atom.pos.y, atom.pos.z]),
                    ))
    if not sulfurs:
        return None

    kept = {identifier(chain, residue) for chain, residue in residues}
    for half, position in sulfurs:
        if half not in kept:
            continue
        for other, other_position in sulfurs:
            if other == half or other in kept:
                continue
            if np.linalg.norm(position - other_position) < cutoff:
                return half
    return None


