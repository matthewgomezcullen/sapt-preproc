import io

import gemmi
from openmm.app import PDBFile
from pdbfixer import PDBFixer


def repair(protein_path):
    """
    Rebuild missing heavy atoms, missing terminal atoms, and missing residues with PDBFixer, and
        hand the result back as a gemmi structure.

    Hydrogens are not added here; protonation is a separate step with its own pH.

    TODO: `fixer.findMissingResidues` does not work without SEQRES records, so chain breaks are
        detected as neither missing nor repaired. Fix.
    """
    fixer = PDBFixer(filename=protein_path)
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # keepIds preserves chain names and sequence numbers, which the cutout keys residues by.
    buffer = io.StringIO()
    PDBFile.writeFile(fixer.topology, fixer.positions, buffer, keepIds=True)
    return gemmi.read_pdb_string(buffer.getvalue()) # pyright: ignore[reportAttributeAccessIssue]
