import io

import gemmi
import openmm
from openmm.app import PDBFile
from pdbfixer import PDBFixer


# The CPU platform sums forces in whatever order its lanes and threads finish in, so a minimisation
# run twice on the same input lands in different places. Reference is single-threaded and repeats 
# exactly.
PLATFORM = openmm.Platform.getPlatformByName("Reference")


def repair(protein_path, seed):
    """
    Rebuild missing heavy atoms, missing terminal atoms, and missing residues with PDBFixer, and
        hand the result back as a gemmi structure.

    Non-standard residues are replaced by the standard ones they were made from.

    `addMissingAtoms` places a rebuilt atom by minimising it, and falls back to Langevin dynamics
        when the minimum leaves atoms on top of each other. This is nondeterministic unless seeded 
        and using a Reference platform.

    TODO: `fixer.findMissingResidues` does not work without SEQRES records, so chain breaks are
        detected as neither missing nor repaired. Fix.
    """
    fixer = PDBFixer(filename=protein_path, platform=PLATFORM)
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=seed)

    # keepIds preserves chain names and sequence numbers, which the cutout keys residues by.
    buffer = io.StringIO()
    PDBFile.writeFile(fixer.topology, fixer.positions, buffer, keepIds=True)
    return gemmi.read_pdb_string(buffer.getvalue()) # pyright: ignore[reportAttributeAccessIssue]
