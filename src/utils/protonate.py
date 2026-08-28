import io

import gemmi
from openmm.app import Modeller, PDBFile


def hydrogens(model, pH):
    """
    Add hydrogens to every residue at `pH`, returning the protonated model and the protonation state
        Modeller chose for each residue.

    Modeller is called directly rather than through `PDBFixer.addMissingHydrogens`, which wraps it
        and discards its return value. `_calculate_charge` reads the states, not the names.

    Going direct also drops PDBFixer's Chemical Component Dictionary lookup, which reaches out to
        the network for any residue Modeller does not know.
    """
    structure = gemmi.Structure() # pyright: ignore[reportAttributeAccessIssue]
    structure.add_model(model)
    structure.setup_entities()
    pdb = PDBFile(io.StringIO(structure.make_pdb_string()))

    # addHydrogens replaces the modeller's topology, so residues() is read off the original.
    residues = list(pdb.topology.residues())
    modeller = Modeller(pdb.topology, pdb.positions)
    variants = modeller.addHydrogens(pH=pH)
    states = {
        (residue.chain.id, int(residue.id), residue.insertionCode): variant
        for residue, variant in zip(residues, variants)
    }

    # keepIds preserves chain names and sequence numbers, which the cutout keys residues by.
    buffer = io.StringIO()
    PDBFile.writeFile(modeller.topology, modeller.positions, buffer, keepIds=True)
    return gemmi.read_pdb_string(buffer.getvalue())[0], states # pyright: ignore[reportAttributeAccessIssue]
