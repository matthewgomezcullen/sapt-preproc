import io
import random

import gemmi
import openmm
from openmm.app import Modeller, PDBFile

from utils.fix import PLATFORM


def add(modeller, pH, seed, variants=None):
    """
    `Modeller.addHydrogens`, made reproducible, returning the state chosen for each residue.

    Modeller starts every hydrogen it adds at a random offset from its parent and minimises from
        there, drawing on the global random module. Left unseeded it places the hydrogens
        differently on every run.

    The generator's state is put back afterwards.
    """
    state = random.getstate()
    random.seed(seed)
    try:
        return modeller.addHydrogens(pH=pH, variants=variants, platform=PLATFORM)
    finally:
        random.setstate(state)


def hydrogens(model, pH, seed):
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
    variants = add(modeller, pH, seed)
    states = {
        (residue.chain.id, int(residue.id), residue.insertionCode): variant
        for residue, variant in zip(residues, variants)
    }

    # keepIds preserves chain names and sequence numbers, which the cutout keys residues by.
    buffer = io.StringIO()
    PDBFile.writeFile(modeller.topology, modeller.positions, buffer, keepIds=True)
    return gemmi.read_pdb_string(buffer.getvalue())[0], states # pyright: ignore[reportAttributeAccessIssue]
