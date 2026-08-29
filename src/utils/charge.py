import io

import gemmi
import openmm
from openmm.app import ForceField, PDBFile
from openmm.unit import elementary_charge


# ff14SB and the caps, ions and water that ship with it. Only the protein templates are reached here.
FORCEFIELD = ForceField("amber14-all.xml")


def net(model):
    """
    The net formal charge of a model, read off the Amber templates its residues match.

    The partial charges of a residue sum to its formal charge, so the sum over the system is q_A.
        Templates are matched on the bond graph, rather than a residue name or a pH rule.

    Every residue, cap and terminus has to correspond to a real Amber residue for `createSystem` to 
        return at all.

    The sum is rounded because it accumulates floating-point error over hundreds of atoms.
    """
    structure = gemmi.Structure() # pyright: ignore[reportAttributeAccessIssue]
    structure.add_model(model)
    structure.setup_entities()
    pdb = PDBFile(io.StringIO(structure.make_pdb_string()))

    system = FORCEFIELD.createSystem(pdb.topology)
    nonbonded = next(
        force for force in system.getForces() if isinstance(force, openmm.NonbondedForce)
    )
    return round(sum(
        nonbonded.getParticleParameters(index)[0].value_in_unit(elementary_charge)
        for index in range(system.getNumParticles())
    ))
