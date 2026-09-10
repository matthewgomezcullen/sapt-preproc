import io

import gemmi
import numpy as np
from openmm import LocalEnergyMinimizer, VerletIntegrator, Context, unit
from openmm.app import Modeller, PDBFile
from rdkit import Chem


FORCEFIELD = "amber14-all.xml"

# Sage 2.0.0, as the PoseBusters paper minimised with.
SMALL_MOLECULE = "openff-2.0.0"

# The paper's convergence criterion, in kJ/mol.
TOLERANCE = 0.01

STEP = 0.001

ANGSTROM = 0.1


def hydrogens(poses):
    return [
        Chem.AddHs(pose, addCoords=True) # pyright: ignore[reportAttributeAccessIssue]
        for pose in poses
    ]


def _parameterise(pose):
    """
    The ligand's force-field template. AM1-BCC is the expensive part, so it is run once and shared.
    """
    from openff.toolkit.topology import Molecule

    molecule = Molecule.from_rdkit(pose, allow_undefined_stereo=True)
    molecule.assign_partial_charges("am1bcc")
    return molecule


def _protein(model):
    structure = gemmi.Structure() # pyright: ignore[reportAttributeAccessIssue]
    structure.add_model(model)
    structure.setup_entities()
    return PDBFile(io.StringIO(structure.make_pdb_string()))


def _coordinates(pose):
    return pose.GetConformer().GetPositions() * ANGSTROM


def _move(pose, coordinates):
    from rdkit.Geometry import Point3D

    moved = Chem.Mol(pose) # pyright: ignore[reportAttributeAccessIssue]
    conformer = moved.GetConformer()
    for index, (x, y, z) in enumerate(coordinates):
        conformer.SetAtomPosition(index, Point3D(x, y, z))
    return moved


def minimise(model, poses):
    """
    Every pose relaxed in the field of the protein, with the protein fixed and the ligand free.
    """
    from openmmforcefields.generators import SystemGenerator

    molecule = _parameterise(poses[0])
    protein = _protein(model)
    modeller = Modeller(protein.topology, protein.positions)
    fixed = modeller.topology.getNumAtoms()
    modeller.add(
        molecule.to_topology().to_openmm(),
        unit.Quantity(_coordinates(poses[0]), unit.nanometer),
    )

    generator = SystemGenerator(
        forcefields=[FORCEFIELD],
        small_molecule_forcefield=SMALL_MOLECULE,
        molecules=[molecule],
    )
    system = generator.create_system(modeller.topology, molecules=[molecule])
    # A zero mass is how OpenMM is told an atom does not move.
    for index in range(fixed):
        system.setParticleMass(index, 0)

    context = Context(system, VerletIntegrator(STEP))
    protein_coordinates = np.asarray(
        modeller.positions.value_in_unit(unit.nanometer)
    )[:fixed]

    minimised = []
    for pose in poses:
        context.setPositions(
            unit.Quantity(
                np.vstack([protein_coordinates, _coordinates(pose)]), unit.nanometer
            )
        )
        LocalEnergyMinimizer.minimize(
            context, TOLERANCE * unit.kilojoule_per_mole
        )
        coordinates = context.getState(getPositions=True).getPositions(
            asNumpy=True
        ).value_in_unit(unit.angstrom)
        minimised.append(_move(pose, coordinates[fixed:]))
    return minimised
