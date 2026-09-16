import io
import warnings

import gemmi
import numpy as np
from openmm import CustomExternalForce, LocalEnergyMinimizer, VerletIntegrator, Context, unit
from openmm.app import Modeller, NoCutoff, PDBFile
from rdkit import Chem


FORCEFIELD = "amber14-all.xml"

# Sage 2.0.0, as the PoseBusters paper minimised with.
SMALL_MOLECULE = "openff-2.0.0"

MAX_ITERATIONS = 1000

STEP = 0.001

ANGSTROM = 0.1

# A tether is quoted in kcal/mol/A^2, as the original paper's tethered minimisations were. OpenMM
# works in kJ/mol and nm and converts the strength itself.
TETHER = unit.kilocalories_per_mole / unit.angstrom**2

# There is no box and no solvent, so nothing is periodic and nothing is cut off. Constraints are
# off because the protein is frozen by zero mass, and OpenMM will not constrain a massless atom;
# they would also stop the hydrogens relaxing.
FORCEFIELD_KWARGS = {
    "constraints": None,
    "rigidWater": False,
    "removeCMMotion": False,
}

NONPERIODIC_KWARGS = {"nonbondedMethod": NoCutoff}


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


def _get_heavy_idxs(pose):
    return [atom.GetIdx() for atom in pose.GetAtoms() if atom.GetAtomicNum() > 1]


def _tether(heavy, offset, strength):
    """
    A restraint holding each heavy atom of a pose to wherever the pose is handed in.

    The hydrogens are left out. Their coordinates should be untethered.

    The reference is a per-particle parameter, so one force serves the whole ensemble and is reset 
        onto each pose in turn. `offset` is where the ligand's atoms start in the combined system.
    """
    force = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", strength * TETHER)
    for reference in ("x0", "y0", "z0"):
        force.addPerParticleParameter(reference)
    for index in heavy:
        force.addParticle(offset + index, [0.0, 0.0, 0.0])
    return force


def _move(pose, coordinates):
    from rdkit.Geometry import Point3D

    moved = Chem.Mol(pose) # pyright: ignore[reportAttributeAccessIssue]
    conformer = moved.GetConformer()
    for index, (x, y, z) in enumerate(coordinates):
        conformer.SetAtomPosition(index, Point3D(x, y, z))
    return moved


def minimise(model, poses, tether=None):
    """
    Every pose relaxed in the field of the protein, with the protein fixed and the ligand free.

    `tether` is the strength, in kcal/mol/A^2, of a restraint holding each pose heavy atom to 
        where it was handed in, so the placement DiffDock produced is kept while the hydrogens and 
        the clash relief stay free. None is the free minimisation, and a strength of zero is the 
        same relaxation through a force that cannot pull.
    """
    from openff.interchange.warnings import PresetChargesAndVirtualSitesWarning
    from openmmforcefields.generators import SystemGenerator

    molecule = _parameterise(poses[0])
    protein = _protein(model)
    modeller = Modeller(protein.topology, protein.positions)
    # Minimise an isolated pocket.
    modeller.topology.setPeriodicBoxVectors(None)
    fixed = modeller.topology.getNumAtoms()
    modeller.add(
        molecule.to_topology().to_openmm(),
        unit.Quantity(_coordinates(poses[0]), unit.nanometer),
    )

    generator = SystemGenerator(
        forcefields=[FORCEFIELD],
        small_molecule_forcefield=SMALL_MOLECULE,
        molecules=[molecule],
        forcefield_kwargs=FORCEFIELD_KWARGS,
        nonperiodic_forcefield_kwargs=NONPERIODIC_KWARGS,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PresetChargesAndVirtualSitesWarning)
        warnings.filterwarnings("ignore", r"`torch\.distributed\.reduce_op`", FutureWarning)
        system = generator.create_system(modeller.topology, molecules=[molecule])
    # A zero mass is how OpenMM is told an atom does not move.
    for index in range(fixed):
        system.setParticleMass(index, 0)

    # Added before the context, which is what reads the system's forces.
    heavy = _get_heavy_idxs(poses[0])
    restraint = None if tether is None else _tether(heavy, fixed, tether)
    if restraint is not None:
        system.addForce(restraint)

    # OpenMM takes its fastest platform, on every core the machine has. OPENMM_DEFAULT_PLATFORM and
    # OPENMM_CPU_THREADS override it.
    context = Context(system, VerletIntegrator(STEP))
    protein_coordinates = np.asarray(
        modeller.positions.value_in_unit(unit.nanometer)
    )[:fixed]

    minimised = []
    for pose in poses:
        coordinates = _coordinates(pose)
        if restraint is not None:
            for particle, index in enumerate(heavy):
                restraint.setParticleParameters(particle, fixed + index, coordinates[index])
            restraint.updateParametersInContext(context)
        context.setPositions(
            unit.Quantity(np.vstack([protein_coordinates, coordinates]), unit.nanometer)
        )
        LocalEnergyMinimizer.minimize(context, maxIterations=MAX_ITERATIONS)
        relaxed = context.getState(getPositions=True).getPositions(
            asNumpy=True
        ).value_in_unit(unit.angstrom)
        minimised.append(_move(pose, relaxed[fixed:]))
    return minimised
