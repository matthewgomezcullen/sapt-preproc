"""
The harmonic restraint a tethered minimisation holds each pose heavy atom with, for utils.mm._tether.

Held against the analytic well rather than against a minimised pose. `mm.minimise` builds its
    context on OpenMM's fastest platform, and the CPU one sums forces in whatever order its threads
    finish in, so two runs over the same input do not land in the same place and nothing comparing
    them can be exact. These are exact, and take milliseconds. What a tether does to a real pose in
    a real pocket is test_mm.py's, behind --prepare-long.
"""

import numpy as np
import pytest
from openmm import Context, LocalEnergyMinimizer, System, VerletIntegrator, unit

from utils.mm import ANGSTROM, _tether

# kcal/mol/A^2, and a displacement in angstroms to hold the particle at.
STRENGTH = 10.0
AWAY = 1.0

# Nothing integrates here; the context needs an integrator to exist.
STEP = 0.001

MASS = 12.0

# A minimised position in angstroms lands this close to its reference.
EXACT = 1e-6

# The CPU platform computes in single precision, so an energy is only exact relative to its size.
PRECISION = 1e-6


def tethered(strength, at, reference=(0.0, 0.0, 0.0)):
    """
    One particle tethered at `strength` to `reference`, placed at `at`. Both in angstroms.
    """
    system = System()
    system.addParticle(MASS)
    restraint = _tether([0], 0, strength)
    system.addForce(restraint)
    context = Context(system, VerletIntegrator(STEP))
    reference_to(restraint, context, reference)
    context.setPositions(unit.Quantity(np.array([at]) * ANGSTROM, unit.nanometer))
    return context, restraint


def reference_to(restraint, context, reference):
    restraint.setParticleParameters(0, 0, np.array(reference) * ANGSTROM)
    restraint.updateParametersInContext(context)


def energy(context):
    return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilocalories_per_mole
    )


def position(context):
    return context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(
        unit.angstrom
    )[0]


def test_the_tether_is_a_harmonic_well_of_the_strength_it_was_given():
    """
    Against 0.5*k*d^2 in the units the strength is quoted in, which is what says OpenMM was handed
        kcal/mol/A^2 and not the kJ/mol/nm^2 it works in.
    """
    for away in (AWAY, 2 * AWAY):
        context, _ = tethered(STRENGTH, (away, 0.0, 0.0))

        assert energy(context) == pytest.approx(0.5 * STRENGTH * away**2, rel=PRECISION)


def test_the_tether_pulls_an_atom_onto_its_reference():
    context, _ = tethered(STRENGTH, (AWAY, 0.0, 0.0))

    LocalEnergyMinimizer.minimize(context)

    assert position(context) == pytest.approx([0.0, 0.0, 0.0], abs=EXACT)


def test_a_tether_of_no_strength_does_not_pull():
    context, _ = tethered(0.0, (AWAY, 0.0, 0.0))
    assert energy(context) == pytest.approx(0.0, abs=EXACT)

    LocalEnergyMinimizer.minimize(context)

    assert position(context) == pytest.approx([AWAY, 0.0, 0.0], abs=EXACT)


def test_the_reference_follows_the_parameters_it_is_reset_with():
    context, restraint = tethered(STRENGTH, (AWAY, 0.0, 0.0))
    LocalEnergyMinimizer.minimize(context)

    reference_to(restraint, context, (AWAY, 0.0, 0.0))
    LocalEnergyMinimizer.minimize(context)

    assert position(context) == pytest.approx([AWAY, 0.0, 0.0], abs=EXACT)


def test_the_tether_holds_the_atoms_it_was_given_where_the_ligand_sits():
    restraint = _tether([0, 2, 5], 100, STRENGTH)

    assert restraint.getNumParticles() == 3
    held = [restraint.getParticleParameters(index)[0] for index in range(3)]
    assert held == [100, 102, 105]
