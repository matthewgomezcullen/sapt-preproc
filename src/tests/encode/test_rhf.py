"""
Solving RHF over a prepared cutout.

None of the bin can be solved here. One Fock build of the smallest, 7USH_82V at 1733 basis
    functions, is 227 s on twelve cores, so a converged SCF is one to two hours. What runs here is
    a full SCF over ACE-VAL-NME, one capped run lifted out of a prepared cutout.

The bin itself is at the bottom, marked hpc and skipped unless asked for.
"""

import numpy as np
import pytest

from conftest import paths
from cutouts import FRAGMENT, SUBSET, fragment, prepare, solved
from encode import EncodeProtein, EncodingError
from prepare import PrepareComplex, PrepareError

# The SCF is a stationary point of a smooth functional, so two runs of the same input agree far
# past this. It is loose only because the integrals are summed in whatever order the threads finish
# in, which moves the last digit or two.
REPRODUCIBLE = 1e-9


def test_rhf_refuses_a_complex_that_was_never_prepared():
    """
    There is no cutout to solve before the pipeline has run.
    """
    encoded = EncodeProtein(PrepareComplex(*paths(FRAGMENT)))

    with pytest.raises(PrepareError):
        encoded.RHF()


def test_rhf_converges_on_a_capped_fragment():
    """
    The whole path runs and reaches a converged closed-shell solution.
    """
    prepared = fragment()

    encoded = solved(prepared)

    assert encoded.mean_field.converged
    assert encoded.energy == pytest.approx(encoded.mean_field.e_tot)
    assert encoded.energy < 0
    assert set(np.unique(encoded.mean_field.mo_occ)) <= {0.0, 2.0}
    assert int(encoded.mean_field.mo_occ.sum()) == prepared.electrons


def test_rhf_puts_the_electrons_where_the_charge_says():
    """
    The converged density integrates to the electron count the preparation derived.
    """
    prepared = fragment()

    encoded = solved(prepared)

    density = encoded.mean_field.make_rdm1()
    overlap = encoded.mol.intor("int1e_ovlp")
    assert np.trace(density @ overlap) == pytest.approx(prepared.electrons, abs=1e-6)


def test_rhf_reaches_a_minimum_rather_than_a_saddle_point():
    """
    The converged solution is stable, so the orbitals AVAS is given are the ground state's.

    A converged SCF is only a stationary point. An internally unstable one has a lower solution it
        did not find.
    """
    encoded = solved(fragment())

    internal, _ = encoded.mean_field.stability()

    assert np.allclose(internal, encoded.mean_field.mo_coeff)


def test_rhf_is_reproducible():
    """
    The same cutout gives the same energy twice.
    """
    prepared = fragment()

    first = EncodeProtein(prepared)
    first.RHF()
    second = EncodeProtein(prepared)
    second.RHF()

    assert first.energy == pytest.approx(second.energy, abs=REPRODUCIBLE)
    assert np.allclose(first.mean_field.mo_energy, second.mean_field.mo_energy, atol=REPRODUCIBLE)


def test_rhf_rejects_an_scf_that_did_not_converge():
    """
    An unconverged SCF is not a result.
    """
    prepared = fragment()
    encoded = EncodeProtein(prepared)
    encoded.rhf_max_cycle = 1

    with pytest.raises(EncodingError):
        encoded.RHF()


# --------------------------------------------------------------------------------------------
# The bin itself. Hours each, for the cluster.
# --------------------------------------------------------------------------------------------


@pytest.mark.hpc
@pytest.mark.parametrize("name", SUBSET)
def test_rhf_converges_over_the_subset(name):
    """
    A converged, closed-shell SCF over a cutout of the bin.
    """
    prepared = prepare(name)

    encoded = solved(prepared)

    assert encoded.mean_field.converged
    assert set(np.unique(encoded.mean_field.mo_occ)) <= {0.0, 2.0}
    assert int(encoded.mean_field.mo_occ.sum()) == prepared.electrons


@pytest.mark.hpc_long_stab
@pytest.mark.parametrize("name", SUBSET)
def test_the_subset_reference_is_a_minimum_rather_than_a_saddle_point(name):
    """
    Ensure the converged solution is stable.

    Computationally very expensive, to split out behind its own flag.
    """
    encoded = solved(prepare(name))

    internal, _ = encoded.mean_field.stability()

    assert np.allclose(internal, encoded.mean_field.mo_coeff)


@pytest.mark.hpc
@pytest.mark.parametrize("name", SUBSET)
def test_subset_orbitals_leave_a_gap_to_correlate_across(name):
    """
    The converged solution has a real gap between the highest occupied and lowest virtual orbital.

    A truncated, charged cutout can close it, and a gap near zero means the closed-shell single
        determinant is the wrong starting point.
    """
    # The solve this shares with the test above is the whole of the job's cost.
    encoded = solved(prepare(name))

    energies, occupations = encoded.mean_field.mo_energy, encoded.mean_field.mo_occ
    gap = energies[occupations == 0].min() - energies[occupations > 0].max()
    assert gap > 0
