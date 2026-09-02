"""
Truncating the active space with selected CI natural occupations.

MP2 hands SHCI a space of nmax orbitals chosen for their MP2 fractionality. SHCI solves it, takes
    the natural occupations of its own one-particle density, and keeps only the orbitals inside the
    window lo <= n_i <= hi, which a single determinant fails to describe. Those above the window 
    are doubly occupied and join the core; those below are empty and join the virtuals.

The window fixes no size, and on a space with no static correlation it can keep nothing at all.
    Degenerate solutions are rejected.

Every fidelity test is against exact diagonalisation of the same space. The fragment is capped to 
    eight orbitals, small enough that the exact answer is close. The bin is marked hpc.
"""

import functools

import numpy as np
import pytest
from pyscf import mcscf

from cutouts import SUBSET, all_carbons, contact_weight, fragment, prepare, solved, window
from encode import EncodeProtein, EncodingError

# The MP2 cap the fragment is carried through. Eight orbitals is 4900 determinants, so the space
# SHCI is asked to select within can also be diagonalised exactly.
CAP = 8

# The original paper's window, and the default in the signature.
PAPER = (0.02, 1.97)

# Windows cut against the fragment's own exact spectrum, which runs
#   1.9933 1.9906 1.9818 1.9761 | 0.0240 0.0183 0.0093 0.0066.
NARROW, NARROWED = (0.015, 1.985), (4, 4)  # (lo, hi), (nelecas, ncas)
WIDE, WIDENED = (0.008, 1.992), (6, 6)

# A window that keeps everything.
EVERYTHING = (0.0, 2.0)

# The signature's selection cutoff.
SELECTION = 1e-4
COARSE = 1e-3

# Agreement with exact diagonalisation of the same space, at the default cutoff.
ENERGY = 1e-5  # absolute, Hartree
OCCUPATION = 1e-5  # absolute, on a natural occupation

# Size constraint for VQE. Currently, abitrary budget rather than derived.
SIMULABLE_ORBITALS = 16


@functools.lru_cache(maxsize=None)
def _capped():
    """
    The fragment through RHF, AVAS over every carbon, and the MP2 cap.
    """
    encoded = EncodeProtein(fragment())
    encoded.RHF()
    encoded.AVAS(targets=all_carbons(encoded.mol))
    encoded.nmax = CAP
    encoded.MP2()
    return encoded


def capped():
    """
    A fresh encoder holding that cached space, so a test can truncate it without disturbing the
        next one.
    """
    cached = _capped()
    encoded = EncodeProtein(cached.prepared)
    encoded.mol = cached.mol
    encoded.mean_field = cached.mean_field
    encoded.energy = cached.energy
    encoded.active_space_size, encoded.active_electrons = cached.active_space_size, cached.active_electrons
    encoded.orbital_initial = cached.orbital_initial.copy()
    encoded.occupations = cached.occupations.copy()
    return encoded


@functools.lru_cache(maxsize=None)
def reference():
    """
    Exact diagonalisation of the capped space, which at eight orbitals is affordable.
    """
    encoded = _capped()
    exact = mcscf.CASCI(encoded.mean_field, encoded.active_space_size, encoded.active_electrons)
    exact.verbose = 0
    exact.kernel(encoded.orbital_initial)
    density = exact.fcisolver.make_rdm1(exact.ci, exact.ncas, exact.nelecas)
    return exact.e_tot, np.sort(np.linalg.eigvalsh(density))[::-1]


@pytest.mark.dice
def test_shci_reproduces_the_exact_solution_of_the_space_it_is_given():
    """
    SHCI recovers the energy and the density full CI would have given. Run with a window 
        that keeps everything, so what is compared is the solve alone.
    """
    energy, occupations = reference()
    encoded = capped()

    encoded.SHCI(eps1=SELECTION, lo=EVERYTHING[0], hi=EVERYTHING[1])

    assert encoded.shci_energy == pytest.approx(energy, abs=ENERGY)
    assert np.allclose(encoded.occupations, occupations, atol=OCCUPATION)


@pytest.mark.dice
def test_shci_lowers_the_energy_the_mean_field_settled_on():
    """
    Selected CI is variational, so it sits above exact and below Hartree-Fock.
    """
    energy, _ = reference()
    encoded = capped()

    encoded.SHCI(eps1=SELECTION, lo=EVERYTHING[0], hi=EVERYTHING[1])

    assert encoded.shci_energy < encoded.energy
    assert encoded.shci_energy >= energy - ENERGY


@pytest.mark.dice
def test_a_tighter_selection_cutoff_is_a_closer_answer():
    """
    Smaller `eps1` increases accuracy.
    """
    energy, occupations = reference()

    coarse, fine = capped(), capped()
    coarse.SHCI(eps1=COARSE, lo=EVERYTHING[0], hi=EVERYTHING[1])
    fine.SHCI(eps1=SELECTION, lo=EVERYTHING[0], hi=EVERYTHING[1])

    assert coarse.shci_energy >= fine.shci_energy >= energy - ENERGY
    assert abs(fine.shci_energy - energy) < abs(coarse.shci_energy - energy)
    assert np.abs(fine.occupations - occupations).max() < np.abs(
        coarse.occupations - occupations
    ).max()


@pytest.mark.dice
def test_shci_is_reproducible():
    """
    The same capped space gives the same truncation twice.
    """
    first, second = capped(), capped()

    first.SHCI(eps1=SELECTION, lo=NARROW[0], hi=NARROW[1])
    second.SHCI(eps1=SELECTION, lo=NARROW[0], hi=NARROW[1])

    assert (first.active_space_size, first.active_electrons) == (second.active_space_size, second.active_electrons)
    assert first.shci_energy == pytest.approx(second.shci_energy)
    assert np.allclose(first.occupations, second.occupations)
    assert np.allclose(first.orbital_initial, second.orbital_initial)


@pytest.mark.dice
def test_shci_keeps_exactly_the_occupations_inside_the_window():
    """
    The exact spectrum the window admits, in descending order.
    """
    _, occupations = reference()
    lo, hi = NARROW
    encoded = capped()

    encoded.SHCI(eps1=SELECTION, lo=lo, hi=hi)

    expected = occupations[(occupations >= lo) & (occupations <= hi)]
    assert encoded.active_space_size == len(expected)
    assert np.allclose(encoded.occupations, expected, atol=OCCUPATION)
    assert np.all(np.diff(encoded.occupations) <= 0)


@pytest.mark.dice
def test_shci_retires_a_pair_for_every_orbital_above_the_window():
    """
    An orbital too full to correlate takes its two electrons into the core.

    An orbital too empty to correlate moves to the virtuals and the electron count doesn't change.
    """
    _, occupations = reference()
    lo, hi = NARROW
    encoded = capped()
    before, core_before = encoded.active_electrons, (encoded.mol.nelectron - encoded.active_electrons) // 2
    retained = encoded.orbital_initial[:, :core_before].copy()

    encoded.SHCI(eps1=SELECTION, lo=lo, hi=hi)

    above = int((occupations > hi).sum())
    core = (encoded.mol.nelectron - encoded.active_electrons) // 2
    assert encoded.active_electrons == before - 2 * above
    assert core == core_before + above
    # The orbitals that were already core are still core, not merely still present.
    overlap = encoded.mol.intor("int1e_ovlp")
    projector = encoded.orbital_initial[:, :core] @ encoded.orbital_initial[:, :core].T @ overlap
    assert np.allclose(projector @ retained, retained, atol=1e-8)


@pytest.mark.dice
def test_shci_leaves_a_space_a_correction_can_be_made_in():
    """
    The window that comes back is closed-shell, non-empty, and not simply full.
    """
    encoded = capped()

    encoded.SHCI(eps1=SELECTION, lo=NARROW[0], hi=NARROW[1])

    assert (encoded.active_electrons, encoded.active_space_size) == NARROWED
    assert not encoded.active_electrons % 2
    assert not (encoded.mol.nelectron - encoded.active_electrons) % 2
    assert 0 < encoded.active_electrons < 2 * encoded.active_space_size


@pytest.mark.dice
def test_a_wider_window_keeps_more_of_the_space():
    """
    Widening the window can only keep more.
    """
    narrow, wide = capped(), capped()

    narrow.SHCI(eps1=SELECTION, lo=NARROW[0], hi=NARROW[1])
    wide.SHCI(eps1=SELECTION, lo=WIDE[0], hi=WIDE[1])

    assert (wide.active_electrons, wide.active_space_size) == WIDENED
    assert wide.active_space_size > narrow.active_space_size
    assert wide.active_electrons > narrow.active_electrons


@pytest.mark.dice
def test_shci_keeps_every_orbital_of_the_molecule():
    """
    The truncation narrows the window.
    """
    encoded = capped()

    encoded.SHCI(eps1=SELECTION, lo=NARROW[0], hi=NARROW[1])

    orbitals = encoded.orbital_initial
    assert orbitals.shape == (encoded.mol.nao, encoded.mol.nao)
    overlap = encoded.mol.intor("int1e_ovlp")
    assert np.allclose(orbitals.T @ overlap @ orbitals, np.eye(encoded.mol.nao), atol=1e-8)


@pytest.mark.dice
def test_shci_leaves_the_correlated_space_it_was_handed():
    """
    Core and active together span what they spanned before.

    Unlike MP2's, a CI density mixes occupied with virtual, so the occupied space itself is not
        preserved and cannot be asserted on. The space MP2 chose is preserved.
    """
    encoded = capped()
    correlated = (encoded.mol.nelectron - encoded.active_electrons) // 2 + encoded.active_space_size
    overlap = encoded.mol.intor("int1e_ovlp")
    before = encoded.orbital_initial[:, :correlated]
    span = before @ before.T @ overlap

    encoded.SHCI(eps1=SELECTION, lo=NARROW[0], hi=NARROW[1])

    after = encoded.orbital_initial[:, :correlated]
    assert np.allclose(after @ after.T @ overlap, span, atol=1e-8)


@pytest.mark.dice
def test_shci_keeps_the_active_space_on_the_contact():
    """
    What survives the truncation still sits on the targeted atoms.

    The fragment's four survivors carry mean weight 0.46 on the target AOs, never below 0.39,
        against the 0.2 AVAS demanded of every orbital it admitted.
    """
    encoded = capped()

    encoded.SHCI(eps1=SELECTION, lo=NARROW[0], hi=NARROW[1])

    weights = contact_weight(encoded.mol, window(encoded), all_carbons(encoded.mol))
    assert weights.mean() >= encoded.avas_threshold


def test_shci_refuses_before_an_active_space_exists():
    """
    There is nothing to solve before AVAS and MP2 have chosen a space.
    """
    encoded = EncodeProtein(fragment())

    with pytest.raises(EncodingError):
        encoded.SHCI()


def test_shci_refuses_without_dice():
    """
    Reject without Dice.
    """
    encoded = capped()
    encoded.dice = None

    with pytest.raises(EncodingError):
        encoded.SHCI()


@pytest.mark.dice
def test_shci_keeps_what_dice_wrote_when_it_is_given_somewhere_to_write(tmp_path):
    """
    Tests that output is written and that something is done.
    """
    encoded = capped()
    encoded.scratch = str(tmp_path)

    encoded.SHCI(eps1=SELECTION, lo=NARROW[0], hi=NARROW[1])

    written = {path.name for path in tmp_path.iterdir()}
    assert "FCIDUMP" in written  # the integrals it was handed
    assert "input.dat" in written  # the schedule and the window it was handed
    assert "output.dat" in written  # what it said about the run
    assert "spatialRDM.0.0.txt" in written  # the density the truncation is made of


@pytest.mark.dice
@pytest.mark.parametrize(
    "lo,hi",
    [
        (1.0, 1.0),  # keeps nothing at all
        (0.0, 0.1),  # keeps only orbitals with no electrons in them
        (1.9, 2.0),  # keeps only orbitals that are already full
    ],
)
def test_shci_refuses_a_window_that_leaves_nothing_to_correct(lo, hi):
    """
    An empty window, an empty space and a closed shell give a CASCI with no excitation in it, 
        so SAPT contributes nothing.
    """
    encoded = capped()

    with pytest.raises(EncodingError):
        encoded.SHCI(eps1=SELECTION, lo=lo, hi=hi)


@pytest.mark.dice
def test_the_paper_window_is_too_narrow_for_a_saturated_cutout():
    """
    The original window keeps one orbital of the fragment, at n = 0.024, so (0e, 1o).

    The window was set on KDM5A, whose active space is built round an open-shell iron centre. A
        saturated peptide has no static correlation for it to find: the fragment's occupations run
        1.9933 to 1.9761 and 0.0240 to 0.0066, and the window falls in the gap between them. The
        same measurement on benzene and phenol keeps six of eight, so what decides this is whether
        a pi system sits on the contact, not the size of the space MP2 hands over.
    """
    encoded = capped()

    with pytest.raises(EncodingError):
        encoded.SHCI(eps1=SELECTION, lo=PAPER[0], hi=PAPER[1])


@pytest.mark.dice
def test_the_default_window_leaves_a_space_worth_correcting():
    """
    The window has to survive a cutout with no pi system on it, forced by the test above.
    """
    encoded = capped()

    encoded.SHCI()

    assert encoded.active_space_size >= 2
    assert 0 < encoded.active_electrons < 2 * encoded.active_space_size
    assert encoded.active_space_size <= SIMULABLE_ORBITALS


# --------------------------------------------------------------------------------------------
# The bin. Needs the SCF, the cap and a fifty-orbital solve, so for the cluster.
# --------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def reduced(name):
    """
    A real cutout carried the whole way, truncated by a window that keeps everything.
    """
    encoded = solved(prepare(name))
    encoded.AVAS()
    encoded.MP2()
    encoded.SHCI(lo=EVERYTHING[0], hi=EVERYTHING[1])
    return encoded


@pytest.mark.dice
@pytest.mark.hpc_long_dice
@pytest.mark.parametrize("name", SUBSET)
def test_shci_solves_the_space_mp2_leaves_on_a_real_cutout(name):
    """
    The SHCI runs to completion over nmax orbitals of a cutout of the bin, and returns a
        density.
    """
    encoded = reduced(name)

    assert encoded.active_space_size == encoded.nmax
    assert encoded.shci_energy < encoded.energy
    occupations = encoded.occupations
    assert len(occupations) == encoded.active_space_size
    assert np.all(occupations > -1e-8)
    assert np.all(occupations < 2 + 1e-8)
    assert np.all(np.diff(occupations) <= 1e-8)
    assert occupations.sum() == pytest.approx(encoded.active_electrons, abs=1e-6)


@pytest.mark.dice
@pytest.mark.hpc_long_dice
@pytest.mark.parametrize("name", SUBSET)
def test_the_window_leaves_the_subset_a_space_a_vqe_could_carry(name):
    """
    VQE can handle the output.
    """
    encoded = reduced(name)
    lo, hi = PAPER

    occupations = encoded.occupations
    kept = occupations[(occupations >= lo) & (occupations <= hi)]
    nelecas = encoded.active_electrons - 2 * int((occupations > hi).sum())

    assert 2 <= len(kept) <= SIMULABLE_ORBITALS
    assert 0 < nelecas < 2 * len(kept)
