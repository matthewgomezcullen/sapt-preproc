"""
Choosing the window the Hamiltonian is encoded over, once Dice has solved the space.

    nelecas = nelecas_solved - 2 |{i : n_i > hi}|.

The original paper cut at fixed thresholds, 0.02 <= n <= 1.97. Given no thresholds, `rewindow`
    derives them: it keeps the `ncas_limit` orbitals, ranked by min(n, 2 - n).

Every cut is taken from the space Dice solved. A window can be widened again.
"""

import os

import numpy as np
import pytest

from cutouts import fragment
from encode import ENCODED, SOLVED, EncodeProtein, EncodingError
from utils import save

NAME = "ACE-VAL-NME"

SOLVED_SPACE = (12, 12)
RHF_ENERGY, CORRELATION, SHCI_ENERGY = -99.0, -0.4, -99.5

OCCUPATIONS = np.array(
    [1.989, 1.976, 1.954, 1.929, 1.896, 1.839, 0.158, 0.102, 0.067, 0.043, 0.026, 0.021]
)

PAPER = (0.02, 1.97)
PAPERED = (8, 10)

# A limit, the space the window derived for it leaves, and the occupations it keeps.
CUTS = [
    (2, (2, 2), [1.839, 0.158]),
    (4, (4, 4), [1.896, 1.839, 0.158, 0.102]),
    # Fractionality is not symmetric about n = 1, so the two sides need not be kept evenly.
    (5, (6, 5), [1.929, 1.896, 1.839, 0.158, 0.102]),
    (6, (6, 6), [1.929, 1.896, 1.839, 0.158, 0.102, 0.067]),
    (8, (8, 8), [1.954, 1.929, 1.896, 1.839, 0.158, 0.102, 0.067, 0.043]),
    (12, SOLVED_SPACE, OCCUPATIONS),
    (20, SOLVED_SPACE, OCCUPATIONS),
]
IDS = ["two", "four", "odd", "six", "eight", "at-the-limit", "under-the-limit"]

NARROW, WIDE = 4, 8


def solved_fragment(limit=None):
    encoded = EncodeProtein(fragment())
    encoded.energy, encoded.correlation, encoded.shci_energy = (
        RHF_ENERGY, CORRELATION, SHCI_ENERGY
    )
    encoded.active_electrons, encoded.active_space_size = SOLVED_SPACE
    encoded.occupations = OCCUPATIONS.copy()
    encoded.orbital_initial = np.eye(len(OCCUPATIONS))
    encoded.solved_space = {key: getattr(encoded, key) for key in SOLVED}
    if limit is not None:
        encoded.ncas_limit = limit
    return encoded


def hamiltonian(ncas):
    """
    A Hamiltonian's worth of integrals over `ncas` orbitals, of the right shape.
    """
    return {"e_core": 0.0, "h1": np.zeros((ncas, ncas)), "h2": np.zeros((ncas,) * 4)}


def keep(directory, limit):
    os.makedirs(directory, exist_ok=True)
    solved = solved_fragment()
    save.save_solved(solved.solved_space, NAME, directory)

    nelecas, ncas = next(space for cut, space, _ in CUTS if cut == limit)
    occupations = next(kept for cut, _, kept in CUTS if cut == limit)
    encoded = dict(
        hamiltonian(ncas),
        active_electrons=nelecas,
        active_space_size=ncas,
        occupations=np.asarray(occupations),
    )
    save.save_encoded({key: encoded[key] for key in ENCODED}, NAME, directory)
    return directory


def test_the_thresholds_asked_for_are_the_window_taken():
    encoded = solved_fragment()

    encoded.rewindow(*PAPER)

    lo, hi = PAPER
    assert (encoded.active_electrons, encoded.active_space_size) == PAPERED
    np.testing.assert_allclose(
        encoded.occupations, [n for n in OCCUPATIONS if lo <= n <= hi]
    )


@pytest.mark.parametrize("limit, space, kept", CUTS, ids=IDS)
def test_no_thresholds_keeps_the_orbitals_a_single_determinant_describes_worst(limit, space, kept):
    encoded = solved_fragment(limit)

    encoded.rewindow()

    assert (encoded.active_electrons, encoded.active_space_size) == space
    np.testing.assert_allclose(encoded.occupations, kept)


def test_the_window_derived_is_the_thresholds_it_keeps_between():
    derived, thresholds = solved_fragment(NARROW), solved_fragment()

    derived.rewindow()
    thresholds.rewindow(derived.occupations.min(), derived.occupations.max())

    assert (thresholds.active_electrons, thresholds.active_space_size) == (
        derived.active_electrons,
        derived.active_space_size,
    )
    np.testing.assert_allclose(thresholds.occupations, derived.occupations)


@pytest.mark.parametrize("lo, hi", [(PAPER[0], None), (None, PAPER[1])], ids=["lo", "hi"])
def test_one_threshold_without_the_other_is_refused(lo, hi):
    encoded = solved_fragment()

    with pytest.raises(EncodingError):
        encoded.rewindow(lo, hi)


def test_a_window_is_cut_from_the_space_dice_solved_rather_than_the_last_one():
    encoded = solved_fragment(NARROW)
    encoded.rewindow()
    narrowed = (encoded.active_electrons, encoded.active_space_size)

    encoded.ncas_limit = WIDE
    encoded.rewindow()

    assert narrowed == next(space for cut, space, _ in CUTS if cut == NARROW)
    assert (encoded.active_electrons, encoded.active_space_size) == next(
        space for cut, space, _ in CUTS if cut == WIDE
    )


def test_a_new_window_leaves_the_space_dice_solved_where_it_is(tmp_path):
    directory = keep(os.path.join(str(tmp_path), NAME), NARROW)
    encoded = EncodeProtein(fragment(), directory)
    encoded.ncas_limit = WIDE

    encoded.rewindow()

    solved = save.load_solved(NAME, directory)
    assert (solved["active_electrons"], solved["active_space_size"]) == SOLVED_SPACE
    np.testing.assert_allclose(solved["occupations"], OCCUPATIONS)


def test_a_new_window_discards_the_hamiltonian_it_invalidates():
    encoded = solved_fragment(NARROW)
    nelecas, ncas = SOLVED_SPACE
    for key, value in hamiltonian(ncas).items():
        setattr(encoded, key, value)
    encoded.casci_energy, encoded.rdm1, encoded.rdm2 = (
        SHCI_ENERGY, np.zeros((ncas, ncas)), np.zeros((ncas,) * 4)
    )

    encoded.rewindow()

    assert (encoded.e_core, encoded.h1, encoded.h2) == (None, None, None)
    assert (encoded.casci_energy, encoded.rdm1, encoded.rdm2) == (None, None, None)
    assert encoded.solved()
    assert not encoded.encoded()
    assert not encoded.correlated()


def test_a_window_that_leaves_nothing_to_correct_is_refused():
    encoded = solved_fragment()

    with pytest.raises(EncodingError):
        encoded.rewindow(1.0, 1.0)


def test_rewindowing_refuses_before_a_space_is_solved():
    encoded = EncodeProtein(fragment())

    with pytest.raises(EncodingError):
        encoded.rewindow()
