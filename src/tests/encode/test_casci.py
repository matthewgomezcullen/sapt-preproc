"""
Solving the encoded active space classically, with CASCI.

CASCI diagonalises the Hamiltonian the encoding kept, exactly, inside the window.

SAPT reads the protein through its orbitals and the ground state's spin-summed one- and two-particle
    density matrices over the active orbitals, so those are kept beside the energy. They follow
    PySCF's convention, rdm1[p, q] = <q+ p> and rdm2[p, q, r, s] = <p+ r+ s q>, in which

    E = e_core + h1[p, q] rdm1[q, p] + 1/2 h2[p, q, r, s] rdm2[p, q, r, s].

The fragment is capped to eight orbitals, as for SHCI. Still iterates, but solved quickly.

A space of more than sixteen orbitals is refused before it is solved. CASCI and VQE cannot solve it.
    The limit is tested over two electrons, which leave a space of that width 256 determinants.
"""

import functools
import os
import shutil

import numpy as np
import pytest
from pyscf import fci, mcscf

from cutouts import all_carbons, fragment
from encode import ENCODED, SOLVED, EncodeProtein, EncodingError
from utils import encode, save

NAME = "ACE-VAL-NME"

NMAX = 8

AS_LIMIT = 16

# Agreement with PySCF's own CASCI. The energy is quadratic in the error of the state, so the 
# density matrices agree only to about the root of the energy's tolerance.
ENERGY = 1e-8  # Hartree
DENSITY = 1e-4

EXACT = 1e-10

# An occupation this far from 0 and 2 is fractional
FRACTIONAL = 1e-3

# Made-up two-orbital, two-electron system. Its lowest state is a triplet, which should be refused.
# Both orbitals share h1=0, so only the repulsion decides which state is lowest.
# U (ON_SITE): repulsion when both electrons share one orbital, (00|00) = (11|11).
# J (COULOMB): repulsion when there is one electron in each orbital, (00|11)
# K (EXCHANGE): Exchange integral, (01|01). Lowers the energy of electrons with parallel spin, and 
# mixes the two states where both electrons share an orbital.
# Triplet (one e- in each orbital, spins parallel) = J - K = 0.4 is the lowest state.
ON_SITE, COULOMB, EXCHANGE = 1.0, 0.5, 0.1  # U, J, K


@functools.lru_cache(maxsize=None)
def _encoded():
    """
    The fragment through RHF, AVAS over every carbon and the MP2 cap, with the integrals H() keeps
        over the space.

    H() maps them onto qubits, which CASCI never reads and which is most of its cost, so they are 
        taken from the helper H() uses.
    """
    encoder = EncodeProtein(fragment())
    encoder.RHF()
    encoder.AVAS(targets=all_carbons(encoder.mol))
    encoder.nmax = NMAX
    encoder.MP2()
    encoder.e_core, encoder.h1, encoder.h2 = encode.integrals(
        encoder.mean_field,
        encoder.orbital_initial,
        encoder.active_space_size,
        encoder.active_electrons,
    )
    return encoder


def encoded():
    cached = _encoded()
    encoder = EncodeProtein(cached.prepared)
    for key in (
        "mol", "mean_field", "energy", "correlation", "active_space_size", "active_electrons",
        "orbital_initial", "occupations", "e_core", "h1", "h2",
    ):
        setattr(encoder, key, getattr(cached, key))
    return encoder


@functools.lru_cache(maxsize=None)
def casci_solved():
    encoder = encoded()
    encoder.CASCI()
    return encoder


@functools.lru_cache(maxsize=None)
def exact():
    """
    The energy and density matrices of PySCF's own CASCI over the same space, which builds its
        integrals from the mean field rather than reading the kept ones.
    """
    cached = _encoded()
    reference = mcscf.CASCI(cached.mean_field, cached.active_space_size, cached.active_electrons)
    reference.verbose = 0
    reference.kernel(cached.orbital_initial)
    rdm1, rdm2 = reference.fcisolver.make_rdm12(
        reference.ci, cached.active_space_size, cached.active_electrons
    )
    return reference.e_tot, rdm1, rdm2


def triplet_encoded():
    """
    An encoder holding a Hamiltonian whose ground state is a triplet: two degenerate orbitals and 
        two electrons, whose exchange favours parallel spins.
    """
    encoder = EncodeProtein(fragment())
    encoder.active_space_size, encoder.active_electrons = 2, 2
    encoder.e_core, encoder.h1 = 0.0, np.zeros((2, 2))
    encoder.h2 = np.zeros((2, 2, 2, 2))
    encoder.h2[0, 0, 0, 0] = encoder.h2[1, 1, 1, 1] = ON_SITE
    encoder.h2[0, 0, 1, 1] = encoder.h2[1, 1, 0, 0] = COULOMB
    for p, q, r, s in [(0, 1, 0, 1), (1, 0, 1, 0), (0, 1, 1, 0), (1, 0, 0, 1)]:
        encoder.h2[p, q, r, s] = EXCHANGE
    return encoder


def encoded_two_electrons_over(ncas):
    """
    An encoder holding a Hamiltonian over `ncas` orbitals and two electrons, whose ground state puts
        both in the lowest orbital.
    """
    encoder = EncodeProtein(fragment())
    encoder.active_space_size, encoder.active_electrons = ncas, 2
    encoder.e_core, encoder.h1 = 0.0, np.diag(np.arange(float(ncas)))
    encoder.h2 = np.zeros((ncas,) * 4)
    return encoder


def solved_past_the_limit(*args, **kwargs):
    raise AssertionError("a space past the limit was solved")


@pytest.fixture(scope="module")
def kept(tmp_path_factory):
    directory = str(tmp_path_factory.mktemp("kept") / NAME)
    encoder = encoded()
    encoder.shci_energy = encoder.energy + encoder.correlation
    save.save_solved({key: getattr(encoder, key) for key in SOLVED}, NAME, directory)
    save.save_encoded({key: getattr(encoder, key) for key in ENCODED}, NAME, directory)
    return directory


def copy_fragment_dir(kept, tmp_path):
    directory = str(tmp_path / NAME)
    shutil.copytree(kept, directory)
    return directory


def test_casci_refuses_before_the_hamiltonian_is_built():
    encoder = encoded()
    encoder.e_core = encoder.h1 = encoder.h2 = None

    with pytest.raises(EncodingError):
        encoder.CASCI()


def test_casci_finds_the_state_pyscf_finds_over_the_same_space():
    encoder = casci_solved()
    energy, rdm1, rdm2 = exact()

    assert encoder.casci_energy == pytest.approx(energy, abs=ENERGY)
    assert np.allclose(encoder.rdm1, rdm1, atol=DENSITY)
    assert np.allclose(encoder.rdm2, rdm2, atol=DENSITY)


def test_the_correlated_energy_lies_below_rhf():
    encoder = casci_solved()

    assert encoder.casci_energy < encoder.energy


def test_the_solution_is_kept_on_the_encoder():
    encoder = encoded()
    ncas = encoder.active_space_size

    energy, rdm1, rdm2 = encoder.CASCI()

    assert energy == encoder.casci_energy
    assert rdm1 is encoder.rdm1
    assert rdm2 is encoder.rdm2
    assert rdm1.shape == (ncas, ncas)
    assert rdm2.shape == (ncas,) * 4


def test_the_density_matrices_recover_the_energy():
    """
    Contracted with the Hamiltonian they were solved over, in PySCF's convention, they give back the
        energy.
    """
    encoder = casci_solved()

    energy = (
        encoder.e_core
        + np.einsum("pq,qp", encoder.h1, encoder.rdm1)
        + 0.5 * np.einsum("pqrs,pqrs", encoder.h2, encoder.rdm2)
    )

    assert energy == pytest.approx(encoder.casci_energy, abs=EXACT)


def test_the_one_particle_density_holds_the_active_electrons():
    encoder = casci_solved()

    assert np.trace(encoder.rdm1) == pytest.approx(encoder.active_electrons, abs=EXACT)


def test_the_two_particle_density_contracts_to_the_one_particle_density():
    """
    Tracing out the second electron of every pair leaves the one-particle density.
    """
    encoder = casci_solved()

    contracted = np.einsum("pqrr->pq", encoder.rdm2)

    assert np.allclose(contracted, (encoder.active_electrons - 1) * encoder.rdm1, atol=EXACT)


def test_the_density_matrices_have_the_symmetry_of_a_real_state():
    """
    The one-particle density is symmetric. The two-particle density is symmetric under swapping its
        electrons, and under transposing both at once.
    """
    encoder = casci_solved()

    assert np.allclose(encoder.rdm1, encoder.rdm1.T, atol=EXACT)
    assert np.allclose(encoder.rdm2, encoder.rdm2.transpose(2, 3, 0, 1), atol=EXACT)
    assert np.allclose(encoder.rdm2, encoder.rdm2.transpose(1, 0, 3, 2), atol=EXACT)


def test_the_natural_occupations_lie_between_zero_and_two_and_some_clear_of_both():
    occupations = np.linalg.eigvalsh(casci_solved().rdm1)

    assert np.all(occupations >= -EXACT)
    assert np.all(occupations <= 2 + EXACT)
    assert np.any((occupations > FRACTIONAL) & (occupations < 2 - FRACTIONAL))


def test_a_space_whose_ground_state_is_not_a_singlet_is_refused():
    encoder = triplet_encoded()

    with pytest.raises(EncodingError):
        encoder.CASCI()

    assert encoder.casci_energy is None


def test_casci_rejects_a_solve_that_did_not_converge():
    encoder = encoded()
    encoder.casci_max_cycle = 1

    with pytest.raises(EncodingError):
        encoder.CASCI()

    assert encoder.casci_energy is None
    assert encoder.rdm1 is None
    assert encoder.rdm2 is None


def test_a_space_of_sixteen_orbitals_is_solved():
    encoder = encoded_two_electrons_over(AS_LIMIT)

    encoder.CASCI()

    assert encoder.casci_energy == pytest.approx(0.0, abs=EXACT)
    assert encoder.rdm1.shape == (AS_LIMIT, AS_LIMIT)


def test_a_space_wider_than_sixteen_orbitals_is_refused_before_it_is_solved(monkeypatch):
    encoder = encoded_two_electrons_over(AS_LIMIT + 1)
    monkeypatch.setattr(fci.direct_spin1.FCISolver, "kernel", solved_past_the_limit)

    with pytest.raises(EncodingError):
        encoder.CASCI()

    assert encoder.casci_energy is None
    assert encoder.rdm1 is None
    assert encoder.rdm2 is None


def test_the_solution_is_kept_and_read_back(kept, tmp_path):
    directory = copy_fragment_dir(kept, tmp_path)
    solved = EncodeProtein(fragment(), directory)
    solved.CASCI()

    read_back = EncodeProtein(fragment(), directory)

    assert os.path.isfile(save.casci_path(NAME, directory))
    assert read_back.casci_energy == solved.casci_energy
    np.testing.assert_array_equal(read_back.rdm1, solved.rdm1)
    np.testing.assert_array_equal(read_back.rdm2, solved.rdm2)


def test_a_new_hamiltonian_discards_the_solution_it_replaces(kept, tmp_path, monkeypatch):
    directory = copy_fragment_dir(kept, tmp_path)
    encoder = EncodeProtein(fragment(), directory)
    encoder.CASCI()
    # The qubit operator is not what is tested here, and it is most of the cost of H().
    monkeypatch.setattr(encode, "qubits", lambda *args, **kwargs: None)

    encoder.encode()

    assert encoder.casci_energy is None
    assert encoder.rdm1 is None
    assert encoder.rdm2 is None
    assert not os.path.isfile(save.casci_path(NAME, directory))
    assert save.load_encoded(NAME, directory) is not None
