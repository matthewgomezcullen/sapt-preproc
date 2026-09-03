"""
The active-space Hamiltonian and its qubit image.

CASCI holds the integrals the window implies: the core energy, the one-electron integrals with the
    frozen core folded in, and the two-electron integrals over the active orbitals. Mapping those
    onto qubits is the last step of the pipeline.

Fidelity is against exact diagonalisation of the same space. The fragment is capped to four
    orbitals, so the operator is eight qubits and the matrix is 256 by 256.

The cap is used rather than the window, so none of this needs Dice.
"""

import functools

import numpy as np
import pytest
from pyscf import ao2mo, mcscf, scf

from cutouts import all_carbons, fragment
from encode import EncodeProtein, EncodingError
from utils import encode

# Small enough to diagonalise here, and the paper's own final size is eight.
CAP = 4

# Agreement with exact diagonalisation, in Hartree.
ENERGY = 1e-9


@functools.lru_cache(maxsize=None)
def capped():
    """
    The fragment through RHF, AVAS over every carbon, and the MP2 cap.
    """
    encoded = EncodeProtein(fragment())
    encoded.RHF()
    encoded.AVAS(targets=all_carbons(encoded.mol))
    encoded.nmax = CAP
    encoded.MP2()
    return encoded


@functools.lru_cache(maxsize=None)
def exact():
    """
    CASCI over the capped space, which is exact at this size.
    """
    encoded = capped()
    correlated = mcscf.CASCI(
        encoded.mean_field, encoded.active_space_size, encoded.active_electrons
    )
    correlated.verbose = 0
    correlated.kernel(encoded.orbital_initial)
    return correlated


def lowest(hamiltonian, nelecas):
    """
    The lowest eigenvalue of a qubit Hamiltonian among the states holding nelecas electrons.

    Counting set bits is agnostic to whether the mapping blocks the spins or interleaves them.
    """
    matrix = hamiltonian.to_matrix()
    sector = [
        state for state in range(matrix.shape[0]) if bin(state).count("1") == nelecas
    ]
    return np.linalg.eigvalsh(matrix[np.ix_(sector, sector)])[0].real


def test_the_integrals_reproduce_the_ones_casci_builds():
    """
    The helper reads the same integrals PySCF's own CASCI does.
    """
    encoded = capped()
    reference = exact()

    e_core, h1, h2 = encode.integrals(
        encoded.mean_field,
        encoded.orbital_initial,
        encoded.active_space_size,
        encoded.active_electrons,
    )

    expected, expected_core = reference.get_h1eff()
    assert e_core == pytest.approx(expected_core)
    assert np.allclose(h1, expected)
    assert np.allclose(h2, ao2mo.restore(1, reference.get_h2eff(), encoded.active_space_size))


def test_the_integrals_do_not_need_a_converged_scf():
    """
    The mean field only carries the molecule, so an unsolved one answers the same.

    This is what lets a stored run reach its Hamiltonian without the checkpoint.
    """
    encoded = capped()
    arguments = (
        encoded.orbital_initial, encoded.active_space_size, encoded.active_electrons
    )

    solved = encode.integrals(encoded.mean_field, *arguments)
    bare = encode.integrals(scf.RHF(encoded.mol), *arguments)

    assert bare[0] == pytest.approx(solved[0])
    assert np.allclose(bare[1], solved[1])
    assert np.allclose(bare[2], solved[2])


def test_the_core_energy_holds_more_than_the_nuclei():
    """
    The frozen electrons are in there too, so it is below the nuclear repulsion alone.
    """
    encoded = capped()

    e_core, _, _ = encode.integrals(
        encoded.mean_field,
        encoded.orbital_initial,
        encoded.active_space_size,
        encoded.active_electrons,
    )

    assert e_core < encoded.mol.energy_nuc()


def test_the_integrals_are_the_size_of_the_active_space():
    encoded = capped()

    _, h1, h2 = encode.integrals(
        encoded.mean_field,
        encoded.orbital_initial,
        encoded.active_space_size,
        encoded.active_electrons,
    )

    ncas = encoded.active_space_size
    assert h1.shape == (ncas, ncas)
    assert h2.shape == (ncas, ncas, ncas, ncas)


def test_a_rebuilt_molecule_gives_the_same_integrals():
    """
    A molecule through dumps and loads is the same molecule.
    """
    from pyscf import gto

    encoded = capped()
    arguments = (
        encoded.orbital_initial, encoded.active_space_size, encoded.active_electrons
    )

    before = encode.integrals(encoded.mean_field, *arguments)
    after = encode.integrals(scf.RHF(gto.loads(encoded.mol.dumps())), *arguments)

    assert after[0] == pytest.approx(before[0])
    assert np.allclose(after[1], before[1])
    assert np.allclose(after[2], before[2])


def test_the_qubit_hamiltonian_holds_the_energy_casci_finds():
    """
    Exact diagonalisation of the mapped operator returns the CASCI energy.
    """
    encoded = capped()

    hamiltonian = encoded.H()

    assert lowest(hamiltonian, encoded.active_electrons) == pytest.approx(
        exact().e_tot, abs=ENERGY
    )


def test_there_are_two_qubits_for_every_orbital():
    """
    RHF is closed shell, so each spatial orbital carries a spin pair.
    """
    encoded = capped()

    assert encoded.H().num_qubits == 2 * encoded.active_space_size


def test_the_hamiltonian_is_hermitian():
    encoded = capped()

    matrix = encoded.H().to_matrix()

    assert np.allclose(matrix, matrix.conj().T)


def test_the_coefficients_are_real():
    """
    Real orbitals give real integrals.
    """
    encoded = capped()

    assert np.allclose(encoded.H().coeffs.imag, 0.0)


def test_the_hamiltonian_is_kept_on_the_encoder():
    """
    Like every other step, which stores what it produced.
    """
    encoded = capped()

    returned = encoded.H()

    assert encoded.hamiltonian is returned
    assert encoded.e_core is not None


def test_jordan_wigner_is_the_mapping_by_default():
    encoded = capped()

    assert encoded.H() == encoded.H(mapping="jordan_wigner")


def test_another_mapping_is_another_operator():
    """
    A different mapping holds the same spectrum, so the energy is asserted too.
    """
    encoded = capped()

    jordan_wigner = encoded.H(mapping="jordan_wigner")
    parity = encoded.H(mapping="parity")

    assert parity != jordan_wigner
    assert lowest(parity, encoded.active_electrons) == pytest.approx(
        exact().e_tot, abs=ENERGY
    )


def test_an_unknown_mapping_is_refused():
    encoded = capped()

    with pytest.raises(EncodingError):
        encoded.H(mapping="not_a_mapping")


def test_the_hamiltonian_refuses_before_an_active_space_exists():
    """
    There is nothing to map before AVAS has chosen a space.
    """
    encoded = EncodeProtein(fragment())

    with pytest.raises(EncodingError):
        encoded.H()


def test_the_active_block_is_contiguous():
    """
    The integrals are read out of columns core to core + ncas, so the space has to sit there.
    """
    encoded = capped()

    core = (encoded.mol.nelectron - encoded.active_electrons) // 2
    overlap = encoded.mol.intor("int1e_ovlp")
    active = encoded.orbital_initial[:, core:core + encoded.active_space_size]

    assert active.shape[1] == encoded.active_space_size
    assert np.allclose(active.T @ overlap @ active, np.eye(encoded.active_space_size), atol=1e-8)
