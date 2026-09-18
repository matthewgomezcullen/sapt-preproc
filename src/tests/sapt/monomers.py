"""
The monomers the SAPT tests are built on.

The water dimer of the original paper's Fig. 4, whose first order electrostatic and exchange
    energies its Table III publishes at SAPT(RHF) and its Table IV at SAPT(CASCI) over a (6e, 6o)
    active space on the compressed monomer, in the same monomer-centred 6-31g basis. Six atoms and
    26 basis functions, so a test can build any integral it wants outright and compare against it.

7LOE_Y84 read from its saved artefact, which SAPT needs carried through CASCI rather than stopped at
    the Hamiltonian.
"""

import functools
import os

import numpy as np
from pyscf import fci, gto, mcscf, scf

from conftest import DATA
from encode import EncodeProtein, SolveLigand
from prepare import PrepareComplex
from utils import sapt

# Fig. 4, the dimer at R_OH = 0.2397 A in the second monomer, which is the geometry Tables III and
# IV are scored at. Angstrom.
EQUILIBRIUM = """O -1.551007 -0.114520  0.000000
H -1.934259  0.762503  0.000000
H -0.599677  0.040712  0.000000"""
COMPRESSED = """O  1.350625  0.111469  0.000000
H  1.433068 -0.009834 -0.189640
H  1.433068 -0.009834  0.189640"""

BASIS = "6-31g"

WATER_ELECTRONS = 10
WATER_FUNCTIONS = 13

# Section II.B's active space, over the compressed monomer alone.
WATER_NCAS = 6
WATER_NELECAS = 6

# Table III and Table IV: the first order electrostatic energy of this dimer, at SAPT(RHF) and with
# the compressed monomer correlated. Hartree.
ELST_RESTRICTED = -0.009707
ELST_CORRELATED = -0.009756

# The same tables' first order exchange energy, in the S^2 approximation. Hartree.
EXCH_RESTRICTED = 0.005482
EXCH_CORRELATED = 0.005559


def pulled_apart(geometry, distance):
    moved = []
    for line in geometry.split("\n"):
        element, x, y, z = line.split()
        moved.append(f"{element} {float(x) + distance} {y} {z}")
    return "\n".join(moved)


SEPARATION = 100.0  # Angstrom

SEPARATED = pulled_apart(COMPRESSED, SEPARATION)


@functools.lru_cache(maxsize=None)
def build_water(geometry):
    return gto.M(atom=geometry, basis=BASIS, verbose=0)


@functools.lru_cache(maxsize=None)
def solve_water(geometry):
    mean_field = scf.RHF(build_water(geometry))
    mean_field.conv_tol = 1e-12
    mean_field.verbose = 0
    mean_field.kernel()
    return mean_field


@functools.lru_cache(maxsize=None)
def casci_water(geometry):
    correlated = mcscf.CASCI(solve_water(geometry), WATER_NCAS, WATER_NELECAS)
    correlated.verbose = 0
    correlated.kernel()
    return correlated


@functools.lru_cache(maxsize=None)
def correlate_water(geometry):
    """
    CASCI's one- and two-particle density matrices over the active space.
    """
    correlated = casci_water(geometry)
    return correlated.fcisolver.make_rdm12(correlated.ci, WATER_NCAS, WATER_NELECAS)


def correlated_water_density(geometry):
    return sapt.density(
        solve_water(geometry).mo_coeff,
        correlate_water(geometry)[0],
        WATER_NCAS,
        WATER_NELECAS,
        WATER_ELECTRONS,
    )


def active_orbitals(geometry):
    core = (WATER_ELECTRONS - WATER_NELECAS) // 2
    return solve_water(geometry).mo_coeff[:, core:core + WATER_NCAS]


def restricted_rdm1(ncas, nelecas):
    return np.diag(np.where(np.arange(ncas) < nelecas // 2, 2.0, 0.0))


def restricted_rdm12(ncas, nelecas):
    """
    The density matrices of the determinant filling the lowest active orbitals. Contracted by FCI 
        from the one CI coefficient rather than written in.
    """
    strings = fci.cistring.num_strings(ncas, nelecas // 2)
    determinant = np.zeros((strings, strings))
    determinant[0, 0] = 1.0 # 0,0 gives the lowest-energy orbital.
    return fci.direct_spin1.make_rdm12(determinant, ncas, (nelecas // 2,) * 2)


def occupied(geometry, correlated):
    """
    A water monomer over its occupied orbitals. The core doubly occupied, and the active space, if 
        `correlated`, in its CASCI ground state.

    The CI vector is carried into the space of core and active orbitals together, and FCI contracts
        the density matrices over it, so the blocks holding a core index come out of the wavefunction
        rather than out of Eq. (28).
    """
    if correlated:
        ncas, nelecas, ci = WATER_NCAS, WATER_NELECAS, casci_water(geometry).ci
    else:
        ncas, nelecas, ci = 0, 0, np.ones((1, 1))
    core = (WATER_ELECTRONS - nelecas) // 2
    orbitals, spin = core + ncas, WATER_ELECTRONS // 2
    # Each active string with the core's orbitals, which come first, filled.
    carried = [
        (int(string) << core) | ((1 << core) - 1)
        for string in fci.cistring.make_strings(range(ncas), nelecas // 2)
    ]
    addresses = fci.cistring.strs2addr(orbitals, spin, np.array(carried, dtype=np.int64))
    strings = fci.cistring.num_strings(orbitals, spin)
    vector = np.zeros((strings, strings))
    vector[np.ix_(addresses, addresses)] = ci
    rdm1, rdm2 = fci.direct_spin1.make_rdm12(vector, orbitals, (spin, spin))
    return build_water(geometry), solve_water(geometry).mo_coeff[:, :orbitals], rdm1, rdm2


def water_dimer():
    return build_water(EQUILIBRIUM), build_water(COMPRESSED)


ENCODED = os.path.join(DATA, "encoded")

COMPLEX = "7LOE_Y84"

# 7LOE_Y84 as its artefacts hold it. The window derives itself to leave `ncas_limit` orbitals of the
# fifty Dice solved, and retires the electrons above it to the core.
CUTOUT_FUNCTIONS = 1695
CUTOUT_ELECTRONS = 1144
CUTOUT_NCAS = 14
POSES = 40
POSE_FUNCTIONS = 113


@functools.lru_cache(maxsize=None)
def read_prepared_complex():
    prepared = PrepareComplex("", [], os.path.join(ENCODED, COMPLEX))
    if not prepared.prepared():
        raise FileNotFoundError(f"No prepared artefact for {COMPLEX} under {ENCODED}")
    return prepared


@functools.lru_cache(maxsize=None)
def read_encoded_cutout():
    """
    7LOE_Y84's protein, read back from its artefact.
    """
    cutout = EncodeProtein(read_prepared_complex(), os.path.join(ENCODED, COMPLEX))
    if not cutout.correlated():
        raise FileNotFoundError(f"No correlated artefact for {COMPLEX} under {ENCODED}")
    return cutout


@functools.lru_cache(maxsize=None)
def read_poses():
    return SolveLigand(read_prepared_complex(), os.path.join(ENCODED, COMPLEX))


def cutout_density():
    cutout = read_encoded_cutout()
    return sapt.density(
        cutout.orbital_initial,
        cutout.rdm1,
        cutout.active_space_size,
        cutout.active_electrons,
        cutout.mol.nelectron,
    )


def cutout_active():
    cutout = read_encoded_cutout()
    return sapt.active(
        cutout.orbital_initial,
        cutout.active_space_size,
        cutout.active_electrons,
        cutout.mol.nelectron,
    )


def cutout_cumulant():
    cutout = read_encoded_cutout()
    return sapt.cumulant(cutout.rdm1, cutout.rdm2)
