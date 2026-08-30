"""
Capping the active space with MP2 natural orbitals, README's MCP-NO step.

AVAS is handed every valence p shell on the contact and returns whatever has weight on it, which
    over the bin is roughly 140 to 270 orbitals against Dice's ceiling of about fifty. The cap
    correlates the AVAS space with MP2, ranks its natural orbitals by fractionality min(n, 2 - n),
    and keeps the nmax most fractional: the orbitals MP2 says carry the correlation.

One trap decides whether any of this is right. AVAS semicanonicalizes its orbitals but returns no
    orbital energies, and mf.mo_energy still holds the canonical SCF values, wrong for the new
    orbitals by up to 0.70 Ha on the fragment. MP2 divides by those energies: fed the stale ones
    it returns a correlation energy 68% too negative and natural occupations two to three times
    too fractional, silently, and at sixteen orbitals kept it changes which orbitals are kept.
    Every comparison here is therefore against a reference that recomputes the energies from the
    Fock matrix, and the tolerances are hundreds of times tighter than that failure.

What runs here is the fragment with every carbon targeted, 35 orbitals from 24 target AOs, capped
    to eight so the truncation has something to do; the default rule reaches only two of its
    atoms. The bin itself needs a converged SCF first and sits at the bottom, marked hpc.
"""

import functools

import numpy as np
import pytest
from pyscf import mp
from pyscf.mcscf import avas

from cutouts import SUBSET, fragment, prepare, solved
from encode import EncodeProtein, EncodingError

# What the rest of the pipeline can actually take. Dice runs comfortably to roughly fifty orbitals
# and the original paper's final active space was eight. The raw AVAS spaces of the bin run to
# five times this, which is the reason this stage exists.
TRACTABLE_ORBITALS = 50

# Small enough to bite the fragment's 35-orbital space, and the size the paper's own pipeline
# ended at. The top eight here come back four occupied- and four virtual-derived: an (8e, 8o)
# window.
CAP = 8

# Agreement with the independent reference. Loose enough that an implementation integrating the
# MP2 differently (density fitting moves the fourth decimal) still agrees, and two orders of
# magnitude tighter than the stale-energy mistake the margin exists to catch.
CORRELATION = 1e-3  # relative, on the correlation energy
OCCUPATION = 1e-4   # absolute, on a natural occupation; the selection boundary gap is 4.8e-4


def all_carbons(mol):
    """
    Every carbon's 2p shell: a target set large enough that the cap has to cut.
    """
    return [
        f"{index} C 2p"
        for index in range(mol.natm)
        if mol.atom_symbol(index) == "C"
    ]


def window(encoded):
    """
    The active columns of the orbital set, located by the electron bookkeeping.
    """
    core = (encoded.mol.nelectron - encoded.nelecas) // 2
    return encoded.orbitals[:, core:core + encoded.ncas]


def contact_weight(mol, vectors, targets):
    """
    Weight of each orbital on the span of the target AOs, in the overlap metric. One per column,
        each within [0, 1].
    """
    overlap = mol.intor("int1e_ovlp")
    indices = np.unique(np.concatenate([mol.search_ao_label(t) for t in targets]))
    projected = overlap[indices] @ vectors
    return np.einsum(
        "ti,ti->i", np.linalg.solve(overlap[np.ix_(indices, indices)], projected), projected
    )


@functools.lru_cache(maxsize=None)
def reference():
    """
    The capped space worked out independently of the implementation: PySCF's AVAS and MP2 called
        directly, with the orbital energies recomputed from the Fock matrix because AVAS does not
        return them. A direct reading of README's step rather than of encode.py.

    Returns the size of the uncapped space, its MP2 correlation energy, and the occupations of the
        CAP most fractional natural orbitals, in descending order.
    """
    encoded = solved(fragment())
    mean_field, mol = encoded.mean_field, encoded.mol
    fock = mean_field.get_fock()  # before any orbital swap, while the density is the converged one
    raw, nelecas, orbitals = avas.avas(mean_field, all_carbons(mol), threshold=encoded.threshold)
    core, occupied = (mol.nelectron - nelecas) // 2, nelecas // 2
    energies = np.diag(orbitals.T @ fock @ orbitals)
    saved = mean_field.mo_coeff, mean_field.mo_energy
    try:
        mean_field.mo_coeff, mean_field.mo_energy = orbitals, energies
        correlated = mp.MP2(
            mean_field, frozen=list(range(core)) + list(range(core + raw, mol.nao))
        )
        correlated.verbose = 0
        correlation = correlated.kernel()[0]
        density = correlated.make_rdm1()[core:core + raw, core:core + raw]
    finally:
        mean_field.mo_coeff, mean_field.mo_energy = saved
    # The occupied-virtual block of an unrelaxed MP2 density is zero, so the blocks hold the whole
    # spectrum and keep each occupation's provenance.
    occupations = np.concatenate([
        np.linalg.eigvalsh(density[:occupied, :occupied]),
        np.linalg.eigvalsh(density[occupied:, occupied:]),
    ])
    kept = sorted(occupations, key=lambda n: min(n, 2 - n), reverse=True)[:CAP]
    return raw, correlation, np.sort(kept)[::-1]


@functools.lru_cache(maxsize=None)
def capped():
    """
    The fragment carried through RHF, AVAS over every carbon, and the MP2 cap at CAP. Its own
        encoder, because it changes settings and the one `solved` shares is read-only.
    """
    encoded = EncodeProtein(fragment())
    encoded.RHF()
    encoded.AVAS(targets=all_carbons(encoded.mol))
    encoded.nmax = CAP
    encoded.MP2()
    return encoded


# --------------------------------------------------------------------------------------------
# The cap, against an independent reading of the same step.
# --------------------------------------------------------------------------------------------


def test_mp2_runs_on_the_energies_avas_never_returned():
    """
    The correlation energy matches a reference whose orbital energies come from the Fock matrix.

    AVAS hands back rotated orbitals and nothing else; mean_field.mo_energy still describes the
        canonical orbitals, up to 0.70 Ha off for the new ones. MP2 fed those returns -0.171 Ha
        where the answer is -0.102 Ha, with nothing visibly wrong. This is the sharpest single
        detector of that mistake.
    """
    _, correlation, _ = reference()
    encoded = capped()

    assert encoded.correlation < 0
    assert encoded.correlation == pytest.approx(correlation, rel=CORRELATION)


def test_mp2_keeps_the_nmax_most_fractional_orbitals():
    """
    The window holds exactly the nmax natural orbitals MP2 ranks most fractional.

    Fractionality min(n, 2 - n) is the whole of the ranking, so this pins the selection and the
        occupations at once. On the fragment the eight keep 0.498 of the summed fractionality of
        the 35-orbital space and 0.231 of its correlation energy: the recorded cost of the cap,
        which nothing downstream can recover.
    """
    raw, _, occupations = reference()
    encoded = capped()

    assert raw > CAP
    assert encoded.ncas == encoded.nmax == CAP
    assert np.allclose(encoded.occupations, occupations, atol=OCCUPATION)


def test_mp2_recounts_the_electrons_the_window_kept():
    """
    The active electrons are the pairs of the occupied-derived orbitals that survived.

    A discarded occupied-derived orbital retires its pair to the core; it does not stay counted.
        A window with no electrons, or with every slot filled, has nothing left to correlate.
    """
    _, _, occupations = reference()
    encoded = capped()

    assert encoded.nelecas == 2 * int(np.sum(occupations > 1.0))
    assert not encoded.nelecas % 2
    assert not (encoded.mol.nelectron - encoded.nelecas) % 2
    assert 0 < encoded.nelecas < 2 * encoded.ncas


# --------------------------------------------------------------------------------------------
# Bookkeeping the truncation has to keep straight.
# --------------------------------------------------------------------------------------------


def test_mp2_keeps_every_orbital_of_the_molecule():
    """
    The cap narrows the window, not the basis.

    A discarded orbital moves to the core or the virtuals, it does not vanish: the Hamiltonian
        built downstream needs the full set to stay a basis.
    """
    encoded = capped()

    orbitals = encoded.orbitals
    assert orbitals.shape == (encoded.mol.nao, encoded.mol.nao)
    overlap = encoded.mol.intor("int1e_ovlp")
    assert np.allclose(orbitals.T @ overlap @ orbitals, np.eye(encoded.mol.nao), atol=1e-8)


def test_mp2_leaves_the_electrons_in_a_rotation_of_the_occupied_space():
    """
    The first N_e / 2 columns still span the SCF occupied space.

    Natural orbitals mix occupied only with occupied and virtual only with virtual, so however the
        cap shuffles them, the occupied span is untouched. A discarded occupied-derived orbital
        filed among the virtuals, or the reverse, breaks this while leaving every count looking
        right, and the core energy downstream is then integrated over the wrong orbitals.
    """
    encoded = capped()

    filled = encoded.mol.nelectron // 2
    overlap = encoded.mol.intor("int1e_ovlp")
    rotated = encoded.orbitals[:, :filled]
    original = encoded.mean_field.mo_coeff[:, :filled]
    assert np.allclose(
        rotated @ rotated.T @ overlap, original @ original.T @ overlap, atol=1e-8
    )


def test_mp2_keeps_the_active_space_on_the_contact():
    """
    What survives the cap still sits on the targeted atoms.

    MP2 ranks by correlation and carries no term for the ligand, and a combination of orbitals
        each above the AVAS threshold can itself fall below it, so this is not automatic. On the
        fragment the retained window carries mean weight 0.65 on the target AOs, never below 0.38,
        against the 0.2 AVAS demanded of every orbital it admitted.
    """
    encoded = capped()

    weights = contact_weight(encoded.mol, window(encoded), all_carbons(encoded.mol))
    assert weights.mean() >= encoded.threshold


def test_mp2_leaves_a_space_the_cap_already_fits():
    """
    A space within nmax passes through whole: same size, same electrons, same span.

    The default rule reaches two of the fragment's atoms and AVAS builds seven orbitals from
        them, so the cap has nothing to cut and the only change is the basis within the window.
    """
    encoded = solved(fragment())
    encoded.AVAS()

    assert encoded.ncas <= encoded.nmax
    before = (encoded.ncas, encoded.nelecas)
    overlap = encoded.mol.intor("int1e_ovlp")
    active = window(encoded)
    span = active @ active.T @ overlap

    encoded.MP2()

    assert (encoded.ncas, encoded.nelecas) == before
    assert len(encoded.occupations) == encoded.ncas
    active = window(encoded)
    assert np.allclose(active @ active.T @ overlap, span, atol=1e-8)


def test_mp2_is_reproducible():
    """
    The same converged cutout gives the same capped space twice.
    """
    encoded = capped()
    first = (
        encoded.ncas,
        encoded.nelecas,
        encoded.correlation,
        encoded.occupations.copy(),
        encoded.orbitals.copy(),
    )

    encoded.AVAS(targets=all_carbons(encoded.mol))
    encoded.MP2()

    assert (encoded.ncas, encoded.nelecas) == first[:2]
    assert encoded.correlation == pytest.approx(first[2])
    assert np.allclose(encoded.occupations, first[3])
    assert np.allclose(encoded.orbitals, first[4])


# --------------------------------------------------------------------------------------------
# Guards.
# --------------------------------------------------------------------------------------------


def test_mp2_refuses_before_an_active_space_exists():
    """
    There is nothing to cap before AVAS has chosen a space.
    """
    encoded = EncodeProtein(fragment())

    with pytest.raises(EncodingError):
        encoded.MP2()


def test_the_default_cap_is_within_what_can_be_solved():
    """
    The default nmax is something SHCI can actually take.

    This is the tractability requirement that used to sit, strictly xfailed, on the target count
        in test_avas.py; the cap is the stage that answers it now.
    """
    encoded = EncodeProtein(fragment())

    assert 0 < encoded.nmax <= TRACTABLE_ORBITALS


# --------------------------------------------------------------------------------------------
# The bin. Needs the SCF first, so hours each, for the cluster.
# --------------------------------------------------------------------------------------------


@pytest.mark.hpc
@pytest.mark.parametrize("name", SUBSET)
def test_mp2_caps_the_subset_to_what_can_be_solved(name):
    """
    On a real cutout the cap is what stands between AVAS and Dice, and it holds.

    The rule targets 96 to 186 AOs over the bin, roughly 140 to 270 AVAS orbitals, so the cap must
        bind, and what survives must still be physical and still on the contact. The occupation
        bounds are a real diagnostic here: an MP2 density is not N-representable, and a breach on
        a charged, small-gap cutout is the method saying the reference is bad.
    """
    encoded = solved(prepare(name))
    encoded.AVAS()
    raw = encoded.ncas

    encoded.MP2()

    assert raw > encoded.nmax
    assert encoded.ncas == encoded.nmax <= TRACTABLE_ORBITALS
    assert not encoded.nelecas % 2
    assert 0 < encoded.nelecas < 2 * encoded.ncas
    assert encoded.correlation < 0

    occupations = encoded.occupations
    assert len(occupations) == encoded.ncas
    assert np.all(occupations > -1e-8)
    assert np.all(occupations < 2 + 1e-8)
    assert np.all(np.diff(occupations) <= 1e-8)

    weights = contact_weight(
        encoded.mol, window(encoded), encoded._generate_target_orbitals()
    )
    assert weights.mean() >= encoded.threshold
