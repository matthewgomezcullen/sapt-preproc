"""
Checkpointing the SCF.

It survives a job that runs out of wall clock. PySCF writes the checkpoint every cycle.

The store is `$DATA/scf` on the cluster and nowhere by default.
"""

import os

import numpy as np
import pytest
from pyscf import gto, scf

from cutouts import SUBSET, fragment, prepare
from encode import EncodeProtein
from utils import encode

# A closed-shell molecule small enough that a solve is free.
WATER = {"atom": "O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24", "basis": "6-31g"}

# Two runs of one SCF agree far past this. It is loose because the integrals are summed in the 
# order the threads finish in.
REPRODUCIBLE = 1e-9


def small(**overrides):
    """
    A water molecule, or a deliberate variation on one.
    """
    return gto.M(**{**WATER, "verbose": 0, **overrides})


@pytest.fixture
def fock_builds(monkeypatch):
    """
    A running count of the Fock builds, which are the whole cost of an SCF.
    """
    builds = []
    original = scf.hf.RHF.get_veff

    def counted(self, *args, **kwargs):
        builds.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(scf.hf.RHF, "get_veff", counted)
    return builds


def test_the_store_follows_data(monkeypatch):
    """
    Use $DATA as path.
    """
    monkeypatch.setenv("DATA", "/data/somewhere")
    monkeypatch.delenv("SCF_CHECKPOINTS", raising=False)

    assert encode.store() == os.path.join("/data/somewhere", "scf")


def test_the_store_can_be_pointed_somewhere_else(monkeypatch):
    """
    Use $SCF_CHECKPOINTS as an override.
    """
    monkeypatch.setenv("DATA", "/data/somewhere")
    monkeypatch.setenv("SCF_CHECKPOINTS", "/scratch/mine")

    assert encode.store() == "/scratch/mine"


def test_there_is_no_store_off_the_cluster(monkeypatch):
    """
    Nothing is cached on a machine that never set `$DATA`.
    """
    monkeypatch.delenv("DATA", raising=False)
    monkeypatch.delenv("SCF_CHECKPOINTS", raising=False)

    assert encode.store() is None


def test_the_same_molecule_takes_the_same_path(tmp_path):
    """
    A checkpoint is found again for an identical molecule.
    """
    store = str(tmp_path)

    assert encode.checkpoint(small(), store) == encode.checkpoint(small(), store)


@pytest.mark.parametrize(
    "overrides",
    [
        {"atom": "O 0 0 0; H 0 0 0.97; H 0.93 0 -0.24"},  # one bond lengthened
        {"basis": "sto-3g"},
        {"charge": 2},
        {"charge": 1, "spin": 1},
        {"atom": "O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24; He 8 0 0"},  # an atom added far away
    ],
)
def test_a_molecule_that_differs_takes_a_different_path(tmp_path, overrides):
    """
    A different molecule returns a different path.
    """
    store = str(tmp_path)

    assert encode.checkpoint(small(), store) != encode.checkpoint(small(**overrides), store)


@pytest.mark.parametrize("irrelevant", [{"verbose": 4}, {"max_memory": 1000}])
def test_what_does_not_change_the_solution_does_not_change_the_path(tmp_path, irrelevant):
    """
    Key does not change if the solution hasn't meaningfully changed.
    """
    store = str(tmp_path)

    assert encode.checkpoint(small(), store) == encode.checkpoint(small(**irrelevant), store)


def test_a_solve_with_nowhere_to_write_still_solves(fock_builds):
    """
    No store is the local default.
    """
    mean_field = encode.rhf(small(), 50)

    assert mean_field.converged
    assert fock_builds


def test_a_solve_leaves_its_checkpoint_behind(tmp_path):
    """
    Leave a checkpoint.
    """
    mol = small()

    encode.rhf(mol, 50, str(tmp_path))

    assert os.path.exists(encode.checkpoint(mol, str(tmp_path)))


def test_the_store_is_made_if_it_is_not_there(tmp_path):
    """
    The first job to run does not find `$DATA/scf` waiting for it.
    """
    store = str(tmp_path / "does" / "not" / "exist")

    encode.rhf(small(), 50, store)

    assert os.path.isdir(store)


def test_a_second_solve_reads_the_first_rather_than_repeating_it(tmp_path, fock_builds):
    """
    The second run does no work at all.
    """
    store = str(tmp_path)
    first = encode.rhf(small(), 50, store)
    fock_builds.clear()

    second = encode.rhf(small(), 50, store)

    assert not fock_builds
    assert second.converged
    assert second.e_tot == first.e_tot
    assert np.array_equal(second.mo_coeff, first.mo_coeff)
    assert np.array_equal(second.mo_energy, first.mo_energy)
    assert np.array_equal(second.mo_occ, first.mo_occ)


def test_what_comes_back_is_a_mean_field_the_rest_of_the_pipeline_can_use(tmp_path):
    """
    Checkpoint content is appropriately structured.
    """
    store = str(tmp_path)
    encode.rhf(small(), 50, store)

    restored = encode.rhf(small(), 50, store)

    assert restored.mol is not None
    assert np.trace(restored.make_rdm1() @ restored.mol.intor("int1e_ovlp")) == pytest.approx(
        restored.mol.nelectron, abs=1e-6
    )
    assert restored.get_fock() is not None


def test_a_different_molecule_is_solved_rather_than_read(tmp_path, fock_builds):
    """
    A different molecule doesn't use a stored checkpoint.
    """
    store = str(tmp_path)
    encode.rhf(small(), 50, store)
    fock_builds.clear()

    other = encode.rhf(small(basis="sto-3g"), 50, store)

    assert fock_builds
    assert other.e_tot == pytest.approx(scf.RHF(small(basis="sto-3g")).kernel(), abs=REPRODUCIBLE)


def test_an_unconverged_solve_is_not_read_back_as_a_converged_one(tmp_path):
    """
    A solve that ran out of cycles is not an answer, and a later run must not mistake it for one.
    """
    store = str(tmp_path)
    encode.rhf(small(), 1, store)

    finished = encode.rhf(small(), 50, store)

    assert finished.converged
    assert finished.e_tot == pytest.approx(scf.RHF(small()).kernel(), abs=REPRODUCIBLE)


def test_an_unconverged_solve_is_carried_on_rather_than_started_again(tmp_path, fock_builds):
    """
    A job that dies on the wall clock is continued.
    """
    store = str(tmp_path)
    encode.rhf(small(), 2, store)
    fock_builds.clear()

    warm = encode.rhf(small(), 50, store)
    resumed = len(fock_builds)
    fock_builds.clear()
    encode.rhf(small(), 50, str(tmp_path / "empty"))
    cold = len(fock_builds)

    assert warm.converged
    assert resumed < cold


def test_a_checkpoint_that_cannot_be_read_is_ignored(tmp_path):
    """
    A broken checkpoint does not break a run.
    """
    store = str(tmp_path)
    mol = small()
    os.makedirs(store, exist_ok=True)
    with open(encode.checkpoint(mol, store), "wb") as handle:
        handle.write(b"not an hdf5 file")

    mean_field = encode.rhf(mol, 50, store)

    assert mean_field.converged
    assert mean_field.e_tot == pytest.approx(scf.RHF(small()).kernel(), abs=REPRODUCIBLE)


def test_a_checkpoint_of_the_wrong_shape_is_ignored(tmp_path):
    """
    The orbitals of a checkpoint fit the molecule asking for them.
    """
    store = str(tmp_path)
    mol = small()
    os.makedirs(store, exist_ok=True)
    encode.rhf(small(basis="sto-3g"), 50, str(tmp_path / "other"))
    os.replace(
        encode.checkpoint(small(basis="sto-3g"), str(tmp_path / "other")),
        encode.checkpoint(mol, store),
    )

    mean_field = encode.rhf(mol, 50, store)

    assert mean_field.converged
    assert mean_field.mo_coeff.shape[0] == mol.nao
    assert mean_field.e_tot == pytest.approx(scf.RHF(small()).kernel(), abs=REPRODUCIBLE)


def test_the_encoder_checkpoints_where_the_environment_says(monkeypatch, tmp_path):
    """
    An encoder picks the store up from `$DATA`.
    """
    monkeypatch.setenv("DATA", str(tmp_path))
    monkeypatch.delenv("SCF_CHECKPOINTS", raising=False)

    encoded = EncodeProtein(fragment())

    assert encoded.checkpoints == os.path.join(str(tmp_path), "scf")


def test_the_encoder_solves_a_cutout_once(tmp_path, fock_builds):
    """
    A second encoder over the same cutout does no work.
    """
    prepared = fragment()
    first = EncodeProtein(prepared)
    first.checkpoints = str(tmp_path)
    first.RHF()
    fock_builds.clear()

    second = EncodeProtein(prepared)
    second.checkpoints = str(tmp_path)
    second.RHF()

    assert not fock_builds
    assert second.energy == first.energy
    assert np.array_equal(second.mean_field.mo_coeff, first.mean_field.mo_coeff)


def test_an_encoder_without_a_store_is_unchanged(monkeypatch, tmp_path):
    """
    The fragment tests run as they always have without a store.
    """
    monkeypatch.delenv("DATA", raising=False)
    monkeypatch.delenv("SCF_CHECKPOINTS", raising=False)

    encoded = EncodeProtein(fragment())
    encoded.RHF()

    assert encoded.checkpoints is None
    assert encoded.mean_field.converged
    assert not list(tmp_path.iterdir())


# --------------------------------------------------------------------------------------------
# The bin itself. The first of these is hours, the second reads the checkpoint.
# --------------------------------------------------------------------------------------------


@pytest.mark.hpc
@pytest.mark.parametrize("name", SUBSET)
def test_the_subset_solve_is_read_back_from_the_store(tmp_path, fock_builds, name):
    """
    A cutout of the bin, solved once and then read.
    """
    prepared = prepare(name)
    first = EncodeProtein(prepared)
    first.checkpoints = str(tmp_path)
    first.RHF()
    fock_builds.clear()

    second = EncodeProtein(prepared)
    second.checkpoints = str(tmp_path)
    second.RHF()

    assert not fock_builds
    assert second.energy == first.energy
    assert second.mean_field.converged
    assert int(second.mean_field.mo_occ.sum()) == prepared.electrons
