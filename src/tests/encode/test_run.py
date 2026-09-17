"""
The driver, run.py.

Dice solves the space at the whole window, 0 <= n <= 2, and the Hamiltonian is encoded over the
    paper's, 0.02 <= n <= 1.97. The solved space stays kept at the window Dice solved it at, which any
    narrower one can be taken from.

    7BJJ_TVW       a preparation kept with none of its inputs in the test data
    6YT6_PKE       a complex whose inputs are in the test data
    ACE-VAL-NME    7BJJ_TVW's fragment, kept as a job with one pose and carried through RHF, AVAS and
                   the MP2 cap, with occupations standing in for Dice's
"""

import copy
import os
import shutil

import numpy as np
import pytest

import filter
import run
from conftest import POSEBUSTERS, PREPARED, paths
from cutouts import EXPECTED, all_carbons, fragment
from encode import EncodeProtein, SolveLigand
from prepare import PrepareComplex
from utils import save

JOB = "job"

KEPT = "7BJJ_TVW"
INPUTS = "6YT6_PKE"

# The fragment's space
NMAX = 8

OCCUPATIONS = [1.99, 1.95, 1.90, 1.60, 0.40, 0.10, 0.05, 0.01]
PAPER = (0.02, 1.97)
WINDOWED = (6, 6)  # (nelecas, ncas)


def keep(out, name, artefact=KEPT):
    directory = os.path.join(out, JOB, name)
    os.makedirs(directory)
    shutil.copyfile(
        save.prepared_path(artefact, os.path.join(PREPARED, artefact)),
        save.prepared_path(name, directory),
    )
    return directory


def entry(name=INPUTS):
    """
    `name` as filter.inventory lists it: (name, protein, poses, native).
    """
    protein, poses = paths(name)
    return name, protein, sorted(poses), os.path.join(POSEBUSTERS, name, f"{name}_ligand.sdf")


def inventory(*entries):
    """
    filter.inventory stub over `entries` alone.
    """

    def listed(named=None):
        return [one for one in entries if not named or one[0] in named], []

    return listed


def recording(calls):
    """
    PrepareComplex.prepare stub.
    """

    def prepare(self):
        calls.append((self.protein_path, list(self.poses_paths)))

    return prepare


def handed_on(calls):
    """
    Stubs for SolveLigand and EncodeProtein.
    """

    class Recorded:
        def __init__(self, prepared, out=None):
            calls.append(prepared)

        def solved(self):
            return True

        def encoded(self):
            return True

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    return Recorded


def refuse(*args, **kwargs):
    raise AssertionError("a stage that was kept ran again")


@pytest.fixture(scope="module")
def solved_job(tmp_path_factory):
    """
    ACE-VAL-NME kept as a job of its own, as far as a solved space: its one pose solved at RHF, and the
        protein through RHF, AVAS over every carbon and the MP2 cap, with OCCUPATIONS for Dice's.
    """
    out = str(tmp_path_factory.mktemp("solved"))
    directory = os.path.join(out, JOB, KEPT)

    # A copy, so the fragment the other modules share is left as it is.
    kept = copy.copy(fragment())
    kept.out = directory
    kept.poses, kept.source, kept.displacement = (
        kept.poses[:1], kept.source[:1], kept.displacement[:1]
    )
    kept.save()

    prepared = PrepareComplex("", [], directory)
    SolveLigand(prepared, directory).RHF()
    protein = EncodeProtein(prepared, directory)
    protein.RHF()
    protein.AVAS(targets=all_carbons(protein.mol))
    protein.nmax = NMAX
    protein.MP2()
    assert (protein.active_electrons, protein.active_space_size) == (NMAX, NMAX)
    protein.shci_energy = protein.energy + protein.correlation
    protein.occupations = np.array(OCCUPATIONS)
    protein.save()
    return out


@pytest.fixture(scope="module")
def encoded_job(solved_job, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("encoded"))
    shutil.copytree(solved_job, out, dirs_exist_ok=True)

    run.run(JOB, out, complexes=[KEPT])

    return out


def test_a_kept_preparation_is_read_back_without_its_inputs(tmp_path, monkeypatch):
    keep(str(tmp_path), KEPT)
    prepared, handed = [], []
    monkeypatch.setattr(filter, "inventory", refuse)
    monkeypatch.setattr(PrepareComplex, "prepare", recording(prepared))
    monkeypatch.setattr(run, "SolveLigand", handed_on(handed))
    monkeypatch.setattr(run, "EncodeProtein", handed_on(handed))

    run.run(JOB, str(tmp_path), complexes=[KEPT])

    assert not prepared
    assert handed
    for given in handed:
        assert given.prepared()
        assert (given.heavy_atoms, given.charge, given.electrons) == EXPECTED[KEPT]


@pytest.mark.parametrize(
    "kept, force", [(False, False), (True, True)], ids=["nothing-kept", "forced"]
)
def test_a_complex_that_has_to_be_prepared_is_prepared_from_its_inputs(
    tmp_path, monkeypatch, kept, force
):
    if kept:
        keep(str(tmp_path), INPUTS)
    prepared = []
    monkeypatch.setattr(filter, "inventory", inventory(entry()))
    monkeypatch.setattr(PrepareComplex, "prepare", recording(prepared))

    run.run(JOB, str(tmp_path), complexes=[INPUTS], force=force, prepare_only=True)

    _, protein, poses, _ = entry()
    assert prepared == [(protein, poses)]


def test_a_complex_neither_kept_nor_in_the_benchmark_set_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(filter, "inventory", inventory())
    monkeypatch.setattr(PrepareComplex, "prepare", refuse)

    with pytest.raises(SystemExit):
        run.run(JOB, str(tmp_path), complexes=[INPUTS], prepare_only=True)


def test_the_hamiltonian_is_encoded_over_the_papers_window(encoded_job):
    encoded = save.load_encoded(KEPT, os.path.join(encoded_job, JOB, KEPT))

    lo, hi = PAPER
    nelecas, ncas = WINDOWED
    assert (encoded["active_electrons"], encoded["active_space_size"]) == WINDOWED
    np.testing.assert_allclose(encoded["occupations"], [n for n in OCCUPATIONS if lo <= n <= hi])
    assert encoded["h1"].shape == (ncas, ncas)
    assert encoded["h2"].shape == (ncas,) * 4


def test_the_solved_space_stays_at_the_window_dice_solved(encoded_job):
    solved = save.load_solved(KEPT, os.path.join(encoded_job, JOB, KEPT))

    assert (solved["active_electrons"], solved["active_space_size"]) == (NMAX, NMAX)
    np.testing.assert_allclose(solved["occupations"], OCCUPATIONS)


def test_a_finished_complex_is_read_back_rather_than_run_again(encoded_job, tmp_path, monkeypatch):
    shutil.copytree(encoded_job, str(tmp_path), dirs_exist_ok=True)
    directory = os.path.join(str(tmp_path), JOB, KEPT)
    before = save.load_encoded(KEPT, directory)
    monkeypatch.setattr(filter, "inventory", refuse)
    monkeypatch.setattr(PrepareComplex, "prepare", refuse)
    monkeypatch.setattr(SolveLigand, "RHF", refuse)
    for stage in ("RHF", "AVAS", "MP2", "SHCI", "H"):
        monkeypatch.setattr(EncodeProtein, stage, refuse)

    run.run(JOB, str(tmp_path), complexes=[KEPT])

    after = save.load_encoded(KEPT, directory)
    assert after.keys() == before.keys()
    for key, value in before.items():
        np.testing.assert_array_equal(after[key], value)
