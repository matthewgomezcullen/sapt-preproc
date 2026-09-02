"""
The driver that runs the pipeline over one complex and keeps what it produces.

These are about what is written and what is read back, not about the physics.

`run` itself reaches Dice, so it is marked. Nothing above it is.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

import run
from cutouts import FRAGMENT, SUBSET, all_carbons, fragment
from encode import EncodingError

# The window the driver solves under. Truncating is arithmetic on the occupations, so one stored 
# run answers for any window.
EVERYTHING = (0.0, 2.0)

# Small enough that the cap has to cut, and quick enough to run here.
FRAGMENT_NMAX = 8


def finished(ncas=4, nao=12):
    """
    An encoder that has been the whole way.
    """
    return SimpleNamespace(
        active_space_size=ncas,
        active_electrons=ncas,
        orbital_initial=np.arange(nao * nao, dtype=float).reshape(nao, nao),
        occupations=np.linspace(1.99, 0.01, ncas),
        energy=-570.5,
        shci_energy=-570.52,
        correlation=-0.31,
    )


def test_the_results_follow_data(monkeypatch):
    """
    A run on the cluster belongs beside the checkpoints it was built from.
    """
    monkeypatch.setenv("DATA", "/data/somewhere")

    assert run.spaces() == os.path.join("/data/somewhere", "spaces")


def test_the_results_land_locally_without_data(monkeypatch):
    """
    Off the cluster there is still somewhere to put them, beside the screen's own table.
    """
    monkeypatch.delenv("DATA", raising=False)

    assert run.spaces() == os.path.join(run.OUT, "spaces")


def test_each_complex_gets_its_own_file():
    """
    The array runs one task per complex, and they don't land on each other.
    """
    assert run.path("7USH_82V", "/out") != run.path("7W06_ITN", "/out")


def test_a_complex_resolves_from_the_tracked_fixtures():
    """
    The full benchmark set is not tracked, so a fresh clone has only these. The cluster is a fresh
        clone, and resolving only against `src/data` is why the first run of the driver died.
    """
    tracked = os.path.join(run.ROOT, "tests", "data")

    for name in SUBSET:
        found = run.find(name, roots=(tracked,))
        assert os.path.isfile(found.protein_path)
        assert found.poses_paths


def test_a_complex_kept_beside_its_own_poses_resolves():
    """
    The fragment is stored whole rather than split across the two benchmark directories.
    """
    tracked = os.path.join(run.ROOT, "tests", "data")

    found = run.find(FRAGMENT, roots=(tracked,))

    assert os.path.isfile(found.protein_path)
    assert found.poses_paths


def test_the_first_root_holding_a_complex_wins(tmp_path):
    """
    A machine with the whole set downloaded uses it; a clone falls back to what it has.
    """
    tracked = os.path.join(run.ROOT, "tests", "data")

    found = run.find(SUBSET[0], roots=(str(tmp_path), tracked))

    assert found.protein_path.startswith(tracked)


def test_a_complex_that_is_nowhere_says_where_it_looked(tmp_path):
    """
    The failure this replaces was a FileNotFoundError out of `os.listdir`, naming one directory
        that the reader had no reason to expect.
    """
    with pytest.raises(EncodingError) as raised:
        run.find("not_a_complex", roots=(str(tmp_path),))

    assert str(tmp_path) in str(raised.value)


def test_a_root_that_does_not_exist_is_passed_over(tmp_path):
    """
    `src/data` is absent wherever the set has not been downloaded, which is the ordinary case.
    """
    tracked = os.path.join(run.ROOT, "tests", "data")

    found = run.find(SUBSET[0], roots=(str(tmp_path / "nothing" / "here"), tracked))

    assert found.poses_paths


def test_what_is_saved_is_what_comes_back(tmp_path):
    """
    Every array survives the round trip exactly.
    """
    written = finished()
    path = run.path(FRAGMENT, str(tmp_path))

    run.save(written, FRAGMENT, path)
    read = run.load(path)

    assert np.array_equal(read["orbitals"], written.orbital_initial)
    assert np.array_equal(read["occupations"], written.occupations)


def test_what_is_saved_is_enough_to_go_on_with(tmp_path):
    """
    The active space, the energies behind it, and which complex it belongs to.
    """
    written = finished()
    path = run.path(FRAGMENT, str(tmp_path))

    run.save(written, FRAGMENT, path)
    read = run.load(path)

    assert read["name"] == FRAGMENT
    assert read["ncas"] == written.active_space_size
    assert read["nelecas"] == written.active_electrons
    assert read["energy"] == written.energy
    assert read["energy_cas"] == written.shci_energy
    assert read["correlation"] == written.correlation


def test_the_window_it_was_solved_under_is_recorded(tmp_path):
    """
    A reader has to know the occupations were never truncated, or it cannot re-window them.
    """
    path = run.path(FRAGMENT, str(tmp_path))

    run.save(finished(), FRAGMENT, path)

    assert tuple(run.load(path)["window"]) == EVERYTHING


def test_the_numbers_come_back_as_numbers(tmp_path):
    """
    A scalar out of an npz is a zero-dimensional array, which compares oddly and prints worse.
    """
    path = run.path(FRAGMENT, str(tmp_path))
    run.save(finished(), FRAGMENT, path)

    read = run.load(path)

    assert isinstance(read["ncas"], int)
    assert isinstance(read["energy"], float)
    assert isinstance(read["name"], str)


def test_the_directory_is_made_if_it_is_not_there(tmp_path):
    """
    The first task to finish creates the dictionary.
    """
    out = str(tmp_path / "not" / "yet")

    run.save(finished(), FRAGMENT, run.path(FRAGMENT, out))

    assert os.path.isdir(out)


def test_an_unfinished_run_is_not_saved(tmp_path):
    """
    A space that never reached Dice is not a result.
    """
    unfinished = finished()
    unfinished.shci_energy = None

    with pytest.raises(EncodingError):
        run.save(unfinished, FRAGMENT, run.path(FRAGMENT, str(tmp_path)))


def test_a_finished_complex_is_left_alone(tmp_path):
    """
    A rerun does not recompute.
    """
    out = str(tmp_path)
    run.save(finished(), FRAGMENT, run.path(FRAGMENT, out))

    assert run.done(FRAGMENT, out)


def test_an_unstarted_complex_is_not_finished(tmp_path):
    assert not run.done(FRAGMENT, str(tmp_path))


def test_a_result_that_cannot_be_read_is_not_finished(tmp_path):
    """
    A file cut off mid-write is not a result.
    """
    out = str(tmp_path)
    os.makedirs(out, exist_ok=True)
    with open(run.path(FRAGMENT, out), "wb") as handle:
        handle.write(b"not an npz")

    assert not run.done(FRAGMENT, out)


@pytest.mark.dice
def test_the_driver_carries_a_complex_the_whole_way(tmp_path):
    """
    Every step, and a file at the end of it holding what the steps produced.
    """
    out = str(tmp_path)

    result = run.run(
        FRAGMENT, out, prepared=fragment(), targets=all_carbons, nmax=FRAGMENT_NMAX
    )

    assert run.done(FRAGMENT, out)
    read = run.load(run.path(FRAGMENT, out))
    assert read["ncas"] == result.active_space_size == FRAGMENT_NMAX
    assert read["orbitals"].shape[0] == result.mol.nao
    assert len(read["occupations"]) == result.active_space_size
    assert read["energy_cas"] < read["energy"]


@pytest.mark.dice
def test_the_whole_window_is_kept(tmp_path):
    """
    The driver truncates nothing.
    """
    out = str(tmp_path)

    run.run(FRAGMENT, out, prepared=fragment(), targets=all_carbons, nmax=FRAGMENT_NMAX)

    occupations = run.load(run.path(FRAGMENT, out))["occupations"]
    assert len(occupations) == FRAGMENT_NMAX
    assert occupations.max() > 1.97  # a paper window would have dropped this one
    assert occupations.min() < 0.02  # and this one


@pytest.mark.dice
def test_a_second_run_reads_rather_than_solves(tmp_path):
    """
    Resumes previous jobs.
    """
    out = str(tmp_path)
    run.run(FRAGMENT, out, prepared=fragment(), targets=all_carbons, nmax=FRAGMENT_NMAX)

    again = run.run(
        FRAGMENT, out, prepared=fragment(), targets=all_carbons, nmax=FRAGMENT_NMAX
    )

    assert again is None


@pytest.mark.dice
def test_force_solves_it_again(tmp_path):
    """
    Alternative force to not resume previous runs. 
    """
    out = str(tmp_path)
    run.run(FRAGMENT, out, prepared=fragment(), targets=all_carbons, nmax=FRAGMENT_NMAX)

    again = run.run(
        FRAGMENT, out, prepared=fragment(), targets=all_carbons, nmax=FRAGMENT_NMAX, force=True
    )

    assert again is not None
    assert again.active_space_size == FRAGMENT_NMAX


def test_dices_log_is_copied_out_of_the_scratch(tmp_path):
    """
    Dice's log is in scratch and thrown away. Copy this over.
    """
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "output.dat").write_text("what Dice said")
    destination = str(tmp_path / "kept.out")

    run.keep(str(scratch), destination)

    assert open(destination).read() == "what Dice said"


def test_a_missing_log_is_not_an_error(tmp_path):
    """
    Dice can raise before writing anything, which should not cause more errors.
    """
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    destination = str(tmp_path / "kept.out")

    run.keep(str(scratch), destination)

    assert not os.path.exists(destination)


def test_the_log_sits_beside_the_result(tmp_path):
    """
    Logs sit with the complex.
    """
    out = str(tmp_path)

    assert os.path.dirname(run.log(FRAGMENT, out)) == os.path.dirname(run.path(FRAGMENT, out))


@pytest.mark.dice
def test_the_run_says_what_it_is_doing(tmp_path, capfd):
    """
    Log contents are correct.
    """
    run.run(
        FRAGMENT, str(tmp_path), prepared=fragment(), targets=all_carbons, nmax=FRAGMENT_NMAX
    )

    printed = capfd.readouterr().out
    for step in ("RHF", "AVAS", "MP2", "SHCI"):
        assert step in printed


@pytest.mark.dice
def test_the_run_reports_the_space_it_reached(tmp_path, capfd):
    """
    Log the active space.
    """
    result = run.run(
        FRAGMENT, str(tmp_path), prepared=fragment(), targets=all_carbons, nmax=FRAGMENT_NMAX
    )

    printed = capfd.readouterr().out
    assert f"{result.active_electrons}e" in printed
    assert f"{result.active_space_size}o" in printed


@pytest.mark.dice
def test_a_run_is_quiet_by_default(tmp_path):
    """
    Default does not log the SCF table.
    """
    result = run.run(
        FRAGMENT, str(tmp_path), prepared=fragment(), targets=all_carbons, nmax=FRAGMENT_NMAX
    )

    assert result.mol.verbose == 0
    assert result.mean_field.verbose == 0


@pytest.mark.dice
def test_verbosity_reaches_pyscf(tmp_path):
    """
    Asserted on the objects, not the output: PySCF binds its stream at import, before pytest's
        capture is in place, so neither capsys nor capfd can read back what it prints.
    """
    result = run.run(
        FRAGMENT,
        str(tmp_path),
        prepared=fragment(),
        targets=all_carbons,
        nmax=FRAGMENT_NMAX,
        verbose=4,
    )

    assert result.verbose == 4
    assert result.mol.verbose == 4
    assert result.mean_field.verbose == 4


@pytest.mark.dice
def test_a_finished_run_keeps_dices_log_and_nothing_else(tmp_path):
    """
    The integrals and the wavefunction should not be kept.
    """
    out = str(tmp_path)

    result = run.run(
        FRAGMENT, out, prepared=fragment(), targets=all_carbons, nmax=FRAGMENT_NMAX
    )

    assert os.path.isfile(run.log(FRAGMENT, out))
    assert not os.path.exists(result.scratch)


# --------------------------------------------------------------------------------------------
# The bin. Hours each, for the cluster.
# --------------------------------------------------------------------------------------------


@pytest.mark.dice
@pytest.mark.hpc_long_run
@pytest.mark.parametrize("name", SUBSET)
def test_the_driver_carries_a_cutout_of_the_bin_the_whole_way(tmp_path, name):
    """
    What the cluster is being asked to do, and what it leaves behind.
    """
    out = str(tmp_path)

    result = run.run(name, out)

    read = run.load(run.path(name, out))
    assert read["ncas"] == result.active_space_size == result.nmax
    assert read["orbitals"].shape[0] == result.mol.nao
    assert 0 < read["nelecas"] < 2 * read["ncas"]
