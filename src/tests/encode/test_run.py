"""
The driver that runs the pipeline over one complex and keeps what it produces.

These are about what is written and what is read back, not about the physics.

The file is written in two halves: the space Dice returned, then the integrals for the Hamiltonian.

`run` itself reaches Dice, so it is marked. Nothing above it is.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import gto

import run
from cutouts import FRAGMENT, SUBSET, all_carbons, fragment
from encode import EncodingError
from utils import encode

# The window the driver solves under. Truncating is arithmetic on the occupations, so one stored
# run answers for any window.
EVERYTHING = (0.0, 2.0)

# The window the Hamiltonian is built over.
PAPER = (0.02, 1.97)

# What the fragment is mapped at instead.
NARROW, NARROWED = (0.015, 1.985), (4, 4)  # (lo, hi), (nelecas, ncas)

# Small enough that the cap has to cut, and quick enough to run here.
FRAGMENT_NMAX = 8

# The stand-in's active space. Two electrons in two of its four orbitals, so it is closed-shell and
# has an excitation in it.
NCAS = 2
NELECAS = 2

# The stand-in's core energy.
E_CORE = -569.75

POSES = ["somewhere/rank1_confidence-0.71.sdf", "somewhere/rank2_confidence-0.77.sdf"]


def molecule():
    """
    Returns a molecule.
    """
    return gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31G", verbose=0)


def finished_shci():
    """
    An encoder that has been through SHCI.
    """
    mol = molecule()
    return SimpleNamespace(
        mol=mol,
        mean_field=None,
        active_space_size=NCAS,
        active_electrons=NELECAS,
        orbital_initial=np.arange(mol.nao * mol.nao, dtype=float).reshape(mol.nao, mol.nao),
        occupations=np.linspace(1.99, 0.01, NCAS),
        energy=-570.5,
        shci_energy=-570.52,
        correlation=-0.31,
        prepared=SimpleNamespace(poses_paths=list(POSES)),
    )


def finished():
    """
    An encoder that has been through its Hamiltonian.

    """
    written = finished_shci()
    written.e_core = E_CORE
    written.h1 = np.arange(NCAS * NCAS, dtype=float).reshape(NCAS, NCAS)
    written.h2 = np.arange(NCAS ** 4, dtype=float).reshape((NCAS,) * 4)
    return written


def test_the_results_follow_data(monkeypatch):
    """
    Results are put in the $DATA directory.
    """
    monkeypatch.setenv("DATA", "/data/somewhere")

    assert run.spaces() == os.path.join("/data/somewhere", "spaces")


def test_the_results_land_locally_without_data(monkeypatch):
    """
    By default, results are put in OUT.
    """
    monkeypatch.delenv("DATA", raising=False)

    assert run.spaces() == os.path.join(run.OUT, "spaces")


def test_each_complex_gets_its_own_file():
    """
    The array runs one task per complex, and each have their own output dir.
    """
    assert run.path("7USH_82V", "/out") != run.path("7W06_ITN", "/out")


def test_a_complex_resolves_from_the_tracked_fixtures():
    """
    The full benchmark set is not tracked, so a fresh clone has only those in `tests`.
    """
    tracked = os.path.join(run.ROOT, "tests", "data")

    for name in SUBSET:
        found = run.find(name, roots=(tracked,))
        assert os.path.isfile(found.protein_path)
        assert found.poses_paths


def test_a_complex_kept_beside_its_own_poses_resolves():
    """
    Fragment is found.
    """
    tracked = os.path.join(run.ROOT, "tests", "data")

    found = run.find(FRAGMENT, roots=(tracked,))

    assert os.path.isfile(found.protein_path)
    assert found.poses_paths


def test_the_first_root_holding_a_complex_wins(tmp_path):
    """
    Prioritise data over tests.
    """
    tracked = os.path.join(run.ROOT, "tests", "data")

    found = run.find(SUBSET[0], roots=(str(tmp_path), tracked))

    assert found.protein_path.startswith(tracked)


def test_a_complex_that_is_nowhere_says_where_it_looked(tmp_path):
    """
    Raise appropriate EncodingError.
    """
    with pytest.raises(EncodingError) as raised:
        run.find("not_a_complex", roots=(str(tmp_path),))

    assert str(tmp_path) in str(raised.value)


def test_a_root_that_does_not_exist_is_passed_over(tmp_path):
    """
    Pass over empty data dir.
    """
    tracked = os.path.join(run.ROOT, "tests", "data")

    found = run.find(SUBSET[0], roots=(str(tmp_path / "nothing" / "here"), tracked))

    assert found.poses_paths


def test_what_is_saved_is_what_comes_back(tmp_path):
    """
    Fragment loaded matches the original.
    """
    written = finished_shci()
    path = run.path(FRAGMENT, str(tmp_path))

    run.save(written, FRAGMENT, path)
    read = run.load(path)

    assert np.array_equal(read["orbitals"], written.orbital_initial)
    assert np.array_equal(read["occupations"], written.occupations)


def test_what_is_saved_is_enough_to_go_on_with(tmp_path):
    """
    The active space, the energies behind it, the integrals SAPT is handed, and which complex it
        all belongs to.
    """
    written = finished()
    path = run.path(FRAGMENT, str(tmp_path))

    run.save(written, FRAGMENT, path)
    run.finish(written, path, EVERYTHING)
    read = run.load(path)

    assert read["name"] == FRAGMENT
    assert read["ncas"] == written.active_space_size
    assert read["nelecas"] == written.active_electrons
    assert read["energy"] == written.energy
    assert read["energy_cas"] == written.shci_energy
    assert read["correlation"] == written.correlation
    assert read["e_core"] == written.e_core
    assert np.array_equal(read["h1"], written.h1)
    assert np.array_equal(read["h2"], written.h2)
    assert read["hamiltonian_ncas"] == written.active_space_size
    assert read["hamiltonian_nelecas"] == written.active_electrons
    assert tuple(read["hamiltonian_window"]) == EVERYTHING


def test_the_window_it_was_solved_under_is_recorded(tmp_path):
    """
    Store the window.
    """
    path = run.path(FRAGMENT, str(tmp_path))

    run.save(finished_shci(), FRAGMENT, path)

    assert tuple(run.load(path)["window"]) == EVERYTHING


def test_the_numbers_come_back_as_numbers(tmp_path):
    """
    Correct data types stored.
    """
    path = run.path(FRAGMENT, str(tmp_path))
    run.save(finished_shci(), FRAGMENT, path)

    read = run.load(path)

    assert isinstance(read["ncas"], int)
    assert isinstance(read["energy"], float)
    assert isinstance(read["name"], str)


def test_the_directory_is_made_if_it_is_not_there(tmp_path):
    """
    The first task to finish creates the dictionary.
    """
    out = str(tmp_path / "not" / "yet")

    run.save(finished_shci(), FRAGMENT, run.path(FRAGMENT, out))

    assert os.path.isdir(out)


def test_an_unfinished_run_is_not_saved(tmp_path):
    """
    Neither half is written before the step behind it has produced anything.
    """
    path = run.path(FRAGMENT, str(tmp_path))
    unfinished = finished_shci()
    unfinished.shci_energy = None
    unmapped = finished()
    unmapped.h2 = None

    with pytest.raises(EncodingError):
        run.save(unfinished, FRAGMENT, path)

    run.save(finished_shci(), FRAGMENT, path)
    with pytest.raises(EncodingError):
        run.finish(unmapped, path, EVERYTHING)


def test_a_finished_complex_is_left_alone(tmp_path):
    """
    A rerun does not recompute, and a run that stopped after SHCI is solved but not finished.
    """
    out = str(tmp_path)
    written = finished()
    path = run.path(FRAGMENT, out)

    run.save(written, FRAGMENT, path)
    assert run.solved(FRAGMENT, out)
    assert not run.done(FRAGMENT, out)

    run.finish(written, path, EVERYTHING)
    assert run.done(FRAGMENT, out)


def test_an_unstarted_complex_is_not_finished(tmp_path):
    assert not run.done(FRAGMENT, str(tmp_path))


def test_a_result_that_cannot_be_read_is_not_finished(tmp_path):
    """
    An unreadble file is not used.
    """
    out = str(tmp_path)
    os.makedirs(out, exist_ok=True)
    with open(run.path(FRAGMENT, out), "wb") as handle:
        handle.write(b"not an npz")

    assert not run.done(FRAGMENT, out)


def test_the_molecule_is_stored(tmp_path):
    """
    The geometry, basis, charge, spin and atom ordering are stored.
    """
    written = finished_shci()
    path = run.path(FRAGMENT, str(tmp_path))

    run.save(written, FRAGMENT, path)

    assert run.load(path)["molecule"] == written.mol.dumps()


def test_the_stored_molecule_rebuilds(tmp_path):
    """
    Rebuild stored molecule correctly.
    """
    written = finished_shci()
    path = run.path(FRAGMENT, str(tmp_path))
    run.save(written, FRAGMENT, path)

    rebuilt = gto.loads(run.load(path)["molecule"])

    assert rebuilt.nao == written.mol.nao
    assert rebuilt.nelectron == written.mol.nelectron
    assert np.allclose(rebuilt.atom_coords(), written.mol.atom_coords())


def test_the_checkpoint_digest_is_stored(tmp_path):
    """
    The checkpoint digest is stored correctly.
    """
    written = finished_shci()
    path = run.path(FRAGMENT, str(tmp_path))

    run.save(written, FRAGMENT, path)

    assert run.load(path)["digest"] == encode.digest(written.mol)


def test_the_digest_survives_a_rebuild():
    """
    A molecule through dumps and loads keeps its digest.
    """
    mol = molecule()

    assert encode.digest(gto.loads(mol.dumps())) == encode.digest(mol)


def test_the_poses_are_stored(tmp_path):
    """
    Which poses defined the cutout. Monomer B comes from these.
    """
    path = run.path(FRAGMENT, str(tmp_path))

    run.save(finished_shci(), FRAGMENT, path)

    assert list(run.load(path)["poses"]) == POSES


def test_a_stored_run_resumes_without_preparing_anything(tmp_path):
    """
    A result holding its molecule doesn't need perparation.
    """
    out = str(tmp_path)
    written = finished_shci()
    run.save(written, FRAGMENT, run.path(FRAGMENT, out))

    resumed = run.resume(FRAGMENT, out)

    assert resumed.active_space_size == NCAS
    assert resumed.active_electrons == NELECAS
    assert resumed.mol.nao == written.mol.nao
    assert np.array_equal(resumed.orbital_initial, written.orbital_initial)


def test_a_resumed_run_reaches_its_hamiltonian(tmp_path):
    """
    A resumed run produces its hamiltonian.
    """
    out = str(tmp_path)
    run.save(finished_shci(), FRAGMENT, run.path(FRAGMENT, out))

    resumed = run.resume(FRAGMENT, out)

    assert resumed.H().num_qubits == 2 * NCAS


def test_resuming_a_complex_that_was_never_run_says_so(tmp_path):
    with pytest.raises(EncodingError):
        run.resume(FRAGMENT, str(tmp_path))


@pytest.mark.dice
def test_the_driver_carries_a_complex_the_whole_way(tmp_path):
    """
    Every step, and a file at the end of it holding the space and the Hamiltonian built over it.
    """
    out = str(tmp_path)
    nelecas, ncas = NARROWED

    result = run.run(
        FRAGMENT, out, prepared=fragment(), targets=all_carbons, nmax=FRAGMENT_NMAX,
        window=NARROW,
    )

    assert run.done(FRAGMENT, out)
    read = run.load(run.path(FRAGMENT, out))
    assert read["ncas"] == FRAGMENT_NMAX
    assert read["orbitals"].shape[0] == result.mol.nao
    assert len(read["occupations"]) == FRAGMENT_NMAX
    assert read["energy_cas"] < read["energy"]

    assert (result.active_electrons, result.active_space_size) == (nelecas, ncas)
    assert (read["hamiltonian_nelecas"], read["hamiltonian_ncas"]) == (nelecas, ncas)
    assert tuple(read["hamiltonian_window"]) == NARROW
    assert read["h1"].shape == (ncas, ncas)
    assert read["h2"].shape == (ncas,) * 4


@pytest.mark.dice
def test_the_whole_window_is_kept(tmp_path):
    """
    The driver truncates nothing it stores; the narrowing the Hamiltonian needs is not written
        back over it.
    """
    out = str(tmp_path)

    run.run(
        FRAGMENT, out, prepared=fragment(), targets=all_carbons, nmax=FRAGMENT_NMAX,
        window=NARROW,
    )

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
    run.run(
        FRAGMENT, out, prepared=fragment(), targets=all_carbons, nmax=FRAGMENT_NMAX,
        window=NARROW,
    )

    again = run.run(
        FRAGMENT, out, prepared=fragment(), targets=all_carbons, nmax=FRAGMENT_NMAX,
        window=NARROW,
    )

    assert again is None


@pytest.mark.dice
def test_force_solves_it_again(tmp_path):
    """
    Alternative force to not resume previous runs. 
    """
    out = str(tmp_path)
    run.run(
        FRAGMENT, out, prepared=fragment(), targets=all_carbons, nmax=FRAGMENT_NMAX,
        window=NARROW,
    )

    again = run.run(
        FRAGMENT, out, prepared=fragment(), targets=all_carbons, nmax=FRAGMENT_NMAX,
        window=NARROW, force=True,
    )

    assert again is not None
    assert again.active_space_size == NARROWED[1]


@pytest.mark.dice
def test_a_run_that_stopped_before_its_hamiltonian_picks_up_where_it_left_off(tmp_path, capfd):
    """
    Banking after SHCI only pays if the next attempt reads it rather than paying Dice again.

    No `prepared`, and the roots point nowhere, so a run that reached for the complex at all would
        raise rather than quietly prepare it a second time.
    """
    out = str(tmp_path)
    path = run.path(FRAGMENT, out)
    run.run(
        FRAGMENT, out, prepared=fragment(), targets=all_carbons, nmax=FRAGMENT_NMAX,
        window=NARROW,
    )
    stored = run.load(path)
    for key in run.HAMILTONIAN:
        stored.pop(key)
    np.savez(path, **stored)
    assert not run.done(FRAGMENT, out)
    capfd.readouterr()

    again = run.run(FRAGMENT, out, window=NARROW, roots=(str(tmp_path / "nothing"),))

    printed = capfd.readouterr().out
    assert again is not None
    assert run.done(FRAGMENT, out)
    assert "RHF" not in printed
    assert "SHCI" not in printed


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
        FRAGMENT, str(tmp_path), prepared=fragment(), targets=all_carbons, nmax=FRAGMENT_NMAX,
        window=NARROW,
    )

    printed = capfd.readouterr().out
    # "integrals" rather than "H", which is a substring of "RHF", and rather than the qubits,
    # which the step builds and does not keep.
    for step in ("RHF", "AVAS", "MP2", "SHCI", "WINDOW", "integrals"):
        assert step in printed


@pytest.mark.dice
def test_the_run_reports_the_space_it_reached(tmp_path, capfd):
    """
    Log the active space.
    """
    result = run.run(
        FRAGMENT, str(tmp_path), prepared=fragment(), targets=all_carbons, nmax=FRAGMENT_NMAX,
        window=NARROW,
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
        FRAGMENT, str(tmp_path), prepared=fragment(), targets=all_carbons, nmax=FRAGMENT_NMAX,
        window=NARROW,
    )

    assert result.mol.verbose == 0
    assert result.mean_field.verbose == 0


@pytest.mark.dice
def test_verbosity_reaches_pyscf(tmp_path):
    """
    Asserted on the objects, because PySCF binds its stream at import, before pytest's capture is 
        in place, so we cannot read back what it prints.
    """
    result = run.run(
        FRAGMENT,
        str(tmp_path),
        prepared=fragment(),
        targets=all_carbons,
        nmax=FRAGMENT_NMAX,
        window=NARROW,
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
        FRAGMENT, out, prepared=fragment(), targets=all_carbons, nmax=FRAGMENT_NMAX,
        window=NARROW,
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
    Full run on the subset of the bin, at the paper's window, which a real cutout survives where
        the saturated fragment does not.
    """
    out = str(tmp_path)

    result = run.run(name, out)

    read = run.load(run.path(name, out))
    assert read["ncas"] == result.nmax
    assert read["orbitals"].shape[0] == result.mol.nao
    assert 0 < read["nelecas"] < 2 * read["ncas"]
    assert tuple(read["hamiltonian_window"]) == PAPER

    ncas = read["hamiltonian_ncas"]
    assert ncas == result.active_space_size < read["ncas"]
    assert 0 < read["hamiltonian_nelecas"] < 2 * ncas
    assert read["h1"].shape == (ncas, ncas)
    assert read["h2"].shape == (ncas,) * 4
