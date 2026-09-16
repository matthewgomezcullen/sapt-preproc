"""
Solving RHF over every pose of the ligand.

Each pose is solved alone and kept the moment it is, in the pose_scf directory of its complex. 

An ensemble resumes from the poses it kept.

7BJJ_TVW's ligand is 15 atoms and 100 basis functions, under half a second a pose. Its molecules are
    built for all forty poses; wherever one is solved, the first three stand in for the rest.
"""

import functools
import itertools
import os

import numpy as np
import pytest
from rdkit import Chem

from cutouts import prepare, read
from encode import EncodingError, SolveLigand
from prepare import PrepareComplex, PrepareError
from utils import encode, save

LIGAND = "7BJJ_TVW"

# The ligand as preparation left it: every pose carries its hydrogens, and none is charged.
LIGAND_POSES = 40
LIGAND_ATOMS = 15
LIGAND_HYDROGENS = 5
LIGAND_ELECTRONS = 70
LIGAND_FUNCTIONS = 100

POSES = 3

REPRODUCIBILE_THRESHOLD = 1e-9

DISTINCT_POSES_THRESHOLD = 1e-5


class Killed(BaseException):
    """
    A job killed partway through a pose, as the process sees it.
    """


def slice_of_poses(start, stop):
    prepared = read(LIGAND)
    prepared.out = None
    prepared.poses = prepared.poses[start:stop]
    prepared.source = prepared.source[start:stop]
    return prepared


@functools.lru_cache(maxsize=None)
def ensemble():
    return slice_of_poses(0, POSES)


@functools.lru_cache(maxsize=None)
def solve_ensemble():
    ligand = SolveLigand(ensemble())
    ligand.RHF()
    return ligand


def without_a_hydrogen(pose):
    edited = Chem.RWMol(pose) # pyright: ignore[reportAttributeAccessIssue]
    edited.RemoveAtom(next(atom.GetIdx() for atom in pose.GetAtoms() if atom.GetAtomicNum() == 1))
    return edited.GetMol()


def rhf_counted(calls, rhf):
    """
    RHF stub, counting every molecule it is handed.
    """
    return lambda mol, *args, **kwargs: calls.append(mol) or rhf(mol, *args, **kwargs)


def test_ligand_refuses_a_complex_that_was_never_prepared():
    ligand = SolveLigand(PrepareComplex("", []))

    with pytest.raises(PrepareError):
        ligand.RHF()


def test_each_pose_is_the_molecule_preparation_left():
    prepared = prepare(LIGAND)
    ligand = SolveLigand(prepared)

    ligand._molecules()

    assert len(ligand.mols) == len(prepared.poses) == LIGAND_POSES
    for mol, pose in zip(ligand.mols, prepared.poses):
        symbols = [mol.atom_symbol(index) for index in range(mol.natm)]
        assert symbols == [atom.GetSymbol() for atom in pose.GetAtoms()]
        assert len(symbols) == LIGAND_ATOMS
        assert symbols.count("H") == LIGAND_HYDROGENS
        assert mol.charge == 0
        assert mol.spin == prepared.spin
        assert mol.nelectron == LIGAND_ELECTRONS
        assert mol.basis == prepared.basis
        assert mol.nao == LIGAND_FUNCTIONS


def test_each_pose_keeps_its_own_coordinates():
    prepared = prepare(LIGAND)
    ligand = SolveLigand(prepared)

    ligand._molecules()

    for mol, pose in zip(ligand.mols, prepared.poses):
        assert np.allclose(mol.atom_coords(unit="Angstrom"), pose.GetConformer().GetPositions())


def test_a_pose_no_closed_shell_can_hold_is_refused():
    """
    A pose with an odd number of electrons cannot be paired by RHF.
    """
    prepared = slice_of_poses(0, 1)
    prepared.poses = [without_a_hydrogen(prepared.poses[0])]

    with pytest.raises(PrepareError):
        SolveLigand(prepared).RHF()


def test_rhf_converges_for_every_pose():
    ligand = solve_ensemble()

    assert ligand.solved()
    assert len(ligand.mean_fields) == POSES
    assert ligand.energies == pytest.approx([mean_field.e_tot for mean_field in ligand.mean_fields])
    for mean_field in ligand.mean_fields:
        assert mean_field.converged
        assert mean_field.e_tot < 0
        assert set(np.unique(mean_field.mo_occ)) <= {0.0, 2.0}
        assert int(mean_field.mo_occ.sum()) == LIGAND_ELECTRONS


def test_each_pose_is_solved_at_its_own_geometry():
    ligand = solve_ensemble()
    alone = SolveLigand(slice_of_poses(POSES - 1, POSES))

    alone.RHF()

    assert alone.energies[0] == pytest.approx(ligand.energies[-1], abs=REPRODUCIBILE_THRESHOLD)
    for first, second in itertools.combinations(ligand.energies, 2):
        assert abs(first - second) > DISTINCT_POSES_THRESHOLD


def test_a_pose_reaches_a_minimum_rather_than_a_saddle_point():
    """
    The converged solution is stable, so the density SAPT is handed is the ground state's.
    """
    mean_field = solve_ensemble().mean_fields[0].copy()

    internal, _ = mean_field.stability()

    assert np.allclose(internal, mean_field.mo_coeff)


def test_rhf_rejects_a_pose_whose_scf_did_not_converge():
    ligand = SolveLigand(ensemble())
    ligand.rhf_max_cycle = 1

    with pytest.raises(EncodingError):
        ligand.RHF()

    assert not ligand.solved()


def test_the_ensemble_holds_no_two_electron_integrals():
    """
    PySCF keeps a molecule's two-electron integrals on its mean field whenever they fit under
        max_memory, as this ligand's do, at 100 MB a pose. Forty poses of 200 basis functions would
        hold 65 GB, so only the solution is kept.
    """
    ligand = solve_ensemble()

    assert all(mean_field._eri is None for mean_field in ligand.mean_fields)


def test_every_pose_is_kept_and_read_back(tmp_path, monkeypatch):
    out = str(tmp_path / LIGAND)
    kept = SolveLigand(ensemble(), out)
    kept.RHF()
    calls = []
    monkeypatch.setattr(encode, "rhf", rhf_counted(calls, encode.rhf))

    read_back = SolveLigand(ensemble(), out)
    assert read_back.solved()
    read_back.RHF()

    assert not calls
    for index, (was, now) in enumerate(zip(kept.mean_fields, read_back.mean_fields)):
        assert os.path.isfile(save.pose_scf_path(LIGAND, out, index))
        assert now.e_tot == was.e_tot
        for key in ("mo_energy", "mo_occ", "mo_coeff"):
            np.testing.assert_array_equal(getattr(now, key), getattr(was, key))


def test_an_interrupted_ensemble_resumes_from_the_poses_it_kept(tmp_path, monkeypatch):
    reference = solve_ensemble().energies
    out = str(tmp_path / LIGAND)
    calls, rhf = [], encode.rhf

    def killed_on_the_second_pose(mol, *args, **kwargs):
        calls.append(mol)
        if len(calls) == 2:
            raise Killed
        return rhf(mol, *args, **kwargs)

    monkeypatch.setattr(encode, "rhf", killed_on_the_second_pose)
    with pytest.raises(Killed):
        SolveLigand(ensemble(), out).RHF()
    kept = [os.path.isfile(save.pose_scf_path(LIGAND, out, index)) for index in range(POSES)]
    assert kept == [True, False, False]

    calls.clear()
    monkeypatch.setattr(encode, "rhf", rhf_counted(calls, rhf))
    resumed = SolveLigand(ensemble(), out)
    resumed.RHF()

    assert len(calls) == POSES - 1
    assert resumed.solved()
    assert resumed.energies == pytest.approx(reference, abs=REPRODUCIBILE_THRESHOLD)
