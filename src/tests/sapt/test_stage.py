"""
The SAPT stage, sapt.py's SAPT, over the water dimer: the compressed monomer, correlated over
    (6e, 6o), stands in for the protein, and the equilibrium monomer for a pose, beside a copy of it
    pulled 100 A away.

Once every pose has all of its scores, the stage keeps them as <complex>_sapt.npz in the complex's
    directory, each against the file its pose came from. A stage handed that directory again reads
    them back rather than scoring the poses again.
"""

import shutil
from types import SimpleNamespace

import numpy as np
import pytest

import monomers
from sapt import SAPT
from utils import sapt, save

NAME = "WATER_DIMER"

NEAR, FAR = "rank1_confidence0.50.sdf", "rank2_confidence-0.25.sdf"

SCORES = ["electrostatics", "exchanges", "cumulants", "int_energies"]

# The tables print six decimal places, so half of the last one is the closest a comparison gets.
PUBLISHED = 1e-6  # Hartree

EXACT = 1e-12

# The far pose barely interacts. Hartree.
APART = 1e-5


def protein():
    rdm1, rdm2 = monomers.correlate_water(monomers.COMPRESSED)
    return SimpleNamespace(
        mol=monomers.build_water(monomers.COMPRESSED),
        orbital_initial=monomers.solve_water(monomers.COMPRESSED).mo_coeff,
        rdm1=rdm1,
        rdm2=rdm2,
        active_space_size=monomers.WATER_NCAS,
        active_electrons=monomers.WATER_NELECAS,
        correlated=lambda: True,
    )


def ligand():
    poses = [monomers.EQUILIBRIUM, monomers.pulled_apart(monomers.EQUILIBRIUM, monomers.SEPARATION)]
    return SimpleNamespace(
        mols=[monomers.build_water(pose) for pose in poses],
        mean_fields=[monomers.solve_water(pose) for pose in poses],
        prepared=SimpleNamespace(source=[NEAR, FAR]),
        solved=lambda: True,
    )


def refuse(*args, **kwargs):
    raise AssertionError("a kept pose was scored again")


@pytest.fixture(scope="module")
def scored_stage(tmp_path_factory):
    stage = SAPT(protein(), ligand(), str(tmp_path_factory.mktemp("scored") / NAME))
    stage.interaction()
    return stage


def test_a_pose_is_scored_as_table_iv_scores_it(scored_stage):
    assert scored_stage.electrostatics[0] == pytest.approx(monomers.ELST_CORRELATED, abs=PUBLISHED)
    assert scored_stage.exchanges[0] == pytest.approx(monomers.EXCH_CORRELATED, abs=PUBLISHED)


def test_the_cumulants_share_is_in_the_exchange_and_kept_apart(scored_stage):
    separable = sapt.exchange(
        monomers.build_water(monomers.COMPRESSED),
        monomers.correlated_water_density(monomers.COMPRESSED),
        monomers.build_water(monomers.EQUILIBRIUM),
        monomers.solve_water(monomers.EQUILIBRIUM).make_rdm1(),
    )

    assert scored_stage.cumulants[0] != 0
    assert scored_stage.exchanges[0] - scored_stage.cumulants[0] == pytest.approx(
        separable, abs=EXACT
    )


def test_each_pose_is_scored_apart(scored_stage):
    for key in SCORES:
        assert len(getattr(scored_stage, key)) == 2
    np.testing.assert_allclose(
        scored_stage.int_energies,
        np.add(scored_stage.electrostatics, scored_stage.exchanges),
        rtol=0,
        atol=EXACT,
    )
    near, far = scored_stage.int_energies
    assert abs(far) < APART < abs(near)


def test_the_scores_are_kept_once_every_pose_has_them(scored_stage):
    record = save.load_sapt(NAME, scored_stage.out)

    assert list(record["source"]) == [NEAR, FAR]
    for key in SCORES:
        np.testing.assert_array_equal(record[key], getattr(scored_stage, key))


@pytest.mark.parametrize("term", ["elst", "exch"])
def test_nothing_is_kept_before_every_score_is_in(tmp_path, term):
    stage = SAPT(protein(), ligand(), str(tmp_path / NAME))

    getattr(stage, term)()
    stage.save()

    assert not stage.scored()
    assert save.load_sapt(NAME, stage.out) is None


def test_a_kept_record_is_read_back_rather_than_scored_again(scored_stage, tmp_path, monkeypatch):
    out = str(tmp_path / NAME)
    shutil.copytree(scored_stage.out, out)
    for term in ("density", "electrostatics", "exchange", "cumulant_exchange"):
        monkeypatch.setattr(sapt, term, refuse)

    stage = SAPT(protein(), ligand(), out)

    assert stage.scored()
    assert stage.interaction() == pytest.approx(scored_stage.int_energies, abs=EXACT)
    for key in SCORES:
        np.testing.assert_array_equal(getattr(stage, key), getattr(scored_stage, key))
