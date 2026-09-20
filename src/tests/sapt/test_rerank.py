"""
Ranking each complex's poses on their first order interaction energy, beside the ranking confidence.py
    gave them.

The rows are confidence.csv's. Each is given its pose's energies, keyed by the complex and the file
    the pose came from, as confidence.py keys its scores, and each complex's poses are ranked on
    E_elst + E_exch, lowest first. The summary adds whether that ranking's first pose is near-native
    to what confidence.py already asks of its own.

sapt.py's run reads confidence.csv out of a screen's directory and every complex's kept scores out of
    the complex's own, and writes sapt.csv and sapt_summary.csv beside confidence.py's tables.
"""

import os

import numpy as np
import pytest

import confidence
import filter
import sapt
from utils import save

NAME, OTHER = "5S8I_2LY", "6ZCY_QF8"

DEPOSITED = "rank1_confidence0.99.sdf"
FIRST, SECOND = "rank10_confidence-3.19.sdf", "rank11_confidence-3.19.sdf"
THIRD = "rank12_confidence-3.21.sdf"

# Either side of the near-native threshold. Angstrom.
NEAR, FAR = 0.0, 2 * filter.NEAR_NATIVE

SCREEN = "reranked"

TABLE, SUMMARY = "sapt.csv", "sapt_summary.csv"


def rows(*poses, name=NAME):
    """
    confidence.csv's rows for one complex. Each pose is given as (source, confidence, rmsd).
    """
    return confidence.rank([
        dict(zip(
            confidence.FIELDS,
            (name, source, *confidence.docked_rank_and_score(source), None, scored, rmsd),
        ))
        for source, scored, rmsd in poses
    ])


def energies(*poses, name=NAME):
    return {
        (name, source): {"elst": elst, "exch": exch, "cumulant": cumulant}
        for source, elst, exch, cumulant in poses
    }


def by_source(rows):
    return {row["source"]: row for row in rows}


def keep(job, *poses, name=NAME):
    """
    Each pose is given as (source, elst, exch, cumulant).
    """
    sources, elst, exch, cumulants = (list(column) for column in zip(*poses))
    save.save_sapt(
        {
            "source": sources,
            "electrostatics": elst,
            "exchanges": exch,
            "cumulants": cumulants,
            "int_energies": np.add(elst, exch),
        },
        name,
        confidence.complex_dir(name, job),
    )


@pytest.fixture
def screen(tmp_path, monkeypatch):
    """
    A screen confidence.py has ranked, of NAME's three poses and OTHER's one. Only NAME was scored,
        and its record holds the poses in another order than confidence.csv does.
    """
    monkeypatch.setattr(filter, "OUT", str(tmp_path))
    job = filter.job_dir(SCREEN)
    confidence.write_table(
        rows((FIRST, 0.75, FAR), (SECOND, -1.0, FAR), (DEPOSITED, -2.0, NEAR))
        + rows((THIRD, 0.5, FAR), name=OTHER),
        os.path.join(job, confidence.TABLE_NAME),
        confidence.FIELDS,
    )
    keep(
        job,
        (DEPOSITED, -0.030, 0.012, -0.001),
        (SECOND, -0.010, 0.004, 0.0),
        (FIRST, -0.020, 0.012, 0.0),
    )
    return SCREEN


def test_join_gives_each_pose_its_energies_and_names_a_pose_that_has_none():
    ranked = rows((FIRST, 0.75, FAR), (SECOND, -1.0, FAR), (DEPOSITED, -2.0, NEAR))

    joined, missing = sapt.join(
        ranked, energies((DEPOSITED, -0.030, 0.012, -0.001), (FIRST, -0.010, 0.004, 0.0))
    )

    deposited = by_source(joined)[DEPOSITED]
    assert (deposited["elst"], deposited["exch"], deposited["cumulant"]) == (-0.030, 0.012, -0.001)
    assert deposited["interaction"] == pytest.approx(-0.030 + 0.012)
    assert deposited["rank_minimised"] == 3
    assert missing == [(NAME, SECOND)]


def test_rank_puts_the_lowest_interaction_energy_first():
    joined, _ = sapt.join(
        rows((SECOND, 0.75, FAR), (DEPOSITED, -1.0, NEAR), (FIRST, -2.0, FAR))
        + rows((THIRD, 0.5, FAR), name=OTHER),
        {
            **energies(
                (SECOND, -0.010, 0.004, 0.0),
                (DEPOSITED, -0.030, 0.012, 0.0),
                (FIRST, -0.010, 0.004, 0.0),
            ),
            **energies((THIRD, -0.020, 0.010, 0.0), name=OTHER),
        },
    )

    ranked = sapt.rank(joined)

    for name, order in [(NAME, [DEPOSITED, FIRST, SECOND]), (OTHER, [THIRD])]:
        theirs = sorted(
            (row for row in ranked if row["name"] == name), key=lambda row: row["rank_sapt"]
        )
        assert [row["source"] for row in theirs] == order
        assert [row["rank_sapt"] for row in theirs] == list(range(1, len(order) + 1))


def test_summarise_asks_whether_the_lowest_energy_pose_is_near_native():
    joined, _ = sapt.join(
        rows((FIRST, 0.75, FAR), (DEPOSITED, -1.0, NEAR)),
        energies((FIRST, -0.010, 0.004, 0.0), (DEPOSITED, -0.030, 0.012, 0.0)),
    )

    summary = sapt.summarise(sapt.rank(joined))

    assert len(summary) == 1
    assert summary[0]["top1_sapt"] is True
    assert summary[0]["top1_minimised"] is False
    assert summary[0]["top1_docked"] is True
    assert summary[0]["near_native"] == 1
    assert summary[0]["fraction"] == pytest.approx(0.5)


def test_summarise_records_a_ranking_that_puts_a_pose_far_from_the_deposited_ligand_first():
    joined, _ = sapt.join(
        rows((FIRST, -1.0, FAR), (DEPOSITED, 0.75, NEAR)),
        energies((FIRST, -0.030, 0.012, 0.0), (DEPOSITED, -0.010, 0.004, 0.0)),
    )

    summary = sapt.summarise(sapt.rank(joined))

    assert summary[0]["top1_sapt"] is False
    assert summary[0]["top1_minimised"] is True


def test_summarise_rates_each_ranking_over_the_pairs_the_near_native_pose_makes():
    joined, _ = sapt.join(
        rows((FIRST, 0.75, FAR), (SECOND, -1.0, FAR), (DEPOSITED, -2.0, NEAR)),
        energies(
            (FIRST, -0.010, 0.004, 0.0),
            (SECOND, -0.005, 0.004, 0.0),
            (DEPOSITED, -0.030, 0.012, 0.0),
        ),
    )

    summary = sapt.summarise(sapt.rank(joined))

    assert summary[0]["discrimination_sapt"] == 1.0
    assert summary[0]["discrimination_minimised"] == 0.0
    # DiffDock scored the other two the same, but they are both far, so they make no pair.
    assert summary[0]["discrimination_docked"] == 1.0


def test_the_tables_round_trip_through_confidence_pys_reader(tmp_path):
    joined, _ = sapt.join(
        rows((DEPOSITED, 0.75, None), (FIRST, -1.0, FAR)),
        energies((DEPOSITED, -0.030, 0.012, -0.001), (FIRST, -0.010, 0.004, 0.0)),
    )
    ranked = sapt.rank(joined)
    summarised = sapt.summarise(ranked)
    poses, summary = str(tmp_path / "sapt.csv"), str(tmp_path / "sapt_summary.csv")

    confidence.write_table(ranked, poses, sapt.FIELDS)
    confidence.write_table(summarised, summary, sapt.SUMMARY_FIELDS)

    assert summarised[0]["top1_sapt"] is None
    assert confidence.read_table(poses) == ranked
    assert confidence.read_table(summary) == summarised


def test_run_writes_both_tables_into_the_screens_directory(screen):
    ranked, summarised = sapt.run(screen)

    job = filter.job_dir(screen)
    assert confidence.read_table(os.path.join(job, TABLE)) == ranked
    assert confidence.read_table(os.path.join(job, SUMMARY)) == summarised


def test_run_ranks_each_pose_on_the_scores_its_complex_kept(screen):
    ranked, summarised = sapt.run(screen)

    assert [row["source"] for row in sorted(ranked, key=lambda row: row["rank_sapt"])] == [
        DEPOSITED, FIRST, SECOND
    ]
    deposited = by_source(ranked)[DEPOSITED]
    assert (deposited["elst"], deposited["exch"], deposited["cumulant"]) == (-0.030, 0.012, -0.001)
    assert (summarised[0]["top1_sapt"], summarised[0]["top1_minimised"]) == (True, False)


def test_a_complex_that_was_never_scored_is_left_out_of_both_tables_and_reported(screen, capsys):
    ranked, summarised = sapt.run(screen)

    assert {row["name"] for row in ranked} == {entry["name"] for entry in summarised} == {NAME}
    assert OTHER in capsys.readouterr().out


def test_run_refuses_a_screen_confidence_py_has_not_ranked(tmp_path, monkeypatch):
    monkeypatch.setattr(filter, "OUT", str(tmp_path))
    keep(filter.job_dir(SCREEN), (DEPOSITED, -0.030, 0.012, 0.0))

    with pytest.raises(SystemExit):
        sapt.run(SCREEN)


def test_run_refuses_a_screen_none_of_whose_complexes_was_scored(screen):
    os.remove(save.sapt_path(NAME, confidence.complex_dir(NAME, filter.job_dir(screen))))

    with pytest.raises(SystemExit):
        sapt.run(screen)
