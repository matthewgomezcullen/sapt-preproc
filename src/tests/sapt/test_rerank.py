"""
Ranking each complex's poses on their first order interaction energy, beside the ranking confidence.py
    gave them.

The rows are confidence.csv's. Each is given its pose's energies, keyed by the complex and the file
    the pose came from, as confidence.py keys its scores, and each complex's poses are ranked on
    E_elst + E_exch, lowest first. The summary adds whether that ranking's first pose is near-native
    to what confidence.py already asks of its own.
"""

import pytest

import confidence
import filter
import sapt

NAME, OTHER = "5S8I_2LY", "6ZCY_QF8"

DEPOSITED = "rank1_confidence0.99.sdf"
FIRST, SECOND = "rank10_confidence-3.19.sdf", "rank11_confidence-3.19.sdf"
THIRD = "rank12_confidence-3.21.sdf"

# Either side of the near-native threshold. Angstrom.
NEAR, FAR = 0.0, 2 * filter.NEAR_NATIVE


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
