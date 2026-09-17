"""
Re-scoring the filtered, minimised poses with DiffDock's confidence model, for confidence.py.

The scoring runs in another interpreter. Everything else is tested here: the export, the RMSD labels, the join, the ranking, the two tables and the figure drawn from them.

    5S8I_2LY    the cheapest structure in the set, and none of its twenty poses is near-native
    6ZCY_QF8    a second complex, for the tests that require more than one
"""

import os
import subprocess

import numpy as np
import pytest
from rdkit.Chem import MolFromMolBlock, MolFromMolFile, MolToMolBlock, SDMolSupplier

import confidence
import filter
import motivation
from conftest import DATA, paths
from utils import save

NAME = "5S8I_2LY"
OTHER = "6ZCY_QF8"

NATIVE = os.path.join(DATA, NAME, f"{NAME}_ligand.sdf")

DEPOSITED = "rank1_confidence0.99.sdf"
FIRST, SECOND = "rank10_confidence-3.19.sdf", "rank11_confidence-3.19.sdf"
THIRD = "rank12_confidence-3.21.sdf"

# Every real pose is about 4.9 A out, and a coordinate survives a mol block only to four decimals.
AS_DOCKED, EXACT = 4.0, 0.01

JOB_DIR_NAME = "plotted"


@pytest.fixture
def job(tmp_path, monkeypatch):
    monkeypatch.setattr(confidence, "JOB", str(tmp_path))
    return str(tmp_path)


@pytest.fixture
def scored_complex(tmp_path, monkeypatch):
    monkeypatch.setattr(filter, "OUT", str(tmp_path))
    monkeypatch.setattr(confidence, "JOB", filter.job_dir(JOB_DIR_NAME))
    protein, poses = paths(NAME)
    monkeypatch.setattr(filter, "inventory", lambda named=None: ([(NAME, protein, poses, NATIVE)], []))
    first, second, deposited = artefact(count=2, deposited=True)
    confidence.write_table(
        [
            {"name": NAME, "source": first, "confidence": 0.5},
            {"name": NAME, "source": second, "confidence": -0.5},
            {"name": NAME, "source": deposited, "confidence": -1.0},
        ],
        os.path.join(filter.job_dir(JOB_DIR_NAME), confidence.SCORES_NAME),
        confidence.SCORE_FIELDS,
    )
    return JOB_DIR_NAME


def artefact(name=NAME, count=3, deposited=False):
    """
    A prepared record holding real poses of the complex. Only the two keys confidence.py reads are
        written; the rest is not tested here.
    """
    _, poses = paths(name)
    chosen = sorted(poses)[:count]
    sources = [os.path.basename(path) for path in chosen]
    if deposited:
        chosen, sources = chosen + [NATIVE], sources + [DEPOSITED]
    molecules = [MolFromMolFile(path, removeHs=False) for path in chosen]
    save.save_prepared(
        {"poses": [MolToMolBlock(molecule) for molecule in molecules], "source": sources},
        name,
        confidence.complex_dir(name),
    )
    return sources


def stored(name=NAME):
    # The poses of a prepared artefact, as confidence.py reads them back.
    record = save.load_prepared(name, confidence.complex_dir(name))
    return [MolFromMolBlock(str(block), removeHs=False) for block in record["poses"]]


def records(path):
    # What was written to an SDF, as (title, molecule) pairs.
    return [(pose.GetProp("_Name"), pose) for pose in SDMolSupplier(path, removeHs=False)]


def rows(*poses, name=NAME):
    """
    Rows as the join leaves them, each pose given as (source, confidence, rmsd). The as-docked rank
        and confidence come off the source, which is where label reads them from too, and the keys
        are FIELDS in order, which is what the table is written from.
    """
    return [
        dict(zip(confidence.FIELDS, (name, source, *confidence.docked_rank_and_score(source), None, scored, rmsd)))
        for source, scored, rmsd in poses
    ]


def by_source(rows):
    return {row["source"]: row for row in rows}


def scoring(written):
    # The scoring pass, stubbed: it writes `written` where the real one would, and logs the command.
    calls = []

    def run(command, **keywords):
        calls.append(command)
        confidence.write_table(written, command[command.index("--scores") + 1], confidence.SCORE_FIELDS)
        return subprocess.CompletedProcess(command, 0)

    return calls, run


def failing(command, **keywords):
    raise subprocess.CalledProcessError(1, command)


def scored(work, job):
    # confidence.score, with its two transient files kept inside the job directory.
    return confidence.score(
        work, python="python", models="models", esm="esm.pt",
        manifest=os.path.join(job, "manifest.csv"), scores=os.path.join(job, "scores.csv"),
    )


def test_export_writes_every_kept_pose_titled_with_the_file_it_came_from(job):
    sources = artefact(count=2)
    before = stored()

    path = confidence.export(NAME)

    assert path == confidence.poses_path(NAME) == os.path.join(
        confidence.complex_dir(NAME), f"{NAME}_poses.sdf"
    )
    assert [title for title, _ in records(path)] == sources
    for was, (_, now) in zip(before, records(path)):
        assert now.GetNumAtoms() == was.GetNumAtoms()
        assert np.allclose(now.GetConformer().GetPositions(), was.GetConformer().GetPositions())


def test_export_rewrites_a_file_left_by_an_earlier_preparation(job):
    artefact(count=2)
    confidence.export(NAME)
    sources = artefact(count=3, deposited=True)

    path = confidence.export(NAME)

    assert [title for title, _ in records(path)] == sources


@pytest.mark.parametrize(
    "source, rank, published",
    [("rank1_confidence-0.38.sdf", 1, -0.38), ("rank10_confidence0.31.sdf", 10, 0.31),
     ("rank40_confidence-10.25.sdf", 40, -10.25)],
)
def test_docked_reads_the_rank_and_confidence_off_the_filename(source, rank, published):
    assert confidence.docked_rank_and_score(source) == (rank, published)


@pytest.mark.parametrize("source", ["rank1.sdf", "5S8I_2LY_relaxed.sdf", "rank1_confidence.sdf"])
def test_docked_refuses_a_name_that_is_not_a_scored_pose(source):
    with pytest.raises(ValueError):
        confidence.docked_rank_and_score(source)


def test_label_measures_every_pose_against_the_deposited_ligand(job):
    """
    The count of near-native poses has to equal filter.csv's.
    """
    sources = artefact(count=3, deposited=True)

    labelled = confidence.initial_rows(NAME, NATIVE)

    by_name = by_source(labelled)
    assert len(by_name) == 4
    # The deposited ligand against itself, and the poses where DiffDock put them.
    assert by_name[DEPOSITED]["rmsd"] < EXACT
    assert all(row["rmsd"] > AS_DOCKED for source, row in by_name.items() if source != DEPOSITED)
    for source in sources:
        assert (by_name[source]["rank_docked"],
                by_name[source]["confidence_docked"]) == confidence.docked_rank_and_score(source)
    near = [row for row in labelled if row["rmsd"] <= filter.NEAR_NATIVE]
    assert len(near) == len(filter.get_near_natives(stored(), NATIVE))


def test_score_hands_every_complex_over_in_one_pass_and_reads_the_answer(job, monkeypatch):
    """
    The confidence model and the language model are loaded once. The protein handed over is the 
        deposited structure.
    """
    mine, theirs = artefact(NAME, count=2), artefact(OTHER, count=2)
    work = [(name, paths(name)[0], confidence.export(name)) for name in (NAME, OTHER)]
    calls, run = scoring([
        {"name": NAME, "source": mine[0], "confidence": -1.5},
        {"name": NAME, "source": mine[1], "confidence": 0.25},
        {"name": OTHER, "source": theirs[0], "confidence": 1.0},
    ])
    monkeypatch.setattr(subprocess, "run", run)

    answer = scored(work, job)

    assert len(calls) == 1
    assert calls[0][:2] == ["python", confidence.SCRIPT]
    assert confidence.read_table(os.path.join(job, "manifest.csv")) == [
        {"name": name, "protein": protein, "sdf": sdf} for name, protein, sdf in work
    ]
    assert answer[(NAME, mine[0])] == -1.5
    assert answer[(OTHER, theirs[0])] == 1.0


def test_score_raises_when_the_scoring_pass_fails(job, monkeypatch):
    artefact(count=2)
    monkeypatch.setattr(subprocess, "run", failing)

    with pytest.raises(subprocess.CalledProcessError):
        scored([(NAME, paths(NAME)[0], confidence.export(NAME))], job)


def test_join_matches_scores_by_name_and_names_a_pose_that_has_none():
    """
    A pose it could not score is dropped and reported.
    """
    labelled = rows((FIRST, None, 4.9), (SECOND, None, 4.8), (DEPOSITED, None, 0.0))

    joined, missing = confidence.join(labelled, {(NAME, DEPOSITED): 0.75, (NAME, FIRST): -1.0})

    assert by_source(joined)[DEPOSITED]["confidence_minimised"] == 0.75
    assert by_source(joined)[FIRST]["confidence_minimised"] == -1.0
    assert missing == [(NAME, SECOND)]


def test_rank_puts_the_most_confident_pose_first():
    """
    A tie is broken by the rank DiffDock gave.
    """
    ranked = confidence.rank(
        rows((SECOND, -1.0, 4.8), (DEPOSITED, 0.75, 0.0), (FIRST, -1.0, 4.9))
        + rows((THIRD, -5.0, 4.9), name=OTHER)
    )

    for name, order in [(NAME, [DEPOSITED, FIRST, SECOND]), (OTHER, [THIRD])]:
        theirs = [row for row in ranked if row["name"] == name]
        assert [row["source"] for row in theirs] == order
        assert [row["rank_minimised"] for row in theirs] == list(range(1, len(order) + 1))


def test_summarise_counts_what_fraction_of_the_ensemble_is_near_native():
    summary = confidence.summarise(confidence.rank(rows(
        (DEPOSITED, 0.75, 0.0), (FIRST, -1.0, 4.9), (SECOND, -2.0, 4.8),
        (THIRD, -3.0, filter.NEAR_NATIVE),
    )))

    assert len(summary) == 1
    assert summary[0]["poses"] == 4
    assert summary[0]["near_native"] == 2
    assert summary[0]["fraction"] == pytest.approx(0.5)
    assert summary[0]["top1_minimised"] is True


def test_summarise_records_what_diffdocks_own_ranking_would_have_picked():
    summary = confidence.summarise(confidence.rank(rows((THIRD, 0.75, 4.9), (FIRST, -2.0, 0.5))))

    # Re-scoring promotes the pose 4.9 A out; DiffDock's own order kept the near-native one first.
    assert summary[0]["top1_minimised"] is False
    assert summary[0]["top1_docked"] is True


def test_the_tables_round_trip_a_missing_rmsd(tmp_path):
    """
    The numbers come back as numbers, a missing RMSD comes back missing, and a top-1 is a boolean. 
    """
    ranked = confidence.rank(rows((DEPOSITED, 0.75, None), (FIRST, -1.0, 4.9)))
    summarised = confidence.summarise(ranked)
    poses, summary = str(tmp_path / "poses.csv"), str(tmp_path / "summary.csv")

    confidence.write_table(ranked, poses, confidence.FIELDS)
    confidence.write_table(summarised, summary, confidence.SUMMARY_FIELDS)

    assert summarised[0]["top1_minimised"] is None
    assert summarised[0]["near_native"] == 0
    assert confidence.read_table(poses) == ranked
    assert confidence.read_table(summary) == summarised


@pytest.mark.parametrize("plot", [True, False])
def test_run_draws_the_figure_into_the_screens_directory_only_when_asked(scored_complex, plot):
    confidence.run(complexes=[NAME], reuse=True, name=scored_complex, plot=plot)

    assert os.path.isfile(f"{motivation.motivation_path(scored_complex, job=True)}.png") is plot


def test_run_plots_the_top1_of_the_rescored_ranking_a_row_a_complex(scored_complex, monkeypatch):
    plotted = []

    def record(rows, *arguments, **keywords):
        plotted.append(rows)
        return f"{motivation.motivation_path(scored_complex, job=True)}.png"

    monkeypatch.setattr(motivation, "plot", record)

    confidence.run(complexes=[NAME], reuse=True, name=scored_complex, plot=True)

    # The deposited ligand is the one near-native pose, and the re-scoring ranked it last.
    assert len(plotted) == 1
    assert [(row["name"], row["top1"]) for row in plotted[0]] == [(NAME, False)]
