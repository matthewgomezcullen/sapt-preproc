"""
Re-scoring the filtered, minimised poses with DiffDock's confidence model, for confidence.py.

The scoring runs in another interpreter. Everything else is tested here: the export, the RMSD labels, the join, the ranking and the two tables.

    5S8I_2LY    the cheapest structure in the set. None of its twenty poses is near-native, every
                one sitting about 4.9 A from the deposited ligand, so the deposited ligand stands in
                whenever a fixture needs a pose the confidence model should pick.
    6ZCY_QF8    a second complex, for the tests that require more than one
"""

import os
import subprocess

import numpy as np
import pytest
from rdkit import Chem

import confidence
import filter
from conftest import DATA, paths
from utils import save

NAME = "5S8I_2LY"
OTHER = "6ZCY_QF8"

NATIVE = os.path.join(DATA, NAME, f"{NAME}_ligand.sdf")

# The deposited ligand entered as a pose.
DEPOSITED = "rank1_confidence0.99.sdf"

# Every real pose of 5S8I_2LY is about 4.9 A out.
AS_DOCKED = 4.0

EXACT = 0.01

CUTOUT = "ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N\n"


@pytest.fixture
def job(tmp_path, monkeypatch):
    """
    The directory the prepared artefacts sit in, standing in for out/filter.
    """
    monkeypatch.setattr(confidence, "JOB", str(tmp_path))
    return str(tmp_path)


def artefact(name=NAME, count=3, deposited=False):
    """
    A prepared record holding real poses of the complex, `source` and all, written where the job
        expects it. `deposited` appends the deposited ligand as a near-native pose.
    """
    _, poses = paths(name)
    chosen = sorted(poses)[:count]
    sources = [os.path.basename(path) for path in chosen]
    molecules = [
        Chem.MolFromMolFile(path, removeHs=False) # pyright: ignore[reportAttributeAccessIssue]
        for path in chosen
    ]
    if deposited:
        sources.append(DEPOSITED)
        molecules.append(
            Chem.MolFromMolFile(NATIVE, removeHs=False) # pyright: ignore[reportAttributeAccessIssue]
        )
    save.save_prepared(
        {
            "cutout": CUTOUT,
            "poses": [
                Chem.MolToMolBlock(molecule) # pyright: ignore[reportAttributeAccessIssue]
                for molecule in molecules
            ],
            "source": sources,
            "charge": -1,
            "electrons": 1188,
            "heavy_atoms": 157,
            "excluded": 2,
            "failed": ["bond_lengths"],
        },
        name,
        confidence.directory(name),
    )
    return sources


def stored(name=NAME):
    """
    The poses of a prepared artefact, as confidence.py reads them back.
    """
    record = save.load_prepared(name, confidence.directory(name))
    return [
        Chem.MolFromMolBlock(str(block), removeHs=False) # pyright: ignore[reportAttributeAccessIssue]
        for block in record["poses"]
    ]


def records(path):
    """
    What was written to an SDF, as (title, molecule) pairs.
    """
    supplier = Chem.SDMolSupplier(path, removeHs=False) # pyright: ignore[reportAttributeAccessIssue]
    return [(molecule.GetProp("_Name"), molecule) for molecule in supplier]


def rows(*poses, name=NAME):
    """
    Rows as the join leaves them, each pose given as (source, confidence, rmsd). The as-docked rank
        and confidence come off the source, which is where label reads them from too.
    """
    made = []
    for source, scored, rmsd in poses:
        rank, published = confidence.docked(source)
        made.append({
            "name": name,
            "source": source,
            "rank_docked": rank,
            "confidence_docked": published,
            "rank_minimised": None,
            "confidence_minimised": scored,
            "rmsd": rmsd,
        })
    return made


def by_source(rows):
    return {row["source"]: row for row in rows}


def scoring(written):
    """
    The scoring pass, stubbed: it writes `written` to the scores file the real one would write, and
        records the command it was asked to run.
    """
    calls = []

    def run(command, **keywords):
        calls.append(command)
        confidence.write(
            written, command[command.index("--scores") + 1], confidence.SCORE_FIELDS
        )
        return subprocess.CompletedProcess(command, 0)

    return calls, run


def scored(work, job, **keywords):
    """
    confidence.score with its two transient files kept inside the job directory.
    """
    return confidence.score(
        work,
        python="python",
        models="models",
        manifest=os.path.join(job, "manifest.csv"),
        scores=os.path.join(job, "scores.csv"),
        **keywords,
    )


def test_directory_is_the_one_the_screen_prepared_into(job):
    assert confidence.directory(NAME) == os.path.join(job, NAME)


def test_export_writes_one_record_for_every_kept_pose(job):
    sources = artefact(count=3)

    path = confidence.export(NAME)

    assert len(records(path)) == len(sources)


def test_export_titles_each_record_with_the_file_the_pose_came_from(job):
    sources = artefact(count=3)

    path = confidence.export(NAME)

    assert [title for title, _ in records(path)] == sources


def test_export_keeps_the_coordinates_and_the_hydrogens(job):
    artefact(count=2)
    before = stored()

    path = confidence.export(NAME)

    for was, (_, now) in zip(before, records(path)):
        assert now.GetNumAtoms() == was.GetNumAtoms()
        assert np.allclose(
            now.GetConformer().GetPositions(), was.GetConformer().GetPositions()
        )


def test_export_sits_beside_the_prepared_artefact(job):
    artefact()

    path = confidence.export(NAME)

    assert path == confidence.poses_path(NAME)
    assert os.path.dirname(path) == confidence.directory(NAME)


def test_export_rewrites_a_file_left_by_an_earlier_preparation(job):
    artefact(count=2)
    confidence.export(NAME)
    sources = artefact(count=3, deposited=True)

    path = confidence.export(NAME)

    assert [title for title, _ in records(path)] == sources


@pytest.mark.parametrize(
    "source, rank, published",
    [
        ("rank1_confidence-0.38.sdf", 1, -0.38),
        ("rank12_confidence-1.00.sdf", 12, -1.00),
        ("rank10_confidence0.31.sdf", 10, 0.31),
        ("rank40_confidence-10.25.sdf", 40, -10.25),
    ],
)
def test_docked_reads_the_rank_and_confidence_off_the_filename(source, rank, published):
    assert confidence.docked(source) == (rank, published)


@pytest.mark.parametrize("source", ["rank1.sdf", "5S8I_2LY_relaxed.sdf", "rank1_confidence.sdf"])
def test_docked_refuses_a_name_that_is_not_a_scored_pose(source):
    with pytest.raises(ValueError):
        confidence.docked(source)


def test_label_measures_every_pose_against_the_deposited_ligand(job):
    artefact(count=3, deposited=True)

    labelled = by_source(confidence.label(NAME, NATIVE))

    assert len(labelled) == 4
    # The deposited ligand against itself, and the poses where DiffDock put them.
    assert labelled[DEPOSITED]["rmsd"] < EXACT
    assert all(row["rmsd"] > AS_DOCKED for source, row in labelled.items() if source != DEPOSITED)


def test_label_agrees_with_the_screen_on_the_near_native_count(job):
    artefact(count=5, deposited=True)
    poses = stored()

    labelled = confidence.label(NAME, NATIVE)

    near = [row for row in labelled if row["rmsd"] <= filter.NEAR_NATIVE]
    assert len(near) == len(filter.near_native(poses, NATIVE))


def test_label_carries_the_rank_and_confidence_diffdock_gave_each_pose(job):
    sources = artefact(count=3)

    labelled = by_source(confidence.label(NAME, NATIVE))

    for source in sources:
        rank, published = confidence.docked(source)
        assert labelled[source]["rank_docked"] == rank
        assert labelled[source]["confidence_docked"] == published


def test_score_runs_the_scoring_pass_once_for_every_complex(job, monkeypatch):
    """
    The confidence model and the language model are gigabytes to load, so they are loaded once.
    """
    artefact(NAME, count=2)
    artefact(OTHER, count=2)
    work = [(name, paths(name)[0], confidence.export(name)) for name in (NAME, OTHER)]
    calls, run = scoring([])
    monkeypatch.setattr(subprocess, "run", run)

    scored(work, job)

    assert len(calls) == 1
    # The interpreter DiffDock was installed into, running the scoring script.
    assert calls[0][:2] == ["python", confidence.SCRIPT]


def test_score_hands_over_the_deposited_protein_and_the_exported_poses(job, monkeypatch):
    """
    The deposited structure, not the capped cutout: the confidence model was trained on whole
        proteins, and the minimised poses are still in the deposited frame.
    """
    artefact(count=2)
    exported = confidence.export(NAME)
    _, run = scoring([])
    monkeypatch.setattr(subprocess, "run", run)

    scored([(NAME, paths(NAME)[0], exported)], job)

    handed = confidence.read(os.path.join(job, "manifest.csv"))
    assert len(handed) == 1
    assert handed[0] == {"name": NAME, "protein": paths(NAME)[0], "sdf": exported}


def test_score_returns_a_confidence_for_every_pose_it_was_given(job, monkeypatch):
    sources = artefact(count=2)
    _, run = scoring([
        {"name": NAME, "source": sources[0], "confidence": -1.5},
        {"name": NAME, "source": sources[1], "confidence": 0.25},
    ])
    monkeypatch.setattr(subprocess, "run", run)

    answer = scored([(NAME, paths(NAME)[0], confidence.export(NAME))], job)

    assert answer == {(NAME, sources[0]): -1.5, (NAME, sources[1]): 0.25}


def test_score_raises_when_the_scoring_pass_fails(job, monkeypatch):
    """
    A pass that died leaves no scores, and an empty table must not read as a set of poses the model
        had nothing good to say about.
    """
    artefact(count=2)

    def failed(command, **keywords):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", failed)

    with pytest.raises(subprocess.CalledProcessError):
        scored([(NAME, paths(NAME)[0], confidence.export(NAME))], job)


def test_join_matches_a_score_to_its_pose_by_name_not_by_order():
    """
    Nothing guarantees the scoring pass answers in the order it was asked, and a join by position
        would attribute one pose's score to another without ever failing.
    """
    labelled = rows(
        ("rank10_confidence-3.19.sdf", None, 4.9),
        ("rank11_confidence-3.19.sdf", None, 4.8),
        (DEPOSITED, None, 0.0),
    )

    joined, missing = confidence.join(labelled, {
        (NAME, DEPOSITED): 0.75,
        (NAME, "rank11_confidence-3.19.sdf"): -2.0,
        (NAME, "rank10_confidence-3.19.sdf"): -1.0,
    })

    assert not missing
    assert by_source(joined)[DEPOSITED]["confidence_minimised"] == 0.75
    assert by_source(joined)["rank10_confidence-3.19.sdf"]["confidence_minimised"] == -1.0


def test_join_reports_a_pose_that_came_back_without_a_score():
    """
    One complex the scoring pass could not handle must not cost the other forty-nine.
    """
    labelled = rows(
        ("rank10_confidence-3.19.sdf", None, 4.9),
        (DEPOSITED, None, 0.0),
    )

    joined, missing = confidence.join(labelled, {(NAME, DEPOSITED): 0.75})

    assert [row["source"] for row in joined] == [DEPOSITED]
    assert missing == [(NAME, "rank10_confidence-3.19.sdf")]


def test_rank_puts_the_most_confident_pose_first():
    ranked = confidence.rank(rows(
        ("rank10_confidence-3.19.sdf", -1.0, 4.9),
        (DEPOSITED, 0.75, 0.0),
        ("rank11_confidence-3.19.sdf", -2.0, 4.8),
    ))

    assert [row["source"] for row in ranked] == [
        DEPOSITED, "rank10_confidence-3.19.sdf", "rank11_confidence-3.19.sdf"
    ]
    assert [row["rank_minimised"] for row in ranked] == [1, 2, 3]


def test_rank_breaks_a_tie_with_the_rank_diffdock_gave():
    """
    Two poses at the same confidence have to order the same way on every run, and DiffDock's own
        ordering is the only tie-break the data carries.
    """
    ranked = confidence.rank(rows(
        ("rank12_confidence-3.21.sdf", -1.0, 4.9),
        ("rank10_confidence-3.19.sdf", -1.0, 4.8),
    ))

    assert [row["rank_docked"] for row in ranked] == [10, 12]


def test_rank_ranks_each_complex_on_its_own():
    ranked = confidence.rank(
        rows((DEPOSITED, -5.0, 0.0))
        + rows(("rank10_confidence-3.19.sdf", 1.0, 4.9), name=OTHER)
    )

    assert {row["name"] for row in ranked if row["rank_minimised"] == 1} == {NAME, OTHER}


def test_summarise_counts_what_fraction_of_the_ensemble_is_near_native():
    """
    The x-axis of the plot, and the rate a random pick off the ensemble would score.
    """
    summary = confidence.summarise(confidence.rank(rows(
        (DEPOSITED, 0.75, 0.0),
        ("rank10_confidence-3.19.sdf", -1.0, 4.9),
        ("rank11_confidence-3.19.sdf", -2.0, 4.8),
        ("rank12_confidence-3.21.sdf", -3.0, 1.5),
    )))

    assert len(summary) == 1
    assert summary[0]["poses"] == 4
    assert summary[0]["near_native"] == 2
    assert summary[0]["fraction"] == pytest.approx(0.5)


def test_summarise_records_whether_the_top_pose_is_near_native():
    near = confidence.summarise(confidence.rank(rows(
        (DEPOSITED, 0.75, 0.0),
        ("rank10_confidence-3.19.sdf", -1.0, 4.9),
    )))
    far = confidence.summarise(confidence.rank(rows(
        (DEPOSITED, -1.0, 0.0),
        ("rank10_confidence-3.19.sdf", 0.75, 4.9),
    )))

    assert near[0]["top1_minimised"] is True
    assert far[0]["top1_minimised"] is False


def test_summarise_counts_a_pose_exactly_on_the_threshold_as_near_native():
    """
    filter.py's near_native is inclusive, so this is too.
    """
    summary = confidence.summarise(confidence.rank(rows(
        ("rank10_confidence-3.19.sdf", 0.75, filter.NEAR_NATIVE),
    )))

    assert summary[0]["near_native"] == 1
    assert summary[0]["top1_minimised"] is True


def test_summarise_records_what_diffdocks_own_ranking_would_have_picked():
    """
    The baseline the re-scoring is measured against: the best-ranked pose that survived the filter,
        scored before it was minimised.
    """
    summary = confidence.summarise(confidence.rank(rows(
        ("rank12_confidence-3.21.sdf", 0.75, 4.9),
        ("rank10_confidence-3.19.sdf", -2.0, 0.5),
    )))

    # Re-scoring promotes the pose 4.9 A out; DiffDock's own order kept the near-native one first.
    assert summary[0]["top1_minimised"] is False
    assert summary[0]["top1_docked"] is True


def test_summarise_leaves_the_top_pose_unknown_when_its_rmsd_is_missing():
    """
    An RMSD that could not be measured is not a failure to find the pocket.
    """
    summary = confidence.summarise(confidence.rank(rows(
        (DEPOSITED, 0.75, None),
        ("rank10_confidence-3.19.sdf", -1.0, 4.9),
    )))

    assert summary[0]["top1_minimised"] is None
    assert summary[0]["near_native"] == 0


def test_the_tables_round_trip(tmp_path):
    """
    The numbers come back as numbers, a missing RMSD comes back missing, and a top-1 comes back a
        boolean rather than the string 'False'.
    """
    ranked = confidence.rank(rows(
        (DEPOSITED, 0.75, None),
        ("rank10_confidence-3.19.sdf", -1.0, 4.9),
    ))
    poses, summary = str(tmp_path / "poses.csv"), str(tmp_path / "summary.csv")

    confidence.write(ranked, poses, confidence.FIELDS)
    confidence.write(confidence.summarise(ranked), summary, confidence.SUMMARY_FIELDS)

    assert confidence.read(poses) == ranked
    assert confidence.read(summary) == confidence.summarise(ranked)
