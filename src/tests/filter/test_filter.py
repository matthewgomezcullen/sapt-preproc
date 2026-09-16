"""
filter.py's --name and --no-mm.

Nothing here prepares a complex. PrepareComplex is stood in for.

    6YT6_PKE    a test-data complex carrying a deposited ligand to measure against
"""

import os
from collections import Counter

import filter
from conftest import POSEBUSTERS, paths

NAME = "6YT6_PKE"

RUN = "no_mm"

TETHER = 10.0

ROWS = [
    {
        "name": "5S8I_2LY", "status": "eligible", "heavy_atoms": 157, "charge": -1,
        "electrons": 1188, "poses": 3, "excluded": 1, "near_native": 2, "rejection": "",
    },
    {
        "name": "6ZCY_QF8", "status": "rejected", "heavy_atoms": None, "charge": None,
        "electrons": None, "poses": None, "excluded": None, "near_native": 0,
        "rejection": "metal in the retained region",
    },
]


def inventory_entry(name=NAME, count=2):
    protein, poses = paths(name)
    native = os.path.join(POSEBUSTERS, name, f"{name}_ligand.sdf")
    return name, protein, sorted(poses)[:count], native


def recording(calls):
    """
    A stand-in for PrepareComplex that records how it was built and prepares nothing.
    """

    class Recorded:
        def __init__(self, protein, poses, out=None, *, mm, tether):
            calls.append({
                "protein": protein, "poses": poses, "out": out, "mm": mm, "tether": tether,
            })
            self.poses = []
            self.heavy_atoms = self.charge = self.electrons = self.excluded = None
            self.failed = Counter()

        def prepared(self):
            return False

        def prepare(self):
            pass

    return Recorded


class Serial:
    """
    ProcessPoolExecutor, run in this process.
    """

    def __init__(self, max_workers=None):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        return False

    def map(self, call, iterable):
        return list(map(call, iterable))


def stubbed(monkeypatch, tmp_path=None):
    calls = []
    if tmp_path is not None:
        monkeypatch.setattr(filter, "OUT", str(tmp_path))
    monkeypatch.setattr(filter, "PrepareComplex", recording(calls))
    return calls


def test_a_run_minimises_by_default(monkeypatch):
    calls = stubbed(monkeypatch)

    filter.screen([inventory_entry()])

    assert [call["mm"] for call in calls] == [True]


def test_a_run_without_mm_says_so_for_every_complex(monkeypatch):
    calls = stubbed(monkeypatch)

    filter.screen([inventory_entry(), inventory_entry()], mm=False)

    assert [call["mm"] for call in calls] == [False, False]


def test_a_parallel_run_carries_the_minimisation_settings_too(monkeypatch):
    calls = stubbed(monkeypatch)
    monkeypatch.setattr(filter, "ProcessPoolExecutor", Serial)

    filter.screen_parallel([inventory_entry()], mm=False, tether=TETHER)

    assert [(call["mm"], call["tether"]) for call in calls] == [(False, TETHER)]


def test_a_run_is_minimised_free_by_default(monkeypatch):
    calls = stubbed(monkeypatch)

    filter.screen([inventory_entry()])

    assert [call["tether"] for call in calls] == [None]


def test_a_run_tethers_every_complex_to_what_it_was_given(monkeypatch):
    calls = stubbed(monkeypatch)

    filter.screen([inventory_entry(), inventory_entry()], tether=TETHER)

    assert [call["tether"] for call in calls] == [TETHER, TETHER]


def test_a_named_run_keeps_its_preparations_apart(tmp_path, monkeypatch):
    calls = stubbed(monkeypatch, tmp_path)

    filter.screen([inventory_entry()], name=RUN)

    assert [call["out"] for call in calls] == [os.path.join(tmp_path, f"filter_{RUN}", NAME)]


def test_a_named_parallel_run_keeps_its_preparations_apart(tmp_path, monkeypatch):
    calls = stubbed(monkeypatch, tmp_path)
    monkeypatch.setattr(filter, "ProcessPoolExecutor", Serial)

    filter.screen_parallel([inventory_entry()], name=RUN)

    assert [call["out"] for call in calls] == [os.path.join(tmp_path, f"filter_{RUN}", NAME)]


def test_an_unnamed_run_keeps_its_preparations_where_it_always_did(tmp_path, monkeypatch):
    calls = stubbed(monkeypatch, tmp_path)

    filter.screen([inventory_entry()])

    assert [call["out"] for call in calls] == [os.path.join(tmp_path, "filter", NAME)]


def test_a_named_run_writes_its_own_table(tmp_path, monkeypatch):
    monkeypatch.setattr(filter, "OUT", str(tmp_path))

    filter.write(ROWS, name=RUN)

    assert os.path.exists(os.path.join(tmp_path, f"filter_{RUN}", "filter.csv"))
    assert not os.path.exists(os.path.join(tmp_path, "filter", "filter.csv"))


def test_a_named_table_reads_back_as_it_was_written(tmp_path, monkeypatch):
    monkeypatch.setattr(filter, "OUT", str(tmp_path))
    filter.write(ROWS, name=RUN)

    assert filter.read(name=RUN) == ROWS
