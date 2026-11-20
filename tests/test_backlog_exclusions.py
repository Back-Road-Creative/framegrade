"""``drive_backlog_excluded.yaml`` and health rules 10-11: the manifest, the store and the
footage on disk cannot silently disagree (backlog_store_orphan, backlog_unlisted_footage)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
import yaml

from gomoveshift import paths
from gomoveshift.health import accounting, run_check
from gomoveshift.health import check as check_mod

NOW = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)
ISO = NOW.isoformat()


def _row(project, *sources):
    return SimpleNamespace(project=project, sources=tuple(str(s) for s in sources))


def _cam(folder: Path, *names: str) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for name in names:
        p = folder / name
        p.write_bytes(b"")
        out.append(p)
    return out


def _exfile(tmp_path, data) -> Path:
    p = tmp_path / "drive_backlog_excluded.yaml"
    p.write_text(yaml.safe_dump(data) if not isinstance(data, str) else data)
    return p


def _entry(*sources, reason="measured: no GPS, checked 2026-11-20", **extra):
    return {"sources": [str(s) for s in sources], "reason": reason, **extra}


def test_excluded_path_sits_beside_the_backlog(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_INPUTS_DIR", str(tmp_path))
    assert paths.drive_backlog_excluded_path() == paths.drive_backlog_path().with_name(
        "drive_backlog_excluded.yaml"
    )


# -- the loader ------------------------------------------------------------------------------


def test_a_missing_file_is_an_empty_list_and_no_problem(tmp_path):
    assert accounting.load_exclusions(tmp_path / "nope.yaml", []) == ([], [])


def test_an_empty_list_is_valid(tmp_path):
    assert accounting.load_exclusions(_exfile(tmp_path, []), []) == ([], [])


def test_a_good_entry_loads_with_its_optional_project(tmp_path):
    f = _exfile(tmp_path, [_entry("/d/a/G1.MP4", project="odd-drive")])
    entries, problems = accounting.load_exclusions(f, [])
    assert problems == []
    assert [(e.sources, e.reason, e.project) for e in entries] == [
        (("/d/a/G1.MP4",), "measured: no GPS, checked 2026-11-20", "odd-drive")
    ]


@pytest.mark.parametrize(
    "data, needle",
    [
        ({"sources": ["/a"]}, "top level"),
        ("[unclosed", "unreadable"),
        (["not a mapping"], "mapping"),
        ([{"sources": ["/a"]}], "reason"),
        ([_entry("/a", reason="")], "reason"),
        ([_entry("/a", reason="  ")], "reason"),
        ([_entry("/a", reason=7)], "reason"),
        ([{"reason": "r"}], "sources"),
        ([_entry(reason="r")], "sources"),
        ([{"sources": "/a", "reason": "r"}], "sources"),
        ([{"sources": [3], "reason": "r"}], "sources"),
        ([{"sources": ["rel/G1.MP4"], "reason": "r"}], "absolute"),
        ([_entry("/a", project=5)], "project"),
        ([_entry("/a", colour="red")], "unknown"),
    ],
)
def test_a_contract_break_is_a_problem_naming_the_entry_never_a_crash(tmp_path, data, needle):
    entries, problems = accounting.load_exclusions(_exfile(tmp_path, data), [])
    assert entries == []
    assert len(problems) == 1 and needle in problems[0]["detail"]


def test_a_source_twice_across_entries_is_a_problem_naming_the_second_entry(tmp_path):
    f = _exfile(tmp_path, [_entry("/a/G1.MP4"), _entry("/a/G1.MP4", "/a/G2.MP4")])
    entries, problems = accounting.load_exclusions(f, [])
    assert [e.sources for e in entries] == [("/a/G1.MP4",)]
    assert [p["entry"] for p in problems] == [2]
    assert "/a/G1.MP4" in problems[0]["detail"]


def test_a_source_twice_in_one_entry_is_a_problem(tmp_path):
    f = _exfile(tmp_path, [_entry("/a/G1.MP4", "/a/G1.MP4")])
    _, problems = accounting.load_exclusions(f, [])
    assert len(problems) == 1 and "twice" in problems[0]["detail"]


def test_a_source_that_is_also_a_manifest_row_source_is_a_problem(tmp_path):
    f = _exfile(tmp_path, [_entry("/a/G1.MP4")])
    entries, problems = accounting.load_exclusions(f, [_row("drive-x", "/a/G1.MP4")])
    assert entries == []
    assert "drive-x" in problems[0]["detail"] and "/a/G1.MP4" in problems[0]["detail"]


def test_problems_become_incident_rows_naming_the_entry(tmp_path):
    f = _exfile(tmp_path, [_entry("/a"), {"sources": ["/b"]}])
    _, problems = accounting.load_exclusions(f, [])
    (row,) = accounting.exclusion_problem_incidents(problems, ISO)
    assert row["kind"] == "backlog_exclusion_invalid"
    assert row["key"] == "backlog_exclusion_invalid:2"
    assert "entry 2" in check_mod._line(row) and "reason" in check_mod._line(row)


# -- rule 10: store project in no row and no exclusion ---------------------------------------


def test_a_store_project_in_no_row_and_no_exclusion_is_an_orphan():
    rows = accounting.store_orphan_incidents(
        ["listed", "stray", "kept"],
        [_row("listed")],
        [accounting.Exclusion(("/a/G1.MP4",), "r", "kept")],
        ISO,
    )
    assert [r["project"] for r in rows] == ["stray"]
    row = rows[0]
    assert row["kind"] == "backlog_store_orphan" and row["key"] == "backlog_store_orphan:stray"
    assert "stray" in check_mod._line(row) and "drive_backlog" in check_mod._line(row)


def test_the_orphan_rule_reads_a_real_store(tmp_path):
    from seo_manager.scheduling.store import ScheduleStore

    db = tmp_path / "schedule.sqlite3"
    tz = ZoneInfo("America/New_York")
    store = ScheduleStore(db, tz, ["usa"])
    store.init()
    store.register_ready("in-store", "usa", 3, now=NOW)
    assert accounting.store_projects(db, tz, ["usa"]) == ["in-store"]


def test_an_unreadable_store_is_a_note_not_a_finding(tmp_path):
    notes: list[str] = []
    bad = tmp_path / "schedule.sqlite3"
    bad.write_text("not a database")
    assert accounting.store_projects(bad, ZoneInfo("UTC"), ["usa"], notes) is None
    assert notes and "backlog_store_orphan not evaluated" in notes[0]


# -- rule 11: camera files in a listed folder that no row or exclusion accounts for ---------


def test_unlisted_camera_files_are_one_sorted_finding_per_directory(tmp_path):
    d = tmp_path / "trip"
    listed, *_ = _cam(d, "GX01.MP4", "GX02.MP4", "zz.insv", "AA.mov", "b.MoV")
    (row,) = accounting.unlisted_footage_incidents([_row("drive", listed)], [], ISO, [])
    assert row["kind"] == "backlog_unlisted_footage"
    assert row["key"] == f"backlog_unlisted_footage:{d}"
    assert row["directory"] == str(d)
    assert row["files"] == ["AA.mov", "GX02.MP4", "b.MoV", "zz.insv"]
    assert "GX02.MP4" in check_mod._line(row)


def test_an_excluded_file_is_accounted_for(tmp_path):
    a, b = _cam(tmp_path / "trip", "GX01.MP4", "GX02.MP4")
    ex = [accounting.Exclusion((str(b),), "r", None)]
    assert accounting.unlisted_footage_incidents([_row("drive", a)], ex, ISO, []) == []


def test_an_exclusion_alone_makes_its_directory_checked(tmp_path):
    a, _b = _cam(tmp_path / "only-excluded", "GX01.MP4", "GX02.MP4")
    ex = [accounting.Exclusion((str(a),), "r", None)]
    (row,) = accounting.unlisted_footage_incidents([], ex, ISO, [])
    assert row["files"] == ["GX02.MP4"]


def test_everything_listed_is_no_finding(tmp_path):
    a, b = _cam(tmp_path / "trip", "GX01.MP4", "GX02.MP4")
    assert accounting.unlisted_footage_incidents([_row("d", a, b)], [], ISO, []) == []


def test_non_camera_files_subdirectories_and_other_folders_are_ignored(tmp_path):
    d = tmp_path / "trip"
    a, *_ = _cam(d, "GX01.MP4", "notes.txt", "x.gpx", "thumb.jpg", "GX03.MP4.part")
    (d / "sub.mp4").mkdir()
    _cam(d / "nested", "GX09.MP4")
    _cam(tmp_path / "elsewhere", "GX10.MP4")
    assert accounting.unlisted_footage_incidents([_row("d", a)], [], ISO, []) == []


def test_an_unreadable_directory_is_a_note_naming_it_not_a_finding(tmp_path):
    gone = tmp_path / "offline-d-drive"
    notes: list[str] = []
    rows = accounting.unlisted_footage_incidents([_row("d", gone / "GX01.MP4")], [], ISO, notes)
    assert rows == []
    assert len(notes) == 1 and str(gone) in notes[0]


def test_listing_never_opens_a_media_file(tmp_path, monkeypatch):
    a, _b = _cam(tmp_path / "trip", "GX01.MP4", "GX02.MP4")
    opened = []
    monkeypatch.setattr(Path, "open", lambda self, *x, **k: opened.append(self))
    monkeypatch.setattr(Path, "read_bytes", lambda self: opened.append(self) or b"")
    accounting.unlisted_footage_incidents([_row("d", a)], [], ISO, [])
    assert opened == []


# -- wired into run_check --------------------------------------------------------------------


def _run(tmp_path, monkeypatch, rows, exclusions):
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    monkeypatch.setenv("GMS_INPUTS_DIR", str(inputs))
    target = paths.drive_backlog_excluded_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if exclusions is not None:
        target.write_text(exclusions)
    monkeypatch.setattr(check_mod, "_backlog_rows", lambda notes: rows)
    cfg = SimpleNamespace(seo=SimpleNamespace(storage_path="relative/seo"))
    return run_check(
        cfg,
        incidents_dir=tmp_path / "inc",
        jobs_dir=tmp_path / "jobs",
        code_root=tmp_path,
        staging_sha="abc123",
        now=NOW,
        notify_fn=lambda *a: True,
    )


def test_check_files_unlisted_footage_and_a_broken_exclusion_then_resolves(tmp_path, monkeypatch):
    a, b = _cam(tmp_path / "trip", "GX01.MP4", "GX02.MP4")
    rows = [_row("drive", a)]
    first = _run(tmp_path, monkeypatch, rows, "- sources: [/x/G.MP4]\n")
    kinds = sorted(r["kind"] for r in first["incidents"])
    assert kinds == ["backlog_exclusion_invalid", "backlog_unlisted_footage"]
    fixed = yaml.safe_dump([_entry(b)])
    second = _run(tmp_path, monkeypatch, rows, fixed)
    assert second["incidents"] == [] and len(second["resolved"]) == 2


def test_check_with_no_exclusions_file_still_runs(tmp_path, monkeypatch):
    a, _b = _cam(tmp_path / "trip", "GX01.MP4", "GX02.MP4")
    out = _run(tmp_path, monkeypatch, [_row("drive", a)], None)
    assert [r["kind"] for r in out["incidents"]] == ["backlog_unlisted_footage"]
