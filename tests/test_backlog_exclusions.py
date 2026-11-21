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
    assert row["key"] == "backlog_exclusion_invalid:/b"  # the first source, not the number
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


def test_unlisted_camera_files_are_one_sorted_finding_per_file(tmp_path):
    d = tmp_path / "trip"
    listed, *_ = _cam(d, "GX01.MP4", "GX02.MP4", "zz.insv", "AA.mov", "b.MoV")
    rows = accounting.unlisted_footage_incidents([_row("drive", listed)], [], ISO, [])
    assert [r["name"] for r in rows] == ["AA.mov", "GX02.MP4", "b.MoV", "zz.insv"]
    row = rows[1]
    assert row["kind"] == "backlog_unlisted_footage"
    assert row["key"] == f"backlog_unlisted_footage:{d}/GX02.MP4"
    assert row["directory"] == str(d)
    assert "GX02.MP4" in check_mod._line(row)


def test_an_excluded_file_is_accounted_for(tmp_path):
    a, b = _cam(tmp_path / "trip", "GX01.MP4", "GX02.MP4")
    ex = [accounting.Exclusion((str(b),), "r", None)]
    assert accounting.unlisted_footage_incidents([_row("drive", a)], ex, ISO, []) == []


def test_an_exclusion_alone_makes_its_directory_checked(tmp_path):
    a, _b = _cam(tmp_path / "only-excluded", "GX01.MP4", "GX02.MP4")
    ex = [accounting.Exclusion((str(a),), "r", None)]
    (row,) = accounting.unlisted_footage_incidents([], ex, ISO, [])
    assert row["name"] == "GX02.MP4"


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


# -- review fixes ----------------------------------------------------------------------------


def _checker(tmp_path, monkeypatch, sent):
    def go(rows, exclusions=None, storage="relative/seo"):
        inputs = tmp_path / "inputs"
        inputs.mkdir(exist_ok=True)
        monkeypatch.setenv("GMS_INPUTS_DIR", str(inputs))
        target = paths.drive_backlog_excluded_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        if exclusions is None:
            target.unlink(missing_ok=True)
        else:
            target.write_text(exclusions)
        monkeypatch.setattr(check_mod, "_backlog_rows", lambda notes: rows)
        cfg = SimpleNamespace(seo=SimpleNamespace(storage_path=storage))
        return run_check(
            cfg,
            incidents_dir=tmp_path / "inc",
            jobs_dir=tmp_path / "jobs",
            code_root=tmp_path,
            staging_sha="abc123",
            now=NOW,
            notify_fn=lambda *a: sent.append(a) or True,
        )

    return go


def test_an_unreadable_folder_keeps_its_open_incident_and_never_renotifies(tmp_path, monkeypatch):
    sent: list = []
    go = _checker(tmp_path, monkeypatch, sent)
    d = tmp_path / "trip"
    a, _b = _cam(d, "GX01.MP4", "GX02.MP4")
    rows = [_row("drive", a)]
    first = go(rows)
    assert len(first["new"]) == 1 and len(sent) == 1
    parked = tmp_path / "trip-offline"
    d.rename(parked)  # D: offline
    off = go(rows)
    assert off["resolved"] == [] and len(off["incidents"]) == 1
    assert any(str(d) in n for n in off["notes"])
    parked.rename(d)  # D: back
    back = go(rows)
    assert back["resolved"] == [] and back["new"] == []
    assert len(sent) == 1


def test_an_unmounted_drive_folder_is_unreadable_not_empty(monkeypatch):
    monkeypatch.setattr(accounting.os.path, "ismount", lambda p: False)
    monkeypatch.setattr(accounting, "_scan", lambda f: pytest.fail("listed an unmounted drive"))
    notes: list[str] = []
    bad: list[str] = []
    rows = accounting.unlisted_footage_incidents(
        [_row("d", "/mnt/z/GMSVidInput/G1.MP4")], [], ISO, notes, bad
    )
    assert rows == [] and bad == ["/mnt/z/gmsvidinput"] and "not mounted" in notes[0]


def test_each_unlisted_file_opens_and_resolves_on_its_own(tmp_path, monkeypatch):
    sent: list = []
    go = _checker(tmp_path, monkeypatch, sent)
    d = tmp_path / "trip"
    c, a = _cam(d, "GX00.MP4", "GX01.MP4")
    rows = [_row("drive", c)]
    first = go(rows)
    assert first["new"] == [f"backlog_unlisted_footage:{a}"]
    (b,) = _cam(d, "GX02.MP4")
    second = go(rows, yaml.safe_dump([_entry(a)]))
    assert second["resolved"] == [f"backlog_unlisted_footage:{a}"]
    assert second["new"] == [f"backlog_unlisted_footage:{b}"]
    assert [e[1]["incident"] for e in sent] == [
        f"backlog_unlisted_footage:{a}",
        f"backlog_unlisted_footage:{b}",
    ]


def test_check_flags_a_stray_store_project_and_keeps_an_excluded_one(tmp_path, monkeypatch):
    from seo_manager.scheduling.store import ScheduleStore

    from gomoveshift.pipeline.seo_slots import STORE_FILENAME, waypost_cadence

    storage = tmp_path / "seo"
    storage.mkdir()
    cfg = SimpleNamespace(seo=SimpleNamespace(storage_path=str(storage)))
    cad = waypost_cadence(cfg)
    region = sorted(cad.regions)[0]
    store = ScheduleStore(storage / STORE_FILENAME, cad.tz, cad.regions)
    store.init()
    for name in ("listed", "stray", "kept"):
        store.register_ready(name, region, 3, now=NOW)
    (a,) = _cam(tmp_path / "trip", "GX01.MP4")
    sent: list = []
    go = _checker(tmp_path, monkeypatch, sent)
    out = go(
        [_row("listed", a)], yaml.safe_dump([_entry("/x/y/G9.MP4", project="kept")]), str(storage)
    )
    orphans = [r["key"] for r in out["incidents"] if r["kind"] == "backlog_store_orphan"]
    assert orphans == ["backlog_store_orphan:stray"]


def test_inserting_an_entry_above_does_not_shift_the_invalid_keys(tmp_path):
    bad = {"sources": ["/z/G1.MP4"]}
    _, before = accounting.load_exclusions(_exfile(tmp_path, [bad]), [])
    _, after = accounting.load_exclusions(_exfile(tmp_path, [_entry("/y/G0.MP4"), bad]), [])
    k = lambda ps: accounting.exclusion_problem_incidents(ps, ISO)[0]["key"]  # noqa: E731
    assert k(before) == k(after) == "backlog_exclusion_invalid:/z/G1.MP4"


def test_a_problem_with_no_usable_source_is_keyed_by_number(tmp_path):
    _, problems = accounting.load_exclusions(_exfile(tmp_path, [_entry("/a"), "junk"]), [])
    (row,) = accounting.exclusion_problem_incidents(problems, ISO)
    assert row["key"] == "backlog_exclusion_invalid:2"


def test_paths_compare_normalised(tmp_path):
    d = tmp_path / "trip"
    a, b = _cam(d, "GX01.MP4", "GX02.MP4")
    messy = f"{d}/./sub/../GX02.MP4"
    assert accounting.unlisted_footage_incidents([_row("d", a, messy)], [], ISO, []) == []
    f = _exfile(tmp_path, [_entry(messy)])
    _, problems = accounting.load_exclusions(f, [_row("drive-x", b)])
    assert problems and "drive-x" in problems[0]["detail"]


def test_a_drive_letter_path_compares_case_insensitively(tmp_path, monkeypatch):
    d = tmp_path / "D"
    (listed,) = _cam(d, "GX01.MP4")
    monkeypatch.setattr(accounting, "_is_drvfs", lambda p: p.startswith(str(tmp_path)))
    row = _row("drive", str(listed).replace("GX01.MP4", "gx01.mp4"))
    assert accounting.unlisted_footage_incidents([row], [], ISO, []) == []
    # a POSIX path elsewhere stays case-sensitive
    monkeypatch.setattr(accounting, "_is_drvfs", lambda p: False)
    assert len(accounting.unlisted_footage_incidents([row], [], ISO, [])) == 1


def test_a_windows_path_is_not_absolute(tmp_path):
    f = _exfile(tmp_path, [_entry("D:\\GMS\\x.MP4")])
    entries, problems = accounting.load_exclusions(f, [])
    assert entries == [] and "absolute" in problems[0]["detail"]


def test_a_slow_listing_is_unreadable_after_the_deadline(tmp_path, monkeypatch):
    import time

    d = tmp_path / "trip"
    (a,) = _cam(d, "GX01.MP4")
    monkeypatch.setattr(accounting, "LIST_DEADLINE_S", 0.2)
    monkeypatch.setattr(accounting, "_scan", lambda f: time.sleep(5) or [])
    notes: list[str] = []
    bad: list[str] = []
    started = time.monotonic()
    rows = accounting.unlisted_footage_incidents([_row("d", a)], [], ISO, notes, bad)
    assert time.monotonic() - started < 2
    assert rows == [] and bad == [str(d)] and "exceeded" in notes[0]


def test_an_unreadable_exclusions_file_files_one_finding_and_skips_rules_10_and_11(
    tmp_path, monkeypatch
):
    sent: list = []
    go = _checker(tmp_path, monkeypatch, sent)
    a, b = _cam(tmp_path / "trip", "GX01.MP4", "GX02.MP4")
    rows = [_row("drive", a)]
    go(rows, yaml.safe_dump([_entry(b)]))  # b excluded: nothing open
    out = go(rows, "[unclosed")
    assert [r["key"] for r in out["incidents"]] == ["backlog_exclusion_invalid:file"]
    assert any("rules 10 and 11 not evaluated" in n for n in out["notes"])
    # an open footage incident is kept, not closed, while the file is unreadable
    (c,) = _cam(tmp_path / "trip", "GX03.MP4")
    go(rows)
    held = go(rows, "[unclosed")
    assert f"backlog_unlisted_footage:{c}" in [r["key"] for r in held["incidents"]]
    assert held["resolved"] == []


def test_rule_10_is_noted_by_name_when_storage_is_relative(tmp_path, monkeypatch):
    go = _checker(tmp_path, monkeypatch, [])
    (a,) = _cam(tmp_path / "trip", "GX01.MP4")
    out = go([_row("drive", a)])
    assert any("rule 10 not evaluated" in n for n in out["notes"])


def test_a_failing_rule_11_does_not_skip_rule_10(tmp_path, monkeypatch):
    from seo_manager.scheduling.store import ScheduleStore

    from gomoveshift.pipeline.seo_slots import STORE_FILENAME, waypost_cadence

    storage = tmp_path / "seo"
    storage.mkdir()
    cfg = SimpleNamespace(seo=SimpleNamespace(storage_path=str(storage)))
    cad = waypost_cadence(cfg)
    store = ScheduleStore(storage / STORE_FILENAME, cad.tz, cad.regions)
    store.init()
    store.register_ready("stray", sorted(cad.regions)[0], 3, now=NOW)
    monkeypatch.setattr(
        accounting,
        "unlisted_footage_incidents",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")),
    )
    go = _checker(tmp_path, monkeypatch, [])
    (a,) = _cam(tmp_path / "trip", "GX01.MP4")
    out = go([_row("drive", a)], None, str(storage))
    keys = [r["key"] for r in out["incidents"] if r["kind"] != "seo_feedback_stale"]
    assert keys == ["backlog_store_orphan:stray"]  # rule 13 is not this test's subject
    assert any("rule 11 not evaluated" in n for n in out["notes"])


# -- B5a2: exclusion edge cases --------------------------------------------------------------


def test_an_entry_that_turns_invalid_keeps_its_folder_listed_and_nothing_closes(
    tmp_path, monkeypatch
):
    sent: list = []
    go = _checker(tmp_path, monkeypatch, sent)
    d = tmp_path / "only-excluded"
    g1, g2 = _cam(d, "G1.MP4", "G2.MP4")
    first = go([], yaml.safe_dump([_entry(g1)]))
    assert first["new"] == [f"backlog_unlisted_footage:{g2}"]
    bad = go([], yaml.safe_dump([_entry(g1, reason="")]))
    keys = {r["key"] for r in bad["incidents"]}
    assert f"backlog_unlisted_footage:{g2}" in keys  # still open, not closed
    assert f"backlog_unlisted_footage:{g1}" in keys  # the invalid entry excludes nothing
    assert f"backlog_exclusion_invalid:{g1}" in keys
    assert bad["resolved"] == []
    fixed = go([], yaml.safe_dump([_entry(g1)]))
    assert f"backlog_unlisted_footage:{g2}" not in fixed["new"]  # no re-notify of G2


def test_an_unreadable_exclusions_file_keeps_open_per_entry_invalid_incidents(
    tmp_path, monkeypatch
):
    sent: list = []
    go = _checker(tmp_path, monkeypatch, sent)
    entry_key = "backlog_exclusion_invalid:/x/G.MP4"
    first = go([], "- sources: [/x/G.MP4]\n")
    assert entry_key in [r["key"] for r in first["incidents"]] and len(sent) == 1
    held = go([], "[unclosed")
    assert entry_key in [r["key"] for r in held["incidents"]]
    assert held["resolved"] == []
    assert len(sent) == 2  # only the :file finding is new
    back = go([], "- sources: [/x/G.MP4]\n")
    assert back["new"] == [] and len(sent) == 2
    assert back["resolved"] == ["backlog_exclusion_invalid:file"]


def test_many_new_unlisted_files_send_one_push_naming_the_count(tmp_path, monkeypatch):
    sent: list = []
    go = _checker(tmp_path, monkeypatch, sent)
    d = tmp_path / "trip"
    listed, *_rest = _cam(d, "G000.MP4", *[f"G{n:03d}.MP4" for n in range(1, 41)])
    out = go([_row("drive", listed)])
    assert len(out["new"]) == 40 and len(out["incidents"]) == 40
    assert len(sent) == 1
    assert "40" in sent[0][2]
    assert "and 30 more" in sent[0][2] and "G001.MP4" in sent[0][2]
    assert all(r["pushed"] for r in out["incidents"])
    assert len(go([_row("drive", listed)])["incidents"]) == 40 and len(sent) == 1


def test_one_new_unlisted_file_keeps_its_own_push(tmp_path, monkeypatch):
    sent: list = []
    go = _checker(tmp_path, monkeypatch, sent)
    a, b = _cam(tmp_path / "trip", "G1.MP4", "G2.MP4")
    go([_row("drive", a)])
    assert [e[1]["incident"] for e in sent] == [f"backlog_unlisted_footage:{b}"]


def test_two_invalid_entries_sharing_a_first_source_get_a_numbered_second_key(tmp_path):
    f = _exfile(tmp_path, [{"sources": ["/b"]}, {"sources": ["/b", "/c"]}])
    _, problems = accounting.load_exclusions(f, [])
    rows = accounting.exclusion_problem_incidents(problems, ISO)
    assert [r["key"] for r in rows] == [
        "backlog_exclusion_invalid:/b",
        "backlog_exclusion_invalid:/b#2",
    ]
    assert rows[0]["file"] != rows[1]["file"]


def test_paths_that_flatten_to_one_safe_name_get_distinct_incident_files(tmp_path, monkeypatch):
    sent: list = []
    go = _checker(tmp_path, monkeypatch, sent)
    p1, k1 = _cam(tmp_path / "a_b", "x.mp4", "keep1.mp4")
    p2, k2 = _cam(tmp_path / "a", "b_x.mp4", "keep2.mp4")
    out = go([_row("drive", k1, k2)])
    assert len(out["incidents"]) == 2
    assert len({r["file"] for r in out["incidents"]}) == 2
    again = go([_row("drive", k1, k2)])
    assert len(again["incidents"]) == 2 and again["resolved"] == []
    inc = list((tmp_path / "inc").glob("backlog_unlisted_footage_*.json"))
    assert len(inc) == 2
