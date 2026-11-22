"""``health sync-short-text`` and ``health check`` around the Short text sync incidents
(unit C5b): the nightly entry point files, resolves and pushes them; ``health check`` only lists."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_queue_sync_short_text import NOW, _live_schedule, _short, _world

from gomoveshift.pipeline import _file_lock
from gomoveshift.pipeline import longform_metadata as lm
from gomoveshift.publishing import queue_short_text as qst


@pytest.fixture(autouse=True)
def _no_published_titles(monkeypatch):
    monkeypatch.setattr(lm, "load_published_titles", lambda: set())


def _run_check(tmp_path, notified):
    from gomoveshift.health.check import run_check

    cfg = SimpleNamespace(seo=SimpleNamespace(storage_path="relative/seo"))
    return run_check(
        cfg,
        incidents_dir=tmp_path / "inc",
        jobs_dir=tmp_path / "jobs",
        code_root=tmp_path,
        staging_sha="abc123",
        now=NOW,
        notify_fn=lambda *a: notified.append(a) or True,
    )


def test_health_check_names_the_sync_incidents_but_never_writes_or_pushes_them(tmp_path):
    from gomoveshift.health import check, short_text_sync
    from gomoveshift.health.store import new_incident, write_row

    assert check._EVENTS["short_text_sync_refused"] == "health_short_text_sync_refused"
    assert check._EVENTS["short_text_sync_crash"] == "health_short_text_sync_crash"
    row = new_incident(
        "short_text_sync_refused",
        "short_text_sync_refused:p1",
        "short_text_sync_refused_p1.json",
        "2026-11-20T00:00:00Z",
        project="p1",
        shorts=[{"delivery_id": 1, "short_index": 1, "reason": "anti-spam gate: x"}],
    )
    row["detail"] = short_text_sync.incident_line(row)
    write_row(tmp_path / "inc", row)
    crash = new_incident(
        "short_text_sync_crash",
        "short_text_sync_crash",
        "short_text_sync_crash.json",
        "t",
        detail="x",
    )
    write_row(tmp_path / "inc", crash)
    before = {p.name: p.read_bytes() for p in (tmp_path / "inc").glob("*.json")}
    notified = []
    result = _run_check(tmp_path, notified)
    assert notified == [] and not result["pushed"] and not result["resolved"]
    assert {p.name: p.read_bytes() for p in (tmp_path / "inc").glob("*.json")} == before
    assert {r["key"] for r in result["incidents"]} == {
        "short_text_sync_refused:p1",
        "short_text_sync_crash",
    }
    # an open sync-owned row is an open incident: it keeps `health check` at exit 1
    assert len(result["incidents"]) == 2


# -- the health entry point ---------------------------------------------------------------


def _change(status, project="p1", idx=1, reason="", transient=False):
    return qst.ShortTextChange(1, project, idx, status, "old", "new", reason, transient=transient)


def _health(tmp_path, monkeypatch, sync, notified):
    from gomoveshift.health import short_text_sync

    monkeypatch.setattr(short_text_sync, "sync_short_text", sync)
    monkeypatch.setattr(short_text_sync, "_brand_location_cap", lambda: None)
    monkeypatch.setattr(
        short_text_sync, "open_queue_db", lambda path: SimpleNamespace(close=lambda: None)
    )
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))

    def run():
        return short_text_sync.run_short_text_sync(
            config,
            incidents_dir=tmp_path / "inc",
            now=NOW,
            notify_fn=lambda ev, ident, line: notified.append((ev, ident["incident"])) or True,
        )

    return run


def _inc(tmp_path):
    return {p.name: json.loads(p.read_text()) for p in sorted((tmp_path / "inc").glob("*.json"))}


def test_a_refused_short_files_one_incident_per_project_and_notifies_once(tmp_path, monkeypatch):
    notified = []
    run = _health(
        tmp_path,
        monkeypatch,
        lambda *a, **k: [
            _change("refused", "p1", 1, "anti-spam gate: boom"),
            _change("refused", "p1", 2, "plan gate: x"),
            _change("changed", "p2", 1),
        ],
        notified,
    )
    result = run()
    assert result["exit"] == 1 and result["refused"] == 2
    rows = _inc(tmp_path)
    assert [r["key"] for r in rows.values()] == ["short_text_sync_refused:p1"]
    assert [s["short_index"] for s in next(iter(rows.values()))["shorts"]] == [1, 2]
    run()
    assert notified == [("health_short_text_sync_refused", "short_text_sync_refused:p1")]


def test_a_refusal_that_clears_on_its_own_files_nothing_and_exits_clean(tmp_path, monkeypatch):
    notified = []
    run = _health(
        tmp_path,
        monkeypatch,
        lambda *a, **k: [
            _change(
                "refused",
                "p1",
                1,
                "youtube row 3 slot 2026-11-21 is inside the lead time",
                transient=True,
            ),
            _change("refused", "p1", 2, "facebook row 4 holds a live lease", transient=True),
            _change(
                "refused",
                "p2",
                1,
                "an earlier sidecar's write failed; retried next run",
                transient=True,
            ),
        ],
        notified,
    )
    result = run()
    assert (result["exit"], result["refused"], result["new"]) == (0, 0, [])
    assert _inc(tmp_path) == {} and notified == []
    mixed = _health(
        tmp_path,
        monkeypatch,
        lambda *a, **k: [
            _change("refused", "p1", 1, "facebook row 4 holds a live lease", transient=True),
            _change("refused", "p1", 2, "anti-spam gate: shared-prefix: x"),
        ],
        notified,
    )
    result = mixed()
    assert (result["exit"], result["refused"]) == (1, 1)
    (row,) = _inc(tmp_path).values()
    assert [s["short_index"] for s in row["shorts"]] == [2]


def test_a_clean_run_resolves_the_open_incidents(tmp_path, monkeypatch):
    notified = []
    state = {"out": [_change("refused", "p1", 1, "anti-spam gate: boom")]}
    run = _health(tmp_path, monkeypatch, lambda *a, **k: state["out"], notified)
    run()
    state["out"] = [_change("unchanged", "p1", 1)]
    result = run()
    assert result["exit"] == 0 and result["resolved"] == ["short_text_sync_refused:p1"]
    assert next(iter(_inc(tmp_path).values()))["resolved_at"]


def test_a_crash_files_the_crash_incident_and_keeps_refusals_open(tmp_path, monkeypatch):
    notified = []
    state = {"out": [_change("refused", "p1", 1, "anti-spam gate: boom")]}

    def sync(*a, **k):
        if isinstance(state["out"], Exception):
            raise state["out"]
        return state["out"]

    run = _health(tmp_path, monkeypatch, sync, notified)
    run()
    state["out"] = RuntimeError("db locked")
    result = run()
    assert result["exit"] == 2
    rows = {r["key"]: r for r in _inc(tmp_path).values()}
    assert "db locked" in rows["short_text_sync_crash"]["detail"]
    assert not rows["short_text_sync_refused:p1"]["resolved_at"]  # unknown, not clean
    state["out"] = []
    result = run()
    rows = {r["key"]: r for r in _inc(tmp_path).values()}
    assert rows["short_text_sync_crash"]["resolved_at"]
    assert rows["short_text_sync_refused:p1"]["resolved_at"]
    assert [e for e, _ in notified].count("health_short_text_sync_crash") == 1


def test_the_failed_notification_is_retried(tmp_path, monkeypatch):
    from gomoveshift.health import short_text_sync

    monkeypatch.setattr(
        short_text_sync,
        "sync_short_text",
        lambda *a, **k: [_change("refused", "p1", 1, "plan gate: b")],
    )
    monkeypatch.setattr(short_text_sync, "_brand_location_cap", lambda: None)
    monkeypatch.setattr(
        short_text_sync, "open_queue_db", lambda path: SimpleNamespace(close=lambda: None)
    )
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
    outcomes = iter([False, True])
    short_text_sync.run_short_text_sync(
        config, incidents_dir=tmp_path / "inc", now=NOW, notify_fn=lambda *a: next(outcomes)
    )
    row = next(iter(_inc(tmp_path).values()))
    assert row["pushed"] is False
    short_text_sync.run_short_text_sync(
        config, incidents_dir=tmp_path / "inc", now=NOW, notify_fn=lambda *a: next(outcomes)
    )
    assert next(iter(_inc(tmp_path).values()))["pushed"] is True


def test_the_health_cli_command_runs_the_sync(tmp_path, monkeypatch, capsys):
    from gomoveshift.cli.handlers.health import cmd_health

    from gomoveshift.health import short_text_sync

    # The CLI reads the real clock, so the slot must too (NOW+72h went stale on 2026-11-22).
    plan, conn = _world(tmp_path, [_short(1, schedule=_live_schedule())])
    conn.close()
    monkeypatch.setattr(short_text_sync, "_brand_location_cap", lambda: None)
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
    args = SimpleNamespace(  # no jobs_dir: the sync-short-text subparser defines none
        health_command="sync-short-text", data_root=tmp_path, incidents_dir=tmp_path / "inc"
    )
    monkeypatch.setattr("gomoveshift.console.notify.notify", lambda *a, **k: True)
    assert cmd_health(args, config) == 0
    assert "sync-short-text: 1 changed" in capsys.readouterr().out


def test_the_entry_point_loads_the_brand_location_cap_and_passes_it(tmp_path, monkeypatch):
    from gomoveshift.health import short_text_sync

    seen = []
    monkeypatch.setattr(short_text_sync, "sync_short_text", lambda *a, **k: seen.append(k) or [])
    monkeypatch.setattr(
        short_text_sync, "open_queue_db", lambda path: SimpleNamespace(close=lambda: None)
    )
    monkeypatch.setattr(
        "gomoveshift.publishing.brand_policy.load_brand_policy",
        lambda brand: SimpleNamespace(per_location_cap=1),
    )
    monkeypatch.setattr("gomoveshift.console.settings.current_brand", lambda: "brand")
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
    short_text_sync.run_short_text_sync(
        config, incidents_dir=tmp_path / "inc", now=NOW, notify_fn=lambda *a: True
    )
    assert [k["per_location_cap"] for k in seen] == [1]


def test_a_generator_that_always_raises_gives_an_incident_and_exit_one(tmp_path, monkeypatch):
    from gomoveshift.health import short_text_sync

    def boom(*a, **k):
        raise RuntimeError("boom")

    plan, conn = _world(tmp_path, [_short(1)])
    conn.close()
    monkeypatch.setattr(qst, "_generate_lf", boom)
    monkeypatch.setattr(short_text_sync, "_brand_location_cap", lambda: None)
    notified = []
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
    result = short_text_sync.run_short_text_sync(
        config,
        incidents_dir=tmp_path / "inc",
        now=NOW,
        notify_fn=lambda ev, ident, line: notified.append(ident["incident"]) or True,
    )
    assert result["exit"] == 1 and result["refused"] == 1
    (row,) = _inc(tmp_path).values()
    assert "generator: boom" in row["detail"] and notified == [row["key"]]


def test_an_open_incident_whose_shorts_change_is_refreshed_without_a_second_push(
    tmp_path, monkeypatch
):
    notified = []
    state = {"out": [_change("refused", "p1", 1, "plan gate: a")]}
    run = _health(tmp_path, monkeypatch, lambda *a, **k: state["out"], notified)
    run()
    state["out"] = [
        _change("refused", "p1", 1, "plan gate: a"),
        _change("refused", "p1", 2, "plan gate: b"),
    ]
    result = run()
    (row,) = _inc(tmp_path).values()
    assert [s["short_index"] for s in row["shorts"]] == [1, 2] and "2 queued Short" in row["detail"]
    assert result["new"] == [] and len(notified) == 1


@pytest.mark.parametrize("where", ["notify", "load_all", "write_row"])
def test_the_entry_point_never_raises(tmp_path, monkeypatch, where):
    from gomoveshift.health import short_text_sync

    notified = []
    run = _health(
        tmp_path,
        monkeypatch,
        lambda *a, **k: [_change("refused", "p1", 1, "plan gate: a")],
        notified,
    )
    if where == "notify":
        config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))

        def run():  # noqa: F811
            def bad(*a):
                raise RuntimeError("push down")

            return short_text_sync.run_short_text_sync(
                config, incidents_dir=tmp_path / "inc", now=NOW, notify_fn=bad
            )

    else:

        def bad(*a, **k):
            raise OSError("disk gone")

        monkeypatch.setattr(short_text_sync, where, bad)
    result = run()
    assert result["exit"] == 2 and result["crashed"]


def test_a_held_store_lock_skips_the_incident_work_and_exits_clean(tmp_path, monkeypatch):
    import fcntl

    notified = []
    run = _health(
        tmp_path,
        monkeypatch,
        lambda *a, **k: [_change("refused", "p1", 1, "plan gate: a")],
        notified,
    )
    with open(tmp_path / "inc.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run()
    assert result["skipped"] and result["exit"] == 0
    assert not (tmp_path / "inc").exists() or _inc(tmp_path) == {}
    assert notified == []
    assert run()["exit"] == 1 and len(_inc(tmp_path)) == 1  # the next run, lock free, files it


# -- a transient refusal that never clears escalates -----------------------------------------

LEASE = "facebook row 4 holds a live lease"


def _transient(project="p1", idx=1):
    return _change("refused", project, idx, LEASE, transient=True)


def _refused_rows(tmp_path):
    return {
        k: r for k, r in {r["key"]: r for r in _inc(tmp_path).values()}.items() if "refused" in k
    }


def test_two_transient_runs_file_no_incident(tmp_path, monkeypatch):
    notified = []
    run = _health(tmp_path, monkeypatch, lambda *a, **k: [_transient()], notified)
    for _ in range(2):
        result = run()
        assert (result["exit"], result["refused"], result["new"]) == (0, 0, [])
    assert _refused_rows(tmp_path) == {} and notified == []
    assert result["incidents"] == []  # the streak counter is not an incident


def test_the_third_transient_run_files_one_incident_pushed_once(tmp_path, monkeypatch):
    from gomoveshift.health.short_text_sync import TRANSIENT_ESCALATE_RUNS

    assert TRANSIENT_ESCALATE_RUNS == 3
    notified = []
    run = _health(tmp_path, monkeypatch, lambda *a, **k: [_transient()], notified)
    run(), run()
    result = run()
    assert result["exit"] == 1 and result["refused"] == 1
    (row,) = _refused_rows(tmp_path).values()
    assert row["key"] == "short_text_sync_refused:p1" and not row["resolved_at"]
    (short,) = row["shorts"]
    assert short["short_index"] == 1
    assert "3 consecutive runs" in short["reason"] and LEASE in short["reason"]
    assert notified == [("health_short_text_sync_refused", "short_text_sync_refused:p1")]
    again = run()  # still stuck on the 4th night: still one open row, never pushed twice
    assert again["exit"] == 1 and len(_refused_rows(tmp_path)) == 1
    assert (
        "4 consecutive runs" in next(iter(_refused_rows(tmp_path).values()))["shorts"][0]["reason"]
    )
    assert len(notified) == 1


def test_a_run_that_no_longer_refuses_the_short_resets_the_count(tmp_path, monkeypatch):
    notified = []
    state = {"out": [_transient()]}
    run = _health(tmp_path, monkeypatch, lambda *a, **k: state["out"], notified)
    run(), run()
    state["out"] = [_change("unchanged", "p1", 1)]
    run()
    state["out"] = [_transient()]
    run(), run()
    assert _refused_rows(tmp_path) == {} and notified == []  # 2 in a row again, not 4
    assert run()["exit"] == 1  # the third consecutive one escalates
    assert len(notified) == 1


def test_an_escalated_incident_resolves_when_the_short_clears(tmp_path, monkeypatch):
    notified = []
    state = {"out": [_transient()]}
    run = _health(tmp_path, monkeypatch, lambda *a, **k: state["out"], notified)
    run(), run(), run()
    state["out"] = []
    result = run()
    assert result["exit"] == 0 and result["resolved"] == ["short_text_sync_refused:p1"]


def test_a_crash_leaves_the_streak_alone(tmp_path, monkeypatch):
    notified = []
    state = {"out": [_transient()]}

    def sync(*a, **k):
        if isinstance(state["out"], Exception):
            raise state["out"]
        return state["out"]

    run = _health(tmp_path, monkeypatch, sync, notified)
    run(), run()
    state["out"] = RuntimeError("boom")
    assert run()["exit"] == 2
    state["out"] = [_transient()]
    assert run()["exit"] == 1  # the crash night proved nothing: this is the 3rd refusal


def test_an_actionable_night_between_transient_ones_keeps_the_streak(tmp_path, monkeypatch):
    notified = []
    cycle = [[_transient()], [_transient()], [_change("refused", "p1", 1, "generator: boom")]]
    state = {"i": 0}

    def sync(*a, **k):
        out = cycle[state["i"] % 3]
        state["i"] += 1
        return out

    run = _health(tmp_path, monkeypatch, sync, notified)
    exits = [run()["exit"] for _ in range(8)]
    assert exits == [0, 0, 1, 1, 1, 1, 1, 1]  # the actionable night counts: never resets to 0
    assert len(notified) == 1  # one open row, pushed once


def test_transient_transient_actionable_transient_keeps_the_row_open_pushed_once(
    tmp_path, monkeypatch
):
    notified = []
    outs = iter(
        [
            [_transient()],
            [_transient()],
            [_change("refused", "p1", 1, "generator: boom")],
            [_transient()],
        ]
    )
    run = _health(tmp_path, monkeypatch, lambda *a, **k: next(outs), notified)
    run(), run(), run()
    result = run()
    assert result["exit"] == 1 and result["refused"] == 1 and result["resolved"] == []
    (row,) = _refused_rows(tmp_path).values()
    assert not row["resolved_at"]
    assert "4 consecutive runs" in row["shorts"][0]["reason"]
    assert len(notified) == 1


def test_the_streak_is_kept_per_short_and_per_project(tmp_path, monkeypatch):
    notified = []
    runs = iter(
        [
            [_transient("p1", 1), _transient("p1", 2), _transient("p2", 1)],
            [_transient("p1", 1), _change("unchanged", "p1", 2), _change("unchanged", "p2", 1)],
            [_transient("p1", 1), _transient("p1", 2), _transient("p2", 1)],
        ]
    )
    run = _health(tmp_path, monkeypatch, lambda *a, **k: next(runs), notified)
    run(), run()
    result = run()
    rows = _refused_rows(tmp_path)
    assert set(rows) == {"short_text_sync_refused:p1"}  # p2#1 and p1#2 are on night 1 only
    assert [s["short_index"] for s in rows["short_text_sync_refused:p1"]["shorts"]] == [1]
    assert result["refused"] == 1


def test_projects_whose_names_share_a_safe_filename_keep_separate_streaks(tmp_path, monkeypatch):
    notified = []
    runs = iter(
        [
            [_transient("a b", 1), _transient("a_b", 1)],
            [_transient("a b", 1), _change("unchanged", "a_b", 1)],
            [_transient("a b", 1), _transient("a_b", 1)],
        ]
    )
    run = _health(tmp_path, monkeypatch, lambda *a, **k: next(runs), notified)
    run(), run()
    result = run()
    assert set(_refused_rows(tmp_path)) == {"short_text_sync_refused:a b"}
    assert result["refused"] == 1


def test_the_store_lock_is_the_portable_file_lock_so_the_module_imports_on_windows():
    # ci/import_smoke.py imports every module on Windows, where fcntl does not exist.
    from gomoveshift.health import short_text_sync

    assert not hasattr(short_text_sync, "fcntl")
    assert short_text_sync.lock_exclusive is _file_lock.lock_exclusive


# -- the run record the stale check reads ---------------------------------------------------


def _stamp(tmp_path):
    from gomoveshift.health import short_text_sync

    return short_text_sync.run_record_path(tmp_path / "inc")


def test_the_run_record_sits_next_to_the_incidents_dir(tmp_path):
    assert _stamp(tmp_path) == tmp_path / "inc.short_text_sync_run.json"


def test_a_completed_run_writes_the_stamp_with_the_counts(tmp_path, monkeypatch):
    out = [_change("changed"), _change("reconciled", idx=2), _change("refused", idx=3, reason="x")]
    run = _health(tmp_path, monkeypatch, lambda *a, **k: out, [])
    result = run()
    stamp = json.loads(_stamp(tmp_path).read_text())
    assert stamp["schema"] == 1 and stamp["changed"] == 2 and stamp["refused"] == result["refused"]
    assert stamp["ran_at"].startswith("2026-") and stamp["crashed"] is False
    assert not list(tmp_path.glob("*.tmp*"))  # atomic: no temp file left behind


def test_a_crashed_sync_still_stamps_the_run_with_crashed_true(tmp_path, monkeypatch):
    state = {"fail": False}

    def sync(*a, **k):
        if state["fail"]:
            raise RuntimeError("db locked")
        return []

    run = _health(tmp_path, monkeypatch, sync, [])
    run()
    assert json.loads(_stamp(tmp_path).read_text())["crashed"] is False
    state["fail"] = True
    assert run()["exit"] == 2
    assert json.loads(_stamp(tmp_path).read_text())["crashed"] is True


def test_the_stamp_time_is_the_clock_at_write_time_unless_now_is_injected(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    from gomoveshift.health import short_text_sync

    cfg = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
    monkeypatch.setattr(short_text_sync, "sync_short_text", lambda *a, **k: [])
    monkeypatch.setattr(short_text_sync, "_brand_location_cap", lambda: None)
    monkeypatch.setattr(
        short_text_sync, "open_queue_db", lambda p: SimpleNamespace(close=lambda: 0)
    )
    short_text_sync.run_short_text_sync(cfg, incidents_dir=tmp_path / "inc", notify_fn=lambda *a: 1)
    ran = datetime.fromisoformat(json.loads(_stamp(tmp_path).read_text())["ran_at"])
    assert abs((datetime.now(timezone.utc) - ran).total_seconds()) < 120
    short_text_sync.run_short_text_sync(
        cfg, incidents_dir=tmp_path / "inc", now=NOW, notify_fn=lambda *a: 1
    )
    assert json.loads(_stamp(tmp_path).read_text())["ran_at"].startswith(NOW.strftime("%Y-%m-%d"))


def test_a_held_store_lock_still_writes_the_stamp(tmp_path, monkeypatch):
    import fcntl

    run = _health(tmp_path, monkeypatch, lambda *a, **k: [], [])
    with open(tmp_path / "inc.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run()
    assert result["skipped"] and _stamp(tmp_path).exists()


def test_a_stamp_write_failure_never_changes_the_exit_code(tmp_path, monkeypatch, caplog):
    from gomoveshift.health import short_text_sync

    blocker = tmp_path / "a-file"
    blocker.write_text("x")  # a directory cannot be made under a regular file
    monkeypatch.setattr(short_text_sync, "run_record_path", lambda d: blocker / "run.json")
    run = _health(tmp_path, monkeypatch, lambda *a, **k: [_change("refused", reason="a: b")], [])
    result = run()
    assert result["exit"] == 1 and not result["crashed"]
    assert "run record" in caplog.text
