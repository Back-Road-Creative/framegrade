"""CLI wiring for ``gomoveshift health backlog``."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gomoveshift.cli.handlers import health
from gomoveshift.cli.parser import _build_parser
from gomoveshift.health import launch


def _args(tmp_path, *extra):
    return _build_parser().parse_args(
        ["health", "backlog", "--region", "usa", "--stages-root", str(tmp_path / "st"),
         "--output-root", str(tmp_path / "out"), "--data-root", str(tmp_path),
         "--code-root", str(tmp_path), "--staging-sha", "abc1234", *extra]
    )  # fmt: skip


def test_parser_requires_region_and_roots():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["health", "backlog", "--region", "usa"])


@pytest.fixture
def wired(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        launch, "load_inputs",
        lambda config, region, ledger_path, now: ("usa", [], [], [], set(), lambda **kw: None),
    )  # fmt: skip
    monkeypatch.setattr(launch, "lease_fns", lambda code_root: (lambda: True, lambda: None))
    monkeypatch.setattr(launch, "render_horizon_h", lambda config: 1680)

    def fake(**kw):
        seen.update(kw)
        return seen.pop("_result")

    monkeypatch.setattr(launch, "run_backlog", fake)
    return seen


def test_dry_run_passes_layout_and_prints_json(tmp_path, capsys, wired):
    wired["_result"] = {"action": "would_launch", "reason": "r", "skipped": []}
    assert health.cmd_health(_args(tmp_path, "--dry-run", "--json"), SimpleNamespace()) == 0
    assert wired["dry_run"] is True and wired["region"] == "usa"
    assert wired["sources_root"] == tmp_path / "out" / "_sources"
    assert wired["jobs_dir"] == tmp_path / "logs" / "jobs"
    assert wired["mirror_sha"] == "abc1234"
    assert json.loads(capsys.readouterr().out)["action"] == "would_launch"
    assert callable(wired["render_gate"])


def test_refusal_prints_storage_and_exits_2(tmp_path, capsys, wired):
    wired["_result"] = {
        "action": "refused", "reason": "storage short", "skipped": [
            {"project": "sc-a", "reason": "in store (ready)"}],
        "storage": [{"device": 69, "free_gb": 10.0, "need_gb": 60.0, "ok": False,
                     "uses": ["output sc-b"]}],
        "storage_bound": "x1.5",
    }  # fmt: skip
    assert health.cmd_health(_args(tmp_path), SimpleNamespace()) == 2
    out = capsys.readouterr().out
    assert "skipped sc-a: in store (ready)" in out
    assert "[SHORT]" in out and "refused: storage short" in out


def test_absent_store_reads_as_no_drives_and_is_not_created(tmp_path):
    made = []
    drives, register = launch.open_store(tmp_path / "schedule.sqlite3", made.append)
    assert drives == [] and not made and not (tmp_path / "schedule.sqlite3").exists()


def test_absent_store_is_created_only_by_register(tmp_path):
    from zoneinfo import ZoneInfo

    from seo_manager.scheduling.store import ScheduleStore

    path = tmp_path / "schedule.sqlite3"
    _, register = launch.open_store(
        path, lambda p: ScheduleStore(p, ZoneInfo("America/New_York"), ["usa"])
    )
    register(project="sc-a", region="usa", status="auditing", lf_count=1)
    drives, _ = launch.open_store(
        path, lambda p: ScheduleStore(p, ZoneInfo("America/New_York"), ["usa"])
    )
    assert [(d.project, d.status) for d in drives] == [("sc-a", "auditing")]


def test_store_retire_sets_the_drive_failed_and_opens_nothing_until_called(tmp_path, monkeypatch):
    from zoneinfo import ZoneInfo

    from seo_manager.scheduling.store import ScheduleStore

    from gomoveshift.pipeline import seo_slots

    tz = ZoneInfo("America/New_York")
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
    monkeypatch.setattr(seo_slots, "waypost_cadence", lambda cfg: SimpleNamespace(tz=tz, regions=["usa"]))  # fmt: skip
    retire = launch.store_retire(config)
    assert not (tmp_path / seo_slots.STORE_FILENAME).exists()
    store = ScheduleStore(tmp_path / seo_slots.STORE_FILENAME, tz, ["usa"])
    store.register_ready(project="sc-a", region="usa", status="auditing", lf_count=1)
    retire("sc-a")
    assert [(d.project, d.status) for d in store.all_drives()] == [("sc-a", "failed")]


def test_the_handler_wires_a_store_retire(tmp_path, wired):
    wired["_result"] = {"action": "noop", "reason": "covered", "skipped": []}
    assert health.cmd_health(_args(tmp_path, "--dry-run"), SimpleNamespace()) == 0
    assert callable(wired["retire"])


def test_the_handler_wires_a_publisher_that_refuses_without_a_store(tmp_path, wired):
    wired["_result"] = {"action": "noop", "reason": "covered", "skipped": []}
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path / "seo")))
    assert health.cmd_health(_args(tmp_path), config) == 0
    out = wired["publish"]("day-a", False)
    assert out["action"] == "refused" and "schedule.sqlite3 does not exist" in out["reason"]
    assert not (tmp_path / "seo").exists()


def test_supersede_needs_a_reason_at_the_parser(tmp_path, capsys):
    with pytest.raises(SystemExit):
        health.cmd_health(_args(tmp_path, "--supersede", "ire-a"), SimpleNamespace())


def test_supersede_and_reason_reach_run_backlog(tmp_path, capsys, wired):
    wired["_result"] = {
        "action": "launched", "reason": "ire-a for slot", "skipped": [],
        "supersede": {"reason": "old render", "moves": [{"from": "/s/ire-a", "to": "/s/_x"}]},
    }  # fmt: skip
    args = _args(tmp_path, "--supersede", "ire-a", "--reason", "old render")
    assert health.cmd_health(args, SimpleNamespace()) == 0
    assert wired["supersede"] == ("ire-a", "old render")
    assert "superseded /s/ire-a -> /s/_x" in capsys.readouterr().out


def test_a_due_replace_retry_is_run_by_the_plain_tick_with_the_replace_flag(tmp_path, wired):
    from datetime import datetime, timezone

    wired["_result"] = {"action": "launched", "reason": "r", "skipped": []}
    launch.mark_replace_retry(tmp_path / "out", "ire-a", "blur fix", datetime(2020, 1, 1, tzinfo=timezone.utc))  # fmt: skip
    assert health.cmd_health(_args(tmp_path), SimpleNamespace()) == 0
    assert wired["supersede"] == ("ire-a", "blur fix") and wired["replace_scheduled"] is True
    assert callable(wired["live_status"])
    # a dry run never starts a retry
    wired["_result"] = {"action": "noop", "reason": "r", "skipped": []}
    health.cmd_health(_args(tmp_path, "--dry-run"), SimpleNamespace())
    assert wired["supersede"] is None and wired["replace_scheduled"] is False


def test_dry_run_supersede_says_would_supersede(tmp_path, capsys, wired):
    wired["_result"] = {
        "action": "would_launch", "reason": "ire-a for slot", "skipped": [],
        "supersede": {"reason": "old render", "moves": [{"from": "/s/ire-a", "to": "/s/_x"}]},
    }  # fmt: skip
    args = _args(tmp_path, "--supersede", "ire-a", "--reason", "old render", "--dry-run")
    assert health.cmd_health(args, SimpleNamespace()) == 0
    out = capsys.readouterr().out
    assert "would supersede /s/ire-a -> /s/_x" in out
    assert "\nsuperseded " not in "\n" + out


def test_no_supersede_passes_none(tmp_path, wired):
    wired["_result"] = {"action": "noop", "reason": "covered", "skipped": []}
    health.cmd_health(_args(tmp_path), SimpleNamespace())
    assert wired["supersede"] is None


def test_retirements_and_holds_reach_the_cron_log(tmp_path, capsys, wired):
    # the cron logs only the text lines; a hold the backlog wrote was invisible there
    wired["_result"] = {
        "action": "noop", "reason": "a drive is running; one heavy drive at a time", "skipped": [],
        "retired": [{"project": "nc-a", "reason": "audit failed: nc-a job 9ee99a65 ended failed"}],
        "held": ["/st/nc-a/nc-a/_stages"],
    }  # fmt: skip
    assert health.cmd_health(_args(tmp_path), SimpleNamespace()) == 0
    out = capsys.readouterr().out
    assert "retired nc-a: audit failed: nc-a job 9ee99a65 ended failed" in out
    assert "held /st/nc-a/nc-a/_stages (no-auto-resume)" in out
    assert out.rstrip().endswith("noop: a drive is running; one heavy drive at a time")


def test_a_dry_run_says_would_retire_and_would_hold(tmp_path, capsys, wired):
    wired["_result"] = {
        "action": "noop", "reason": "covered", "skipped": [],
        "retired": [{"project": "nc-a", "reason": "r"}], "held": ["/st/nc-a/nc-a/_stages"],
    }  # fmt: skip
    assert health.cmd_health(_args(tmp_path, "--dry-run"), SimpleNamespace()) == 0
    out = capsys.readouterr().out
    assert "would retire nc-a: r" in out and "would hold /st/nc-a/nc-a/_stages" in out


def test_the_handler_passes_the_render_horizon(tmp_path, wired):
    wired["_result"] = {"action": "noop", "reason": "covered", "skipped": []}
    assert health.cmd_health(_args(tmp_path, "--dry-run"), SimpleNamespace()) == 0
    assert wired["horizon_h"] == 1680


def test_failed_audits_reach_the_cron_log_and_a_launch_exits_zero(tmp_path, capsys, wired):
    wired["_result"] = {
        "action": "launched", "reason": "ire-b for slot", "skipped": [],
        "failed_audits": [{"project": "ire-a", "reason": "audit failed: ire-a job 1 ended failed"}],
    }  # fmt: skip
    assert health.cmd_health(_args(tmp_path), SimpleNamespace()) == 0
    out = capsys.readouterr().out
    assert "failed audit ire-a: audit failed: ire-a job 1 ended failed" in out
    assert out.rstrip().endswith("launched: ire-b for slot")


def test_replace_scheduled_reaches_run_backlog(tmp_path, wired):
    wired["_result"] = {"action": "launched", "reason": "r", "skipped": []}
    args = _args(tmp_path, "--supersede", "ire-a", "--reason", "x", "--replace-scheduled")
    assert health.cmd_health(args, SimpleNamespace()) == 0
    assert wired["replace_scheduled"] is True and callable(wired["live_status"])


def test_replace_scheduled_needs_supersede(tmp_path):
    with pytest.raises(SystemExit):
        health.cmd_health(_args(tmp_path, "--replace-scheduled"), SimpleNamespace())


def _marker_wired(monkeypatch, tmp_path, calls, rows=()):
    monkeypatch.setattr(
        launch, "load_inputs",
        lambda config, region, ledger_path, now: ("usa", list(rows), [], [], set(), lambda **kw: None),
    )  # fmt: skip
    monkeypatch.setattr(launch, "lease_fns", lambda code_root: (lambda: True, lambda: None))
    monkeypatch.setattr(launch, "render_horizon_h", lambda config: 1680)
    answers = iter([{"action": "refused", "reason": "cannot supersede ire-a: pinned", "skipped": []},
                    {"action": "launched", "reason": "day-b for slot", "skipped": []}])  # fmt: skip

    def fake(**kw):
        calls.append((kw["supersede"], kw["replace_scheduled"]))
        return next(answers)

    monkeypatch.setattr(launch, "run_backlog", fake)


def test_a_marker_tick_that_does_not_launch_falls_through_to_the_plain_tick(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    calls = []
    _marker_wired(monkeypatch, tmp_path, calls)
    launch.mark_replace_retry(tmp_path / "out", "ire-a", "blur fix", datetime(2020, 1, 1, tzinfo=timezone.utc), region="usa")  # fmt: skip
    assert health.cmd_health(_args(tmp_path), SimpleNamespace()) == 0
    assert calls == [(("ire-a", "blur fix"), True), (None, False)]  # the region still launches


def test_the_handler_gives_the_marker_lookup_the_ticks_region(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    calls = []
    _marker_wired(monkeypatch, tmp_path, calls)
    launch.mark_replace_retry(tmp_path / "out", "ire-a", "blur fix", datetime(2020, 1, 1, tzinfo=timezone.utc), region="ireland")  # fmt: skip
    monkeypatch.setattr(launch, "run_backlog", lambda **kw: calls.append((kw["supersede"], kw["replace_scheduled"])) or {"action": "noop", "reason": "covered", "skipped": []})  # fmt: skip
    assert health.cmd_health(_args(tmp_path), SimpleNamespace()) == 0
    assert calls == [(None, False)]  # an ireland marker never hijacks a usa tick
