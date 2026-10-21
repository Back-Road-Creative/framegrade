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
