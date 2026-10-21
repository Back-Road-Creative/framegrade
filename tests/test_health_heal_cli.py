"""CLI wiring for ``gomoveshift health heal``."""

from __future__ import annotations

from types import SimpleNamespace

from gomoveshift.cli.handlers import health
from gomoveshift.cli.parser import _build_parser


def test_parser_registers_heal_flags():
    args = _build_parser().parse_args(["health", "heal", "--dry-run", "--json"])
    assert args.health_command == "heal" and args.dry_run and args.json


def test_cmd_health_heal_dispatches_and_prints_json(tmp_path, capsys, monkeypatch):
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return {"launched": [], "would_launch": ["k"], "attempts": [], "notes": []}

    monkeypatch.setattr(health, "run_heal", fake)
    args = _build_parser().parse_args(
        ["health", "heal", "--dry-run", "--json", "--data-root", str(tmp_path)]
    )
    assert health.cmd_health(args, SimpleNamespace()) == 0
    assert seen["dry_run"] is True and seen["jobs_dir"] == tmp_path / "logs" / "jobs"
    assert '"would_launch"' in capsys.readouterr().out


def test_recipe_flag_registers_then_heals_only_that_drive(tmp_path, capsys, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        health, "register_recipe", lambda inc, path, now_iso: "drive_failed:x:recipe"
    )
    monkeypatch.setattr(
        health, "run_heal",
        lambda **kw: seen.update(kw) or {"launched": [], "would_launch": [], "attempts": [], "notes": [], "outcomes": []},
    )  # fmt: skip
    args = _build_parser().parse_args(
        ["health", "heal", "--recipe", str(tmp_path / "resume.json"), "--data-root", str(tmp_path)]
    )
    assert health.cmd_health(args, SimpleNamespace()) == 0
    assert seen["only_key"] == "drive_failed:x:recipe"


def test_recipe_flag_with_an_unreadable_recipe_exits_nonzero(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(health, "register_recipe", lambda inc, path, now_iso: None)
    args = _build_parser().parse_args(
        ["health", "heal", "--recipe", str(tmp_path / "resume.json"), "--data-root", str(tmp_path)]
    )
    assert health.cmd_health(args, SimpleNamespace()) == 2
    assert "no readable resume.json" in capsys.readouterr().out


def test_recipe_dry_run_writes_no_incident_row(tmp_path, capsys):
    from gomoveshift.health import resume

    stages = tmp_path / "ev" / "stages" / "demo-drive-b" / "_stages"
    resume.write_recipe(stages, ["drive", "/in/demo-drive-b"], cwd="/w", sha="old", now_iso="t")
    resume.mark_phase(stages, "phase_a")
    inc = tmp_path / "inc"
    args = _build_parser().parse_args(
        ["health", "heal", "--recipe", str(stages / "resume.json"), "--dry-run", "--json",
         "--incidents-dir", str(inc), "--data-root", str(tmp_path), "--code-root", str(tmp_path),
         "--staging-sha", "abc1234"]
    )  # fmt: skip
    assert health.cmd_health(args, SimpleNamespace()) == 0
    assert not inc.exists() or not any(inc.iterdir())
    assert "drive_failed:demo-drive-b:recipe" in capsys.readouterr().out
