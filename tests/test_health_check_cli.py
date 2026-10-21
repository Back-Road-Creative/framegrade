"""CLI wiring for ``gomoveshift health check``."""

from __future__ import annotations

import json
from types import SimpleNamespace

from gomoveshift.cli import _COMMAND_MAP
from gomoveshift.cli.handlers.health import cmd_health
from gomoveshift.cli.parser import _build_parser


def test_parser_registers_health_check():
    assert _COMMAND_MAP["health"] is cmd_health
    args = _build_parser().parse_args(["health", "check", "--json", "--staging-sha", "x"])
    assert args.command == "health" and args.json and args.staging_sha == "x"


def test_cmd_health_exit_codes(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_NOTIFY_URL", "")
    monkeypatch.setenv("HOME", str(tmp_path))
    args = _build_parser().parse_args(
        [
            "health",
            "check",
            "--json",
            "--data-root",
            str(tmp_path),
            "--incidents-dir",
            str(tmp_path / "inc"),
            "--staging-sha",
            "x",
        ]
    )
    cfg = SimpleNamespace(seo=SimpleNamespace(storage_path="rel"))
    assert cmd_health(args, cfg) == 0
    log = tmp_path / "l.log"
    log.write_text("Drive pipeline failed: boom\n")
    d = tmp_path / "logs" / "jobs" / "r"
    d.mkdir(parents=True)
    (d / "job.json").write_text(
        json.dumps(
            {
                "schema": "job/1",
                "run_id": "r",
                "name": "drive-demo",
                "argv": ["drive"],
                "created_at": 1.0,
                "expect": [str(log)],
                "result": {"state": "failed", "exit_code": 1, "ended_at": 2.0},
            }
        )
    )
    assert cmd_health(args, cfg) == 1
