"""The CLI and cron drive entry points write ``run_origin.json`` themselves.

A reader that only discovers run dirs would show every CLI/cron run as ``unknown``; the writer at
the launch site is what makes provenance real. Console launches say ``console`` through the
environment, cron says ``cron``, and a bare CLI run says ``cli``.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest

from gomoveshift import run_origin
from gomoveshift.console import jobs, runs
from tests.test_cli_lf_day_propagation import _drive_args, _run_cmd_drive


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(run_origin.ORIGIN_ENV, raising=False)


def _sidecar(directory):
    return json.loads((directory / run_origin.SIDECAR).read_text())


def test_a_bare_run_is_recorded_as_cli_with_its_argv_and_start(tmp_path):
    run_origin.write_sidecar(tmp_path, argv=["/usr/bin/gomoveshift", "drive", "d"])
    data = _sidecar(tmp_path)
    assert data["origin"] == "cli"
    assert data["argv"] == ["gomoveshift", "drive"]
    assert data["started_at"]


@pytest.mark.parametrize("origin", ["cron", "console", "cli"])
def test_the_environment_names_the_origin(tmp_path, monkeypatch, origin):
    monkeypatch.setenv(run_origin.ORIGIN_ENV, origin)
    run_origin.write_sidecar(tmp_path, argv=["x"])
    assert _sidecar(tmp_path)["origin"] == origin


def test_an_unrecognised_environment_value_falls_back_to_cli(tmp_path, monkeypatch):
    monkeypatch.setenv(run_origin.ORIGIN_ENV, "spaceship")
    run_origin.write_sidecar(tmp_path, argv=["x"])
    assert _sidecar(tmp_path)["origin"] == "cli"


def test_an_existing_sidecar_is_never_overwritten_so_a_resume_keeps_its_origin(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(run_origin.ORIGIN_ENV, "cron")
    run_origin.write_sidecar(tmp_path, argv=["first"])
    monkeypatch.setenv(run_origin.ORIGIN_ENV, "cli")
    run_origin.write_sidecar(tmp_path, argv=["second"])
    assert _sidecar(tmp_path)["origin"] == "cron"
    assert _sidecar(tmp_path)["argv"] == ["first"]


def test_an_unwritable_dir_never_breaks_the_run(tmp_path):
    run_origin.write_sidecar(tmp_path / "missing" / "dir", argv=["x"])  # no raise


def test_the_console_reads_what_the_writer_wrote(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "data"))
    root = tmp_path / "root"
    (root / "trip" / "_gates").mkdir(parents=True)
    monkeypatch.setenv(run_origin.ORIGIN_ENV, "cron")
    run_origin.write_sidecar(root / "trip", argv=["gomoveshift", "drive", "d"])
    store = jobs.JobStore(tmp_path / "data" / "jobs.sqlite3")
    runs.reconcile(store, roots=[root])
    row = next(r for r in runs.list_runs(store) if r["output_dir"] == str(root / "trip"))
    assert row["origin"] == "cron"


def test_the_drive_command_writes_the_sidecar_into_its_run_dir(tmp_path, monkeypatch):
    out = tmp_path / "out"
    args = _drive_args(directory=str(tmp_path / "day1"), output_dir=str(out), dry_run=False)
    monkeypatch.setattr(sys, "argv", ["gomoveshift", "drive", "day1"])
    from gomoveshift.config import Config

    _run_cmd_drive(args, Config())
    assert _sidecar(out / "day1")["origin"] == "cli"


def test_a_dry_run_leaves_no_sidecar(tmp_path):
    out = tmp_path / "out"
    args = _drive_args(directory=str(tmp_path / "day1"), output_dir=str(out), dry_run=True)
    from gomoveshift.config import Config

    with patch.object(run_origin, "write_sidecar") as write:
        _run_cmd_drive(args, Config())
    write.assert_not_called()


def test_a_console_launched_job_passes_the_console_origin_to_its_child(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "data"))
    seen = {}

    class FakeProc:
        pid = None

    def fake_popen(argv, **kw):
        seen["env"] = kw["env"]
        return FakeProc()

    monkeypatch.setattr(jobs.subprocess, "Popen", fake_popen)
    store = jobs.JobStore(tmp_path / "data" / "jobs.sqlite3")
    monkeypatch.setattr(jobs, "_validate_argv", lambda argv: None)
    jobs.launch(store, "drive", ["gomoveshift", "drive", "d"], tmp_path / "logs", env={"A": "1"})
    assert seen["env"][run_origin.ORIGIN_ENV] == "console"
    assert seen["env"]["A"] == "1"


def test_the_sidecar_keeps_the_subcommand_but_no_paths_or_secret_like_args(tmp_path):
    run_origin.write_sidecar(
        tmp_path,
        argv=[
            "/home/x/venv/bin/gomoveshift",
            "pipeline",
            "/mnt/d/day/clip.mp4",
            "--token",
            "s3cr3t",
        ],
    )
    raw = (tmp_path / run_origin.SIDECAR).read_text()
    assert _sidecar(tmp_path)["argv"] == ["gomoveshift", "pipeline"]
    for leaked in ("/home/x", "/mnt/d", "s3cr3t", "--token"):
        assert leaked not in raw


def test_a_flag_first_argv_stores_only_the_program_name(tmp_path):
    run_origin.write_sidecar(tmp_path, argv=["/x/gomoveshift", "--verbose", "drive"])
    assert _sidecar(tmp_path)["argv"] == ["gomoveshift"]


def test_the_pipeline_command_writes_the_sidecar_into_its_output_dir(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from gomoveshift.cli.handlers import pipeline as handler

    video = tmp_path / "clip.mp4"
    video.touch()
    out = tmp_path / "out"
    args = SimpleNamespace(video=video, no_tmux=True, output_dir=out)
    config = SimpleNamespace(seo=SimpleNamespace(validate_paths_exist=lambda: None))
    monkeypatch.setattr(handler, "is_in_tmux", lambda: True)
    monkeypatch.setattr(sys, "argv", ["gomoveshift", "pipeline", str(video)])

    class _StopError(Exception):
        pass

    def _stop(**_kw):
        raise _StopError

    monkeypatch.setattr(handler, "configure_logging", _stop)
    with pytest.raises(_StopError):
        handler.cmd_pipeline(args, config)
    assert _sidecar(out)["argv"] == ["gomoveshift", "pipeline"]
