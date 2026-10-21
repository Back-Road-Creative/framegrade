"""Resume recipe (``resume.json``): written by a drive for itself, read by ``health heal``."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from types import SimpleNamespace
from unittest import mock

from gomoveshift.health import resume


def test_write_recipe_drops_fresh_and_records_context(tmp_path):
    path = resume.write_recipe(
        tmp_path / "_stages",
        ["drive", "/in/demo-drive-a", "--fresh", "--publish", "fri"],
        cwd="/work/mirror",
        sha="abc1234",
        now_iso="2026-01-01T00:00:00+00:00",
    )
    assert path == tmp_path / "_stages" / "resume.json"
    got = json.loads(path.read_text())
    assert got["argv"] == [
        sys.executable,
        "-m",
        "gomoveshift",
        "drive",
        "/in/demo-drive-a",
        "--publish",
        "fri",
    ]
    assert got["cwd"] == "/work/mirror" and got["sha"] == "abc1234"
    assert got["started_at"] == "2026-01-01T00:00:00+00:00" and got["phase_reached"] is None


def test_write_recipe_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert resume.write_recipe(blocker / "sub", ["drive"], cwd="/", sha=None, now_iso="t") is None


def test_mark_phase_updates_and_ignores_missing(tmp_path):
    resume.mark_phase(tmp_path, "phase_a")  # no recipe yet: silent no-op
    path = resume.write_recipe(tmp_path, ["drive"], cwd="/", sha=None, now_iso="t")
    resume.mark_phase(tmp_path, "phase_bc_day_first")
    assert json.loads(path.read_text())["phase_reached"] == "phase_bc_day_first"


def test_read_recipe_rejects_unreadable_or_argv_less(tmp_path):
    assert resume.read_recipe(tmp_path / "nope.json") is None
    bad = tmp_path / "resume.json"
    bad.write_text("{not json")
    assert resume.read_recipe(bad) is None
    bad.write_text(json.dumps({"argv": "drive", "cwd": "/"}))
    assert resume.read_recipe(bad) is None
    bad.write_text(json.dumps({"argv": ["drive"], "cwd": "/"}))
    assert resume.read_recipe(bad)["argv"] == ["drive"]


def test_recipe_line_round_trips(tmp_path):
    line = resume.recipe_line(tmp_path / "_stages" / "resume.json")
    assert resume.parse_recipe_path(["noise", f"2026 INFO {line}"]) == str(
        tmp_path / "_stages" / "resume.json"
    )
    assert resume.parse_recipe_path(["nothing here"]) is None


def test_drive_handler_writes_recipe_before_the_run(tmp_path):
    from gomoveshift.cli.handlers import drive

    args = Namespace(output_dir=tmp_path / "out", directory="/in/Demo Drive A", phase=None)
    cfg = SimpleNamespace(paths=SimpleNamespace(intermediate_folder="_stages"))
    with mock.patch.object(drive.sys, "argv", ["gomoveshift", "drive", "/in/x", "--fresh"]):
        path = drive._write_resume_recipe(args, cfg)
    got = json.loads(path.read_text())
    assert "--fresh" not in got["argv"] and got["argv"][-2:] == ["drive", "/in/x"]
    assert path.parent == drive._stages_base(args, cfg)


def test_drive_handler_skips_write_for_a_single_phase_run(tmp_path):
    """A heal-launched ``--phase`` run must not overwrite the full recipe it was built from."""
    from gomoveshift.cli.handlers import drive

    args = Namespace(output_dir=tmp_path / "out", directory="/in/demo", phase="D")
    cfg = SimpleNamespace(paths=SimpleNamespace(intermediate_folder="_stages"))
    assert drive._write_resume_recipe(args, cfg) is None
    assert not (drive._stages_base(args, cfg) / "resume.json").exists()


def test_orchestrator_save_state_marks_last_completed_phase(tmp_path):
    from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

    resume.write_recipe(tmp_path, ["drive"], cwd="/", sha=None, now_iso="t")
    orch = object.__new__(DrivePipelineOrchestrator)
    state = SimpleNamespace(phases_completed=["phase_a", "phase_bc_day_first"])
    with mock.patch("gomoveshift.pipeline.drive_orchestrator.save_drive_state"):
        orch._save_state(tmp_path / "pipeline_state.json", state)
    assert (
        json.loads((tmp_path / "resume.json").read_text())["phase_reached"] == "phase_bc_day_first"
    )
