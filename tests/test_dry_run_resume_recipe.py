"""A discovery-only drive never writes the resume recipe; a heal relaunch survives cross-fs.

Regression: ``drive --label-dry-run`` on an evidence copy rewrote the copy's
``resume.json`` (``phase_reached: None``), so a later ``health heal`` refused "cannot map phase
None" -- a run that renders nothing replaced the recipe of the run that did. ``--list-lfs``,
``--label-dry-run`` and ``--dry-run`` leave an existing recipe byte-identical and create none.

The heal relaunch drops ``--allow-cross-fs-source`` (``heal._HOST_ONLY_FLAGS``). That flag only
lowers the cross-filesystem PERFORMANCE report from WARNING to INFO, so the relaunch must still
pass the source-footage check when the footage sits on a different filesystem (symlinks on the work volume pointing at the footage volume).
"""

from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config
from gomoveshift.health import heal, resume
from gomoveshift.pipeline import preflight

_PRIOR = {
    "argv": ["py", "-m", "gomoveshift", "drive", "/in/demo"],
    "cwd": "/",
    "phase_reached": "phase_a",
}


def _cmd_drive(argv, cfg):
    from gomoveshift.cli.handlers import drive as drive_handler

    args = _build_parser().parse_args(argv)
    with (
        patch.object(drive_handler, "is_in_tmux", return_value=True),
        patch.object(drive_handler, "configure_logging"),
        patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator") as orch_cls,
        patch.object(drive_handler.sys, "argv", ["gomoveshift", *argv]),
    ):
        orch_cls.return_value.run.return_value = SimpleNamespace(
            success=True, stages_completed=[], drive_plan=None
        )
        drive_handler.cmd_drive(args, cfg)
    return drive_handler._stages_base(args, cfg) / resume.RECIPE_NAME


def _argv(tmp_path, *flags):
    src = tmp_path / "Demo Drive"
    src.mkdir(exist_ok=True)
    return [
        "drive",
        str(src),
        "--output-dir",
        str(tmp_path / "out"),
        "--region",
        "ireland",
        "--publish",
        "sun",
        "--force",
        "--no-tmux",
        "--no-review",
        "--no-auto-publish-journal",
        *flags,
    ]


@pytest.fixture
def stages(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_INTERMEDIATE_DIR", str(tmp_path / "cache"))
    return tmp_path


@pytest.mark.parametrize("flag", ["--label-dry-run", "--list-lfs", "--dry-run"])
def test_discovery_run_leaves_an_existing_recipe_byte_identical(stages, flag):
    argv = _argv(stages, flag)
    path = _cmd_drive(_argv(stages), Config())  # a real drive wrote the recipe first
    path.write_text(json.dumps(_PRIOR), encoding="utf-8")
    before = path.read_bytes()
    assert _cmd_drive(argv, Config()) == path
    assert path.read_bytes() == before


@pytest.mark.parametrize("flag", ["--label-dry-run", "--list-lfs", "--dry-run"])
def test_discovery_run_creates_no_recipe(stages, flag):
    assert not _cmd_drive(_argv(stages, flag), Config()).exists()


def test_real_drive_still_writes_the_recipe(stages):
    path = _cmd_drive(_argv(stages, "--fresh"), Config())
    got = json.loads(path.read_text())
    assert got["phase_reached"] is None and "--fresh" not in got["argv"]


def test_labels_only_save_never_marks_the_recipe_phase(tmp_path):
    from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

    resume.write_recipe(tmp_path, ["drive"], cwd="/", sha=None, now_iso="t")
    before = (tmp_path / resume.RECIPE_NAME).read_bytes()
    orch = object.__new__(DrivePipelineOrchestrator)
    orch._labels_only = True
    with patch("gomoveshift.pipeline.drive_orchestrator.save_drive_state"):
        orch._save_state(
            tmp_path / "pipeline_state.json", SimpleNamespace(phases_completed=["phase_a"])
        )
    assert (tmp_path / resume.RECIPE_NAME).read_bytes() == before


def test_heal_relaunch_without_cross_fs_flag_passes_source_check(tmp_path, monkeypatch, caplog):
    dvol = tmp_path / "dvol" / "trip"
    dvol.mkdir(parents=True)
    links = tmp_path / "cache" / "sources" / "trip"
    links.mkdir(parents=True)
    for name in ("GX010001.MP4", "GX010002.MP4"):
        (dvol / name).write_bytes(b"\0" * 64)
        (links / name).symlink_to(dvol / name)
    d_root = os.path.realpath(tmp_path / "dvol")
    monkeypatch.setattr(
        preflight, "_device_of", lambda p: 2 if os.path.realpath(p).startswith(d_root) else 1
    )
    code_root = tmp_path / "code"
    py = code_root / heal.SERVICE_DIR / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    recipe = ["py", "-m", "gomoveshift", *_argv(tmp_path), "--allow-cross-fs-source"]
    recipe[4] = str(links)

    argv, _cwd = heal.bind_to_code_root(code_root, recipe)
    args = _build_parser().parse_args(argv[3:])
    assert args.allow_cross_fs_source is False

    (tmp_path / "stg").mkdir()
    with caplog.at_level(logging.INFO, logger=preflight.logger.name):
        preflight.check_source_filesystem(
            links, tmp_path / "stg", allow_cross_fs=args.allow_cross_fs_source
        )
    perf = [r for r in caplog.records if "PERFORMANCE" in r.getMessage()]
    assert [r.levelno for r in perf] == [logging.WARNING]
