"""``gomoveshift cleanup <project> --abandoned`` reclaims a killed run's stage cache.

2026-11-08: a relaunch of ``nctopa-2026-03-05-rec0063`` under a different
``GMS_INTERMEDIATE_DIR`` left the earlier run's 139 GB ``_stages`` orphaned —
no run lock held, no live ``gomoveshift drive``, ``phase_d`` never reached.
``cleanup <project> --min-age 0`` answered ``SKIP … Pipeline not complete`` and
exit 2: every incomplete project was untouchable, so the only reclaim path was a
hand ``rm`` of a path the agent cannot type.

Rule pinned here: with ``--abandoned`` AND a named project, an incomplete stage
dir is planned for removal when every rule holds — no completion marker, no run
lock held by a live pid, no live ``gomoveshift drive`` process anywhere, no
``no-auto-resume`` hold, old enough for ``--min-age`` (unless a backlog supersede record names
the project: that run is dead by record). The first failing rule is
the skip reason. Without the flag the project is skipped as before, with a hint.
``--confirm`` removes the ``_stages`` dir only; the output dir and every other
project are untouched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gomoveshift.cli.handlers import cleanup as cleanup_handler
from gomoveshift.health.resume import HOLD_NAME
from gomoveshift.pipeline.cleanup import PipelineCleanup

OLD = 1_700_000_000


def _killed_run(
    base: Path, name: str = "nctopa-2026-03-05-rec0063", *, phases=("phase_a",)
) -> Path:
    project = base / name
    stages = project / "_stages"
    lfs = stages / "day_2026-03-05" / "regular_lfs"
    lfs.mkdir(parents=True)
    (stages / "episode_01_2026-03-05.mp4").write_bytes(b"e" * 4000)
    (lfs / "lf_01_no_stops.mp4").write_bytes(b"l" * 3000)
    (stages / "pipeline_state.json").write_text(json.dumps({"phases_completed": list(phases)}))
    (stages / "resume.json").write_text(json.dumps({"schema": "resume-recipe/1", "argv": []}))
    os.utime(stages, (OLD, OLD))
    return project


def _cleanup(tmp_path: Path, **kw) -> PipelineCleanup:
    return PipelineCleanup(
        output_base=tmp_path / "out",
        staging_dir=tmp_path / "stg",
        stage_cache_base=tmp_path / "cache",
        **kw,
    )


def _lock(project: Path, pid: int) -> None:
    (project / "_stages" / "pipeline_state.lock").write_text(json.dumps({"pid": pid}))


def test_without_the_flag_the_skip_says_how_to_reclaim_an_abandoned_run(tmp_path):
    project = _killed_run(tmp_path / "cache")
    plan = _cleanup(tmp_path).plan_cleanup([project], min_age_days=0)
    assert plan.files == [] and plan.abandoned == []
    (skip,) = plan.skipped
    assert "Pipeline not complete" in skip["reason"]
    assert "--abandoned" in skip["reason"]


def test_a_killed_run_is_planned_as_one_stage_dir_with_its_size(tmp_path):
    project = _killed_run(tmp_path / "cache")
    plan = _cleanup(tmp_path, drive_running_fn=lambda: False).plan_cleanup(
        [project], min_age_days=0, abandoned=True
    )
    assert plan.files == [] and plan.skipped == []
    (entry,) = plan.abandoned
    assert entry["path"] == str(project / "_stages")
    assert entry["size"] == 7000 + len(
        (project / "_stages" / "pipeline_state.json").read_bytes()
    ) + len((project / "_stages" / "resume.json").read_bytes())
    assert "abandoned" in entry["reason"]


def test_a_complete_project_is_never_abandoned(tmp_path):
    project = _killed_run(tmp_path / "cache", phases=("phase_a", "phase_d"))
    plan = _cleanup(tmp_path, drive_running_fn=lambda: False).plan_cleanup(
        [project], min_age_days=0, abandoned=True
    )
    assert plan.abandoned == []


def test_a_run_lock_held_by_a_live_pid_refuses(tmp_path):
    project = _killed_run(tmp_path / "cache")
    _lock(project, os.getpid())
    plan = _cleanup(tmp_path, drive_running_fn=lambda: False).plan_cleanup(
        [project], min_age_days=0, abandoned=True
    )
    assert plan.abandoned == []
    (skip,) = plan.skipped
    assert "run lock" in skip["reason"] and str(os.getpid()) in skip["reason"]


def test_a_stale_run_lock_does_not_refuse(tmp_path):
    project = _killed_run(tmp_path / "cache")
    _lock(project, 1)  # pid 1 is init: alive, so stub the probe
    with patch("gomoveshift.pipeline.orchestrator_lock._pid_is_alive", return_value=False):
        plan = _cleanup(tmp_path, drive_running_fn=lambda: False).plan_cleanup(
            [project], min_age_days=0, abandoned=True
        )
    assert [e["path"] for e in plan.abandoned] == [str(project / "_stages")]


def test_a_live_drive_anywhere_refuses(tmp_path):
    project = _killed_run(tmp_path / "cache")
    plan = _cleanup(tmp_path, drive_running_fn=lambda: True).plan_cleanup(
        [project], min_age_days=0, abandoned=True
    )
    assert plan.abandoned == []
    (skip,) = plan.skipped
    assert "drive" in skip["reason"] and "running" in skip["reason"]


def test_a_held_run_is_waiting_not_abandoned(tmp_path):
    project = _killed_run(tmp_path / "cache")
    (project / "_stages" / HOLD_NAME).write_text("operator hold")
    plan = _cleanup(tmp_path, drive_running_fn=lambda: False).plan_cleanup(
        [project], min_age_days=0, abandoned=True
    )
    assert plan.abandoned == []
    (skip,) = plan.skipped
    assert HOLD_NAME in skip["reason"]


def test_a_young_run_is_still_too_recent(tmp_path):
    project = _killed_run(tmp_path / "cache")
    now = int(__import__("time").time())
    os.utime(project / "_stages", (now, now))
    plan = _cleanup(tmp_path, drive_running_fn=lambda: False).plan_cleanup(
        [project], min_age_days=7, abandoned=True
    )
    assert plan.abandoned == []
    assert any("Too recent" in s["reason"] for s in plan.skipped)


def _supersede_record(where: Path, project: str) -> None:
    (where / "superseded.json").write_text(
        json.dumps({"schema": "backlog-supersede/1", "project": project})
    )


def test_a_young_run_the_backlog_superseded_clears_the_age_gate(tmp_path):
    """2026-11-10: rec0063's superseded run (moved under ``_superseded/<name>-<stamp>``, nested
    layout) was 2 days old, so ``--abandoned`` answered ``Too recent`` and freed nothing while the
    live run waited on that 139 GB. The supersede record already says the run is dead."""
    moved = tmp_path / "cache" / "_superseded" / "nctopa-2026-03-05-rec0063-20261110T160126Z"
    project = _killed_run(moved)
    now = int(__import__("time").time())
    os.utime(project / "_stages", (now, now))
    _supersede_record(moved, project.name)
    plan = _cleanup(tmp_path, drive_running_fn=lambda: False).plan_cleanup(
        [project], min_age_days=7, abandoned=True
    )
    assert [e["path"] for e in plan.abandoned] == [str(project / "_stages")]


def test_a_supersede_record_for_another_project_keeps_the_age_gate(tmp_path):
    moved = tmp_path / "cache" / "_superseded" / "x-20261110T160126Z"
    project = _killed_run(moved)
    now = int(__import__("time").time())
    os.utime(project / "_stages", (now, now))
    _supersede_record(moved, "some-other-drive")
    plan = _cleanup(tmp_path, drive_running_fn=lambda: False).plan_cleanup(
        [project], min_age_days=7, abandoned=True
    )
    assert plan.abandoned == []
    assert any("Too recent" in s["reason"] for s in plan.skipped)


def test_a_superseded_run_is_still_refused_while_a_drive_runs(tmp_path):
    moved = tmp_path / "cache" / "_superseded" / "nctopa-2026-03-05-rec0063-20261110T160126Z"
    project = _killed_run(moved)
    _supersede_record(moved, project.name)
    plan = _cleanup(tmp_path, drive_running_fn=lambda: True).plan_cleanup(
        [project], min_age_days=7, abandoned=True
    )
    assert plan.abandoned == []
    assert any("running" in s["reason"] for s in plan.skipped)


def test_execute_removes_the_stage_dir_only(tmp_path):
    cache = tmp_path / "cache"
    project = _killed_run(cache)
    other = _killed_run(cache, "other-drive-2026-03-06-rec0070")
    out_project = tmp_path / "out" / project.name
    out_project.mkdir(parents=True)
    (out_project / "youtube_uploads.json").write_text("{}")
    cleanup = _cleanup(tmp_path, drive_running_fn=lambda: False)
    plan = cleanup.plan_cleanup([project], min_age_days=0, abandoned=True)
    assert [e["path"] for e in plan.abandoned] == [str(project / "_stages")]
    assert cleanup.execute_plan(plan) == 1
    assert not (project / "_stages").exists()
    assert (other / "_stages" / "pipeline_state.json").exists()
    assert (out_project / "youtube_uploads.json").exists()


def test_execute_never_follows_a_symlinked_stage_dir(tmp_path):
    cache = tmp_path / "cache"
    project = _killed_run(cache)
    real = tmp_path / "elsewhere"
    (project / "_stages").rename(real)
    (project / "_stages").symlink_to(real)
    cleanup = _cleanup(tmp_path, drive_running_fn=lambda: False)
    plan = cleanup.plan_cleanup([project], min_age_days=0, abandoned=True)
    assert plan.abandoned == []
    assert any("symlink" in s["reason"] for s in plan.skipped)
    assert (real / "pipeline_state.json").exists()


def _cli(tmp_path: Path, project: Path, **flags):
    out, stg = tmp_path / "out", tmp_path / "stg"
    out.mkdir(exist_ok=True)
    stg.mkdir(exist_ok=True)
    args = SimpleNamespace(
        project=project.name if project else None,
        confirm=False,
        min_age=0,
        archive_base=None,
        abandoned=False,
    )
    for k, v in flags.items():
        setattr(args, k, v)
    with (
        patch("gomoveshift.paths.gms_output_dir", return_value=out),
        patch("gomoveshift.paths.gms_staging_dir", return_value=stg),
        patch("gomoveshift.paths.gms_intermediate_base", return_value=tmp_path / "cache"),
        patch("gomoveshift.health.heal._pgrep_drive", return_value=False),
    ):
        return cleanup_handler.cmd_cleanup(args, None)


def test_cli_dry_run_lists_the_abandoned_dir_and_deletes_nothing(tmp_path, capsys):
    project = _killed_run(tmp_path / "cache")
    rc = _cli(tmp_path, project, abandoned=True)
    text = capsys.readouterr().out
    assert rc == 0, text
    assert f"WOULD REMOVE ABANDONED STAGE DIR: {project / '_stages'}" in text
    assert "dry run" in text
    assert (project / "_stages" / "pipeline_state.json").exists()


def test_cli_confirm_removes_it_and_reports(tmp_path, capsys):
    project = _killed_run(tmp_path / "cache")
    rc = _cli(tmp_path, project, abandoned=True, confirm=True)
    text = capsys.readouterr().out
    assert rc == 0, text
    assert f"REMOVE ABANDONED STAGE DIR: {project / '_stages'}" in text
    assert not (project / "_stages").exists()


def test_cli_abandoned_without_a_project_name_is_refused(tmp_path, capsys):
    _killed_run(tmp_path / "cache")
    rc = _cli(tmp_path, None, abandoned=True)
    err = capsys.readouterr().err
    assert rc == 1
    assert "--abandoned" in err and "project" in err


def test_cli_without_the_flag_keeps_todays_exit_2_and_hints(tmp_path, capsys):
    project = _killed_run(tmp_path / "cache")
    rc = _cli(tmp_path, project)
    text = capsys.readouterr().out
    assert rc == 2
    assert "SKIP" in text and "--abandoned" in text


def test_cli_parser_accepts_the_flag():
    from gomoveshift.cli.parser import _build_parser

    args = _build_parser().parse_args(["cleanup", "nctopa-2026-03-05-rec0063", "--abandoned"])
    assert args.abandoned is True and args.project == "nctopa-2026-03-05-rec0063"
    assert _build_parser().parse_args(["cleanup"]).abandoned is False


def test_default_drive_probe_is_the_heal_modules():
    """The live-drive probe is ``health.heal._pgrep_drive`` — one pattern for the whole codebase,
    looked up at call time so a patch on heal is honoured."""
    from gomoveshift.pipeline import cleanup as cleanup_mod

    with patch("gomoveshift.health.heal._pgrep_drive", return_value=True) as probe:
        assert cleanup_mod._live_drive_probe() is True
    probe.assert_called_once_with()


def test_importing_publishing_queue_first_does_not_cycle_through_cleanup():
    """CI (#257 run 42565899313): ``tests/e2e/conftest.py`` imports ``publishing.queue`` first;
    queue → pipeline._file_lock → pipeline/__init__ → pipeline.cleanup → health.heal → cadence →
    publishing.queue (partially initialised) raised ImportError. The heal import stays lazy."""
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import gomoveshift.publishing.queue; import gomoveshift.pipeline.cleanup",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
