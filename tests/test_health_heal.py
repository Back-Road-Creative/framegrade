"""``health heal``: audit-only auto-resume of a failed drive, behind five guards.

Every side effect (launcher, notify, git, process probe, slot, clock) is injected; nothing here
launches a real job or pushes."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gomoveshift.health import heal, resume
from gomoveshift.health.store import load_all, new_incident, write_row

NOW = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)
FUTURE = datetime(2026, 1, 16, 12, 0, tzinfo=timezone.utc)
SHA = "abcdef1234567890"  # pragma: allowlist secret


class Rig:
    def __init__(self, tmp_path: Path, *, drive_args=None, venv=True, **row_over):
        self.tmp = tmp_path
        self.svc = tmp_path
        self.python = self.svc / ".venv" / "bin" / "python"
        if venv:
            self.python.parent.mkdir(parents=True)
            self.python.write_text("")
        self.inc = tmp_path / "inc"
        self.jobs = tmp_path / "jobs"
        self.launches: list = []
        self.pushes: list = []
        self.running = False
        self.changed: list = []  # git is pathspec-filtered: only generator files can appear
        self.slot = FUTURE
        self.launch_ok = True
        stages = tmp_path / "stages"
        resume.write_recipe(
            stages,
            drive_args or ["drive", "/in/demo-drive-a", "--publish", "fri"],
            cwd="/work",
            sha="old",
            now_iso="t",
        )
        row = new_incident(
            "drive_failed", "drive_failed:demo-drive-a:r1", "demo-drive-a.json", "t",
            drive="demo-drive-a", phase="phase_d", gate=None, error="boom", log_path=None,
            job_name="gms-fri-demo", job_run_id="r1", failed_at=None, staging_sha="old",
            resume_recipe=str(stages / "resume.json"), resume_attempts=[],
        )  # fmt: skip
        row.update(row_over)
        write_row(self.inc, row)

    def run(self, *, sha=SHA, dry_run=False, now=NOW, only_key=None):
        return heal.run_heal(
            incidents_dir=self.inc, jobs_dir=self.jobs, code_root=self.tmp, mirror_sha=sha,
            now=now, dry_run=dry_run, only_key=only_key,
            launcher=lambda argv, cwd: (self.launches.append((argv, cwd)) or (self.launch_ok, "")),
            notify_fn=lambda e, i, s: self.pushes.append((e, s)) or True,
            git_changed=lambda a, b, paths: self.changed,
            drive_running=lambda jobs: self.running,
            slot_fn=lambda row, recipe: self.slot,
        )  # fmt: skip

    def row(self):
        return load_all(self.inc)[0][0]


def test_launches_audit_only_resume_with_exact_argv_and_records_attempt(tmp_path):
    rig = Rig(tmp_path)
    out = rig.run()
    assert out["launched"] == ["drive_failed:demo-drive-a:r1"]
    argv, cwd = rig.launches[0]
    tail = argv[argv.index("--") + 1 :]
    assert tail[-2:] == ["--skip-upload", "--no-youtube-upload"] and "--phase" not in tail
    assert "--fresh" not in tail and tail[3:6] == ["drive", "/in/demo-drive-a", "--publish"]
    assert argv[argv.index("--name") + 1] == "gms-heal-demo-drive-a-abcdef1"
    assert "--heavy" in argv and "--detach" in argv and argv[argv.index("--stall") + 1] == "86400"
    assert argv[argv.index("--log-root") + 1] == str(rig.jobs)
    assert tail[:3] == [str(rig.python), "-m", "gomoveshift"]
    assert cwd == str(rig.svc)
    att = rig.row()["resume_attempts"][0]
    assert att["action"] == "launched" and att["mirror_sha"] == SHA
    assert att["job_name"] == "gms-heal-demo-drive-a-abcdef1" and att["slot"] == FUTURE.isoformat()
    assert att["reason"] and att["at"]
    assert [e for e, _ in rig.pushes] == ["health_heal_launched"]


def test_refuses_without_a_readable_recipe_and_never_guesses(tmp_path):
    rig = Rig(tmp_path, resume_recipe=None)
    out = rig.run()
    assert rig.launches == [] and out["launched"] == []
    att = rig.row()["resume_attempts"][0]
    assert att["action"] == "refused" and "resume.json" in att["reason"]
    Path(tmp_path / "stages" / "resume.json").write_text("{bad")
    rig2 = Rig(tmp_path / "b")
    Path(tmp_path / "b" / "stages" / "resume.json").write_text("{bad")
    rig2.run()
    assert rig2.launches == [] and rig2.row()["resume_attempts"][0]["action"] == "refused"


def test_an_unmappable_phase_still_resumes_every_remaining_phase(tmp_path):
    rig = Rig(tmp_path, phase=None)
    rig.run()
    argv, _cwd = rig.launches[0]
    assert "--phase" not in argv[argv.index("--") + 1 :]
    assert rig.row()["resume_attempts"][0]["action"] == "launched"


def test_a_phase_a_or_bc_failure_resumes_every_phase_not_the_failed_one_alone(tmp_path):
    # `--phase A` is the drive CLI's single-phase mode: phase A passes, then the DoD gate fails
    # (lf_count 0) because B-D never ran. Heal must run the plain resume instead.
    for i, phase in enumerate(["phase_a", "phase_bc_day_first", "Phase B", "phase_d"]):
        rig = Rig(tmp_path / str(i), phase=phase)
        rig.run()
        argv, _cwd = rig.launches[0]
        tail = argv[argv.index("--") + 1 :]
        assert "--phase" not in tail and tail[-2:] == ["--skip-upload", "--no-youtube-upload"]


def test_no_auto_resume_marker_refuses_on_every_mirror_sha(tmp_path):
    rig = Rig(tmp_path)
    (tmp_path / "stages" / resume.HOLD_NAME).write_text("one authorized run; no repeat\n")
    out = rig.run()
    rig.run(sha="1234567fedcba")  # pragma: allowlist secret
    assert rig.launches == [] and out["launched"] == []
    att = rig.row()["resume_attempts"]
    assert [a["action"] for a in att] == ["refused", "refused"]
    assert resume.HOLD_NAME in att[0]["reason"]


def test_no_auto_resume_marker_shows_in_the_dry_run(tmp_path):
    rig = Rig(tmp_path)
    (tmp_path / "stages" / resume.HOLD_NAME).write_text("")
    out = rig.run(dry_run=True)
    assert out["would_launch"] == [] and out["attempts"][0]["action"] == "refused"


def test_guard_a_once_per_mirror_sha(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    rig.run()
    assert len(rig.launches) == 1
    assert len(rig.row()["resume_attempts"]) == 1  # a repeat skip is not re-recorded
    rig.run(sha="1234567fedcba")  # pragma: allowlist secret
    assert len(rig.launches) == 2


def _failed_heal_row(rig: Rig, n: int) -> None:
    # `health check` files a failed heal job as its own drive_failed row (keyed by that job's
    # name), still pointing at the drive's one recipe.
    first = next(r for r in load_all(rig.inc)[0] if r["key"] == "drive_failed:demo-drive-a:r1")
    write_row(
        rig.inc,
        {
            **first,
            "key": f"drive_failed:gms-heal-demo-drive-a-abcdef1:h{n}",
            "file": f"heal-{n}.json",
            "drive": "gms-heal-demo-drive-a-abcdef1",
            "job_name": "gms-heal-demo-drive-a-abcdef1",
            "job_run_id": f"h{n}",
            "resume_attempts": [],
        },
    )


def test_guard_a_is_once_per_recipe_not_once_per_row(tmp_path):
    # US-64, 11-06: each failed heal became a new row, and each new row healed again on the same
    # SHA, so one deterministic failure re-ran hourly and kept the staging sync deferred forever.
    rig = Rig(tmp_path)
    rig.run()
    _failed_heal_row(rig, 1)
    out = rig.run()
    assert len(rig.launches) == 1 and out["launched"] == []
    _failed_heal_row(rig, 2)
    rig.run()
    assert len(rig.launches) == 1
    rig.run(sha="1234567fedcba")  # pragma: allowlist secret
    assert len(rig.launches) == 2  # new code: exactly one more resume of that recipe


def test_a_heal_of_a_heal_does_not_repeat_the_audit_only_flags(tmp_path):
    rig = Rig(
        tmp_path, drive_args=["drive", "/in/demo-drive-a", "--skip-upload", "--no-youtube-upload"]
    )
    rig.run()
    tail = rig.launches[0][0][rig.launches[0][0].index("--") + 1 :]
    assert tail.count("--skip-upload") == 1 and tail.count("--no-youtube-upload") == 1


def test_guard_b_skips_while_a_drive_job_is_running(tmp_path):
    rig = Rig(tmp_path)
    rig.running = True
    rig.run()
    att = rig.row()["resume_attempts"][0]
    assert rig.launches == [] and att["action"] == "skipped" and "running" in att["reason"]
    rig.running = False
    rig.run()
    assert len(rig.launches) == 1


def test_guard_c_skips_once_the_publish_slot_has_passed(tmp_path):
    rig = Rig(tmp_path)
    rig.slot = datetime(2026, 1, 9, 12, 0, tzinfo=timezone.utc)
    rig.run()
    att = rig.row()["resume_attempts"][0]
    assert rig.launches == [] and att["action"] == "skipped" and "slot" in att["reason"]
    assert att["slot"] == rig.slot.isoformat()


def test_guard_c_unknown_slot_is_a_recorded_skip(tmp_path):
    rig = Rig(tmp_path)
    rig.slot = None
    rig.run()
    assert rig.launches == [] and rig.row()["resume_attempts"][0]["action"] == "skipped"


def test_guard_d_metadata_gate_needs_a_generator_change(tmp_path):
    rig = Rig(tmp_path, gate="Metadata quality")
    rig.run()
    att = rig.row()["resume_attempts"][0]
    assert rig.launches == [] and att["action"] == "skipped" and "generator" in att["reason"]
    rig.changed = [heal.GENERATOR_PATHS[0] + "x.py"]
    rig.run(sha="1234567fedcba")  # pragma: allowlist secret
    assert len(rig.launches) == 1


def test_guard_d_does_not_hold_back_a_non_metadata_gate(tmp_path):
    rig = Rig(tmp_path, gate="Audio loudness")
    rig.run()
    assert len(rig.launches) == 1


def test_guard_d_passes_the_generator_paths_to_git(tmp_path):
    rig = Rig(tmp_path, gate="metadata")
    seen = []
    heal.run_heal(
        incidents_dir=rig.inc, jobs_dir=rig.jobs, code_root=tmp_path, mirror_sha=SHA, now=NOW,
        launcher=lambda a, c: (True, ""), notify_fn=lambda *a: True,
        git_changed=lambda a, b, paths: seen.append((a, b, tuple(paths))) or [],
        drive_running=lambda j: False, slot_fn=lambda r, c: FUTURE,
    )  # fmt: skip
    assert seen == [("old", SHA, tuple(heal.GENERATOR_PATHS))]


def test_outcome_is_pushed_once_when_the_heal_job_finishes(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    name = rig.row()["resume_attempts"][0]["job_name"]
    rig.run()
    assert [e for e, _ in rig.pushes] == ["health_heal_launched"]  # still running: no outcome
    d = rig.jobs / "h1"
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({
        "schema": "job/1", "run_id": "h1", "name": name, "argv": ["x"], "created_at": 9e9,
        "result": {"state": "succeeded", "exit_code": 0},
    }))  # fmt: skip
    rig.run()
    rig.run()
    assert [e for e, _ in rig.pushes] == ["health_heal_launched", "health_heal_outcome"]
    assert rig.row()["resume_attempts"][0]["outcome"] == "succeeded"


def test_dry_run_launches_nothing_and_writes_nothing(tmp_path):
    rig = Rig(tmp_path)
    before = rig.row()
    out = rig.run(dry_run=True)
    assert rig.launches == [] and rig.pushes == [] and rig.row() == before
    assert out["would_launch"] == ["drive_failed:demo-drive-a:r1"]


def test_launcher_failure_is_a_recorded_skip(tmp_path):
    rig = Rig(tmp_path)
    rig.launch_ok = False
    rig.run()
    att = rig.row()["resume_attempts"][0]
    assert att["action"] == "skipped" and "launch" in att["reason"] and rig.pushes == []


def test_resolved_rows_are_left_alone(tmp_path):
    rig = Rig(tmp_path, resolved_at="t")
    rig.run()
    assert rig.launches == []


def _running_record(jobs: Path, run_id: str, worker: dict) -> None:
    d = jobs / run_id
    d.mkdir(parents=True)
    log = jobs / f"{run_id}.log"
    log.write_text("INFO gomoveshift.pipeline.drive_orchestrator Phase A start\n")
    (d / "job.json").write_text(json.dumps({
        "schema": "job/1", "run_id": run_id, "name": "demo-audit", "argv": ["bash", "x.sh"],
        "created_at": 1.0, "expect": [str(log)], "result": {"state": "running"},
        "nanny": {"pid": 1, "start_ticks": "1", "boot_id": "b"}, "worker": worker,
    }))  # fmt: skip


def test_stale_running_record_with_dead_worker_is_not_a_running_drive(tmp_path):
    # A record left at "running" by a killed nanny must not block heal forever.
    jobs = tmp_path / "jobs"
    _running_record(jobs, "old", {"pid": 7, "start_ticks": "9", "boot_id": "b"})
    assert (
        heal.default_drive_running(jobs, alive_fn=lambda ident: False, pgrep_fn=lambda: False)
        is False
    )


def test_running_record_with_live_worker_is_a_running_drive(tmp_path):
    jobs = tmp_path / "jobs"
    _running_record(jobs, "live", {"pid": 7, "start_ticks": "9", "boot_id": "b"})
    alive = lambda ident: ident.get("pid") == 7  # noqa: E731
    assert heal.default_drive_running(jobs, alive_fn=alive, pgrep_fn=lambda: False) is True


# ── The code that runs is the code whose SHA is recorded ─────────────
# A recipe names the interpreter and cwd of the run that wrote it -- for an
# evidence run, a pinned worktree. Launching those verbatim would run the old
# worktree's code while the attempt records ``code_sha(code_root)``.


def test_resume_runs_code_root_interpreter_not_the_recipe_writers(tmp_path):
    rig = Rig(tmp_path)
    recipe = json.loads((tmp_path / "stages" / "resume.json").read_text())
    recipe["argv"][0] = "/old/worktree/service/.venv/bin/python"
    recipe["cwd"] = "/old/worktree/service"
    (tmp_path / "stages" / "resume.json").write_text(json.dumps(recipe))
    rig.run()
    argv, cwd = rig.launches[0]
    tail = argv[argv.index("--") + 1 :]
    assert tail[:3] == [str(rig.python), "-m", "gomoveshift"]
    assert not any(a.startswith("/old/worktree") for a in argv) and cwd == str(rig.svc)
    att = rig.row()["resume_attempts"][0]
    assert att["action"] == "launched" and att["mirror_sha"] == SHA
    assert att["interpreter"] == str(rig.python)


def test_cross_fs_override_is_not_carried_into_a_resume(tmp_path):
    rig = Rig(tmp_path, drive_args=["drive", "/in/demo-drive-a", "--allow-cross-fs-source"])
    rig.run()
    argv, _cwd = rig.launches[0]
    assert "--allow-cross-fs-source" not in argv
    assert argv[argv.index("--") + 4 : argv.index("--") + 6] == ["drive", "/in/demo-drive-a"]


def test_refuses_when_code_root_has_no_service_interpreter(tmp_path):
    rig = Rig(tmp_path, venv=False)
    out = rig.run()
    assert rig.launches == [] and out["launched"] == []
    att = rig.row()["resume_attempts"][0]
    assert att["action"] == "refused"
    assert str(rig.python) in att["reason"] and "virtualenv" in att["reason"]


def test_refuses_a_recipe_that_is_not_a_gomoveshift_module_run(tmp_path):
    rig = Rig(tmp_path)
    recipe = json.loads((tmp_path / "stages" / "resume.json").read_text())
    recipe["argv"] = ["/usr/bin/bash", "/old/launcher.sh", "drive", "/in/demo-drive-a"]
    (tmp_path / "stages" / "resume.json").write_text(json.dumps(recipe))
    rig.run()
    assert rig.launches == []
    att = rig.row()["resume_attempts"][0]
    assert att["action"] == "refused" and "-m gomoveshift" in att["reason"]


# ── One named drive, no incident row needed ─────────────────────────
# An evidence run launched outside the job runner never gets a drive_failed row, so the
# hourly heal can't see it. ``--recipe`` registers that one drive and heals only it,
# through the same guards and the same code-root binding.


def _second_recipe(tmp_path: Path, phase="phase_a") -> Path:
    stages = tmp_path / "ev" / "stages" / "demo-drive-b" / "_stages"
    resume.write_recipe(stages, ["drive", "/in/demo-drive-b"], cwd="/w", sha="old", now_iso="t")
    if phase:
        resume.mark_phase(stages, phase)
    return stages / "resume.json"


def test_recipe_target_registers_one_row_and_heals_only_that_drive(tmp_path):
    rig = Rig(tmp_path)
    key = heal.register_recipe(rig.inc, _second_recipe(tmp_path), now_iso="t")
    assert key == "drive_failed:demo-drive-b:recipe"
    out = rig.run(only_key=key)
    assert out["launched"] == [key] and len(rig.launches) == 1
    argv, cwd = rig.launches[0]
    tail = argv[argv.index("--") + 1 :]
    assert tail[3:5] == ["drive", "/in/demo-drive-b"] and "--phase" not in tail
    assert cwd == str(rig.svc)
    other = next(r for r in load_all(rig.inc)[0] if r["drive"] == "demo-drive-a")
    assert other["resume_attempts"] == []


def test_registering_a_recipe_twice_keeps_its_attempt_history(tmp_path):
    rig = Rig(tmp_path)
    path = _second_recipe(tmp_path)
    key = heal.register_recipe(rig.inc, path, now_iso="t")
    rig.run(only_key=key)
    assert heal.register_recipe(rig.inc, path, now_iso="t2") == key
    row = next(r for r in load_all(rig.inc)[0] if r["key"] == key)
    assert [a["action"] for a in row["resume_attempts"]] == ["launched"]


def test_unreadable_recipe_is_not_registered(tmp_path):
    rig = Rig(tmp_path)
    bad = tmp_path / "nope" / "resume.json"
    assert heal.register_recipe(rig.inc, bad, now_iso="t") is None
    assert len(load_all(rig.inc)[0]) == 1


def test_recipe_target_still_obeys_the_publish_slot_guard(tmp_path):
    # --recipe narrows which row heals; it bypasses no guard. A passed or
    # unknown slot is a recorded skip for that row too.
    rig = Rig(tmp_path)
    key = heal.register_recipe(rig.inc, _second_recipe(tmp_path), now_iso="t")
    rig.slot = None
    out = rig.run(only_key=key)
    row = next(r for r in load_all(rig.inc)[0] if r["key"] == key)
    assert rig.launches == [] and out["launched"] == []
    assert row["resume_attempts"][-1]["reason"] == "publish slot unknown"
    rig.slot = datetime(2026, 1, 9, 12, 0, tzinfo=timezone.utc)
    rig.run(only_key=key)
    row = next(r for r in load_all(rig.inc)[0] if r["key"] == key)
    assert rig.launches == [] and "has passed" in row["resume_attempts"][-1]["reason"]


# A dry run is the pre-launch evidence an operator grades before a resume: it must show the
# exact command, its interpreter and cwd, and the mirror SHA it would record -- not just a key.


def test_dry_run_reports_the_exact_launch_it_would_make(tmp_path):
    rig = Rig(
        tmp_path, drive_args=["drive", "/in/demo-drive-a", "--allow-cross-fs-source", "--force"]
    )
    recipe = json.loads((tmp_path / "stages" / "resume.json").read_text())
    recipe["argv"][0] = "/old/worktree/service/.venv/bin/python"
    recipe["cwd"] = "/old/worktree/service"
    (tmp_path / "stages" / "resume.json").write_text(json.dumps(recipe))
    before = rig.row()
    out = rig.run(dry_run=True)
    assert rig.launches == [] and rig.row() == before
    (att,) = out["attempts"]
    assert att["key"] == "drive_failed:demo-drive-a:r1" and att["mirror_sha"] == SHA
    assert att["interpreter"] == str(rig.python) and att["cwd"] == str(rig.svc)
    tail = att["argv"][att["argv"].index("--") + 1 :]
    assert tail[:3] == [str(rig.python), "-m", "gomoveshift"]
    assert "--allow-cross-fs-source" not in att["argv"]
    assert not any(a.startswith("/old/worktree") for a in att["argv"])
    assert tail[-2:] == ["--skip-upload", "--no-youtube-upload"] and "--phase" not in tail
    assert out["would_launch"] == ["drive_failed:demo-drive-a:r1"]


def test_job_runner_defaults_to_the_code_root_script_and_honours_the_env(tmp_path, monkeypatch):
    monkeypatch.delenv(heal.JOB_RUNNER_ENV, raising=False)
    assert heal.job_runner(tmp_path) == tmp_path / "bin" / "job-runner.py"
    custom = tmp_path / "elsewhere" / "run.py"
    monkeypatch.setenv(heal.JOB_RUNNER_ENV, str(custom))
    assert heal.job_runner(tmp_path) == custom
    rig = Rig(tmp_path)
    rig.run()
    argv, _cwd = rig.launches[0]
    assert argv[argv.index("python3") : argv.index("python3") + 2] == ["python3", str(custom)]


def test_resume_keeps_the_recipes_staging_override(tmp_path):
    rig = Rig(tmp_path)
    path = tmp_path / "stages" / "resume.json"
    recipe = json.loads(path.read_text())
    stages = "/mnt/c/Stages/demo-drive-a"
    recipe["env"] = {"GMS_INTERMEDIATE_DIR": stages}
    path.write_text(json.dumps(recipe))
    rig.run()
    argv, _cwd = rig.launches[0]
    assert argv[:3] == ["env", f"GMS_INTERMEDIATE_DIR={stages}", "GMS_LAUNCHED_BY=health-heal"]
    assert argv[3] == "python3" and argv[4].endswith("bin/job-runner.py")


def test_resume_without_a_recorded_env_still_carries_the_heal_launcher_marker(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    argv, _cwd = rig.launches[0]
    assert argv[:3] == ["env", "GMS_LAUNCHED_BY=health-heal", "python3"]


def test_a_recorded_launcher_in_the_recipe_cannot_replace_the_heal_marker(tmp_path):
    rig = Rig(tmp_path)
    path = tmp_path / "stages" / "resume.json"
    recipe = json.loads(path.read_text())
    recipe["env"] = {"GMS_LAUNCHED_BY": "console"}
    path.write_text(json.dumps(recipe))
    rig.run()
    argv, _cwd = rig.launches[0]
    assert "GMS_LAUNCHED_BY=health-heal" in argv and "GMS_LAUNCHED_BY=console" not in argv


def _checkpoint(rig: Rig, **versions) -> None:
    """A completed checkpoint beside the rig's recipe, stamped with ``versions``."""
    from gomoveshift.pipeline.orchestrator_fingerprint import PHASE_CODE_VERSIONS

    stamps = {**PHASE_CODE_VERSIONS, **versions}
    (rig.tmp / "stages" / "pipeline_state.json").write_text(
        json.dumps(
            {
                "phases_completed": ["phase_a", "phase_bc_day_first", "phase_d"],
                "phase_code_versions": stamps,
            }
        ),
        encoding="utf-8",
    )


def test_a_version_stale_checkpoint_resumes_every_phase_not_one(tmp_path):
    # `--phase D` on a checkpoint whose phase_bc stamp is stale stops at the prerequisite
    # check (the orchestrator drops the stale phase_bc), so a heal must run the plain resume:
    # the stale phase and the phases built from it re-run, the current ones are reused.
    from gomoveshift.pipeline.orchestrator_fingerprint import PHASE_CODE_VERSIONS

    rig = Rig(tmp_path)
    _checkpoint(rig, phase_bc_day_first=PHASE_CODE_VERSIONS["phase_bc_day_first"] - 1)
    out = rig.run()
    assert out["launched"] == ["drive_failed:demo-drive-a:r1"]
    argv, _cwd = rig.launches[0]
    tail = argv[argv.index("--") + 1 :]
    assert "--phase" not in tail
    assert tail[-2:] == ["--skip-upload", "--no-youtube-upload"]
    att = rig.row()["resume_attempts"][0]
    assert att["stale_phases"] == ["phase_bc_day_first", "phase_d"]
    assert "phase_bc_day_first" in att["reason"]


def test_a_current_checkpoint_resumes_plain_with_no_stale_phases(tmp_path):
    rig = Rig(tmp_path)
    _checkpoint(rig)
    rig.run()
    argv, _cwd = rig.launches[0]
    assert argv[-2:] == ["--skip-upload", "--no-youtube-upload"] and "--phase" not in argv
    assert "stale_phases" not in rig.row()["resume_attempts"][0]


def test_a_stale_checkpoint_resumes_even_when_the_phase_is_unmappable(tmp_path):
    from gomoveshift.pipeline.orchestrator_fingerprint import PHASE_CODE_VERSIONS

    rig = Rig(tmp_path, phase=None)
    _checkpoint(rig, phase_a=PHASE_CODE_VERSIONS["phase_a"] - 1)
    rig.run()
    argv, _cwd = rig.launches[0]
    assert "--phase" not in argv[argv.index("--") + 1 :]


def test_an_unreadable_checkpoint_is_not_stale(tmp_path):
    rig = Rig(tmp_path)
    (rig.tmp / "stages" / "pipeline_state.json").write_text("{not json", encoding="utf-8")
    rig.run()
    argv, _cwd = rig.launches[0]
    assert "--phase" not in argv[argv.index("--") + 1 :]


def test_re_registering_a_resolved_recipe_reopens_it_with_its_history(tmp_path):
    # An operator's --recipe is never a silent no-op: a row `health check` resolved
    # (e.g. after a phase-A-only heal "succeeded") opens again and heals.
    rig = Rig(tmp_path)
    path = _second_recipe(tmp_path)
    key = heal.register_recipe(rig.inc, path, now_iso="t")
    rig.run(only_key=key)
    row = next(r for r in load_all(rig.inc)[0] if r["key"] == key)
    row["resolved_at"] = "t1"
    write_row(rig.inc, row)
    assert heal.register_recipe(rig.inc, path, now_iso="t2") == key
    row = next(r for r in load_all(rig.inc)[0] if r["key"] == key)
    assert not row.get("resolved_at") and row["reopened_at"] == "t2"
    assert [a["action"] for a in row["resume_attempts"]] == ["launched"]
    assert rig.run(sha="newer", only_key=key)["launched"] == [key]


# ── The process fallback matches a drive, never a health call that names one ──
# 2026-11-07: ``health backlog --supersede ... --reason "queue next Ireland drive"``
# matched the old ``gomoveshift.* drive `` pattern on its own command line, so
# every such call answered "a drive is running" with no drive alive.


def _pgrep_pids(pattern: str) -> set[int]:
    done = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, check=False)
    return {int(p) for p in done.stdout.split()}


@pytest.mark.skipif(shutil.which("pgrep") is None, reason="needs pgrep")
@pytest.mark.parametrize(
    "argv, is_drive",
    [
        (["-m", "gomoveshift", "drive", "/mnt/d/_gms-sources/x", "--region", "usa"], True),
        ([".venv/bin/gomoveshift", "--log-json", "drive", "/mnt/d/x"], True),
        (["gomoveshift", "health", "backlog", "--reason", "next Ireland drive", "-n"], False),
        (["gomoveshift", "health", "publish", "p", "--reason", "re-run the drive now"], False),
    ],
)
def test_pgrep_pattern_matches_a_drive_process_only(argv, is_drive):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", *argv])
    try:
        for _ in range(50):  # the new pid's cmdline is readable almost at once
            if Path(f"/proc/{proc.pid}/cmdline").read_bytes().count(b"\0") > 3:
                break
            time.sleep(0.02)
        assert (proc.pid in _pgrep_pids(heal.DRIVE_PROCESS_PATTERN)) is is_drive
    finally:
        proc.kill()
        proc.wait()
