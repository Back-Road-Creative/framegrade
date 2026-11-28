"""``health heal``: audit-only auto-resume of a failed drive, behind five guards.

Every side effect (launcher, notify, git, process probe, slot, clock) is injected; nothing here
launches a real job or pushes."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.health import heal, resume, retry_policy
from gomoveshift.health.store import load_all, new_incident, write_row

NOW = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)
FUTURE = datetime(2026, 1, 16, 12, 0, tzinfo=timezone.utc)
SHA = "abcdef1234567890"  # pragma: allowlist secret
NEW_SHA = "1234567fedcba"  # pragma: allowlist secret


class Rig:
    def __init__(self, tmp_path: Path, *, drive_args=None, venv=True, legacy=False, **row_over):
        self.legacy = legacy  # filed before fingerprints: the row carries no failure_fingerprint
        self.filed = False
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
        self.conflict = None  # what ``conflict_fn`` says: a live drive on this recipe's own dirs
        self.changed: list = []  # git is pathspec-filtered: only generator files can appear
        self.slot = FUTURE
        self.launch_ok = True
        self.alive = True  # what the injected ``proc_alive`` says of any nanny identity
        self.installed = None  # the venv's gomoveshift-video version; None = unknown
        self.pinned = None  # the checkout's pinned version; None = unknown
        self.siblings: dict = {}  # seo_manager / redactcam versions; {} = unknown
        self.manifest = None  # sha of the drive's manifest row; None = unknown
        self.manifest_row = None  # the drive's ``drive_backlog.yaml`` row; None = not found
        stages = tmp_path / "stages"
        for src in ("demo-drive-a", "demo-drive-b"):  # the recipes' source dirs, under code_root
            (tmp_path / "in" / src).mkdir(parents=True, exist_ok=True)
        resume.write_recipe(
            stages,
            drive_args or ["drive", "in/demo-drive-a", "--publish", "fri"],
            cwd="/work",
            sha="old",
            now_iso="t",
        )
        row = new_incident(
            "drive_failed", "drive_failed:demo-drive-a:r1", "demo-drive-a.json", "t",
            drive="demo-drive-a", phase="phase_d", gate=None, error="OSError: No space left on device",
            log_path=None,
            job_name="gms-fri-demo", job_run_id="r1", failed_at=None, staging_sha="old",
            resume_recipe=str(stages / "resume.json"), resume_attempts=[],
        )  # fmt: skip
        row.update(row_over)
        write_row(self.inc, row)

    def file_now(self):
        """What ``health check`` does when it files the row: stamp the fingerprint it failed under."""
        row = self.row()
        row["failure_fingerprint"] = retry_policy.failure_stamp(
            row, python=str(self.python), config=None,
            installed_version_fn=lambda p: self.installed, sibling_versions_fn=lambda p: self.siblings,
            manifest_sha_fn=lambda r: self.manifest, config_sha_fn=lambda c: "cfg",
        )  # fmt: skip
        write_row(self.inc, row)
        self.filed = True

    def run(self, *, sha=SHA, dry_run=False, now=NOW, only_key=None):
        if not (self.legacy or self.filed):
            self.file_now()
        return heal.run_heal(
            incidents_dir=self.inc, jobs_dir=self.jobs, code_root=self.tmp, mirror_sha=sha,
            now=now, dry_run=dry_run, only_key=only_key,
            launcher=lambda argv, cwd: (self.launches.append((argv, cwd)) or (self.launch_ok, "")),
            notify_fn=lambda e, i, s: self.pushes.append((e, s)) or True,
            git_changed=lambda a, b, paths: self.changed,
            conflict_fn=lambda jobs, recipe_path, recipe: self.conflict,
            slot_fn=lambda row, recipe: self.slot,
            installed_version_fn=lambda python: self.installed,
            pinned_version_fn=lambda code_root: self.pinned,
            sibling_versions_fn=lambda python: self.siblings,
            manifest_sha_fn=lambda row: self.manifest,
            manifest_row_fn=lambda row: self.manifest_row,
            config_sha_fn=lambda config: "cfg",
            proc_alive=lambda ident: self.alive,
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
    assert "--fresh" not in tail and tail[3:6] == ["drive", "in/demo-drive-a", "--publish"]
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
    assert [a["action"] for a in att] == ["refused"]  # an unchanged refusal is not re-recorded
    assert resume.HOLD_NAME in att[0]["reason"]


def test_no_auto_resume_marker_shows_in_the_dry_run(tmp_path):
    rig = Rig(tmp_path)
    (tmp_path / "stages" / resume.HOLD_NAME).write_text("")
    out = rig.run(dry_run=True)
    assert out["would_launch"] == [] and out["attempts"][0]["action"] == "refused"


def test_guard_a_waits_out_its_backoff_on_one_mirror_sha(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    rig.run()
    rig.run()
    assert len(rig.launches) == 1
    # the backoff skip is recorded once (the owner's rule: waits are visible, never silent)
    assert len(rig.row()["resume_attempts"]) == 2
    rig.run(sha="1234567fedcba")  # pragma: allowlist secret
    assert len(rig.launches) == 1  # an unrelated mirror commit is not a reason to relaunch
    rig.installed = "0.3.999"
    rig.run(sha="1234567fedcba", now=_at(heal.HEAL_RETRY_BASE_S + 60))  # pragma: allowlist secret
    assert len(rig.launches) == 2


def test_guard_a_is_keyed_on_the_installed_version_not_the_sha_alone(tmp_path):
    rig = Rig(tmp_path)
    rig.installed = "0.3.101"
    rig.run()
    rig.run()
    assert len(rig.launches) == 1  # same SHA, same installed version: once
    assert rig.row()["resume_attempts"][0]["installed_version"] == "0.3.101"
    rig.installed = "0.3.102"  # the deferred install landed; the SHA did not move
    rig.run()
    assert len(rig.launches) == 2
    assert rig.row()["resume_attempts"][-1]["installed_version"] == "0.3.102"
    rig.run()
    assert len(rig.launches) == 2


def test_an_unknown_version_turning_known_is_one_changed_fingerprint(tmp_path):
    rig = Rig(tmp_path, error="boom")  # unknown class: one launch per fingerprint
    rig.run()  # installed unknown
    rig.run()
    assert len(rig.launches) == 1
    rig.installed = "0.3.102"
    rig.run()
    assert len(rig.launches) == 2
    rig.run()
    assert len(rig.launches) == 2


def test_install_pending_refuses_while_the_pin_differs_from_the_venv(tmp_path):
    rig = Rig(tmp_path)
    rig.pinned, rig.installed = "0.3.102", "0.3.101"
    out = rig.run()
    assert rig.launches == [] and out["launched"] == []
    att = rig.row()["resume_attempts"][0]
    assert att["action"] == "skipped"
    assert att["reason"] == "install pending: checkout pins 0.3.102, venv has 0.3.101"
    rig.run()
    assert len(rig.row()["resume_attempts"]) == 1  # an unchanged skip is not re-recorded
    rig.installed = "0.3.102"  # install landed
    rig.run()
    assert len(rig.launches) == 1


def test_install_pending_shows_in_the_dry_run(tmp_path):
    rig = Rig(tmp_path)
    rig.pinned, rig.installed = "0.3.102", "0.3.101"
    out = rig.run(dry_run=True)
    assert out["would_launch"] == []
    assert out["attempts"][0]["reason"].startswith("install pending:")


def test_unknown_pin_or_installed_version_does_not_hold_a_resume(tmp_path):
    rig = Rig(tmp_path)
    rig.pinned, rig.installed = None, "0.3.101"
    rig.run()
    assert len(rig.launches) == 1


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
    assert len(rig.launches) == 1  # an unrelated mirror commit: nothing
    rig.installed = "0.3.999"
    rig.run()
    assert len(rig.launches) == 2  # new code: exactly one more resume of that recipe


def test_a_heal_of_a_heal_does_not_repeat_the_audit_only_flags(tmp_path):
    rig = Rig(
        tmp_path, drive_args=["drive", "in/demo-drive-a", "--skip-upload", "--no-youtube-upload"]
    )
    rig.run()
    tail = rig.launches[0][0][rig.launches[0][0].index("--") + 1 :]
    assert tail.count("--skip-upload") == 1 and tail.count("--no-youtube-upload") == 1


def test_guard_b_skips_while_a_drive_on_the_same_dirs_is_running(tmp_path):
    rig = Rig(tmp_path)
    rig.conflict = "a drive on the same stages dir is running (pid 7)"
    rig.run()
    att = rig.row()["resume_attempts"][0]
    assert rig.launches == [] and att["action"] == "skipped" and att["reason"] == rig.conflict
    rig.conflict = None
    rig.run()
    assert len(rig.launches) == 1


def test_guard_c_a_passed_slot_still_launches_the_audit_only_resume(tmp_path):
    # The drive CLI keeps only the weekday check for an audit-only run (--skip-upload with
    # --no-youtube-upload; handlers/drive.py `_launch_anchor_refusal`), so a passed slot never
    # refuses a heal resume; `health backlog`/`health publish` claim a fresh slot afterwards.
    rig = Rig(tmp_path)
    rig.slot = datetime(2026, 1, 9, 12, 0, tzinfo=timezone.utc)
    rig.run()
    att = rig.row()["resume_attempts"][0]
    assert len(rig.launches) == 1 and att["action"] == "launched"
    assert att["slot"] == rig.slot.isoformat()


def test_guard_c_an_unknown_slot_still_launches_and_records_no_slot(tmp_path):
    rig = Rig(tmp_path)
    rig.slot = None
    rig.run()
    att = rig.row()["resume_attempts"][0]
    assert len(rig.launches) == 1 and att["action"] == "launched" and att["slot"] is None


def test_guard_d_metadata_gate_needs_a_generator_change(tmp_path):
    rig = Rig(tmp_path, gate="Metadata quality")
    rig.run()
    att = rig.row()["resume_attempts"][0]
    assert rig.launches == [] and att["action"] == "skipped" and "generator" in att["reason"]
    rig.changed = [heal.GENERATOR_PATHS[0] + "x.py"]
    rig.run(sha="1234567fedcba")  # pragma: allowlist secret
    assert len(rig.launches) == 1


def test_guard_d_unchanged_generators_retry_once_per_max_wait_never_never(tmp_path):
    rig = Rig(tmp_path, gate="Metadata quality")
    rig.run()
    assert rig.launches == []
    rig.run(now=NOW + timedelta(seconds=heal.HEAL_RETRY_MAX_S - 60))
    assert rig.launches == []
    assert len(rig.row()["resume_attempts"]) == 1  # the repeat skip is not re-recorded
    rig.run(now=NOW + timedelta(seconds=heal.HEAL_RETRY_MAX_S + 60))
    assert len(rig.launches) == 1
    # launched: the next try waits the cap again, even on a new SHA with the same generators
    rig.run(sha=NEW_SHA, now=NOW + timedelta(seconds=heal.HEAL_RETRY_MAX_S + 3600))
    assert len(rig.launches) == 1
    rig.run(sha=NEW_SHA, now=NOW + timedelta(seconds=2 * heal.HEAL_RETRY_MAX_S + 300))
    assert len(rig.launches) == 2


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
        conflict_fn=lambda j, rp, recipe: None, slot_fn=lambda r, c: FUTURE,
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
    rig.file_now()
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


def _running_record(jobs: Path, run_id: str, worker: dict, out: str = "/out/demo") -> None:
    d = jobs / run_id
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({
        "schema": "job/1", "run_id": run_id, "name": "demo-audit", "created_at": 1.0,
        "argv": ["py", "-m", "gomoveshift", "drive", "/src/demo", "--output-dir", out],
        "result": {"state": "running"},
        "nanny": {"pid": 1, "start_ticks": "1", "boot_id": "b"}, "worker": worker,
    }))  # fmt: skip


_RECIPE = {"argv": ["py", "-m", "gomoveshift", "drive", "/src/demo", "--output-dir", "/out/demo"], "cwd": "/"}  # fmt: skip


def _conflict(tmp_path, alive=lambda ident: True):
    return heal.drive_conflict(
        tmp_path / "jobs", tmp_path / "st" / "resume.json", _RECIPE, alive_fn=alive,
        holder_fn=lambda lock: None,
    )  # fmt: skip


def test_stale_running_record_with_dead_worker_is_not_a_conflict(tmp_path):
    # A record left at "running" by a killed nanny must not block heal forever.
    _running_record(tmp_path / "jobs", "old", {"pid": 7, "start_ticks": "9", "boot_id": "b"})
    assert _conflict(tmp_path, alive=lambda ident: False) is None


def test_a_live_job_on_the_same_output_dir_is_a_conflict(tmp_path):
    _running_record(tmp_path / "jobs", "live", {"pid": 7, "start_ticks": "9", "boot_id": "b"})
    assert "same output dir" in _conflict(tmp_path)


def test_a_live_job_of_another_drive_is_never_a_conflict(tmp_path):
    _running_record(tmp_path / "jobs", "other", {"pid": 7}, out="/out/another-drive")
    assert _conflict(tmp_path) is None


def test_a_live_run_lock_on_the_stages_or_project_dir_is_a_conflict(tmp_path):
    for held in (tmp_path / "st" / "pipeline_state.lock", Path("/out/demo/pipeline_project.lock")):  # fmt: skip
        why = heal.drive_conflict(
            tmp_path / "jobs", tmp_path / "st" / "resume.json", _RECIPE,
            holder_fn=lambda lock, held=held: {"pid": 7} if lock == held else None,
        )  # fmt: skip
        assert why and "pid 7" in why


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
    rig = Rig(tmp_path, drive_args=["drive", "in/demo-drive-a", "--allow-cross-fs-source"])
    rig.run()
    argv, _cwd = rig.launches[0]
    assert "--allow-cross-fs-source" not in argv
    assert argv[argv.index("--") + 4 : argv.index("--") + 6] == ["drive", "in/demo-drive-a"]


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
    recipe["argv"] = ["/usr/bin/bash", "/old/launcher.sh", "drive", "in/demo-drive-a"]
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
    resume.write_recipe(stages, ["drive", "in/demo-drive-b"], cwd="/w", sha="old", now_iso="t")
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
    assert tail[3:5] == ["drive", "in/demo-drive-b"] and "--phase" not in tail
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


def test_recipe_target_still_obeys_the_backoff_and_holds(tmp_path):
    # --recipe narrows which row heals; it bypasses no guard. A held recipe still refuses and a
    # launched one still waits out its backoff (a passed or unknown slot is no longer a guard).
    rig = Rig(tmp_path)
    second = _second_recipe(tmp_path)
    key = heal.register_recipe(rig.inc, second, now_iso="t")
    rig.slot = None
    out = rig.run(only_key=key)
    assert out["launched"] == [key]
    rig.run(only_key=key)
    assert len(rig.launches) == 1
    row = next(r for r in load_all(rig.inc)[0] if r["key"] == key)
    assert row["resume_attempts"][-1]["reason"].startswith("unknown failure, fingerprint")
    (second.parent / resume.HOLD_NAME).write_text("stop\n")
    rig.run(only_key=key, now=NOW + timedelta(days=3))
    row = next(r for r in load_all(rig.inc)[0] if r["key"] == key)
    assert len(rig.launches) == 1 and row["resume_attempts"][-1]["action"] == "refused"


# A dry run is the pre-launch evidence an operator grades before a resume: it must show the
# exact command, its interpreter and cwd, and the mirror SHA it would record -- not just a key.


def test_dry_run_reports_the_exact_launch_it_would_make(tmp_path):
    rig = Rig(
        tmp_path, drive_args=["drive", "in/demo-drive-a", "--allow-cross-fs-source", "--force"]
    )
    recipe = json.loads((tmp_path / "stages" / "resume.json").read_text())
    recipe["argv"][0] = "/old/worktree/service/.venv/bin/python"
    recipe["cwd"] = "/old/worktree/service"
    (tmp_path / "stages" / "resume.json").write_text(json.dumps(recipe))
    rig.file_now()
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
    rig.installed = "0.3.999"
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


PREFLIGHT_LOG = (
    "[INFO] gomoveshift Pre-flight: validating environment...\n"
    "[ERROR] gomoveshift.pipeline.drive_orchestrator Pre-flight check failed: RENDER BLOCKED: "
    "1 blocker found before any encode:\n"
    "  1. [staging disk] Insufficient cache space at /stage: 249.8 GB free, need ~351.4 GB\n"
    "DRIVE FAILED: Pre-flight failed: RENDER BLOCKED\n"
)
ENCODE_LOG = "[INFO] PHASE DONE phase_a in 12s\n" + PREFLIGHT_LOG


def _tick(n: int) -> datetime:
    """The clock of the n-th heal run (the n-th attempt is made at ``_tick(n - 1)``)."""
    return NOW + timedelta(minutes=45 * n)


def _finish_heal_job(rig: Rig, n: int, log: str, *, state="failed") -> None:
    """The heal job the last launch started, finished with ``log`` as its output.log."""
    name = heal.job_name("demo-drive-a", SHA)
    d = rig.jobs / f"h{n}"
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({
        "schema": "job/1", "run_id": f"h{n}", "name": name, "argv": ["x"],
        "created_at": _tick(n).timestamp() - 300, "result": {"state": state, "exit_code": 1},
    }))  # fmt: skip
    (d / "output.log").write_text(log)


def test_a_preflight_failed_heal_relaunches_on_the_next_run_at_the_same_sha(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    _finish_heal_job(rig, 1, PREFLIGHT_LOG)
    out = rig.run(now=_tick(1))
    atts = rig.row()["resume_attempts"]
    assert atts[0]["outcome"] == "preflight_failed"
    assert atts[0]["outcome_detail"].startswith("1. [staging disk] Insufficient cache space")
    assert len(rig.launches) == 2 and out["launched"] == ["drive_failed:demo-drive-a:r1"]
    assert atts[1]["action"] == "launched" and "outcome" not in atts[1]
    rig.run(now=_tick(2))  # the relaunch is not judged by the previous run's finished record
    assert len(rig.launches) == 2


def test_an_encode_stage_failure_does_not_relaunch_at_the_same_sha(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    _finish_heal_job(rig, 1, ENCODE_LOG)
    rig.run()
    assert len(rig.launches) == 1
    assert rig.row()["resume_attempts"][0]["outcome"] == "failed"


def test_a_failure_with_no_log_is_a_plain_failure_not_a_preflight_one(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    _finish_heal_job(rig, 1, "")
    (rig.jobs / "h1" / "output.log").unlink()
    rig.run()
    assert len(rig.launches) == 1
    assert rig.row()["resume_attempts"][0]["outcome"] == "failed"


def test_preflight_retries_are_capped_then_wait_the_max_backoff_never_forever(tmp_path):
    rig = Rig(tmp_path)
    for n in range(1, heal.HEAL_PREFLIGHT_RETRIES + 1):
        rig.run(now=_tick(n - 1))
        assert len(rig.launches) == n
        _finish_heal_job(rig, n, PREFLIGHT_LOG)
    last_at = _tick(heal.HEAL_PREFLIGHT_RETRIES - 1)
    rig.run(now=_tick(heal.HEAL_PREFLIGHT_RETRIES))
    rig.run(now=_tick(heal.HEAL_PREFLIGHT_RETRIES + 1))
    assert len(rig.launches) == heal.HEAL_PREFLIGHT_RETRIES
    last = rig.row()["resume_attempts"][-1]
    retry_at = last_at + timedelta(seconds=heal.HEAL_RETRY_MAX_S)
    assert last["action"] == "skipped"
    assert last["reason"] == (
        f"pre-flight failed 8x on this code; retry at {heal.utc_iso(retry_at.timestamp())}"
    )
    rig.run(now=retry_at - timedelta(minutes=5))
    assert len(rig.launches) == heal.HEAL_PREFLIGHT_RETRIES
    rig.run(now=retry_at + timedelta(minutes=5))
    assert len(rig.launches) == heal.HEAL_PREFLIGHT_RETRIES + 1
    rig.installed = "0.3.999"
    rig.run(now=retry_at + timedelta(minutes=6))
    assert len(rig.launches) == heal.HEAL_PREFLIGHT_RETRIES + 2  # new code is immediate


def test_a_preflight_failed_retry_still_obeys_the_other_guards(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    _finish_heal_job(rig, 1, PREFLIGHT_LOG)
    rig.conflict = "a drive on the same stages dir is running (pid 7)"
    rig.run(now=_tick(1))
    assert len(rig.launches) == 1
    assert rig.row()["resume_attempts"][-1]["reason"] == rig.conflict


def test_the_outcome_notice_says_failed_pre_flight_with_the_blocker(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    _finish_heal_job(rig, 1, PREFLIGHT_LOG)
    rig.conflict = "a drive on the same stages dir is running (pid 7)"
    rig.run(now=_tick(1))
    text = [s for e, s in rig.pushes if e == "health_heal_outcome"][0]
    assert "failed pre-flight: 1. [staging disk] Insufficient cache space" in text


def test_the_preflight_detail_is_truncated_to_300_chars(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    _finish_heal_job(rig, 1, PREFLIGHT_LOG.replace("need ~351.4 GB", "x" * 500))
    rig.conflict = "a drive on the same stages dir is running (pid 7)"
    rig.run(now=_tick(1))
    assert len(rig.row()["resume_attempts"][0]["outcome_detail"]) == 300


def _at(seconds: float) -> datetime:
    return NOW + timedelta(seconds=seconds)


def test_same_code_within_the_backoff_does_not_launch_and_names_the_next_attempt(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    rig.run(now=_at(heal.HEAL_RETRY_BASE_S - 60))
    rig.run(now=_at(heal.HEAL_RETRY_BASE_S - 30))
    assert len(rig.launches) == 1
    atts = rig.row()["resume_attempts"]
    assert [a["action"] for a in atts] == ["launched", "skipped"]  # the repeat skip is deduped
    assert atts[1]["reason"] == (
        f"retry backoff: next attempt at {heal.utc_iso(_at(heal.HEAL_RETRY_BASE_S).timestamp())}"
    )


def test_same_code_past_the_backoff_launches_again_on_the_same_job_name(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    rig.run(now=_at(heal.HEAL_RETRY_BASE_S + 60))
    assert len(rig.launches) == 2
    names = [a[0][a[0].index("--name") + 1] for a in rig.launches]
    assert names[0] == names[1] == heal.job_name("demo-drive-a", SHA)


def test_the_backoff_doubles_from_one_hour_and_caps_at_a_day(tmp_path, monkeypatch):
    monkeypatch.setitem(retry_policy.BUDGET, "transient", 99)  # the curve, not the budget
    assert (heal.HEAL_RETRY_BASE_S, heal.HEAL_RETRY_MAX_S) == (3600, 86400)
    rig = Rig(tmp_path)
    clock = 0.0
    waits = []
    for _ in range(8):
        before = len(rig.launches)
        rig.run(now=_at(clock))
        assert len(rig.launches) == before + 1
        # one second short of the wait: nothing; the next loop, at the wait itself: a launch
        n = before + 1
        want = min(3600 * 2 ** (n - 1), 86400)
        waits.append(want)
        rig.run(now=_at(clock + want - 1))
        assert len(rig.launches) == n
        clock += want
    assert waits == [3600, 7200, 14400, 28800, 57600, 86400, 86400, 86400]


def test_the_backoff_is_counted_across_every_row_sharing_the_recipe(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    _failed_heal_row(rig, 1)
    rig.run(now=_at(3600 + 60))  # attempt 2 (1h after attempt 1)
    assert len(rig.launches) == 2
    _failed_heal_row(rig, 2)
    # n is now 2 for the recipe: the next wait is 2h from the newest attempt, whichever row ran it
    rig.run(now=_at(3600 + 60 + 7200 - 60))
    assert len(rig.launches) == 2
    rig.run(now=_at(3600 + 60 + 7200 + 60))
    assert len(rig.launches) == 3


def test_a_changed_fingerprint_retries_immediately_inside_the_backoff(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    rig.run(sha="1234567fedcba", now=_at(60))  # pragma: allowlist secret
    assert len(rig.launches) == 1  # a new mirror SHA alone no longer resets the backoff
    rig.manifest = "row-v2"
    rig.run(sha="1234567fedcba", now=_at(120))  # pragma: allowlist secret
    assert len(rig.launches) == 2  # a changed manifest row does


def test_a_reused_job_name_maps_each_outcome_to_its_own_run_by_time(tmp_path):
    rig = Rig(tmp_path)
    name = heal.job_name("demo-drive-a", SHA)

    def finish(run_id, created, state, code):
        d = rig.jobs / run_id
        d.mkdir(parents=True)
        (d / "job.json").write_text(json.dumps({
            "schema": "job/1", "run_id": run_id, "name": name, "argv": ["x"],
            "created_at": created.timestamp(), "result": {"state": state, "exit_code": code},
        }))  # fmt: skip
        (d / "output.log").write_text(ENCODE_LOG)

    rig.run()
    finish("a", NOW, "failed", 1)
    second = _at(heal.HEAL_RETRY_BASE_S + 60)
    rig.run(now=second)  # records attempt 1 as failed, launches attempt 2
    finish("b", second, "succeeded", 0)
    rig.run(now=_at(heal.HEAL_RETRY_BASE_S + 120))
    atts = rig.row()["resume_attempts"]
    assert [a["outcome"] for a in atts if a["action"] == "launched"] == ["failed", "succeeded"]


# ── Retry by failure class and fingerprint, never by mirror SHA ──


def _fail(rig: Rig, n: int, **over) -> None:
    """A heal job failed: `health check` files it as a row of its own under the same recipe."""
    _failed_heal_row(rig, n)
    rows = {r["key"]: r for r in load_all(rig.inc)[0]}
    row = rows[f"drive_failed:gms-heal-demo-drive-a-abcdef1:h{n}"]
    row.update(over)
    write_row(rig.inc, row)


def test_a_deterministic_failure_is_not_relaunched_by_an_unrelated_mirror_commit(tmp_path):
    rig = Rig(tmp_path, error="LF 1: final-text review refused (place_outside_text)")
    for i, sha in enumerate([SHA, NEW_SHA, "99999991", "99999992"]):
        out = rig.run(sha=sha, now=_at(i * 90000))
        assert out["launched"] == []
    assert rig.launches == []
    att = rig.row()["resume_attempts"][0]
    assert att["action"] == "refused" and att["failure_class"] == "deterministic"
    assert att["reason"].startswith(f"deterministic failure, fingerprint {att['fp_hash'][:12]}")
    assert len(rig.row()["resume_attempts"]) == 1  # one refusal line, not one per mirror commit


def test_a_changed_version_or_manifest_row_relaunches_a_deterministic_failure_once(tmp_path):
    rig = Rig(tmp_path, error="BLUR GATE: blur_enabled but the cached day master has no marker")
    rig.run()
    assert rig.launches == []
    rig.manifest = "row-v2"  # the manifest row was fixed
    rig.run(now=_at(100))
    rig.run(now=_at(100 + 90000))
    assert len(rig.launches) == 1
    _fail(rig, 1, error="BLUR GATE: again", phase="phase_d")
    rig.run(now=_at(200 + 90000))
    assert len(rig.launches) == 1  # failed the same way under the same fingerprint
    rig.installed = "0.3.200"  # a release
    rig.run(now=_at(300 + 90000))
    assert len(rig.launches) == 2


def test_a_transient_failure_retries_at_most_three_times_with_the_backoff(tmp_path):
    rig = Rig(tmp_path)  # "No space left on device"
    clock = 0.0
    for n in (1, 2, 3):
        rig.run(now=_at(clock))
        assert len(rig.launches) == n
        clock += 3600 * 2 ** (n - 1) + 60
    rig.run(now=_at(clock))
    rig.run(now=_at(clock + 7 * 86400))
    assert len(rig.launches) == 3
    last = rig.row()["resume_attempts"][-1]
    assert last["action"] == "refused" and "3 of 3 attempts used" in last["reason"]
    rig.siblings = {"redactcam": "9.9.9"}
    rig.run(now=_at(clock + 7 * 86400 + 60))
    assert len(rig.launches) == 4  # a changed dependency version


def test_an_unknown_failure_is_retried_once_per_fingerprint(tmp_path):
    rig = Rig(tmp_path, error="something nobody classified")
    rig.run()
    rig.run(now=_at(90000))
    rig.run(sha=NEW_SHA, now=_at(180000))
    assert len(rig.launches) == 1
    assert rig.row()["resume_attempts"][0]["failure_class"] == "unknown"
    rig.siblings = {"seo_manager": "0.9.9"}
    rig.run(now=_at(270000))
    assert len(rig.launches) == 2


def test_a_publish_only_failure_is_refused_with_its_class_and_reason(tmp_path):
    rig = Rig(tmp_path, error="YouTubeQuotaError: quotaExceeded")
    out = rig.run()
    att = rig.row()["resume_attempts"][0]
    assert rig.launches == [] and out["launched"] == []
    assert att["action"] == "refused" and att["failure_class"] == "publish_only"
    assert "heal never publishes" in att["reason"] and att["fp_hash"][:12] in att["reason"]


def test_each_attempt_records_the_fingerprint_and_the_row_keeps_its_first_failure(tmp_path):
    rig = Rig(tmp_path)
    rig.installed, rig.siblings, rig.manifest = "0.3.113", {"redactcam": "0.2.0"}, "m1"
    rig.run()
    att, row = rig.row()["resume_attempts"][0], rig.row()
    assert att["fingerprint"] == {
        "versions": {"gomoveshift": "0.3.113", "redactcam": "0.2.0"},
        "manifest_sha": "m1", "config_sha": "cfg", "phase": "phase_d",
    }  # fmt: skip
    assert att["fp_hash"] == row["failure_fingerprint"]["hash"]
    assert row["failure_fingerprint"]["class"] == "transient"


def test_an_incident_from_before_fingerprints_keeps_working(tmp_path):
    rig = Rig(tmp_path, error="boom")  # no failure_fingerprint, an old attempt without fp_hash
    row = rig.row()
    row["resume_attempts"] = [
        {"at": "t", "mirror_sha": SHA, "installed_version": None, "action": "launched",
         "reason": "old", "job_name": "x", "slot": None},
    ]  # fmt: skip
    write_row(rig.inc, row)
    out = rig.run(now=_at(100))
    assert out["launched"] == [row["key"]]  # unknown: its first fingerprinted attempt
    rig.run(now=_at(200))
    assert len(rig.launches) == 1


def test_the_classifier_table(tmp_path):
    cases = {
        "BLUR GATE: insert clip x carries no .blurred marker": "deterministic",
        "FinalTextReviewError: LF 2: final-text review refused (x)": "deterministic",
        "Day 2026-01-01: kept footage 10s is under full_drive_part_min_sec=600s": "deterministic",
        "BatchDurationIntegrityError: sum disagrees": "deterministic",
        "OSError: [Errno 28] No space left on device": "transient",
        "ffmpeg: Cannot allocate memory": "transient",
        "Windows drive /mnt/d is not accessible after 3 probe attempt(s)": "transient",
        "OSError: [Errno 5] Input/output error": "transient",
        "PlaceLookupUnavailableError: every endpoint down": "transient",
        "YouTubeQuotaError: quotaExceeded": "publish_only",
        "DriveUploadError: EOF": "publish_only",
        "ValueError: nobody knows": "unknown",
        "": "unknown",
    }
    for text, want in cases.items():
        assert retry_policy.classify_failure(text) == want, text
    assert retry_policy.classify_failure(None, "phase_d", "Metadata quality") == "unknown"


def test_an_interrupted_repair_swap_is_transient_and_relaunched_but_a_plain_blur_gate_is_not(
    tmp_path,
):
    from gomoveshift.pipeline import blur_gate

    sidecar = tmp_path / "day_x_blur_regions.json"
    blur_gate.repair_swap_pending_path(sidecar).write_text("{}")
    swap = blur_gate.repair_swap_error(sidecar)
    assert retry_policy.classify_failure(swap) == "transient"
    assert retry_policy.classify_failure("BLUR GATE (cabin): 1 of 9 EXPOSED") == "deterministic"
    rig = Rig(tmp_path / "swap", error=swap)
    out = rig.run(now=_at(100))
    assert out["launched"] == [rig.row()["key"]] and len(rig.launches) == 1
    plain = Rig(tmp_path / "plain", error="BLUR GATE (cabin): 1 of 9 EXPOSED")
    plain.run(now=_at(100))
    plain.run(now=_at(100 + 90000))
    # Rig.run stamps the row on its first run, so a deterministic failure is refused outright
    assert plain.launches == []
    assert plain.row()["resume_attempts"][-1]["action"] == "refused"


def test_a_privacy_detection_that_failed_on_a_host_cause_is_transient_a_leak_is_not():
    head = "BLUR GATE: privacy detection failed for day 2026-02-24 master ("
    tail = ") — refusing to encode an unblurred master. Fix the detector/model and re-run"
    for cause in ("CUDA out of memory", "[Errno 5] Input/output error", "ENOSPC", "Killed"):
        assert retry_policy.classify_failure(head + cause + tail) == "transient", cause
    assert retry_policy.classify_failure(head + "ValueError: bad model" + tail) == "deterministic"
    for text in (
        "BLUR GATE (cabin): 1 of 9 EXPOSED",
        "BLUR GATE: blur_enabled but the cached day master has no marker",
        "BLUR GATE (frame pairing): the source has 10 frames, the privacy mask 9 (OOM ROOM)",
    ):
        assert retry_policy.classify_failure(text) == "deterministic", text


def test_a_day_under_the_minimum_lf_is_deterministic_at_every_gate():
    # a day under the 480 s minimum LF can never fill a long-form: preflight and the
    # phase_a GPS-gap gate name min_lf_duration, and retrying the same footage never helps
    for text in (
        "Day 2026-03-04: raw source footage 300s is under min_lf_duration=480s, so the day "
        "can never fill a long-form. Refused before ingest.",
        "Day 2026-03-04: at most 200s of footage survives the GPS-gap cut (90s has no usable "
        "fix within 30s), under min_lf_duration=480s; the",
        "Day 2026-03-04: kept (stop-masked) footage 300s is under full_drive_part_min_sec=3600s "
        "and no best_moments window reaches the 480s minimum LF duration.",
    ):
        assert retry_policy.classify_failure(text) == "deterministic", text


def test_a_legacy_deterministic_row_gets_one_launch_then_is_refused_under_the_same_fingerprint(
    tmp_path,
):
    rig = Rig(tmp_path, legacy=True, error="LF 1: final-text review refused (place_outside_text)")
    out = rig.run()
    assert out["launched"] == [rig.row()["key"]] and len(rig.launches) == 1
    assert "failure_fingerprint" not in rig.row()  # heal never stamps a row it did not file
    _fail(rig, 1, error="LF 1: final-text review refused (place_outside_text)")
    out = rig.run(now=_at(100))
    assert out["launched"] == [] and len(rig.launches) == 1
    assert rig.row()["resume_attempts"][-1]["action"] == "refused"


def test_a_row_filed_under_a_fingerprint_is_refused_while_it_holds_and_launches_once_when_it_changes(
    tmp_path,
):
    rig = Rig(tmp_path, error="BLUR GATE: no marker")
    rig.installed = "0.3.1"
    rig.file_now()
    assert rig.run()["launched"] == [] and rig.launches == []
    rig.installed = "0.3.2"  # a release since the failure
    assert len(rig.run(now=_at(100))["launched"]) == 1
    assert len(rig.launches) == 1


def test_a_deterministic_refusal_raises_one_failure_unattended_that_does_not_grow(tmp_path):
    from gomoveshift.health import check

    rig = Rig(tmp_path, error="BLUR GATE: no marker")
    for i in range(4):
        rig.run(now=_at(i * 90000))
    rows = load_all(rig.inc)[0]
    drive_rows = [r for r in rows if r["kind"] == "drive_failed"]
    counts = []
    for i in range(4):
        now = _at(i * 90000)
        notes: list = []
        counts.append(len(check.unattended_incidents(drive_rows, now, now.isoformat(), notes)))
    assert counts == [1, 1, 1, 1]


# -- V19: a recipe-less publish or heal incident closes instead of re-refusing every tick ---------


@pytest.mark.parametrize("job_name", ["gms-publish-demo-drive-a", "gms-heal-gms-backlog-demo-a-1111111"])  # fmt: skip
def test_a_recipe_less_publish_or_heal_job_incident_is_closed_with_a_reason(tmp_path, job_name):
    rig = Rig(tmp_path, resume_recipe=None, job_name=job_name)
    out = rig.run()
    assert rig.launches == [] and out["launched"] == []
    row = rig.row()
    assert row["resolved_at"] and "no resume recipe" in row["resolution_reason"]
    assert row["resume_attempts"][0]["action"] == "closed"
    rig.run(now=NOW + timedelta(hours=1))  # resolved: never touched again
    assert len(rig.row()["resume_attempts"]) == 1


def test_a_recipe_less_publish_incident_dry_run_writes_nothing(tmp_path):
    rig = Rig(tmp_path, resume_recipe=None, job_name="gms-publish-demo-drive-a")
    out = rig.run(dry_run=True)
    assert rig.row()["resolved_at"] is None and out["attempts"][0]["action"] == "closed"


def test_a_recipe_less_backlog_job_incident_stays_open_and_refused(tmp_path):
    rig = Rig(tmp_path, resume_recipe=None, job_name="gms-backlog-demo-drive-a")
    rig.run()
    row = rig.row()
    assert row["resolved_at"] is None and row["resume_attempts"][0]["action"] == "refused"


def test_a_heal_of_a_heal_reuses_the_base_drive_name(tmp_path):
    nested = "gms-heal-gms-heal-demo-drive-a-1111111-2222222"
    rig = Rig(tmp_path, drive=nested, job_name=nested)
    rig.run()
    argv = rig.launches[0][0]
    assert argv[argv.index("--name") + 1] == "gms-heal-demo-drive-a-abcdef1"


# -- A run killed by a host restart is retried, not counted as used ----------------------------
NANNY = {"pid": 4242, "start_ticks": "77", "boot_id": "old-boot"}


def _running_heal_job(rig: Rig, n: int, *, nanny=NANNY) -> None:
    """The heal job the last launch started, left at "running" (its nanny died with the host)."""
    d = rig.jobs / f"i{n}"
    d.mkdir(parents=True)
    rec = {
        "schema": "job/1", "run_id": f"i{n}", "name": heal.job_name("demo-drive-a", SHA),
        "argv": ["x"], "created_at": _tick(n).timestamp() - 300, "result": {"state": "running"},
    }  # fmt: skip
    if nanny is not None:
        rec["nanny"] = nanny
    (d / "job.json").write_text(json.dumps(rec))


def _unknown_row(tmp_path) -> Rig:
    # class ``unknown``: one attempt per fingerprint, so a counted attempt is a refusal
    return Rig(tmp_path, error="SomethingNobodyClassified")


def test_dead_nanny_running_record_is_interrupted_and_pushed_once(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    _running_heal_job(rig, 1)
    rig.alive = False
    rig.run(now=_tick(1))
    att = rig.row()["resume_attempts"][0]
    assert att["outcome"] == "interrupted" and att["outcome_at"]
    rig.run(now=_tick(2))
    outcomes = [s for e, s in rig.pushes if e == "health_heal_outcome"]
    assert len(outcomes) == 1 and "interrupted" in outcomes[0]
    assert "will be retried" in outcomes[0]


def test_live_nanny_running_record_gets_no_outcome(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    _running_heal_job(rig, 1)
    rig.alive = True
    rig.run(now=_tick(1))
    assert "outcome" not in rig.row()["resume_attempts"][0]
    assert [e for e, _ in rig.pushes] == ["health_heal_launched"]


def test_running_record_without_a_nanny_identity_stays_unknown(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    _running_heal_job(rig, 1, nanny=None)
    rig.alive = False
    rig.run(now=_tick(1))
    assert "outcome" not in rig.row()["resume_attempts"][0]


def test_an_interrupted_attempt_does_not_block_a_relaunch_on_the_same_fingerprint(tmp_path):
    rig = _unknown_row(tmp_path)
    rig.run()
    _running_heal_job(rig, 1)
    rig.alive = False
    out = rig.run(now=_tick(1))  # 45 min later: inside the 1 h backoff a used attempt buys
    atts = rig.row()["resume_attempts"]
    assert atts[0]["outcome"] == "interrupted"
    assert len(rig.launches) == 2 and out["launched"] == ["drive_failed:demo-drive-a:r1"]
    assert atts[1]["action"] == "launched"


def test_a_failed_attempt_still_counts_against_the_budget(tmp_path):
    rig = _unknown_row(tmp_path)
    rig.run()
    _finish_heal_job(rig, 1, ENCODE_LOG)
    rig.run(now=_tick(1))
    assert len(rig.launches) == 1
    last = rig.row()["resume_attempts"][-1]
    assert last["action"] == "refused" and "1 of 1 attempts used" in last["reason"]


def test_interrupted_relaunches_are_capped_per_fingerprint(tmp_path):
    rig = _unknown_row(tmp_path)
    rig.alive = False
    cap = retry_policy.HEAL_INTERRUPTED_RETRIES
    for n in range(1, cap + 1):
        rig.run(now=_tick(n - 1))
        assert len(rig.launches) == n
        _running_heal_job(rig, n)
    rig.run(now=_tick(cap))  # the cap-th interrupt is recorded, then one more relaunch is allowed
    assert len(rig.launches) == cap + 1
    _running_heal_job(rig, cap + 1)
    rig.run(now=_tick(cap + 1))  # past the cap the interrupted attempt counts like a failure
    rig.run(now=_tick(cap + 2))
    assert len(rig.launches) == cap + 1
    atts = rig.row()["resume_attempts"]
    assert [a.get("outcome") for a in atts[: cap + 1]] == ["interrupted"] * (cap + 1)
    assert "attempts used" in atts[-1]["reason"]


def test_past_the_interrupted_cap_an_interrupted_attempt_counts_like_a_failure():
    cap = retry_policy.HEAL_INTERRUPTED_RETRIES
    attempts = [
        {"action": "launched", "fp_hash": "f", "outcome": "interrupted"} for _ in range(cap)
    ]
    assert retry_policy.launched_count(attempts, "f") == 0
    assert retry_policy.launched_count(attempts + attempts[:1], "f") == 1


# 2026-11-26: gomoveshiftsite#334 renamed the Ireland GPX files (enhanced-day14 -> day15) and
# Galley Head's heal resumed its recipe as written: job 75fb04ce exited 2 at 12:45 UTC,
# "--day-gpx: day GPX not found: .../ireland-2025-09-enhanced-day14.gpx". A recipe GPX that no
# longer exists is taken from the drive's backlog row when that row names a file that does.


def _gpx_rig(tmp_path, recipe_gpx, flag="--day-gpx"):
    args = ["drive", "in/demo-drive-a", "--publish", "fri", flag, *recipe_gpx]
    return Rig(tmp_path, drive_args=args)


def _tail(rig):
    argv, _cwd = rig.launches[0]
    return argv[argv.index("--") + 1 :]


def test_a_recipe_day_gpx_that_is_gone_is_taken_from_the_backlog_row(tmp_path):
    new = tmp_path / "gpx" / "day15.gpx"
    new.parent.mkdir()
    new.write_text("<gpx/>")
    rig = _gpx_rig(tmp_path, [str(tmp_path / "gpx" / "enhanced-day14.gpx")])
    rig.manifest_row = SimpleNamespace(day_gpx=str(new), bracket_gpx=())
    rig.run()
    tail = _tail(rig)
    assert tail[tail.index("--day-gpx") + 1] == str(new)
    att = rig.row()["resume_attempts"][0]
    assert att["action"] == "launched"
    assert att["gpx_refreshed"] == {"--day-gpx": [str(new)]}


def test_a_recipe_bracket_gpx_pair_that_is_gone_is_taken_from_the_backlog_row(tmp_path):
    pair = [tmp_path / "before.gpx", tmp_path / "after.gpx"]
    for f in pair:
        f.write_text("<gpx/>")
    rig = _gpx_rig(tmp_path, ["/gone/a.gpx", "/gone/b.gpx"], flag="--bracket-gpx")
    rig.manifest_row = SimpleNamespace(day_gpx="", bracket_gpx=tuple(str(f) for f in pair))
    rig.run()
    tail = _tail(rig)
    i = tail.index("--bracket-gpx")
    assert tail[i + 1 : i + 3] == [str(f) for f in pair]


@pytest.mark.parametrize("case", ["recipe file exists", "row file missing too", "no row"])
def test_a_recipe_gpx_is_kept_unless_it_is_gone(tmp_path, case):
    old = tmp_path / "old.gpx"
    if case == "recipe file exists":
        old.write_text("<gpx/>")
    other = tmp_path / "other.gpx"
    if case != "row file missing too":
        other.write_text("<gpx/>")
    rig = _gpx_rig(tmp_path, [str(old)])
    if case != "no row":
        rig.manifest_row = SimpleNamespace(day_gpx=str(other), bracket_gpx=())
    rig.run()
    att = rig.row()["resume_attempts"][0]
    assert "gpx_refreshed" not in att
    if case != "recipe file exists":  # gone with nothing to replace it: refused, never launched
        assert not rig.launches and att["action"] == "refused"
        assert f"--day-gpx {old} no longer exists" in att["reason"]
        return
    tail = _tail(rig)
    assert tail[tail.index("--day-gpx") + 1] == str(old)


def test_default_manifest_row_reads_the_backlog_row_of_the_heals_origin(monkeypatch, tmp_path):
    from seo_manager.scheduling import backlog

    rows = [SimpleNamespace(project="demo-drive-a", day_gpx="/g/day15.gpx")]
    monkeypatch.setattr(backlog, "load", lambda path: rows)
    row = {"drive": "gms-heal-gms-backlog-demo-drive-a-abc1234"}
    assert retry_policy.default_manifest_row(row) is rows[0]
    assert retry_policy.default_manifest_row({"drive": "elsewhere"}) is None


def test_heal_refuses_through_the_shared_install_pending_helper(tmp_path):
    rig = Rig(tmp_path)
    rig.pinned, rig.installed = "0.3.167", "0.3.163"
    rig.run()
    assert rig.launches == []
    assert rig.row()["resume_attempts"][0]["reason"] == heal.install_pending("0.3.163", "0.3.167")
    assert heal.install_pending("0.3.167", "0.3.167") is None
    assert heal.install_pending(None, "0.3.167") is None
    assert heal.install_pending("0.3.163", None) is None


def test_the_row_a_stale_publish_refusal_files_is_re_audited_by_the_hourly_heal(tmp_path):
    # Old Head to Courtmacsherry, 2026-11-27: ``health publish`` files this row when it refuses a
    # passed audit as stale. The hourly heal (no ``--recipe``) must launch it, not refuse it as
    # publish-only or deterministic from the refusal's own text.
    rig = Rig(tmp_path)
    why = ("health publish refused: demo-drive-b's checkpoint is stale under the current code "
           "(phase_bc_day_first, phase_d re-run); re-audit it (health heal) before publishing")  # fmt: skip
    key = heal.register_recipe(rig.inc, _second_recipe(tmp_path), now_iso="t", error=why)
    row = next(r for r in load_all(rig.inc)[0] if r["key"] == key)
    assert row["error"] == why
    out = rig.run()
    assert key in out["launched"], out
