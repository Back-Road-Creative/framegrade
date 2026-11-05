"""``health publish``: one operator command approves an auditing drive, claims its Waypost slot and
launches Phase D, and the hourly tick closes the claim. A temp sqlite store and a fake launcher;
nothing here uploads or starts a job."""

from __future__ import annotations

import json
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import pytest
from seo_manager.core.profile import Cadence, RegionCadence
from seo_manager.scheduling.store import ScheduleStore

from gomoveshift import run_origin
from gomoveshift.health import publish

ET = timezone(timedelta(hours=-4))
NOW = datetime(2026, 10, 20, 20, 0, tzinfo=ET)  # Tue evening: the next free slot is Fri 10-23
CAD = Cadence(regions={"usa": RegionCadence(dow=(4, 1), time=time(9, 0))}, lead_hours=72)


class Rig:
    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.store = ScheduleStore(tmp_path / "schedule.sqlite3", CAD.tz, CAD.regions)
        self.store.init()
        self.stages, self.out, self.jobs = tmp_path / "st", tmp_path / "out", tmp_path / "jobs"
        py = tmp_path / ".venv" / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("")
        self.launches: list = []
        self.launch_ok = True
        self.running = False

    def audit(self, project="day-a", status="auditing", recipe=True):
        out = self.out / project
        self.store.register_ready(
            project, "usa", 1, output_dir=str(out), status=status, mirror_sha="abc1234",
            lf_mode="full_drive", ingest_mode="gopro",
        )  # fmt: skip
        if recipe:
            stages = self.stages / project / project / "_stages"
            stages.mkdir(parents=True)
            argv = [
                "/old/py", "-m", "gomoveshift", "drive", str(self.out / "_sources" / project),
                "--region", "usa", "--publish", "mon", "--lf-day", "mon", "--ingest-mode",
                "gopro", "--lf-mode", "full_drive", "--output-dir", str(out), "--skip-upload",
                "--no-youtube-upload", "--no-review", "--force", "--no-tmux",
                "--day-gpx", "/gpx/a.gpx", "--camera-utc-offset-hours", "-4",
            ]  # fmt: skip
            body = {"argv": argv, "cwd": "/old", "env": {"GMS_INTERMEDIATE_DIR": str(self.stages / project)}}  # fmt: skip
            (stages / "resume.json").write_text(json.dumps(body))
        return out

    def launcher(self, argv, cwd, env):
        self.launches.append((argv, cwd, env))
        return self.launch_ok, "" if self.launch_ok else "runner: boom"

    def run(self, project="day-a", *, dry_run=False):
        return publish.run_publish(
            project=project, region="usa", store=self.store, cadence=CAD, stages_root=self.stages,
            code_root=self.tmp, jobs_dir=self.jobs, now=NOW, dry_run=dry_run,
            drive_running=lambda jobs: self.running, launcher=self.launcher,
        )  # fmt: skip

    def snapshot(self):
        drives = [(d.project, d.status, d.claim_id) for d in self.store.all_drives()]
        return drives, [(b.project, b.slot_at) for b in self.store.bookings()]


def test_happy_path_approves_claims_and_launches_phase_d(tmp_path):
    rig = Rig(tmp_path)
    rig.audit()
    res = rig.run()
    assert res["action"] == "launched", res
    d = rig.store.drive("day-a")
    assert d.status == "claimed" and d.claim_id == res["claim_id"]
    assert [b.slot_at.date().isoformat() for b in rig.store.bookings()] == ["2026-10-23"]
    argv, cwd, env = rig.launches[0]
    assert argv[argv.index("--name") + 1] == "gms-publish-day-a"
    drive = argv[argv.index("--") + 1 :]
    assert drive[0] == str(tmp_path / ".venv" / "bin" / "python")
    assert drive[1:4] == ["-m", "gomoveshift", "drive"]
    assert drive[drive.index("--lf-publish-date") + 1] == "2026-10-23"
    assert drive[drive.index("--phase") + 1] == "D"
    for flag in ("--youtube-upload", "--force", "--no-tmux", "--day-gpx", "--camera-utc-offset-hours"):  # fmt: skip
        assert drive.count(flag) == 1
    assert "--skip-upload" not in drive and "--no-youtube-upload" not in drive
    assert drive[drive.index("--publish") + 1] == drive[drive.index("--lf-day") + 1] == "fri"
    assert drive[drive.index("--output-dir") + 1] == str(rig.out / "day-a")
    assert env["GMS_INTERMEDIATE_DIR"] == str(rig.stages / "day-a")
    assert env[run_origin.LAUNCHED_BY_ENV] in run_origin.LAUNCHERS


def test_oldest_ready_drive_is_someone_elses_and_is_refused_untouched(tmp_path):
    rig = Rig(tmp_path)
    rig.audit("older", status="ready", recipe=False)
    rig.audit("day-a")
    before = rig.snapshot()
    res = rig.run()
    assert res["action"] == "refused" and "older" in res["reason"]
    assert rig.snapshot() == before and not rig.launches


def test_race_claim_of_another_drive_is_undone(tmp_path, monkeypatch):
    rig = Rig(tmp_path)
    rig.audit("racer", status="auditing", recipe=False)
    rig.audit("day-a")
    real = publish.claim_or_reason

    def claim_other(store, *a, **kw):  # a drive that became ready first wins the claim
        store.set_status("racer", "ready")
        store.set_status("day-a", "auditing")
        return real(store, *a, **kw)

    monkeypatch.setattr(publish, "claim_or_reason", claim_other)
    res = rig.run()
    assert res["action"] == "refused" and "racer" in res["reason"]
    assert {d.project: d.status for d in rig.store.all_drives()} == {"racer": "ready", "day-a": "auditing"}  # fmt: skip
    assert rig.store.bookings() == [] and not rig.launches


@pytest.mark.parametrize("status", ["ready", "claimed", "complete"])
def test_only_an_auditing_drive_can_be_published(tmp_path, status):
    rig = Rig(tmp_path)
    rig.audit(status="ready")
    if status != "ready":
        rig.store.claim_drive("day-a", [datetime(2026, 10, 23, 9, tzinfo=ET)], "c1", now=NOW)
        if status != "claimed":
            rig.store.finish_claim("c1", ok=status == "complete")
    before = rig.snapshot()
    res = rig.run()
    assert res["action"] == "refused" and status in res["reason"]
    assert rig.snapshot() == before and not rig.launches


def test_unknown_project_and_missing_recipe_are_refused_before_any_write(tmp_path):
    rig = Rig(tmp_path)
    assert "not in the schedule store" in rig.run("ghost")["reason"]
    rig.audit(recipe=False)
    before = rig.snapshot()
    res = rig.run()
    assert res["action"] == "refused" and "resume.json" in res["reason"]
    assert rig.snapshot() == before


def test_dry_run_reports_the_steps_and_changes_nothing(tmp_path):
    rig = Rig(tmp_path)
    rig.audit()
    before = rig.snapshot()
    res = rig.run(dry_run=True)
    assert res["action"] == "would_publish" and len(res["steps"]) == 4
    assert rig.snapshot() == before and not rig.launches


def test_launch_failure_puts_the_drive_back_to_auditing(tmp_path):
    rig = Rig(tmp_path)
    rig.audit()
    rig.launch_ok = False
    res = rig.run()
    assert res["action"] == "refused" and "launch failed" in res["reason"]
    assert rig.store.drive("day-a").status == "auditing" and rig.store.bookings() == []


def test_a_running_drive_refuses_before_any_write(tmp_path):
    rig = Rig(tmp_path)
    rig.audit()
    rig.running = True
    before = rig.snapshot()
    assert rig.run()["action"] == "refused" and rig.snapshot() == before


# -- reconcile (the hourly tick) ------------------------------------------------------------


def _job(rig, project, state, code, created):
    d = rig.jobs / f"{project}-{state}-{created}"
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({
        "schema": "job/1", "run_id": d.name, "name": f"gms-publish-{project}",
        "created_at": created, "result": {"state": state, "exit_code": code},
    }))  # fmt: skip


def _published(rig, project):
    out = rig.out / project
    out.mkdir(parents=True, exist_ok=True)
    (out / "youtube_uploads.json").write_text(json.dumps({"lf1": {"video_id": "abc"}}))


def _claimed(tmp_path):
    rig = Rig(tmp_path)
    rig.audit()
    assert rig.run()["action"] == "launched"
    return rig, rig.store.drive("day-a").claimed_at.timestamp() + 5


def test_reconcile_completes_a_claim_whose_job_uploaded(tmp_path):
    rig, t = _claimed(tmp_path)
    _job(rig, "day-a", "succeeded", 0, t)
    _published(rig, "day-a")
    assert publish.reconcile_publishes(rig.store, rig.jobs) == [{"project": "day-a", "status": "complete"}]  # fmt: skip
    assert rig.store.drive("day-a").status == "complete" and len(rig.store.bookings()) == 1
    assert publish.reconcile_publishes(rig.store, rig.jobs) == []  # idempotent


def test_reconcile_fails_and_releases_when_the_job_failed(tmp_path):
    rig, t = _claimed(tmp_path)
    _job(rig, "day-a", "failed", 1, t)
    assert publish.reconcile_publishes(rig.store, rig.jobs) == [{"project": "day-a", "status": "failed"}]  # fmt: skip
    assert rig.store.drive("day-a").status == "failed" and rig.store.bookings() == []


def test_reconcile_waits_for_a_running_job_and_ignores_a_stale_one(tmp_path):
    rig, t = _claimed(tmp_path)
    assert publish.reconcile_publishes(rig.store, rig.jobs) == []  # no job record yet
    _job(rig, "day-a", "failed", 1, t - 3600)  # an older attempt, before this claim
    _job(rig, "day-a", "running", None, t)
    assert publish.reconcile_publishes(rig.store, rig.jobs) == []
    assert rig.store.drive("day-a").status == "claimed"


def test_reconcile_leaves_a_claim_with_no_publish_job_alone(tmp_path):
    rig = Rig(tmp_path)
    rig.audit(status="ready")
    rig.store.claim_drive("day-a", [datetime(2026, 10, 23, 9, tzinfo=ET)], "c1", now=NOW)
    _job(rig, "other", "failed", 1, NOW.timestamp())
    assert publish.reconcile_publishes(rig.store, rig.jobs) == []


# -- CLI wiring -------------------------------------------------------------------------------


def test_cli_parses_and_refuses_with_exit_2_when_the_store_is_absent(tmp_path, capsys, monkeypatch):
    from types import SimpleNamespace

    from gomoveshift.cli.handlers import health
    from gomoveshift.cli.parser import _build_parser
    from gomoveshift.pipeline import seo_slots

    args = _build_parser().parse_args(
        ["health", "publish", "day-a", "--region", "usa", "--stages-root", str(tmp_path),
         "--data-root", str(tmp_path), "--dry-run"]
    )  # fmt: skip
    assert (args.project, args.dry_run, args.lead_hours) == ("day-a", True, None)
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path / "seo")))
    monkeypatch.setattr(seo_slots, "waypost_cadence", lambda cfg: CAD)
    assert health.cmd_health(args, config) == 2
    assert "does not exist" in capsys.readouterr().out


# -- a ready drive already uploaded outside a claim (Beaufort, 2026-11-05) ----------------------


def _uploaded_outside_a_claim(rig, project="beaufort"):
    """A ``ready`` row whose upload pass ran outside any claim: the receipt sits one level down
    (the run dir) and Waypost booked the video under that run dir's name, not the drive's."""
    out = rig.audit(project, status="ready", recipe=False)
    (out / "src").mkdir(parents=True)
    at = "2026-10-27T20:00:00+00:00"
    (out / "src" / "youtube_uploads.json").write_text(json.dumps({"1": {"video_id": "XIT4", "publish_at": at}}))  # fmt: skip
    rig.store.book("src", 1, "usa", datetime.fromisoformat(at), video_id="XIT4", booked_at=NOW)


def test_publish_retires_an_older_ready_drive_that_is_already_uploaded(tmp_path):
    rig = Rig(tmp_path)
    _uploaded_outside_a_claim(rig)
    rig.audit("day-a")
    res = rig.run()
    assert res["action"] == "launched", res
    assert rig.store.drive("beaufort").status == "complete"
    assert rig.store.drive("day-a").status == "claimed"
    assert {b.project: b.video_id for b in rig.store.bookings()}["beaufort"] == "XIT4"
    assert "src" not in {b.project for b in rig.store.bookings()}  # the stray booking is adopted


def test_publish_dry_run_names_the_retirement_and_changes_nothing(tmp_path):
    rig = Rig(tmp_path)
    _uploaded_outside_a_claim(rig)
    rig.audit("day-a")
    before = rig.snapshot()
    res = rig.run(dry_run=True)
    assert res["action"] == "would_publish", res
    assert any(s.startswith("retire: beaufort") for s in res["steps"])
    assert rig.snapshot() == before


def test_reconcile_retires_a_ready_drive_that_is_already_uploaded(tmp_path):
    rig = Rig(tmp_path)
    _uploaded_outside_a_claim(rig)
    assert publish.reconcile_publishes(rig.store, rig.jobs) == [{"project": "beaufort", "status": "complete"}]  # fmt: skip
    assert rig.store.drive("beaufort").status == "complete"
    assert publish.reconcile_publishes(rig.store, rig.jobs) == []  # idempotent


def test_a_receipt_without_publish_at_is_left_ready(tmp_path):
    rig = Rig(tmp_path)
    rig.audit("older", status="ready", recipe=False)
    _published(rig, "older")  # video id but no publish_at: Waypost cannot book it
    assert publish.reconcile_publishes(rig.store, rig.jobs) == []
    assert rig.store.drive("older").status == "ready"


def test_a_checkpoint_the_current_code_re_runs_is_refused_before_any_write(tmp_path):
    """Kenmare, 2026-11-05: phase_bc_day_first went 3 -> 4 after the audit, so Phase D refused
    its prerequisite after the claim was booked. The publish now refuses first: re-audit it."""
    from gomoveshift.pipeline.orchestrator_fingerprint import PHASE_CODE_VERSIONS

    rig = Rig(tmp_path)
    rig.audit()
    stale = {"phase_a": PHASE_CODE_VERSIONS["phase_a"], "phase_bc_day_first": PHASE_CODE_VERSIONS["phase_bc_day_first"] - 1}  # fmt: skip
    state = rig.stages / "day-a" / "day-a" / "_stages" / "pipeline_state.json"
    state.write_text(json.dumps({"phases_completed": ["phase_a", "phase_bc_day_first"], "phase_code_versions": stale}))  # fmt: skip
    before = rig.snapshot()
    res = rig.run()
    assert res["action"] == "refused" and "phase_bc_day_first" in res["reason"], res
    assert "re-audit" in res["reason"]
    assert rig.snapshot() == before and not rig.launches


def _failed_publish(rig):
    rig.audit(status="ready")
    rig.store.claim_drive("day-a", [datetime(2026, 10, 23, 9, tzinfo=ET)], "c1", now=NOW)
    rig.store.finish_claim("c1", ok=False)
    assert rig.store.drive("day-a").status == "failed" and rig.store.bookings() == []


def test_a_failed_publish_with_no_upload_is_reopened_and_launched(tmp_path):
    # A Phase D job that died before uploading leaves the drive failed; its audit still holds,
    # so the next ``health publish`` reopens it (auditing) and claims a slot again.
    rig = Rig(tmp_path)
    _failed_publish(rig)
    res = rig.run()
    assert res["action"] == "launched", res
    assert res["steps"][0].startswith("reopen: day-a failed -> auditing")
    assert rig.store.drive("day-a").status == "claimed" and len(rig.launches) == 1


def test_reopening_a_failed_drive_is_dry_run_safe(tmp_path):
    rig = Rig(tmp_path)
    _failed_publish(rig)
    before = rig.snapshot()
    res = rig.run(dry_run=True)
    assert res["action"] == "would_publish" and res["steps"][0].startswith("reopen:")
    assert rig.snapshot() == before and not rig.launches


def test_a_failed_drive_with_an_upload_receipt_is_still_refused(tmp_path):
    rig = Rig(tmp_path)
    _failed_publish(rig)
    _published(rig, "day-a")
    before = rig.snapshot()
    res = rig.run()
    assert res["action"] == "refused" and "upload receipt" in res["reason"]
    assert rig.snapshot() == before and not rig.launches
