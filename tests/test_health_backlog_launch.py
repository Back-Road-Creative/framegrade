"""``health backlog``: launch the next ``drive_backlog.yaml`` row for a region's uncovered LF
slot, audit only. Every side effect (lease, process probe, disk, preflight, launcher, store
write, clock) is injected; nothing here launches a real job."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import shutil
from types import SimpleNamespace

import pytest
from seo_manager.scheduling.backlog import BacklogRow

from gomoveshift.health import heal, launch
from gomoveshift.pipeline.preflight import PreflightBlockedError

ET = timezone(timedelta(hours=-4))
NOW = datetime(2026, 10, 20, 20, 0, tzinfo=ET)
FRI = datetime(2026, 10, 23, 9, 0, tzinfo=ET)
TUE = datetime(2026, 10, 27, 9, 0, tzinfo=ET)
GB = 1024**3


class Rig:
    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.footage = tmp_path / "footage"
        self.footage.mkdir(parents=True, exist_ok=True)
        self.stages = tmp_path / "stages"
        self.stages.mkdir(exist_ok=True)
        self.out = tmp_path / "out"
        self.out.mkdir(exist_ok=True)
        self.sources = tmp_path / "sources"
        python = tmp_path / ".venv" / "bin" / "python"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text("")
        self.rows = [self.row("day-a", 10), self.row("day-b", 4)]
        self.drives: list = []
        self.slots = [(FRI, "fri"), (TUE, "tue")]
        self.ledger: set = set()
        self.running = False
        self.lease_free = True
        self.leases: list = []
        self.devices = {self.stages: (1, 500.0), self.out: (2, 500.0)}
        self.preflight_error = None
        self.preflights: list = []
        self.launches: list = []
        self.launch_ok = True
        self.registered: list = []
        self.retired: list = []
        self.installed = "0.3.25"
        self.latest = "0.3.25"
        self.supersede = None
        self.replace_scheduled = False
        self.live = None  # replace_scheduled: video ids -> {id: state}
        self.replace_store = None  # replace_scheduled: a fake Waypost store
        self.gate_reason = None
        self.gates: list = []
        self.publish_result = None  # set -> run_backlog gets a publisher returning it
        self.publishes: list = []
        self.relaunch = None  # a ``launch.FailedRelaunch`` -> run_backlog gets it

    def row(self, project, gb, region="usa"):
        files = []
        for i in range(2):
            f = self.footage / f"{project}-GX0{i}.MP4"
            with open(f, "wb") as fh:
                fh.truncate(int(gb * GB / 2))
            files.append(str(f))
        return BacklogRow(project, region, tuple(files), "gopro", "full_drive")

    def disk(self, path):
        path = Path(path)
        for root, value in self.devices.items():
            if path == root or root in path.parents:
                return value
        raise AssertionError(f"no device for {path}")

    def lease_acquire(self):
        self.leases.append("acquire")
        return self.lease_free

    def preflight(self, source_dir, output_dir, *, staging_dir, ingest_mode, python):
        self.preflights.append((source_dir, output_dir, staging_dir, ingest_mode, python))
        if self.preflight_error:
            raise self.preflight_error

    def render_gate(self, subset, row):
        self.gates.append((subset, row.project))
        if isinstance(self.gate_reason, dict):  # per-row refusals
            return self.gate_reason.get(row.project)
        return self.gate_reason

    def launcher(self, argv, cwd, env):
        self.launches.append((argv, cwd, env))
        return self.launch_ok, "" if self.launch_ok else "runner: boom"

    def kwargs(self, *, dry_run=False, region="usa"):
        return dict(
            region=region, rows=self.rows, drives=self.drives, free_slots=self.slots,
            known_projects=self.ledger, stages_root=self.stages, output_root=self.out,
            sources_root=self.sources, code_root=self.tmp, jobs_dir=self.tmp / "jobs",
            mirror_sha="abc1234", now=NOW, dry_run=dry_run,
            lease_acquire=self.lease_acquire,
            lease_release=lambda: self.leases.append("release"),
            drive_running=lambda: self.running, disk=self.disk, preflight=self.preflight,
            render_gate=self.render_gate, launcher=self.launcher,
            register=lambda **kw: self.registered.append(kw), retire=self.retired.append,
            installed_version=lambda python: self.installed,
            latest_version=lambda: self.latest,
            **({"supersede": self.supersede} if self.supersede else {}),
            **({"replace_scheduled": True, "live_status": self.live} if self.replace_scheduled else {}),
            **({"replace_store": lambda: self.replace_store} if self.replace_store else {}),
            **({"publish": self.publish} if self.publish_result else {}),
            **({"failed_relaunch": self.relaunch} if self.relaunch else {}),
        )  # fmt: skip

    def publish(self, project, dry_run):
        self.publishes.append((project, dry_run))
        return {"project": project, "steps": [], **self.publish_result}

    def run(self, *, dry_run=False, region="usa"):
        return launch.run_backlog(**self.kwargs(dry_run=dry_run, region=region))

    def stage(self, run, source_files, *, hold=False):
        """A prior run's stage root: subset dir of symlinks + its resume recipe."""
        subset = self.tmp / "prior" / run
        subset.mkdir(parents=True)
        for f in source_files:
            (subset / Path(f).name).symlink_to(f)
        stages = self.stages / run / "src" / "_stages"
        stages.mkdir(parents=True)
        argv = ["/py", "-m", "gomoveshift", "drive", str(subset), "--region", "usa"]
        (stages / "resume.json").write_text(json.dumps({"argv": argv}))
        if hold:
            (stages / launch.HOLD_NAME).write_text("one attempt only")
        return stages


def drive(project, status, region="usa", output_dir=None):
    return SimpleNamespace(project=project, region=region, status=status, output_dir=output_dir)


def uploaded(tmp_path, name, records, sub=""):
    """An output dir whose Phase D receipt (``youtube_uploads.json``) holds ``records``."""
    out = tmp_path / name
    (out / sub).mkdir(parents=True, exist_ok=True)
    (out / sub / "youtube_uploads.json").write_text(records if isinstance(records, str) else json.dumps(records))  # fmt: skip
    return str(out)


def test_launches_first_eligible_row_for_the_first_uncovered_slot(tmp_path):
    rig = Rig(tmp_path)
    res = rig.run()
    assert res["action"] == "launched", res
    assert (res["project"], res["lf_day"]) == ("day-a", "fri")
    argv, cwd, env = rig.launches[0]
    assert argv[argv.index("--name") + 1] == "gms-backlog-day-a"
    drive_args = argv[argv.index("--") + 1 :]
    assert drive_args[1:4] == ["-m", "gomoveshift", "drive"]
    assert drive_args[4] == str(rig.sources / "day-a")
    for flag in ("--skip-upload", "--no-youtube-upload", "--no-review", "--force", "--no-tmux"):
        assert flag in drive_args
    assert drive_args[drive_args.index("--lf-day") + 1] == "fri"
    assert drive_args[drive_args.index("--publish") + 1] == "fri"
    assert drive_args[drive_args.index("--ingest-mode") + 1] == "gopro"
    assert drive_args[drive_args.index("--lf-mode") + 1] == "full_drive"
    assert "--youtube-upload" not in drive_args
    assert env["GMS_INTERMEDIATE_DIR"] == str(rig.stages / "day-a")
    # The job runner's --expect: the drive's last encode (shorts) under this row's output dir, so
    # an exit 0 that wrote no fresh shorts records failed-no-output, not succeeded.
    expect = argv[argv.index("--expect") + 1]
    assert expect == str(rig.out / "day-a" / "**" / "shorts" / "**" / "*.mp4")
    assert argv.index("--expect") < argv.index("--")
    links = sorted(p.name for p in (rig.sources / "day-a").iterdir())
    assert links == sorted(Path(f).name for f in rig.rows[0].sources)
    assert rig.registered == [
        dict(
            project="day-a", region="usa", lf_count=1, status="auditing",
            output_dir=str(rig.out / "day-a"), mirror_sha="abc1234",
            lf_mode="full_drive", ingest_mode="gopro",
        )
    ]  # fmt: skip
    assert rig.leases == ["acquire", "release"]
    assert rig.preflights[0][2] == rig.stages / "day-a" / "day-a" / "_stages"


def test_dry_run_changes_nothing_and_takes_no_lease(tmp_path):
    rig = Rig(tmp_path)
    res = rig.run(dry_run=True)
    assert res["action"] == "would_launch"
    assert res["argv"] and not rig.launches and not rig.registered and not rig.preflights
    assert not rig.sources.exists() and rig.leases == []


def test_lease_held_is_a_noop_and_never_released_by_us(tmp_path):
    rig = Rig(tmp_path)
    rig.lease_free = False
    res = rig.run()
    assert res["action"] == "noop" and "lease" in res["reason"]
    assert rig.leases == ["acquire"] and not rig.launches


def test_running_drive_is_a_noop(tmp_path):
    rig = Rig(tmp_path)
    rig.running = True
    res = rig.run()
    assert res["action"] == "noop" and "running" in res["reason"]
    assert not rig.launches and rig.leases == ["acquire", "release"]


def test_repeat_invocation_after_a_launch_is_a_noop(tmp_path):
    rig = Rig(tmp_path)
    assert rig.run()["action"] == "launched"
    rig.drives.append(drive("day-a", "auditing"))  # what register() wrote
    res = rig.run()
    assert res["action"] == "noop" and "day-a" in res["reason"]
    assert len(rig.launches) == 1


def live_worker():
    """The identity ``heal._proc_alive`` reads for a process that is alive: this test process."""
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    ticks = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]
    return {"pid": os.getpid(), "boot_id": boot, "start_ticks": ticks}


def job(rig, run_id, project, state, created_at, exit_code=1, name=None, worker_pid=4242, live=False):  # fmt: skip
    """A job-runner record for the backlog drive of ``project`` (``worker_pid=None``: jobrun's own
    launch failure, the worker never started; ``live``: its worker is a process that is alive)."""
    d = rig.tmp / "jobs" / run_id
    d.mkdir(parents=True)
    result = {"state": state, "exit_code": None if state == "running" else exit_code}
    rec = {"schema": "job/1", "run_id": run_id, "name": name or f"gms-backlog-{project}",
           "created_at": created_at, "worker": live_worker() if live else {"pid": worker_pid}, "result": result}  # fmt: skip
    (d / "job.json").write_text(json.dumps(rec))


# 2026-11-10: Clonakilty's backlog job failed 11-08; heal resumed it as
# gms-heal-gms-backlog-<project>-c4aedcb, which completed 11-10 06:20. Every tick still refused
# "audit failed ... job 4847741a" (only the gms-backlog- name was read), exit 2, so the region's
# queue reported a failure while the drive sat audited, waiting for health publish.


def test_a_heal_resume_that_succeeded_after_the_failed_audit_clears_the_refusal(tmp_path):
    rig = Rig(tmp_path)
    rig.drives.append(drive("day-a", "auditing"))
    job(rig, "old", "day-a", "failed", 100.0)
    job(rig, "heal", "day-a", "succeeded", 200.0, exit_code=0,
        name="gms-heal-gms-backlog-day-a-c4aedcb")  # fmt: skip
    res = rig.run()
    assert res["action"] == "noop" and "audit in progress" in res["reason"]


def test_a_heal_of_a_heal_counts_and_a_failed_one_is_listed_naming_it(tmp_path):
    rig = Rig(tmp_path)
    rig.drives.append(drive("day-a", "auditing"))
    job(rig, "old", "day-a", "succeeded", 100.0, exit_code=0)
    job(rig, "heal2", "day-a", "failed", 300.0,
        name="gms-heal-gms-heal-gms-backlog-day-a-c4aedcb-49f2171")  # fmt: skip
    res = rig.run()
    assert res["action"] == "launched" and "heal2" in res["failed_audits"][0]["reason"], res


# 2026-11-17: Clonakilty's heal resumes failed overnight; the operator relaunched it with
# `gms-health-cron.sh heal-recipe <resume.json>`. A recipe-registered row names the drive by its
# stages dir, so the job was gms-heal-<project>-7e3f368 (no gms-backlog- inside). It succeeded at
# 05:25 ET, yet the 05:45 tick still refused "audit failed ... job 07d33eab", exit 2.


def test_a_recipe_heal_named_after_the_stages_dir_counts_as_the_audit(tmp_path):
    rig = Rig(tmp_path)
    rig.drives.append(drive("day-a", "auditing"))
    job(rig, "old", "day-a", "failed", 100.0)
    job(rig, "recipe", "day-a", "succeeded", 200.0, exit_code=0, name="gms-heal-day-a-7e3f368")
    res = rig.run()
    assert res["action"] == "noop" and "audit in progress" in res["reason"]


def test_a_bare_project_name_or_another_projects_recipe_heal_is_not_this_audit(tmp_path):
    rig = Rig(tmp_path)
    rig.drives.append(drive("day-a", "auditing"))
    job(rig, "old", "day-a", "failed", 100.0)
    job(rig, "bare", "day-a", "succeeded", 200.0, exit_code=0, name="day-a")
    job(rig, "other", "day-a", "succeeded", 300.0, exit_code=0, name="gms-heal-day-ab-7e3f368")
    res = rig.run()
    assert res["action"] == "launched" and "old" in res["failed_audits"][0]["reason"], res


def test_another_projects_heal_or_a_longer_project_name_is_not_this_audit(tmp_path):
    rig = Rig(tmp_path)
    rig.drives.append(drive("day-a", "auditing"))
    job(rig, "old", "day-a", "failed", 100.0)
    job(rig, "b", "day-a", "succeeded", 200.0, exit_code=0, name="gms-heal-gms-backlog-day-b-c4aedcb")  # fmt: skip
    job(rig, "c", "day-a", "succeeded", 300.0, exit_code=0, name="gms-backlog-day-a-c4aedcb")
    job(rig, "d", "day-a", "succeeded", 400.0, exit_code=0, name="gms-heal-gms-backlog-day-a")
    res = rig.run()
    assert res["action"] == "launched" and "old" in res["failed_audits"][0]["reason"], res


# 2026-11-10: Clonakilty's audit passed at 06:20, then sat ``auditing`` holding Ireland's next
# slot ("audit in progress") until a person ran ``health publish``, whose first try at 09:39 was
# refused ``outside_lead_window`` (slot Sun 11-15 2 PM ET, window opens Thu 11-12 2 PM ET). The
# hourly backlog tick now asks ``health publish`` itself once the newest audit has succeeded.


def _audited(rig, **publish_result):
    rig.drives.append(drive("day-a", "auditing"))
    job(rig, "ok", "day-a", "succeeded", 100.0, exit_code=0)
    rig.publish_result = publish_result


def test_a_succeeded_audit_is_published_by_the_backlog_tick(tmp_path):
    rig = Rig(tmp_path)
    _audited(rig, action="launched", reason="day-a for 2026-10-23T09:00:00-04:00",
             argv=["python3", "jobrun.py"], slot="2026-10-23T09:00:00-04:00")  # fmt: skip
    res = rig.run()
    assert rig.publishes == [("day-a", False)]
    assert res["action"] == "launched" and res["project"] == "day-a", res
    assert "published: day-a for 2026-10-23" in res["reason"]
    assert res["publish"]["slot"] == "2026-10-23T09:00:00-04:00"
    assert not rig.launches and not rig.registered  # the publisher launched Phase D, not backlog
    assert launch.exit_code(res) == 0


# 2026-11-19: Kerry-to-Cork's audit passed and its publish job (Phase D) failed at 08:01 ET on a
# venv rebuilt under it. ``health check`` closed the claim ``failed`` and every later backlog tick
# skipped it "in the schedule store (failed)": an audited drive's publish was abandoned. A listed
# drive whose newest publish job failed is published again (``health publish`` reopens it).


def _failed_publish(rig, state="failed", **publish_result):
    rig.drives.append(drive("day-a", "failed"))
    job(rig, "ok", "day-a", "succeeded", 100.0, exit_code=0)
    job(rig, "pub", "day-a", state, 200.0, name="gms-publish-day-a")
    rig.publish_result = publish_result or {"action": "launched", "reason": "day-a for slot"}


def test_a_failed_publish_of_a_listed_drive_is_published_again(tmp_path):
    rig = Rig(tmp_path)
    _failed_publish(rig)
    res = rig.run()
    assert rig.publishes == [("day-a", False)]
    assert res["action"] == "launched" and res["project"] == "day-a", res
    assert not rig.launches and not rig.registered


# 2026-11-23 (Kerry-to-Cork): its publish was refused (``mask_frame_count_unknown``, then a
# duplicate title) and the refusal RETURNED from the failed-publish loop, so every tick ended
# there and morning-kerry/afternoon-kerry never launched. A refused drive stays unpublished, but
# it no longer holds its region: it is reported under ``failed_publishes`` and the tick goes on.

MASK_REFUSAL = "mask_frame_count_unknown: no container to count"


def _refused_failed_publish(rig, project="day-a"):
    rig.drives.append(drive(project, "failed"))
    job(rig, f"ok-{project}", project, "succeeded", 100.0, exit_code=0)
    job(rig, f"pub-{project}", project, "failed", 200.0, name=f"gms-publish-{project}")
    rig.publish_result = {"action": "refused", "reason": MASK_REFUSAL}


def test_a_refused_failed_publish_does_not_hold_the_region(tmp_path):
    rig = Rig(tmp_path)
    _refused_failed_publish(rig)
    res = rig.run()
    assert rig.publishes == [("day-a", False)]
    assert res["action"] == "launched" and res["project"] == "day-b", res
    assert [a[0][a[0].index("--name") + 1] for a in rig.launches] == ["gms-backlog-day-b"]
    assert [r["project"] for r in rig.registered] == ["day-b"]


def test_a_refused_publish_is_reported_beside_the_launch(tmp_path, caplog):
    rig = Rig(tmp_path)
    _refused_failed_publish(rig)
    with caplog.at_level("WARNING", logger=launch.logger.name):
        res = rig.run()
    assert res["action"] == "launched", res
    assert res["failed_publishes"][0]["project"] == "day-a"
    assert MASK_REFUSAL in res["failed_publishes"][0]["reason"]
    assert "publish" not in res  # the refused drive's publish result is not the tick's
    assert any(
        "day-a" in r.getMessage() and MASK_REFUSAL in r.getMessage() and r.levelname == "WARNING"
        for r in caplog.records
    )


def test_a_refused_publish_is_not_relaunched_by_the_row_loop_in_the_same_tick(tmp_path):
    rig = Rig(tmp_path)
    _refused_failed_publish(rig)
    rig.rows[:] = [rig.row("day-a", 10), rig.row("day-b", 4)]  # the refused drive is listed first
    res = rig.run()
    assert res["project"] == "day-b" and len(rig.launches) == 1
    assert all(r["project"] != "day-a" for r in rig.registered)
    assert any(s["project"] == "day-a" for s in res["skipped"])  # named, never silently passed


def test_two_refused_failed_drives_are_both_reported_and_a_fresh_row_launches(tmp_path):
    rig = Rig(tmp_path)
    rig.rows.append(rig.row("day-c", 4))
    rig.rows.append(rig.row("day-d", 4))
    _refused_failed_publish(rig, "day-a")
    _refused_failed_publish(rig, "day-c")
    res = rig.run()
    assert [p for p, _ in rig.publishes] == ["day-a", "day-c"]
    assert [f["project"] for f in res["failed_publishes"]] == ["day-a", "day-c"]
    assert res["action"] == "launched" and res["project"] == "day-b", res


def test_a_refused_publish_with_no_fresh_row_is_reported_not_silent(tmp_path):
    rig = Rig(tmp_path)
    rig.rows[:] = [r for r in rig.rows if r.project == "day-a"]
    _refused_failed_publish(rig)
    res = rig.run()
    # still fail-closed: nothing published or relaunched
    assert not rig.launches and not rig.registered
    assert res["failed_publishes"][0]["project"] == "day-a", res


def test_an_auditing_drives_refused_publish_does_not_hold_the_region(tmp_path):
    rig = Rig(tmp_path)
    _audited(rig, action="refused", reason=MASK_REFUSAL)
    res = rig.run()
    assert rig.publishes == [("day-a", False)]  # asked once, never re-asked by a later loop
    assert res["action"] == "launched" and res["project"] == "day-b", res
    assert [f["project"] for f in res["failed_publishes"]] == ["day-a"]
    assert MASK_REFUSAL in res["failed_publishes"][0]["reason"]
    assert [r["project"] for r in rig.registered] == ["day-b"]


def test_a_refused_failed_drive_with_no_resume_json_is_superseded_from_source(tmp_path):
    # the refused drive's stage dir holds only calibration.json, so heal can never resume it and
    # ``FailedRelaunch`` relaunches it from source once its failure fingerprint changed
    from gomoveshift.health import retry_policy
    from gomoveshift.health.store import new_incident, write_row

    rig = Rig(tmp_path)
    rig.rows[:] = [rig.row("day-a", 10)]
    _refused_failed_publish(rig)
    (rig.stages / "day-a" / "day-a" / "_stages").mkdir(parents=True)
    (rig.stages / "day-a" / "day-a" / "_stages" / "calibration.json").write_text("{}")
    (rig.out / "day-a").mkdir(exist_ok=True)
    old = {"versions": {"gomoveshift": "0.3.25"}, "manifest_sha": "m1", "config_sha": "c1", "phase": "x"}  # fmt: skip
    incidents = rig.tmp / "incidents"
    row = new_incident("drive_failed", "drive_failed:day-a:r1", "day-a.json", "2026-11-07T10:00:00Z",
                       drive="day-a", phase="x", error="BLUR GATE", resume_recipe=None, resume_attempts=[])  # fmt: skip
    row["failure_fingerprint"] = {"hash": retry_policy.fingerprint_hash(old), "fingerprint": old, "class": "deterministic"}  # fmt: skip
    write_row(incidents, row)
    rig.relaunch = launch.FailedRelaunch(
        incidents, rig.stages, fingerprint_fn=lambda project, phase: {**old, "manifest_sha": "m2"},
        now=NOW,
    )  # fmt: skip
    res = rig.run()
    assert [f["project"] for f in res["failed_publishes"]] == ["day-a"]
    assert (res["action"], res["project"]) == ("launched", "day-a"), res
    assert "relaunch" in res and len(rig.launches) == 1


def test_a_slot_wait_or_noop_publish_still_returns_and_launches_nothing(tmp_path):
    for result in (
        {"action": "refused", "reason": "no claim: outside_lead_window (next slot x)"},
        {"action": "noop", "reason": "nothing to do"},
    ):
        rig = Rig(tmp_path / result["action"])
        _failed_publish(rig, **result)
        res = rig.run()
        assert rig.publishes == [("day-a", False)] and not rig.launches, res
        assert "failed_publishes" not in res


def test_a_backoff_wait_is_not_a_refusal(tmp_path):
    rig = Rig(tmp_path)
    _unfinished_publish(rig, tmp_path, age_s=60)
    res = rig.run()
    assert rig.publishes == [] and "failed_publishes" not in res
    assert res["publish_backoff"][0]["project"] == "day-a"
    # a replace backoff never held the region: the next fresh row launches in the same tick
    assert res["action"] == "launched" and res["project"] == "day-b", res


def test_a_failed_publish_retry_dry_run_asks_for_the_dry_run(tmp_path):
    rig = Rig(tmp_path)
    _failed_publish(rig, action="would_publish", reason="day-a (dry run: nothing changed)")
    assert rig.run(dry_run=True)["action"] == "would_publish"
    assert rig.publishes == [("day-a", True)]


def test_a_failed_drive_with_no_failed_publish_is_not_published(tmp_path):
    # set ``failed`` by hand, or its publish job is still running: never re-published
    for state in (None, "running"):
        rig = Rig(tmp_path / str(state))
        rig.drives.append(drive("day-a", "failed"))
        job(rig, "ok", "day-a", "succeeded", 100.0, exit_code=0)
        if state:
            job(rig, "pub", "day-a", state, 200.0, name="gms-publish-day-a")
        rig.publish_result = {"action": "launched", "reason": "x"}
        rig.run()
        assert rig.publishes == []


def test_a_failed_publish_waits_for_a_running_drive(tmp_path):
    rig = Rig(tmp_path)
    _failed_publish(rig)
    rig.running = True
    res = rig.run()
    assert rig.publishes == [] and "running" in res["reason"]


# 2026-11-20 root cause (Kerry-to-Cork): the retry keyed on the publish job's ``state`` being
# ``failed`` or its exit code being non-zero. A publish job the runner ended any other way
# (cancelled, stalled, killed with no exit code) is as dead as a failed one, but matched neither,
# so the failed row with a passed audit and no upload receipt was never published again.


@pytest.mark.parametrize("state, code", [("cancelled", None), ("stalled", None), ("killed", None), ("failed", None)])  # fmt: skip
def test_a_publish_job_that_ended_without_an_exit_code_is_published_again(tmp_path, state, code):
    rig = Rig(tmp_path)
    rig.drives.append(drive("day-a", "failed"))
    job(rig, "ok", "day-a", "succeeded", 100.0, exit_code=0)
    job(rig, "pub", "day-a", state, 200.0, exit_code=code, name="gms-publish-day-a")
    rig.publish_result = {"action": "launched", "reason": "day-a for slot"}
    res = rig.run()
    assert rig.publishes == [("day-a", False)], res
    assert not rig.launches and not rig.registered  # the PUBLISH retries; never a render


def test_a_failed_publish_row_with_an_upload_receipt_is_never_published_again(tmp_path):
    rig = Rig(tmp_path)
    out = tmp_path / "out" / "day-a"
    out.mkdir(parents=True)
    (out / "youtube_uploads.json").write_text(json.dumps({"1": {"video_id": "V1"}}))
    rig.drives.append(drive("day-a", "failed", output_dir=str(out)))
    job(rig, "ok", "day-a", "succeeded", 100.0, exit_code=0)
    job(rig, "pub", "day-a", "cancelled", 200.0, exit_code=None, name="gms-publish-day-a")
    rig.publish_result = {"action": "launched", "reason": "x"}
    rig.run()
    assert rig.publishes == []


def test_publish_job_prefix_matches_the_publisher():
    from gomoveshift.health import publish

    assert launch.PUBLISH_JOB_PREFIX == publish.JOB_PREFIX


def test_a_slot_outside_the_lead_window_waits_and_is_not_a_failure(tmp_path):
    rig = Rig(tmp_path)
    _audited(rig, action="refused",
             reason="no claim: outside_lead_window (next slot 2026-11-15T14:00:00-05:00)")  # fmt: skip
    res = rig.run()
    assert res["action"] == "noop", res
    assert "audit passed: day-a waits for its Waypost slot" in res["reason"]
    assert "outside_lead_window" in res["reason"] and "2026-11-15T14:00" in res["reason"]
    assert launch.exit_code(res) == 0 and not rig.launches


def test_any_other_publish_refusal_is_a_refusal_naming_it(tmp_path):
    rig = Rig(tmp_path)
    rig.rows[:] = [r for r in rig.rows if r.project == "day-a"]  # no fresh row behind it
    _audited(rig, action="refused", reason="day-a's checkpoint is stale under the current code (B re-run)")  # fmt: skip
    res = rig.run()
    # a refused publish no longer holds the region: reported, and nothing launches
    assert res["failed_publishes"][0]["project"] == "day-a"
    assert "checkpoint is stale" in res["failed_publishes"][0]["reason"]
    assert not rig.launches and not rig.registered


def test_a_dry_run_asks_the_publisher_for_its_dry_run(tmp_path):
    rig = Rig(tmp_path)
    _audited(rig, action="would_publish", reason="day-a (dry run: nothing changed)")
    res = rig.run(dry_run=True)
    assert rig.publishes == [("day-a", True)]
    assert res["action"] == "would_publish" and "day-a (dry run" in res["reason"]
    assert launch.exit_code(res) == 0


@pytest.mark.parametrize("state", ["running", None])
def test_an_audit_still_running_or_with_no_record_is_never_published(tmp_path, state):
    rig = Rig(tmp_path)
    rig.drives.append(drive("day-a", "auditing"))
    if state:
        job(rig, "now", "day-a", state, 100.0)
    rig.publish_result = {"action": "launched", "reason": "x"}
    res = rig.run()
    assert rig.publishes == [] and res["action"] == "noop" and "audit in progress" in res["reason"]


def test_a_failed_audit_or_one_waiting_at_the_ost_gate_is_never_published(tmp_path):
    rig = Rig(tmp_path)
    rig.drives.append(drive("day-a", "auditing"))
    job(rig, "bad", "day-a", "failed", 100.0)
    rig.publish_result = {"action": "launched", "reason": "x"}
    assert rig.run()["action"] == "launched" and rig.publishes == []  # day-b launches instead
    job(rig, "gate", "day-a", "failed", 200.0, exit_code=launch.drive_jobs.EXIT_WAITING_AT_GATE)
    (rig.tmp / "jobs" / "gate" / "output.log").write_text("DRIVE WAITING: OST audit\n")
    rig.run()
    assert rig.publishes == []


def test_a_running_drive_holds_the_publish_too(tmp_path):
    rig = Rig(tmp_path)
    _audited(rig, action="launched", reason="x")
    rig.running = True
    res = rig.run()
    assert rig.publishes == [] and "running" in res["reason"]


# 2026-11-17: ireland-2025-09-19-old-head-courtmacsherry failed on an Overpass outage and every
# Ireland row behind it was refused, hourly. A failed audit no longer holds its region.


def test_a_failed_audit_is_listed_and_the_next_row_launches(tmp_path):
    rig = Rig(tmp_path)
    rig.drives.append(drive("day-a", "auditing"))
    job(rig, "fbc9336e", "day-a", "failed", 100.0)
    res = rig.run()
    assert (res["action"], res["project"]) == ("launched", "day-b"), res
    (bad,) = res["failed_audits"]
    assert bad["project"] == "day-a" and "fbc9336e" in bad["reason"] and "exit 1" in bad["reason"]
    assert "health heal" in bad["reason"]
    assert len(rig.launches) == 1 and rig.registered[0]["project"] == "day-b"
    assert rig.retired == [] and launch.exit_code(res) == 0


def test_a_failed_audit_then_one_in_progress_waits_on_the_running_one(tmp_path):
    rig = Rig(tmp_path)
    rig.rows.append(rig.row("day-c", 3))
    rig.drives += [drive("day-a", "auditing"), drive("day-b", "auditing")]
    job(rig, "bad", "day-a", "failed", 100.0)
    job(rig, "now", "day-b", "running", 200.0)
    res = rig.run()
    assert res["action"] == "noop" and "audit in progress: day-b" in res["reason"], res
    assert [f["project"] for f in res["failed_audits"]] == ["day-a"] and not rig.launches


def test_a_failed_audit_then_a_passed_one_publishes_the_passed_one(tmp_path):
    rig = Rig(tmp_path)
    rig.drives += [drive("day-a", "auditing"), drive("day-b", "auditing")]
    job(rig, "bad", "day-a", "failed", 100.0)
    job(rig, "ok", "day-b", "succeeded", 100.0, exit_code=0)
    rig.publish_result = {"action": "launched", "reason": "day-b for slot"}
    res = rig.run()
    assert rig.publishes == [("day-b", False)] and res["action"] == "launched", res
    assert [f["project"] for f in res["failed_audits"]] == ["day-a"]
    assert launch.exit_code(res) == 0


def test_a_failed_audit_whose_newer_audit_succeeded_is_published(tmp_path):
    rig = Rig(tmp_path)
    _audited(rig, action="launched", reason="day-a for slot")
    job(rig, "old", "day-a", "failed", 50.0)
    res = rig.run()
    assert rig.publishes == [("day-a", False)] and "failed_audits" not in res


def test_every_auditing_row_failed_and_no_row_left_is_a_noop_listing_them(tmp_path):
    rig = Rig(tmp_path)
    rig.drives += [drive("day-a", "auditing"), drive("day-b", "auditing")]
    job(rig, "a", "day-a", "failed", 100.0)
    job(rig, "b", "day-b", "failed", 100.0)
    res = rig.run()
    assert res["action"] == "noop" and "no eligible backlog row" in res["reason"], res
    assert [f["project"] for f in res["failed_audits"]] == ["day-a", "day-b"]
    assert launch.exit_code(res) == 0 and not rig.launches


def test_a_relaunched_audit_after_a_failure_is_still_a_noop(tmp_path):
    rig = Rig(tmp_path)
    rig.drives.append(drive("day-a", "auditing"))
    job(rig, "old", "day-a", "failed", 100.0)
    job(rig, "new", "day-a", "running", 200.0)
    res = rig.run()
    assert res["action"] == "noop" and "audit in progress" in res["reason"]


def test_an_audit_with_no_job_record_or_a_succeeded_job_stays_a_noop(tmp_path):
    rig = Rig(tmp_path)
    rig.drives.append(drive("day-a", "auditing"))
    assert rig.run()["action"] == "noop"
    job(rig, "ok", "day-a", "succeeded", 100.0, exit_code=0)
    job(rig, "other", "day-b", "failed", 300.0)
    assert rig.run()["action"] == "noop"


# 2026-11-07: nctopa-2026-03-04's audit failed for good (under the floor once GPS-less spans were
# cut), so its row was dropped from the backlog. The store row stayed ``auditing``, every
# ``health backlog --region usa`` refused on it, and ``--supersede`` refused "no usa row ... in
# the backlog": no tracked way out but a hand ``set-status``.


def test_a_failed_audit_with_no_backlog_row_is_retired_and_the_next_row_launches(tmp_path):
    rig = Rig(tmp_path)
    rig.drives.append(drive("dropped", "auditing"))
    job(rig, "9ee99a65", "dropped", "failed", 100.0)
    res = rig.run()
    assert rig.retired == ["dropped"]
    assert res["retired"] == [{"project": "dropped", "reason": res["retired"][0]["reason"]}]
    assert "9ee99a65" in res["retired"][0]["reason"]
    assert res["action"] == "launched" and res["project"] == "day-a"


def test_a_dry_run_names_the_retirement_but_writes_nothing(tmp_path):
    rig = Rig(tmp_path)
    rig.drives.append(drive("dropped", "auditing"))
    job(rig, "9ee99a65", "dropped", "failed", 100.0)
    res = rig.run(dry_run=True)
    assert rig.retired == [] and [r["project"] for r in res["retired"]] == ["dropped"]
    assert res["project"] == "day-a" and not rig.launches


# A retired drive's recipe stayed live: ``health heal`` re-resumed the dead 03-04 audit on every
# new mirror SHA (jobs ba3a9dbf, 15c1c9b0, 1affffde), each holding the heavy slot ~35 min.
def test_retiring_holds_the_drives_recipe_so_heal_never_resumes_it(tmp_path):
    rig = Rig(tmp_path)
    rig.drives.append(drive("dropped", "auditing"))
    job(rig, "9ee99a65", "dropped", "failed", 100.0)
    stages = rig.stages / "dropped" / "dropped" / "_stages"
    stages.mkdir(parents=True)
    (stages / "resume.json").write_text(json.dumps({"argv": ["drive", "/x"]}))
    (rig.stages / "dropped" / "other" / "_stages").mkdir(parents=True)  # no recipe: left alone
    assert rig.run(dry_run=True)["retired"] and not (stages / launch.HOLD_NAME).exists()
    rig.run()
    assert "9ee99a65" in (stages / launch.HOLD_NAME).read_text()
    assert heal.resume.is_held(stages / "resume.json")
    assert not (rig.stages / "dropped" / "other" / "_stages" / launch.HOLD_NAME).exists()


# 11-08: heal launched the dead 03-04 resume (jobs 8e6e2e9f, 12fbd6e1) two seconds before each
# hourly backlog, so the backlog always stopped at "a drive is running" and never held it; the
# resume failed in ~11 s and heal relaunched it on the next mirror SHA, starving both regions.
def test_retiring_and_holding_run_even_while_a_drive_is_running(tmp_path):
    rig = Rig(tmp_path)
    rig.running = True
    rig.drives.append(drive("dropped", "auditing"))
    rig.drives.append(drive("gone", "failed"))
    job(rig, "9ee99a65", "dropped", "failed", 100.0)
    held = []
    for project in ("dropped", "gone"):
        stages = rig.stages / project / project / "_stages"
        stages.mkdir(parents=True)
        (stages / "resume.json").write_text("{}")
        held.append(stages)
    res = rig.run()
    assert res["reason"] == "a drive is running; one heavy drive at a time"
    assert rig.retired == ["dropped"] and not rig.launches
    assert res["held"] == [str(s) for s in held]
    assert all(heal.resume.is_held(s / "resume.json") for s in held)


def test_a_rowless_drive_already_failed_in_the_store_gets_the_hold_too(tmp_path):
    # retired before holds existed (or set ``failed`` by hand): no later run retires it again
    rig = Rig(tmp_path)
    rig.drives.append(drive("dropped", "failed"))
    stages = rig.stages / "dropped" / "dropped" / "_stages"
    stages.mkdir(parents=True)
    (stages / "resume.json").write_text("{}")
    assert rig.run(dry_run=True)["held"] == [str(stages)]
    assert not (stages / launch.HOLD_NAME).exists()
    res = rig.run()
    assert res["held"] == [str(stages)] and res["action"] == "launched" and rig.retired == []
    assert "failed" in (stages / launch.HOLD_NAME).read_text()
    assert "held" not in rig.run()  # an existing hold is left as it is


def test_a_failed_drive_whose_row_is_still_listed_gets_no_hold(tmp_path):
    rig = Rig(tmp_path)
    rig.drives.append(drive("day-b", "failed"))
    stages = rig.stages / "day-b" / "day-b" / "_stages"
    stages.mkdir(parents=True)
    (stages / "resume.json").write_text("{}")
    assert "held" not in rig.run() and not (stages / launch.HOLD_NAME).exists()


def test_retiring_a_drive_with_no_stages_still_retires_it(tmp_path):
    rig = Rig(tmp_path)
    rig.drives.append(drive("dropped", "auditing"))
    job(rig, "9ee99a65", "dropped", "failed", 100.0)
    assert rig.run()["action"] == "launched" and rig.retired == ["dropped"]


def test_a_failed_audit_whose_row_is_still_listed_is_never_retired(tmp_path):
    rig = Rig(tmp_path)
    rig.drives.append(drive("day-a", "auditing"))
    job(rig, "fbc9336e", "day-a", "failed", 100.0)
    res = rig.run()
    assert res["action"] == "launched" and res["failed_audits"] and rig.retired == []


def test_a_rowless_audit_that_is_running_or_unrecorded_is_never_retired(tmp_path):
    rig = Rig(tmp_path)
    rig.drives.append(drive("dropped", "auditing"))
    assert rig.run()["action"] == "noop" and rig.retired == []
    job(rig, "old", "dropped", "failed", 100.0)
    job(rig, "new", "dropped", "running", 200.0)
    res = rig.run()
    assert "audit in progress" in res["reason"] and rig.retired == []


def test_a_rowless_failed_audit_in_another_region_is_left_alone(tmp_path):
    rig = Rig(tmp_path)
    rig.drives.append(drive("ie-gone", "auditing", region="ireland"))
    job(rig, "x", "ie-gone", "failed", 100.0)
    assert rig.run()["action"] == "launched" and rig.retired == []


def test_auditing_in_another_region_does_not_block(tmp_path):
    rig = Rig(tmp_path)
    rig.drives.append(drive("ie-x", "auditing", region="ireland"))
    assert rig.run()["action"] == "launched"


def test_ready_drives_cover_slots_in_order(tmp_path):
    rig = Rig(tmp_path)
    rig.drives.append(drive("old", "ready"))
    res = rig.run()
    assert (res["action"], res["lf_day"], res["slot"]) == ("launched", "tue", TUE.isoformat())
    rig2 = Rig(tmp_path / "two")
    rig2.drives += [drive("old", "ready"), drive("old2", "ready")]
    res2 = rig2.run()
    assert res2["action"] == "noop" and "covered" in res2["reason"]


def test_an_uploaded_auditing_drive_does_not_hold_the_slot(tmp_path):
    # 2026-11-02: sc-2026-02-21-ace-basin stayed ``auditing`` after Phase D uploaded both
    # parts, and every backlog call answered "audit in progress" for the open usa slot.
    rig = Rig(tmp_path)
    out = uploaded(tmp_path, "ace", {"1": {"video_id": "ZPaIwzcE_kU"}})
    rig.drives.append(drive("done", "auditing", output_dir=out))
    res = rig.run()
    assert (res["action"], res["lf_day"]) == ("launched", "fri"), res
    assert res["published"] == ["done"]


def test_an_uploaded_ready_drive_does_not_cover_a_slot(tmp_path):
    # The receipt may sit one level down, under the render's project dir (F1 output ``src``).
    rig = Rig(tmp_path)
    out = uploaded(tmp_path, "f1", {"1": {"video_id": "XIT4Fa9aUUU"}}, sub="src")
    rig.drives.append(drive("done", "ready", output_dir=out))
    res = rig.run()
    assert (res["action"], res["lf_day"]) == ("launched", "fri"), res


@pytest.mark.parametrize("records", [{}, "{not json", {"1": {"video_id": None}}])
def test_an_empty_or_unreadable_receipt_still_holds_the_slot(tmp_path, records):
    rig = Rig(tmp_path)
    rig.drives.append(drive("day-x", "auditing", output_dir=uploaded(tmp_path, "x", records)))
    res = rig.run()
    assert res["action"] == "noop" and "audit in progress: day-x" in res["reason"]


def test_an_uploaded_drive_still_skips_its_own_backlog_row(tmp_path):
    rig = Rig(tmp_path)
    out = uploaded(tmp_path, "a", {"1": {"video_id": "v1"}})
    rig.drives.append(drive("day-a", "auditing", output_dir=out))
    res = rig.run()
    assert res["project"] == "day-b"
    assert res["skipped"][0] == {"project": "day-a", "reason": "in the schedule store (auditing)"}


def test_slots_past_the_horizon_or_before_now_do_not_count(tmp_path):
    rig = Rig(tmp_path)
    rig.slots = [(NOW - timedelta(hours=1), "tue"), (NOW + timedelta(days=8), "wed")]
    res = rig.run()
    assert res["action"] == "noop" and "covered" in res["reason"]


def test_skips_store_ledger_staged_and_held_rows_with_reasons(tmp_path):
    rig = Rig(tmp_path)
    rig.rows = [rig.row(n, 1) for n in ("in-store", "in-ledger", "staged", "held", "free")]
    rig.drives.append(drive("in-store", "complete"))
    rig.ledger = {"in-ledger"}
    rig.stage("prior-run", rig.rows[2].sources)
    held = rig.stages / "held" / "held" / "_stages"
    held.mkdir(parents=True)
    (held / launch.HOLD_NAME).write_text("")
    res = rig.run()
    assert res["project"] == "free", res
    reasons = {s["project"]: s["reason"] for s in res["skipped"]}
    assert "complete" in reasons["in-store"]
    assert "ledger" in reasons["in-ledger"]
    assert "prior-run" in reasons["staged"] and "resume.json" in reasons["staged"]
    assert "hold" in reasons["held"]


def test_a_hold_on_a_prior_stage_root_wins_over_staged(tmp_path):
    rig = Rig(tmp_path)
    rig.stage("prior-run", rig.rows[0].sources, hold=True)
    res = rig.run()
    assert res["project"] == "day-b"
    assert "hold" in res["skipped"][0]["reason"]


def test_a_partially_staged_row_is_still_skipped(tmp_path):
    rig = Rig(tmp_path)
    rig.stage("prior-run", rig.rows[0].sources[:1])
    assert rig.run()["project"] == "day-b"


def test_a_missing_source_skips_the_row(tmp_path):
    rig = Rig(tmp_path)
    Path(rig.rows[0].sources[0]).unlink()
    res = rig.run()
    assert res["project"] == "day-b"
    assert "missing" in res["skipped"][0]["reason"]


def _gpx_row(rig, gpx):
    row = rig.rows[0]
    return BacklogRow(
        row.project, row.region, row.sources, "generic", row.lf_mode,
        day_gpx=str(gpx), camera_utc_offset_hours=-4.0,
    )  # fmt: skip


def test_a_day_gpx_row_launches_with_the_track_and_camera_offset(tmp_path):
    rig = Rig(tmp_path)
    gpx = tmp_path / "day12.gpx"
    gpx.write_text("<gpx/>")
    rig.rows[0] = _gpx_row(rig, gpx)
    res = rig.run()
    assert res["action"] == "launched", res
    drive_args = rig.launches[0][0][rig.launches[0][0].index("--") + 1 :]
    assert drive_args[drive_args.index("--day-gpx") + 1] == str(gpx)
    assert drive_args[drive_args.index("--camera-utc-offset-hours") + 1] == "-4.0"
    assert drive_args[drive_args.index("--ingest-mode") + 1] == "generic"


def test_a_row_without_a_day_gpx_passes_neither_flag(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    drive_args = rig.launches[0][0]
    assert "--day-gpx" not in drive_args and "--camera-utc-offset-hours" not in drive_args


def _placeless_row(rig, label):
    row = rig.rows[0]
    return BacklogRow(
        row.project, row.region, row.sources, "generic", row.lf_mode,
        placeless_label=label,
    )  # fmt: skip


def test_a_claim_free_placeless_row_is_skipped_pointing_at_bracket_gpx(tmp_path):
    rig = Rig(tmp_path)
    rig.rows[0] = _placeless_row(rig, "Somewhere On The Road")
    res = rig.run()
    assert (res["action"], res["project"]) == ("launched", "day-b"), res
    (skip,) = res["skipped"]
    assert skip["project"] == "day-a"
    assert "bracket_gpx" in skip["reason"] and "last resort" in skip["reason"]
    assert "--placeless-label" not in rig.launches[0][0]


# 2026-11-08: nctopa-2026-03-05-rec0063 rendered "Through The Valleys And Ridges" over a drive that
# is mostly divided interstate and farmland. A placeless label is operator-typed and nothing in the
# pipeline measures it, so the launcher passes only a claim-free one; a descriptive label is skipped
# at the backlog, before any render, and the row behind it still launches.
def test_a_descriptive_placeless_label_is_skipped_before_any_render(tmp_path):
    rig = Rig(tmp_path)
    rig.rows[0] = _placeless_row(rig, "Through The Valleys And Ridges")
    res = rig.run()
    assert (res["action"], res["project"]) == ("launched", "day-b"), res
    assert res["skipped"] == [
        {
            "project": "day-a",
            "reason": launch.placeless_label_skip_reason("Through The Valleys And Ridges"),
        }
    ]
    assert rig.gates == [(rig.sources / "day-b", "day-b")]  # never reached the render gate
    assert "--placeless-label" not in rig.launches[0][0]


def test_a_placeless_label_outside_the_vocabulary_is_skipped_at_the_backlog_not_at_launch(tmp_path):
    rig = Rig(tmp_path)
    rig.rows[0] = _placeless_row(rig, "Through Roanoke")
    res = rig.run()
    assert (res["action"], res["project"]) == ("launched", "day-b"), res
    reason = res["skipped"][0]["reason"]
    assert reason.startswith("placeless_label 'Through Roanoke': ")
    assert "'roanoke' are outside" in reason and "bracket_gpx" in reason


@pytest.mark.parametrize(
    "label, expected",
    [
        ("Out Through Open Country", "claims Open, Country"),
        ("Somewhere In The Hills", "claims Hills"),
        ("Back Roads In The Morning Fog", "claims Back, Morning, Fog"),
        ("Over The Ridges", "'over' are outside the placeless vocabulary"),
    ],
)
def test_placeless_label_skip_reason(label, expected):
    reason = launch.placeless_label_skip_reason(label)
    assert reason is not None and expected in reason, reason


def test_a_row_without_a_placeless_label_passes_no_label(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    assert "--placeless-label" not in rig.launches[0][0]


def _labeled_row(rig, per_lf, location=""):
    row = rig.rows[0]
    return BacklogRow(
        row.project, row.region, row.sources, "generic", row.lf_mode,
        per_lf_locations=tuple(per_lf), location=location,
    )  # fmt: skip


# nctopa-2026-03-05-rec0063 (GPS-less, no track): its labels are the operator's, typed once into
# the backlog row, and the launcher hands them to the drive. Nothing is typed at launch time.
def test_a_row_with_per_lf_locations_launches_with_each_label_and_the_drive_wide_one(tmp_path):
    rig = Rig(tmp_path)
    rig.rows[0] = _labeled_row(
        rig,
        [(1, "Example Town to Beta Pass, Example County, Exampleland"),
         (2, "Beta Pass to Gamma Falls, Example County, Exampleland")],
        location="Example County, Exampleland",
    )  # fmt: skip
    res = rig.run()
    assert res["action"] == "launched", res
    drive_args = rig.launches[0][0][rig.launches[0][0].index("--") + 1 :]
    per_lf = [drive_args[i + 1] for i, a in enumerate(drive_args) if a == "--per-lf-location"]
    assert per_lf == [
        "1=Example Town to Beta Pass, Example County, Exampleland",
        "2=Beta Pass to Gamma Falls, Example County, Exampleland",
    ]
    assert drive_args[drive_args.index("--location") + 1] == "Example County, Exampleland"
    assert "--placeless-label" not in drive_args and "--day-gpx" not in drive_args


def test_a_row_with_per_lf_locations_and_no_drive_wide_label_passes_only_the_per_lf_flags(tmp_path):
    rig = Rig(tmp_path)
    rig.rows[0] = _labeled_row(rig, [(1, "Example Town, Example County")])
    res = rig.run()
    assert res["action"] == "launched", res
    drive_args = rig.launches[0][0]
    assert "--per-lf-location" in drive_args and "--location" not in drive_args


def test_a_row_without_location_fields_passes_neither_flag(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    drive_args = rig.launches[0][0]
    assert "--per-lf-location" not in drive_args and "--location" not in drive_args


# The drive's own label rules (``cli.parser._validate_per_lf_location_args``) judge the row
# before any render: a refused row is skipped at the backlog with the parser's reason, never
# failed hours later at the launch job's argparse, and the row behind it still launches.
@pytest.mark.parametrize(
    "per_lf, location, expected",
    [
        ([(1, "To Beta Pass, Example County")], "", "starts with a bare preposition"),
        ([(1, "Beta Pass, Example County")], "Exampleland", "must have >= 2 components"),
        ([(1, "Beta Pass, Example County")], "Beta Pass, Example County", "equals --location"),
        (
            [(1, "Beta Pass, Example County, Exampleland")],
            "Other County, Exampleland",
            "is not a comma-suffix",
        ),
    ],
)
def test_a_row_the_drive_would_refuse_is_skipped_before_any_render(
    tmp_path, per_lf, location, expected
):
    rig = Rig(tmp_path)
    rig.rows[0] = _labeled_row(rig, per_lf, location=location)
    res = rig.run()
    assert (res["action"], res["project"]) == ("launched", "day-b"), res
    (skip,) = res["skipped"]
    assert skip["project"] == "day-a" and skip["reason"].startswith("location labels: ")
    assert expected in skip["reason"], skip["reason"]
    assert rig.gates == [(rig.sources / "day-b", "day-b")]  # never reached the render gate
    assert "--per-lf-location" not in rig.launches[0][0]


def test_location_labels_skip_reason_is_none_for_a_row_the_drive_accepts():
    row = BacklogRow(
        "p", "usa", ("/v.mp4",), "generic", "full_drive",
        per_lf_locations=((1, "Beta Pass, Example County, Exampleland"),),
        location="Example County, Exampleland",
    )  # fmt: skip
    assert launch.location_labels_skip_reason(row) is None


def test_a_missing_day_gpx_skips_the_row(tmp_path):
    rig = Rig(tmp_path)
    rig.rows[0] = _gpx_row(rig, tmp_path / "gone.gpx")
    res = rig.run()
    assert res["project"] == "day-b"
    assert res["skipped"][0]["reason"] == f"missing day_gpx {tmp_path / 'gone.gpx'}"


def test_backlog_exhausted_is_a_noop_naming_every_skip(tmp_path):
    rig = Rig(tmp_path)
    rig.ledger = {"day-a", "day-b"}
    res = rig.run()
    assert res["action"] == "noop" and "no eligible" in res["reason"]
    assert [s["project"] for s in res["skipped"]] == ["day-a", "day-b"]


def test_other_region_rows_are_ignored(tmp_path):
    rig = Rig(tmp_path)
    rig.rows = [rig.row("ie-1", 1, region="ireland"), rig.row("us-1", 1)]
    assert rig.run()["project"] == "us-1"


@pytest.mark.parametrize("action", ["launched", "refused"])
def test_staging_boundary_free_equal_to_need_passes(tmp_path, action):
    rig = Rig(tmp_path)
    rig.rows = [rig.row("day-a", 10)]
    need = launch.staging_need_gb(rig.rows[0].sources)
    assert need == pytest.approx(30.0)  # 10 GB x 3.0 = 30, the 30 GB floor
    rig.devices[rig.stages] = (1, need if action == "launched" else need - 0.01)
    res = rig.run()
    assert res["action"] == action, res
    if action == "refused":
        assert "storage" in res["reason"] and not rig.launches and not rig.sources.exists()


def test_staging_need_scales_with_day_source_above_the_floor(tmp_path):
    rig = Rig(tmp_path)
    row = rig.row("big", 40)
    assert launch.staging_need_gb(row.sources) == pytest.approx(120.0)


def test_a_drive_the_master_encode_cannot_fit_is_refused_upfront(tmp_path):
    rig = Rig(tmp_path)
    rig.rows = [rig.row("rec0063", 138.4)]  # 2026-11-10: launched on 333.8 GB, failed at encode
    rig.devices[rig.stages] = (1, 333.8)
    res = rig.run()
    assert res["action"] == "refused" and "storage" in res["reason"], res
    assert not rig.launches


def test_shared_device_sums_staging_and_output_needs(tmp_path):
    rig = Rig(tmp_path)
    rig.rows = [rig.row("day-a", 40)]  # staging 120 + output floor 50 = 170 on one device
    rig.devices = {rig.stages: (7, 169.0), rig.out: (7, 169.0)}
    res = rig.run()
    assert res["action"] == "refused" and "170.0" in res["reason"]
    storage = res["storage"]
    assert len(storage) == 1 and storage[0]["need_gb"] == pytest.approx(170.0)
    rig.devices = {rig.stages: (7, 170.0), rig.out: (7, 170.0)}
    assert rig.run()["action"] == "launched"


def test_distinct_devices_are_checked_separately(tmp_path):
    rig = Rig(tmp_path)
    rig.rows = [rig.row("day-a", 40)]
    rig.devices = {rig.stages: (1, 120.0), rig.out: (2, 49.0)}
    res = rig.run()
    assert res["action"] == "refused"
    by_dev = {s["device"]: s for s in res["storage"]}
    assert by_dev[1]["ok"] and not by_dev[2]["ok"]


def test_storage_is_reported_on_dry_run_with_the_bound_named(tmp_path):
    res = Rig(tmp_path).run(dry_run=True)
    assert res["storage"] and "x 3.0" in res["storage_bound"]


def test_preflight_blocked_refuses_without_launching(tmp_path):
    rig = Rig(tmp_path)
    rig.preflight_error = PreflightBlockedError([("staging disk", "need ~200 GB")])
    res = rig.run()
    assert res["action"] == "refused" and "need ~200 GB" in res["reason"]
    assert not rig.launches and not rig.registered and rig.leases[-1] == "release"


def test_launch_failure_refuses_and_registers_nothing(tmp_path):
    rig = Rig(tmp_path)
    rig.launch_ok = False
    res = rig.run()
    assert res["action"] == "refused" and "runner: boom" in res["reason"]
    assert not rig.registered and rig.leases[-1] == "release"


def test_an_existing_subset_dir_is_reused_when_it_matches(tmp_path):
    rig = Rig(tmp_path)
    assert rig.run()["action"] == "launched"
    rig2 = Rig(tmp_path)  # fresh rig, same paths: the dir from the first launch is reused
    rig2.rows = rig.rows
    assert rig2.run()["action"] == "launched"


def test_a_mismatched_subset_dir_is_refused_never_rewritten(tmp_path):
    rig = Rig(tmp_path)
    subset = rig.sources / "day-a"
    subset.mkdir(parents=True)
    (subset / "stray.MP4").write_text("x")
    res = rig.run()
    assert res["action"] == "refused" and "stray.MP4" in res["reason"]
    assert (subset / "stray.MP4").exists() and not rig.launches


def test_the_day_gpx_slicers_own_sidecars_do_not_mismatch_the_subset_dir(tmp_path):
    # A drive with --day-gpx writes <stem>.gpx beside each video in the subset dir before
    # ingest; a relaunch of that row (e.g. after the drive refused) must still match.
    rig = Rig(tmp_path)
    assert rig.run()["action"] == "launched"
    subset = rig.sources / "day-a"
    for video in [p for p in subset.iterdir() if p.suffix == ".MP4"]:
        video.with_suffix(".gpx").write_text(
            '<?xml version="1.0"?>\n<gpx creator="gms-day-gpx-slicer">\n</gpx>\n'
        )
    rig2 = Rig(tmp_path)
    rig2.rows = rig.rows
    assert rig2.run()["action"] == "launched"


def test_a_sidecar_the_slicer_did_not_write_still_mismatches(tmp_path):
    rig = Rig(tmp_path)
    assert rig.run()["action"] == "launched"
    subset = rig.sources / "day-a"
    (subset / "day-a-GX00.gpx").write_text('<gpx creator="someone-else"></gpx>')
    (subset / "day-z-GX00.gpx").write_text('<gpx creator="gms-day-gpx-slicer"></gpx>')
    rig2 = Rig(tmp_path)
    rig2.rows = rig.rows
    res = rig2.run()
    assert res["action"] == "refused"
    assert "day-a-GX00.gpx" in res["reason"] and "day-z-GX00.gpx" in res["reason"]


def test_no_service_interpreter_refuses(tmp_path):
    rig = Rig(tmp_path)
    (tmp_path / ".venv" / "bin" / "python").unlink()
    res = rig.run()
    assert res["action"] == "refused" and "interpreter" in res["reason"]


def test_an_exception_mid_run_still_releases_the_lease(tmp_path):
    rig = Rig(tmp_path)

    def boom():
        raise OSError("probe failed")

    with pytest.raises(OSError):
        launch.run_backlog(
            region="usa", rows=rig.rows, drives=[], free_slots=rig.slots, known_projects=set(),
            stages_root=rig.stages, output_root=rig.out, sources_root=rig.sources,
            code_root=rig.tmp, jobs_dir=rig.tmp, mirror_sha="x", now=NOW, dry_run=False,
            lease_acquire=rig.lease_acquire, lease_release=lambda: rig.leases.append("release"),
            drive_running=boom, disk=rig.disk, preflight=rig.preflight, render_gate=rig.render_gate, launcher=rig.launcher,
            register=lambda **kw: None, retire=lambda project: None,
        )  # fmt: skip
    assert rig.leases == ["acquire", "release"]


def test_exit_code_maps_refused_to_2():
    assert launch.exit_code({"action": "refused"}) == 2
    for action in ("launched", "would_launch", "noop"):
        assert launch.exit_code({"action": action}) == 0


def test_preflight_runs_under_the_drive_interpreter(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    assert rig.preflights[0][4] == rig.tmp / ".venv" / "bin" / "python"


def test_default_preflight_reports_the_child_blockers(tmp_path):
    fake = tmp_path / "python"
    fake.write_text(
        '#!/bin/sh\necho \'[["GoPro telemetry parser", "gopro-overlay missing"]]\'\nexit 3\n'
    )
    fake.chmod(0o755)
    with pytest.raises(PreflightBlockedError, match="gopro-overlay missing"):
        launch.default_preflight(
            tmp_path / "src", tmp_path / "out", staging_dir=tmp_path / "st",
            ingest_mode="gopro", python=fake,
        )  # fmt: skip


def test_default_preflight_crash_is_not_a_pass(tmp_path):
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\necho boom >&2\nexit 1\n")
    fake.chmod(0o755)
    with pytest.raises(RuntimeError, match="boom"):
        launch.default_preflight(
            tmp_path / "src", tmp_path / "out", staging_dir=tmp_path / "st",
            ingest_mode="gopro", python=fake,
        )  # fmt: skip


def test_report_slots_take_the_day_from_waypost_not_local_weekday_math():
    """The LF day passed to ``--lf-day`` is the schedule report's own ``day``, so slot
    decisions stay in seo_manager.scheduling (tests/test_scheduling_single_source.py)."""
    report = {"next_available": [
        {"region": "usa", "day": "fri", "slot": "2026-10-23T15:00:00-04:00"},
        {"region": "ireland", "day": "sun", "slot": "2026-10-25T14:00:00+00:00"},
        {"region": "usa", "day": "tue", "slot": None},
    ]}  # fmt: skip
    assert launch.report_slots(report, "usa") == [(datetime(2026, 10, 23, 15, 0, tzinfo=ET), "fri")]


def test_launch_goes_through_the_configured_job_runner(tmp_path, monkeypatch):
    runner = tmp_path / "elsewhere" / "run.py"
    monkeypatch.setenv(heal.JOB_RUNNER_ENV, str(runner))
    rig = Rig(tmp_path)
    assert rig.run()["action"] == "launched"
    argv = rig.launches[0][0]
    assert argv[:2] == ["python3", str(runner)]


def test_lease_script_defaults_to_the_code_root_and_honours_the_env(tmp_path, monkeypatch):
    monkeypatch.delenv(launch.LEASE_SCRIPT_ENV, raising=False)
    assert launch.lease_script(tmp_path) == tmp_path / "bin" / "lease.sh"
    custom = tmp_path / "elsewhere" / "lease.sh"
    monkeypatch.setenv(launch.LEASE_SCRIPT_ENV, str(custom))
    assert launch.lease_script(tmp_path) == custom


def test_the_launch_env_marks_the_child_as_backlog_started(tmp_path):
    rig = Rig(tmp_path)
    assert rig.run()["action"] == "launched"
    assert rig.launches[0][2]["GMS_LAUNCHED_BY"] == "health-backlog"


def test_an_installed_pin_older_than_the_newest_release_refuses(tmp_path):
    rig = Rig(tmp_path)
    rig.installed, rig.latest = "0.3.24", "0.3.25"
    res = rig.run()
    assert res["action"] == "refused" and not rig.launches
    assert "0.3.24" in res["reason"] and "0.3.25" in res["reason"]


def test_a_newer_install_than_the_latest_tag_launches(tmp_path):
    rig = Rig(tmp_path)
    rig.installed, rig.latest = "0.3.26", "0.3.25"
    assert rig.run()["action"] == "launched"


def test_an_unknown_latest_release_refuses_rather_than_guessing(tmp_path):
    rig = Rig(tmp_path)

    def offline():
        raise OSError("network is unreachable")

    res = launch.run_backlog(**{**rig.kwargs(), "latest_version": offline})
    assert res["action"] == "refused" and not rig.launches
    assert "latest" in res["reason"] and "network is unreachable" in res["reason"]


def test_an_unreadable_installed_version_refuses(tmp_path):
    rig = Rig(tmp_path)
    res = launch.run_backlog(**{**rig.kwargs(), "installed_version": lambda python: None})
    assert res["action"] == "refused" and not rig.launches


def test_a_dry_run_needs_no_release_lookup(tmp_path):
    rig = Rig(tmp_path)

    def boom():
        raise AssertionError("dry run must not look up releases")

    res = launch.run_backlog(**{**rig.kwargs(dry_run=True), "latest_version": boom})
    assert res["action"] == "would_launch"


@pytest.mark.parametrize(
    "tags,want",
    [
        ("a\trefs/tags/v0.3.9\nb\trefs/tags/v0.3.25\nc\trefs/tags/v0.3.10\n", "0.3.25"),
        ("a\trefs/tags/workspaces/release-1\nb\trefs/tags/v1.0.0\n", "1.0.0"),
    ],
)
def test_latest_release_is_the_highest_version_tag(tags, want):
    assert launch.latest_release_from_tags(tags) == want


def test_no_version_tags_is_an_error():
    with pytest.raises(ValueError):
        launch.latest_release_from_tags("a\trefs/tags/nightly\n")


def test_a_day_gpx_row_without_an_offset_omits_the_flag_rather_than_passing_none(tmp_path):
    rig = Rig(tmp_path)
    gpx = tmp_path / "day12.gpx"
    gpx.write_text("<gpx/>")
    row = rig.rows[0]
    rig.rows[0] = BacklogRow(row.project, row.region, row.sources, "generic", row.lf_mode, day_gpx=str(gpx))  # fmt: skip
    assert rig.run()["action"] == "launched"
    drive_args = rig.launches[0][0]
    assert "--day-gpx" in drive_args and "--camera-utc-offset-hours" not in drive_args
    assert "None" not in drive_args


# --supersede: a row's own stale staging (an earlier run of the same project, e.g. a render on an
# older release) is moved aside with a recorded reason, then the row launches fresh.

STAMP = "20261021T000000Z"  # NOW in UTC


def _stale(rig, project="day-a", *, uploads="{}", hold=False):
    stages = rig.stage(project, rig.rows[0].sources, hold=hold)
    recipe = json.loads((stages / "resume.json").read_text())
    recipe.update(sha="8151c48", phase_reached="phase_d", started_at="2026-11-01T11:18:23Z")
    (stages / "resume.json").write_text(json.dumps(recipe))
    out = rig.out / project
    (out / "shorts").mkdir(parents=True)
    (out / "shorts" / "s01.mp4").write_text("old")
    (out / "youtube_uploads.json").write_text(uploads)
    return stages


def test_without_supersede_the_rows_own_stale_stage_still_skips_it(tmp_path):
    rig = Rig(tmp_path)
    _stale(rig)
    res = rig.run()
    assert res["project"] == "day-b"
    assert "already staged" in res["skipped"][0]["reason"]


def test_supersede_moves_the_rows_stage_and_output_aside_and_launches_it(tmp_path):
    rig = Rig(tmp_path)
    _stale(rig)
    rig.supersede = ("day-a", "0.3.23 render: wrong place labels")
    res = rig.run()
    assert (res["action"], res["project"]) == ("launched", "day-a"), res
    moved_stage = rig.stages / launch.SUPERSEDED_DIR / f"day-a-{STAMP}"
    moved_out = rig.out / launch.SUPERSEDED_DIR / f"day-a-{STAMP}"
    assert not (rig.stages / "day-a").exists() and not (rig.out / "day-a" / "shorts").exists()
    assert (moved_stage / "src" / "_stages" / "resume.json").is_file()
    assert (moved_out / "shorts" / "s01.mp4").read_text() == "old"
    for where in (moved_stage, moved_out):
        record = json.loads((where / launch.SUPERSEDE_RECORD).read_text())
        assert record["schema"] == "backlog-supersede/1"
        assert record["project"] == "day-a"
        assert record["reason"] == "0.3.23 render: wrong place labels"
        assert record["at"] == NOW.isoformat()
        assert record["held"] is False
        assert record["recipe"]["sha"] == "8151c48"
        assert record["recipe"]["phase_reached"] == "phase_d"
        assert {m["to"] for m in record["moves"]} == {str(moved_stage), str(moved_out)}
    assert res["supersede"]["moves"] == record["moves"]
    assert rig.launches and rig.registered[0]["project"] == "day-a"
    assert rig.launches[0][2]["GMS_INTERMEDIATE_DIR"] == str(rig.stages / "day-a")


def test_a_superseded_stage_is_never_read_as_staged(tmp_path):
    rig = Rig(tmp_path)
    one_level = rig.stages / launch.SUPERSEDED_DIR / "old-run" / "_stages"
    one_level.mkdir(parents=True)
    argv = ["/py", "-m", "gomoveshift", "drive", str(rig.footage), "--region", "usa"]
    (one_level / "resume.json").write_text(json.dumps({"argv": argv}))
    assert launch.staged_sources(rig.stages) == []


def test_supersede_dry_run_plans_the_moves_and_changes_nothing(tmp_path):
    rig = Rig(tmp_path)
    _stale(rig)
    rig.supersede = ("day-a", "stale")
    res = rig.run(dry_run=True)
    assert (res["action"], res["project"]) == ("would_launch", "day-a"), res
    assert [m["from"] for m in res["supersede"]["moves"]] == [
        str(rig.stages / "day-a"), str(rig.out / "day-a")]  # fmt: skip
    assert (rig.stages / "day-a").is_dir() and (rig.out / "day-a" / "shorts").is_dir()
    assert not (rig.stages / launch.SUPERSEDED_DIR).exists()


def test_supersede_of_a_held_own_stage_launches_and_records_the_hold(tmp_path):
    rig = Rig(tmp_path)
    _stale(rig, hold=True)
    rig.supersede = ("day-a", "replace the held run")
    res = rig.run()
    assert res["action"] == "launched", res
    moved = rig.stages / launch.SUPERSEDED_DIR / f"day-a-{STAMP}"
    assert json.loads((moved / launch.SUPERSEDE_RECORD).read_text())["held"] is True


@pytest.mark.parametrize("reason", ["", "   "])
def test_supersede_needs_a_reason(tmp_path, reason):
    rig = Rig(tmp_path)
    _stale(rig)
    rig.supersede = ("day-a", reason)
    res = rig.run()
    assert res["action"] == "refused" and "reason" in res["reason"]
    assert (rig.stages / "day-a").is_dir() and not rig.launches


def test_supersede_of_a_row_that_is_not_next_refuses_and_moves_nothing(tmp_path):
    rig = Rig(tmp_path)
    rig.stage("day-b", rig.rows[1].sources)
    rig.supersede = ("day-b", "stale")
    res = rig.run()
    assert res["action"] == "refused" and "day-a" in res["reason"], res
    assert (rig.stages / "day-b").is_dir() and not rig.launches


def test_supersede_of_an_unknown_project_refuses(tmp_path):
    rig = Rig(tmp_path)
    rig.supersede = ("nope", "stale")
    res = rig.run()
    assert res["action"] == "refused" and "nope" in res["reason"] and not rig.launches


@pytest.mark.parametrize("where", ["ledger", "store"])
def test_supersede_never_moves_a_published_or_booked_project(tmp_path, where):
    rig = Rig(tmp_path)
    _stale(rig)
    if where == "ledger":
        rig.ledger = {"day-a"}
    else:
        rig.drives.append(drive("day-a", "complete"))
    rig.supersede = ("day-a", "stale")
    res = rig.run()
    assert res["action"] == "refused", res
    assert (rig.stages / "day-a").is_dir() and not rig.launches


def test_supersede_refuses_an_output_with_upload_records(tmp_path):
    rig = Rig(tmp_path)
    _stale(rig, uploads=json.dumps({"lf01": "abcDEF12345"}))
    rig.supersede = ("day-a", "stale")
    res = rig.run()
    assert res["action"] == "refused" and "youtube_uploads.json" in res["reason"], res
    assert (rig.stages / "day-a").is_dir() and not rig.launches


def test_supersede_refuses_an_unreadable_upload_record(tmp_path):
    rig = Rig(tmp_path)
    _stale(rig, uploads="{not json")
    rig.supersede = ("day-a", "stale")
    res = rig.run()
    assert res["action"] == "refused" and "youtube_uploads.json" in res["reason"], res


def test_supersede_with_nothing_staged_refuses(tmp_path):
    rig = Rig(tmp_path)
    rig.supersede = ("day-a", "stale")
    res = rig.run()
    assert res["action"] == "refused" and "nothing to supersede" in res["reason"], res
    assert not rig.launches


def test_supersede_still_skips_a_row_staged_by_a_held_sibling_run(tmp_path):
    rig = Rig(tmp_path)
    _stale(rig)
    rig.stage("other-run", rig.rows[0].sources, hold=True)
    rig.supersede = ("day-a", "stale")
    res = rig.run()
    assert res["action"] == "refused" and "other-run" in res["reason"], res
    assert (rig.stages / "day-a").is_dir()


def test_supersede_moves_nothing_when_a_later_check_refuses(tmp_path):
    rig = Rig(tmp_path)
    _stale(rig)
    rig.installed, rig.latest = "0.3.24", "0.3.25"
    rig.supersede = ("day-a", "stale")
    res = rig.run()
    assert res["action"] == "refused" and not rig.launches
    assert (rig.stages / "day-a").is_dir()
    assert not (rig.stages / launch.SUPERSEDED_DIR).exists()


def test_supersede_refuses_when_the_target_already_exists(tmp_path):
    rig = Rig(tmp_path)
    _stale(rig)
    (rig.stages / launch.SUPERSEDED_DIR / f"day-a-{STAMP}").mkdir(parents=True)
    rig.supersede = ("day-a", "stale")
    res = rig.run()
    assert res["action"] == "refused" and "already exists" in res["reason"], res
    assert (rig.stages / "day-a").is_dir() and not rig.launches


# --supersede also sets aside a dead sibling: an earlier run of the same footage under another
# project name, which would otherwise keep the row refused as "already staged".


def _sibling(rig, name="old-name", *, uploads=None, hold=False):
    stages = rig.stage(name, rig.rows[0].sources, hold=hold)
    out = rig.out / name
    out.mkdir(parents=True)
    (out / "drive.log").write_text("phases=1/3")
    if uploads is not None:
        (out / "youtube_uploads.json").write_text(uploads)
    return stages


def test_supersede_sets_a_sibling_run_of_the_same_footage_aside(tmp_path):
    rig = Rig(tmp_path)
    _sibling(rig)
    rig.supersede = ("day-a", "same footage, new name")
    res = rig.run()
    assert res["action"] == "launched", res
    moves = {m["from"]: m["to"] for m in res["supersede"]["moves"]}
    assert moves == {
        str(rig.stages / "old-name"): str(rig.stages / launch.SUPERSEDED_DIR / f"old-name-{STAMP}"),
        str(rig.out / "old-name"): str(rig.out / launch.SUPERSEDED_DIR / f"old-name-{STAMP}"),
    }
    assert not (rig.stages / "old-name").exists() and not (rig.out / "old-name").exists()
    for dst in moves.values():
        record = json.loads((Path(dst) / launch.SUPERSEDE_RECORD).read_text())
        assert record["reason"] == "same footage, new name"
        assert [m["from"] for m in record["moves"]] == list(moves)


def test_supersede_dry_run_plans_the_sibling_moves_and_changes_nothing(tmp_path):
    rig = Rig(tmp_path)
    _sibling(rig)
    rig.supersede = ("day-a", "stale")
    res = rig.run(dry_run=True)
    assert [m["from"] for m in res["supersede"]["moves"]] == [
        str(rig.stages / "old-name"),
        str(rig.out / "old-name"),
    ], res
    assert (rig.stages / "old-name").is_dir() and (rig.out / "old-name").is_dir()
    assert not (rig.stages / launch.SUPERSEDED_DIR).exists()


def test_supersede_refuses_a_sibling_whose_output_has_upload_records(tmp_path):
    rig = Rig(tmp_path)
    _sibling(rig, uploads=json.dumps({"s01": {"video_id": "abc"}}))
    rig.supersede = ("day-a", "stale")
    res = rig.run()
    assert res["action"] == "refused" and "old-name" in res["reason"], res
    assert "upload records" in res["reason"]
    assert (rig.stages / "old-name").is_dir() and not rig.launches


def test_supersede_refuses_a_held_sibling(tmp_path):
    rig = Rig(tmp_path)
    _sibling(rig, hold=True)
    rig.supersede = ("day-a", "stale")
    res = rig.run()
    assert res["action"] == "refused" and "hold" in res["reason"], res
    assert (rig.stages / "old-name").is_dir() and not rig.launches


@pytest.mark.parametrize("where", ["store", "ledger"])
def test_supersede_refuses_a_sibling_in_the_store_or_ledger(tmp_path, where):
    rig = Rig(tmp_path)
    _sibling(rig)
    if where == "store":
        rig.drives.append(drive("old-name", "complete"))
    else:
        rig.ledger = {"old-name"}
    rig.supersede = ("day-a", "stale")
    res = rig.run()
    assert res["action"] == "refused" and "old-name" in res["reason"], res
    assert (rig.stages / "old-name").is_dir() and not rig.launches


def test_without_supersede_a_sibling_run_still_skips_the_row(tmp_path):
    rig = Rig(tmp_path)
    _sibling(rig)
    res = rig.run()
    assert res["project"] == "day-b"
    assert "already staged by" in res["skipped"][0]["reason"]
    assert (rig.stages / "old-name").is_dir()


RAW_SHORT = (
    "Day 2025-09-15: raw source footage 2866s is under full_drive_part_min_sec=3600s, so the "
    "day can never fill a part."
)


def test_render_gate_refusal_launches_nothing_and_registers_nothing(tmp_path):
    rig = Rig(tmp_path)
    rig.gate_reason = RAW_SHORT
    res = rig.run()
    assert (res["action"], res["project"]) == ("refused", "day-a"), res
    assert res["reason"] == f"render gate: {RAW_SHORT}"
    assert rig.gates == [(rig.sources / "day-a", "day-a"), (rig.sources / "day-b", "day-b")]
    assert [s["project"] for s in res["skipped"]] == ["day-a", "day-b"]
    assert rig.launches == [] and rig.registered == [] and rig.preflights == []
    assert not rig.sources.exists()  # no subset dir written for a refused row


def test_a_render_gate_refused_row_does_not_stall_the_region(tmp_path):
    # 2026-11-03: sc-2026-02-22-ace-basin (1417 s of footage) was refused at the top of the
    # usa backlog on every hourly tick, and no row behind it was ever launched
    rig = Rig(tmp_path)
    rig.gate_reason = {"day-a": RAW_SHORT}
    res = rig.run()
    assert (res["action"], res["project"]) == ("launched", "day-b"), res
    assert res["skipped"] == [{"project": "day-a", "reason": f"render gate: {RAW_SHORT}"}]
    assert len(rig.launches) == 1 and not (rig.sources / "day-a").exists()


def test_a_dry_run_names_the_row_behind_a_render_gate_refusal(tmp_path):
    rig = Rig(tmp_path)
    rig.gate_reason = {"day-a": RAW_SHORT}
    res = rig.run(dry_run=True)
    assert (res["action"], res["project"]) == ("would_launch", "day-b"), res


def test_render_gate_refuses_a_dry_run_too(tmp_path):
    # a would_launch must never name a row the drive will refuse
    rig = Rig(tmp_path)
    rig.gate_reason = RAW_SHORT
    res = rig.run(dry_run=True)
    assert res["action"] == "refused" and res["reason"].startswith("render gate: "), res


def test_render_gate_refusal_leaves_the_superseded_run_in_place(tmp_path):
    # 2026-11-02: the Ireland supersede moved the old renders aside, launched, and only then
    # did the drive's own preflight refuse the row's 2866 s of raw footage
    rig = Rig(tmp_path)
    _stale(rig)
    rig.supersede = ("day-a", "0.3.23 render: wrong place labels")
    rig.gate_reason = RAW_SHORT
    res = rig.run()
    assert res["action"] == "refused", res
    assert (rig.stages / "day-a").is_dir() and (rig.out / "day-a" / "shorts").is_dir()
    assert not (rig.stages / launch.SUPERSEDED_DIR).exists()
    assert not (rig.out / launch.SUPERSEDED_DIR).exists()
    assert rig.launches == [] and rig.registered == []


def test_config_render_gate_applies_the_rows_modes_to_a_copy(tmp_path, monkeypatch):
    from gomoveshift.config import DriveConfig
    from gomoveshift.pipeline import orchestrator_preflight

    seen = {}

    def fake(config, source_dir):
        seen.update(
            ingest=config.drive.ingest_mode, mode=config.drive.lf_selection_mode, src=source_dir
        )
        return "short"

    monkeypatch.setattr(orchestrator_preflight, "day_under_part_min_message", fake)
    drive_cfg = DriveConfig(ingest_mode="gopro", lf_selection_mode="best_moments")
    config = SimpleNamespace(drive=drive_cfg)
    row = BacklogRow("day-a", "ireland", ("/x/a.mp4",), "generic", "full_drive")
    assert launch.config_render_gate(config)(tmp_path, row) == "short"
    assert seen == {"ingest": "generic", "mode": "full_drive", "src": tmp_path}
    assert (config.drive.ingest_mode, config.drive.lf_selection_mode) == ("gopro", "best_moments")


def test_config_render_gate_scans_the_rows_sources_before_the_subset_exists(tmp_path, monkeypatch):
    # a dry run never creates the subset dir, and must not pass the gate for want of one
    from gomoveshift.config import DriveConfig
    from gomoveshift.pipeline import orchestrator_preflight

    clips = [tmp_path / "GX010032.MP4", tmp_path / "GX010033.MP4"]
    for c in clips:
        c.write_bytes(b"")
    seen = {}

    def fake(config, source_dir):
        seen["links"] = sorted((p.name, os.path.realpath(p)) for p in source_dir.iterdir())
        return None

    monkeypatch.setattr(orchestrator_preflight, "day_under_part_min_message", fake)
    config = SimpleNamespace(drive=DriveConfig())
    row = BacklogRow("day-a", "ireland", tuple(map(str, clips)), "generic", "full_drive")
    assert launch.config_render_gate(config)(tmp_path / "missing", row) is None
    assert seen["links"] == [(c.name, os.path.realpath(c)) for c in clips]
    assert not (tmp_path / "missing").exists()


# --supersede of a failed audit: the row's store entry is ``auditing`` and its newest job failed,
# e.g. 2026-11-02 the Ireland row's drive refused its own day before ingest. Superseding it moves
# that run aside and relaunches the row as it now reads (re-registering the same store row).


def _failed(rig, project="day-a"):
    _stale(rig, project)
    rig.drives.append(drive(project, "auditing"))
    job(rig, "e76bef98", project, "failed", 100.0)
    rig.supersede = (project, "full_drive refused 2866 s; relaunch as best_moments")


def test_supersede_relaunches_a_failed_audit(tmp_path):
    rig = Rig(tmp_path)
    _failed(rig)
    res = rig.run()
    assert (res["action"], res["project"]) == ("launched", "day-a"), res
    assert "e76bef98" in res["relaunch"]
    assert len(rig.launches) == 1 and rig.registered[0]["project"] == "day-a"
    assert (rig.stages / launch.SUPERSEDED_DIR / f"day-a-{STAMP}").is_dir()
    assert not (rig.stages / "day-a").exists()


def test_supersede_of_a_failed_audit_dry_run_moves_nothing(tmp_path):
    rig = Rig(tmp_path)
    _failed(rig)
    res = rig.run(dry_run=True)
    assert res["action"] == "would_launch" and "e76bef98" in res["relaunch"], res
    assert (rig.stages / "day-a").is_dir() and not rig.launches


def test_a_failed_audit_still_refuses_a_supersede_of_another_row(tmp_path):
    rig = Rig(tmp_path)
    _failed(rig)
    rig.supersede = ("day-b", "something else")
    res = rig.run()
    assert res["action"] == "refused" and "e76bef98" in res["reason"], res
    assert not rig.launches


def test_supersede_never_replaces_an_audit_that_is_running(tmp_path):
    rig = Rig(tmp_path)
    _failed(rig)
    job(rig, "again", "day-a", "running", 200.0)
    res = rig.run()
    assert res["action"] == "noop" and "audit in progress" in res["reason"], res
    assert (rig.stages / "day-a").is_dir() and not rig.launches


def test_supersede_still_refuses_a_ready_store_row(tmp_path):
    rig = Rig(tmp_path)
    _stale(rig)
    rig.drives.append(drive("day-a", "ready"))
    rig.slots = [(FRI, "fri"), (TUE, "tue"), (TUE + timedelta(days=1), "wed")]
    rig.supersede = ("day-a", "re-render")
    res = rig.run()
    assert res["action"] == "refused" and "in the schedule store (ready)" in res["reason"], res


# 2026-11-08: nctopa-2026-03-05-rec0063's audit stopped at the OST audit gate (exit 5, "DRIVE
# WAITING:"), which every placeless drive does: no GPS, no system review, so a person confirms its
# OST. The backlog called that "audit failed ... relaunch it", and nothing resumed it once confirmed.
GATE_LINE = (
    "DRIVE WAITING: ost_audit — OST audit pending for day(s) ['2026-03-05']: review each "
    "ost_audit_<date>.json sidecar and create the matching .confirmed file — next: review the "
    "audit, confirm it, then re-run the same command to resume"
)
ENDED = 1000.0


def _waiting(rig, project="day-a", *, hold=False):
    """``project``'s audit stopped at the OST gate: its job ended exit 5 on the waiting line."""
    stages = rig.stage(project, rig.rows[0].sources, hold=hold)
    recipe = json.loads((stages / "resume.json").read_text())
    recipe["argv"] += [
        "--output-dir",
        str(rig.out / project),
        "--placeless-label",
        "Over The Ridges",
    ]
    recipe.update(cwd="/old/svc", env={"GMS_INTERMEDIATE_DIR": str(rig.stages / project)})
    (stages / "resume.json").write_text(json.dumps(recipe))
    sidecar = stages / "ost_audit_2026-03-05.json"
    sidecar.write_text("{}")
    os.utime(sidecar, (ENDED - 60, ENDED - 60))
    rig.drives.append(drive(project, "auditing"))
    job(rig, "1a3f8a41", project, "failed", ENDED - 3600, exit_code=5)
    d = rig.tmp / "jobs" / "1a3f8a41"
    rec = json.loads((d / "job.json").read_text())
    rec["result"]["ended_at"] = ENDED
    (d / "job.json").write_text(json.dumps(rec))
    (d / "output.log").write_text(
        f"01:44:24 [INFO] gomoveshift.cli.handlers.drive {GATE_LINE}\n{GATE_LINE}\n"
    )
    return stages, sidecar


def _confirm(sidecar, at):
    marker = sidecar.with_suffix(".confirmed")
    marker.write_text("")
    os.utime(marker, (at, at))
    return marker


def test_an_audit_waiting_at_the_ost_gate_is_not_called_failed(tmp_path):
    rig = Rig(tmp_path)
    _, sidecar = _waiting(rig)
    res = rig.run()
    assert res["action"] == "noop" and launch.exit_code(res) == 0, res
    assert "failed" not in res["reason"] and "relaunch" not in res["reason"]
    assert "OST audit" in res["reason"] and str(sidecar) in res["reason"]
    assert str(sidecar.with_suffix(".confirmed")) in res["reason"]
    assert not rig.launches and not rig.retired


def test_a_confirmed_ost_audit_resumes_the_waiting_drive_from_its_stages(tmp_path):
    rig = Rig(tmp_path)
    stages, sidecar = _waiting(rig)
    _confirm(sidecar, ENDED + 600)
    res = rig.run()
    assert res["action"] == "launched" and res["project"] == "day-a", res
    assert "resum" in res["reason"]
    ((argv, cwd, env),) = rig.launches
    assert argv[argv.index("--name") + 1] == "gms-backlog-day-a"
    drive_argv = argv[argv.index("--") + 1 :]
    assert drive_argv[:4] == [
        str(rig.tmp / ".venv" / "bin" / "python"),
        "-m",
        "gomoveshift",
        "drive",
    ]
    assert "--placeless-label" in drive_argv and "--skip-upload" in drive_argv
    assert argv[argv.index("--expect") + 1] == str(
        rig.out / "day-a" / "**" / "shorts" / "**" / "*.mp4"
    )
    assert env["GMS_INTERMEDIATE_DIR"] == str(rig.stages / "day-a")
    assert not rig.registered  # the store row is already ``auditing``


def test_a_confirmation_older_than_the_wait_never_resumes_it_again(tmp_path):
    # The drive stopped AFTER the marker was made (the audit text changed and the gate reopened):
    # resuming again would only stop at the same gate, every tick.
    rig = Rig(tmp_path)
    _, sidecar = _waiting(rig)
    _confirm(sidecar, ENDED - 30)
    res = rig.run()
    assert res["action"] == "noop" and "OST audit" in res["reason"], res
    assert not rig.launches


def test_an_operator_hold_keeps_a_confirmed_waiting_drive_where_it_is(tmp_path):
    rig = Rig(tmp_path)
    _, sidecar = _waiting(rig, hold=True)
    _confirm(sidecar, ENDED + 600)
    res = rig.run()
    assert res["action"] == "noop" and "hold" in res["reason"], res
    assert not rig.launches


def test_a_confirmed_waiting_drive_dry_run_launches_nothing(tmp_path):
    rig = Rig(tmp_path)
    _, sidecar = _waiting(rig)
    _confirm(sidecar, ENDED + 600)
    res = rig.run(dry_run=True)
    assert res["action"] == "would_launch" and res["argv"], res
    assert not rig.launches


def test_a_waiting_audit_with_no_backlog_row_is_never_retired(tmp_path):
    rig = Rig(tmp_path)
    _waiting(rig)
    rig.rows = [r for r in rig.rows if r.project != "day-a"]
    res = rig.run()
    assert not rig.retired and not rig.launches
    assert res["action"] == "noop" and "OST audit" in res["reason"], res


def test_a_waiting_audit_can_still_be_superseded(tmp_path):
    rig = Rig(tmp_path)
    _waiting(rig)
    rig.supersede = ("day-a", "relaunch as best_moments")
    res = rig.run()
    assert (res["action"], res["project"]) == ("launched", "day-a"), res
    assert "1a3f8a41" in res["relaunch"]


def test_a_slot_past_seven_days_is_covered_by_default(tmp_path):
    rig = Rig(tmp_path)
    rig.slots = [(NOW + timedelta(days=10), "sun")]
    res = rig.run()
    assert res["action"] == "noop" and "covered" in res["reason"], res
    assert f"inside {launch.COVERAGE_HORIZON_H} h" in res["reason"]


def test_a_wider_horizon_renders_ahead_for_a_later_slot(tmp_path):
    """JP 2026-11-14: render the backlog through as fast as the box allows. A horizon as wide
    as the publish lead window lets the next drive start for a slot weeks out."""
    rig = Rig(tmp_path)
    later = NOW + timedelta(days=40)
    rig.slots = [(later, "sun")]
    res = launch.run_backlog(**rig.kwargs(), horizon_h=70 * 24)
    assert (res["action"], res["slot"], res["lf_day"]) == ("launched", later.isoformat(), "sun"), (
        res
    )


def test_a_wider_horizon_still_reports_its_own_hours_when_covered(tmp_path):
    rig = Rig(tmp_path)
    rig.slots = [(NOW + timedelta(days=80), "sun")]
    res = launch.run_backlog(**rig.kwargs(), horizon_h=70 * 24)
    assert res["action"] == "noop" and "inside 1680 h" in res["reason"], res


@pytest.mark.parametrize(("lead", "want"), [(72, launch.COVERAGE_HORIZON_H), (1680, 1680)])
def test_render_horizon_is_the_wider_of_seven_days_and_the_lead_window(monkeypatch, lead, want):
    from gomoveshift.pipeline import seo_slots

    monkeypatch.setattr(
        seo_slots, "waypost_cadence", lambda config: SimpleNamespace(lead_hours=lead)
    )
    assert launch.render_horizon_h(object()) == want


# --- B6: a launch refused for lack of staging space raises an incident --------------------------


def _space_incidents(tmp_path):
    from gomoveshift.health.store import load_all

    return [r for r in load_all(tmp_path / "inc")[0] if r["kind"] == launch.STORAGE_KIND]


def _storage_run(rig, tmp_path, **kw):
    return launch.run_backlog(**{**rig.kwargs(**kw), "incidents_dir": tmp_path / "inc"})


def test_a_storage_refusal_raises_one_open_incident_per_region(tmp_path):
    rig = Rig(tmp_path)
    rig.devices[rig.stages] = (1, 5.0)
    res = _storage_run(rig, tmp_path)
    assert res["action"] == "refused" and res["reason"].startswith("storage:")
    (row,) = _space_incidents(tmp_path)
    assert row["key"] == "backlog_storage_refused:usa" and not row["resolved_at"]
    assert row["project"] == "day-a" and "5.0 GB free" in row["reason"]
    _storage_run(rig, tmp_path)  # a repeat refusal files nothing new
    assert len(_space_incidents(tmp_path)) == 1


def test_a_launch_resolves_the_storage_incident(tmp_path):
    rig = Rig(tmp_path)
    rig.devices[rig.stages] = (1, 5.0)
    _storage_run(rig, tmp_path)
    rig.devices[rig.stages] = (1, 500.0)
    assert _storage_run(rig, tmp_path)["action"] == "launched"
    (row,) = _space_incidents(tmp_path)
    assert row["resolved_at"]


def test_space_freed_resolves_it_even_when_the_launch_then_stops_elsewhere(tmp_path):
    rig = Rig(tmp_path)
    rig.devices[rig.stages] = (1, 5.0)
    _storage_run(rig, tmp_path)
    rig.devices[rig.stages] = (1, 500.0)
    rig.launch_ok = False
    assert _storage_run(rig, tmp_path)["action"] == "refused"
    assert _space_incidents(tmp_path)[0]["resolved_at"]


def test_another_refusal_leaves_the_storage_incident_open(tmp_path):
    rig = Rig(tmp_path)
    rig.devices[rig.stages] = (1, 5.0)
    _storage_run(rig, tmp_path)
    rig.lease_free = False  # a noop that measured nothing proves nothing
    _storage_run(rig, tmp_path)
    assert not _space_incidents(tmp_path)[0]["resolved_at"]


def test_a_dry_run_and_a_run_with_no_incidents_dir_write_nothing(tmp_path):
    rig = Rig(tmp_path)
    rig.devices[rig.stages] = (1, 5.0)
    _storage_run(rig, tmp_path, dry_run=True)
    assert not (tmp_path / "inc").exists()
    assert rig.run()["action"] == "refused"
    assert not (tmp_path / "inc").exists()


def test_a_reopened_shortage_after_resolution_files_a_fresh_unpushed_incident(tmp_path):
    rig = Rig(tmp_path)
    rig.devices[rig.stages] = (1, 5.0)
    _storage_run(rig, tmp_path)
    rig.devices[rig.stages] = (1, 500.0)
    _storage_run(rig, tmp_path)
    rig.devices[rig.stages] = (1, 5.0)
    rig.registered.clear()
    rig.rows = [rig.row("day-c", 4)]
    _storage_run(rig, tmp_path)
    (row,) = _space_incidents(tmp_path)
    assert not row["resolved_at"] and not row["pushed"]


# --supersede --replace-scheduled: an output whose upload records are all public (kept) or still
# scheduled well ahead (replaced, live-confirmed) is set aside too, with the plan recorded.


def _rec(vid, when):
    return {"video_id": vid, "url": f"https://youtu.be/{vid}", "publish_at": when.isoformat(),
            "title": f"T {vid}", "playlist_id": "PL1"}  # fmt: skip


PAST = NOW - timedelta(days=2)
FAR = NOW + timedelta(days=9)


def _replace_rig(tmp_path, uploads, live=None, flag=True):
    rig = Rig(tmp_path)
    _stale(rig, uploads=json.dumps(uploads))
    rig.supersede = ("day-a", "blur mask was wrong")
    rig.replace_scheduled = flag
    rig.live = live or (lambda ids: {i: "scheduled" for i in ids})
    if flag:
        rig.drives.append(drive("day-a", "complete"))  # M3: only a complete row is replaced
        rig.replace_store = ReopenStore(rig)
    return rig


def test_replace_flag_absent_keeps_the_refusal(tmp_path):
    rig = _replace_rig(tmp_path, {"2": _rec("b", FAR)}, flag=False)
    res = rig.run()
    assert res["action"] == "refused" and "holds upload records" in res["reason"], res


def test_replace_all_scheduled_is_accepted_and_the_plan_is_durable(tmp_path):
    rig = _replace_rig(tmp_path, {"2": _rec("b", FAR), "3": _rec("c", FAR + timedelta(days=7))})
    res = rig.run()
    assert (res["action"], res["project"]) == ("launched", "day-a"), res
    moved = rig.out / launch.SUPERSEDED_DIR / f"day-a-{STAMP}"
    record = json.loads((moved / launch.SUPERSEDE_RECORD).read_text())
    assert record["schema"] == "backlog-supersede/2"
    assert record["replace"]["replace"]["2"]["video_id"] == "b"
    assert record["replace"]["replace"]["2"]["playlist_id"] == "PL1"
    fresh = json.loads((rig.out / "day-a" / "replace_plan.json").read_text())
    assert set(fresh["replace"]) == {"2", "3"} and fresh["kept"] == {}


def test_replace_mix_keeps_the_public_lf(tmp_path):
    rig = _replace_rig(tmp_path, {"1": _rec("a", PAST), "2": _rec("b", FAR)})
    res = rig.run()
    assert res["action"] == "launched", res
    fresh = json.loads((rig.out / "day-a" / "replace_plan.json").read_text())
    assert set(fresh["kept"]) == {"1"} and set(fresh["replace"]) == {"2"}


@pytest.mark.parametrize("why", ["near", "public", "unreadable", "no_live"])
def test_replace_refuses_and_moves_nothing(tmp_path, why):
    uploads = {"2": _rec("b", FAR)}
    live = None
    if why == "near":
        uploads = {"2": _rec("b", NOW + timedelta(hours=2))}
    elif why == "public":
        live = lambda ids: {i: "public" for i in ids}  # noqa: E731
    elif why == "unreadable":
        uploads = {"2": "bareVideoId"}
    rig = _replace_rig(tmp_path, uploads, live)
    if why == "no_live":
        rig.live = None
    res = rig.run()
    assert res["action"] == "refused" and "cannot supersede" in res["reason"], res
    assert (rig.stages / "day-a").is_dir() and not rig.launches


def test_replace_unreadable_file_refuses(tmp_path):
    rig = Rig(tmp_path)
    _stale(rig, uploads="{not json")
    rig.supersede, rig.replace_scheduled = ("day-a", "x"), True
    rig.live = lambda ids: {i: "scheduled" for i in ids}
    assert rig.run()["action"] == "refused"


# --replace-scheduled on a ``complete`` store row: Waypost reopens it (``reopen_for_replace``)
# instead of ``register`` (which refuses complete rows), dry-run first, before any rename.


class ReopenStore:
    """A fake Waypost store: records each ``reopen_for_replace`` and what the disk looked like.
    The parameter list is Waypost's own (``ScheduleStore.reopen_for_replace``, waypost PR #117):
    an unknown keyword raises ``TypeError`` here exactly as it would there."""

    def __init__(self, rig, refuse_dry=None, refuse_real=None):
        self.rig, self.refuse_dry, self.refuse_real, self.calls = rig, refuse_dry, refuse_real, []

    def reopen_for_replace(
        self, project, *, expected, replace, plan_digest, reason, output_dir, now,
        margin_h=6.0, dry_run=False,
    ):  # fmt: skip
        stage_present = (self.rig.stages / project).is_dir()
        self.calls.append({"project": project, "stage_present": stage_present, "expected": expected,
                           "replace": replace, "plan_digest": plan_digest, "reason": reason,
                           "output_dir": output_dir, "now": now, "margin_h": margin_h,
                           "dry_run": dry_run})  # fmt: skip
        return self.refuse_dry if dry_run else self.refuse_real


def _complete_run(rig, store):
    rig.replace_store = store
    return rig.run()


def _three_lfs():
    return {"1": _rec("a", PAST), "2": _rec("b", FAR), "3": _rec("c", FAR + timedelta(days=7))}


def test_complete_row_dry_runs_the_reopen_before_any_rename_then_reopens_for_real(tmp_path):
    import hashlib

    rig = _replace_rig(tmp_path, _three_lfs())
    store = ReopenStore(rig)
    res = _complete_run(rig, store)
    assert res["action"] == "launched", res
    dry, real = store.calls
    assert dry["dry_run"] is True and dry["stage_present"] is True  # nothing renamed yet
    assert real["dry_run"] is False and not real["stage_present"]  # after the renames
    assert rig.registered == []  # register would raise on a complete row
    plan_file = rig.out / "day-a" / "replace_plan.json"
    for call in (dry, real):
        assert call["expected"] == {1: "a", 2: "b", 3: "c"}
        assert call["replace"] == {2, 3}
        assert call["plan_digest"] == hashlib.sha256(plan_file.read_bytes()).hexdigest()
        assert call["reason"] == "blur mask was wrong"
        assert call["output_dir"] == str(rig.out / "day-a")
    assert dry["now"] == NOW and real["now"] > NOW  # the real reopen re-reads the clock


def test_a_dry_run_refusal_moves_nothing_and_never_registers(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    store = ReopenStore(rig, refuse_dry="LF2 is not booked as expected")
    res = _complete_run(rig, store)
    assert res["action"] == "refused" and "LF2 is not booked as expected" in res["reason"], res
    assert [c["dry_run"] for c in store.calls] == [True]
    assert (rig.stages / "day-a").is_dir() and (rig.out / "day-a").is_dir()
    assert not (rig.out / launch.SUPERSEDED_DIR).exists()
    assert not rig.launches and not rig.registered


def test_a_refused_real_reopen_fails_loudly_and_never_registers(tmp_path, caplog):
    rig = _replace_rig(tmp_path, _three_lfs())
    store = ReopenStore(rig, refuse_real="drifted")
    with caplog.at_level("ERROR"):
        res = _complete_run(rig, store)
    assert res["action"] == "refused" and "drifted" in res["reason"], res
    assert launch.exit_code(res) != 0 and rig.registered == []
    assert "drifted" in caplog.text


def test_a_claimed_row_is_still_refused_under_replace(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    rig.drives[:] = [drive("day-a", "claimed")]
    store = rig.replace_store
    res = rig.run()
    assert res["action"] == "refused" and "claimed" in res["reason"], res
    assert not store.calls and (rig.stages / "day-a").is_dir()


def test_a_complete_row_with_no_replace_plan_is_still_refused(tmp_path):
    rig = _replace_rig(tmp_path, {})  # empty uploads file: nothing to plan
    store = ReopenStore(rig)
    res = _complete_run(rig, store)
    assert res["action"] == "refused" and "complete" in res["reason"], res
    assert not store.calls and (rig.stages / "day-a").is_dir() and not rig.launches


def test_a_store_without_reopen_for_replace_is_refused_before_any_rename(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    res = _complete_run(rig, object())
    assert res["action"] == "refused" and "reopen_for_replace" in res["reason"], res
    assert "repin" in res["reason"]
    assert (rig.stages / "day-a").is_dir() and not rig.launches and rig.registered == []


# --replace-scheduled review fixes: M2 (a refusal after the renames is undone), M3 (only a
# ``complete`` row), m2 (fresh clock), m3 (dry run asks Waypost), m5 (incidents).


def _replace_incidents():
    from gomoveshift.health.store import default_data_root, default_incidents_dir, load_all

    rows, _bad = load_all(default_incidents_dir(default_data_root()))
    return [r for r in rows if r.get("kind") == "replace_scheduled_partial"]


def _untouched(rig):
    """The drive exactly as before the attempt: stages, output, receipt, no leftovers."""
    assert (rig.stages / "day-a").is_dir() and (rig.out / "day-a" / "shorts" / "s01.mp4").is_file()
    held = json.loads((rig.out / "day-a" / "youtube_uploads.json").read_text())
    assert held and not (rig.out / "day-a" / "replace_plan.json").exists()
    assert not (rig.out / "day-a" / launch.SUPERSEDE_RECORD).exists()
    assert not list((rig.stages / "day-a").rglob(launch.SUPERSEDE_RECORD))
    for root in (rig.stages, rig.out):
        assert not (root / launch.SUPERSEDED_DIR).exists() or not any(
            (root / launch.SUPERSEDED_DIR).iterdir()
        )


@pytest.mark.parametrize("how", ["preflight", "launcher"])
def test_a_refusal_after_the_renames_restores_the_drive_so_a_retry_can_plan(tmp_path, how):
    rig = _replace_rig(tmp_path, _three_lfs())
    if how == "preflight":
        rig.preflight_error = PreflightBlockedError([("storage", "no space")])
    else:
        rig.launch_ok = False
    res = rig.run()
    assert res["action"] == "refused", res
    _untouched(rig)
    # the real reopen now runs after preflight and before the launch (a launcher failure follows it)
    asked = [c["dry_run"] for c in rig.replace_store.calls]
    assert rig.registered == [] and asked == ([True] if how == "preflight" else [True, False])
    assert launch.due_replace_retry(rig.out, NOW + timedelta(days=2)) == ("day-a", "blur mask was wrong")  # fmt: skip
    rig.preflight_error, rig.launch_ok = None, True  # the blocker is fixed: the retry plans again
    res = rig.run()
    assert res["action"] == "launched", res
    assert (rig.out / "day-a" / "replace_plan.json").is_file()


def test_a_restore_that_cannot_finish_opens_an_incident(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    rig.launch_ok = False
    real_rename = Path.rename

    def rename(self, target):
        if SUPERSEDED_MARK in str(self) and "_stages" not in str(self):
            raise OSError("device busy")
        return real_rename(self, target)

    SUPERSEDED_MARK = launch.SUPERSEDED_DIR
    monkeypatch.setattr(Path, "rename", rename)
    res = rig.run()
    assert res["action"] == "refused" and "restore" in res["reason"], res
    (row,) = _replace_incidents()
    assert row["drive"] == "day-a" and not row["resolved_at"] and "device busy" in row["error"]


@pytest.mark.parametrize("status", ["ready", "auditing", "failed", None])
def test_replace_needs_the_drives_own_store_row_complete(tmp_path, status):
    rig = _replace_rig(tmp_path, _three_lfs())
    rig.drives[:] = [drive("day-a", status)] if status else []
    res = rig.run()
    assert res["action"] != "launched" and not rig.launches and not rig.replace_store.calls, res
    _untouched(rig)
    if status in ("ready", None):
        assert res["action"] == "refused" and "complete" in res["reason"], res


def test_a_refused_real_reopen_launches_nothing_restores_the_drive_and_is_retried(tmp_path):
    # SPEC CHANGE (fixup 4): the real reopen runs BEFORE the launch, so a refusal no longer leaves
    # an audit running on a row that is still complete: the old run goes back, a marker retries it.
    rig = _replace_rig(tmp_path, _three_lfs())
    rig.replace_store = ReopenStore(rig, refuse_real="drifted")
    res = rig.run()
    assert res["action"] == "refused" and "drifted" in res["reason"], res
    assert rig.launches == [] and rig.registered == []
    _untouched(rig)
    rows = _replace_incidents()
    assert {r["code"] for r in rows} == {"reopen_refused", "replace_retry"}
    assert all(not r["resolved_at"] for r in rows)
    assert launch.due_replace_retry(rig.out, NOW) is None  # backoff: not hot
    assert launch.due_replace_retry(rig.out, NOW + timedelta(days=2)) == ("day-a", "blur mask was wrong")  # fmt: skip
    rig.replace_store = ReopenStore(rig)  # Waypost accepts now: the retry launches and clears it
    assert rig.run()["action"] == "launched"
    assert launch.due_replace_retry(rig.out, NOW + timedelta(days=9)) is None
    assert not [r for r in _replace_incidents() if r["code"] in ("reopen_refused", "replace_retry") and not r["resolved_at"]]  # fmt: skip


def test_the_backoff_grows_with_each_failed_attempt(tmp_path):
    for _ in range(3):
        launch.mark_replace_retry(tmp_path, "day-a", "why", NOW)
    assert launch.due_replace_retry(tmp_path, NOW + timedelta(seconds=1)) is None
    assert launch.due_replace_retry(tmp_path, NOW + timedelta(days=30)) == ("day-a", "why")


def test_a_launch_failure_after_a_real_reopen_gives_the_row_back_complete(tmp_path):
    from types import SimpleNamespace

    from gomoveshift.health import publish

    rig = _replace_rig(tmp_path, _three_lfs())
    rows = {"day-a": SimpleNamespace(project="day-a", status="complete", output_dir=str(rig.out / "day-a"))}  # fmt: skip

    class Store(ReopenStore):
        def reopen_for_replace(self, project, **kw):
            if not kw["dry_run"]:
                rows[project].status = "auditing"
            return super().reopen_for_replace(project, **kw)

        def all_drives(self):
            return list(rows.values())

        def retire_published(self, project, held, now=None):
            rows[project].status = "complete"
            return "complete"

    rig.replace_store = Store(rig)
    rig.launch_ok = False
    res = rig.run()
    assert res["action"] == "refused" and "launch failed" in res["reason"], res
    assert rows["day-a"].status == "complete"  # given back at once, from the set-aside receipts
    _untouched(rig)
    # and when that call fails, the ordinary retirement closes it from the restored receipts
    rows["day-a"].status = "auditing"
    assert publish.retire_uploaded(rig.replace_store) == [{"project": "day-a", "status": "complete"}]  # fmt: skip
    assert launch.due_replace_retry(rig.out, NOW + timedelta(days=2)) is not None


def test_a_record_inside_the_margin_is_kept_and_the_rest_replaced(tmp_path):
    uploads = {"1": _rec("a", PAST), "2": _rec("b", NOW + timedelta(hours=2)), "3": _rec("c", FAR)}
    rig = _replace_rig(tmp_path, uploads)
    assert rig.run()["action"] == "launched"
    plan = json.loads((rig.out / "day-a" / "replace_plan.json").read_text())
    assert set(plan["replace"]) == {"3"} and set(plan["kept"]) == {"1", "2"}
    assert plan["margin_kept"] == ["2"]
    (row,) = [r for r in _replace_incidents() if r["code"] == "margin_kept"]
    assert row["drive"] == "day-a" and not row["resolved_at"]
    assert rig.replace_store.calls[-1]["replace"] == {3}


def test_the_real_reopen_gets_a_fresh_clock(tmp_path, monkeypatch):
    later = NOW + timedelta(minutes=47)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return later

    monkeypatch.setattr(launch, "datetime", Clock)
    rig = _replace_rig(tmp_path, _three_lfs())
    assert rig.run()["action"] == "launched"
    dry, real = rig.replace_store.calls
    assert dry["now"] == NOW and real["now"] == later


def test_a_dry_run_asks_waypost_and_reports_its_refusal(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    rig.replace_store = ReopenStore(rig, refuse_dry="LF2 is not booked as expected")
    res = rig.run(dry_run=True)
    assert res["action"] == "refused" and "LF2 is not booked as expected" in res["reason"], res
    _untouched(rig)


def test_a_dry_run_that_waypost_accepts_would_launch_and_writes_nothing(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    res = rig.run(dry_run=True)
    assert res["action"] == "would_launch", res
    assert [c["dry_run"] for c in rig.replace_store.calls] == [True]
    _untouched(rig)


# 2026-11-22: a replace whose LF is not finished must be retried by the machine. A failed publish of
# such a drive is republished with the heal backoff (never hot), and a reopen that raises leaves an
# incident naming the set-aside run, which a retry of --replace-scheduled plans from.


class RaisingStore(ReopenStore):
    def reopen_for_replace(self, project, **kw):
        if kw.get("dry_run"):
            return None
        self.calls.append({"project": project, **kw})
        raise RuntimeError("database is locked")


def test_a_reopen_that_raises_is_retried_then_the_whole_attempt_is_retried_later(tmp_path, monkeypatch):  # fmt: skip
    pauses = []
    monkeypatch.setattr(launch.time, "sleep", pauses.append)
    rig = _replace_rig(tmp_path, _three_lfs())
    rig.replace_store = RaisingStore(rig)
    res = rig.run()
    assert res["action"] == "refused" and "database is locked" in res["reason"], res
    assert rig.launches == [] and len(rig.replace_store.calls) == 3  # retried, not abandoned
    assert [p for p in pauses if p in launch._REOPEN_RETRY_S] == list(launch._REOPEN_RETRY_S)
    _untouched(rig)  # the drive is back as it was: the next attempt plans again from it
    assert {r["code"] for r in _replace_incidents()} == {"reopen_refused", "replace_retry"}
    rig.replace_store = ReopenStore(rig)
    assert rig.run()["action"] == "launched"


def test_a_launcher_that_raises_puts_the_drive_back(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())

    def boom(argv, cwd, env):
        raise OSError("no such runner")

    rig.launcher = boom
    res = rig.run()
    assert res["action"] == "refused" and "no such runner" in res["reason"], res
    assert (rig.stages / "day-a").is_dir() and not (rig.out / launch.SUPERSEDED_DIR).exists()


def _unfinished_publish(rig, tmp_path, *, age_s):
    out = tmp_path / "out-day-a"
    out.mkdir()
    (out / "youtube_uploads.json").write_text(json.dumps({"1": {"video_id": "V1"}}))
    plan = {"schema": "replace-scheduled/1", "kept": {"1": {"video_id": "V1"}},
            "replace": {"2": {"video_id": "old2"}}, "progress": {"2": {"state": "uploaded"}}}  # fmt: skip
    (out / "replace_plan.json").write_text(json.dumps(plan))
    rig.drives.append(drive("day-a", "failed", output_dir=str(out)))
    job(rig, "ok", "day-a", "succeeded", 100.0, exit_code=0)
    job(rig, "pub", "day-a", "failed", NOW.timestamp() - age_s, name="gms-publish-day-a")
    rig.publish_result = {"action": "launched", "reason": "day-a for slot"}


def test_an_unfinished_replace_is_republished_after_the_heal_backoff_not_before(tmp_path):
    rig = Rig(tmp_path / "a")
    _unfinished_publish(rig, tmp_path / "a", age_s=60)
    res = rig.run()
    assert rig.publishes == []  # one publish job a minute ago: the first retry waits an hour
    assert res["publish_backoff"][0]["project"] == "day-a" and 0 < res["publish_backoff"][0]["wait_s"] <= heal._retry_wait_s(1)  # fmt: skip
    rig2 = Rig(tmp_path / "b")
    _unfinished_publish(rig2, tmp_path / "b", age_s=heal._retry_wait_s(1) + 60)
    rig2.run()
    assert rig2.publishes == [("day-a", False)]  # due: re-enters Phase D with no human step


def test_a_failed_replace_whose_manifest_row_was_removed_is_not_given_up(tmp_path):
    rig = Rig(tmp_path)
    _unfinished_publish(rig, tmp_path, age_s=heal._retry_wait_s(1) + 60)
    rig.rows[:] = []  # the manifest row is gone: the drive is no longer listed
    res = rig.run()
    assert rig.publishes == [("day-a", False)] and "held" not in res  # retried, no GAVE_UP_HOLD
    assert not res.get("held") and not list(rig.stages.rglob(launch.HOLD_NAME))  # no hold on disk


def test_the_publish_backoff_counts_only_jobs_after_this_plans_creation(tmp_path):
    rig = Rig(tmp_path)
    _unfinished_publish(rig, tmp_path, age_s=heal._retry_wait_s(1) + 60)
    out = Path(rig.drives[0].output_dir)
    plan = json.loads((out / "replace_plan.json").read_text())
    for n in range(4):  # four failed publishes of an EARLIER replace of this project
        job(
            rig, f"old{n}", "day-a", "failed", NOW.timestamp() - 90000 - n, name="gms-publish-day-a"
        )
    plan["created_at"] = NOW.timestamp() - 86400
    (out / "replace_plan.json").write_text(json.dumps(plan))
    job(
        rig, "ok2", "day-a", "succeeded", NOW.timestamp() - 86000, exit_code=0
    )  # this plan's own audit
    rig.run()
    assert rig.publishes == [("day-a", False)]  # only this plan's one job counts: its wait is over


def test_an_unfinished_replace_never_counts_as_uploaded(tmp_path):
    rig = Rig(tmp_path)
    _unfinished_publish(rig, tmp_path, age_s=60)
    out = rig.drives[0].output_dir
    assert launch._uploaded(out) is False and launch.replace_unfinished(out) == {2: "uploaded"}
    plan = json.loads((Path(out) / "replace_plan.json").read_text())
    plan["progress"]["2"]["state"] = "done"
    (Path(out) / "replace_plan.json").write_text(json.dumps(plan))
    assert launch._uploaded(out) is True and launch.replace_unfinished(out) == {}


# 2026-11-22 fixup 5: a retry marker never stalls a region. One wrapper (``launch.run_tick``) runs a
# due marker as ``--supersede --replace-scheduled`` and decides: launched -> done; a permanent
# refusal -> clear the marker and resolve its incident; any other -> bump it. Either way the same
# tick then falls through to the plain backlog, so the region's own launch/relaunch/audit paths run.


def _retry_tick(rig, *, dry_run=False):
    def run(sup, rep):
        rig.supersede, rig.replace_scheduled = sup, rep
        return rig.run(dry_run=dry_run)

    return launch.run_tick(
        stages_root=rig.stages,
        output_root=rig.out, now=NOW, region="usa", regions={}, dry_run=dry_run,
        supersede=None, replace=False, run=run,
    )  # fmt: skip


def _marker(rig, project="day-a", n=1, region="usa", age=timedelta(days=3)):
    for _ in range(n):
        launch.mark_replace_retry(rig.out, project, "blur mask was wrong", NOW - age, region=region)  # fmt: skip


def _attempts(rig, project="day-a"):
    return json.loads(launch._retry_marker(rig.out, project).read_text())["attempts"]


def _retry_incidents():
    from gomoveshift.health.store import default_data_root, default_incidents_dir, load_all

    rows, _bad = load_all(default_incidents_dir(default_data_root()))
    return [r for r in rows if r.get("kind") == "replace_scheduled_partial" and r.get("code") == "replace_retry"]  # fmt: skip


def test_all_public_uploads_clear_the_marker_resolve_its_incident_and_the_row_launches(tmp_path):
    rig = _replace_rig(tmp_path, {"1": _rec("a", PAST)})
    _marker(rig)
    launch.replace_scheduled_mod.record_incident("day-a", "replace_retry", "earlier failure")
    res = _retry_tick(rig)
    assert (res["action"], res["project"]) == ("launched", "day-b"), res  # same tick, no stall
    assert launch.due_replace_retry(rig.out, NOW + timedelta(days=30), "usa") is None
    assert not launch._retry_marker(rig.out, "day-a").exists()
    (row,) = _retry_incidents()
    assert row["resolved_at"] and "nothing to replace" in row["outcome"], row


def test_a_transient_refusal_bumps_the_marker_and_the_plain_tick_still_launches(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    rig.replace_store = ReopenStore(rig, refuse_dry="LF2 is not booked as expected")
    _marker(rig)
    res = _retry_tick(rig)
    assert (res["action"], res["project"]) == ("launched", "day-b"), res
    assert _attempts(rig) == 2  # bumped once (the refusal site and the wrapper never both bump)
    assert launch.due_replace_retry(rig.out, NOW, "usa") is None  # backoff: not hot
    later = NOW + timedelta(seconds=heal._retry_wait_s(2) + 1)
    assert launch.due_replace_retry(rig.out, later, "usa") == ("day-a", "blur mask was wrong")


def test_a_refusal_that_forgot_to_bump_is_bumped_by_the_wrapper(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    rig.installed = "0.3.24"  # a stale pin: refused with no marker call of its own
    _marker(rig)
    res = _retry_tick(rig)
    assert res["action"] == "refused", res
    assert _attempts(rig) == 2 and launch.due_replace_retry(rig.out, NOW, "usa") is None


def test_one_stuck_marker_never_starves_the_next(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    rig.replace_store = ReopenStore(rig, refuse_dry="drifted")
    _marker(rig, "day-a", age=timedelta(days=4))
    _marker(rig, "ire-b", age=timedelta(days=3))
    _retry_tick(rig)
    assert launch.due_replace_retry(rig.out, NOW, "usa") == ("ire-b", "blur mask was wrong")


def test_a_marker_for_another_region_does_not_hijack_the_tick(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig, "ire-a", region="ireland")
    res = _retry_tick(rig)
    assert (res["action"], res["project"]) == ("launched", "day-b"), res
    assert rig.supersede is None and _attempts(rig, "ire-a") == 1  # untouched
    assert launch.due_replace_retry(rig.out, NOW + timedelta(days=30), "ireland") == ("ire-a", "blur mask was wrong")  # fmt: skip


def test_a_marker_without_a_region_reads_as_its_manifest_regions(tmp_path):
    launch.mark_replace_retry(tmp_path, "ire-a", "why", NOW)  # older code: no region
    later = NOW + timedelta(days=30)
    assert launch.due_replace_retry(tmp_path, later, "usa", {"ire-a": "ireland"}) is None
    assert launch.due_replace_retry(tmp_path, later, "ireland", {"ire-a": "ireland"}) == ("ire-a", "why")  # fmt: skip
    assert launch.due_replace_retry(tmp_path, later, "usa", {}) is None  # unresolvable: unclaimed
    assert launch.due_replace_retry(tmp_path, later, None, {}) == (
        "ire-a",
        "why",
    )  # no region filter


def test_a_crash_after_the_renames_leaves_a_marker_a_later_tick_finds(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    rig.preflight_error = KeyboardInterrupt()  # not an Exception: the process dies mid-replace
    with pytest.raises(KeyboardInterrupt):
        rig.run()
    assert (rig.out / launch.SUPERSEDED_DIR).exists()  # the run is set aside, the crash left it so
    assert launch.due_replace_retry(rig.out, NOW + timedelta(days=2), "usa") == ("day-a", "blur mask was wrong")  # fmt: skip


def test_a_passed_audit_older_than_the_plan_never_publishes_the_fresh_directory(tmp_path):
    rig = Rig(tmp_path)
    out = tmp_path / "fresh"
    out.mkdir()
    plan = {"schema": "replace-scheduled/1", "kept": {}, "created_at": NOW.timestamp() - 60,
            "replace": {"2": {"video_id": "old2"}}}  # fmt: skip
    (out / "replace_plan.json").write_text(json.dumps(plan))
    rig.drives.append(drive("day-a", "auditing", output_dir=str(out)))
    job(
        rig, "old", "day-a", "succeeded", NOW.timestamp() - 9000, exit_code=0
    )  # the OLD run's audit
    rig.publish_result = {"action": "launched", "reason": "day-a for slot"}
    res = rig.run()
    assert rig.publishes == [], res
    job(rig, "new", "day-a", "succeeded", NOW.timestamp() - 30, exit_code=0)  # this replace's own
    rig.run()
    assert rig.publishes == [("day-a", False)]


class RaisingDryStore(ReopenStore):
    def reopen_for_replace(self, project, **kw):
        if kw.get("dry_run"):
            self.calls.append({"project": project, **kw})
            raise RuntimeError("database is locked")
        return super().reopen_for_replace(project, **kw)


def test_a_dry_reopen_that_raises_is_retried_then_bumps_the_marker_not_the_tick(tmp_path, monkeypatch):  # fmt: skip
    monkeypatch.setattr(launch.time, "sleep", lambda s: None)
    rig = _replace_rig(tmp_path, _three_lfs())
    rig.replace_store = RaisingDryStore(rig)
    res = rig.run()  # must return, never raise
    assert res["action"] == "refused" and "database is locked" in res["reason"], res
    assert len(rig.replace_store.calls) == 3  # retried like the real reopen
    assert _attempts(rig) == 1 and (rig.stages / "day-a").is_dir()  # marker bumped, nothing moved


def test_a_restore_that_cannot_finish_keeps_the_marker_for_the_next_tick(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    rig.launch_ok = False
    real_rename = Path.rename

    def rename(self, target):
        if launch.SUPERSEDED_DIR in str(self) and "_stages" not in str(self):
            raise OSError("device busy")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", rename)
    assert rig.run()["action"] == "refused"
    assert launch.due_replace_retry(rig.out, NOW + timedelta(days=2), "usa") == ("day-a", "blur mask was wrong")  # fmt: skip


def test_every_record_margin_kept_says_so_and_is_permanent():
    soon = {"1": _rec("a", NOW + timedelta(hours=2))}
    plan, why = launch.replace_scheduled_mod.classify_uploads(soon, NOW, lambda ids: {})
    assert plan is None and "margin" in why and "already public" not in why, why
    assert launch.refusal_is_permanent(why)
    public, why2 = launch.replace_scheduled_mod.classify_uploads({"1": _rec("a", PAST)}, NOW, lambda ids: {})  # fmt: skip
    assert public is None and "already public" in why2 and launch.refusal_is_permanent(why2)
    assert not launch.refusal_is_permanent("cannot supersede day-a: the next eligible usa row is day-b")  # fmt: skip


def test_a_dry_run_refusal_leaves_a_retry_marker(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    rig.replace_store = ReopenStore(rig, refuse_dry="LF2 is not booked as expected")
    assert rig.run()["action"] == "refused"
    assert launch.due_replace_retry(rig.out, NOW + timedelta(days=2), "usa") == ("day-a", "blur mask was wrong")  # fmt: skip


# 2026-11-22 fixup 6: only a replace's OWN launch clears its marker; a crash between the renames and
# the plan write never becomes a permanent give-up; a project gone from the region's rows is the
# permanent refusal even while another row is eligible; a raise after a launch never re-marks.


def test_another_drives_publish_in_the_marker_attempt_never_clears_the_marker(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    rig.rows.append(rig.row("day-c", 1))
    rig.drives.append(drive("day-c", "auditing"))
    job(rig, "c1", "day-c", "succeeded", NOW.timestamp() - 100, exit_code=0)
    rig.publish_result = {"action": "launched", "reason": "day-c published"}
    launch.replace_scheduled_mod.record_incident("day-a", "replace_retry", "earlier failure")
    _marker(rig)
    res = _retry_tick(rig)
    assert (res["action"], res["project"]) == ("launched", "day-c"), res  # returned as is
    assert "replace_retry" not in res and rig.publishes == [("day-c", False)]  # no fall-through
    assert _attempts(rig) == 1 and not rig.launches and (rig.stages / "day-a").is_dir()
    assert not _retry_incidents()[0].get("resolved_at")
    rig.publish_result = None  # the next tick: day-c is published (complete), the replace launches
    rig.drives[-1] = drive("day-c", "complete")
    res2 = _retry_tick(rig)
    assert res2["action"] == "launched" and res2.get("project") == "day-a", res2
    assert not launch._retry_marker(rig.out, "day-a").exists() and len(rig.launches) == 1
    assert _retry_incidents()[0].get("resolved_at")


def test_a_marker_whose_row_left_the_manifest_is_permanent_even_beside_an_eligible_row(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    rig.rows[:] = [r for r in rig.rows if r.project != "day-a"]  # day-b stays eligible
    _marker(rig)
    res = _retry_tick(rig)
    assert res["replace_retry"]["outcome"] == "cleared", res["replace_retry"]
    assert "no usa row day-a in the backlog" in res["replace_retry"]["reason"]
    assert not launch._retry_marker(rig.out, "day-a").exists()


def _crash_after_the_renames(rig, monkeypatch, exc):
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(launch.replace_scheduled_mod, "write_plan", boom)


def test_a_crash_between_the_renames_and_the_plan_write_is_finished_by_the_next_tick(tmp_path, monkeypatch):  # fmt: skip
    rig = _replace_rig(tmp_path, _three_lfs())
    real = launch.replace_scheduled_mod.write_plan
    _crash_after_the_renames(rig, monkeypatch, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        rig.run()
    monkeypatch.setattr(launch.replace_scheduled_mod, "write_plan", real)
    assert launch._retry_marker(rig.out, "day-a").exists() and not (rig.out / "day-a").exists()
    later = NOW + timedelta(days=2)

    def run(sup, rep):
        rig.supersede, rig.replace_scheduled = sup, rep
        return launch.run_backlog(**{**rig.kwargs(), "now": later})

    res = launch.run_tick(stages_root=rig.stages, output_root=rig.out, now=later, region="usa", regions={}, dry_run=False,
                          supersede=None, replace=False, run=run)  # fmt: skip
    assert (res["action"], res["project"]) == ("launched", "day-a"), res
    assert not launch._retry_marker(rig.out, "day-a").exists()
    plan = json.loads((rig.out / "day-a" / "replace_plan.json").read_text())
    assert set(plan["replace"]) == {"2", "3"}  # planned from the set-aside run's records


def test_a_failing_supersede_is_put_back_and_retried_never_raised(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _crash_after_the_renames(rig, monkeypatch, OSError("disk full"))
    res = rig.run()  # must return, never raise
    assert res["action"] == "refused" and "disk full" in res["reason"], res
    assert (rig.stages / "day-a").is_dir() and not rig.launches  # the old run is back
    assert _attempts(rig) == 1 and not launch.refusal_is_permanent(res["reason"])


def test_nothing_to_supersede_with_no_set_aside_run_is_permanent(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    shutil.rmtree(rig.stages / "day-a")
    shutil.rmtree(rig.out / "day-a")  # no stages, no output, and nothing set aside
    _marker(rig)
    res = _retry_tick(rig)
    assert res["replace_retry"]["outcome"] == "cleared", res["replace_retry"]
    assert "nothing to supersede" in res["replace_retry"]["reason"]


def test_a_raise_after_the_replace_launched_never_remarks_or_falls_through(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    calls = []

    def run(sup, rep):
        calls.append(sup)
        rig.supersede, rig.replace_scheduled = sup, rep
        out = rig.run()
        if sup and out.get("action") == "launched":
            raise OSError("stage_roots: disk hiccup after the launch")
        return out

    res = launch.run_tick(stages_root=rig.stages, output_root=rig.out, now=NOW, region="usa", regions={}, dry_run=False,
                          supersede=None, replace=False, run=run)  # fmt: skip
    assert not launch._retry_marker(rig.out, "day-a").exists(), "a launched replace got its marker back"  # fmt: skip
    assert len(rig.launches) == 1 and calls == [("day-a", "blur mask was wrong")], calls
    assert res["action"] == "launched" and res["project"] == "day-a", res


def test_a_raise_inside_the_marker_call_bumps_the_marker_and_never_crashes_the_tick(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)

    def run(sup, rep):
        if sup:
            raise RuntimeError("seo store exploded")
        rig.supersede, rig.replace_scheduled = None, False
        return rig.run()

    res = launch.run_tick(stages_root=rig.stages, output_root=rig.out, now=NOW, region="usa", regions={}, dry_run=False,
                          supersede=None, replace=False, run=run)  # fmt: skip
    assert (res["action"], res["project"]) == ("launched", "day-b"), res  # fell through
    assert res["replace_retry"]["outcome"] == "bumped" and "seo store exploded" in res["replace_retry"]["reason"]  # fmt: skip
    assert _attempts(rig) == 2


def test_one_call_bumps_the_marker_once_however_many_sites_ask(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    rig.launch_ok = False  # the pre-rename mark, then the refusal's retry_later: one bump total
    _marker(rig)
    assert _retry_tick(rig)["replace_retry"]["outcome"] == "bumped"
    assert _attempts(rig) == 2


def test_the_oldest_due_marker_runs_first_whatever_its_name(tmp_path):
    launch.mark_replace_retry(tmp_path, "day-a", "why-a", NOW - timedelta(days=3), region="usa")
    launch.mark_replace_retry(tmp_path, "ire-b", "why-b", NOW - timedelta(days=4), region="usa")
    assert launch.due_replace_retry(tmp_path, NOW, "usa") == ("ire-b", "why-b")


@pytest.mark.parametrize(
    "why",
    [
        "cannot supersede day-a: nothing to supersede: no day-a under the stages or output root",
        "cannot supersede day-a: no usa row day-a in the backlog",
        "cannot supersede day-a: the schedule store row is complete and no upload record gives a replace plan; only Waypost can reopen it",  # fmt: skip
        "every record is already public: nothing to replace",
    ],
)
def test_the_permanent_refusals_are_permanent(why):
    assert launch.refusal_is_permanent(why)


def test_the_margin_refusals_say_exactly_which_case_it_is():
    near = {"1": _rec("a", NOW + timedelta(hours=2))}
    _, only = launch.replace_scheduled_mod.classify_uploads(near, NOW, lambda ids: {})
    assert only == "every record is scheduled inside the 6 h margin (kept by design, each publishes on its own): nothing to replace"  # fmt: skip
    _, mixed = launch.replace_scheduled_mod.classify_uploads({**near, "2": _rec("b", PAST)}, NOW, lambda ids: {})  # fmt: skip
    assert mixed == "every record is public or inside the 6 h margin (kept by design): nothing to replace"  # fmt: skip


def test_resolve_incidents_closes_only_the_named_codes():
    mod = launch.replace_scheduled_mod
    mod.record_incident("day-a", "replace_retry", "x")
    mod.record_incident("day-a", "restore_failed", "y")
    assert mod.resolve_incidents("day-a", drive_level=True, codes=("replace_retry",), outcome="given up: z") == 1  # fmt: skip
    from gomoveshift.health.store import default_data_root, default_incidents_dir, load_all

    rows, _bad = load_all(default_incidents_dir(default_data_root()))
    by = {r["code"]: r for r in rows if r.get("kind") == "replace_scheduled_partial"}
    assert by["replace_retry"]["resolved_at"] and by["replace_retry"]["outcome"] == "given up: z"
    assert not by["restore_failed"].get("resolved_at")


def test_a_regionless_marker_whose_row_is_in_another_region_is_left_for_that_region(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    launch.mark_replace_retry(rig.out, "ire-a", "why", NOW - timedelta(days=3))  # no region

    def run(sup, rep):
        rig.supersede, rig.replace_scheduled = sup, rep
        return rig.run()

    res = launch.run_tick(stages_root=rig.stages, output_root=rig.out, now=NOW, region="usa", regions={"ire-a": "ireland"}, dry_run=False, supersede=None, replace=False, run=run)  # fmt: skip
    assert (res["action"], res["project"]) == ("launched", "day-b"), res  # the plain tick ran
    held = json.loads(launch._retry_marker(rig.out, "ire-a").read_text())
    assert "region" not in held and held["attempts"] == 1  # untouched, never stamped


# Fixup 7: a set-aside run always carries its record, belongs to its own project, and a launched
# replace never re-runs.


def _tick_at(rig, when, jobs_dir=None):
    def run(sup, rep):
        rig.supersede, rig.replace_scheduled = sup, rep
        return launch.run_backlog(**{**rig.kwargs(), "now": when})

    return launch.run_tick(stages_root=rig.stages, output_root=rig.out, now=when, region="usa", regions={}, dry_run=False, supersede=None, replace=False, run=run, jobs_dir=jobs_dir)  # fmt: skip


def _crash_on_rename(monkeypatch, rig, *, after):
    """Die (not an ``Exception``: the process is gone) on the first rename into a ``_superseded``
    dir under the output root, before it happens or right after it."""
    real = Path.rename
    aside = rig.out / launch.SUPERSEDED_DIR

    def crash(self, target):
        if aside in Path(target).parents:
            if after:
                real(self, target)
            raise KeyboardInterrupt()
        return real(self, target)

    monkeypatch.setattr(Path, "rename", crash)
    return real, aside


def test_a_crash_right_after_the_out_rename_never_gives_the_replace_up(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    real, aside = _crash_on_rename(monkeypatch, rig, after=True)
    with pytest.raises(KeyboardInterrupt):
        rig.run()
    monkeypatch.setattr(Path, "rename", real)
    moved = next(aside.glob("day-a-*"))
    assert (moved / launch.SUPERSEDE_RECORD).exists()  # the record went in BEFORE the rename
    assert launch._retry_marker(rig.out, "day-a").exists()
    res = _tick_at(rig, NOW + timedelta(days=2))
    assert (res["action"], res["project"]) == ("launched", "day-a"), res
    assert not launch._retry_marker(rig.out, "day-a").exists() and len(rig.launches) == 1
    assert set(json.loads((rig.out / "day-a" / "replace_plan.json").read_text())["replace"]) == {"2", "3"}  # fmt: skip


def test_a_crash_after_the_record_write_before_the_rename_leaves_a_harmless_record(tmp_path, monkeypatch):  # fmt: skip
    rig = _replace_rig(tmp_path, _three_lfs())
    real, _aside = _crash_on_rename(monkeypatch, rig, after=False)
    with pytest.raises(KeyboardInterrupt):
        rig.run()
    monkeypatch.setattr(Path, "rename", real)
    stray = rig.out / "day-a" / launch.SUPERSEDE_RECORD  # written into the dir still in place
    assert stray.exists() and launch._retry_marker(rig.out, "day-a").exists()
    assert launch._superseded_uploads("day-a", rig.out / "day-a") is None  # ignored outside the set-aside dir  # fmt: skip
    res = _tick_at(rig, NOW + timedelta(days=2))
    assert (res["action"], res["project"]) == ("launched", "day-a"), res
    assert not launch._retry_marker(rig.out, "day-a").exists() and len(rig.launches) == 1
    assert not stray.exists()  # it went into the set-aside run with the dir


def _aside(out, name, project, uploads):
    d = out / launch.SUPERSEDED_DIR / name
    d.mkdir(parents=True)
    (d / launch.SUPERSEDE_RECORD).write_text(json.dumps({"schema": "backlog-supersede/2", "project": project}))  # fmt: skip
    (d / "youtube_uploads.json").write_text(json.dumps(uploads))
    return d


def test_the_set_aside_fallback_never_reads_another_projects_run(tmp_path):
    out = tmp_path / "out"
    mine = _aside(out, "day-a-20261021T000000Z", "day-a", {"2": {"video_id": "mine"}})
    _aside(out, "day-a-long-20261001T000000Z", "day-a-long", {"2": {"video_id": "OTHER"}})
    _aside(out, "day-a-20261030T000000Z-failed-attempt-20261030T000000Z", "day-a", {"2": {"video_id": "X"}})  # fmt: skip
    assert launch._superseded_uploads("day-a", out / "day-a") == mine / "youtube_uploads.json"
    assert launch._superseded_uploads("day-a-long", out / "day-a-long").parent.name.startswith("day-a-long-")  # fmt: skip
    assert launch._superseded_uploads("day", out / "day") is None


def test_the_set_aside_fallback_needs_a_record_naming_its_own_project(tmp_path):
    out = tmp_path / "out"
    _aside(out, "day-a-20261021T000000Z", "someone-else", {"2": {"video_id": "OTHER"}})
    assert launch._superseded_uploads("day-a", out / "day-a") is None


def test_a_regionless_marker_with_no_row_is_given_up_and_recorded_durably(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    launch.mark_replace_retry(rig.out, "ghost-a", "why", NOW - timedelta(days=3))  # no region
    res = _tick_at(rig, NOW)
    assert (res["action"], res["project"]) == ("launched", "day-b"), res  # the plain tick ran
    assert not launch._retry_marker(rig.out, "ghost-a").exists()
    assert res["given_up_replace_retry"] == ["ghost-a"] and "unclaimed_replace_retry" not in res
    from gomoveshift.health.store import default_data_root, default_incidents_dir, load_all

    rows, _bad = load_all(default_incidents_dir(default_data_root()))
    ghost = [r for r in rows if r.get("drive") == "ghost-a" and r.get("code") == "replace_retry"]
    assert len(ghost) == 1 and ghost[0]["resolved_at"], ghost
    assert ghost[0]["outcome"] == "given up: no row in any region"


def test_a_dry_run_never_gives_a_regionless_marker_up(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    launch.mark_replace_retry(rig.out, "ghost-a", "why", NOW - timedelta(days=3))
    launch.run_tick(stages_root=rig.stages, output_root=rig.out, now=NOW, region="usa", regions={}, dry_run=True, supersede=None, replace=False, run=lambda s, r: rig.run(dry_run=True))  # fmt: skip
    assert launch._retry_marker(rig.out, "ghost-a").exists()


def test_a_raise_after_another_drives_publish_never_remarks_or_falls_through(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    rig.rows.append(rig.row("day-c", 1))
    rig.drives.append(drive("day-c", "auditing"))
    job(rig, "c1", "day-c", "succeeded", NOW.timestamp() - 100, exit_code=0)
    rig.publish_result = {"action": "launched", "reason": "day-c published"}
    _marker(rig)

    def run(sup, rep):
        rig.supersede, rig.replace_scheduled = sup, rep
        out = rig.run()
        if sup and out.get("action") == "launched":
            raise OSError("stage_roots hiccup after day-c's publish")
        return out

    res = launch.run_tick(stages_root=rig.stages, output_root=rig.out, now=NOW, region="usa", regions={}, dry_run=False, supersede=None, replace=False, run=run)  # fmt: skip
    assert res["action"] == "launched" and res["project"] == "day-c" and "replace_retry" not in res
    assert rig.publishes == [("day-c", False)] and not rig.launches, (rig.publishes, rig.launches)
    assert _attempts(rig) == 1


def test_a_raise_after_the_launcher_returned_ok_never_re_marks_or_falls_through(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)

    def run(sup, rep):
        rig.supersede, rig.replace_scheduled = sup, rep
        out = rig.run()
        raise OSError("post-launch hiccup")

    res = launch.run_tick(stages_root=rig.stages, output_root=rig.out, now=NOW, region="usa", regions={}, dry_run=False, supersede=None, replace=False, run=run)  # fmt: skip
    assert res["action"] == "launched" and "replace_retry" not in res and len(rig.launches) == 1


def test_a_crash_after_the_launch_before_the_clear_never_replaces_twice(tmp_path, monkeypatch):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    real = launch.clear_replace_retry

    def die(*a, **k):
        raise KeyboardInterrupt()

    monkeypatch.setattr(launch, "clear_replace_retry", die)
    with pytest.raises(KeyboardInterrupt):
        _retry_tick(rig)
    monkeypatch.setattr(launch, "clear_replace_retry", real)
    assert len(rig.launches) == 1
    held = json.loads(launch._retry_marker(rig.out, "day-a").read_text())
    assert held["launching_at"] == NOW.timestamp()  # stamped BEFORE the launcher ran
    job(rig, "new", "day-a", "running", NOW.timestamp() + 5)  # the launcher's job exists now
    rig.running = True
    res = _tick_at(rig, NOW + timedelta(days=5), jobs_dir=rig.tmp / "jobs")
    assert len(rig.launches) == 1, "the marker replaced the fresh run a second time"
    assert not launch._retry_marker(rig.out, "day-a").exists()
    assert res["replace_retry"]["outcome"] == "cleared", res


def test_a_launch_stamp_with_no_job_since_it_is_retried_not_trusted(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    launch.stamp_replace_launch(rig.out, "day-a", NOW.timestamp())  # died before the launcher ran
    job(rig, "old", "day-a", "succeeded", NOW.timestamp() - 9000, exit_code=0)  # the old run's
    res = _tick_at(rig, NOW + timedelta(days=5), jobs_dir=rig.tmp / "jobs")
    assert (res["action"], res["project"]) == ("launched", "day-a"), res
    assert len(rig.launches) == 1 and not launch._retry_marker(rig.out, "day-a").exists()


def test_a_failed_launcher_drops_the_launch_stamp(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    rig.launch_ok = False
    _marker(rig)
    _retry_tick(rig)
    assert "launching_at" not in json.loads(launch._retry_marker(rig.out, "day-a").read_text())


def test_the_wrappers_own_clear_is_the_second_line_when_the_in_run_clear_did_not_take(tmp_path, monkeypatch):  # fmt: skip
    """``_run`` clears the marker on the replace's launch and the wrapper clears it again: the
    second is not dead, it is what removes the marker when the first unlink failed."""
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    real, calls = launch.clear_replace_retry, []

    def flaky(out, project):
        calls.append(project)
        if len(calls) > 1:
            real(out, project)

    monkeypatch.setattr(launch, "clear_replace_retry", flaky)
    res = _retry_tick(rig)
    assert (res["action"], res["project"]) == ("launched", "day-a"), res
    assert calls == ["day-a", "day-a"] and not launch._retry_marker(rig.out, "day-a").exists()


def test_a_same_project_launch_that_is_not_the_replace_leaves_the_marker(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    out = launch.run_tick(stages_root=rig.stages, output_root=rig.out, now=NOW, region="usa", regions={}, dry_run=False, supersede=None, replace=False, run=lambda s, r: {"action": "launched", "project": "day-a", "skipped": [], "reason": "day-a published"})  # fmt: skip
    assert out["action"] == "launched" and "replace_retry" not in out
    assert _attempts(rig) == 1  # a publish of the same project is not the replace


def test_a_replaced_rows_old_passed_audit_never_publishes_through_the_failed_loop(tmp_path):
    rig = Rig(tmp_path)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    plan = {"schema": "replace-scheduled/1", "kept": {}, "created_at": NOW.timestamp() - 60, "replace": {"2": {"video_id": "old2"}}}  # fmt: skip
    (fresh / "replace_plan.json").write_text(json.dumps(plan))
    rig.drives.append(drive("day-a", "failed", output_dir=str(fresh)))
    job(rig, "old", "day-a", "succeeded", NOW.timestamp() - 9000, exit_code=0)  # the OLD run's
    job(
        rig, "pub", "day-a", "failed", NOW.timestamp() - 8000, name="gms-publish-day-a"
    )  # before the plan: no backoff
    rig.publish_result = {"action": "launched", "reason": "day-a for slot"}
    rig.run()
    assert rig.publishes == []


def test_a_dropped_rows_old_failed_audit_never_retires_the_replaced_run(tmp_path):
    rig = Rig(tmp_path)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    plan = {"schema": "replace-scheduled/1", "kept": {}, "created_at": NOW.timestamp() - 60, "replace": {"2": {"video_id": "old2"}}}  # fmt: skip
    (fresh / "replace_plan.json").write_text(json.dumps(plan))
    rig.drives.append(drive("day-x", "auditing", output_dir=str(fresh)))  # no manifest row
    job(rig, "old", "day-x", "failed", NOW.timestamp() - 9000)  # the set-aside run's
    rig.run()
    assert rig.retired == []


# Fixup 8: a supersede that fails before its move never buries the live run, and a launched or
# kept-margin replace leaves its record whatever dies.


def _named(rig, project="day-a"):
    return [a for a, _c, _e in rig.launches if f"gms-backlog-{project}" in a]


def _launcher_writing_jobs(rig, created_at):
    """The real job runner writes ``job.json`` (``gms-backlog-<project>``) before it returns."""
    made = []

    def launcher(argv, cwd, env):
        rig.launches.append((argv, cwd, env))
        made.append(1)
        d = rig.tmp / "jobs" / f"run{len(made)}"
        d.mkdir(parents=True)
        rec = {"schema": "job/1", "run_id": f"run{len(made)}", "name": argv[argv.index("--name") + 1], "created_at": created_at(), "result": {"state": "running", "exit_code": None}}  # fmt: skip
        (d / "job.json").write_text(json.dumps(rec))
        return rig.launch_ok, "" if rig.launch_ok else "runner: TimeoutExpired"

    rig.launcher = launcher


def _first_record_write_fails(monkeypatch):
    import errno

    real, fired = Path.write_text, []

    def full(self, *a, **k):
        if self.name == launch.SUPERSEDE_RECORD and not fired:
            fired.append(self)
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", full)
    return lambda: monkeypatch.setattr(Path, "write_text", real)


@pytest.mark.parametrize("prior_aside", [True, False])
def test_a_record_write_failure_before_any_move_leaves_the_old_run_and_retries(tmp_path, monkeypatch, prior_aside):  # fmt: skip
    from gomoveshift.pipeline.cleanup import _superseded_by_record

    rig = _replace_rig(tmp_path, _three_lfs())
    if prior_aside:  # an earlier supersede left set-aside runs: ``_superseded`` already exists
        (rig.out / launch.SUPERSEDED_DIR / "older-20260101T000000Z").mkdir(parents=True)
        (rig.stages / launch.SUPERSEDED_DIR / "older-20260101T000000Z").mkdir(parents=True)
    _marker(rig)
    restore = _first_record_write_fails(monkeypatch)
    _retry_tick(rig)
    restore()
    live = rig.out / "day-a"
    assert (live / "youtube_uploads.json").exists() and (live / "shorts" / "s01.mp4").exists()
    assert (rig.stages / "day-a").is_dir() and not list((rig.out / launch.SUPERSEDED_DIR).glob("day-a-*"))  # fmt: skip
    assert not (live / launch.SUPERSEDE_RECORD).exists() and not _superseded_by_record(live)
    assert launch._retry_marker(rig.out, "day-a").exists()
    for days in (2, 4):  # never "nothing to supersede", never given up
        if not launch._retry_marker(rig.out, "day-a").exists():
            break
        res = _tick_at(rig, NOW + timedelta(days=days))
        if res["action"] == "launched" and res["project"] == "day-a":
            break
    assert len(_named(rig)) == 1 and not launch._retry_marker(rig.out, "day-a").exists()


def test_a_failed_supersede_after_the_first_move_still_puts_back_and_cleans_records(tmp_path, monkeypatch):  # fmt: skip
    from gomoveshift.pipeline.cleanup import _superseded_by_record

    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    real = Path.rename
    hit = []

    def die_on_out(self, target):
        if self == rig.out / "day-a" and not hit:
            hit.append(1)
            raise OSError("EXDEV")
        return real(self, target)

    monkeypatch.setattr(Path, "rename", die_on_out)
    _retry_tick(rig)
    monkeypatch.setattr(Path, "rename", real)
    live = rig.out / "day-a"
    assert (live / "youtube_uploads.json").exists() and (rig.stages / "day-a").is_dir()
    assert not (live / launch.SUPERSEDE_RECORD).exists() and not _superseded_by_record(live)


def test_a_crash_after_the_launch_before_the_clear_still_records_the_kept_margin(tmp_path, monkeypatch):  # fmt: skip
    near = NOW + timedelta(hours=3)
    rig = _replace_rig(tmp_path, {"1": _rec("a", near), "2": _rec("b", FAR), "3": _rec("c", FAR + timedelta(days=7))})  # fmt: skip
    _launcher_writing_jobs(rig, lambda: NOW.timestamp() + 3)
    _marker(rig)
    monkeypatch.setattr(launch, "clear_replace_retry", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))  # fmt: skip
    with pytest.raises(KeyboardInterrupt):
        _retry_tick(rig)
    kept = _kept_rows()
    assert len(kept) == 1 and kept[0]["lf"] == 1, kept  # recorded BEFORE the clear that died


def _kept_rows():
    from gomoveshift.health.store import default_data_root, default_incidents_dir, load_all

    rows, _bad = load_all(default_incidents_dir(default_data_root()))
    return [r for r in rows if r.get("drive") == "day-a" and r.get("code") == "margin_kept"]


def test_the_launch_stamp_branch_records_the_kept_margin_once(tmp_path, monkeypatch):
    near = NOW + timedelta(hours=3)
    rig = _replace_rig(tmp_path, {"1": _rec("a", near), "2": _rec("b", FAR), "3": _rec("c", FAR + timedelta(days=7))})  # fmt: skip
    _launcher_writing_jobs(rig, lambda: NOW.timestamp() + 3)
    _marker(rig)
    real = launch.clear_replace_retry
    monkeypatch.setattr(launch, "clear_replace_retry", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))  # fmt: skip
    with pytest.raises(KeyboardInterrupt):
        _retry_tick(rig)
    monkeypatch.setattr(launch, "clear_replace_retry", real)
    # the first path's record is lost with the crash: only the stamp branch is left
    for f in _kept_files():
        f.unlink()
    assert not _kept_rows()
    res = _tick_at(rig, NOW + timedelta(days=1), jobs_dir=rig.tmp / "jobs")
    assert res["replace_retry"]["outcome"] == "cleared" and len(_named(rig)) == 1
    assert len(_kept_rows()) == 1
    _tick_at(rig, NOW + timedelta(days=2), jobs_dir=rig.tmp / "jobs")
    assert len(_kept_rows()) == 1  # idempotent: never a duplicate


def _kept_files():
    from gomoveshift.health.store import default_data_root, default_incidents_dir

    return list(default_incidents_dir(default_data_root()).glob("*margin_kept*.json"))


def test_a_launcher_timeout_after_its_job_was_written_never_replaces_twice(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    rig.launch_ok = False  # default_launcher: (False, TimeoutExpired) after the runner detached
    _launcher_writing_jobs(rig, lambda: NOW.timestamp() + 3)
    _marker(rig)
    res = _retry_tick(rig)
    assert (res["action"], res["project"]) == ("launched", "day-a"), res
    assert not launch._retry_marker(rig.out, "day-a").exists()
    # the fresh dir is in use, not put back
    assert (rig.out / "day-a" / "replace_plan.json").exists()
    assert not list((rig.out / "day-a").glob("youtube_uploads.json"))
    rig.launch_ok = True
    _tick_at(rig, NOW + timedelta(days=2), jobs_dir=rig.tmp / "jobs")
    assert len(_named(rig)) == 1, [a[a.index("--name") + 1] for a, _c, _e in rig.launches]


def test_a_raise_after_the_launcher_ok_with_the_marker_left_never_falls_through(tmp_path, monkeypatch):  # fmt: skip
    """The launch sink alone must tell "launched" from "no launch": the marker is still there."""
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    monkeypatch.setattr(launch, "clear_replace_retry", lambda *a, **k: None)

    def run(sup, rep):
        rig.supersede, rig.replace_scheduled = sup, rep
        rig.run()
        raise OSError("post-launch hiccup")

    res = launch.run_tick(stages_root=rig.stages, output_root=rig.out, now=NOW, region="usa", regions={}, dry_run=False, supersede=None, replace=False, run=run)  # fmt: skip
    assert res["action"] == "launched" and res["project"] == "day-a" and "replace_retry" not in res
    assert len(rig.launches) == 1  # no plain tick behind the drive just launched


def test_a_stamp_plus_another_projects_job_after_it_is_retried_not_cleared(tmp_path):
    rig = _replace_rig(tmp_path, _three_lfs())
    _marker(rig)
    launch.stamp_replace_launch(rig.out, "day-a", NOW.timestamp())  # died before the launcher ran
    job(rig, "other", "day-b", "running", NOW.timestamp() + 5)  # ANOTHER project's job, later
    res = _tick_at(rig, NOW + timedelta(days=5), jobs_dir=rig.tmp / "jobs")
    assert (res["action"], res["project"]) == ("launched", "day-a"), res
    assert res.get("replace_retry", {}).get("outcome") != "cleared" and len(_named(rig)) == 1
