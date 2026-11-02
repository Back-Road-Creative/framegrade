"""``health backlog``: launch the next ``drive_backlog.yaml`` row for a region's uncovered LF
slot, audit only. Every side effect (lease, process probe, disk, preflight, launcher, store
write, clock) is injected; nothing here launches a real job."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
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
        self.installed = "0.3.25"
        self.latest = "0.3.25"

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
            launcher=self.launcher,
            register=lambda **kw: self.registered.append(kw),
            installed_version=lambda python: self.installed,
            latest_version=lambda: self.latest,
        )  # fmt: skip

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


def drive(project, status, region="usa"):
    return SimpleNamespace(project=project, region=region, status=status)


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


def job(rig, run_id, project, state, created_at, exit_code=1):
    """A job-runner record for the backlog drive of ``project``."""
    d = rig.tmp / "jobs" / run_id
    d.mkdir(parents=True)
    result = {"state": state, "exit_code": None if state == "running" else exit_code}
    rec = {"schema": "job/1", "run_id": run_id, "name": f"gms-backlog-{project}",
           "created_at": created_at, "result": result}  # fmt: skip
    (d / "job.json").write_text(json.dumps(rec))


def test_an_audit_whose_latest_job_failed_is_refused_naming_the_job(tmp_path):
    rig = Rig(tmp_path)
    rig.drives.append(drive("day-a", "auditing"))
    job(rig, "fbc9336e", "day-a", "failed", 100.0)
    res = rig.run()
    assert res["action"] == "refused"
    assert "day-a" in res["reason"] and "fbc9336e" in res["reason"] and "exit 1" in res["reason"]
    assert not rig.launches and not rig.registered
    assert launch.exit_code(res) == 2


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
    assert need == pytest.approx(30.0)  # 10 GB x 1.5 = 15 < the 30 GB floor
    rig.devices[rig.stages] = (1, need if action == "launched" else need - 0.01)
    res = rig.run()
    assert res["action"] == action, res
    if action == "refused":
        assert "storage" in res["reason"] and not rig.launches and not rig.sources.exists()


def test_staging_need_scales_with_day_source_above_the_floor(tmp_path):
    rig = Rig(tmp_path)
    row = rig.row("big", 40)
    assert launch.staging_need_gb(row.sources) == pytest.approx(60.0)


def test_shared_device_sums_staging_and_output_needs(tmp_path):
    rig = Rig(tmp_path)
    rig.rows = [rig.row("day-a", 40)]  # staging 60 + output floor 50 = 110 on one device
    rig.devices = {rig.stages: (7, 109.0), rig.out: (7, 109.0)}
    res = rig.run()
    assert res["action"] == "refused" and "110.0" in res["reason"]
    storage = res["storage"]
    assert len(storage) == 1 and storage[0]["need_gb"] == pytest.approx(110.0)
    rig.devices = {rig.stages: (7, 110.0), rig.out: (7, 110.0)}
    assert rig.run()["action"] == "launched"


def test_distinct_devices_are_checked_separately(tmp_path):
    rig = Rig(tmp_path)
    rig.rows = [rig.row("day-a", 40)]
    rig.devices = {rig.stages: (1, 60.0), rig.out: (2, 49.0)}
    res = rig.run()
    assert res["action"] == "refused"
    by_dev = {s["device"]: s for s in res["storage"]}
    assert by_dev[1]["ok"] and not by_dev[2]["ok"]


def test_storage_is_reported_on_dry_run_with_the_bound_named(tmp_path):
    res = Rig(tmp_path).run(dry_run=True)
    assert res["storage"] and "1.5" in res["storage_bound"]


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
            drive_running=boom, disk=rig.disk, preflight=rig.preflight, launcher=rig.launcher,
            register=lambda **kw: None,
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
