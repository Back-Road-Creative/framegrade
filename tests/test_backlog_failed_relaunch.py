"""``health backlog`` relaunches a failed drive from source once its failure fingerprint changes
(rec0060-0061, 2026-11-20: store row failed since 11-07, stage dir holding only calibration.json,
no resume.json, so ``health heal`` could never reach it)."""

from __future__ import annotations

import json

import pytest
from test_health_backlog_launch import STAMP, NOW, Rig, drive, job, uploaded

from gomoveshift.health import launch, retry_policy
from gomoveshift.health.store import load_all, new_incident, write_row

OLD = {
    "versions": {"gomoveshift": "0.3.25"},
    "manifest_sha": "m1",
    "config_sha": "c1",
    "phase": "x",
}
NEW = {**OLD, "manifest_sha": "m2"}


def _relaunch(rig, now_fp, *, dry_run=False):
    rig.incidents = rig.tmp / "incidents"
    rig.relaunch = launch.FailedRelaunch(
        rig.incidents, rig.stages, fingerprint_fn=lambda project, phase: now_fp,
        now=NOW, dry_run=dry_run,
    )  # fmt: skip
    return rig.relaunch


def _failed_row(rig, stamped=OLD, *, klass="deterministic", status="failed", project="day-a"):
    """A store row ``status`` whose stage dir holds only calibration.json, and the incident row
    ``health check`` filed for it (``stamped``: its failure fingerprint, None = legacy)."""
    rig.drives.append(drive(project, status))
    (rig.stages / project / project / "_stages").mkdir(parents=True)
    (rig.stages / project / project / "_stages" / "calibration.json").write_text("{}")
    (rig.out / project).mkdir(exist_ok=True)
    rig.incidents = rig.tmp / "incidents"
    row = new_incident(
        "drive_failed", f"drive_failed:{project}:r1", f"{project}.json", "2026-11-07T10:00:00Z",
        drive=project, phase="x", error="BLUR GATE", resume_recipe=None, resume_attempts=[],
    )  # fmt: skip
    if stamped:
        row["failure_fingerprint"] = {
            "hash": retry_policy.fingerprint_hash(stamped), "fingerprint": stamped, "class": klass,
        }  # fmt: skip
    write_row(rig.incidents, row)


def _attempts(rig):
    rows, _ = load_all(rig.incidents)
    return [a for r in rows if r["kind"] == launch.RELAUNCH_KIND for a in r["resume_attempts"]]


def test_a_failed_row_with_a_changed_fingerprint_relaunches_once(tmp_path):
    rig = Rig(tmp_path)
    _failed_row(rig)
    _relaunch(rig, NEW)
    res = rig.run()
    assert (res["action"], res["project"]) == ("launched", "day-a"), res
    assert len(rig.launches) == 1
    assert (rig.stages / launch.SUPERSEDED_DIR / f"day-a-{STAMP}").is_dir()
    assert not (rig.stages / "day-a").exists()
    assert [r["status"] for r in rig.registered] == ["auditing"]
    assert [a["action"] for a in _attempts(rig)] == ["launched"]


def test_a_second_tick_after_the_relaunch_launches_nothing(tmp_path):
    rig = Rig(tmp_path)
    rig.rows = rig.rows[:1]  # only day-a: nothing else may launch
    _failed_row(rig)
    _relaunch(rig, NEW)
    rig.run()
    rig.drives[:] = [drive("day-a", "failed")]  # it failed again, under the same code
    rig.launches.clear()
    (rig.stages / "day-a" / "day-a" / "_stages").mkdir(parents=True)
    res = rig.run()
    assert not rig.launches and res["action"] != "launched", res


def test_the_same_fingerprint_launches_nothing_and_records_one_refusal(tmp_path):
    rig = Rig(tmp_path)
    rig.rows = rig.rows[:1]  # only day-a: nothing else may launch
    _failed_row(rig)
    _relaunch(rig, OLD)
    for _ in range(3):
        res = rig.run()
        assert res["action"] != "launched", res
    assert not rig.launches and (rig.stages / "day-a").is_dir()
    got = _attempts(rig)
    assert [a["action"] for a in got] == ["refused"], got  # deduped across ticks
    assert "deterministic" in got[0]["reason"]


def test_a_legacy_unstamped_failed_row_gets_one_launch(tmp_path):
    rig = Rig(tmp_path)
    _failed_row(rig, stamped=None)
    _relaunch(rig, NEW)
    res = rig.run()
    assert res["action"] == "launched" and len(rig.launches) == 1, res


def test_a_failed_row_with_no_incident_row_gets_one_launch(tmp_path):
    rig = Rig(tmp_path)
    _failed_row(rig, stamped=None)
    for p in rig.incidents.glob("*.json"):
        p.unlink()
    _relaunch(rig, NEW)
    assert rig.run()["action"] == "launched"


@pytest.mark.parametrize("status", ["published", "booked", "ready"])
def test_a_published_booked_or_ready_row_is_untouched(tmp_path, status):
    rig = Rig(tmp_path)
    _failed_row(rig, status=status)
    _relaunch(rig, NEW)
    rig.run()
    assert (rig.stages / "day-a").is_dir() and not (rig.stages / launch.SUPERSEDED_DIR).exists()
    assert all(r["project"] != "day-a" for r in rig.registered)


def test_a_failed_row_with_an_upload_receipt_is_untouched(tmp_path):
    rig = Rig(tmp_path)
    rig.rows = rig.rows[:1]  # only day-a: nothing else may launch
    _failed_row(rig)
    rig.drives[-1].output_dir = uploaded(tmp_path, "pub", {"lf1": {"video_id": "abc"}})
    _relaunch(rig, NEW)
    rig.run()
    assert (rig.stages / "day-a").is_dir() and not rig.launches


def test_a_failed_row_with_a_readable_resume_recipe_is_heals_to_resume(tmp_path):
    rig = Rig(tmp_path)
    rig.rows = rig.rows[:1]  # only day-a: nothing else may launch
    _failed_row(rig)
    stages = rig.stages / "day-a" / "day-a" / "_stages"
    (stages / "resume.json").write_text(json.dumps({"argv": ["/py", "-m", "x"], "cwd": "/w"}))
    _relaunch(rig, NEW)
    res = rig.run()
    assert res["action"] != "launched" and not rig.launches, res


def test_an_auditing_row_with_a_failed_newest_audit_and_a_relaunch_due_launches(tmp_path):
    rig = Rig(tmp_path)
    _failed_row(rig, status="auditing")
    job(rig, "dead1", "day-a", "failed", 100.0)
    _relaunch(rig, NEW)
    res = rig.run()
    assert (res["action"], res["project"]) == ("launched", "day-a"), res
    assert "failed_audits" not in res and len(rig.launches) == 1
    assert (rig.stages / launch.SUPERSEDED_DIR / f"day-a-{STAMP}").is_dir()


def test_an_auditing_row_with_no_job_record_and_a_relaunch_due_launches(tmp_path):
    rig = Rig(tmp_path)
    _failed_row(rig, status="auditing")
    _relaunch(rig, NEW)
    res = rig.run()
    assert res["action"] == "launched" and "audit in progress" not in res.get("reason", "")


def test_an_auditing_row_with_a_running_audit_is_never_relaunched(tmp_path):
    rig = Rig(tmp_path)
    _failed_row(rig, status="auditing")
    job(rig, "live", "day-a", "running", 100.0)
    _relaunch(rig, NEW)
    res = rig.run()
    assert "audit in progress" in res["reason"] and not rig.launches


def test_an_auditing_row_with_a_failed_audit_and_the_same_fingerprint_is_still_listed(tmp_path):
    rig = Rig(tmp_path)
    _failed_row(rig, status="auditing")
    job(rig, "dead1", "day-a", "failed", 100.0)
    _relaunch(rig, OLD)
    res = rig.run()
    assert res["failed_audits"][0]["project"] == "day-a"
    assert (res["action"], res["project"]) == ("launched", "day-b"), res


def test_dry_run_moves_and_writes_nothing(tmp_path):
    rig = Rig(tmp_path)
    _failed_row(rig)
    _relaunch(rig, NEW, dry_run=True)
    before = sorted(p.name for p in rig.incidents.glob("*"))
    res = rig.run(dry_run=True)
    assert res["action"] == "would_launch" and res["project"] == "day-a", res
    assert (rig.stages / "day-a").is_dir() and not (rig.stages / launch.SUPERSEDED_DIR).exists()
    assert not rig.launches and not rig.registered
    assert sorted(p.name for p in rig.incidents.glob("*")) == before


def test_a_dry_run_refusal_writes_no_record(tmp_path):
    rig = Rig(tmp_path)
    _failed_row(rig)
    _relaunch(rig, OLD, dry_run=True)
    rig.run(dry_run=True)
    assert _attempts(rig) == []


def test_a_running_drive_blocks_the_relaunch(tmp_path):
    rig = Rig(tmp_path)
    _failed_row(rig)
    _relaunch(rig, NEW)
    rig.running = True
    rig.run()
    assert (rig.stages / "day-a").is_dir() and not rig.launches
