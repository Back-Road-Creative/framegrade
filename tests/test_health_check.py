"""``gomoveshift health check``: detection, idempotent push, never raises."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from gomoveshift.health import cadence, run_check
from gomoveshift.health import check as check_mod
from gomoveshift.health.drives import parse_log
from gomoveshift.health.store import load_all
from gomoveshift.publishing.queue import open_queue_db

NOW = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)


def _job(
    jobs: Path,
    run_id: str,
    name: str,
    log: Path,
    *,
    state="failed",
    rc=1,
    created=1000.0,
    argv=("gomoveshift", "drive", "/in/demo"),
):
    d = jobs / run_id
    d.mkdir(parents=True)
    (d / "job.json").write_text(
        json.dumps(
            {
                "schema": "job/1",
                "run_id": run_id,
                "name": name,
                "argv": list(argv),
                "created_at": created,
                "expect": [str(log)],
                "result": {"state": state, "exit_code": rc, "ended_at": created + 60},
            }
        )
    )


def _log(path: Path, *lines: str) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


def _run(tmp_path, notify, **kw):
    cfg = SimpleNamespace(seo=SimpleNamespace(storage_path="relative/seo"))
    return run_check(
        cfg,
        incidents_dir=tmp_path / "inc",
        jobs_dir=tmp_path / "jobs",
        code_root=tmp_path,
        staging_sha="abc123",
        now=NOW,
        notify_fn=notify,
        **kw,
    )


def _failed_drive(tmp_path):
    log = _log(
        tmp_path / "a.log",
        "project_name=demo-drive-a",
        "running phase_d",
        "Audio loudness HARD FAIL - too loud",
        "2026-01-09 ERROR Drive pipeline failed: Audio loudness HARD FAIL",
    )
    _job(tmp_path / "jobs", "r1", "drive-demo-drive-a", log)


def test_parse_log_reads_drive_phase_gate_error():
    got = parse_log(
        [
            "project_name=demo-drive-a",
            "phase_d start",
            "Audio loudness HARD FAIL x",
            "T ERROR Drive pipeline failed: boom",
        ]
    )
    assert got == {
        "drive": "demo-drive-a",
        "phase": "phase_d",
        "gate": "Audio loudness",
        "error": "Drive pipeline failed: boom",
    }


def test_failed_drive_writes_incident_row_and_pushes_once(tmp_path):
    _failed_drive(tmp_path)
    sent = []
    first = _run(tmp_path, lambda e, i, s: sent.append((e, s)) or True)
    row = json.loads((tmp_path / "inc" / "demo-drive-a.json").read_text())
    assert row["kind"] == "drive_failed" and row["drive"] == "demo-drive-a"
    assert row["phase"] == "phase_d" and row["gate"] == "Audio loudness"
    assert row["job_run_id"] == "r1" and row["staging_sha"] == "abc123"
    assert row["resume_attempts"] == [] and row["log_path"].endswith("a.log")
    assert row["error"].startswith("Drive pipeline failed:")
    assert len(sent) == 1 and sent[0][0] == "health_drive_failed"
    assert len(first["incidents"]) == 1
    _run(tmp_path, lambda e, i, s: sent.append((e, s)) or True)
    assert len(sent) == 1  # idempotent


def test_failed_push_stays_unrecorded_and_retries(tmp_path):
    _failed_drive(tmp_path)
    calls = []
    _run(tmp_path, lambda *a: (calls.append(a), False)[1])
    row = json.loads((tmp_path / "inc" / "demo-drive-a.json").read_text())
    assert row["pushed"] is False
    _run(tmp_path, lambda *a: (calls.append(a), True)[1])
    assert len(calls) == 2
    assert json.loads((tmp_path / "inc" / "demo-drive-a.json").read_text())["pushed"] is True


def test_nonzero_exit_needs_marker_when_state_not_failed(tmp_path):
    clean = _log(tmp_path / "c.log", "all fine")
    _job(tmp_path / "jobs", "r2", "drive-b", clean, state="exited", rc=2)
    assert _run(tmp_path, lambda *a: True)["incidents"] == []
    marked = _log(tmp_path / "m.log", "Drive pipeline failed: nope")
    _job(tmp_path / "jobs", "r3", "drive-c", marked, state="exited", rc=2, created=2000.0)
    assert len(_run(tmp_path, lambda *a: True)["incidents"]) == 1


def test_later_successful_run_resolves_incident(tmp_path):
    _failed_drive(tmp_path)
    _run(tmp_path, lambda *a: True)
    ok = _log(tmp_path / "ok.log", "done")
    _job(tmp_path / "jobs", "r9", "drive-demo-drive-a", ok, state="succeeded", rc=0, created=5000.0)
    out = _run(tmp_path, lambda *a: True)
    assert out["incidents"] == [] and len(out["resolved"]) == 1
    rows, _ = load_all(tmp_path / "inc")
    assert rows[0]["resolved_at"]


def test_unreadable_records_are_counted_not_raised(tmp_path):
    (tmp_path / "jobs" / "bad").mkdir(parents=True)
    (tmp_path / "jobs" / "bad" / "job.json").write_text("{not json")
    (tmp_path / "inc").mkdir()
    (tmp_path / "inc" / "x.json").write_text("[]")
    out = _run(tmp_path, lambda *a: True)
    assert out["unreadable"] == 2 and out["incidents"] == []
    assert any("relative" in n for n in out["notes"])


def test_unbooked_slot_needs_ready_drive_and_48h_window():
    report = {
        "next_available": [
            {"region": "r1", "day": "Sat", "slot": (NOW + timedelta(hours=30)).isoformat()},
            {"region": "r1", "day": "Sun", "slot": (NOW + timedelta(hours=60)).isoformat()},
            {"region": "r2", "day": "Sat", "slot": (NOW + timedelta(hours=10)).isoformat()},
        ]
    }
    rows = cadence.unbooked_slots(report, {"r1": 1, "r2": 0}, NOW, NOW.isoformat())
    assert [r["region"] for r in rows] == ["r1"]
    assert rows[0]["kind"] == "lf_slot_unbooked" and rows[0]["file"].startswith("slot_r1_")


def test_uncovered_slot_pages_an_empty_region_and_names_the_next_drive():
    """Rule 2 only pages a region that HAS a ready drive; a region with none left is the
    worse gap (an audit takes days), so it pages a week out and names the backlog row."""
    report = {
        "next_available": [
            {"region": "usa", "day": "Fri", "slot": (NOW + timedelta(hours=80)).isoformat()},
            {"region": "usa", "day": "Tue", "slot": (NOW + timedelta(hours=150)).isoformat()},
            {"region": "ireland", "day": "Sun", "slot": (NOW + timedelta(hours=20)).isoformat()},
            {"region": "far", "day": "Sun", "slot": (NOW + timedelta(hours=200)).isoformat()},
        ]
    }
    rows = cadence.uncovered_slots(
        report,
        {"usa": 0, "ireland": 1, "far": 0},
        {"usa": "sc-2026-02-21-ace-basin", "far": None},
        NOW,
        NOW.isoformat(),
    )
    assert [(r["region"], r["next_drive"]) for r in rows] == [("usa", "sc-2026-02-21-ace-basin")]
    assert rows[0]["kind"] == "lf_slot_uncovered"
    assert rows[0]["slot"] == (NOW + timedelta(hours=80)).isoformat()


def test_uncovered_slot_with_no_backlog_row_says_so():
    report = {"next_available": [{"region": "usa", "slot": (NOW + timedelta(hours=5)).isoformat()}]}
    rows = cadence.uncovered_slots(report, {"usa": 0}, {}, NOW, NOW.isoformat())
    assert rows[0]["next_drive"] is None
    assert "no usa row" in check_mod._line(rows[0])


def _queue(tmp_path, *, updated_hours_ago, scheduled="2026-01-01T00:00:00+00:00", paused=False):
    conn = open_queue_db(tmp_path / "q")
    with conn:
        conn.execute(
            "INSERT INTO deliveries (project_name, short_index, platform, plan_hash, status,"
            " scheduled_at, created_at, updated_at) VALUES ('p', 1, 'youtube', 'h', 'pending', ?, ?, ?)",
            (scheduled, NOW.isoformat(), (NOW - timedelta(hours=updated_hours_ago)).isoformat()),
        )
        if paused:
            conn.execute(
                "INSERT INTO platform_pauses (platform, paused_at) VALUES ('youtube', 'x')"
            )
    conn.close()
    return tmp_path / "q" / "social_queue.sqlite3"


def test_queue_stalled_when_due_work_and_quiet(tmp_path):
    row = cadence.queue_stalled(_queue(tmp_path, updated_hours_ago=30), NOW, NOW.isoformat())
    assert row["kind"] == "queue_stalled" and row["due_deliveries"] == 1


def test_queue_not_stalled_when_recent_future_or_paused(tmp_path):
    assert cadence.queue_stalled(_queue(tmp_path, updated_hours_ago=2), NOW, "t") is None


def test_queue_future_work_is_not_due(tmp_path):
    db = _queue(tmp_path, updated_hours_ago=30, scheduled=(NOW + timedelta(days=1)).isoformat())
    assert cadence.queue_stalled(db, NOW, "t") is None


def test_queue_paused_platform_is_not_due(tmp_path):
    assert (
        cadence.queue_stalled(_queue(tmp_path, updated_hours_ago=30, paused=True), NOW, "t") is None
    )


def test_launcher_run_drive_is_detected_from_its_log(tmp_path):
    # Real drives run through a launcher script: neither argv nor the job name says "drive".
    log = _log(
        tmp_path / "demo-LIVE.log",
        "2026-01-09 INFO gomoveshift.pipeline.drive_orchestrator   Phase A start",
        "2026-01-09 ERROR gomoveshift.pipeline.drive_orchestrator   Drive pipeline failed: boom",
    )
    _job(tmp_path / "jobs", "r1", "demo-sun-audit", log, argv=("bash", "/x/launch.sh", "audit"))
    pushes = []
    _run(tmp_path, lambda *a: pushes.append(a) or True)
    rows, _ = load_all(tmp_path / "inc")
    assert [r["job_name"] for r in rows if r["kind"] == "drive_failed"] == ["demo-sun-audit"]


def test_failed_non_drive_job_raises_no_incident(tmp_path):
    log = _log(tmp_path / "list.log", "listing LFs", "Traceback: boom")
    _job(tmp_path / "jobs", "r1", "demo-listlfs", log, argv=("bash", "/x/list.sh"))
    _run(tmp_path, lambda *a: True)
    rows, _ = load_all(tmp_path / "inc")
    assert [r for r in rows if r["kind"] == "drive_failed"] == []


def test_incident_row_records_the_resume_recipe_path_from_the_drive_log(tmp_path):
    log = _log(
        tmp_path / "a.log",
        "project_name=demo-drive-a",
        f"INFO resume recipe: {tmp_path}/stages/resume.json",
        "ERROR Drive pipeline failed: boom",
    )
    _job(tmp_path / "jobs", "r1", "drive-demo-drive-a", log)
    out = _run(tmp_path, lambda *a: True)
    assert out["incidents"][0]["resume_recipe"] == f"{tmp_path}/stages/resume.json"


def test_row_without_a_recipe_line_records_none(tmp_path):
    _failed_drive(tmp_path)
    assert _run(tmp_path, lambda *a: True)["incidents"][0]["resume_recipe"] is None


def _recovered_pair(tmp_path, *, share):
    """A failed audit and a LATER succeeded publish under a DIFFERENT job name."""
    recipe = f"INFO resume recipe: {tmp_path}/stages/resume.json"
    fail_log = _log(
        tmp_path / "audit.log", "project_name=demo-drive-a", recipe, "Drive pipeline failed: x"
    )
    ok_log = _log(
        tmp_path / ("audit.log" if share == "log" else "publish.log"),
        "project_name=demo-drive-a",
        *([recipe] if share == "stages" else []),
        "all done",
    )
    _job(tmp_path / "jobs", "r1", "gms-fri-1002-audit", fail_log, created=1000.0)
    _job(
        tmp_path / "jobs",
        "r2",
        "gms-fri-1002-publish",
        ok_log,
        state="succeeded",
        rc=0,
        created=2000.0,
    )


def test_later_success_under_another_name_resolves_via_shared_stages_dir(tmp_path):
    _recovered_pair(tmp_path, share="stages")
    assert _run(tmp_path, lambda *a: True)["incidents"] == []


def test_later_success_under_another_name_resolves_via_shared_log(tmp_path):
    _recovered_pair(tmp_path, share="log")
    assert _run(tmp_path, lambda *a: True)["incidents"] == []


def test_unrelated_later_success_does_not_resolve(tmp_path):
    _recovered_pair(tmp_path, share="none")
    assert len(_run(tmp_path, lambda *a: True)["incidents"]) == 1


def test_an_earlier_success_does_not_resolve_a_later_failure(tmp_path):
    _recovered_pair(tmp_path, share="stages")
    (tmp_path / "jobs" / "r2" / "job.json").write_text(
        (tmp_path / "jobs" / "r2" / "job.json").read_text().replace("2000.0", "500.0")
    )
    assert len(_run(tmp_path, lambda *a: True)["incidents"]) == 1
