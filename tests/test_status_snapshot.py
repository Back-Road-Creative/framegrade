"""``gomoveshift status``: one read-only snapshot of everything live.

Each section names where it read from and when; a store it cannot read says
``unavailable`` with the reason instead of reading as empty or healthy. Nothing
here writes: the queue is opened read-only and a missing file is never created.
All data is synthetic.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from gomoveshift.cli.handlers.status import cmd_status
from gomoveshift.publishing.queue import enqueue, open_queue_db
from gomoveshift.status import snapshot

NOW = "2026-11-13T18:00:00+00:00"
SECTIONS = ("jobs", "incidents", "queue", "runway", "backlog", "machine")


def _config(tmp_path):
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path / "seo")))


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Every store pointed at an empty tmp dir, so no test reads the real box."""
    monkeypatch.setenv("GMS_DATA_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("GMS_INPUTS_DIR", str(tmp_path / "inputs"))
    monkeypatch.setenv("GMS_STATUS_GPU_LOCK_PATH", str(tmp_path / "gpu.lock"))
    for name in ("GMS_JOBS_DIR", "GMS_HEALTH_INCIDENTS_DIR", "GMS_CRON_LOGS_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GMS_STATUS_VOLUMES", f"Scratch={tmp_path}")
    monkeypatch.setattr(snapshot, "_runway_report", lambda config: {"queue_ends": None})
    return tmp_path


def _job(jobs_dir, name, *, state=None, exit_code=None, created=1794600000.0, log=None):
    job_dir = jobs_dir / name
    job_dir.mkdir(parents=True)
    rec = {
        "schema": "job/1",
        "name": name,
        "run_id": f"run-{name}",
        "argv": ["python", "-m", "gomoveshift", "drive"],
        "created_at": created,
        "heartbeat_at": created + 60,
        "expect": [],
        "result": {"state": state, "exit_code": exit_code, "ended_at": created + 90}
        if state
        else None,
    }
    (job_dir / "job.json").write_text(json.dumps(rec))
    if log is not None:
        (job_dir / "output.log").write_text(log)
    return job_dir


def test_every_section_says_where_and_when_it_read(env):
    snap = snapshot.build_snapshot(_config(env), now=NOW)
    assert snap["schema"] == "gms-status/1" and snap["as_of"] == NOW
    for name in SECTIONS:
        assert snap[name]["as_of"] == NOW, name
        assert snap[name]["source"], name


def test_a_missing_store_is_unavailable_never_empty(env):
    snap = snapshot.build_snapshot(_config(env), now=NOW)
    for name in ("jobs", "incidents", "queue", "backlog"):
        assert "unavailable" in snap[name], name
    assert "running" not in snap["jobs"] and "counts" not in snap["queue"]


def test_one_broken_reader_does_not_take_the_others_down(env, monkeypatch):
    def boom(config):
        raise RuntimeError("ledger locked")

    monkeypatch.setattr(snapshot, "_runway_report", boom)
    snap = snapshot.build_snapshot(_config(env), now=NOW)
    assert snap["runway"]["unavailable"] == "RuntimeError: ledger locked"
    assert "volumes" in snap["machine"]


def test_jobs_split_running_from_ended_and_name_the_drive(env, monkeypatch):
    jobs = env / "jobs"
    monkeypatch.setenv("GMS_JOBS_DIR", str(jobs))
    _job(jobs, "live", log="project_name=synth-coast\nrunning phase_bc_day_first\n")
    _job(jobs, "old-fail", state="failed", exit_code=1, created=1794500000.0)
    _job(jobs, "old-ok", state="succeeded", exit_code=0, created=1794400000.0)
    (jobs / "junk").mkdir()
    (jobs / "junk" / "job.json").write_text("{not json")

    section = snapshot.build_snapshot(_config(env), now=NOW)["jobs"]

    assert section["source"] == str(jobs)
    assert [j["name"] for j in section["running"]] == ["live"]
    assert section["running"][0]["drive"] == "synth-coast"
    assert section["running"][0]["phase"] == "phase_bc_day_first"
    assert [j["name"] for j in section["recent"]] == ["old-fail", "old-ok"]
    assert section["recent"][0]["state"] == "failed" and section["recent"][0]["exit_code"] == 1
    assert section["counts"] == {"running": 1, "failed": 1, "succeeded": 1}
    assert section["unreadable"] == 1


def test_jobs_dir_defaults_under_the_data_root(env):
    jobs = env / "root" / "logs" / "jobs"
    _job(jobs, "only", state="succeeded", exit_code=0)
    section = snapshot.build_snapshot(_config(env), now=NOW)["jobs"]
    assert section["source"] == str(jobs) and section["counts"] == {"succeeded": 1}


def test_incidents_list_only_the_open_ones(env, monkeypatch):
    inc = env / "incidents"
    inc.mkdir()
    monkeypatch.setenv("GMS_HEALTH_INCIDENTS_DIR", str(inc))
    base = {"schema": "health-incident/1", "kind": "drive_failed", "detected_at": NOW}
    (inc / "a.json").write_text(
        json.dumps({**base, "key": "k-open", "file": "a.json", "resolved_at": None, "error": "x"})
    )
    (inc / "b.json").write_text(
        json.dumps({**base, "key": "k-done", "file": "b.json", "resolved_at": NOW})
    )
    (inc / "c.json").write_text("nope")

    section = snapshot.build_snapshot(_config(env), now=NOW)["incidents"]

    assert [row["key"] for row in section["open"]] == ["k-open"]
    assert section["resolved"] == 1 and section["unreadable"] == 1


def _queue(tmp_path):
    conn = open_queue_db(tmp_path / "seo")
    rows = [
        ("synth-a", 1, "youtube", "queued", "2026-11-14T15:00:00+00:00"),
        ("synth-a", 2, "youtube", "queued", "2026-11-15T15:00:00+00:00"),
        ("synth-a", 1, "tiktok", "awaiting_manual_publish", "2026-11-10T15:00:00+00:00"),
        ("synth-a", 2, "tiktok", "awaiting_manual_publish", "2026-11-13T17:00:00+00:00"),
        ("synth-a", 1, "facebook", "published", "2026-11-01T15:00:00+00:00"),
    ]
    for project, index, platform, status, when in rows:
        delivery_id = enqueue(conn, project, index, platform, "HASH0")
        conn.execute(
            "UPDATE deliveries SET status=?, scheduled_at=? WHERE id=?",
            (status, when, delivery_id),
        )
    conn.commit()
    conn.close()


def test_queue_counts_by_platform_overdue_and_next_up(env):
    _queue(env)
    section = snapshot.build_snapshot(_config(env), now=NOW)["queue"]

    assert section["counts"] == {
        "facebook": {"published": 1},
        "tiktok": {"awaiting_manual_publish": 2},
        "youtube": {"queued": 2},
    }
    assert [(r["platform"], r["short_index"]) for r in section["overdue_manual"]] == [("tiktok", 1)]
    assert [r["scheduled_at"][:10] for r in section["next"]] == ["2026-11-14", "2026-11-15"]


def test_reading_the_queue_changes_nothing_on_disk(env):
    _queue(env)
    db = env / "seo" / "social_queue.sqlite3"
    os.chmod(db, 0o444)
    before = (db.stat().st_mtime_ns, db.read_bytes())
    section = snapshot.build_snapshot(_config(env), now=NOW)["queue"]
    assert "unavailable" not in section
    assert (db.stat().st_mtime_ns, db.read_bytes()) == before


def test_a_missing_queue_file_is_not_created(env):
    snapshot.build_snapshot(_config(env), now=NOW)
    assert not (env / "seo").exists()


def test_runway_carries_the_schedule_report(env, monkeypatch):
    report = {"queue_ends": "2026-12-20", "scheduled_longforms": [], "scheduled_shorts_count": 4}
    monkeypatch.setattr(snapshot, "_runway_report", lambda config: report)
    section = snapshot.build_snapshot(_config(env), now=NOW)["runway"]
    assert section["queue_ends"] == "2026-12-20" and section["scheduled_shorts_count"] == 4
    assert section["source"] == "gomoveshift schedule --json"


def test_runway_counts_days_from_the_reports_own_today(env, monkeypatch):
    report = {"as_of": "2026-11-13T13:00:00-05:00", "queue_ends": "2026-12-20"}
    monkeypatch.setattr(snapshot, "_runway_report", lambda config: dict(report))
    assert snapshot.build_snapshot(_config(env), now=NOW)["runway"]["days_of_runway"] == 37

    report["queue_ends"] = None
    assert snapshot.build_snapshot(_config(env), now=NOW)["runway"]["days_of_runway"] is None


def test_backlog_counts_rows_by_region(env):
    path = env / "inputs" / "gomoveshift" / "console" / "brands" / "gomoveshift"
    path.mkdir(parents=True)
    (path / "drive_backlog.yaml").write_text(
        "- {project: synth-1, region: usa}\n"
        "- {project: synth-2, region: usa}\n"
        "- {project: synth-3, region: ireland}\n"
    )
    section = snapshot.build_snapshot(_config(env), now=NOW)["backlog"]
    assert section["rows"] == 3 and section["by_region"] == {"ireland": 1, "usa": 2}
    assert section["projects"][0] == {"project": "synth-1", "region": "usa"}


def test_machine_reports_free_space_and_an_unreadable_volume(env, monkeypatch):
    monkeypatch.setenv("GMS_STATUS_VOLUMES", f"C:={env},D:={env / 'absent'}")
    volumes = snapshot.build_snapshot(_config(env), now=NOW)["machine"]["volumes"]
    assert volumes[0]["label"] == "C:" and volumes[0]["free_bytes"] > 0
    assert volumes[0]["total_bytes"] >= volumes[0]["free_bytes"]
    assert volumes[1]["label"] == "D:" and "unavailable" in volumes[1]


def test_machine_names_the_gpu_holder_or_says_free(env):
    lock = env / "gpu.lock"
    assert "unavailable" in snapshot.build_snapshot(_config(env), now=NOW)["machine"]["gpu"]
    lock.write_text("{}")
    assert snapshot.build_snapshot(_config(env), now=NOW)["machine"]["gpu"]["held"] is False
    lock.write_text(json.dumps({"owner": "gomoveshift", "pid": 7, "acquired": NOW}))
    gpu = snapshot.build_snapshot(_config(env), now=NOW)["machine"]["gpu"]
    assert gpu["held"] is True and gpu["owner"] == "gomoveshift" and gpu["pid"] == 7


def test_machine_reads_each_cron_log_last_line_and_last_failure(env, monkeypatch):
    logs = env / "cron"
    logs.mkdir()
    monkeypatch.setenv("GMS_CRON_LOGS_DIR", str(logs))
    (logs / "gms-health-cron.log").write_text(
        "[2026-11-13T17:45:14] FAILED: backlog usa exit 2\n"
        "[2026-11-13T17:45:16] backlog ireland: exit 0\n"
    )
    (logs / "gms-journal-cron.log").write_text("[2026-11-13T18:17:12] nothing to publish\n")
    (logs / "unrelated.log").write_text("x\n")

    crons = snapshot.build_snapshot(_config(env), now=NOW)["machine"]["cron"]

    assert [c["name"] for c in crons] == ["gms-health", "gms-journal"]
    assert crons[0]["last_at"] == "2026-11-13T17:45:16"
    assert crons[0]["last_line"] == "backlog ireland: exit 0"
    assert crons[0]["last_failure"] == {
        "at": "2026-11-13T17:45:14",
        "line": "FAILED: backlog usa exit 2",
    }
    assert crons[1]["last_failure"] is None


def test_the_command_prints_json_or_a_readable_summary(env, capsys):
    _queue(env)
    assert cmd_status(SimpleNamespace(json=True), _config(env)) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == "gms-status/1"
    assert cmd_status(SimpleNamespace(json=False), _config(env)) == 0
    out = capsys.readouterr().out
    assert "Publishing" in out and "tiktok" in out and "unavailable" in out


def test_status_is_a_registered_subcommand():
    from gomoveshift.cli import _COMMAND_MAP
    from gomoveshift.cli.parser import _build_parser

    args = _build_parser().parse_args(["status", "--json"])
    assert args.json is True and _COMMAND_MAP["status"] is cmd_status


def test_a_missing_ledger_is_unavailable_not_an_empty_schedule(env, monkeypatch):
    from gomoveshift.pipeline import content_ledger

    monkeypatch.undo()
    monkeypatch.setattr(content_ledger, "_default_ledger_path", lambda: env / "absent.json")
    section = snapshot._section(NOW, snapshot.RUNWAY_SOURCE, lambda: snapshot._runway(None))
    assert section["unavailable"] == f"FileNotFoundError: no file {env / 'absent.json'}"
    assert "queue_ends" not in section
