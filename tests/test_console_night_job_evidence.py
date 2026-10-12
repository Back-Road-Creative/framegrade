"""Console restarts recover only trusted night-project worker outcomes."""

import json
import sys
from types import SimpleNamespace

import pytest

from gomoveshift.console import jobs, settings
from gomoveshift.pipeline.photos.night_project import NightStore


@pytest.fixture
def night_job(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "console_data_dir", lambda: tmp_path)
    monkeypatch.setattr(settings, "current_brand", lambda: "brand")
    monkeypatch.setattr(jobs.notify, "notify_job_failed", lambda row: None)
    monkeypatch.setattr(jobs.notify, "notify_job_interrupted", lambda row: None)
    monkeypatch.setattr(jobs, "_PROCESSES", {})
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("{}")
    projects = NightStore(tmp_path / "night-projects")
    project = projects.create("brand", [source], [tmp_path], "natural")
    argv = [
        sys.executable,
        "-m",
        "gomoveshift",
        "photos",
        "night",
        "--project",
        project,
        "--workspace",
        str(projects.root),
        "--brand",
        "brand",
    ]
    store = jobs.JobStore(tmp_path / "jobs.sqlite3")
    identifier = store.create("night", argv, "", brand="brand")
    store.update(identifier, status="running")
    return store, identifier, projects, project


def set_outcome(projects, project, status):
    row = projects.get("brand", project)
    row["status"] = status
    row["items"][0]["status"] = "complete" if status == "complete" else status
    projects.save(row)


@pytest.mark.parametrize(
    "outcome, expected, rc",
    [
        ("complete", "done", 0),
        ("failed", "failed", 1),
        ("partial", "failed", 1),
        ("blocked", "failed", 1),
        ("paused", "paused", 2),
        ("waiting", "waiting", 2),
        ("cancelled", "cancelled", 1),
    ],
)
def test_restart_adopts_durable_outcome(night_job, outcome, expected, rc):
    store, identifier, projects, project = night_job
    set_outcome(projects, project, outcome)
    jobs.sweep_orphaned_jobs(store)
    row = store.get(identifier)
    assert row["status"] == expected
    assert row["rc"] == rc
    assert row["outcome_evidence"] == f"night_project:{outcome}"


@pytest.mark.parametrize("outcome", ["paused", "waiting"])
def test_reaped_rc2_is_not_a_failure(night_job, monkeypatch, outcome):
    store, identifier, projects, project = night_job
    set_outcome(projects, project, outcome)
    jobs._track(store, identifier, SimpleNamespace(poll=lambda: 2))
    failures = []
    monkeypatch.setattr(jobs.notify, "notify_job_failed", failures.append)
    jobs.reap_finished(store)
    assert store.get(identifier)["status"] == outcome
    assert failures == []


def test_reaped_cancel_is_not_reported_as_worker_failure(night_job, monkeypatch):
    store, identifier, projects, project = night_job
    set_outcome(projects, project, "cancelled")
    jobs._track(store, identifier, SimpleNamespace(poll=lambda: 1))
    failures = []
    monkeypatch.setattr(jobs.notify, "notify_job_failed", failures.append)

    jobs.reap_finished(store)

    assert store.get(identifier)["status"] == "cancelled"
    assert failures == []


@pytest.mark.parametrize("change", ["brand", "workspace", "missing", "duplicate", "queued"])
def test_untrusted_or_nonterminal_evidence_never_claims_success(night_job, change):
    store, identifier, projects, project = night_job
    set_outcome(projects, project, "complete")
    argv = store.get(identifier)["argv"]
    if change == "brand":
        argv[-1] = "other-brand"
    elif change == "workspace":
        argv[8] = str(projects.root.parent / "untrusted")
    elif change == "missing":
        argv[6] = "b" * 64
    elif change == "duplicate":
        argv.extend(["--brand", "brand"])
    elif change == "queued":
        set_outcome(projects, project, "queued")
    store.update(identifier, argv=json.dumps(argv))
    jobs.sweep_orphaned_jobs(store)
    assert store.get(identifier)["status"] == "interrupted"


def test_live_worker_is_not_overridden_by_old_project_record(night_job, monkeypatch):
    store, identifier, projects, project = night_job
    set_outcome(projects, project, "complete")
    store.update(identifier, pid=123, started_fingerprint="same")
    monkeypatch.setattr(jobs, "_process_matches", lambda *args: True)
    jobs.sweep_orphaned_jobs(store)
    assert store.get(identifier)["status"] == "running"


def test_watched_worker_exit_uses_durable_outcome(night_job, monkeypatch):
    store, identifier, projects, project = night_job
    set_outcome(projects, project, "complete")
    store.update(identifier, pid=123, started_fingerprint="gone")
    monkeypatch.setattr(jobs, "_process_matches", lambda *args: False)
    jobs.reap_finished(store)
    assert store.get(identifier)["status"] == "done"


def test_foreign_job_brand_never_adopts_current_brand_project(night_job):
    store, identifier, projects, project = night_job
    set_outcome(projects, project, "complete")
    store.update(identifier, brand="other-brand")
    jobs.sweep_orphaned_jobs(store)
    assert store.get(identifier)["status"] == "interrupted"


def test_complete_typed_products_are_durable_job_evidence(night_job):
    store, identifier, projects, project = night_job
    row = projects.get("brand", project)
    row.update(
        schema=2,
        kind="night_products",
        status="complete",
        tasks=[
            {"product": "best_stills", "status": "complete"},
            {"product": "timelapse", "status": "complete"},
        ],
    )
    row.pop("items")
    projects.save(row)

    jobs.sweep_orphaned_jobs(store)

    assert store.get(identifier)["status"] == "done"
