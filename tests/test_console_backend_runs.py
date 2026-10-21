"""Backend runs not launched from the console are still visible.

Gate discovery used to read only the console job store's own argvs, so a run started from the CLI
or cron (no job row) never showed on /drives/gates/pending and never alerted the phone. The
registry in ``console.runs`` records every output dir with provenance (console / cli / cron /
unknown), found by reconciling the job store plus a bounded scan of the configured output root.
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from gomoveshift.console import jobs, runs
from gomoveshift.console.app import create_app


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path / "root"))
    (tmp_path / "root").mkdir()


def _gate(directory, name="gate_cd_2026-08-01", question="Played each LF?"):
    (directory / "_gates").mkdir(parents=True, exist_ok=True)
    (directory / "_gates" / f"{name}.question.json").write_text(json.dumps({"question": question}))
    return directory


def _store(tmp_path):
    return jobs.JobStore(tmp_path / "data" / "jobs.sqlite3")


def test_a_cli_run_gate_with_no_job_row_is_listed_as_pending(tmp_path):
    cli_run = _gate(tmp_path / "root" / "trip-cli")
    resp = TestClient(create_app()).get("/drives/gates/pending")
    assert resp.status_code == 200
    assert str(cli_run) in resp.text
    assert "Played each LF?" in resp.text
    assert "unknown" in resp.text  # origin is shown, and honest about not knowing


def test_the_ticker_alerts_for_a_cli_run_gate(tmp_path, monkeypatch):
    cli_run = _gate(tmp_path / "root" / "trip-cli")
    seen = []
    monkeypatch.setattr(jobs.notify, "check_pending_gates", lambda dirs, **kw: seen.append(dirs))
    jobs.Ticker(_store(tmp_path), tmp_path / "logs", tick_seconds=0)._observe_gates()
    assert seen and str(cli_run) in seen[0]


def test_console_jobs_register_as_console_with_their_argv_and_start(tmp_path):
    store = _store(tmp_path)
    out = tmp_path / "root" / "trip-console"
    job_id = store.create("drive", ["gms", "drive", "d", "--output-dir", str(out)], log_path="")
    runs.reconcile(store)
    row = next(r for r in runs.list_runs(store) if r["output_dir"] == str(out))
    assert row["origin"] == "console"
    assert row["argv"] == ["gms", "drive", "d", "--output-dir", str(out)]
    assert row["started_at"] == store.get(job_id)["created_at"]


def test_a_scanned_dir_without_provenance_is_unknown_not_guessed(tmp_path):
    cli_run = _gate(tmp_path / "root" / "trip-cli")
    store = _store(tmp_path)
    runs.reconcile(store)
    row = next(r for r in runs.list_runs(store) if r["output_dir"] == str(cli_run))
    assert row["origin"] == "unknown"
    assert row["started_at"] is None
    assert row["argv"] is None


def test_a_run_origin_sidecar_supplies_provenance(tmp_path):
    cron_run = _gate(tmp_path / "root" / "trip-cron")
    (cron_run / "run_origin.json").write_text(
        json.dumps(
            {"origin": "cron", "argv": ["gms", "drive"], "started_at": "2026-10-16T01:00:00Z"}
        )
    )
    store = _store(tmp_path)
    runs.reconcile(store)
    row = next(r for r in runs.list_runs(store) if r["output_dir"] == str(cron_run))
    assert (row["origin"], row["started_at"]) == ("cron", "2026-10-16T01:00:00Z")


def test_a_sidecar_cannot_claim_an_unknown_origin_value(tmp_path):
    odd = _gate(tmp_path / "root" / "trip-odd")
    (odd / "run_origin.json").write_text(json.dumps({"origin": "admin"}))
    store = _store(tmp_path)
    runs.reconcile(store)
    row = next(r for r in runs.list_runs(store) if r["output_dir"] == str(odd))
    assert row["origin"] == "unknown"


def test_reconcile_is_idempotent_and_never_duplicates(tmp_path):
    _gate(tmp_path / "root" / "trip-cli")
    store = _store(tmp_path)
    for _ in range(3):
        runs.reconcile(store)
    assert len(runs.list_runs(store)) == 1
    # a restart (fresh store object on the same file) changes nothing
    runs.reconcile(_store(tmp_path))
    assert len(runs.list_runs(_store(tmp_path))) == 1


def test_known_provenance_is_never_downgraded_or_overwritten(tmp_path):
    out = _gate(tmp_path / "root" / "trip")
    store = _store(tmp_path)
    store.create("drive", ["gms", "drive", "d", "--output-dir", str(out)], log_path="")
    runs.reconcile(store)
    (out / "run_origin.json").write_text(json.dumps({"origin": "cron", "argv": ["x"]}))
    runs.reconcile(store)
    row = next(r for r in runs.list_runs(store) if r["output_dir"] == str(out))
    assert row["origin"] == "console"


def test_unknown_is_upgraded_once_provenance_appears(tmp_path):
    out = _gate(tmp_path / "root" / "trip")
    store = _store(tmp_path)
    runs.reconcile(store)
    (out / "run_origin.json").write_text(
        json.dumps(
            {"origin": "cli", "argv": ["gms", "drive"], "started_at": "2026-10-16T02:00:00Z"}
        )
    )
    runs.reconcile(store)
    row = runs.list_runs(store)[0]
    assert (row["origin"], row["started_at"]) == ("cli", "2026-10-16T02:00:00Z")


def test_only_the_configured_root_is_scanned(tmp_path):
    outside = _gate(tmp_path / "elsewhere" / "trip")
    store = _store(tmp_path)
    runs.reconcile(store)
    assert str(outside) not in runs.known_output_dirs(store)


def test_a_dir_outside_the_registry_is_still_refused_by_containment(tmp_path):
    outside = _gate(tmp_path / "elsewhere" / "trip")
    resp = TestClient(create_app()).get(
        "/drives/gates", params={"output_dir": str(outside), "date_key": "2026-08-01"}
    )
    assert resp.status_code in (400, 403, 404)


def test_registry_table_survives_a_reopened_database(tmp_path):
    _gate(tmp_path / "root" / "trip-cli")
    runs.reconcile(_store(tmp_path))
    conn = sqlite3.connect(tmp_path / "data" / "jobs.sqlite3")
    assert conn.execute("SELECT COUNT(*) FROM backend_runs").fetchone()[0] == 1


def test_the_ticker_loop_itself_observes_gates_from_the_registry(tmp_path, monkeypatch):
    """One real loop pass: reverting the loop's call to job-store-only discovery turns this red."""
    cli_run = _gate(tmp_path / "root" / "trip-cli")
    seen = []
    monkeypatch.setattr(jobs, "run_due", lambda *a, **k: None)
    monkeypatch.setattr(jobs, "reap_finished", lambda *a, **k: None)
    monkeypatch.setattr("gomoveshift.console.night_supervision.supervise", lambda *a, **k: None)
    monkeypatch.setattr(jobs.notify, "check_queue_regressions", lambda *a, **k: None)
    monkeypatch.setattr(jobs.notify, "check_missed_slots", lambda *a, **k: None)
    monkeypatch.setattr(jobs.notify, "check_pending_gates", lambda dirs, **kw: seen.append(dirs))
    ticker = jobs.Ticker(_store(tmp_path), tmp_path / "logs", tick_seconds=0, wait=lambda s: None)
    ticker._stop_event.set = lambda: None  # keep the loop condition live for exactly one pass
    passes = []

    def one_pass(_seconds):
        passes.append(1)
        ticker._stop_event.is_set = lambda: True

    ticker._wait = one_pass
    ticker._run()
    assert passes == [1]
    assert seen and str(cli_run) in seen[0]


def test_a_run_nested_deeper_than_the_scan_depth_is_not_registered(tmp_path):
    shallow = _gate(tmp_path / "root" / "a" / "b")
    deep = _gate(tmp_path / "root" / "a" / "b" / "c")
    store = _store(tmp_path)
    runs.reconcile(store)
    registered = {r["output_dir"] for r in runs.list_runs(store)}
    assert str(shallow) in registered
    assert str(deep) not in registered


def test_a_restart_rescan_adds_no_job_and_no_registry_row(tmp_path):
    _gate(tmp_path / "root" / "trip-cli")
    store = _store(tmp_path)
    runs.reconcile(store)

    def counts(s):
        with sqlite3.connect(s.db_path) as conn:
            return (
                conn.execute("SELECT count(*) FROM jobs").fetchone()[0],
                conn.execute("SELECT count(*) FROM backend_runs").fetchone()[0],
            )

    before = counts(store)
    restarted = _store(tmp_path)  # a fresh process opening the same file
    jobs.Ticker(restarted, tmp_path / "logs", tick_seconds=0)._observe_gates()
    jobs.Ticker(restarted, tmp_path / "logs", tick_seconds=0)._observe_gates()
    assert counts(restarted) == before == (0, 1)
