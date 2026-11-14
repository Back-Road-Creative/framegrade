"""/jobs also lists the host job-runner records (cron, heal, CLI drives), read-only.

All data is synthetic: a tmp jobs dir pointed at by ``GMS_JOBS_DIR``.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from gomoveshift.console import pubqueue
from gomoveshift.console.app import create_app

BASE = 1794600000.0  # 2026-11-14T20:00:00Z -> 15:00 EST


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    monkeypatch.setenv("GMS_JOBS_DIR", str(tmp_path / "hostjobs"))
    monkeypatch.setattr(
        pubqueue.Config,
        "load",
        staticmethod(
            lambda *a, **k: SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
        ),
    )


def _record(tmp_path, rid, name, *, state=None, exit_code=None, created=BASE, log=None):
    d = tmp_path / "hostjobs" / rid
    d.mkdir(parents=True, exist_ok=True)
    rec = {
        "schema": "job/1",
        "name": name,
        "run_id": f"run-{rid}",
        "argv": ["python", "-m", "gomoveshift", "drive"],
        "created_at": created,
        "heartbeat_at": created + 60,
        "expect": [],
        "result": {"state": state, "exit_code": exit_code, "ended_at": created + 90}
        if state
        else None,
    }
    (d / "job.json").write_text(json.dumps(rec))
    if log is not None:
        (d / "output.log").write_text(log)
    return d


def _client():
    return TestClient(create_app())


def test_host_records_listed_with_origin_state_and_newest_first(tmp_path):
    _record(tmp_path, "aaa-run", "cron-run", created=BASE)
    _record(tmp_path, "bbb-fail", "heal-fail", state="failed", exit_code=2, created=BASE + 10)
    _record(tmp_path, "ccc-ok", "drive-ok", state="succeeded", exit_code=0, created=BASE + 20)
    resp = _client().get("/jobs")
    assert resp.status_code == 200
    text = resp.text
    assert "Host jobs (started outside this console)" in text
    assert "host runner" in text
    for name in ("cron-run", "heal-fail", "drive-ok"):
        assert name in text
    assert text.index("drive-ok") < text.index("heal-fail") < text.index("cron-run")
    assert "running" in text and "failed" in text and "succeeded" in text
    assert "15:00 EST" in text  # Eastern, not UTC
    assert 'href="/jobs/host/ccc-ok"' in text
    assert "console" in text.lower()  # existing table marked


def test_host_list_capped_at_fifty_newest(tmp_path):
    for i in range(55):
        _record(tmp_path, f"r{i:03d}", f"job-n{i:03d}x", created=BASE + i)
    text = _client().get("/jobs").text
    assert "job-n054x" in text and "job-n005x" in text
    assert "job-n004x" not in text and "job-n000x" not in text


def test_unreadable_dir_says_unavailable_not_empty(tmp_path):
    text = _client().get("/jobs").text  # directory does not exist
    assert "unavailable:" in text
    assert "Host jobs (started outside this console)" in text


def test_unreadable_records_are_counted(tmp_path):
    _record(tmp_path, "good", "good-job")
    bad = tmp_path / "hostjobs" / "bad"
    bad.mkdir()
    (bad / "job.json").write_text("{not json")
    text = _client().get("/jobs").text
    assert "good-job" in text
    assert "1 unreadable" in text


def test_detail_shows_last_100_lines_and_fields(tmp_path):
    log = "".join(f"line {i}\n" for i in range(1, 151))
    _record(tmp_path, "abc-123", "heal-fail", state="failed", exit_code=2, log=log)
    resp = _client().get("/jobs/host/abc-123")
    assert resp.status_code == 200
    text = resp.text
    assert "line 150" in text and "line 51\n" in text
    assert "line 50\n" not in text
    assert "heal-fail" in text and "failed" in text and "host runner" in text


def test_detail_is_read_only(tmp_path):
    _record(tmp_path, "abc-123", "cron-run", log="x\n")
    text = _client().get("/jobs/host/abc-123").text.lower()
    assert "<form" not in text
    assert "cancel" not in text and "re-run" not in text and "rerun" not in text
    assert "http-equiv=" not in text


@pytest.mark.parametrize(
    "path",
    [
        "/jobs/host/nope",
        "/jobs/host/%2e%2e",
        "/jobs/host/..%2Fsecret",
        "/jobs/host/%2Fetc%2Fpasswd",
        "/jobs/host/a%2Fb",
    ],
)
def test_bad_or_unknown_ids_are_404(tmp_path, path):
    _record(tmp_path, "abc-123", "cron-run")
    (tmp_path / "secret").mkdir()
    assert _client().get(path).status_code == 404


def test_directory_without_job_json_is_404(tmp_path):
    (tmp_path / "hostjobs" / "empty").mkdir(parents=True)
    assert _client().get("/jobs/host/empty").status_code == 404


def test_int_job_route_still_works_and_host_route_is_not_swallowed(tmp_path):
    client = _client()
    assert client.get("/jobs/999").status_code == 404  # unknown console job, not 422
    _record(tmp_path, "abc-123", "cron-run")
    assert client.get("/jobs/host/abc-123").status_code == 200


def test_jobs_page_has_no_meta_refresh_and_launch_form_stays(tmp_path):
    _record(tmp_path, "abc-123", "cron-run")
    text = _client().get("/jobs").text
    assert "http-equiv=" not in text.lower()
    assert 'action="/jobs"' in text


def test_get_handlers_write_nothing(tmp_path):
    _record(tmp_path, "abc-123", "cron-run", log="x\n")
    before = sorted(p.name for p in (tmp_path / "hostjobs").rglob("*"))
    client = _client()
    client.get("/jobs")
    client.get("/jobs/host/abc-123")
    assert sorted(p.name for p in (tmp_path / "hostjobs").rglob("*")) == before


def test_dot_segments_never_resolve_to_a_record(tmp_path):
    from gomoveshift.console import jobs

    _record(tmp_path, "abc-123", "cron-run")
    for bad in ("..", ".", "", "a/b", "../hostjobs/abc-123", "abc-123/", "abc 123"):
        assert jobs.get_host_record(bad) is None, bad
    # The client folds `/jobs/host/..` into `/jobs/` before sending; no id route is reached.
    assert _client().get("/jobs/host/").status_code in (404, 422)


def test_detail_lookup_survives_an_unresolvable_jobs_dir(monkeypatch):
    from gomoveshift.console import jobs
    from gomoveshift.status import snapshot

    def boom():
        raise RuntimeError("no data root")

    monkeypatch.setattr(snapshot, "jobs_dir", boom)
    assert jobs.get_host_record("abc-123") is None
