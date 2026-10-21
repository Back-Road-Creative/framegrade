"""Console Shorts tab -- launch + CSV preview/byte-identical download.

Shorts have no standalone CLI subcommand (grep ``gomoveshift/cli/__init__.py``
``_COMMAND_MAP``: only ``drive``/``pipeline`` produce shorts). The tab
therefore launches the SAME argv the Drives tab does -- the console never
invents a second launch path -- and adds read-only access to the shorts
Metricool CSV a run already wrote (``drive_phases/phase_d.py``'s
``<project>_shorts_metricool.csv``).
"""

from fastapi.testclient import TestClient

from gomoveshift.console import drives, jobs, shorts
from gomoveshift.console.app import create_app

_FIXTURE_CSV = (
    "Post text,Picture Url 1,Publish Date,YouTube Video Category\n"
    'Sullivan County TN drive,https://example.com/a.mp4,2026-08-01 12:00,""\n'
    'Blount County TN drive,https://example.com/b.mp4,2026-08-01 18:00,""\n'
)


def _client() -> TestClient:
    return TestClient(create_app())


def test_build_shorts_argv_matches_the_drive_invocation():
    argv = shorts.build_shorts_argv("/data/day1", output_dir="/data/out", region="usa")
    assert argv == drives.build_drive_argv("/data/day1", output_dir="/data/out", region="usa")
    assert "drive" in jobs.ALLOWED_SUBCOMMANDS


def test_build_shorts_argv_forwards_publish_and_lf_day():
    argv = shorts.build_shorts_argv(
        "/data/day1", output_dir="/data/out", publish="tue", lf_day="tue"
    )
    assert argv == drives.build_drive_argv(
        "/data/day1", output_dir="/data/out", publish="tue", lf_day="tue"
    )
    assert "--publish" in argv and "--lf-day" in argv


def test_build_shorts_argv_accepts_shorts_target_and_gap():
    argv = shorts.build_shorts_argv(
        "/data/day1", output_dir="/data/out", shorts_target=6, shorts_gap=45.0
    )
    assert argv[argv.index("--shorts-target") + 1] == "6"
    assert argv[argv.index("--shorts-gap") + 1] == "45.0"


def test_build_shorts_argv_full_render_forwards_explicit_safety_choices():
    argv = shorts.build_shorts_argv(
        "/data/day1",
        output_dir="/data/out",
        dry_run=False,
        confirm_plan=True,
        skip_upload=True,
        ost_system_review=True,
    )
    assert "--force" in argv
    assert "--skip-upload" in argv and "--no-youtube-upload" in argv
    assert "--ost-system-review" in argv
    assert "--ost-audit-confirmed" not in argv
    assert "--no-review" in argv
    dry_argv = shorts.build_shorts_argv(
        "/data/day1", output_dir="/data/out", dry_run=True, confirm_plan=True
    )
    assert "--dry-run" in dry_argv and "--force" not in dry_argv


def test_shorts_form_shows_safe_full_render_choices_off_by_default():
    page = _client().get("/shorts")
    assert page.status_code == 200
    for control in ("confirm_plan", "skip_upload", "ost_system_review"):
        assert f'name="{control}"' in page.text
        assert f'name="{control}" checked' not in page.text


def test_find_shorts_csv_locates_the_project_csv(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    assert shorts.find_shorts_csv(output_dir) is None
    csv_path = output_dir / "drive_day1_shorts_metricool.csv"
    csv_path.write_text(_FIXTURE_CSV, encoding="utf-8")
    assert shorts.find_shorts_csv(output_dir) == csv_path


def test_read_shorts_csv_rows_parses_header_and_body(tmp_path):
    csv_path = tmp_path / "x_shorts_metricool.csv"
    csv_path.write_text(_FIXTURE_CSV, encoding="utf-8")
    header, rows = shorts.read_shorts_csv_rows(csv_path)
    assert header == ["Post text", "Picture Url 1", "Publish Date", "YouTube Video Category"]
    assert len(rows) == 2
    assert rows[0][0] == "Sullivan County TN drive"


def test_shorts_run_launches_through_the_allowlist(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))

    class _FakePopen:
        def poll(self):
            return 0

    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: _FakePopen())

    resp = _client().post(
        "/shorts/run",
        data={
            "brand": "gomoveshift",
            "directory": str(tmp_path / "day1"),
            "output_dir": str(tmp_path / "out"),
            "region": "usa",
            "dry_run": "on",
            "publish": "tue",
            "lf_day": "tue",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    job_id = int(resp.headers["location"].rsplit("/", 1)[-1])
    row = jobs.get_job(jobs.JobStore(tmp_path / "jobs.sqlite3"), job_id)
    assert row["name"] == "shorts"
    assert str(tmp_path / "day1") in row["argv"]
    assert "--dry-run" in row["argv"] and "--no-review" in row["argv"]
    assert "--publish" in row["argv"] and "--lf-day" in row["argv"]


def test_shorts_full_run_refuses_without_plan_confirmation(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    resp = _client().post(
        "/shorts/run",
        data={
            "brand": "gomoveshift",
            "directory": str(tmp_path / "day1"),
            "output_dir": str(tmp_path / "out"),
            "publish": "tue",
            "lf_day": "tue",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert "Confirm the dry-run plan" in resp.text
    assert str(tmp_path / "day1") in resp.text
    assert jobs.list_jobs(jobs.JobStore(tmp_path / "jobs.sqlite3")) == []


def test_shorts_full_run_records_upload_hold_and_system_review(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))

    class _FakePopen:
        def poll(self):
            return 0

    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: _FakePopen())
    resp = _client().post(
        "/shorts/run",
        data={
            "brand": "gomoveshift",
            "directory": str(tmp_path / "day1"),
            "output_dir": str(tmp_path / "out"),
            "publish": "tue",
            "lf_day": "tue",
            "confirm_plan": "on",
            "skip_upload": "on",
            "ost_system_review": "on",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    job_id = int(resp.headers["location"].rsplit("/", 1)[-1])
    argv = jobs.get_job(jobs.JobStore(tmp_path / "jobs.sqlite3"), job_id)["argv"]
    assert "--force" in argv
    assert "--skip-upload" in argv and "--no-youtube-upload" in argv
    assert "--ost-system-review" in argv
    assert "--ost-audit-confirmed" not in argv


def test_shorts_form_keeps_safety_choices_after_input_error(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    resp = _client().post(
        "/shorts/run",
        data={
            "brand": "gomoveshift",
            "directory": str(tmp_path / "day1"),
            "output_dir": str(tmp_path / "out"),
            "publish": "tue",
            "lf_day": "tue",
            "shorts_target": "six",
            "confirm_plan": "on",
            "skip_upload": "on",
            "ost_system_review": "on",
        },
    )
    assert resp.status_code == 400
    assert str(tmp_path / "day1") in resp.text
    for control in ("confirm_plan", "skip_upload", "ost_system_review"):
        assert f'name="{control}" checked' in resp.text
    assert jobs.list_jobs(jobs.JobStore(tmp_path / "jobs.sqlite3")) == []


def test_shorts_run_refuses_when_neither_publish_nor_lf_day_given(tmp_path, monkeypatch):
    """The drive handler hard-refuses an
    unscheduled launch -- the Shorts tab must not hand it one."""
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))

    class _FakePopen:
        def poll(self):
            return 0

    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: _FakePopen())

    resp = _client().post(
        "/shorts/run",
        data={
            "brand": "gomoveshift",
            "directory": str(tmp_path / "day1"),
            "output_dir": str(tmp_path / "out"),
        },
        follow_redirects=False,
    )
    assert resp.status_code == 400
    store = jobs.JobStore(tmp_path / "jobs.sqlite3")
    assert jobs.list_jobs(store) == []


def test_shorts_run_accepts_shorts_target_gap_and_brand(shipped_brands, tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))

    class _FakePopen:
        def poll(self):
            return 0

    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: _FakePopen())

    resp = _client().post(
        "/shorts/run",
        data={
            "brand": "examplebrand",
            "directory": str(tmp_path / "day1"),
            "output_dir": str(tmp_path / "out"),
            "shorts_target": "6",
            "shorts_gap": "45",
            "dry_run": "on",
            "publish": "fri",
            "lf_day": "fri",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    job_id = int(resp.headers["location"].rsplit("/", 1)[-1])
    row = jobs.get_job(jobs.JobStore(tmp_path / "jobs.sqlite3"), job_id)
    assert "--shorts-target" in row["argv"] and "--shorts-gap" in row["argv"]


def test_shorts_run_rejects_an_unknown_brand(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    resp = _client().post(
        "/shorts/run",
        data={
            "brand": "not-a-real-brand",
            "directory": str(tmp_path / "day1"),
            "output_dir": str(tmp_path / "out"),
        },
        follow_redirects=False,
    )
    assert resp.status_code == 400


def test_shorts_tab_renders_the_pipeline_flags_fieldset(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    resp = _client().get("/shorts")
    assert resp.status_code == 200
    assert "Pipeline flags" in resp.text
    assert "--shorts-target" in resp.text


def _register_output_dir(tmp_path, output_dir) -> None:
    store = jobs.JobStore(tmp_path / "jobs.sqlite3")
    store.create("shorts", ["gms", "shorts", "d", "--output-dir", str(output_dir)], log_path="")


def test_shorts_csv_route_renders_table_from_fixture_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "drive_day1_shorts_metricool.csv").write_text(_FIXTURE_CSV, encoding="utf-8")
    _register_output_dir(tmp_path, output_dir)
    resp = _client().get("/shorts/csv", params={"output_dir": str(output_dir)})
    assert resp.status_code == 200
    assert "Sullivan County TN drive" in resp.text
    assert "Blount County TN drive" in resp.text
    assert "Post text" in resp.text


def test_shorts_csv_route_404s_on_missing_output_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    missing = tmp_path / "nope"
    resp = _client().get("/shorts/csv", params={"output_dir": str(missing)})
    assert resp.status_code == 404
    assert str(missing) not in resp.text


def test_shorts_csv_route_rejects_a_foreign_output_dir_without_echoing_path(tmp_path, monkeypatch):
    """An output_dir no job this console launched used is refused, even
    though the directory and CSV genuinely exist -- and the refusal body must
    not echo the submitted path back."""
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    foreign_dir = tmp_path / "foreign-tenant-out"
    foreign_dir.mkdir()
    (foreign_dir / "drive_day1_shorts_metricool.csv").write_text(_FIXTURE_CSV, encoding="utf-8")
    # No job row references foreign_dir -- this console never launched a run there.
    resp = _client().get("/shorts/csv", params={"output_dir": str(foreign_dir)})
    assert resp.status_code == 404
    assert str(foreign_dir) not in resp.text
    assert "Sullivan County TN drive" not in resp.text


def test_shorts_csv_route_404s_when_no_csv_written(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _register_output_dir(tmp_path, output_dir)
    resp = _client().get("/shorts/csv", params={"output_dir": str(output_dir)})
    assert resp.status_code == 404
    assert str(output_dir) not in resp.text


def test_shorts_csv_download_returns_byte_identical_content(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    csv_path = output_dir / "drive_day1_shorts_metricool.csv"
    csv_path.write_bytes(_FIXTURE_CSV.encode("utf-8"))
    _register_output_dir(tmp_path, output_dir)
    resp = _client().get("/shorts/csv/download", params={"output_dir": str(output_dir)})
    assert resp.status_code == 200
    assert resp.content == csv_path.read_bytes()
    assert resp.headers["content-type"].startswith("text/csv")


def test_shorts_csv_download_404s_when_no_csv_written(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _register_output_dir(tmp_path, output_dir)
    resp = _client().get("/shorts/csv/download", params={"output_dir": str(output_dir)})
    assert resp.status_code == 404
    assert str(output_dir) not in resp.text


def test_shorts_csv_download_rejects_a_foreign_output_dir_and_returns_no_bytes(
    tmp_path, monkeypatch
):
    """A foreign output_dir's CSV must never be served -- assert the
    body carries no CSV bytes, not just the status code."""
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    foreign_dir = tmp_path / "foreign-tenant-out"
    foreign_dir.mkdir()
    csv_path = foreign_dir / "drive_day1_shorts_metricool.csv"
    csv_path.write_bytes(_FIXTURE_CSV.encode("utf-8"))
    # No job row references foreign_dir.
    resp = _client().get("/shorts/csv/download", params={"output_dir": str(foreign_dir)})
    assert resp.status_code == 404
    assert str(foreign_dir) not in resp.text
    assert resp.content != csv_path.read_bytes()
    assert b"Sullivan County TN drive" not in resp.content
