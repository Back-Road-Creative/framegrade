"""Console Journeys & site tab -- site PR list + journal job launch."""

import sys

import pytest
from fastapi.testclient import TestClient

from gomoveshift.console import journeys
from gomoveshift.console.app import create_app


class _FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _client() -> TestClient:
    return TestClient(create_app())


def test_list_site_prs_parses_fixture_json():
    fixture = (
        '[{"number": 12, "title": "sync 2026-08-30", "url": "https://x/12"}, '
        '{"number": 9, "title": "stub: example-2026", "url": "https://x/9"}]'
    )
    calls = []

    def fake_runner(argv, **kwargs):
        calls.append(argv)
        return _FakeResult(returncode=0, stdout=fixture)

    prs, error = journeys.list_site_prs(repo="example-org/example-site", runner=fake_runner)

    assert error == ""
    assert [pr["number"] for pr in prs] == [12, 9]
    assert calls[0][:3] == ["gh", "pr", "list"]
    assert "example-org/example-site" in calls[0]


def test_list_site_prs_gh_missing_degrades_to_empty_with_message():
    def fake_runner(argv, **kwargs):
        raise FileNotFoundError("gh not found")

    prs, error = journeys.list_site_prs(repo="x/y", runner=fake_runner)
    assert prs == []
    assert "gh" in error.lower()


def test_list_site_prs_gh_failure_degrades_to_empty_with_message():
    def fake_runner(argv, **kwargs):
        return _FakeResult(returncode=1, stderr="authentication required")

    prs, error = journeys.list_site_prs(repo="x/y", runner=fake_runner)
    assert prs == []
    assert "authentication required" in error


def test_list_site_prs_unparseable_output_degrades_to_empty_with_message():
    def fake_runner(argv, **kwargs):
        return _FakeResult(returncode=0, stdout="not json")

    prs, error = journeys.list_site_prs(repo="x/y", runner=fake_runner)
    assert prs == []
    assert error


def test_site_repo_env_override(monkeypatch):
    monkeypatch.setenv("GMS_SITE_REPO", "example-org/other-site")
    assert journeys.site_repo() == "example-org/other-site"


def test_site_repo_has_no_default(monkeypatch):
    monkeypatch.delenv("GMS_SITE_REPO", raising=False)
    assert journeys.site_repo() == ""


def test_list_site_prs_not_configured_names_the_variable(monkeypatch):
    monkeypatch.delenv("GMS_SITE_REPO", raising=False)

    def fake_runner(argv, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("gh must not run without a configured repo")

    prs, error = journeys.list_site_prs(runner=fake_runner)
    assert prs == []
    assert "GMS_SITE_REPO" in error


def test_build_sync_site_argv_matches_parser_contract():
    argv = journeys.build_sync_site_argv("/path/to/site")
    assert argv == [
        sys.executable,
        "-m",
        "gomoveshift",
        "sync-site",
        "--site-path",
        "/path/to/site",
    ]


def test_build_sync_site_argv_requires_site_path():
    with pytest.raises(ValueError):
        journeys.build_sync_site_argv("")


def test_build_publish_journal_replay_argv_matches_parser_contract():
    argv = journeys.build_publish_journal_replay_argv()
    assert argv == [sys.executable, "-m", "gomoveshift", "publish-journal", "--replay-pending"]


def test_journeys_tab_renders_prs_and_launch_buttons(monkeypatch):
    monkeypatch.setattr(
        journeys,
        "list_site_prs",
        lambda repo=None, runner=None: ([{"number": 3, "title": "sync", "url": "https://x/3"}], ""),
    )
    response = _client().get("/journeys")
    assert response.status_code == 200
    assert "sync" in response.text
    assert "sync-site" in response.text
    assert "publish-journal" in response.text


def test_journeys_tab_gh_error_renders_empty_state(monkeypatch):
    monkeypatch.setattr(
        journeys, "list_site_prs", lambda repo=None, runner=None: ([], "gh not found")
    )
    response = _client().get("/journeys")
    assert response.status_code == 200
    assert "gh not found" in response.text


def test_journeys_sync_site_launch_redirects_to_job(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_STAGING_DIR", str(tmp_path))
    # site_path comes from GMS_SITE_PATH (library.py's config-only rule), not
    # a posted form field -- see test_console_journeys_site_path.py.
    monkeypatch.setenv("GMS_SITE_PATH", str(tmp_path))
    response = _client().post("/journeys/sync-site", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"].startswith("/jobs/")


def test_journeys_publish_journal_replay_launch_redirects_to_job(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_STAGING_DIR", str(tmp_path))
    response = _client().post("/journeys/publish-journal-replay", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"].startswith("/jobs/")
