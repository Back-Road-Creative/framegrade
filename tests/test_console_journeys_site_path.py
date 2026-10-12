"""The journal sync's site path is config (GMS_SITE_PATH), never a posted
form field -- same tenant-boundary rule as the Media library
(``library.resolve_site_path``): a form field could name any path on the
box, including another brand's checkout.
"""

from fastapi.testclient import TestClient

from gomoveshift.console import jobs
from gomoveshift.console.app import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def _job_store(tmp_path) -> jobs.JobStore:
    return jobs.JobStore(tmp_path / "_console" / "jobs.sqlite3")


def test_sync_site_uses_configured_path_not_posted_one(monkeypatch, tmp_path):
    configured = tmp_path / "configured-site"
    configured.mkdir()
    monkeypatch.setenv("GMS_SITE_PATH", str(configured))
    monkeypatch.setenv("GMS_STAGING_DIR", str(tmp_path))

    response = _client().post(
        "/journeys/sync-site",
        data={"site_path": str(tmp_path / "posted-elsewhere")},
        follow_redirects=False,
    )

    assert response.status_code == 303
    rows = jobs.list_jobs(_job_store(tmp_path))
    assert rows
    argv = rows[0]["argv"]
    assert str(configured) in argv
    assert str(tmp_path / "posted-elsewhere") not in argv


def test_sync_site_posted_path_is_not_honoured(monkeypatch, tmp_path):
    # Even with no GMS_SITE_PATH configured, a posted site_path must never
    # reach argv -- the form field is inert, not merely deprioritised.
    monkeypatch.delenv("GMS_SITE_PATH", raising=False)
    monkeypatch.setenv("GMS_STAGING_DIR", str(tmp_path))

    response = _client().post(
        "/journeys/sync-site",
        data={"site_path": str(tmp_path / "attacker-supplied")},
    )

    assert response.status_code == 400
    assert jobs.list_jobs(_job_store(tmp_path)) == []


def test_sync_site_unset_config_refuses_with_operator_message(monkeypatch, tmp_path):
    monkeypatch.delenv("GMS_SITE_PATH", raising=False)
    monkeypatch.setenv("GMS_STAGING_DIR", str(tmp_path))

    response = _client().post("/journeys/sync-site")

    assert response.status_code == 400
    assert "GMS_SITE_PATH" in response.text
    assert "Settings" in response.text
