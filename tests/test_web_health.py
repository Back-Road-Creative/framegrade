"""/health is a dependency-aware liveness check.

Returns 200 when ffmpeg + ffprobe are on PATH and the scratch dir is
writable. Returns 503 with a JSON body naming the failing check otherwise.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_client(tmp_path):
    from gomoveshift.config import Config
    from gomoveshift.web.app import create_app

    config = Config()
    config.paths.temp_folder = str(tmp_path / "scratch")
    app = create_app(config=config)
    return TestClient(app)


def test_health_ok_when_all_deps_present(app_client, monkeypatch):
    monkeypatch.setattr("gomoveshift.web.app.shutil.which", lambda name: f"/usr/bin/{name}")
    response = app_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_503_when_ffmpeg_missing(app_client, monkeypatch):
    def which(name):
        return None if name == "ffmpeg" else f"/usr/bin/{name}"

    monkeypatch.setattr("gomoveshift.web.app.shutil.which", which)
    response = app_client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert "ffmpeg" in body["failed"]


def test_health_503_when_ffprobe_missing(app_client, monkeypatch):
    def which(name):
        return None if name == "ffprobe" else f"/usr/bin/{name}"

    monkeypatch.setattr("gomoveshift.web.app.shutil.which", which)
    response = app_client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert "ffprobe" in body["failed"]


def test_health_503_when_scratch_dir_unwritable(tmp_path, monkeypatch):
    from gomoveshift.config import Config
    from gomoveshift.web.app import create_app

    monkeypatch.setattr("gomoveshift.web.app.shutil.which", lambda name: f"/usr/bin/{name}")

    config = Config()
    # Point at a path under a non-existent parent that cannot be created
    # (root-owned /proc subdir is not writable by the test user).
    config.paths.temp_folder = "/proc/1/no_such_dir"
    app = create_app(config=config)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert "scratch_dir" in body["failed"]


def test_health_response_lists_only_failing_checks(app_client, monkeypatch):
    def which(name):
        return None if name == "ffmpeg" else f"/usr/bin/{name}"

    monkeypatch.setattr("gomoveshift.web.app.shutil.which", which)
    response = app_client.get("/health")
    body = response.json()
    assert body["failed"] == ["ffmpeg"]


def test_health_uses_config_temp_folder(tmp_path, monkeypatch):
    """Liveness check writes a sentinel under config.paths.temp_folder."""
    from gomoveshift.config import Config
    from gomoveshift.web.app import create_app

    monkeypatch.setattr("gomoveshift.web.app.shutil.which", lambda name: f"/usr/bin/{name}")

    scratch = tmp_path / "scratch"
    config = Config()
    config.paths.temp_folder = str(scratch)
    app = create_app(config=config)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert Path(scratch).exists()
