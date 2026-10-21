"""The console refuses an unreadable drive source before queueing a render.

A run once used a source dir of symlinks into a drive the worker could not
read; the failure surfaced only after the job was queued. The console's own
check is the cheap first gate (its user may differ from the worker's); the
host worker's preflight stays authoritative.
"""

import os

import pytest
from fastapi.testclient import TestClient

from gomoveshift.console.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    launched = []
    monkeypatch.setattr(
        "gomoveshift.console.app.jobs.launch", lambda *a, **k: launched.append(a) or 42
    )
    c = TestClient(create_app())
    c.launched = launched
    return c


def _post(client, src, tmp_path, **extra):
    data = {
        "directory": str(src),
        "output_dir": str(tmp_path / "out"),
        "publish": "tue",
        "lf_day": "tue",
        "dry_run": "on",
    }
    data.update(extra)
    return client.post("/drives/run", data=data, follow_redirects=False)


def test_dangling_symlink_refused_before_queueing(client, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "gone.mp4").symlink_to(tmp_path / "missing" / "gone.mp4")
    resp = _post(client, src, tmp_path)
    assert resp.status_code == 400
    assert str(src / "gone.mp4") in resp.text
    assert client.launched == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads everything")
def test_unreadable_file_refused_and_each_path_named(client, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    locked = tmp_path / "locked.mov"
    locked.write_bytes(b"x")
    os.chmod(locked, 0)
    (src / "link.mov").symlink_to(locked)
    (src / "dangling.mp4").symlink_to(tmp_path / "nope.mp4")
    try:
        resp = _post(client, src, tmp_path, dry_run="", confirm_plan="on")
    finally:
        os.chmod(locked, 0o600)
    assert resp.status_code == 400
    assert str(src / "link.mov") in resp.text and "not readable" in resp.text
    assert str(src / "dangling.mp4") in resp.text
    assert client.launched == []


def test_readable_symlinks_accepted(client, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    src = tmp_path / "src"
    src.mkdir()
    for name in ("a.mp4", "b.MOV"):
        (real / name).write_bytes(b"x")
        (src / name).symlink_to(real / name)
    (src / "notes.txt").symlink_to(tmp_path / "ignored-dangling.txt")
    resp = _post(client, src, tmp_path)
    assert resp.status_code == 303
    assert len(client.launched) == 1
