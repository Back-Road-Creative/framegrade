"""Console route for the standalone-asset enqueue door."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from gomoveshift.console import jobs, pubqueue
from gomoveshift.console.app import create_app


def _config(tmp_path):
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))


def _client() -> TestClient:
    return TestClient(create_app())


def _register_output_dir(tmp_path, output_dir) -> None:
    store = jobs.JobStore(tmp_path / "jobs.sqlite3")
    store.create("drive", ["gms", "drive", "d", "--output-dir", str(output_dir)], log_path="")


def _launch_recorder(monkeypatch):
    launched: dict = {}
    monkeypatch.setattr(
        jobs,
        "launch",
        lambda store, name, argv, log_dir, env=None: launched.setdefault(name, argv) or 1,
    )
    return launched


def test_enqueue_asset_refuses_a_video_path_from_another_tenants_run(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    launched = _launch_recorder(monkeypatch)
    mine = tmp_path / "out"
    mine.mkdir()
    _register_output_dir(tmp_path, mine)
    foreign = tmp_path / "other-tenant-out"
    foreign.mkdir()
    foreign_video = foreign / "route.mp4"

    resp = _client().post(
        "/publishing/enqueue-asset",
        data={
            "video_path": str(foreign_video),
            "platform": "youtube",
            "at": "2026-09-05T14:00:00+00:00",
            "title": "t",
            "confirm": "abc",
        },
        follow_redirects=False,
    )

    assert resp.status_code == 404
    assert str(foreign_video) not in resp.text
    assert launched == {}


def test_enqueue_asset_launches_the_real_subcommand(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    launched = _launch_recorder(monkeypatch)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    video = output_dir / "route.mp4"
    _register_output_dir(tmp_path, output_dir)

    resp = _client().post(
        "/publishing/enqueue-asset",
        data={
            "video_path": str(video),
            "platform": "youtube",
            "at": "2026-09-05T14:00:00+00:00",
            "title": "A Drive",
            "description": "desc",
            "confirm": "the-hash",
        },
        follow_redirects=False,
    )

    assert resp.status_code == 303
    argv = launched["enqueue-asset"]
    assert argv[3:] == [
        "enqueue-asset",
        str(video.resolve()),
        "--at",
        "2026-09-05T14:00:00+00:00",
        "--title",
        "A Drive",
        "--description",
        "desc",
        "--confirm",
        "the-hash",
        "--apply",
        "--platform",
        "youtube",
    ]
