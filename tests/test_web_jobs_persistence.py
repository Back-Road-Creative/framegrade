"""Web jobs survive an app restart via atomic JSON write."""

from __future__ import annotations

from pathlib import Path

import pytest

from gomoveshift.models import ProcessingJob
from gomoveshift.web import app as web_app_module


@pytest.fixture(autouse=True)
def _isolated_jobs_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("GMS_WEB_JOBS_PATH", str(tmp_path / "web_jobs.json"))
    # Each test sees a fresh in-memory map.
    web_app_module.jobs.clear()
    yield
    web_app_module.jobs.clear()


def test_persist_then_reload_roundtrip(tmp_path: Path) -> None:
    job = ProcessingJob(video_path=tmp_path / "vid.mp4", status="analyzing")
    web_app_module.jobs["video.mp4"] = job
    web_app_module._persist_jobs_to_disk()

    # Snapshot file exists.
    snapshot = Path(web_app_module._jobs_state_path())
    assert snapshot.exists()

    # Wipe in-memory state, reload from disk.
    web_app_module.jobs.clear()
    web_app_module._load_jobs_from_disk()
    assert "video.mp4" in web_app_module.jobs
    assert web_app_module.jobs["video.mp4"].status == "analyzing"


def test_load_with_missing_file_is_noop() -> None:
    web_app_module._load_jobs_from_disk()  # no raise
    assert web_app_module.jobs == {}


def test_load_with_corrupt_file_logs_and_recovers(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    snapshot = Path(web_app_module._jobs_state_path())
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text("not-json{{{")
    import logging as _logging

    caplog.set_level(_logging.WARNING, logger=web_app_module.logger.name)
    web_app_module._load_jobs_from_disk()
    assert web_app_module.jobs == {}
    assert any("web_jobs.json parse failed" in m for m in caplog.messages)


def test_persist_uses_atomic_replace(tmp_path: Path) -> None:
    """Confirm the snapshot file is created via tmp+rename (no half-writes).

    We can't directly observe os.replace in a unit test, but we CAN verify
    that after persist, no ``web_jobs.*.tmp`` siblings remain.
    """
    job = ProcessingJob(video_path=tmp_path / "vid.mp4", status="analyzing")
    web_app_module.jobs["video.mp4"] = job
    web_app_module._persist_jobs_to_disk()

    snapshot = Path(web_app_module._jobs_state_path())
    leftovers = list(snapshot.parent.glob("web_jobs.*.tmp"))
    assert leftovers == [], leftovers
    assert snapshot.exists()
