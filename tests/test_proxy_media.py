"""A low-res review proxy at ingest, and GoPro camera
model/firmware captured alongside each clip's other ffprobe-derived facts.

Both are optional-by-construction: a proxy build failure WARNs and review
falls back to the original file; a missing camera tag leaves the field
``None`` rather than raising. Neither runs real ffmpeg/ffprobe here --
subprocess is monkeypatched and only the built argv / return value is
asserted.
"""

import json
import subprocess

from gomoveshift.console import drives, jobs
from gomoveshift.console.app import create_app
from fastapi.testclient import TestClient

from gomoveshift.pipeline.ingest.gopro_parser import extract_camera_metadata
from gomoveshift.pipeline.proxy_media import build_review_proxy

DATE_KEY = "2026-08-01"


class _FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_extract_camera_metadata_reads_model_and_firmware(monkeypatch):
    probe = {
        "format": {
            "tags": {
                "com.gopro.model": "HERO11 Black",
                "firmware": "H22.01.02.32.00",
            }
        },
        "streams": [],
    }

    def fake_run(cmd, **kwargs):
        assert cmd[0] == "ffprobe"
        return _FakeResult(returncode=0, stdout=json.dumps(probe))

    monkeypatch.setattr(subprocess, "run", fake_run)

    model, firmware = extract_camera_metadata("clip.mp4")

    assert model == "HERO11 Black"
    assert firmware == "H22.01.02.32.00"


def test_extract_camera_metadata_missing_tags_returns_none(monkeypatch):
    probe = {"format": {"tags": {}}, "streams": []}

    monkeypatch.setattr(
        subprocess, "run", lambda cmd, **kwargs: _FakeResult(returncode=0, stdout=json.dumps(probe))
    )

    model, firmware = extract_camera_metadata("clip.mp4")

    assert model is None
    assert firmware is None


def test_extract_camera_metadata_ffprobe_failure_returns_none(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, **kwargs: _FakeResult(returncode=1, stdout="", stderr="boom")
    )

    model, firmware = extract_camera_metadata("clip.mp4")

    assert model is None
    assert firmware is None


def test_build_review_proxy_argv_is_small_muted_and_faststart(tmp_path, monkeypatch):
    stages_dir = tmp_path / "_stages"
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        out = cmd[-1]
        with open(out, "wb") as f:
            f.write(b"fake-proxy-bytes")
        return _FakeResult(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    src = tmp_path / "clip.mp4"
    src.write_bytes(b"source-bytes")

    proxy = build_review_proxy(src, stages_dir)

    assert proxy is not None
    assert proxy.exists()
    assert proxy.is_relative_to(stages_dir)

    cmd = captured["cmd"]
    assert cmd[0] == "ffmpeg"
    assert "-an" in cmd
    joined = " ".join(cmd)
    assert "scale=640" in joined
    assert "-crf" in cmd and "28" in cmd
    assert "+faststart" in joined


def test_build_review_proxy_never_writes_under_output_dir(tmp_path, monkeypatch):
    """The proxy path must be derived from the stages root, never from a
    sibling ``output_dir`` the caller might also pass around."""
    stages_dir = tmp_path / "_stages"
    output_dir = tmp_path / "out"

    def fake_run(cmd, **kwargs):
        out = cmd[-1]
        with open(out, "wb") as f:
            f.write(b"fake")
        return _FakeResult(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")

    proxy = build_review_proxy(src, stages_dir)

    assert proxy is not None
    assert str(proxy).startswith(str(stages_dir))
    assert not str(proxy).startswith(str(output_dir))


def test_build_review_proxy_failure_warns_and_returns_none(tmp_path, monkeypatch, caplog):
    def fake_run(cmd, **kwargs):
        return _FakeResult(returncode=1, stderr="ffmpeg exploded")

    monkeypatch.setattr(subprocess, "run", fake_run)

    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")

    with caplog.at_level("WARNING"):
        proxy = build_review_proxy(src, tmp_path / "_stages")

    assert proxy is None
    assert any("proxy" in rec.message.lower() for rec in caplog.records)


def test_build_review_proxy_subprocess_exception_warns_and_returns_none(
    tmp_path, monkeypatch, caplog
):
    def fake_run(cmd, **kwargs):
        raise OSError("ffmpeg not found")

    monkeypatch.setattr(subprocess, "run", fake_run)

    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")

    with caplog.at_level("WARNING"):
        proxy = build_review_proxy(src, tmp_path / "_stages")

    assert proxy is None


def _write_tiny_video(path) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 32))
    assert writer.isOpened(), "could not open VideoWriter for fixture"
    for _ in range(5):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    assert path.exists() and path.stat().st_size > 0


def _gate(tmp_path, monkeypatch, *, with_proxy=False):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    output_dir = tmp_path / "out"
    video = output_dir / "lf_01.mp4"
    _write_tiny_video(video)
    if with_proxy:
        proxy = output_dir / "lf_01_proxy.mp4"
        proxy.write_bytes(b"proxy-bytes-smaller-than-original")
    metadata = output_dir / "day_lf01_youtube.json"
    metadata.write_text(json.dumps({"title": "Sample Drive"}), encoding="utf-8")
    (output_dir / "_gates").mkdir(parents=True, exist_ok=True)
    (output_dir / "_gates" / f"gate_cd_{DATE_KEY}.question.json").write_text(
        json.dumps(
            {
                "question": f"Played each LF and reviewed all metadata for {DATE_KEY}?",
                "choices": ["approve", "reject"],
                "artifacts": [str(video), str(metadata)],
            }
        ),
        encoding="utf-8",
    )
    store = jobs.JobStore(tmp_path / "jobs.sqlite3")
    store.create("drive", ["gms", "drive", "d", "--output-dir", str(output_dir)], log_path="")
    return output_dir, video


def _question(output_dir):
    return drives.read_gate_question(output_dir, DATE_KEY)


def test_review_playback_prefers_the_proxy_when_present(tmp_path, monkeypatch):
    output_dir, video = _gate(tmp_path, monkeypatch, with_proxy=True)

    resolved = drives.resolve_gate_video_media(output_dir, _question(output_dir), 0)

    assert resolved is not None
    path, content_type = resolved
    assert path.name == "lf_01_proxy.mp4"
    assert content_type == "video/mp4"


def test_review_playback_falls_back_to_the_original_without_a_proxy(tmp_path, monkeypatch):
    output_dir, video = _gate(tmp_path, monkeypatch, with_proxy=False)

    resolved = drives.resolve_gate_video_media(output_dir, _question(output_dir), 0)

    assert resolved is not None
    path, _ = resolved
    assert path == video.resolve()


def test_the_route_streams_the_proxy_bytes_not_the_original(tmp_path, monkeypatch):
    output_dir, video = _gate(tmp_path, monkeypatch, with_proxy=True)

    response = TestClient(create_app()).get(
        "/drives/gates/video",
        params={"output_dir": str(output_dir), "date_key": DATE_KEY, "index": 0},
    )

    assert response.status_code == 200
    assert response.content == (output_dir / "lf_01_proxy.mp4").read_bytes()
    assert response.content != video.read_bytes()
