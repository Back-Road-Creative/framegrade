"""``probe_video_summary``: the engine-side video probe the publish-side
``fingerprint_media`` and the night-product validator share, so the engine
needs no import of ``gomoveshift.publishing`` (``tests/test_media_boundary.py``)."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from gomoveshift.pipeline.photos import night_products
from gomoveshift.pipeline.validation.ffprobe_client import probe_video_summary

_OK = (
    '{"format": {"duration": "2.5"}, "streams": [{"codec_type": "audio"}, '
    '{"codec_type": "video", "codec_name": "hevc", "width": 3840, "height": 2160}]}'
)


def _run_returning(stdout: str, returncode: int = 0):
    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=returncode, stdout=stdout, stderr="bad")

    return _fake_run


def test_summary_reads_first_video_stream(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _run_returning(_OK))
    assert probe_video_summary(tmp_path / "a.mp4", timeout=5) == (2.5, 3840, 2160, "hevc")


@pytest.mark.parametrize(
    ("stdout", "returncode", "needle"),
    [
        ("", 1, "ffprobe failed"),
        ('{"format": {"duration": "2"}, "streams": []}', 0, "no video stream"),
        ('{"format": {"duration": "x"}, "streams": [{"codec_type": "video"}]}', 0, "duration"),
        ('{"format": {"duration": "2"}, "streams": [{"codec_type": "video"}]}', 0, "incomplete"),
    ],
)
def test_summary_failures_name_label_and_file(tmp_path, monkeypatch, stdout, returncode, needle):
    path = tmp_path / "a.mp4"
    monkeypatch.setattr(subprocess, "run", _run_returning(stdout, returncode))
    with pytest.raises(RuntimeError, match=f"my label: .*{needle}|my label: {needle}") as exc:
        probe_video_summary(path, timeout=5, label="my label")
    assert str(path) in str(exc.value)


def test_night_validator_reports_probe_fields_without_hashing(tmp_path, monkeypatch):
    path = tmp_path / "t.mp4"
    path.write_bytes(b"x")
    monkeypatch.setattr(subprocess, "run", _run_returning(_OK))
    monkeypatch.setattr(
        "gomoveshift.pipeline.ffmpeg_runner.validate_video",
        lambda *a, **k: SimpleNamespace(valid=True, duration=2.5),
    )
    observed = night_products._validate_media(
        night_products.NightProduct.TIMELAPSE, path, {"duration_s": 2.5}
    )
    assert observed == {"duration_s": 2.5, "width": 3840, "height": 2160, "codec": "hevc"}
