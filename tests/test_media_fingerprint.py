"""Production ``MediaFingerprint`` builder.

Only a mocked ffprobe path is used for the error branches (missing binary,
non-zero exit, no video stream, unparseable duration); the happy path runs
a real ffprobe against a tiny fixture rendered with ffmpeg, so a wrong field
name in the ffprobe JSON parsing would actually be caught.
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone

import pytest

from gomoveshift.publishing.media_fingerprint import (
    MediaVerificationError,
    fingerprint_media,
    verify_media,
)
from gomoveshift.publishing.models import (
    MediaFingerprint,
    PlatformPayloads,
    RouteSelection,
    ScheduleWindow,
    ShortPublishRecord,
)

_SCHEDULED = datetime(2026, 9, 1, 19, 0, tzinfo=timezone.utc)


def _record(media: MediaFingerprint) -> ShortPublishRecord:
    return ShortPublishRecord(
        project_name="proj",
        lf_index=0,
        short_index=1,
        media=media,
        schedule=ScheduleWindow(_SCHEDULED, "America/Chicago"),
        recording_date="2026-08-01",
        sampled_regions=("a",),
        parent_lf_video_id="vid",
        parent_lf_publish_at=_SCHEDULED,
        payloads=PlatformPayloads(),
        consent_hash="c",
        reviewed_plan_hash="HASH1",
        routes=RouteSelection(selected=("youtube",)),
    )


def _ffmpeg(args: list[str]) -> None:
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-y", *args], capture_output=True, text=True, timeout=60
    )
    assert p.returncode == 0, p.stderr[-2000:]


@pytest.fixture
def tiny_mp4(tmp_path):
    # 2 frames at 1 fps, 64x48 h264 — small enough to be a fast fixture,
    # real enough that ffprobe reports a genuine video stream/codec.
    dst = tmp_path / "fixture.mp4"
    _ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=64x48:d=2:r=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(dst),
        ]
    )
    return dst


def test_fingerprint_media_happy_path(tiny_mp4):
    fp = fingerprint_media(tiny_mp4)

    # path stored as given (str(Path) as passed in) — this value feeds the
    # plan hash, so two runs against the same relative/absolute argument
    # must produce the same string; str(path) on the Path we were handed
    # (no os.path.abspath) keeps that stable without depending on cwd.
    assert fp.path == str(tiny_mp4)
    assert fp.size_bytes == tiny_mp4.stat().st_size

    expected_sha256 = hashlib.sha256(tiny_mp4.read_bytes()).hexdigest()
    assert fp.sha256 == expected_sha256

    assert fp.width == 64
    assert fp.height == 48
    assert fp.codec == "h264"
    assert 1.5 <= fp.duration_seconds <= 2.5


def test_fingerprint_media_missing_file_raises(tmp_path):
    missing = tmp_path / "nope.mp4"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        fingerprint_media(missing)


def test_fingerprint_media_ffprobe_nonzero_exit_raises(tiny_mp4, monkeypatch):
    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="corrupt moov atom")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match=str(tiny_mp4)):
        fingerprint_media(tiny_mp4)


def test_fingerprint_media_no_video_stream_raises(tiny_mp4, monkeypatch):
    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            returncode=0,
            stdout='{"format": {"duration": "2.0"}, "streams": []}',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match=str(tiny_mp4)):
        fingerprint_media(tiny_mp4)


def test_fingerprint_media_unparseable_duration_raises(tiny_mp4, monkeypatch):
    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            returncode=0,
            stdout=(
                '{"format": {"duration": "not-a-number"}, '
                '"streams": [{"codec_type": "video", "codec_name": "h264", '
                '"width": 64, "height": 48}]}'
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match=str(tiny_mp4)):
        fingerprint_media(tiny_mp4)


def test_fingerprint_media_incomplete_stream_raises(tiny_mp4, monkeypatch):
    """A probe whose video stream lacks width/height/codec_name must fail
    naming the file. Without this, int(None) raised a bare TypeError that
    named neither the field nor the path."""

    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            returncode=0,
            stdout=(
                '{"format": {"duration": "2.0"}, '
                '"streams": [{"codec_type": "video", "codec_name": "h264"}]}'
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match=str(tiny_mp4)):
        fingerprint_media(tiny_mp4)


# --- verify_media ------------------------------------------------


def test_verify_media_passes_when_file_matches_the_fingerprint(tmp_path):
    path = tmp_path / "short.mp4"
    path.write_bytes(b"matching bytes")
    media = MediaFingerprint(
        str(path),
        path.stat().st_size,
        2.0,
        1080,
        1920,
        "h264",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    verify_media(_record(media))  # must not raise


def test_verify_media_raises_when_media_missing(tmp_path):
    missing = tmp_path / "gone.mp4"
    media = MediaFingerprint(str(missing), 1, 2.0, 1080, 1920, "h264", "a" * 64)
    with pytest.raises(MediaVerificationError, match=str(missing)):
        verify_media(_record(media))


def test_verify_media_raises_on_fingerprint_mismatch(tmp_path):
    path = tmp_path / "swapped.mp4"
    path.write_bytes(b"whatever is on disk now")
    media = MediaFingerprint(str(path), path.stat().st_size, 2.0, 1080, 1920, "h264", "b" * 64)
    with pytest.raises(MediaVerificationError, match="sha256 mismatch"):
        verify_media(_record(media))
