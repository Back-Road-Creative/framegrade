"""A raw concat retries once when the drvfs mount returns ENOMEM.

Failure mode: ffmpeg writing a ~25 GB stream-copy concat onto a WSL
``/mnt/c`` mount got ``Cannot allocate memory`` from the filesystem with RAM
free. It is a transient write failure, so one clean retry is right; any other
failure, or a second ENOMEM, still fails at once.
"""

import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.day_master_encoder import _concat_raw_episodes
from gomoveshift.pipeline.ffmpeg_runner import run_ffmpeg_retry_enomem
from gomoveshift.pipeline.ingest import Episode, GoProChapter, GoProRecording
from gomoveshift.pipeline.ingest.concatenator import concat_episode
from datetime import datetime

_ENOMEM = (
    "[aost#0:1/copy @ 0x1] Error submitting a packet to the muxer: Cannot allocate memory\n"
    "[out#0/mp4 @ 0x2] Task finished with error code: -12 (Cannot allocate memory)\n"
)
_OTHER = "[in#0 @ 0x1] Invalid data found when processing input\n"


class _Ffmpeg:
    """Fake subprocess.run: one scripted (rc, stderr) per call; writes the output."""

    def __init__(self, script, text=True):
        self.script, self.calls, self.text = list(script), [], text
        self.seen_at_call = []

    def __call__(self, cmd, **_kw):
        self.calls.append(cmd)
        out = Path(cmd[-1])
        self.seen_at_call.append(out.exists())
        rc, err = self.script.pop(0)
        out.write_bytes(b"\x01" * 128)
        return MagicMock(returncode=rc, stderr=err if self.text else err.encode())


def test_retries_once_on_enomem_and_removes_partial(tmp_path, caplog):
    out = tmp_path / "x.mp4"
    fake = _Ffmpeg([(1, _ENOMEM), (0, "")])
    with patch.object(subprocess, "run", fake), caplog.at_level(logging.WARNING):
        result = run_ffmpeg_retry_enomem(["ffmpeg", "-i", "a", str(out)], out, timeout=5)
    assert result.returncode == 0
    assert len(fake.calls) == 2
    assert fake.seen_at_call == [False, False]  # partial removed before the retry
    assert any(str(out) in r.message and r.levelno == logging.WARNING for r in caplog.records)


def test_non_enomem_failure_is_not_retried(tmp_path):
    out = tmp_path / "x.mp4"
    fake = _Ffmpeg([(1, _OTHER), (0, "")])
    with patch.object(subprocess, "run", fake):
        result = run_ffmpeg_retry_enomem(["ffmpeg", str(out)], out, timeout=5)
    assert result.returncode == 1
    assert len(fake.calls) == 1


def test_second_enomem_fails(tmp_path):
    out = tmp_path / "x.mp4"
    fake = _Ffmpeg([(1, _ENOMEM), (1, _ENOMEM), (0, "")])
    with patch.object(subprocess, "run", fake):
        result = run_ffmpeg_retry_enomem(["ffmpeg", str(out)], out, timeout=5)
    assert result.returncode == 1
    assert len(fake.calls) == 2


def test_error_code_minus_12_alone_is_recognised(tmp_path):
    # ffmpeg prints "error code: -12" with a colon; the errno text may be absent.
    out = tmp_path / "x.mp4"
    fake = _Ffmpeg([(1, "[out#0/mp4 @ 0x2] Task finished with error code: -12\n"), (0, "")])
    with patch.object(subprocess, "run", fake):
        result = run_ffmpeg_retry_enomem(["ffmpeg", str(out)], out, timeout=5)
    assert result.returncode == 0
    assert len(fake.calls) == 2


def test_bytes_stderr_is_recognised(tmp_path):
    out = tmp_path / "x.mp4"
    fake = _Ffmpeg([(1, _ENOMEM), (0, "")], text=False)
    with patch.object(subprocess, "run", fake):
        result = run_ffmpeg_retry_enomem(["ffmpeg", str(out)], out, timeout=5)
    assert result.returncode == 0


def test_day_master_raw_concat_retries(tmp_path):
    src = tmp_path / "ep.mp4"
    src.write_bytes(b"\x00")
    fake = _Ffmpeg([(1, _ENOMEM), (0, "")], text=False)
    with patch.object(subprocess, "run", fake):
        ok = _concat_raw_episodes([src], tmp_path / "raw.mp4", tmp_path / "work")
    assert ok is True
    assert len(fake.calls) == 2


def test_episode_concat_retries(tmp_path):
    rec = GoProRecording(recording_num=62)
    rec.chapters = []
    for n in (1, 2):
        f = tmp_path / f"GX0{n}0062.MP4"
        f.write_bytes(b"\x00" * 64)
        rec.chapters.append(GoProChapter(f, n, 62, 1200.0, datetime(2026, 3, 4, 8, 0)))
    ep = Episode(index=1, recordings=[rec], date="2026-03-04")
    fake = _Ffmpeg([(1, _ENOMEM), (0, "")])
    with patch.object(subprocess, "run", fake):
        path = concat_episode(ep, tmp_path / "stages")
    assert path.exists()
    assert len(fake.calls) == 2


def test_episode_concat_other_failure_raises_at_once(tmp_path):
    rec = GoProRecording(recording_num=62)
    rec.chapters = []
    for n in (1, 2):
        f = tmp_path / f"GX0{n}0062.MP4"
        f.write_bytes(b"\x00" * 64)
        rec.chapters.append(GoProChapter(f, n, 62, 1200.0, datetime(2026, 3, 4, 8, 0)))
    ep = Episode(index=1, recordings=[rec], date="2026-03-04")
    fake = _Ffmpeg([(1, _OTHER), (0, "")])
    with patch.object(subprocess, "run", fake), pytest.raises(RuntimeError):
        concat_episode(ep, tmp_path / "stages")
    assert len(fake.calls) == 1
