"""A cached episode concat is reused only when it is a whole, readable MP4.

Failure mode: a run stopped mid-concat left a
truncated ``_stages/episode_01_2026-03-04.mp4`` (no moov atom). The relaunch
reused it because it was non-empty, and Phase A died with "produced ZERO score
samples" after the whole ingest. The concat now writes to a partial name and
renames on success, and a cached concat that does not probe to the episode's
length is set aside and rebuilt.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.ingest import Episode, GoProChapter, GoProRecording
from gomoveshift.pipeline.ingest.concatenator import concat_episode

_PROBE = "gomoveshift.pipeline.ingest.concatenator.get_clip_duration"
_CT = datetime(2026, 3, 4, 8, 0)


def _episode(tmp_path: Path) -> Episode:
    rec = GoProRecording(recording_num=62)
    rec.chapters = []
    for n, dur in ((1, 1200.0), (2, 1350.0)):
        f = tmp_path / f"GX0{n}0062.MP4"
        f.write_bytes(b"\x00" * 64)
        rec.chapters.append(GoProChapter(f, n, 62, dur, _CT))
    return Episode(index=1, recordings=[rec], date="2026-03-04")


def _ffmpeg_writes(calls: list):
    def _run(cmd, **_kw):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"\x01" * 128)
        return MagicMock(returncode=0, stderr="")

    return _run


def test_truncated_cache_is_set_aside_and_rebuilt(tmp_path):
    ep = _episode(tmp_path)
    out = tmp_path / "episode_01_2026-03-04.mp4"
    out.write_bytes(b"\x00" * 4096)  # killed mid-write: non-empty, unreadable
    calls: list = []
    with (
        patch(_PROBE, return_value=0.0),
        patch("subprocess.run", side_effect=_ffmpeg_writes(calls)),
    ):
        result = concat_episode(ep, tmp_path)
    assert result == out
    assert len(calls) == 1
    assert (tmp_path / "episode_01_2026-03-04.mp4.invalid").exists()
    assert out.read_bytes() == b"\x01" * 128


def test_short_cache_is_rebuilt(tmp_path):
    ep = _episode(tmp_path)
    out = tmp_path / "episode_01_2026-03-04.mp4"
    out.write_bytes(b"\x00" * 4096)
    calls: list = []
    with (
        patch(_PROBE, return_value=1200.0),
        patch("subprocess.run", side_effect=_ffmpeg_writes(calls)),
    ):
        concat_episode(ep, tmp_path)
    assert len(calls) == 1


def test_whole_cache_is_reused_without_ffmpeg(tmp_path):
    ep = _episode(tmp_path)
    out = tmp_path / "episode_01_2026-03-04.mp4"
    out.write_bytes(b"\x00" * 4096)
    with patch(_PROBE, return_value=2549.4), patch("subprocess.run") as run:
        result = concat_episode(ep, tmp_path)
    assert result == out
    run.assert_not_called()
    assert not (tmp_path / "episode_01_2026-03-04.mp4.invalid").exists()


def test_ffmpeg_writes_a_partial_name_then_renames(tmp_path):
    ep = _episode(tmp_path)
    out = tmp_path / "episode_01_2026-03-04.mp4"
    calls: list = []
    with patch("subprocess.run", side_effect=_ffmpeg_writes(calls)):
        result = concat_episode(ep, tmp_path)
    assert result == out and out.exists()
    target = Path(calls[0][-1])
    assert target != out
    assert target.suffix == ".mp4"  # ffmpeg picks the muxer from the extension
    assert not target.exists()


def test_failed_ffmpeg_leaves_no_file_at_the_final_name(tmp_path):
    ep = _episode(tmp_path)
    out = tmp_path / "episode_01_2026-03-04.mp4"

    def _run(cmd, **_kw):
        Path(cmd[-1]).write_bytes(b"\x01" * 16)  # partial bytes, then die
        return MagicMock(returncode=1, stderr="Killed")

    with patch("subprocess.run", side_effect=_run), pytest.raises(RuntimeError):
        concat_episode(ep, tmp_path)
    assert not out.exists()
