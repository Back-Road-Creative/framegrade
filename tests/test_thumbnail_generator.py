"""Thumbnail generator tests.

Verifies the new ``gomoveshift.pipeline.drive_phases.thumbnail_generator``:
  - ``probe_frame_brightness`` returns a numeric value for HEVC 10-bit inputs.
  - ``pick_body_timestamps`` picks timestamps inside the body (not the
    first/last 5%).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from gomoveshift.pipeline.drive_phases.thumbnail_generator import (
    pick_body_timestamps,
    probe_frame_brightness,
)


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


@pytest.fixture
def hevc_10bit_clip(tmp_path: Path) -> Path:
    """Encode a tiny 10-bit HEVC clip. Skip if libx265 unavailable."""
    if not _ffmpeg_available():
        pytest.skip("ffmpeg unavailable")
    out = tmp_path / "clip10.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=20:size=640x360:rate=30",
        "-c:v",
        "libx265",
        "-pix_fmt",
        "yuv420p10le",
        "-x265-params",
        "profile=main10:log-level=error",
        str(out),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        pytest.skip("ffmpeg timed out producing 10-bit clip")
    if r.returncode != 0 or not out.exists() or out.stat().st_size < 1024:
        pytest.skip(f"libx265 unavailable: {r.stderr[:200]}")
    return out


def test_probe_frame_brightness_hevc_10bit(hevc_10bit_clip: Path):
    """Brightness probe must return a non-None numeric value for 10-bit HEVC."""
    timestamps = [5, 10, 15]
    values = [probe_frame_brightness(str(hevc_10bit_clip), ts) for ts in timestamps]
    non_none = [v for v in values if v is not None]
    assert len(non_none) >= 3, f"expected 3 brightness values, got {values}"
    for v in non_none:
        assert 0.0 <= v <= 255.0


def test_pick_body_timestamps_excludes_head_tail():
    # 600s video: 5% = 30s margin → all picks in [30, 570]
    ts = pick_body_timestamps(duration=600.0, count=3)
    assert len(ts) == 3
    for t in ts:
        assert 30.0 <= t <= 570.0


def test_pick_body_timestamps_handles_short_video():
    # Very short clip — should still return count values and not crash
    ts = pick_body_timestamps(duration=10.0, count=3)
    assert len(ts) == 3
    for t in ts:
        assert 0.0 < t < 10.0


def test_pick_body_timestamps_legacy_unchanged_when_no_mask():
    """With neither mask arg supplied, the pick is byte-for-byte legacy."""
    assert pick_body_timestamps(600.0, count=3) == pick_body_timestamps(600.0, count=3)
    # Legacy 600s/3: margins 30s, evenly spaced across [30, 570].
    assert pick_body_timestamps(600.0, count=3) == [30.0, 300.0, 570.0]


def test_pick_body_timestamps_never_samples_masked_region():
    """A masked region is never sampled; every pick is unmasked."""

    # STOP covers the whole first half [0, 300]; only [300, 600] is moving.
    def is_masked(t: float) -> bool:
        return t < 300.0

    ts = pick_body_timestamps(600.0, count=3, is_masked=is_masked)
    assert len(ts) == 3
    for t in ts:
        assert not is_masked(t), f"picked a masked timestamp {t}"
        assert 30.0 <= t <= 570.0


def test_pick_body_timestamps_unmasked_intervals_restricts_sampling():
    """``unmasked_intervals`` restricts every pick to a kept range."""
    kept = [(400.0, 560.0)]
    ts = pick_body_timestamps(600.0, count=3, unmasked_intervals=kept)
    assert len(ts) == 3
    for t in ts:
        assert 400.0 <= t <= 560.0


def test_pick_body_timestamps_all_masked_falls_back_to_legacy():
    """If the whole body is masked, fall back to legacy (never crash)."""
    ts = pick_body_timestamps(600.0, count=3, is_masked=lambda t: True)
    assert ts == pick_body_timestamps(600.0, count=3)
