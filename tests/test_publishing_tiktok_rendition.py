"""The TikTok upload needs its own H.264 rendition — no network; one test runs real ffmpeg when it is installed."""

from __future__ import annotations

import shutil
import subprocess  # noqa: S404
from pathlib import Path

import pytest

from gomoveshift.publishing.tiktok_rendition import (
    INSTAGRAM_MAX_BITRATE_BPS,
    TikTokRenditionError,
    ensure_h264_rendition,
    ensure_instagram_rendition,
    instagram_rendition_path,
    rendition_path,
)


def _src(tmp_path: Path, name: str = "short_02.mp4", data: bytes = b"hevc-bytes") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _recorder(*, writes: bytes = b"h264-bytes"):
    """Stands in for the ffmpeg run: records argv and creates the output."""
    calls: list[list[str]] = []

    def run(argv: list[str]) -> None:
        calls.append(argv)
        Path(argv[-1]).write_bytes(writes)

    return run, calls


def test_an_h264_source_is_uploaded_as_is(tmp_path):
    """No transcode when the master is already H.264 — re-encoding a
    compliant file would only cost quality and minutes."""
    src = _src(tmp_path)
    run, calls = _recorder()
    assert ensure_h264_rendition(src, codec="h264", run=run) == src
    assert calls == []


def test_an_hevc_source_is_transcoded_to_a_sibling_rendition(tmp_path):
    """A large HEVC short can reach the TikTok inbox yet never open; the
    same short as a smaller H.264 opens and publishes. HEVC is inside TikTok's stated limits,
    so nothing rejects it — the app just never renders it, which looks
    exactly like a delivery that worked."""
    src = _src(tmp_path)
    run, calls = _recorder()

    out = ensure_h264_rendition(src, codec="hevc", run=run)

    assert out == rendition_path(src) != src
    assert out.read_bytes() == b"h264-bytes"
    assert len(calls) == 1
    argv = calls[0]
    assert argv[0] == "ffmpeg"
    assert str(src) in argv
    assert "libx264" in argv, "TikTok recommends H.264; that is the whole point"
    assert "yuv420p" in argv, "10-bit HEVC would otherwise carry into an unplayable H.264"
    assert "+faststart" in argv, "the app streams the file before the composer renders"
    assert argv[-1] == str(out)


def test_an_existing_rendition_is_reused(tmp_path):
    src = _src(tmp_path)
    run, calls = _recorder()
    first = ensure_h264_rendition(src, codec="hevc", run=run)
    second = ensure_h264_rendition(src, codec="hevc", run=run)
    assert first == second
    assert len(calls) == 1, "one transcode per short, not one per drain pass"


def test_a_rendition_older_than_its_source_is_rebuilt(tmp_path):
    """A re-rendered short must not upload the previous run's video."""
    src = _src(tmp_path)
    stale = rendition_path(src)
    stale.write_bytes(b"stale")
    import os

    os.utime(stale, (0, 0))
    run, calls = _recorder()

    out = ensure_h264_rendition(src, codec="hevc", run=run)

    assert out.read_bytes() == b"h264-bytes"
    assert len(calls) == 1


def test_a_transcode_that_writes_nothing_raises_rather_than_uploading_it(tmp_path):
    """An empty file would upload happily and land an unopenable draft —
    the failure this whole module exists to stop."""
    src = _src(tmp_path)

    def run(argv: list[str]) -> None:
        Path(argv[-1]).write_bytes(b"")

    with pytest.raises(TikTokRenditionError, match="empty"):
        ensure_h264_rendition(src, codec="hevc", run=run)


def test_a_missing_source_is_named(tmp_path):
    with pytest.raises(TikTokRenditionError, match="short_02.mp4"):
        ensure_h264_rendition(tmp_path / "short_02.mp4", codec="hevc", run=lambda argv: None)


def _arg(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_instagram_over_cap_master_is_transcoded_under_the_cap(tmp_path):
    src = _src(tmp_path)
    run, calls = _recorder()
    out = ensure_instagram_rendition(src, probe=lambda p: 29_600_000, run=run)
    assert out == instagram_rendition_path(src)
    assert out.name == "short_02.instagram-h264.mp4"
    argv = calls[0]
    assert _arg(argv, "-c:v") == "libx264"
    assert _arg(argv, "-pix_fmt") == "yuv420p"
    assert _arg(argv, "-movflags") == "+faststart"
    assert int(_arg(argv, "-maxrate").rstrip("M")) * 1_000_000 <= INSTAGRAM_MAX_BITRATE_BPS


def test_instagram_under_cap_master_is_returned_untouched(tmp_path):
    src = _src(tmp_path)
    run, calls = _recorder()
    assert ensure_instagram_rendition(src, probe=lambda p: 21_900_000, run=run) == src
    assert calls == []


def test_instagram_unknown_bitrate_transcodes(tmp_path):
    src = _src(tmp_path)
    run, calls = _recorder()
    out = ensure_instagram_rendition(src, probe=lambda p: None, run=run)
    assert out == instagram_rendition_path(src)
    assert len(calls) == 1


def test_instagram_fresh_rendition_is_reused_and_stale_rebuilt(tmp_path):
    import os

    src = _src(tmp_path)
    target = instagram_rendition_path(src)
    target.write_bytes(b"old")
    run, calls = _recorder()
    os.utime(src, (1000, 1000))
    os.utime(target, (2000, 2000))
    assert ensure_instagram_rendition(src, probe=lambda p: 30_000_000, run=run) == target
    assert calls == []
    os.utime(target, (500, 500))
    ensure_instagram_rendition(src, probe=lambda p: 30_000_000, run=run)
    assert len(calls) == 1


def test_instagram_empty_output_raises(tmp_path):
    src = _src(tmp_path)
    run, _ = _recorder(writes=b"")
    with pytest.raises(TikTokRenditionError):
        ensure_instagram_rendition(src, probe=lambda p: 30_000_000, run=run)


def test_instagram_known_bitrate_skips_the_probe(tmp_path):
    """The delivery's fingerprint already gives the average bitrate
    (size * 8 / duration, both just verified against the file), so no ffprobe
    runs; a broken probe would otherwise transcode every upload unnoticed."""
    src = _src(tmp_path)
    run, calls = _recorder()

    def probe(_path):
        raise AssertionError("probed a short whose bitrate was already known")

    assert ensure_instagram_rendition(src, bitrate_bps=21_900_000, probe=probe, run=run) == src
    assert calls == []


# 2026-11-27: the Day 10 day reel's master is 2160x3840 at 60 fps. Its Instagram rendition kept
# that frame (still tagged H.264 level 4.1, which cannot hold it; Meta's Reels spec caps width at
# 1920 px), and both posts failed: the first container sat IN_PROGRESS until it expired, the
# recovery's upload was refused ``ProcessingFailedError``. Pipeline shorts (1080x1920, 29.97 fps)
# published. The rendition is capped at 1080x1920 and 30 fps, a master over that frame is never
# uploaded as is, and a cached rendition over it is rebuilt.


def test_instagram_rendition_caps_the_frame_at_1080x1920_and_30_fps(tmp_path):
    src = _src(tmp_path)
    run, calls = _recorder()
    ensure_instagram_rendition(src, probe=lambda p: 58_000_000, frame=lambda p: None, run=run)
    argv = calls[0]
    vf = _arg(argv, "-vf")
    assert "scale=" in vf and "1080" in vf and "1920" in vf, vf
    assert "force_original_aspect_ratio=decrease" in vf, vf
    assert vf.endswith("setsar=1"), vf
    assert _arg(argv, "-fpsmax") == "30"


def test_an_under_cap_master_over_the_frame_is_still_transcoded(tmp_path):
    src = _src(tmp_path)
    run, calls = _recorder()
    out = ensure_instagram_rendition(
        src, bitrate_bps=21_900_000, frame=lambda p: (2160, 3840, 60.0), run=run
    )
    assert out == instagram_rendition_path(src) and len(calls) == 1


def test_an_under_cap_master_inside_the_frame_is_returned_untouched(tmp_path):
    src = _src(tmp_path)
    run, calls = _recorder()
    out = ensure_instagram_rendition(
        src, bitrate_bps=21_900_000, frame=lambda p: (1080, 1920, 30000 / 1001), run=run
    )
    assert out == src and calls == []


def test_a_fresh_rendition_over_the_frame_is_rebuilt(tmp_path):
    import os

    src = _src(tmp_path)
    target = instagram_rendition_path(src)
    target.write_bytes(b"4k60 rendition from before the cap")
    os.utime(src, (1000, 1000))
    os.utime(target, (2000, 2000))
    frames = {src: (2160, 3840, 60.0), target: (2160, 3840, 60.0)}
    run, calls = _recorder()
    out = ensure_instagram_rendition(src, probe=lambda p: 58_000_000, frame=frames.get, run=run)
    assert out == target and len(calls) == 1


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="no ffmpeg")
def test_a_real_4k60_master_comes_out_1080x1920_at_30_fps(tmp_path):
    src = tmp_path / "short_01.mp4"
    subprocess.run(  # noqa: S603
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=2160x3840:rate=60:duration=0.5",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-shortest",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            str(src),
        ],  # fmt: skip
        check=True,
    )
    out = ensure_instagram_rendition(src, bitrate_bps=58_000_000)
    probe = subprocess.run(  # noqa: S603
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate",
            "-of",
            "csv=p=0",
            str(out),
        ],  # fmt: skip
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert probe == "1080,1920,30/1", probe
