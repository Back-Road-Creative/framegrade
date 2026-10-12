"""``/api/frame`` must extract the frame the TIMESTAMP names.

The endpoint used to seek with ``cap.set(CAP_PROP_POS_FRAMES, int(t * fps))``.
That arithmetic is only correct when the source is constant-frame-rate AND the
container's average frame rate is the real one. On a variable-rate source — a
GoPro capture, anything assembled from clips of differing cadence — the average
is a fiction and the seek lands hundreds of milliseconds away: measured on the
fixture below, ``t=1.5`` resolves to frame 22 instead of frame 35.

That is an operator-facing defect, not a cosmetic one: the reviewer approves an
LF/short window against the frame this endpoint renders, so a mis-seek means
approving against content the timestamp does not contain.

Both fixtures are luma ramps with a single keyframe, so a wrong frame is a
visible step rather than a rounding wobble, and each test proves its own
discrimination by asserting the neighbouring frame would fail the assertion.
"""

from __future__ import annotations

import shutil
import subprocess

import cv2
import numpy as np
import pytest

import gomoveshift.web.app as web_app

STEP = 4  # luma units between consecutive frames

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _encode(src, frames: int, extra: list[str]) -> None:
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pixel_format", "gray",
         "-video_size", "64x64", "-framerate", "30", "-i", "-", *extra,
         "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0",
         "-g", "1000", "-pix_fmt", "yuv420p", str(src)],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    for i in range(frames):
        proc.stdin.write(np.full((64, 64), i * STEP, np.uint8).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0 and src.stat().st_size > 0


@pytest.fixture
def vfr_video(tmp_path):
    """60 frames: the first 30 at 30 fps, the rest at 10 fps. Average frame rate
    (~15) matches neither half, which is exactly what breaks index arithmetic."""
    src = tmp_path / "media" / "vfr.mp4"
    src.parent.mkdir()
    _encode(
        src, 60,
        ["-vf", "setpts='if(lt(N,30), N/30/TB, (1 + (N-30)/10)/TB)'", "-fps_mode", "vfr"],
    )  # fmt: skip
    return src


@pytest.fixture
def cfr_video(tmp_path):
    """40 frames of plain 30 fps — the case the old seek already got right, kept
    so the fix is not allowed to regress it."""
    src = tmp_path / "media" / "cfr.mp4"
    src.parent.mkdir()
    _encode(src, 40, [])
    return src


def _mean_luma(jpeg: bytes) -> float:
    arr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert arr is not None and arr.size, "helper returned bytes that are not a decodable JPEG"
    return float(arr.mean())


def _frame_mean(src, n: int) -> float:
    """Ground truth for frame ``n``, selected by index rather than by seek."""
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(src), "-vf", f"select=eq(n\\,{n})",
         "-vsync", "0", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "2", "-"],
        capture_output=True,
    )  # fmt: skip
    return _mean_luma(out.stdout)


def _assert_lands_on(src, timestamp: float, n: int) -> float:
    """The grab must return the frame on screen at ``timestamp``. Whether a
    timestamp sitting exactly on a boundary resolves to ``n`` or ``n+1`` is a
    tie-break convention, not accuracy, so both are accepted — anything further
    away is a real mis-seek."""
    got = _mean_luma(web_app._read_frame_sync(src, timestamp))
    want = [_frame_mean(src, n), _frame_mean(src, n + 1)]
    assert min(abs(got - w) for w in want) < 1.5, f"got {got}, wanted one of {want}"
    return got


@pytest.mark.parametrize("timestamp,n,arithmetic_n", [(1.5, 35, 22), (2.5, 45, 38), (3.5, 55, 53)])
def test_variable_rate_source_lands_on_the_frame_the_timestamp_names(
    vfr_video, timestamp, n, arithmetic_n
):
    got = _assert_lands_on(vfr_video, timestamp, n)
    # Self-proving: ``int(timestamp * container_average_fps)`` — the seek this
    # replaced — picks ``arithmetic_n``, and that frame is far enough away that
    # it could not have satisfied the assertion above.
    assert abs(got - _frame_mean(vfr_video, arithmetic_n)) > 2.5


@pytest.mark.parametrize("n", [6, 15, 30])
def test_constant_rate_source_stays_accurate(cfr_video, n):
    """The case index arithmetic already got right — the fix must not regress it."""
    _assert_lands_on(cfr_video, (n + 0.5) / 30, n)


def test_unreadable_input_still_yields_none(tmp_path):
    bogus = tmp_path / "not-a-video.mp4"
    bogus.write_bytes(b"not a real video")
    assert web_app._read_frame_sync(bogus, 0.0) is None
