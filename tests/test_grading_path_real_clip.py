"""A real clip goes through the whole grading path and the grade reaches the pixels.

Every other grading test feeds hand-made statistics, hand-made LUTs or a mocked
ffmpeg, so a break between the stages (analysis that measures the wrong thing,
a LUT that is baked but never wired into the encode, an encode that drops the
filter) passes them all. This one runs the production calls, in order, on a
real file:

    sample_frames -> analyze_segments -> compute_correction   (_analyze_color)
    -> params_to_lut3d_cube -> build_filter_chain              (_bake_global_filter)
    -> encode_video                                            (libx265 Main 10)

The source is a flat, blue-cast clip, so the expected grade is known: warm the
white balance and pull the blue-over-red cast down. The assertions read the
encoded file back, not the filter string.

It skips without ffmpeg/libx265 on a workstation and FAILS in CI (``CI`` set,
as GitHub Actions does), so a broken ffmpeg install cannot turn the only
real-encode grading check into a silent skip.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

W, H, FPS, DURATION = 320, 180, 30, 30
# Flat and blue: red and green pulled down, contrast squeezed.
CAST = "colorchannelmixer=rr=0.75:gg=0.9:bb=1.0,eq=contrast=0.6:brightness=-0.05"
PROBE_TIMES = (5.0, 15.0, 25.0)


def _unavailable(reason: str) -> None:
    if os.environ.get("CI"):
        pytest.fail(f"{reason}; CI must run the real grading path, not skip it")
    pytest.skip(reason)


def _require_ffmpeg() -> None:
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        _unavailable("ffmpeg/ffprobe not on PATH")
    encoders = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30
    ).stdout
    if "libx265" not in encoders:
        _unavailable("ffmpeg lacks libx265")


def _mean_rgb(path: Path, at: float) -> np.ndarray:
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            str(at),
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-vf",
            f"scale={W}:{H}:out_range=full,format=rgb24",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        check=True,
        timeout=60,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(H, W, 3).astype(float).mean(axis=(0, 1))


def _probe(path: Path) -> dict[str, str]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height:format=duration",
            "-of",
            "default=noprint_wrappers=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


@pytest.mark.slow
def test_a_real_blue_cast_clip_is_graded_end_to_end(tmp_path: Path) -> None:
    _require_ffmpeg()
    from gomoveshift.config import Config
    from gomoveshift.pipeline import single_pass as sp
    from gomoveshift.pipeline.color_filters import encode_video

    src = tmp_path / "src.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=s={W}x{H}:r={FPS}:d={DURATION},{CAST},format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(src),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )

    proc = sp.SinglePassProcessor(Config())
    # Hosted CI has no GPU: sample with a software decode. Every other stage is real.
    with patch.object(sp, "detect_nvenc", return_value=False):
        filter_str, method, corrections, lut_dir = proc._analyze_color(src)
    try:
        assert method == "auto_global", (method, filter_str)
        assert len(corrections) == 1
        grade = corrections[0]
        # A blue cast is answered by warming: more red, less blue.
        assert grade.white_balance_r > 0 and grade.white_balance_b < 0, grade
        assert lut_dir is not None and (lut_dir / "global.cube").is_file()
        assert f"lut3d={lut_dir / 'global.cube'}" in filter_str, filter_str

        out = tmp_path / "graded.mp4"
        assert encode_video(src, out, filter_str, W, H, float(FPS), float(DURATION))
    finally:
        if lut_dir is not None:
            shutil.rmtree(lut_dir, ignore_errors=True)

    info = _probe(out)
    assert (info["codec_name"], info["pix_fmt"]) == ("hevc", "yuv420p10le"), info
    assert (int(info["width"]), int(info["height"])) == (W, H), info
    assert abs(float(info["duration"]) - DURATION) < 0.2, info

    for at in PROBE_TIMES:
        r0, _, b0 = _mean_rgb(src, at)
        r1, _, b1 = _mean_rgb(out, at)
        # Measured on this fixture: the cast is ~33 levels in, ~16 out. Require at
        # least a 30% cut so a grade that never reaches the encode (cast unchanged)
        # or one applied backwards (cast grows) fails.
        assert b0 - r0 > 20, (at, b0 - r0)
        assert b1 - r1 < 0.7 * (b0 - r0), (at, b0 - r0, b1 - r1)
