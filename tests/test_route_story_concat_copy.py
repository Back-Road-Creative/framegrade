"""The film's parts are joined, not re-encoded, and are kept near-lossless until delivery.

Previously every frame went JPEG -> x264 crf 23 part clip -> x264 crf 23
concat -> crf 18 delivery: two wasted lossy generations, and a whole-film
encode pass spent on the concat alone. Parts now share one near-lossless
video spec so the concat can stream-copy them; ``encode_final`` stays the one
lossy delivery encode.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.route_story import assembly

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _frames(directory: Path, count: int, seed: int) -> Path:
    directory.mkdir()
    rng = np.random.default_rng(seed)
    for index in range(count):
        # Detail everywhere: a flat frame survives any codec, noise does not.
        # Grey noise, so the error measured is the codec's, not yuv420p's
        # chroma subsampling (which every delivery shares regardless).
        grey = rng.integers(0, 256, size=(180, 320), dtype=np.uint8)
        pixels = np.repeat(grey[:, :, None], 3, axis=2)
        Image.fromarray(pixels).save(directory / f"{index:06d}.jpg", quality=95)
    return directory


def _framemd5(path: Path) -> list[str]:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v", "-f", "framemd5", "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        line.rsplit(",", 1)[-1].strip() for line in result.stdout.splitlines() if line[:1] != "#"
    ]


def _first_frame(path: Path) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(180, 320, 3)


@needs_ffmpeg
def test_concat_decodes_to_exactly_the_parts_it_joined(tmp_path) -> None:
    first = tmp_path / "a.mp4"
    second = tmp_path / "b.mp4"
    assembly._encode_frames(_frames(tmp_path / "fa", 12, 1), first, 30)
    assembly._encode_frames(_frames(tmp_path / "fb", 9, 2), second, 30)
    joined = tmp_path / "joined.mp4"

    assembly._concat([first, second], joined)

    assert _framemd5(joined) == _framemd5(first) + _framemd5(second)


@needs_ffmpeg
def test_a_part_clip_is_near_lossless(tmp_path) -> None:
    frames = _frames(tmp_path / "f", 3, 3)
    clip = tmp_path / "part.mp4"
    assembly._encode_frames(frames, clip, 30)
    source = np.asarray(Image.open(frames / "000000.jpg").convert("RGB"), dtype=np.int16)
    decoded = _first_frame(clip).astype(np.int16)
    # Measured on this grey noise: crf 23 costs tens of levels.
    # Under this bound is no generation a viewer can see.
    assert float(np.abs(source - decoded).mean()) < 3.0


def test_every_part_encoder_shares_one_video_spec() -> None:
    # Stream-copy is only valid when every joined part has the same codec,
    # profile, pixel format, colour tags and timescale -- so they come from
    # one list, never from two hand-kept copies.
    assert "-c:v" in assembly._PART_VIDEO_ARGS
    assert "-video_track_timescale" in assembly._PART_VIDEO_ARGS
