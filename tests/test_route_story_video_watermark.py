"""The placed source clip carries the maker's mark too -- the one film surface
that is never saved through ``assembly._save_frame``."""

from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.route_story import assembly

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _clip(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=0x305070:s=320x180:d=0.5:r=10",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip


def _first_frame(path: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1", "-f", "image2pipe",
         "-vcodec", "png", "-"],
        check=True,
        capture_output=True,
    ).stdout  # fmt: skip
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.int16)


@needs_ffmpeg
def test_the_source_clip_is_marked_bottom_right_only(tmp_path) -> None:
    source = tmp_path / "clip.mp4"
    _clip(source)
    plain = tmp_path / "plain.mp4"
    marked = tmp_path / "marked.mp4"

    assembly._encode_source_video(source, plain, 10, 320, 180)
    assembly._encode_source_video(source, marked, 10, 320, 180, watermark="Made with GoMoveShift")

    before, after = _first_frame(plain), _first_frame(marked)
    corner = (slice(150, 180), slice(200, 320))
    assert np.abs(after[corner] - before[corner]).max() > 60
    assert np.abs(after[:90, :160] - before[:90, :160]).max() <= 2
    assert not list(tmp_path.glob("*.png")), "the mark's scratch image outlived the encode"
