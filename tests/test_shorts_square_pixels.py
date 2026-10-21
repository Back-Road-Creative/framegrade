"""Shorts and platform variants must encode square pixels (SAR 1:1).

The 9:16 crop of a 4K source is 1214x2160; scaling that to 1080x1920 makes
ffmpeg set SAR 1214:1215 to keep the display aspect, so every published Short
probed non-square. Synthetic fixtures only.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline.ffmpeg_filters import build_crop_filter
from gomoveshift.pipeline.validation.shorts_validator import ShortsValidator
from gomoveshift.publishing.tiktok_rendition import _ffmpeg_argv, _instagram_argv


@pytest.mark.parametrize("src", [(3840, 2160), (1920, 1080), (2704, 1520), (1921, 1081)])
def test_crop_filter_forces_square_pixels_after_scale(src):
    chain = build_crop_filter(*src).split(",")
    assert chain[-1] == "setsar=1"
    assert chain[0].startswith("crop=")
    assert len(chain) == 3


@pytest.mark.parametrize("builder", [_ffmpeg_argv, _instagram_argv])
def test_h264_variants_force_square_pixels(builder):
    argv = builder(Path("in.mp4"), Path("out.mp4"))
    assert argv[argv.index("-vf") + 1] == "setsar=1"


def _data(sar):
    stream = {"codec_type": "video", "width": 1080, "height": 1920}
    if sar is not None:
        stream["sample_aspect_ratio"] = sar
    return {"streams": [stream]}


@pytest.mark.parametrize("sar", ["1:1", None])
def test_resolution_gate_accepts_square_pixels(sar):
    gate = ShortsValidator(MagicMock())._check_short_resolution(Path("short_01.mp4"), _data(sar))
    assert gate.passed


@pytest.mark.parametrize("sar", ["1214:1215", "N/A", "0:1"])
def test_resolution_gate_rejects_non_square_pixels(sar):
    gate = ShortsValidator(MagicMock())._check_short_resolution(Path("short_01.mp4"), _data(sar))
    assert not gate.passed
    assert "SAR" in gate.message
