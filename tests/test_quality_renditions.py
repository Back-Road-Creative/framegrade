"""Rendition table + honest-resolution rule.

Literals deliberately left for a follow-up: the
3840/2160/1920x1080/1080x1920 values in encoding_constants.py, single_pass.py,
montage_builder.py, drive_phases/*, route_story/presets.py + film.py,
shorts_validator.py (reads ec.SHORTS_*), and astro_timelapse.py's
``TIMELAPSE_HEIGHT_DEFAULT`` (a render-cost cap, not a rendition).
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline import encoding_constants as ec
from gomoveshift.pipeline import quality
from gomoveshift.pipeline.quality import HONEST_RESOLUTION_RULE, RENDITIONS, rendition_for
from gomoveshift.pipeline.validation.longform_validator import LongformValidator

PKG = Path(quality.__file__).resolve().parents[1]


def _gate(width: int, height: int):
    ffprobe = MagicMock()
    ffprobe.probe.return_value = {
        "format": {"bit_rate": "60000000"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "hevc",
                "width": width,
                "height": height,
                "profile": "Main 10",
                "pix_fmt": "yuv420p10le",
                "color_range": "tv",
                "color_space": "bt709",
                "color_primaries": "bt709",
                "color_transfer": "bt709",
            },
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    }
    return LongformValidator(ffprobe).check_video_quality(Path("/fake/lf.mp4"))


def test_lf_gate_checks_width_and_aspect():
    assert _gate(3840, 2160).passed
    low = _gate(1920, 1080)
    assert not low.passed and "1920x1080" in low.message and "below 4K" in low.message
    dci = _gate(4096, 2160)
    assert not dci.passed and "4096x2160" in dci.message and "256:135" in dci.message
    tall = _gate(3840, 2400)
    assert not tall.passed and "3840x2400" in tall.message and "8:5" in tall.message


def test_rendition_table_values():
    lf = rendition_for("drive_longform_4k")
    assert (lf.width, lf.height, lf.codec, lf.aspect) == (3840, 2160, "hevc", (16, 9))
    assert lf.min_bitrate_kbps == ec.LONGFORM_MIN_BITRATE_4K
    assert rendition_for("youtube_longform").aspect is None
    assert rendition_for("youtube_longform").min_bitrate_kbps == ec.LONGFORM_MIN_BITRATE_4K
    assert rendition_for("youtube_short").aspect == (9, 16)
    assert rendition_for("route_story_short").min_bitrate_kbps == ec.SHORTS_MIN_BITRATE
    assert RENDITIONS is ec.RENDITIONS
    with pytest.raises(KeyError):
        rendition_for("nope")


def test_one_rendition_class_and_table():
    classes, tables = [], []
    for path in PKG.rglob("*.py"):
        text = path.read_text()
        classes += [path for m in re.finditer(r"^class Rendition\b", text, re.M)]
        tables += [path for m in re.finditer(r"^RENDITIONS\b[^\n]*=", text, re.M)]
    assert [p.name for p in classes] == ["encoding_constants.py"]
    assert [p.name for p in tables] == ["encoding_constants.py"]


# parser.py and shorts_validator.py only mention 1080x1920 in prose (help / docstrings);
# they carry no executable literal, so they are not scanned line-by-line.
OWNED = [
    "pipeline/validation/longform_validator.py",
    "pipeline/photos/astro_timelapse.py",
    "pipeline/route_story/layers.py",
    "config.py",
]


def test_rendition_table_is_single_source():
    pairs = re.compile(r"\b(3840|1920|1080)\s*[x×,]\s*(2160|1080|1920)\b|\b3840\b")
    offenders = []
    for rel in OWNED:
        for n, line in enumerate((PKG / rel).read_text().splitlines(), 1):
            if pairs.search(line):
                offenders.append(f"{rel}:{n}: {line.strip()}")
    assert offenders == []
    src = (PKG / OWNED[0]).read_text()
    assert "2160" not in re.sub(r"#.*", "", src)


def test_honest_resolution_rule_named():
    assert HONEST_RESOLUTION_RULE == "never claim a native resolution the source lacked"
    for rel in (
        "pipeline/photos/astro_timelapse.py",
        "pipeline/route_story/layers.py",
        "cli/parser.py",
    ):
        text = (PKG / rel).read_text()
        assert "HONEST_RESOLUTION_RULE" in text or "quality.py" in text, rel
