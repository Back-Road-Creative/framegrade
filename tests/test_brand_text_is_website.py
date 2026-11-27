"""The brand line burned into reels, Shorts and long-forms reads "gomoveshift.com".

Owner, 2026-11-27: the bottom of a reel said "GoMoveShift", which tells a viewer
nothing about where to find the rest; it should name the website. Every burn
surface reads the brand words from one place (``ffmpeg_filters.BRAND_TEXT``,
the ``MediaProfile`` default and the shipped ``gomoveshift`` brand template), so
this pins all of them, the route-story film's maker's mark included, and checks
the longer text still fits inside the frame at 9:16 and 16:9.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from PIL import ImageFont

from gomoveshift.pipeline import ffmpeg_filters, ost_applier
from gomoveshift.pipeline.route_story import hud
from gomoveshift.publishing import brand_policy
from gomoveshift.publishing.brand_policy import MediaProfile

WEBSITE = "gomoveshift.com"
SHIPPED = Path(brand_policy.__file__).resolve().parent.parent / "console" / "brands"


def test_brand_text_names_the_website_everywhere_it_is_set():
    assert ffmpeg_filters.BRAND_TEXT == WEBSITE
    assert MediaProfile().brand_text == WEBSITE
    raw = yaml.safe_load((SHIPPED / "gomoveshift" / "brand.yaml").read_text())
    assert raw["media"]["brand_text"] == WEBSITE
    assert WEBSITE in hud.DEFAULT_WATERMARK_TEXT
    assert "GoMoveShift" not in hud.DEFAULT_WATERMARK_TEXT


@pytest.mark.parametrize("layout", ["short", "regular_lf"])
def test_burned_brand_line_reads_the_website(layout):
    chain = ost_applier._build_drawtext_filter(
        location="Kerry to Cork",
        layout=ost_applier.LAYOUTS[layout],
        font_path=ost_applier.FONT_PATH,
        video_width=1080 if layout == "short" else 3840,
        video_duration=60.0,
    )
    brand = next(p for p in chain.split(",drawtext=") if f"y=h*{ost_applier._BRAND_Y_FRAC}" in p)
    assert f"text='{WEBSITE}'" in brand
    assert "GoMoveShift" not in chain


@pytest.mark.parametrize("width,height", [(1080, 1920), (2160, 3840), (1920, 1080), (3840, 2160)])
def test_brand_line_fits_the_frame(width, height):
    font_path = Path(ffmpeg_filters.FONT_PATH)
    if not font_path.is_file():
        pytest.skip("the burn font is not installed on this host")
    size = int(ost_applier._BRAND_FONTSIZE * ost_applier._font_scale(width))
    left, _, right, _ = ImageFont.truetype(str(font_path), size).getbbox(WEBSITE)
    # Centred, with at least 10% of the width clear on each side.
    assert right - left <= width * 0.8, (width, height, right - left)
