"""Privacy-blur composite FILTER construction for ``SinglePassProcessor``.

These builders produce the ``filter_complex`` that the stage-9 encode will use
(wired in the encode step) when a blur mask is present: faces/plates are blurred via
``alphamerge`` + ``overlay`` — NOT ``maskedmerge`` (which promotes the gray
mask to yuv420p with chroma pinned at 128, half-blending chroma across the
whole frame). Blur lands after color/fade but before the OST/CTA text, which
must stay sharp.
"""

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline.color_filters import cfr_filter
from gomoveshift.pipeline.single_pass import BLUR_OVERLAY_FORMAT, SinglePassProcessor

KW = dict(
    color_filter="curves=m='0/0.03 1/0.98'",
    location="County Kerry, Ireland",
    font_path="/usr/share/fonts/DejaVuSans.ttf",
    duration=600.0,
    fade_video=True,
    video_width=3840,
    skip_ost=False,
)


@pytest.fixture
def proc():
    return SinglePassProcessor(Config())


class TestBlurFilter:
    @pytest.mark.parametrize("width,expected", [(1920, "boxblur=12:2"), (3840, "boxblur=24:2")])
    def test_radius_scales_with_width(self, proc, width, expected):
        assert proc._build_blur_filter(width) == expected

    def test_radius_floored_and_configurable(self, proc):
        proc.config.drive.blur_strength = 1
        assert proc._build_blur_filter(320) == "boxblur=2:2"  # max(2, round(0.16))
        proc.config.drive.blur_strength = 30
        assert proc._build_blur_filter(1920) == "boxblur=30:2"


class TestBlurFilterComplex:
    def test_shape_and_ordering(self, proc):
        fc = proc._build_blur_filter_complex(blur_filter="boxblur=24:2", **KW)
        assert "alphamerge" in fc and "overlay" in fc
        assert "maskedmerge" not in fc  # the chroma-corrupting design we rejected
        assert "[1:v]format=gray[m]" in fc and "[0:v]" in fc  # mask is input 1
        # blur sits after color/scale/fade, before the text overlays
        assert fc.index("scale=in_range=full") < fc.index("split") < fc.index("boxblur")
        assert fc.index("boxblur") < fc.index("drawtext")
        assert fc.rstrip().endswith("[vout]")

    def test_skip_ost_adds_no_text_between_overlay_and_vout(self, proc):
        kw = {**KW, "skip_ost": True, "location": "", "fade_video": False}
        fc = proc._build_blur_filter_complex(blur_filter="boxblur=24:2", **kw)
        assert "drawtext" not in fc
        # The composite pins its overlay format and runs straight into the
        # tail; with no OST the tail is only the CFR conversion, which every
        # chain ends with.
        assert f"overlay=format={BLUR_OVERLAY_FORMAT}[ov]" in fc
        assert fc.rstrip().endswith(f"[ov]{cfr_filter()}[vout]")

    def test_overlays_identical_to_no_blur_path(self, proc):
        """Drift guard: blur fc draws the exact same OST/CTA text as the -vf path."""
        vf = proc._build_video_filter(**KW)
        fc = proc._build_blur_filter_complex(blur_filter="boxblur=24:2", **KW)
        vf_dt = [f for f in vf.split(",") if f.startswith("drawtext")]
        # Split on the label, not on the overlay's option string — the latter
        # carries pinned pixel formats that are free to change.
        fc_tail = fc.split(";[ov]")[1].rsplit("[vout]", 1)[0]
        fc_dt = [f for f in fc_tail.split(",") if f.startswith("drawtext")]
        assert vf_dt == fc_dt and len(vf_dt) >= 2
