"""The privacy-blur branch runs at quarter resolution.

Measured: the full-frame
``boxblur`` is the wall of the blur composite; blurring a 1/4-size copy and
scaling it back measured 3.89x the full composite's fps with the masked area
at least as blurred. Only the ``[b2]`` branch changes; the base/alpha/overlay
pins and the untouched ``[b1]`` branch must not.
"""

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline.single_pass import (
    BLUR_ALPHA_FORMAT,
    BLUR_BASE_FORMAT,
    BLUR_BRANCH_DOWNSCALE,
    BLUR_OVERLAY_FORMAT,
    SinglePassProcessor,
)

KW = dict(
    color_filter="null",
    location="",
    font_path="",
    duration=6.0,
    fade_video=False,
    skip_ost=True,
)


@pytest.fixture
def proc():
    return SinglePassProcessor(Config())


def test_downscale_constant_is_four():
    assert BLUR_BRANCH_DOWNSCALE == 4


def test_blur_branch_is_blurred_at_quarter_resolution(proc):
    fc = proc._build_blur_filter_complex(blur_filter="boxblur=40:2", video_width=3840, **KW)
    assert (
        "[b2]scale=iw/4:ih/4:flags=bilinear,boxblur=10:2,scale=iw*4:ih*4:flags=bilinear[bl]" in fc
    )
    assert "boxblur=40:2" not in fc


def test_radius_is_scaled_and_floored_at_one(proc):
    fc = proc._build_blur_filter_complex(blur_filter="boxblur=2:2", video_width=1920, **KW)
    assert "boxblur=1:2" in fc  # round(2/4) = 0, floored to 1


def test_pins_and_untouched_branch_are_unchanged(proc):
    fc = proc._build_blur_filter_complex(blur_filter="boxblur=24:2", video_width=3840, **KW)
    assert f"format={BLUR_BASE_FORMAT}[base]" in fc
    assert "[base]split[b1][b2]" in fc
    assert f"[bl][m]alphamerge,format={BLUR_ALPHA_FORMAT}[bla]" in fc
    assert f"[b1][bla]overlay=format={BLUR_OVERLAY_FORMAT}[ov]" in fc
    assert "[1:v]format=gray[m]" in fc
    # the b1 branch reaches overlay with no filter of its own
    assert "[b1]scale" not in fc and "[b1]boxblur" not in fc


def test_width_not_divisible_by_four_keeps_full_resolution_blur(proc):
    fc = proc._build_blur_filter_complex(blur_filter="boxblur=24:2", video_width=1922, **KW)
    assert "[b2]boxblur=24:2[bl]" in fc


def test_height_not_divisible_by_four_keeps_full_resolution_blur(proc):
    # the scale-back must reproduce the mask's exact size, so both dimensions gate
    fc = proc._build_blur_filter_complex(
        blur_filter="boxblur=24:2", video_width=3840, video_height=2161, **KW
    )
    assert "[b2]boxblur=24:2[bl]" in fc
