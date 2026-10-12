"""Long-form burned-OST overlay word-wrap.

The long-form OST overlay rendered the location title as a SINGLE centered
``drawtext`` with no wrapping — a long ``"A to B via Road, ST"`` label (valid
under the 100-char ``lf_title`` budget) overflowed the frame. ``ost_applier``
now wraps on PIL metrics pinned to the exact ffmpeg ``FONT_PATH`` and stacks one
centered ``drawtext`` per line, all sharing the title's alpha/enable window.
"""

from gomoveshift.pipeline.ost_applier import (
    FONT_PATH,
    _LOCATION_FONTSIZE,
    _LOCATION_Y_FRAC,
    _build_drawtext_filter,
    _location_drawtext_filters,
)


def _loc_drawtexts(vfilter: str) -> list[str]:
    """The drawtext clauses that carry the location title (y anchored on
    _LOCATION_Y_FRAC), excluding brand/CTA/URL clauses."""
    return [d for d in vfilter.split(",drawtext=") if f"h*{_LOCATION_Y_FRAC}" in d]


class TestLocationDrawtextFilters:
    def test_short_location_single_line(self):
        out = _location_drawtext_filters(
            "Cork, Ireland", FONT_PATH, _LOCATION_FONTSIZE, 1920, _LOCATION_Y_FRAC, "1", "1"
        )
        assert len(out) == 1
        assert "text='Cork, Ireland'" in out[0]

    def test_long_location_wraps_to_multiple_lines(self):
        long_loc = "Cataloochee Valley to Maggie Valley via Blue Ridge Parkway, NC"
        out = _location_drawtext_filters(
            long_loc, FONT_PATH, _LOCATION_FONTSIZE, 1920, _LOCATION_Y_FRAC, "1", "1"
        )
        assert len(out) >= 2
        # Every line shares the title anchor, font, alpha + enable.
        for d in out:
            assert f"fontfile='{FONT_PATH}'" in d
            assert f"h*{_LOCATION_Y_FRAC}" in d
            assert "alpha='1'" in d
            assert "enable='1'" in d
        # Lines stack vertically (distinct y offsets above/below the anchor).
        ys = [d.split("y=")[1].split(":")[0] for d in out]
        assert len(set(ys)) == len(ys)

    def test_pins_the_passed_font(self):
        out = _location_drawtext_filters(
            "Somewhere", "/custom/font.ttf", _LOCATION_FONTSIZE, 1920, _LOCATION_Y_FRAC, "1", "1"
        )
        assert "fontfile='/custom/font.ttf'" in out[0]


class TestBuildDrawtextFilterWrapping:
    def test_montage_long_location_emits_multiple_location_drawtexts(self):
        long_loc = "Cataloochee Valley to Maggie Valley via Blue Ridge Parkway, NC"
        vfilter = _build_drawtext_filter(
            location=long_loc,
            layout={},
            font_path=FONT_PATH,
            video_width=1920,
            video_duration=120.0,
            montage_duration=15.0,
        )
        assert len(_loc_drawtexts(vfilter)) >= 2

    def test_legacy_short_location_single_drawtext(self):
        layout = {
            "location_start_sec": 0.0,
            "location_duration_sec": 5.0,
        }
        vfilter = _build_drawtext_filter(
            location="Cork, Ireland",
            layout=layout,
            font_path=FONT_PATH,
            video_width=1920,
            video_duration=120.0,
        )
        assert len(_loc_drawtexts(vfilter)) == 1
