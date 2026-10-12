"""LF and shorts OST text get the same contrast treatment.

The two burn surfaces render the SAME brand text with different legibility:

  * shorts (``ffmpeg_filters.build_ost_filter``) draw a border **plus** a drop
    shadow — ``shadowcolor=black@0.6:shadowx=4:shadowy=4`` at 48pt, 3px at 38pt;
  * ``ost_applier._build_drawtext_filter`` — every long-form entry, and the
    apply_ost shorts layout — drew a border only.

On a bright sky a white glyph with a thin dark border loses its edge; the drop
shadow is what keeps it readable, which is why the shorts surface has always
had one. These tests pin the shadow onto every drawtext entry the OST applier
emits, at the shorts offset when the frame is 1920 wide (scale 1.0), and pin
that the existing border is NOT traded away for it.
"""

from __future__ import annotations

import pytest

from gomoveshift.pipeline import ost_applier


# Values copied from ffmpeg_filters.build_ost_filter — the shorts burn surface
# this change is bringing the long-form surface into line with.
SHORTS_SHADOW_COLOR = "black@0.6"
SHORTS_LOC_FONTSIZE = 48
SHORTS_LOC_SHADOW_PX = 4
SHORTS_CTA_FONTSIZE = 38
SHORTS_CTA_SHADOW_PX = 3


def _entries(chain: str) -> list[str]:
    """Split a comma-joined filter chain into its drawtext entries."""
    return [e for e in chain.split("drawtext=") if e.strip()]


def _lf_chain(width: int = 1920) -> str:
    return ost_applier._build_drawtext_filter(
        location="Healy Pass, County Cork",
        layout=ost_applier.LAYOUTS["regular_lf"],
        font_path="/fake/font.ttf",
        video_width=width,
        video_duration=600.0,
        montage_duration=15.0,
    )


def _short_chain(width: int = 1080) -> str:
    return ost_applier._build_drawtext_filter(
        location="Healy Pass, County Cork",
        layout=ost_applier.LAYOUTS["short"],
        font_path="/fake/font.ttf",
        video_width=width,
        video_duration=35.0,
    )


class TestEveryEntryCarriesAShadow:
    @pytest.mark.parametrize("chain_fn", [_lf_chain, _short_chain], ids=["regular_lf", "short"])
    def test_all_drawtext_entries_have_a_shadow(self, chain_fn):
        entries = _entries(chain_fn())
        assert entries, "expected at least one drawtext entry"
        for entry in entries:
            assert "shadowcolor=" in entry, f"drawtext entry without a shadow: {entry}"
            assert "shadowx=" in entry and "shadowy=" in entry, entry

    def test_shadow_colour_matches_the_shorts_surface(self):
        for entry in _entries(_lf_chain()):
            assert f"shadowcolor={SHORTS_SHADOW_COLOR}" in entry, entry

    def test_shadow_is_square(self):
        """x and y offsets are equal — a diagonal drop, as on shorts."""
        import re

        for entry in _entries(_lf_chain()):
            x = re.search(r"shadowx=(\d+)", entry)
            y = re.search(r"shadowy=(\d+)", entry)
            assert x and y, entry
            assert x.group(1) == y.group(1), entry


class TestOffsetMatchesTheShortsSurface:
    def test_location_offset_equals_the_shorts_value_at_reference_width(self):
        """At 1920 wide the font scale is 1.0, so 48pt must give the shorts 4px."""
        assert ost_applier._font_scale(1920) == 1.0
        assert ost_applier._LOCATION_FONTSIZE == SHORTS_LOC_FONTSIZE
        assert ost_applier._shadow_offset(SHORTS_LOC_FONTSIZE) == SHORTS_LOC_SHADOW_PX

    def test_cta_sized_text_reproduces_the_shorts_cta_offset(self):
        assert ost_applier._shadow_offset(SHORTS_CTA_FONTSIZE) == SHORTS_CTA_SHADOW_PX

    def test_offset_grows_with_the_font_scale(self):
        """A 4K LF scales fonts up; the shadow must scale with them, not stay 4px."""
        small = _entries(_lf_chain(width=1920))[0]
        large = _entries(_lf_chain(width=3840))[0]
        import re

        assert int(re.search(r"shadowx=(\d+)", large).group(1)) > int(
            re.search(r"shadowx=(\d+)", small).group(1)
        )

    def test_offset_is_never_zero(self):
        """A tiny watermark still needs a shadow — rounding must not erase it."""
        assert ost_applier._shadow_offset(4) >= 1


class TestBorderIsNotTradedAway:
    def test_border_still_present_on_every_entry(self):
        for entry in _entries(_lf_chain()):
            assert "borderw=" in entry and "bordercolor=" in entry, entry

    def test_title_border_weight_unchanged(self):
        """Regression belt: the shadow is additive, not a border replacement."""
        chain = _lf_chain()
        assert "borderw=3:bordercolor=black@0.8" in chain
