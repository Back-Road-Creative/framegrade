"""OST-render cleanups: montage clamp, empty-location hard-fail, shared escaping.

* Montage clamp — clamp ``montage_duration <= video_duration`` so a sub-15s LF
  fades the title out within the clip instead of holding it past EOF.
* Empty location — an empty ``lf_location`` reaching ``apply_ost`` hard-fails loud
  (was a silent stream-copy SUCCESS shipping a title-less video).
* Shared escaping — long-form ``_escape_drawtext`` now delegates to the shared
  ``escape_ffmpeg_text`` (escapes ``{``/``}``, ``'`` parity).
"""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from gomoveshift.pipeline import ost_applier
from gomoveshift.pipeline.ffmpeg_filters import escape_ffmpeg_text


def _find_location_entry(chain: str) -> str:
    for e in chain.split("drawtext="):
        if f"y=h*{ost_applier._LOCATION_Y_FRAC}" in e:
            return e
    raise AssertionError(f"LOCATION drawtext entry not found:\n{chain}")


def _extract_enable_expr(entry: str) -> str:
    m = re.search(r"enable='([^']*)'", entry)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# montage_duration clamped to video_duration
# ---------------------------------------------------------------------------


class TestMontageDurationClampedToVideo:
    def test_sub_15s_lf_clamps_montage_to_video_duration(self):
        """A 10s LF with the fixed 15s montage_duration must clamp the
        enable window + crossfade to the video length so the title fades
        out within the clip rather than holding to t=15 (past EOF)."""
        chain = ost_applier._build_drawtext_filter(
            location="Sample Location",
            layout=ost_applier.LAYOUTS["regular_lf"],
            font_path=ost_applier.FONT_PATH,
            video_width=3840,
            video_duration=10.0,
            montage_duration=15.0,
            crossfade_sec=0.5,
        )
        loc = _find_location_entry(chain)
        assert _extract_enable_expr(loc) == "between(t,0,10.0)", (
            "sub-15s LF must clamp the LOCATION enable window to video_duration"
        )

    def test_normal_lf_montage_window_unchanged(self):
        """An 817s LF keeps the full 15s montage window (no clamp)."""
        chain = ost_applier._build_drawtext_filter(
            location="Sample Location",
            layout=ost_applier.LAYOUTS["regular_lf"],
            font_path=ost_applier.FONT_PATH,
            video_width=3840,
            video_duration=817.0,
            montage_duration=15.0,
            crossfade_sec=0.5,
        )
        loc = _find_location_entry(chain)
        assert _extract_enable_expr(loc) == "between(t,0,15.0)"


# ---------------------------------------------------------------------------
# Empty location hard-fails instead of silent stream-copy SUCCESS
# ---------------------------------------------------------------------------


class TestEmptyLocationHardFails:
    def _make_input(self, tmp_path):
        p = tmp_path / "in.mp4"
        p.write_bytes(b"\x00" * 1024)
        return p

    def test_empty_location_raises(self, tmp_path):
        in_mp4 = self._make_input(tmp_path)
        with patch("gomoveshift.pipeline.ost_applier.Path.exists", return_value=True):
            with pytest.raises(ValueError, match="OST-EMPTY-LOCATION"):
                ost_applier.apply_ost(
                    input_path=in_mp4,
                    output_path=tmp_path / "out.mp4",
                    location="",
                    layout="regular_lf",
                )

    def test_whitespace_location_raises(self, tmp_path):
        in_mp4 = self._make_input(tmp_path)
        with patch("gomoveshift.pipeline.ost_applier.Path.exists", return_value=True):
            with pytest.raises(ValueError, match="OST-EMPTY-LOCATION"):
                ost_applier.apply_ost(
                    input_path=in_mp4,
                    output_path=tmp_path / "out.mp4",
                    location="   ",
                    layout="regular_lf",
                )

    def test_empty_location_does_not_stream_copy(self, tmp_path):
        """The old silent-success path called _stream_copy; it must not now.
        (Font-missing graceful degradation — the surviving stream-copy path —
        is covered by test_ost_applier.test_stream_copy_path_has_no_loudnorm.)"""
        in_mp4 = self._make_input(tmp_path)
        with (
            patch("gomoveshift.pipeline.ost_applier.Path.exists", return_value=True),
            patch("gomoveshift.pipeline.ost_applier._stream_copy") as mock_copy,
        ):
            with pytest.raises(ValueError):
                ost_applier.apply_ost(
                    input_path=in_mp4,
                    output_path=tmp_path / "out.mp4",
                    location="",
                    layout="regular_lf",
                )
        mock_copy.assert_not_called()


# ---------------------------------------------------------------------------
# Single shared escape function; { } escaped; ' parity
# ---------------------------------------------------------------------------


class TestEscapeConsolidation:
    def test_escape_drawtext_is_the_shared_function(self):
        """_escape_drawtext must delegate to the shared escape_ffmpeg_text
        so the two surfaces cannot diverge again."""
        for sample in ["plain", "100% off", "Route {A}", "O'Brien's Pass", "a:b"]:
            assert ost_applier._escape_drawtext(sample) == escape_ffmpeg_text(sample)

    def test_long_form_escapes_curly_braces(self):
        """Long-form path now escapes { } (was a render-time specifier
        injection on a literal 'Route {A}')."""
        assert ost_applier._escape_drawtext("Route {A}") == "Route \\{A\\}"

    def test_apostrophe_uses_standard_ffmpeg_escape(self):
        """The divergent '-escape collapses to the documented '\\'' form."""
        assert ost_applier._escape_drawtext("O'Brien") == "O'\\''Brien"
