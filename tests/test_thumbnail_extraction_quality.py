"""Thumbnail frame extraction stops softening and stops guessing at range.

Three defects, one hop: the 4K→1080 downscale used swscale's default bicubic
(the exact softening already fixed for shorts with ``flags=lanczos``); the
tv-range video → full-range JPEG conversion was left to implicit swscale
behaviour instead of being stated; and the font search was an independent
hardcoded literal that tried Windows paths first, rather than the canonical
``ffmpeg_filters.FONT_PATH`` the OST burn uses.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from gomoveshift.pipeline import thumbnail_text_overlay as tto
from gomoveshift.pipeline.drive_phases import thumbnail_generator as tg
from gomoveshift.pipeline.encoding_constants import LONGFORM_COLOR_RANGE
from gomoveshift.pipeline.ffmpeg_filters import FONT_PATH


def _vf(cmd: list[str]) -> str:
    return cmd[cmd.index("-vf") + 1]


class TestExtractionFilter:
    def _capture(self) -> list[str]:
        seen: list[list[str]] = []

        def fake_run(cmd, **kw):
            seen.append(list(cmd))
            return MagicMock(returncode=0, stderr="")

        with patch("gomoveshift.pipeline.thumbnail_text_overlay.subprocess.run", fake_run):
            with patch.object(Path, "exists", return_value=True):
                tto.extract_frame_from_video("/src.mp4", "/out.jpg", timestamp=42)
        return seen[0]

    def test_downscale_uses_lanczos_not_default_bicubic(self):
        assert "flags=lanczos" in _vf(self._capture())

    def test_range_conversion_is_explicit_tv_to_full(self):
        vf = _vf(self._capture())
        assert f"in_range={LONGFORM_COLOR_RANGE}" in vf
        assert "out_range=pc" in vf


class TestBrightnessProbe:
    def test_probe_states_the_same_range_conversion(self):
        seen: list[list[str]] = []

        def fake_run(cmd, **kw):
            seen.append(list(cmd))
            return MagicMock(returncode=0, stderr="")

        with patch(
            "gomoveshift.pipeline.drive_phases.thumbnail_generator.subprocess.run", fake_run
        ):
            tg.probe_frame_brightness("/src.mp4", 10.0)

        vf = _vf(seen[0])
        assert f"in_range={LONGFORM_COLOR_RANGE}" in vf and "out_range=pc" in vf


class TestCanonicalFont:
    def test_canonical_burn_font_is_searched_first(self):
        """Thumbnail text must use the font the OST burn uses, not Arial."""
        assert tto._font_search_paths()[0] == FONT_PATH

    def test_get_font_returns_the_canonical_face(self):
        if not Path(FONT_PATH).is_file():  # pragma: no cover - host without DejaVu
            return
        assert tto._get_font(72, bold=True).path == FONT_PATH
