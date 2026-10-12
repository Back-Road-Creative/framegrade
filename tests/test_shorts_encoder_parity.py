"""Shorts encoder parity with the long-form NVENC master.

Shorts are cut from the graded 10-bit day master and the encoding-constants
docstring claims their quality "matches the long-form encode". That claim only
ever held for preset / CQ / profile / pix_fmt:

  * the rate ceiling was a flat ``8M`` / ``12M`` pair that predates the
    CQ18 + 10-bit raise — roughly HALF the long-form 1080p floor for the same
    2.07 MP frame (1080x1920 vs 1920x1080), so VBV starved the CQ the encoder
    was asking for. Long-form already fixed exactly this by moving to a
    floor x ratio model;
  * none of the long-form NVENC quality flags (lookahead, spatial/temporal AQ,
    B-frame refs, two-pass) reached the shorts command.
"""

from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline import encoding_constants as ec
from gomoveshift.pipeline.shorts_extractor import render_short_ffmpeg


def _kbps(rate_token: str) -> float:
    """Parse an ffmpeg rate token (``30000k`` / ``8M``) into kbps."""
    token = rate_token.strip()
    if token.endswith(("k", "K")):
        return float(token[:-1])
    if token.endswith(("m", "M")):
        return float(token[:-1]) * 1000.0
    return float(token) / 1000.0


# ── Rate model ───────────────────────────────────────────────────────────


class TestShortsRateModel:
    """SHORTS_MAXRATE / SHORTS_BUFSIZE derive from a floor, like long-form."""

    def test_floor_is_the_longform_1080p_floor(self):
        """Same pixel count as a 1080p LF — same bitrate floor."""
        assert ec.SHORTS_MIN_BITRATE == ec.LONGFORM_MIN_BITRATE_1080P

    def test_ratios_match_longform(self):
        assert ec.SHORTS_MAXRATE_RATIO == ec.LONGFORM_MAXRATE_RATIO
        assert ec.SHORTS_BUFSIZE_RATIO == ec.LONGFORM_BUFSIZE_RATIO

    def test_maxrate_and_bufsize_are_derived_not_flat(self):
        assert ec.SHORTS_MAXRATE == f"{int(ec.SHORTS_MIN_BITRATE * ec.SHORTS_MAXRATE_RATIO)}k"
        assert ec.SHORTS_BUFSIZE == f"{int(ec.SHORTS_MIN_BITRATE * ec.SHORTS_BUFSIZE_RATIO)}k"

    def test_maxrate_clears_the_longform_1080p_floor(self):
        """The old flat 8M ceiling sat BELOW the 15 Mbps long-form 1080p floor."""
        assert _kbps(ec.SHORTS_MAXRATE) >= ec.LONGFORM_MIN_BITRATE_1080P

    def test_bufsize_exceeds_maxrate(self):
        assert _kbps(ec.SHORTS_BUFSIZE) > _kbps(ec.SHORTS_MAXRATE)


# ── Emitted command ──────────────────────────────────────────────────────


@pytest.fixture
def fake_source(tmp_path):
    src = tmp_path / "source.mp4"
    src.write_bytes(b"\x00" * 1024)
    return src


def _render_cmd(fake_source, tmp_path) -> list[str]:
    """Return the ffmpeg command render_short_ffmpeg emits (encode stops at rc=1)."""
    with (
        patch("gomoveshift.pipeline.shorts_extractor.subprocess.run") as mock_run,
        patch("gomoveshift.pipeline.shorts_extractor._detect_nvenc_cached", return_value=True),
        patch(
            "gomoveshift.pipeline.shorts_extractor.get_video_dimensions",
            return_value=(3840, 2160),
        ),
    ):
        mock_run.return_value = MagicMock(returncode=1, stderr="stop here", stdout="")
        render_short_ffmpeg(
            source_video=fake_source,
            output_path=tmp_path / "out" / "short.mp4",
            start_time=10.0,
            duration=35.0,
            location="County Kerry, Ireland",
        )
    return list(mock_run.call_args[0][0])


class TestShortsNvencQualityFlags:
    """The long-form NVENC quality configuration reaches the shorts encode."""

    # Mirrors color_filters._nvenc_args — the long-form NVENC configuration.
    EXPECTED = {
        "-tune": "hq",
        "-rc-lookahead": "32",
        "-spatial-aq": "1",
        "-temporal-aq": "1",
        "-b_ref_mode": "2",
        "-bf": "4",
        "-multipass": "2",
    }

    @pytest.mark.parametrize("flag,value", sorted(EXPECTED.items()))
    def test_flag_present_with_longform_value(self, flag, value, fake_source, tmp_path):
        cmd = _render_cmd(fake_source, tmp_path)
        assert flag in cmd, f"{flag} missing from the shorts encode command"
        assert cmd[cmd.index(flag) + 1] == value

    def test_rate_control_args_come_from_constants(self, fake_source, tmp_path):
        cmd = _render_cmd(fake_source, tmp_path)
        assert cmd[cmd.index("-maxrate") + 1] == ec.SHORTS_MAXRATE
        assert cmd[cmd.index("-bufsize") + 1] == ec.SHORTS_BUFSIZE

    def test_still_nvenc_vbr_at_the_frozen_cq(self, fake_source, tmp_path):
        """Regression belt: the parity flags must not disturb the frozen quality."""
        cmd = _render_cmd(fake_source, tmp_path)
        assert cmd[cmd.index("-c:v") + 1] == "hevc_nvenc"
        assert cmd[cmd.index("-rc") + 1] == "vbr"
        assert cmd[cmd.index("-profile:v") + 1] == ec.SHORTS_PROFILE
        assert cmd[cmd.index("-pix_fmt") + 1] == ec.SHORTS_PIX_FMT
