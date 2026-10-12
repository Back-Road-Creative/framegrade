"""
TDD Tests for encoding improvements — shorts file size reduction + long-form NVENC fix.

Change 1 (shorts): CQ from SHORTS_CRF, 10-bit (main10/p010le), muted (no audio stream),
maxrate/bufsize cap.
Change 2 (long-form): NVENC preset p7, CQ from LONGFORM_NVENC_CQ.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Change 1: Shorts encoding — reduce file size
# =============================================================================


class TestShortsDefaults:
    """Verify module-level defaults for shorts encoding."""

    def test_default_crf_matches_shorts_constant(self):
        """DEFAULT_CRF must match SHORTS_CRF from encoding_constants."""
        from gomoveshift.pipeline.encoding_constants import SHORTS_CRF
        from gomoveshift.pipeline.shorts_extractor import DEFAULT_CRF

        assert DEFAULT_CRF == SHORTS_CRF, f"Expected DEFAULT_CRF={SHORTS_CRF}, got {DEFAULT_CRF}"

    # test_default_audio_bitrate_exists was removed: shorts render MUTED
    # (audio is not encoded), so shorts_extractor no longer imports/exposes a
    # per-render audio-bitrate default — its subject no longer exists.


class TestShortsTenBitColor:
    """Shorts must encode 10-bit (main10/p010le), matching the graded master.

    Flipped from the old 8-bit contract: shorts are cut from the graded 10-bit
    day master, and 8-bit CQ 23 visibly banded graded skies.
    """

    @patch("gomoveshift.pipeline.shorts_extractor.subprocess.run")
    @patch("gomoveshift.pipeline.shorts_extractor.get_video_dimensions", return_value=(1920, 1080))
    @patch("gomoveshift.pipeline.ffmpeg_runner.validate_video")
    def test_nvenc_uses_main10_profile(self, mock_val, mock_dims, mock_run, tmp_path):
        """Encode must use profile 'main10' (10-bit), not 'main' (8-bit)."""
        from gomoveshift.pipeline.shorts_extractor import render_short_ffmpeg

        mock_val_result = MagicMock()
        mock_val_result.valid = True
        mock_val_result.duration = 30.0
        mock_val.return_value = mock_val_result

        output = tmp_path / "test.mp4"
        output.write_bytes(b"x" * 1024)

        mock_run.return_value = MagicMock(returncode=0)

        render_short_ffmpeg(
            source_video=Path("/fake/video.mp4"),
            output_path=output,
            start_time=0.0,
            duration=30.0,
            location="Test",
            use_nvenc=True,
        )

        cmd = mock_run.call_args[0][0]
        # Find profile:v value
        idx = cmd.index("-profile:v")
        assert cmd[idx + 1] == "main10", f"NVENC profile should be 'main10', got '{cmd[idx + 1]}'"

    @patch("gomoveshift.pipeline.shorts_extractor.subprocess.run")
    @patch("gomoveshift.pipeline.shorts_extractor.get_video_dimensions", return_value=(1920, 1080))
    @patch("gomoveshift.pipeline.ffmpeg_runner.validate_video")
    def test_nvenc_uses_p010le(self, mock_val, mock_dims, mock_run, tmp_path):
        """Encode must use pix_fmt 'p010le' (10-bit NVENC surface), not 'yuv420p'."""
        from gomoveshift.pipeline.shorts_extractor import render_short_ffmpeg

        mock_val_result = MagicMock()
        mock_val_result.valid = True
        mock_val_result.duration = 30.0
        mock_val.return_value = mock_val_result

        output = tmp_path / "test.mp4"
        output.write_bytes(b"x" * 1024)

        mock_run.return_value = MagicMock(returncode=0)

        render_short_ffmpeg(
            source_video=Path("/fake/video.mp4"),
            output_path=output,
            start_time=0.0,
            duration=30.0,
            location="Test",
            use_nvenc=True,
        )

        cmd = mock_run.call_args[0][0]
        idx = cmd.index("-pix_fmt")
        assert cmd[idx + 1] == "p010le", f"NVENC pix_fmt should be 'p010le', got '{cmd[idx + 1]}'"

    @patch("gomoveshift.pipeline.shorts_extractor.subprocess.run")
    @patch("gomoveshift.pipeline.shorts_extractor.get_video_dimensions", return_value=(1920, 1080))
    @patch("gomoveshift.pipeline.ffmpeg_runner.validate_video")
    def test_no_hwaccel_uses_main10_profile(self, mock_val, mock_dims, mock_run, tmp_path):
        """use_nvenc=False (CPU decode) still encodes profile 'main10'."""
        from gomoveshift.pipeline.shorts_extractor import render_short_ffmpeg

        mock_val_result = MagicMock()
        mock_val_result.valid = True
        mock_val_result.duration = 30.0
        mock_val.return_value = mock_val_result

        output = tmp_path / "test.mp4"
        output.write_bytes(b"x" * 1024)

        mock_run.return_value = MagicMock(returncode=0)

        render_short_ffmpeg(
            source_video=Path("/fake/video.mp4"),
            output_path=output,
            start_time=0.0,
            duration=30.0,
            location="Test",
            use_nvenc=False,
        )

        cmd = mock_run.call_args[0][0]
        idx = cmd.index("-profile:v")
        assert cmd[idx + 1] == "main10", f"Profile should be 'main10', got '{cmd[idx + 1]}'"

    @patch("gomoveshift.pipeline.shorts_extractor.subprocess.run")
    @patch("gomoveshift.pipeline.shorts_extractor.get_video_dimensions", return_value=(1920, 1080))
    @patch("gomoveshift.pipeline.ffmpeg_runner.validate_video")
    def test_no_hwaccel_uses_p010le(self, mock_val, mock_dims, mock_run, tmp_path):
        """use_nvenc=False (CPU decode) still encodes pix_fmt 'p010le'."""
        from gomoveshift.pipeline.shorts_extractor import render_short_ffmpeg

        mock_val_result = MagicMock()
        mock_val_result.valid = True
        mock_val_result.duration = 30.0
        mock_val.return_value = mock_val_result

        output = tmp_path / "test.mp4"
        output.write_bytes(b"x" * 1024)

        mock_run.return_value = MagicMock(returncode=0)

        render_short_ffmpeg(
            source_video=Path("/fake/video.mp4"),
            output_path=output,
            start_time=0.0,
            duration=30.0,
            location="Test",
            use_nvenc=False,
        )

        cmd = mock_run.call_args[0][0]
        idx = cmd.index("-pix_fmt")
        assert cmd[idx + 1] == "p010le", f"Pix_fmt should be 'p010le', got '{cmd[idx + 1]}'"


class TestShortsAudioBitrate:
    """Shorts render MUTED: no audio stream is encoded,
    so no audio-bitrate flag appears at all."""

    @patch("gomoveshift.pipeline.shorts_extractor.subprocess.run")
    @patch("gomoveshift.pipeline.shorts_extractor.get_video_dimensions", return_value=(1920, 1080))
    @patch("gomoveshift.pipeline.ffmpeg_runner.validate_video")
    def test_shorts_have_no_audio_bitrate(self, mock_val, mock_dims, mock_run, tmp_path):
        """Muted shorts carry no -b:a / -c:a; the encode is -an."""
        from gomoveshift.pipeline.shorts_extractor import render_short_ffmpeg

        mock_val_result = MagicMock()
        mock_val_result.valid = True
        mock_val_result.duration = 30.0
        mock_val.return_value = mock_val_result

        output = tmp_path / "test.mp4"
        output.write_bytes(b"x" * 1024)

        mock_run.return_value = MagicMock(returncode=0)

        render_short_ffmpeg(
            source_video=Path("/fake/video.mp4"),
            output_path=output,
            start_time=0.0,
            duration=30.0,
            location="Test",
            use_nvenc=False,
        )

        cmd = mock_run.call_args[0][0]
        assert "-b:a" not in cmd, f"muted shorts must have no -b:a: {cmd}"
        assert "-c:a" not in cmd, f"muted shorts must have no -c:a: {cmd}"
        assert "-an" in cmd, f"muted shorts must pass -an: {cmd}"


class TestShortsMaxrateCap:
    """Shorts should have maxrate/bufsize cap to prevent bitrate spikes."""

    @patch("gomoveshift.pipeline.shorts_extractor.subprocess.run")
    @patch("gomoveshift.pipeline.shorts_extractor.get_video_dimensions", return_value=(1920, 1080))
    @patch("gomoveshift.pipeline.ffmpeg_runner.validate_video")
    def test_nvenc_has_maxrate_cap(self, mock_val, mock_dims, mock_run, tmp_path):
        """NVENC path caps the rate from the frozen shorts constants."""
        from gomoveshift.pipeline import encoding_constants as ec
        from gomoveshift.pipeline.shorts_extractor import render_short_ffmpeg

        mock_val_result = MagicMock()
        mock_val_result.valid = True
        mock_val_result.duration = 30.0
        mock_val.return_value = mock_val_result

        output = tmp_path / "test.mp4"
        output.write_bytes(b"x" * 1024)

        mock_run.return_value = MagicMock(returncode=0)

        render_short_ffmpeg(
            source_video=Path("/fake/video.mp4"),
            output_path=output,
            start_time=0.0,
            duration=30.0,
            location="Test",
            use_nvenc=True,
        )

        cmd = mock_run.call_args[0][0]
        assert "-maxrate" in cmd, "NVENC path missing -maxrate flag"
        idx = cmd.index("-maxrate")
        assert cmd[idx + 1] == ec.SHORTS_MAXRATE, f"maxrate not constants-sourced: {cmd[idx + 1]}"

        assert "-bufsize" in cmd, "NVENC path missing -bufsize flag"
        idx = cmd.index("-bufsize")
        assert cmd[idx + 1] == ec.SHORTS_BUFSIZE, f"bufsize not constants-sourced: {cmd[idx + 1]}"

    @patch("gomoveshift.pipeline.shorts_extractor.subprocess.run")
    @patch("gomoveshift.pipeline.shorts_extractor.get_video_dimensions", return_value=(1920, 1080))
    @patch("gomoveshift.pipeline.ffmpeg_runner.validate_video")
    def test_cpu_has_maxrate_cap(self, mock_val, mock_dims, mock_run, tmp_path):
        """CPU path caps the rate from the frozen shorts constants."""
        from gomoveshift.pipeline import encoding_constants as ec
        from gomoveshift.pipeline.shorts_extractor import render_short_ffmpeg

        mock_val_result = MagicMock()
        mock_val_result.valid = True
        mock_val_result.duration = 30.0
        mock_val.return_value = mock_val_result

        output = tmp_path / "test.mp4"
        output.write_bytes(b"x" * 1024)

        mock_run.return_value = MagicMock(returncode=0)

        render_short_ffmpeg(
            source_video=Path("/fake/video.mp4"),
            output_path=output,
            start_time=0.0,
            duration=30.0,
            location="Test",
            use_nvenc=False,
        )

        cmd = mock_run.call_args[0][0]
        assert "-maxrate" in cmd, "CPU path missing -maxrate flag"
        idx = cmd.index("-maxrate")
        assert cmd[idx + 1] == ec.SHORTS_MAXRATE, f"maxrate not constants-sourced: {cmd[idx + 1]}"

        assert "-bufsize" in cmd, "CPU path missing -bufsize flag"
        idx = cmd.index("-bufsize")
        assert cmd[idx + 1] == ec.SHORTS_BUFSIZE, f"bufsize not constants-sourced: {cmd[idx + 1]}"


class TestShortsCqMatchesCrf:
    """NVENC -cq should use DEFAULT_CRF value."""

    @patch("gomoveshift.pipeline.shorts_extractor.subprocess.run")
    @patch("gomoveshift.pipeline.shorts_extractor.get_video_dimensions", return_value=(1920, 1080))
    @patch("gomoveshift.pipeline.ffmpeg_runner.validate_video")
    def test_nvenc_cq_uses_default_crf(self, mock_val, mock_dims, mock_run, tmp_path):
        """NVENC -cq value should match DEFAULT_CRF."""
        from gomoveshift.pipeline.shorts_extractor import DEFAULT_CRF, render_short_ffmpeg

        mock_val_result = MagicMock()
        mock_val_result.valid = True
        mock_val_result.duration = 30.0
        mock_val.return_value = mock_val_result

        output = tmp_path / "test.mp4"
        output.write_bytes(b"x" * 1024)

        mock_run.return_value = MagicMock(returncode=0)

        render_short_ffmpeg(
            source_video=Path("/fake/video.mp4"),
            output_path=output,
            start_time=0.0,
            duration=30.0,
            location="Test",
            use_nvenc=True,
        )

        cmd = mock_run.call_args[0][0]
        idx = cmd.index("-cq")
        assert cmd[idx + 1] == str(DEFAULT_CRF), (
            f"NVENC -cq should be '{DEFAULT_CRF}', got '{cmd[idx + 1]}'"
        )

    @patch("gomoveshift.pipeline.shorts_extractor.subprocess.run")
    @patch("gomoveshift.pipeline.shorts_extractor.get_video_dimensions", return_value=(1920, 1080))
    @patch("gomoveshift.pipeline.ffmpeg_runner.validate_video")
    def test_nvenc_preset_matches_constant(self, mock_val, mock_dims, mock_run, tmp_path):
        """NVENC -preset must be SHORTS_PRESET (p7, matching long-form), not hardcoded p5."""
        from gomoveshift.pipeline.encoding_constants import SHORTS_PRESET
        from gomoveshift.pipeline.shorts_extractor import render_short_ffmpeg

        mock_val_result = MagicMock()
        mock_val_result.valid = True
        mock_val_result.duration = 30.0
        mock_val.return_value = mock_val_result

        output = tmp_path / "test.mp4"
        output.write_bytes(b"x" * 1024)

        mock_run.return_value = MagicMock(returncode=0)

        render_short_ffmpeg(
            source_video=Path("/fake/video.mp4"),
            output_path=output,
            start_time=0.0,
            duration=30.0,
            location="Test",
            use_nvenc=True,
        )

        cmd = mock_run.call_args[0][0]
        idx = cmd.index("-preset")
        assert cmd[idx + 1] == SHORTS_PRESET, (
            f"NVENC -preset should be '{SHORTS_PRESET}', got '{cmd[idx + 1]}'"
        )
        assert cmd[idx + 1] == "p7", f"Shorts preset must be p7, got '{cmd[idx + 1]}'"


# =============================================================================
# Change 2: Long-form encoding — NVENC preset and CQ fixes
# =============================================================================


class TestLongFormNVENCQuality:
    """Long-form NVENC should use p7 with full quality params by default."""

    def test_nvenc_default_preset_is_p7(self):
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=True)
        idx = args.index("-preset")
        assert args[idx + 1] == "p7"

    def test_nvenc_default_cq_is_18(self):
        """NVENC CQ must be 18 (YouTube-optimized ~40-50 Mbps at 4K)."""
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=True)
        idx = args.index("-cq")
        assert args[idx + 1] == "18"

    def test_nvenc_has_multipass(self):
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=True)
        assert "-multipass" in args
        idx = args.index("-multipass")
        assert args[idx + 1] == "2"

    def test_nvenc_has_lookahead(self):
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=True)
        assert "-rc-lookahead" in args
        idx = args.index("-rc-lookahead")
        assert args[idx + 1] == "32"

    def test_nvenc_has_spatial_temporal_aq(self):
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=True)
        assert "-spatial-aq" in args
        assert "-temporal-aq" in args

    def test_nvenc_has_bframe_refs(self):
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=True)
        assert "-bf" in args
        assert "-b_ref_mode" in args
        idx = args.index("-bf")
        assert args[idx + 1] == "4"

    def test_fast_param_removed(self):
        import inspect

        from gomoveshift.pipeline.color import get_encoding_args

        sig = inspect.signature(get_encoding_args)
        assert "fast" not in sig.parameters, "get_encoding_args should not have 'fast' parameter"


# =============================================================================
# Regression: bt709/tv color-metadata flags must survive in encode commands.
# Dropping these tags would ship silently — YouTube then guesses range/matrix
# and the grade shifts on playback.
# =============================================================================

_BT709_TV_FLAGS = [
    ("-color_range", "tv"),
    ("-colorspace", "bt709"),
    ("-color_primaries", "bt709"),
    ("-color_trc", "bt709"),
]


class TestColorMetadataFlags:
    """bt709/tv color metadata is mandatory on long-form and shorts encodes."""

    def test_longform_constants_are_bt709_tv(self):
        """Frozen constants must stay bt709/tv — hard values, not self-reference."""
        from gomoveshift.pipeline import encoding_constants as ec

        assert ec.LONGFORM_COLOR_RANGE == "tv"
        assert ec.LONGFORM_COLORSPACE == "bt709"
        assert ec.LONGFORM_COLOR_PRIMARIES == "bt709"
        assert ec.LONGFORM_COLOR_TRC == "bt709"

    def test_nvenc_encoding_args_carry_color_metadata(self):
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=True)
        for flag, value in _BT709_TV_FLAGS:
            assert flag in args, f"{flag} missing from NVENC encoding args"
            idx = args.index(flag)
            assert args[idx + 1] == value, f"{flag} should be '{value}', got '{args[idx + 1]}'"

    def test_x265_encoding_args_carry_color_metadata(self):
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=False)
        for flag, value in _BT709_TV_FLAGS:
            assert flag in args, f"{flag} missing from x265 encoding args"
            idx = args.index(flag)
            assert args[idx + 1] == value, f"{flag} should be '{value}', got '{args[idx + 1]}'"

    @patch("gomoveshift.pipeline.shorts_extractor.subprocess.run")
    @patch("gomoveshift.pipeline.shorts_extractor.get_video_dimensions", return_value=(1920, 1080))
    @patch("gomoveshift.pipeline.ffmpeg_runner.validate_video")
    def test_shorts_encode_cmd_carries_color_metadata(
        self, mock_val, mock_dims, mock_run, tmp_path
    ):
        """Shorts encode command must tag limited-range BT.709 (tv/bt709)."""
        from gomoveshift.pipeline.shorts_extractor import render_short_ffmpeg

        mock_val_result = MagicMock()
        mock_val_result.valid = True
        mock_val_result.duration = 30.0
        mock_val.return_value = mock_val_result

        output = tmp_path / "test.mp4"
        output.write_bytes(b"x" * 1024)

        mock_run.return_value = MagicMock(returncode=0)

        render_short_ffmpeg(
            source_video=Path("/fake/video.mp4"),
            output_path=output,
            start_time=0.0,
            duration=30.0,
            location="Test",
            use_nvenc=True,
        )

        cmd = mock_run.call_args[0][0]
        for flag, value in _BT709_TV_FLAGS:
            assert flag in cmd, f"{flag} missing from shorts encode cmd"
            idx = cmd.index(flag)
            assert cmd[idx + 1] == value, f"{flag} should be '{value}', got '{cmd[idx + 1]}'"


# =============================================================================
# Color-range / color-space handling.
# =============================================================================

import json  # noqa: E402

import pytest  # noqa: E402


def _sp_processor():
    """Minimal SinglePassProcessor (config is a MagicMock — no real encode).

    Construction probes NVENC. An unavailable encoder (the production render
    holding the GPU) is an environment precondition, not a filter-string
    defect: skip with the probe's reason so the leg stays honest.
    """
    from gomoveshift.pipeline.color_filters import NVENCUnavailableError
    from gomoveshift.pipeline.single_pass import SinglePassProcessor

    config = MagicMock()
    config.output.audio_bitrate = "192k"
    try:
        return SinglePassProcessor(config)
    except NVENCUnavailableError as exc:
        pytest.skip(f"NVENC encoder unavailable on this box: {exc}")


class TestProcessorNvencPrecondition:
    """``SinglePassProcessor`` probes NVENC at construction. On a machine
    whose GPU is busy the encoder can be unavailable, so an
    unavailable encoder is an environment precondition for every filter
    assertion built on ``_sp_processor()`` -- a skip with the probe's
    reason, never a red leg and never a weakened assertion."""

    def test_unavailable_encoder_skips_with_the_probe_reason(self):
        from gomoveshift.pipeline import single_pass as sp
        from gomoveshift.pipeline.color_filters import NVENCUnavailableError

        with patch.object(
            sp, "detect_nvenc", side_effect=NVENCUnavailableError("NVENC probe failed")
        ):
            with pytest.raises(pytest.skip.Exception, match="NVENC probe failed"):
                _sp_processor()

    def test_available_encoder_constructs(self):
        from gomoveshift.pipeline import single_pass as sp

        with patch.object(sp, "detect_nvenc", return_value=True):
            assert _sp_processor()._has_nvenc() is True


class TestScaleRangeToken:
    """_scale_range_token maps ffprobe color_range → ffmpeg scale in_range."""

    def test_full_and_limited_and_unknown(self):
        from gomoveshift.pipeline.single_pass import _scale_range_token

        assert _scale_range_token("pc") == "full"
        assert _scale_range_token("jpeg") == "full"
        assert _scale_range_token("") == "full"  # untagged GoPro → full
        assert _scale_range_token("unknown") == "full"
        assert _scale_range_token("tv") == "limited"
        assert _scale_range_token("mpeg") == "limited"
        assert _scale_range_token("LIMITED") == "limited"


class TestConditionalRangeConversion:
    """The full→limited squeeze must follow the source range, not be
    unconditional — an already-limited source must NOT be double-compressed."""

    def test_full_source_squeezes_once(self):
        vf = _sp_processor()._build_video_filter(
            color_filter="null",
            location="",
            font_path="",
            duration=60.0,
            fade_video=False,
            skip_ost=True,
            source_range="full",
        )
        assert "scale=in_range=full:out_range=limited" in vf

    def test_limited_source_is_a_noop(self):
        vf = _sp_processor()._build_video_filter(
            color_filter="null",
            location="",
            font_path="",
            duration=60.0,
            fade_video=False,
            skip_ost=True,
            source_range="limited",
        )
        # in==out → ffmpeg no-op; NOT the full→limited double squeeze.
        assert "scale=in_range=limited:out_range=limited" in vf
        assert "in_range=full" not in vf

    def test_default_is_full_backward_compatible(self):
        vf = _sp_processor()._build_video_filter(
            color_filter="null",
            location="",
            font_path="",
            duration=60.0,
            fade_video=False,
            skip_ost=True,
        )
        assert "scale=in_range=full:out_range=limited" in vf


class TestSdrSourceGate:
    """_assert_sdr_source refuses HDR / non-BT.709 before the encode."""

    @pytest.mark.parametrize(
        "info",
        [
            {"color_trc": "smpte2084"},  # PQ / HDR10
            {"color_trc": "arib-std-b67"},  # HLG
            {"color_primaries": "bt2020"},  # wide gamut
            {"color_trc": "SMPTE2084", "color_primaries": "bt2020"},  # case-insensitive
        ],
    )
    def test_hdr_source_raises(self, info):
        with pytest.raises(ValueError, match="HDR / non-BT.709"):
            _sp_processor()._assert_sdr_source(info)

    @pytest.mark.parametrize(
        "info",
        [
            {},  # untagged GoPro
            {"color_trc": "bt709", "color_primaries": "bt709"},
            {"color_trc": "", "color_primaries": ""},
            {"color_trc": "bt470bg"},  # SD PAL — SDR, allowed
        ],
    )
    def test_sdr_source_passes(self, info):
        _sp_processor()._assert_sdr_source(info)  # must not raise


class TestVideoInfoColorFields:
    """_get_video_info surfaces color_range / color_trc / color_primaries."""

    def test_probe_returns_color_signalling(self):
        payload = {
            "format": {"duration": "10.0", "bit_rate": "50000000"},
            "streams": [
                {
                    "height": 2160,
                    "width": 3840,
                    "pix_fmt": "yuv420p10le",
                    "color_range": "pc",
                    "color_transfer": "bt709",
                    "color_primaries": "bt709",
                    "r_frame_rate": "30000/1001",
                }
            ],
        }
        with patch("gomoveshift.pipeline.single_pass.subprocess.run") as mrun:
            mrun.return_value = MagicMock(returncode=0, stdout=json.dumps(payload))
            info = _sp_processor()._get_video_info(Path("/fake.mp4"))
        assert info["color_range"] == "pc"
        assert info["color_trc"] == "bt709"
        assert info["color_primaries"] == "bt709"


class TestCrossfadeStitchRangeTag:
    """StopProcessor tags the crossfade output with the SOURCE range (no pixel
    conversion happens), not a hardcoded 'tv' that mislabels full-range GoPro."""

    def _stop_processor(self):
        from gomoveshift.config import Config
        from gomoveshift.pipeline.stop_processor import StopProcessor

        return StopProcessor(Config(), crossfade_duration=0.5)

    @pytest.mark.parametrize(
        "reported,expected",
        [("pc", "pc"), ("tv", "tv"), ("", "pc"), ("unknown", "pc"), ("limited", "tv")],
    )
    def test_get_color_range_maps_source(self, reported, expected):
        payload = {"streams": [{"color_range": reported}]}
        with patch("gomoveshift.pipeline.stop_processor.subprocess.run") as mrun:
            mrun.return_value = MagicMock(returncode=0, stdout=json.dumps(payload))
            assert self._stop_processor()._get_color_range(Path("/fake.mp4")) == expected

    def test_probe_failure_defaults_to_full(self):
        with patch("gomoveshift.pipeline.stop_processor.subprocess.run") as mrun:
            mrun.return_value = MagicMock(returncode=1, stdout="")
            assert self._stop_processor()._get_color_range(Path("/fake.mp4")) == "pc"
