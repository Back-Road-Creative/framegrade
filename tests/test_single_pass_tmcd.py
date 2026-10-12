"""Tests for -write_tmcd 0 flag in SinglePassProcessor final ffmpeg command.

GoPro-derived MP4s have a QuickTime timecode track (codec_tag_string=tmcd,
stream index 2). Without -write_tmcd 0, the assembled FINAL.mp4 retains the
tmcd track which then causes ~2x PTS inflation when fed into the concat
demuxer in drive_assembler.

This file tests that SinglePassProcessor.process() includes -write_tmcd 0
immediately before -movflags +faststart in the final encode command.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from gomoveshift.config import Config
from gomoveshift.pipeline.single_pass import SinglePassProcessor


class TestSinglePassWriteTmcd:
    """SinglePassProcessor must include -write_tmcd 0 in the final encode command."""

    def _make_config(self):
        return Config()

    def _make_processor(self):
        return SinglePassProcessor(self._make_config())

    @patch("gomoveshift.pipeline.single_pass.validate_video")
    @patch("gomoveshift.pipeline.single_pass.run_ffmpeg_encode")
    def test_single_pass_includes_write_tmcd_zero(
        self,
        mock_encode,
        mock_validate,
        tmp_path,
    ):
        """-write_tmcd 0 must appear in the final ffmpeg command before -movflags.

        Verifies the fix for GoPro tmcd track retention in FINAL.mp4 files
        that causes ~2x PTS inflation in the drive assembler concat step.
        """
        # Capture the command passed to run_ffmpeg_encode
        captured_cmd = []

        def fake_encode(cmd, **kwargs):
            captured_cmd.extend(cmd)
            result = MagicMock()
            result.success = True
            result.stderr_tail = ""
            return result

        mock_encode.side_effect = fake_encode

        mock_val = MagicMock()
        mock_val.valid = True
        mock_val.duration = 60.0
        mock_val.file_size = 500 * 1024 * 1024  # 500 MB
        mock_val.video_codec = "hevc"
        mock_validate.return_value = mock_val

        processor = self._make_processor()

        video_info = {
            "duration": 60.0,
            "height": 1080,
            "width": 1920,
            "fps": 30.0,
            "bitrate_kbps": 0,
        }

        with (
            patch.object(processor, "_has_nvenc", return_value=False),
            patch.object(
                processor,
                "_analyze_color",
                return_value=("null", "none"),
            ),
            patch.object(processor, "_measure_audio", return_value={}),
            patch.object(processor, "_find_font", return_value=None),
        ):
            processor.process(
                video_path=tmp_path / "input.mp4",
                output_path=tmp_path / "output_FINAL.mp4",
                location="Test Location",
                video_info=video_info,
                skip_color=True,
                skip_ost=True,
                skip_audio=True,
            )

        assert captured_cmd, "run_ffmpeg_encode was never called — process() returned early"
        assert "-write_tmcd" in captured_cmd, (
            "SinglePassProcessor.process() must include -write_tmcd in the final "
            "ffmpeg command to strip the GoPro QuickTime timecode track (tmcd). "
            "Without it, concat demuxer produces ~2x PTS inflation in assembled LFs."
        )
        tmcd_idx = captured_cmd.index("-write_tmcd")
        assert captured_cmd[tmcd_idx + 1] == "0", (
            f"-write_tmcd value must be '0', got '{captured_cmd[tmcd_idx + 1]}'"
        )

    @patch("gomoveshift.pipeline.single_pass.validate_video")
    @patch("gomoveshift.pipeline.single_pass.run_ffmpeg_encode")
    def test_write_tmcd_appears_before_movflags(
        self,
        mock_encode,
        mock_validate,
        tmp_path,
    ):
        """-write_tmcd 0 must appear immediately before -movflags +faststart."""
        captured_cmd = []

        def fake_encode(cmd, **kwargs):
            captured_cmd.extend(cmd)
            result = MagicMock()
            result.success = True
            result.stderr_tail = ""
            return result

        mock_encode.side_effect = fake_encode

        mock_val = MagicMock()
        mock_val.valid = True
        mock_val.duration = 60.0
        mock_val.file_size = 500 * 1024 * 1024
        mock_val.video_codec = "hevc"
        mock_validate.return_value = mock_val

        processor = self._make_processor()

        video_info = {
            "duration": 60.0,
            "height": 1080,
            "width": 1920,
            "fps": 30.0,
            "bitrate_kbps": 0,
        }

        with (
            patch.object(processor, "_has_nvenc", return_value=False),
            patch.object(
                processor,
                "_analyze_color",
                return_value=("null", "none"),
            ),
            patch.object(processor, "_measure_audio", return_value={}),
            patch.object(processor, "_find_font", return_value=None),
        ):
            processor.process(
                video_path=tmp_path / "input.mp4",
                output_path=tmp_path / "output_FINAL.mp4",
                location="Test Location",
                video_info=video_info,
                skip_color=True,
                skip_ost=True,
                skip_audio=True,
            )

        assert captured_cmd, "run_ffmpeg_encode was never called"
        assert "-write_tmcd" in captured_cmd, "-write_tmcd must be in the command"
        assert "-movflags" in captured_cmd, "-movflags must be in the command"

        tmcd_idx = captured_cmd.index("-write_tmcd")
        movflags_idx = captured_cmd.index("-movflags")
        assert tmcd_idx < movflags_idx, (
            f"-write_tmcd must appear before -movflags; "
            f"tmcd at index {tmcd_idx}, movflags at index {movflags_idx}"
        )
