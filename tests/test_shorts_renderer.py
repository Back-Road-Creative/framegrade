"""Tests for ShortsRenderer — FFmpeg-based clip rendering and thumbnails."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.analysis.shorts_renderer import ShortsRenderer


@pytest.fixture
def renderer():
    return ShortsRenderer()


class TestRenderClip:
    """Tests for ShortsRenderer.render_clip."""

    @patch("gomoveshift.analysis.shorts_renderer.subprocess.run")
    def test_render_clip_success(self, mock_run, renderer, tmp_path):
        """Successful render returns True when ffmpeg exits 0 and output exists."""
        source = tmp_path / "input.mp4"
        source.touch()
        output = tmp_path / "output.mp4"
        output.touch()  # Simulate ffmpeg creating the file

        mock_run.return_value = MagicMock(returncode=0)

        result = renderer.render_clip(source, output, start_time=10.0, end_time=25.0)

        assert result is True
        # Verify ffmpeg was called with correct args
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-ss" in cmd
        assert "10.0" in cmd
        assert "-t" in cmd
        assert "15.0" in cmd  # duration = 25 - 10
        assert str(source) in cmd
        assert str(output) in cmd

    @patch("gomoveshift.analysis.shorts_renderer.subprocess.run")
    def test_render_clip_ffmpeg_failure(self, mock_run, renderer, tmp_path):
        """Non-zero exit code returns False."""
        source = tmp_path / "input.mp4"
        source.touch()
        output = tmp_path / "output.mp4"

        mock_run.return_value = MagicMock(returncode=1)

        result = renderer.render_clip(source, output, start_time=0.0, end_time=10.0)
        assert result is False

    @patch("gomoveshift.analysis.shorts_renderer.subprocess.run")
    def test_render_clip_output_not_created(self, mock_run, renderer, tmp_path):
        """Returns False when ffmpeg exits 0 but output file doesn't exist."""
        source = tmp_path / "input.mp4"
        source.touch()
        output = tmp_path / "output.mp4"
        # Don't create output — simulates ffmpeg silently failing

        mock_run.return_value = MagicMock(returncode=0)

        result = renderer.render_clip(source, output, start_time=0.0, end_time=5.0)
        assert result is False

    @patch("gomoveshift.analysis.shorts_renderer.subprocess.run")
    def test_render_clip_exception(self, mock_run, renderer, tmp_path):
        """Subprocess exception (e.g. timeout) returns False."""
        source = tmp_path / "input.mp4"
        source.touch()
        output = tmp_path / "output.mp4"

        mock_run.side_effect = TimeoutError("ffmpeg timed out")

        result = renderer.render_clip(source, output, start_time=0.0, end_time=5.0)
        assert result is False

    @patch("gomoveshift.analysis.shorts_renderer.subprocess.run")
    def test_render_clip_codec_args(self, mock_run, renderer, tmp_path):
        """Verify codec/preset/quality arguments in ffmpeg command."""
        source = tmp_path / "input.mp4"
        source.touch()
        output = tmp_path / "output.mp4"
        output.touch()

        mock_run.return_value = MagicMock(returncode=0)
        renderer.render_clip(source, output, start_time=0.0, end_time=10.0)

        cmd = mock_run.call_args[0][0]
        assert "libx264" in cmd
        assert "fast" in cmd
        assert "23" in cmd
        assert "aac" in cmd
        assert "128k" in cmd


class TestExtractThumbnail:
    """Tests for ShortsRenderer.extract_thumbnail."""

    @patch("gomoveshift.analysis.shorts_renderer.subprocess.run")
    def test_extract_thumbnail_success(self, mock_run, renderer, tmp_path):
        """Successful thumbnail extraction returns True."""
        source = tmp_path / "input.mp4"
        source.touch()
        output = tmp_path / "thumb.jpg"
        output.touch()

        mock_run.return_value = MagicMock(returncode=0)

        result = renderer.extract_thumbnail(source, output, timestamp=5.5)
        assert result is True

        cmd = mock_run.call_args[0][0]
        assert "-vframes" in cmd
        assert "1" in cmd
        assert "5.5" in cmd

    @patch("gomoveshift.analysis.shorts_renderer.subprocess.run")
    def test_extract_thumbnail_failure(self, mock_run, renderer, tmp_path):
        """Non-zero exit returns False."""
        source = tmp_path / "input.mp4"
        source.touch()
        output = tmp_path / "thumb.jpg"

        mock_run.return_value = MagicMock(returncode=1)

        result = renderer.extract_thumbnail(source, output, timestamp=5.5)
        assert result is False

    @patch("gomoveshift.analysis.shorts_renderer.subprocess.run")
    def test_extract_thumbnail_exception(self, mock_run, renderer, tmp_path):
        """Subprocess exception returns False."""
        source = tmp_path / "input.mp4"
        source.touch()
        output = tmp_path / "thumb.jpg"

        mock_run.side_effect = OSError("command not found")

        result = renderer.extract_thumbnail(source, output, timestamp=0.0)
        assert result is False
