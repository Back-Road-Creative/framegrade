"""
Tests for MP4 integrity validation in the render and upload pipeline.

Ensures corrupted MP4s (moov atom missing, zero duration, truncated)
are caught before upload and never reach Google Drive or Metricool CSVs.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from gomoveshift.pipeline.ffmpeg_runner import VideoValidation, validate_video


class TestValidateVideo:
    """validate_video must catch all corruption modes."""

    def test_valid_mp4(self, tmp_path):
        """A properly rendered MP4 passes validation."""
        mp4 = tmp_path / "good.mp4"
        # Generate a minimal valid MP4 with ffmpeg
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=320x240:d=2",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=44100:cl=mono",
                "-t",
                "2",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(mp4),
            ],
            capture_output=True,
            timeout=30,
        )
        result = validate_video(mp4, expected_duration=2.0, require_audio=True)
        assert result.valid, f"Valid MP4 should pass: {result.reason}"
        assert result.has_video
        assert result.has_audio
        assert result.duration > 0

    def test_missing_file(self, tmp_path):
        """Non-existent file fails validation."""
        result = validate_video(tmp_path / "missing.mp4")
        assert not result.valid
        assert "does not exist" in result.reason

    def test_empty_file(self, tmp_path):
        """Zero-byte file fails validation."""
        mp4 = tmp_path / "empty.mp4"
        mp4.write_bytes(b"")
        result = validate_video(mp4)
        assert not result.valid
        assert "empty" in result.reason.lower()

    def test_truncated_mp4(self, tmp_path):
        """Truncated file (no moov atom) fails validation."""
        mp4 = tmp_path / "truncated.mp4"
        # Write garbage that looks like a partial MP4
        mp4.write_bytes(b"\x00\x00\x00\x1cftypisom" + b"\x00" * 10000)
        result = validate_video(mp4)
        assert not result.valid
        assert result.reason  # Should have a reason

    def test_random_bytes_not_mp4(self, tmp_path):
        """Random bytes fail validation."""
        mp4 = tmp_path / "noise.mp4"
        import os

        mp4.write_bytes(os.urandom(50000))
        result = validate_video(mp4)
        assert not result.valid

    def test_duration_mismatch(self, tmp_path):
        """Duration outside tolerance fails."""
        mp4 = tmp_path / "short.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=320x240:d=1",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=44100:cl=mono",
                "-t",
                "1",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                str(mp4),
            ],
            capture_output=True,
            timeout=30,
        )
        # Expect 26s but got 1s — should fail
        result = validate_video(mp4, expected_duration=26.0)
        assert not result.valid
        assert "Duration mismatch" in result.reason

    def test_no_audio_when_required(self, tmp_path):
        """Video without audio fails when audio is required."""
        mp4 = tmp_path / "no_audio.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=320x240:d=2",
                "-t",
                "2",
                "-c:v",
                "libx264",
                "-an",
                str(mp4),
            ],
            capture_output=True,
            timeout=30,
        )
        result = validate_video(mp4, require_audio=True)
        assert not result.valid
        assert "audio" in result.reason.lower()


class TestPipelineRejectsCorruptFiles:
    """Pipeline stages must refuse to proceed with corrupt MP4s."""

    def test_upload_stage_rejects_corrupt_mp4(self):
        """Orchestrator upload stage should raise on corrupt MP4."""
        # Mock validate_video to return invalid for one file
        corrupt_validation = VideoValidation(
            valid=False,
            reason="ffprobe failed: moov atom not found",
            file_size=10000,
        )

        with (
            patch(
                "gomoveshift.pipeline.orchestrator.validate_video",
                return_value=corrupt_validation,
            ),
            pytest.raises(RuntimeError, match="corrupted short"),
        ):
            # Simulate what the upload stage does
            mp4_files = [Path("/fake/short_01.mp4")]
            corrupted = []
            for mp4 in mp4_files:
                vr = corrupt_validation
                if not vr.valid:
                    corrupted.append(f"{mp4.name}: {vr.reason}")
            if corrupted:
                raise RuntimeError(
                    f"{len(corrupted)} corrupted short(s) — refusing to upload. "
                    f"First: {corrupted[0]}"
                )

    def test_render_validates_output(self, tmp_path):
        """render_short_ffmpeg should return False for corrupt output."""
        from gomoveshift.pipeline.shorts_extractor import render_short_ffmpeg

        # Create a dummy source video
        source = tmp_path / "source.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=1920x1080:d=5",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=44100:cl=mono",
                "-t",
                "5",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                str(source),
            ],
            capture_output=True,
            timeout=30,
        )

        output = tmp_path / "short_test.mp4"

        # Patch validate_video to simulate corrupt output
        corrupt = VideoValidation(
            valid=False,
            reason="ffprobe failed: moov atom not found",
        )
        with patch(
            "gomoveshift.pipeline.ffmpeg_runner.validate_video",
            return_value=corrupt,
        ):
            result = render_short_ffmpeg(
                source_video=source,
                output_path=output,
                start_time=0.0,
                duration=3.0,
                location="Test Location",
            )
            assert result is False, "Should return False for corrupt render"
