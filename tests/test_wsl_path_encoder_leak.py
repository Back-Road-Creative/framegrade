"""Tests for three related regressions: WSL copy path, encoder metadata leak, hardcoded cache path.

1. WSL copy failure leaves ctx.current_video pointing to /mnt/ path
2. Encoder metadata file not cleaned up in finally block
3. Hardcoded path to sibling GPS project cache
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── WSL copy failure leaves bad path ─────────────────────────


class TestWSLCopyPathAssignment:
    """ctx.current_video must only be set AFTER successful WSL copy."""

    def _make_ctx(self, tmp_path):
        """Create a minimal PipelineContext-like object for testing."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.context import PipelineContext, PipelinePaths

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        staging_dir = tmp_path / "staging"
        staging_dir.mkdir()

        paths = PipelinePaths(
            output_base=tmp_path,
            project_dir=project_dir,
            stages_dir=tmp_path / "stages",
            state_file=tmp_path / "state.json",
            staging_dir=staging_dir,
        )

        # Create a fake source video
        source = tmp_path / "source.mp4"
        source.write_bytes(b"\x00" * 1024)

        config = Config()
        ctx = PipelineContext(
            project_name="test_project",
            source_video=source,
            source_hash="abc123",
            config=config,
            location="Test Location",
            publish_day="Monday",
            paths=paths,
            current_video=source,
        )
        # Fake video properties
        ctx.video_properties = MagicMock(
            bitrate_kbps=5000, duration=60.0, height=1080, width=1920, fps=30.0
        )
        return ctx

    def test_wsl_copy_success_sets_wsl_path(self):
        """ctx.current_video = final_output must NOT appear before the /mnt/ WSL check."""
        import inspect

        from gomoveshift.pipeline.stages.single_pass_encode import SinglePassEncodeStage

        source = inspect.getsource(SinglePassEncodeStage.run)
        lines = source.split("\n")
        # Find the quality gate section and the /mnt/ check
        quality_gate_line = None
        mnt_check_line = None
        bare_assignment_line = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "QUALITY GATE" in stripped:
                quality_gate_line = i
            if 'str(final_output).startswith("/mnt/")' in stripped:
                mnt_check_line = i
            # Look for bare assignment between quality gate and /mnt/ check
            if (
                quality_gate_line is not None
                and mnt_check_line is None
                and stripped == "ctx.current_video = final_output"
            ):
                bare_assignment_line = i

        assert bare_assignment_line is None, (
            "ctx.current_video = final_output appears BEFORE the /mnt/ WSL check "
            f"(line ~{bare_assignment_line}). A failed WSL copy would leave "
            "ctx.current_video pointing to the /mnt/ path."
        )

    def test_wsl_copy_failure_warning_mentions_original_path(self):
        """When WSL copy fails, the warning should mention using original /mnt/ path."""
        import inspect

        from gomoveshift.pipeline.stages.single_pass_encode import SinglePassEncodeStage

        source = inspect.getsource(SinglePassEncodeStage.run)
        assert "using original /mnt/ path" in source, (
            "WSL copy failure path should log that it's using the original /mnt/ path"
        )


class TestSkipColorWarnsUngraded:
    """--skip-color yields an UNGRADED FINAL (color_filter='null'); the stage
    must warn loudly so a diagnostic output can't be mistaken for a production
    master."""

    def test_skip_color_path_warns_ungraded(self):
        import inspect

        from gomoveshift.pipeline.stages.single_pass_encode import SinglePassEncodeStage

        source = inspect.getsource(SinglePassEncodeStage.run)
        assert "UNGRADED" in source, "the --skip-color path must warn that the FINAL is ungraded"
        # The warning must be guarded by skip_color so it never fires on a
        # normal, graded run.
        idx_guard = source.find("if skip_color:")
        idx_warn = source.find("UNGRADED")
        assert 0 <= idx_guard < idx_warn, "the UNGRADED warning must be guarded by `if skip_color:`"


# ── Encoder metadata file never cleaned up ───────────────────


class TestEncoderMetadataCleanup:
    """metadata_file must be cleaned up in the finally block of concatenate()."""

    def test_metadata_file_cleaned_in_finally(self):
        """The finally block must clean up metadata_file, not just concat_file."""
        import inspect

        from gomoveshift.assembly.encoder import VideoEncoder

        source = inspect.getsource(VideoEncoder.concatenate)
        # Find the finally block and check it references metadata_file cleanup
        finally_idx = source.index("finally:")
        finally_block = source[finally_idx:]
        assert "metadata_file" in finally_block, (
            "The finally block of concatenate() must clean up metadata_file"
        )

    def test_concatenate_with_chapters_cleans_metadata(self, tmp_path):
        """Integration: metadata file should not exist after concatenate() returns."""
        from gomoveshift.assembly.encoder import Chapter, VideoEncoder

        encoder = VideoEncoder()
        # Create fake segment files
        seg1 = tmp_path / "seg1.mp4"
        seg1.write_bytes(b"\x00" * 100)

        output = tmp_path / "output.mp4"
        chapters = [Chapter(start_time=0.0, title="Start")]

        metadata_file = output.parent / f"{output.stem}_chapters.txt"

        # The encode will fail (not real video), but finally block should still clean up
        encoder.concatenate([seg1], output, chapters=chapters)

        assert not metadata_file.exists(), (
            f"metadata_file {metadata_file} should be cleaned up in finally block"
        )


# ── Hardcoded path to sibling project ────────────────────────


class TestChapterGeneratorCachePath:
    """ChapterGenerator must not hardcode path to sibling GPS project."""

    def test_no_hardcoded_gps_cache_path(self):
        """generator.py must NOT contain hardcoded path to the sibling GPS project."""
        import inspect

        from gomoveshift.pipeline.chapters.generator import ChapterGenerator

        source = inspect.getsource(ChapterGenerator.__init__)
        assert "/home/" not in source, (
            "ChapterGenerator.__init__ must not hardcode path to sibling GPS project"
        )

    def test_config_has_geocoding_cache_dir(self):
        """DriveConfig should have a geocoding_cache_dir field."""
        from gomoveshift.config import DriveConfig

        dc = DriveConfig()
        assert hasattr(dc, "geocoding_cache_dir"), (
            "DriveConfig must have a geocoding_cache_dir field"
        )
        # Default should point to the known GPS cache
        assert (
            "geocoding" in dc.geocoding_cache_dir.lower()
            or "cache" in dc.geocoding_cache_dir.lower()
        )

    def test_cache_falls_back_to_temp_when_path_missing(self):
        """When configured cache path doesn't exist, fall back to temp directory."""
        from gomoveshift.config import Config

        config = Config()
        # Override to a non-existent path
        config.drive.geocoding_cache_dir = "/nonexistent/path/.cache"

        with patch("gomoveshift.pipeline.chapters.generator.NominatimRoadsClient"):
            with patch("gomoveshift.pipeline.chapters.generator.GeocodingCache") as mock_cache:
                from gomoveshift.pipeline.chapters.generator import ChapterGenerator

                gen = ChapterGenerator(config=config)
                # The cache_path passed to GeocodingCache should NOT be under /nonexistent/
                call_kwargs = mock_cache.call_args
                actual_path = str(
                    call_kwargs[1].get("cache_path", call_kwargs[0][0] if call_kwargs[0] else "")
                )
                assert "/nonexistent/" not in actual_path, (
                    f"Should fall back to temp dir, not use missing path: {actual_path}"
                )
