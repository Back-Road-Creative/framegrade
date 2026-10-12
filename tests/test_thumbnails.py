"""Tests for thumbnail generation quality gate.

The stage runs upstream of the Drive-upload stage, so breakage caught
here costs a re-run instead of shipping a video with no usable
thumbnail. Covers:
- generation raising -> stage FAILS (never "success" with no output)
- empty / missing styled dir -> stage FAILS
- zero-byte or < 10KB (corrupt) thumbnail -> stage FAILS
- healthy thumbnails -> stage succeeds (no-false-positive guard)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


MIN_THUMBNAIL_SIZE = 10 * 1024  # 10KB


@pytest.fixture
def thumb_stage():
    """Import and instantiate the thumbnail stage."""
    from gomoveshift.pipeline.stages.thumbnails import ThumbnailGenerationStage

    return ThumbnailGenerationStage()


@pytest.fixture
def ctx_with_thumbdir(tmp_path):
    """Build a minimal PipelineContext mock with a temp project dir."""
    ctx = MagicMock()
    ctx.paths.project_dir = tmp_path
    ctx.video_properties = None
    ctx.current_video = Path("/fake/video.mp4")
    ctx.route_title = "Test Route"
    ctx.location = "Somewhere"
    ctx.project_name = "test"
    ctx.skip_stages = set()
    ctx.stages_completed = set()
    # Ensure the styled dir exists
    (tmp_path / "thumbnails" / "styled").mkdir(parents=True, exist_ok=True)
    return ctx


def _styled(ctx, valid=9):
    """Populate styled/ with *valid* healthy thumbnails; return the dir."""
    d = ctx.paths.project_dir / "thumbnails" / "styled"
    for i in range(valid):
        (d / f"thumb_{i:03d}.jpg").write_bytes(b"\xff" * (MIN_THUMBNAIL_SIZE + 1))
    return d


class TestThumbnailQualityGate:
    """Broken thumbnails hard-fail the stage instead of warning past it."""

    @pytest.mark.parametrize("payload", [b"", b"\xff\xd8" * 100])
    def test_fails_on_unusable_thumbnail(self, thumb_stage, ctx_with_thumbdir, payload):
        """Zero-byte and sub-10KB thumbnails both fail, naming the file."""
        (_styled(ctx_with_thumbdir) / "thumb_bad.jpg").write_bytes(payload)

        result = thumb_stage.run(ctx_with_thumbdir)

        assert result.success is False
        assert result.fatal is True
        assert "thumb_bad.jpg" in result.error

    def test_fails_when_no_thumbnails_produced(self, thumb_stage, ctx_with_thumbdir):
        """An empty styled/ dir fails — nothing to ship."""
        result = thumb_stage.run(ctx_with_thumbdir)

        assert result.success is False
        assert "thumbnail" in result.error.lower()

    def test_fails_when_styled_dir_missing(self, thumb_stage, ctx_with_thumbdir):
        """A missing styled/ dir (generation produced nothing) fails."""
        d = _styled(ctx_with_thumbdir)  # >= 9 short-circuits generation
        for f in d.iterdir():
            f.unlink()
        d.rmdir()

        assert thumb_stage.run(ctx_with_thumbdir).success is False

    def test_fails_when_generation_raises(self, thumb_stage, ctx_with_thumbdir):
        """An exception inside generation no longer returns success."""
        with patch(
            "gomoveshift.pipeline.orchestrator.generate_thumbnails_from_video",
            side_effect=RuntimeError("ffmpeg exploded"),
        ):
            result = thumb_stage.run(ctx_with_thumbdir)

        assert result.success is False
        assert "ffmpeg exploded" in result.error

    def test_healthy_thumbnails_pass(self, thumb_stage, ctx_with_thumbdir):
        """No false positives: >= 10KB thumbnails succeed."""
        _styled(ctx_with_thumbdir)

        result = thumb_stage.run(ctx_with_thumbdir)

        assert result.success is True
        assert result.error == ""
