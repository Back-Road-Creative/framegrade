"""Tests for PreRenderReviewStage."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline.context import PipelineContext, PipelinePaths
from gomoveshift.pipeline.review_gate import ReviewFailedError, ReviewGate
from gomoveshift.pipeline.stages.pre_render_review import PreRenderReviewStage


def _make_ctx(**overrides):
    paths = PipelinePaths(
        output_base=Path("/tmp/test"),
        project_dir=Path("/tmp/test/project"),
        stages_dir=Path("/tmp/test/stages"),
        state_file=Path("/tmp/test/state.json"),
        staging_dir=Path("/tmp/test/staging"),
    )
    defaults = dict(
        project_name="test",
        source_video=Path("/tmp/test.mp4"),
        source_hash="abc123",
        config=MagicMock(),
        location="Springfield to Shelbyville via US-41",
        publish_day="monday",
        paths=paths,
    )
    defaults.update(overrides)
    return PipelineContext(**defaults)


class TestPreRenderReviewStage:
    """Tests for the pre-render review gate stage."""

    def test_skip_when_in_skip_stages(self):
        """Stage is skipped when name is in ctx.skip_stages."""
        stage = PreRenderReviewStage()
        ctx = _make_ctx(skip_stages={"pre_render_review"})
        assert stage.should_run(ctx) is False

    def test_skip_when_already_completed(self):
        """Stage is skipped when name is in ctx.stages_completed."""
        stage = PreRenderReviewStage()
        ctx = _make_ctx(stages_completed=["pre_render_review"])
        assert stage.should_run(ctx) is False

    def test_fails_when_reviewer_rejects(self, monkeypatch):
        """Stage fails when ReviewGate raises ReviewFailedError (user says no)."""

        def reject_prompt(self_gate, msg):
            raise ReviewFailedError(f"Human review rejected: {msg}")

        monkeypatch.setattr(ReviewGate, "_prompt_required", reject_prompt)
        stage = PreRenderReviewStage()
        ctx = _make_ctx()
        result = stage.run(ctx)
        assert result.success is False
        assert result.fatal is True
        assert "Review rejected" in result.error

    def test_passes_when_reviewer_approves(self):
        """Stage passes when ReviewGate approves (conftest auto-approves _prompt_required)."""
        stage = PreRenderReviewStage()
        ctx = _make_ctx()
        result = stage.run(ctx)
        assert result.success is True

    def test_fails_on_empty_location(self):
        """Stage fails on empty location text (ReviewGate._fail raises ReviewFailedError)."""
        stage = PreRenderReviewStage()
        ctx = _make_ctx(location="")
        result = stage.run(ctx)
        assert result.success is False
        assert result.fatal is True
        assert "empty" in result.error.lower() or "Review rejected" in result.error

    def test_fails_on_placeholder_text(self):
        """Stage fails on placeholder text ('TODO' in location)."""
        stage = PreRenderReviewStage()
        ctx = _make_ctx(location="TODO fix this location")
        result = stage.run(ctx)
        assert result.success is False
        assert result.fatal is True
        assert "placeholder" in result.error.lower() or "Review rejected" in result.error
