"""Tests for the sub-pipeline scaffold (the base class and the first two sub-pipelines).

Covers the SubPipeline base class behavior — skip/complete gating,
state-save callback firing, fatal-abort propagation — and verifies the
two first-generation sub-pipelines reference exactly the expected slice of
STAGE_ORDER.

These tests use fake stages to exercise the loop logic in isolation, so
they don't need real video files or ffmpeg.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

from gomoveshift.pipeline.stage_protocol import BaseStage, StageResult
from gomoveshift.pipeline.stages import STAGE_ORDER
from gomoveshift.pipeline.sub_pipelines import (
    IngestAndPrep,
    LightMetadata,
    PublishAndArchive,
    RenderAndDerive,
    SubPipeline,
    SubPipelineResult,
)


# ── Fake context: just what SubPipeline.run() reads/writes ──────────────


@dataclass
class FakeCtx:
    skip_stages: set = field(default_factory=set)
    stages_completed: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    # Sentinel for the state-save guard. None until PropertyDetectionStage runs.
    video_properties: Optional[object] = None


def _saver(saved: list):
    """Return a save_state callback that appends each call's
    stages_completed snapshot to ``saved`` for assertion."""

    def save(ctx: FakeCtx) -> None:
        saved.append(list(ctx.stages_completed))

    return save


# ── Fake stages with controllable outcomes ──────────────────────────────


class _FakeStage(BaseStage):
    """Stage that records calls and returns a configurable StageResult."""

    def __init__(self, name: str, result: Optional[StageResult] = None):
        self.name = name
        self._result = result or StageResult(success=True)
        self.run_called = False

    def run(self, ctx):  # type: ignore[override]
        self.run_called = True
        return self._result


def _sub_with(*stages: BaseStage) -> SubPipeline:
    sp = SubPipeline()
    sp.name = "test_sub"
    sp.stages = list(stages)
    return sp


# ── SubPipeline base behavior ───────────────────────────────────────────


class TestSubPipelineRunLoop:
    def test_completes_all_success_stages_and_records_them(self):
        ctx = FakeCtx()
        saved: list = []
        s1, s2 = _FakeStage("a"), _FakeStage("b")
        result = _sub_with(s1, s2).run(ctx, _saver(saved))
        assert result == SubPipelineResult(success=True)
        assert ctx.stages_completed == ["a", "b"]
        # save_state callback fires once per successful stage.
        assert len(saved) == 2

    def test_skips_stages_already_completed(self):
        ctx = FakeCtx(stages_completed=["a"])
        saved: list = []
        s1, s2 = _FakeStage("a"), _FakeStage("b")
        _sub_with(s1, s2).run(ctx, _saver(saved))
        assert s1.run_called is False
        assert s2.run_called is True
        assert ctx.stages_completed == ["a", "b"]
        # Only one save — for the new stage.
        assert len(saved) == 1

    def test_skips_stages_in_skip_set(self):
        ctx = FakeCtx(skip_stages={"a"})
        saved: list = []
        s1, s2 = _FakeStage("a"), _FakeStage("b")
        _sub_with(s1, s2).run(ctx, _saver(saved))
        assert s1.run_called is False
        assert s2.run_called is True
        assert ctx.stages_completed == ["b"]

    def test_idempotent_when_stage_re_adds_itself(self):
        # Some stages (e.g. single_pass_encode) append their own sub-stage
        # names. The loop must not double-add the wrapper name.
        ctx = FakeCtx()

        class _SelfAddingStage(_FakeStage):
            def run(self, ctx):
                ctx.stages_completed.append(self.name)
                return StageResult(success=True)

        s = _SelfAddingStage("x")
        _sub_with(s).run(ctx, _saver([]))
        assert ctx.stages_completed.count("x") == 1

    def test_fatal_failure_aborts_with_propagated_error(self):
        ctx = FakeCtx()
        s1 = _FakeStage("a", StageResult(success=False, error="boom", fatal=True))
        s2 = _FakeStage("b")
        result = _sub_with(s1, s2).run(ctx, _saver([]))
        assert result.success is False
        assert result.fatal is True
        assert result.error == "boom"
        assert s2.run_called is False  # subsequent stages never run

    def test_non_fatal_failure_continues_and_does_not_mark_complete(self):
        ctx = FakeCtx()
        s1 = _FakeStage("a", StageResult(success=False, error="soft", fatal=False))
        s2 = _FakeStage("b")
        saved: list = []
        result = _sub_with(s1, s2).run(ctx, _saver(saved))
        assert result.success is True
        assert s2.run_called is True
        assert ctx.stages_completed == ["b"]
        # No save for the soft-failed stage — only for s2.
        assert len(saved) == 1


# ── First sub-pipeline group identity ──────────────────────────────────────────


class TestFirstGroupSubPipelineSlices:
    """Lock the stage assignments so accidental reordering breaks loudly."""

    def test_ingest_and_prep_owns_stages_0_through_3(self):
        names = [s.name for s in IngestAndPrep().stages]
        assert names == [
            "validation",
            "analytics_sync",
            "detect_properties",
            "copy_to_wsl",
        ]

    def test_light_metadata_owns_stages_4_through_9(self):
        names = [s.name for s in LightMetadata().stages]
        assert names == [
            "stop_removal",
            "blur_detection",
            "chapter_generation",
            "metadata_generation",
            "schedule_generation",
            "pre_render_review",
        ]

    def test_sub_pipelines_cover_first_ten_stages_of_STAGE_ORDER(self):
        # Sub-pipeline ordering must match the canonical STAGE_ORDER slice
        # so resume across the boundary is deterministic.
        expected = [s.name for s in STAGE_ORDER[:10]]
        actual = [s.name for s in IngestAndPrep().stages] + [s.name for s in LightMetadata().stages]
        assert actual == expected

    def test_pre_render_review_is_the_human_gate_boundary(self):
        # The boundary between light-metadata and render-and-derive is the
        # human review gate. Single_pass_encode must NOT be in
        # light-metadata — it's the first irreversible stage.
        light_names = {s.name for s in LightMetadata().stages}
        assert "pre_render_review" in light_names
        assert "single_pass_encode" not in light_names


# ── Second sub-pipeline group identity ──────────────────────────────────────────


class TestSecondGroupSubPipelineSlices:
    """Lock the stage assignments for render-and-derive + publish-and-archive."""

    def test_render_and_derive_owns_stages_9_through_12(self):
        names = [s.name for s in RenderAndDerive().stages]
        assert names == [
            "single_pass_encode",
            "thumbnail_generation",
            "shorts_analysis",
            "social_generation",
        ]

    def test_publish_and_archive_owns_stages_13_through_17(self):
        names = [s.name for s in PublishAndArchive().stages]
        # validation_gate runs FIRST — before the irreversible drive_upload —
        # so its audio/content hard-fail can block the upload, not just the
        # local D: archive copy.
        assert names == [
            "validation_gate",
            "drive_upload",
            "csv_generation",
            "d_drive_copy",
            "cleanup",
        ]

    def test_validation_gate_precedes_d_drive_copy_inside_publish_slice(self):
        # The STAGE_ORDER-level invariant is in test_stage_ordering. This
        # variant pins the same constraint inside the sub-pipeline so a
        # future re-order of publish-and-archive can't silently violate it.
        names = [s.name for s in PublishAndArchive().stages]
        assert names.index("validation_gate") < names.index("d_drive_copy")

    def test_four_sub_pipelines_cover_all_of_STAGE_ORDER_in_order(self):
        # The full sequence must match STAGE_ORDER exactly — same names,
        # same order — so removing the legacy loop is observably a
        # no-op for stage ordering / resume semantics.
        expected = [s.name for s in STAGE_ORDER]
        actual = (
            [s.name for s in IngestAndPrep().stages]
            + [s.name for s in LightMetadata().stages]
            + [s.name for s in RenderAndDerive().stages]
            + [s.name for s in PublishAndArchive().stages]
        )
        assert actual == expected


# ── Orchestrator helper: _save_state_after_stage guard ──────────────────


class TestSaveStateGuard:
    """Verify the shared save helper preserves the pre-detect-properties guard.

    Before PropertyDetectionStage runs, ``ctx.video_properties is None``
    and writing would produce an invalid state file (bitrate_kbps=0 would
    fail can_resume's later check). The helper must early-return.
    """

    def test_helper_does_nothing_when_video_properties_is_none(self):
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator()
        save_calls: list = []

        def _spy(**kwargs):
            save_calls.append(kwargs)

        orch._save_state = _spy  # type: ignore[assignment]

        @dataclass
        class _Ctx:
            video_properties: Optional[object] = None

        orch._save_state_after_stage(_Ctx())
        assert save_calls == []

    def test_helper_writes_state_when_video_properties_populated(self):
        from pathlib import Path

        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator
        from gomoveshift.pipeline.video_detect import VideoProperties

        orch = PipelineOrchestrator()
        save_calls: list = []

        def _spy(**kwargs):
            save_calls.append(kwargs)

        orch._save_state = _spy  # type: ignore[assignment]

        @dataclass
        class _Paths:
            state_file: Path = Path("/tmp/state.json")

        @dataclass
        class _Ctx:
            paths: _Paths = field(default_factory=_Paths)
            project_name: str = "p"
            source_video: Path = Path("/tmp/src.mp4")
            source_hash: str = "abc"
            stages_completed: list = field(default_factory=list)
            current_video: Path = Path("/tmp/cur.mp4")
            video_properties: VideoProperties = field(
                default_factory=lambda: VideoProperties(
                    width=1,
                    height=1,
                    fps=1.0,
                    duration=1.0,
                    codec="h264",
                    audio_codec="aac",
                    file_size_bytes=1,
                )
            )
            timings: dict = field(default_factory=dict)
            params: dict = field(default_factory=dict)
            keep_segments: Optional[list] = None

        orch._save_state_after_stage(_Ctx())
        assert len(save_calls) == 1
        assert save_calls[0]["project_name"] == "p"


# ── Sub-pipelines are wired into the orchestrator import path ───────────


def test_sub_pipelines_importable_from_orchestrator_module():
    """Regression guard against the import-wiring bug class."""
    from gomoveshift.pipeline import orchestrator as orch_mod

    assert hasattr(orch_mod, "IngestAndPrep")
    assert hasattr(orch_mod, "LightMetadata")
    assert hasattr(orch_mod, "RenderAndDerive")
    assert hasattr(orch_mod, "PublishAndArchive")


def test_orchestrator_run_no_longer_iterates_STAGE_ORDER():
    """The sub-pipeline refactor removed the legacy ``for stage in STAGE_ORDER`` loop.

    Catching its reappearance via source inspection is more direct than
    asserting on stage execution counts — and survives unrelated changes
    to the run() body.
    """
    import inspect

    from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

    src = inspect.getsource(PipelineOrchestrator.run)
    # Only fail on actual iteration — references in comments / docstrings
    # that point readers back to STAGE_ORDER are fine and worth keeping.
    assert "for stage in STAGE_ORDER" not in src, (
        "PipelineOrchestrator.run() must drive sub-pipelines, not iterate "
        "STAGE_ORDER directly. Re-extract into a sub-pipeline if you need "
        "a new stage."
    )


def test_orchestrator_module_does_not_import_STAGE_ORDER():
    """STAGE_ORDER is no longer needed at the orchestrator layer.

    Keeping the import would invite drift back toward a flat loop.
    """
    from gomoveshift.pipeline import orchestrator as orch_mod

    assert not hasattr(orch_mod, "STAGE_ORDER")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
