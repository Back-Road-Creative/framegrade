"""Regression tests: an unreadable resumed drive_plan must not
silently succeed and wipe persisted scheduling locks.

A resume rehydrates ``drive_plan`` from JSON; if reconstruction raises
(schema/shape mismatch), the pre-fix code caught the exception, logged,
and returned ``None`` — which the caller then treated exactly like a
*legitimately empty* plan: it wiped ``drive_schedule`` (the run-wide
BASE-DATE + identity-times lock), ``upload_manifests``,
``metadata_per_lf`` and ``social_posts_per_lf``, then returned normally
so Phase D was reported as complete.

These tests assert the fixed contract: a reconstruction FAILURE must
leave all four fields untouched and must not let Phase D return as if it
succeeded. A genuinely empty plan (no dict shape error) keeps its
existing wipe-and-return behaviour unchanged — see the characterization
test at the bottom.
"""

from __future__ import annotations

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline.drive_phases.phase_d import (
    DrivePlanReconstructionError,
    PhaseD,
)
from gomoveshift.pipeline.drive_state_model import PipelineState


def _make_state(tmp_path, *, drive_plan):
    state = PipelineState(
        source_dir=tmp_path / "src",
        output_base=tmp_path / "out",
        stages_base=tmp_path / "stages",
        region="ireland",
        publish_day="wed",
        drive_plan=drive_plan,
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "out").mkdir()
    (tmp_path / "stages").mkdir()
    return state


# An unparseable resumed drive_plan dict — RegularLFPlan.from_dict raises
# on the non-dict entry, mirroring test_phase_d_reconstruct_and_csv_helpers.
_UNREADABLE_DRIVE_PLAN = {"regular_lfs": ["not-a-dict"]}


class TestUnreadableDrivePlanPreservesLocks:
    def test_drive_schedule_unchanged_on_reconstruction_failure(self, tmp_path):
        state = _make_state(tmp_path, drive_plan=_UNREADABLE_DRIVE_PLAN)
        sentinel_schedule = object()
        state["drive_schedule"] = sentinel_schedule

        with pytest.raises(DrivePlanReconstructionError):
            PhaseD(Config()).run(state)

        assert state["drive_schedule"] is sentinel_schedule

    def test_upload_state_unchanged_on_reconstruction_failure(self, tmp_path):
        state = _make_state(tmp_path, drive_plan=_UNREADABLE_DRIVE_PLAN)
        state["upload_manifests"] = {1: "sentinel-manifest"}
        state["metadata_per_lf"] = {1: "sentinel-metadata"}
        state["social_posts_per_lf"] = {1: "sentinel-posts"}

        with pytest.raises(DrivePlanReconstructionError):
            PhaseD(Config()).run(state)

        assert state["upload_manifests"] == {1: "sentinel-manifest"}
        assert state["metadata_per_lf"] == {1: "sentinel-metadata"}
        assert state["social_posts_per_lf"] == {1: "sentinel-posts"}

    def test_phase_d_does_not_report_success_on_reconstruction_failure(self, tmp_path):
        """Phase D must not return normally — a caller relying on a clean
        return to append 'phase_d' to phases_completed must never see one
        for a run whose plan could not be read."""
        state = _make_state(tmp_path, drive_plan=_UNREADABLE_DRIVE_PLAN)

        with pytest.raises(DrivePlanReconstructionError):
            PhaseD(Config()).run(state)


class TestEmptyPlanCharacterization:
    """A genuinely empty (not unreadable) plan keeps today's behaviour.

    This is a characterization test, not new coverage of the fix — it
    should pass before and after the fix.
    """

    def test_none_drive_plan_still_wipes_and_returns_normally(self, tmp_path):
        state = _make_state(tmp_path, drive_plan=None)
        state["drive_schedule"] = object()
        state["upload_manifests"] = {1: "old"}
        state["metadata_per_lf"] = {1: "old"}
        state["social_posts_per_lf"] = {1: "old"}
        state["youtube_uploads"] = {1: {"video_id": "existing"}}

        PhaseD(Config()).run(state)  # must not raise

        assert state["drive_schedule"] is None
        assert state["upload_manifests"] == {}
        assert state["metadata_per_lf"] == {}
        assert state["social_posts_per_lf"] == {}
        # Prior upload history is preserved.
        assert state["youtube_uploads"] == {1: {"video_id": "existing"}}
