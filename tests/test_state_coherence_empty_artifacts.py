"""State coherence guard for empty phase artifacts.

The orchestrator must NOT append a phase to ``state["phases_completed"]``
when that phase produced empty/missing artifacts. A failure mode this prevents:

  1. Phase BC ran, all 4 days failed mid-encode (NVENC + tmp.mp4 ext bugs).
  2. ``run_all_days`` returned 0 day_artifacts.
  3. Orchestrator still appended ``phase_bc_day_first`` to phases_completed.
  4. Phase D failed on sidecar integrity (plan had 7 LFs, no LF artifacts on disk).
  5. Resume: orchestrator saw BC in phases_completed, skipped BC, Phase D
     failed again the same way. Operator hand-edited state TWICE.

This module covers two layers:

  * **Append-time guard** in :class:`DrivePipelineOrchestrator` —
    :meth:`_phase_produced_required_artifacts` returning False blocks the
    append. Covers phase_a (empty episode_files & drive_scores), BC
    (empty day_artifacts), phase_d (empty youtube_uploads when
    ``youtube_upload=True``).
  * **Resume-time invariant** in
    :func:`gomoveshift.pipeline.state_invariants.assert_resume_state_coherent`
    — if a state file slipped past the append guard (older revisions, or
    operator hand-edits), the resume check raises StateCoherenceError so
    the run never proceeds with a phantom phase entry.
"""

from __future__ import annotations

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
from gomoveshift.pipeline.exceptions import StateCoherenceError
from gomoveshift.pipeline.state_invariants import assert_resume_state_coherent
from _state_helpers import make_state


# ── Append-time guard: BC ────────────────────────────────────────────


class TestAppendGuardPhaseBC:
    """Orchestrator must NOT append phase_bc_day_first when day_artifacts is empty."""

    def test_bc_with_empty_day_artifacts_blocks_append(self):
        orch = DrivePipelineOrchestrator(config=Config())
        state = make_state(
            day_artifacts=[],
            regular_lf_paths=[],
        )
        ok, reason = orch._phase_produced_required_artifacts("phase_bc_day_first", state)
        assert ok is False
        assert "day_artifacts" in reason

    def test_bc_with_missing_day_artifacts_blocks_append(self):
        """Key absent altogether is the same shape as empty list."""
        orch = DrivePipelineOrchestrator(config=Config())
        state: dict = {}
        ok, reason = orch._phase_produced_required_artifacts("phase_bc_day_first", state)
        assert ok is False

    def test_bc_with_populated_day_artifacts_allows_append(self):
        orch = DrivePipelineOrchestrator(config=Config())
        state = make_state(
            day_artifacts=[{"date": "2026-02-20", "master_path": "/x/d1.mp4"}],
            regular_lf_paths=["/out/lf_00.mp4"],
        )
        ok, reason = orch._phase_produced_required_artifacts("phase_bc_day_first", state)
        assert ok is True


# ── Append-time guard: phase_a ───────────────────────────────────────


class TestAppendGuardPhaseA:
    """phase_a must populate at least episode_files OR drive_scores."""

    def test_phase_a_empty_episode_files_and_drive_scores_blocks(self):
        orch = DrivePipelineOrchestrator(config=Config())
        state = make_state(episode_files=[], drive_scores=None)
        ok, reason = orch._phase_produced_required_artifacts("phase_a", state)
        assert ok is False

    def test_phase_a_with_episode_files_allows(self):
        orch = DrivePipelineOrchestrator(config=Config())
        state = make_state(episode_files=["/tmp/ep0.mp4"], drive_scores=None)
        ok, reason = orch._phase_produced_required_artifacts("phase_a", state)
        assert ok is True

    def test_phase_a_with_drive_scores_allows(self):
        """drive_scores is an ndarray-ish payload; truthiness is enough."""
        orch = DrivePipelineOrchestrator(config=Config())
        state = make_state(episode_files=[], drive_scores=[0.1, 0.2, 0.3])
        ok, reason = orch._phase_produced_required_artifacts("phase_a", state)
        assert ok is True


# ── Append-time guard: phase_d ───────────────────────────────────────


class TestAppendGuardPhaseD:
    """phase_d must populate youtube_uploads when youtube_upload is True.

    When the operator opted into uploads, an empty youtube_uploads dict
    means phase_d ran but failed silently — appending it would lock in
    the failure across resume.
    """

    def test_phase_d_youtube_upload_true_empty_uploads_blocks(self):
        orch = DrivePipelineOrchestrator(config=Config())
        state = make_state(youtube_upload=True, youtube_uploads={})
        ok, reason = orch._phase_produced_required_artifacts("phase_d", state)
        assert ok is False
        assert "youtube_uploads" in reason

    def test_phase_d_youtube_upload_true_populated_uploads_allows(self):
        orch = DrivePipelineOrchestrator(config=Config())
        state = make_state(
            youtube_upload=True,
            youtube_uploads={0: {"video_id": "abc"}},
        )
        ok, reason = orch._phase_produced_required_artifacts("phase_d", state)
        assert ok is True

    def test_phase_d_skip_upload_empty_uploads_allows(self):
        """When uploads were intentionally skipped, empty is fine."""
        orch = DrivePipelineOrchestrator(config=Config())
        state = make_state(
            youtube_upload=False,
            skip_upload=True,
            youtube_uploads={},
        )
        ok, reason = orch._phase_produced_required_artifacts("phase_d", state)
        assert ok is True


# ── Append-time guard: unknown / b_preview phases ────────────────────


class TestAppendGuardOtherPhases:
    """Phases without a coherence rule pass through unchanged."""

    def test_phase_b_preview_has_no_artifact_rule(self):
        orch = DrivePipelineOrchestrator(config=Config())
        ok, _ = orch._phase_produced_required_artifacts("phase_b_preview", {})
        assert ok is True

    def test_unknown_phase_passes_through(self):
        orch = DrivePipelineOrchestrator(config=Config())
        ok, _ = orch._phase_produced_required_artifacts("phase_x_future", {})
        assert ok is True


# ── Resume-time invariant: BC + empty day_artifacts ──────────────────


class TestResumeCoherenceEmptyDayArtifacts:
    """If a state file slipped past the append guard, resume must catch it."""

    def test_bc_in_phases_completed_with_empty_day_artifacts_raises(self):
        """BC marked complete with empty day_artifacts and expected dates must raise.

        Note: the rule only fires when there were expected dates (real
        work to do). The zero-qualifying-footage early-exit path is
        legitimately covered by the existing Rule 5.
        """
        state = make_state(
            phases_completed=["phase_a", "phase_bc_day_first"],
            day_artifacts=[],
            episode_dates=["2026-02-20", "2026-02-21"],
        )
        with pytest.raises(StateCoherenceError) as exc_info:
            assert_resume_state_coherent(state)
        msg = str(exc_info.value)
        assert "phase_bc_day_first" in msg
        assert "day_artifacts" in msg

    def test_bc_in_phases_completed_with_missing_day_artifacts_raises(self):
        """Key absent is treated the same as empty list (with expected dates)."""
        state = make_state(
            phases_completed=["phase_bc_day_first"],
            episode_dates=["2026-02-20"],
        )
        with pytest.raises(StateCoherenceError):
            assert_resume_state_coherent(state)

    def test_bc_with_populated_day_artifacts_passes(self):
        state = make_state(
            phases_completed=["phase_a", "phase_bc_day_first"],
            day_artifacts=[{"date": "2026-02-20", "master_path": "/x/d1.mp4"}],
            episode_dates=["2026-02-20"],
            per_day_completed={"2026-02-20": {"master_path": "/x/d1.mp4"}},
        )
        # Should not raise the new empty-artifacts rule.
        assert_resume_state_coherent(state)

    def test_bc_with_per_day_completed_but_empty_day_artifacts_passes(self):
        """Backward-compat: state files predating day_artifacts tracking
        still pass when per_day_completed is populated."""
        state = make_state(
            phases_completed=["phase_bc_day_first"],
            day_artifacts=[],
            episode_dates=["2026-02-20"],
            per_day_completed={"2026-02-20": {"master_path": "/x/d1.mp4"}},
        )
        assert_resume_state_coherent(state)  # no raise

    def test_zero_qualifying_footage_early_exit_passes(self):
        """The legitimate zero-qualifying-footage path: BC marks complete
        with empty artifacts AND no expected dates. Must NOT raise — see
        Rule 5 in ``assert_state_invariants``."""
        state = make_state(
            phases_completed=["phase_bc_day_first"],
            day_artifacts=[],
            per_day_completed={},
        )
        assert_resume_state_coherent(state)  # no raise

    def test_bc_not_in_phases_completed_is_noop_even_with_empty_day_artifacts(self):
        """The rule must only fire when BC claims completion."""
        state = make_state(
            phases_completed=["phase_a"],
            day_artifacts=[],
            episode_dates=["2026-02-20"],
        )
        assert_resume_state_coherent(state)  # no raise

    def test_legitimate_full_success_state_passes(self):
        """End-to-end success state with all four phases: no false positive."""
        state = make_state(
            phases_completed=[
                "phase_a",
                "phase_bc_day_first",
                "phase_d",
            ],
            episode_dates=["2026-02-20", "2026-02-21"],
            per_day_completed={
                "2026-02-20": {"master_path": "/x/d1.mp4"},
                "2026-02-21": {"master_path": "/x/d2.mp4"},
            },
            day_artifacts=[
                {"date": "2026-02-20", "master_path": "/x/d1.mp4"},
                {"date": "2026-02-21", "master_path": "/x/d2.mp4"},
            ],
            regular_lf_paths=["/out/lf_00.mp4", "/out/lf_01.mp4"],
            shorts_per_lf={0: ["/out/s0.mp4"], 1: ["/out/s1.mp4"]},
            youtube_uploads={0: {"video_id": "a"}, 1: {"video_id": "b"}},
        )
        assert_resume_state_coherent(state)  # no raise
