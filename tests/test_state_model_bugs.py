"""Tests for PipelineState dict-protocol bugs and orchestrator state fixes.

Covers:
- __contains__ with falsy non-None values (False, 0, empty set)
- get() returning False not default
- update() applying int-key coercions
- Resume preserving user args
- Empty episode processed_files type
- Phase count log matching mode
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.drive_state_model import PipelineState


# ── __contains__ with falsy values ────────────────────────────────────


class TestContainsFalsyValues:
    def test_contains_false(self):
        state = PipelineState()
        state.fresh = False
        assert "fresh" in state

    def test_contains_zero(self):
        state = PipelineState()
        state["some_count"] = 0
        # fresh defaults to False (non-None), so it should be "in"
        assert "fresh" in state  # defaults to False

    def test_contains_empty_set(self):
        state = PipelineState()
        state.delivered_lfs = set()
        assert "delivered_lfs" in state

    def test_contains_empty_list(self):
        state = PipelineState()
        state.regular_lf_paths = []
        assert "regular_lf_paths" in state

    def test_contains_none_is_absent(self):
        state = PipelineState()
        # phases_completed defaults to None
        assert "phases_completed" not in state


# ── get() with falsy values ──────────────────────────────────────────


class TestGetFalsyValues:
    def test_get_returns_false_not_default(self):
        state = PipelineState()
        state.skip_upload = False
        assert state.get("skip_upload", True) is False

    def test_get_returns_empty_set_not_default(self):
        state = PipelineState()
        state.delivered_lfs = set()
        result = state.get("delivered_lfs", {1, 2, 3})
        assert result == set()

    def test_get_returns_none_for_none_field(self):
        """``get`` now distinguishes None-valued from missing.

        Previously, ``get("phases_completed", [])`` returned ``[]`` because
        the implementation conflated ``None`` with absent. The dict-API
        contract (and Python's own ``dict.get``) returns the stored value
        even when it's ``None``; ``default`` only fires when the key is
        genuinely absent.
        """
        state = PipelineState()
        # phases_completed defaults to None — get must return None, not [].
        assert state.get("phases_completed", []) is None

    def test_get_returns_default_for_missing_attribute(self):
        """Missing attributes still fall through to ``default``."""
        state = PipelineState()
        assert state.get("nonexistent_field", "fallback") == "fallback"
        # And without a default, missing → None (matches dict.get).
        assert state.get("nonexistent_field") is None

    def test_get_returns_empty_list_not_default(self):
        state = PipelineState()
        state.regular_lf_paths = []
        result = state.get("regular_lf_paths", ["fallback"])
        assert result == []


# ── update() coercion ────────────────────────────────────────────────


class TestUpdateCoercion:
    def test_update_coerces_int_keys_from_dict(self):
        state = PipelineState()
        raw = {
            "processed_files": {"0": "/tmp/ep_000.mp4", "1": "/tmp/ep_001.mp4"},
            "shorts_per_lf": {"1": ["short.mp4"]},
            "delivered_lfs": [1, 2, 3],
        }
        state.update(raw)
        assert all(isinstance(k, int) for k in state.processed_files.keys())
        assert all(isinstance(k, int) for k in state.shorts_per_lf.keys())
        assert isinstance(state.delivered_lfs, set)

    def test_update_coerces_path_fields(self):
        state = PipelineState()
        raw = {"source_dir": "/data/src", "output_base": "/data/out"}
        state.update(raw)
        assert isinstance(state.source_dir, Path)
        assert isinstance(state.output_base, Path)

    def test_update_coerces_tuple_fields(self):
        state = PipelineState()
        raw = {"file_boundaries": [[0.0, 120.0], [120.0, 300.0]]}
        state.update(raw)
        assert state.file_boundaries == [(0.0, 120.0), (120.0, 300.0)]


# ── Empty episode processed_files type ───────────────────────────────


# TestEmptyEpisodeProcessedFiles removed — phase_b.py was deleted
# (non-day-first removal). The single surviving pipeline only drives
# phase_bc_day_first, which does not populate processed_files the same way.


# ── Resume preserves user args ───────────────────────────────────────


class TestResumePreservesUserArgs:
    def test_resume_preserves_constructor_args(self, tmp_path):
        """When resuming, user-supplied args must not be clobbered by checkpoint."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline._drive_state import save_drive_state
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
        from gomoveshift.pipeline.drive_state_model import PipelineState

        # Simulate a saved checkpoint with old args. Both states use
        # publish_day='wed' because Layer-1 invariants require publish_day
        # to be in LONG_FORM_DAYS (currently {Wed=2}). The arg-preservation
        # contract under test is about source_dir/output_base/region/location
        # divergence anyway; publish_day divergence is not required to
        # exercise the resume path.
        state_file = tmp_path / "pipeline_state.json"
        old_state = PipelineState(
            source_dir=Path("/old/src"),
            output_base=Path("/old/out"),
            publish_day="wed",
            region="usa",
            location="Old Town",
            phases_completed=["phase_a"],
        )
        save_drive_state(state_file, old_state)

        # The artifact-drift gate (drive_orchestrator.py::_try_resume) drops
        # any phase whose declared PHASE_ARTIFACTS are missing on disk so the
        # phase re-runs cleanly. phase_a declares ``drive_scores.npz``; the
        # checkpoint says phase_a is complete, so the artifact must exist
        # under stages_dir (here: tmp_path). The downstream score-reload
        # path also opens it via ``np.load``, so a real (empty) npz archive
        # is required — a zero-byte stub trips ``zipfile.BadZipFile``.
        import numpy as np

        np.savez(tmp_path / "drive_scores.npz", scores=np.array([0.0]))

        # Create orchestrator and a new state with different user args
        orch = DrivePipelineOrchestrator(config=Config())
        new_state = PipelineState(
            source_dir=Path("/new/src"),
            output_base=Path("/new/out"),
            publish_day="wed",
            region="ireland",
            location="New Town",
        )

        stages_completed = []
        resumed = orch._try_resume(new_state, state_file, tmp_path, stages_completed, fresh=False)

        # User args must be preserved (not clobbered by checkpoint)
        assert new_state.source_dir == Path("/new/src")
        assert new_state.output_base == Path("/new/out")
        assert new_state.publish_day == "wed"
        assert new_state.region == "ireland"
        assert new_state.location == "New Town"
        # But checkpoint data should be merged
        assert "phase_a" in resumed


# ── Phase count log ──────────────────────────────────────────────────


class TestPhaseCountLog:
    def test_phase_count_matches_day_first_mode(self, tmp_path):
        """Day-first is the only surviving mode: 3 phases (phase_a, phase_bc_day_first, phase_d).

        The legacy 4-phase manifest (phase_b + phase_c) was deleted.
        """
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

        config = Config()
        orch = DrivePipelineOrchestrator(config=config)
        phase_defs = orch._select_phase_manifest(tmp_path / "state.json")
        assert len(phase_defs) == 3
