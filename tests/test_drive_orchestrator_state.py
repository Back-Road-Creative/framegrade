"""Tests for DrivePipelineOrchestrator._save_state atomic write delegation.

Verifies that the orchestrator delegates to save_drive_state (atomic
mkstemp + os.replace) rather than performing a bare open/write.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from _state_helpers import make_state
from gomoveshift.config import Config
from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator


class TestSaveStateUsesAtomicWrite:
    """_save_state must delegate to save_drive_state for atomic writes."""

    def test_save_state_uses_atomic_write(self, tmp_path):
        """Mock save_drive_state and verify _save_state calls it."""
        orch = DrivePipelineOrchestrator(config=Config())
        state_file = tmp_path / "pipeline_state.json"
        state = make_state(phases_completed=["phase_a"], source_dir="/tmp/src")

        with patch("gomoveshift.pipeline.drive_orchestrator.save_drive_state") as mock_save:
            orch._save_state(state_file, state)
            # ``_save_state`` threads the
            # break-glass ``skip_invariants`` kwarg captured on the
            # orchestrator instance. Default is False when the
            # orchestrator has not been run through the public run()
            # method that sets ``self._skip_invariants``.
            mock_save.assert_called_once_with(state_file, state, skip_invariants=False)

    def test_save_state_survives_crash(self, tmp_path):
        """A crash during write must not corrupt the existing state file.

        Strategy: write valid state, then simulate a crash by patching
        json.dump to raise IOError mid-write.  The atomic pattern
        (mkstemp + os.replace) ensures the original file stays intact.
        """
        from gomoveshift.pipeline._drive_state import save_drive_state

        orch = DrivePipelineOrchestrator(config=Config())
        state_file = tmp_path / "pipeline_state.json"
        original_state = make_state(phases_completed=["phase_a"], region="ireland")

        # Write valid initial state
        save_drive_state(state_file, original_state)
        assert state_file.exists()

        # Simulate a crash mid-write: json.dump raises IOError
        import gomoveshift.pipeline._drive_state as ds_mod

        with patch.object(ds_mod.json, "dump", side_effect=IOError("disk full")):
            with pytest.raises(IOError, match="disk full"):
                # phase_b_preview is the canonical current phase name;
                # legacy "phase_b" would fail Layer-1 invariants before the
                # IOError path (which is not what this test is probing).
                orch._save_state(
                    state_file,
                    make_state(phases_completed=["phase_a", "phase_b_preview"]),
                )

        # Original state file must be intact
        data = json.loads(state_file.read_text())
        assert data["phases_completed"] == ["phase_a"]
        assert data["region"] == "ireland"

        # No leftover .tmp files
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Leftover .tmp files: {tmp_files}"

    def test_save_state_creates_parent_dirs(self, tmp_path):
        """save_drive_state creates parent directories as needed."""
        orch = DrivePipelineOrchestrator(config=Config())
        state_file = tmp_path / "nested" / "deep" / "pipeline_state.json"
        state = make_state(phases_completed=[])

        orch._save_state(state_file, state)

        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["phases_completed"] == []
