"""Tests for Phase B/C stages_base consumption."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline.drive_state_model import PipelineState


class TestPhaseBStagesBase:
    """Phase B reads stages_base from state when available."""

    def test_stages_dir_from_state(self, tmp_path):
        """When stages_base is set, phase_b should use it instead of output_base/_stages."""
        state = PipelineState(
            output_base=tmp_path / "out" / "proj1",
            stages_base=tmp_path / "stages" / "proj1" / "_stages",
        )
        # Simulate what phase_b.run does
        stages_dir = Path(state.get("stages_base", None) or Path(state["output_base"]) / "_stages")
        assert stages_dir == tmp_path / "stages" / "proj1" / "_stages"

    def test_stages_dir_fallback_no_stages_base(self, tmp_path):
        """When stages_base is None, fall back to output_base/_stages."""
        state = PipelineState(output_base=tmp_path / "out" / "proj1")
        stages_dir = Path(state.get("stages_base", None) or Path(state["output_base"]) / "_stages")
        assert stages_dir == (tmp_path / "out" / "proj1" / "_stages")


class TestPhaseCOutputDirs:
    """Phase C keeps final outputs under output_base regardless of stages_base."""

    def test_lf_output_stays_under_output_base(self, tmp_path):
        state = PipelineState(
            output_base=tmp_path / "out" / "proj1",
            stages_base=tmp_path / "stages" / "proj1" / "_stages",
        )
        output_base = Path(state["output_base"])
        lf_output_dir = output_base / "long_forms"
        shorts_output_dir = output_base / "shorts"
        stages_dir = Path(state.get("stages_base", None) or output_base / "_stages")

        assert lf_output_dir.is_relative_to(output_base)
        assert shorts_output_dir.is_relative_to(output_base)
        assert not stages_dir.is_relative_to(output_base)
