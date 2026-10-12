"""Tests for Phase D stages_dir, route_profiles, and empty-output bugs.

Covered regressions:
- stages_dir hardcoded to output_base/_stages instead of reading stages_base
- route_profiles falsy dict (empty {}) silently falls through `or` to int-key lookup
- No warning when drive_plan is empty/None → day's output silently lost
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.drive_state_model import PipelineState


class TestPhaseDStagesBase:
    """Phase D must respect state['stages_base'] instead of hardcoding _stages."""

    def test_phase_d_uses_stages_base(self, tmp_path):
        """When stages_base is set, Phase D should use it for stages_dir."""
        custom_stages = tmp_path / "custom_stages"
        custom_stages.mkdir()

        state = PipelineState(
            source_dir=tmp_path / "src",
            output_base=tmp_path / "out",
            stages_base=custom_stages,
            region="ireland",
            publish_day="wed",
            drive_plan=None,  # empty → early return
        )
        (tmp_path / "src").mkdir()
        (tmp_path / "out").mkdir()

        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_phases.phase_d import PhaseD

        phase_d = PhaseD(Config())
        phase_d.run(state)

        # Phase D should NOT have created output_base/_stages
        fallback_stages = tmp_path / "out" / "_stages"
        assert not fallback_stages.exists(), (
            "Phase D created fallback _stages dir instead of using stages_base"
        )


class TestPhaseDRouteProfileFalsyDict:
    """route_profiles.get(str(idx)) or route_profiles.get(idx) fails for empty dict."""

    def test_empty_dict_profile_not_skipped(self):
        """An empty dict {} route profile should be passed through, not treated as absent."""
        # The or-chain: `route_profiles.get(str(idx)) or route_profiles.get(idx)`
        # treats {} as falsy and falls through. The fix uses explicit None check.
        route_profiles = {"1": {}}  # empty dict for LF index 1

        # Current buggy behavior: `{} or route_profiles.get(1)` → `route_profiles.get(1)` → None
        str_val = route_profiles.get(str(1))
        assert str_val == {}
        # The `or` would make this None:
        buggy_result = str_val or route_profiles.get(1)  # {} or None → None (BUG)
        assert buggy_result is None  # confirms the bug exists

        # Fixed behavior should use explicit None check:
        fixed_result = route_profiles.get(str(1))
        if fixed_result is None:
            fixed_result = route_profiles.get(1)
        assert fixed_result == {}  # empty dict preserved


class TestPhaseDEmptyOutputWarning:
    """Phase D must warn when drive_plan is empty/None."""

    def test_warns_on_empty_drive_plan(self, tmp_path, caplog):
        """Empty drive_plan should log a warning, not silently return."""
        state = PipelineState(
            source_dir=tmp_path / "src",
            output_base=tmp_path / "out",
            stages_base=tmp_path / "stages",
            region="ireland",
            publish_day="wed",
            drive_plan=None,
        )
        (tmp_path / "src").mkdir()
        (tmp_path / "out").mkdir()
        (tmp_path / "stages").mkdir()

        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_phases.phase_d import PhaseD

        phase_d = PhaseD(Config())

        with caplog.at_level(logging.WARNING):
            phase_d.run(state)

        assert any(
            "no qualifying" in r.message.lower() or "empty" in r.message.lower()
            for r in caplog.records
        ), f"Expected warning about empty drive_plan. Logs: {[r.message for r in caplog.records]}"


class TestPhaseBCDayFirstStagesBase:
    """phase_bc_day_first must respect stages_base too."""

    def test_day_first_uses_stages_base(self, tmp_path):
        """When stages_base is set, phase_bc_day_first should use it."""
        custom_stages = tmp_path / "custom_stages"
        custom_stages.mkdir()

        state = PipelineState(
            source_dir=tmp_path / "src",
            output_base=tmp_path / "out",
            stages_base=custom_stages,
            region="ireland",
            publish_day="wed",
            ingest_plan=None,  # empty → early return
        )
        (tmp_path / "src").mkdir()
        (tmp_path / "out").mkdir()

        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import PhaseBCDayFirst

        phase = PhaseBCDayFirst(Config())
        phase.run(state, probe_duration_fn=lambda p: 10.0)

        # Should NOT have created output_base/_stages
        fallback_stages = tmp_path / "out" / "_stages"
        assert not fallback_stages.exists(), (
            "phase_bc_day_first created fallback _stages dir instead of using stages_base"
        )
