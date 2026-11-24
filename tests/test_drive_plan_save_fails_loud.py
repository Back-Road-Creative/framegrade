"""A drive_plan.json that cannot be written fails phase_bc before any encode, with its cause.

It used to be a WARNING: the unloadable plan stayed on disk and every resume dropped
phase_bc_day_first and phase_d again (an endless silent re-drop), after another full render.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from gomoveshift.config import Config
from gomoveshift.pipeline.drive_phases import phase_bc_day_first
from test_phase_bc_shorts_per_lf import _build_state_with_drive_plan


class _UnsavablePlan(SimpleNamespace):
    def save(self, path):
        raise OSError(28, "No space left on device")


@pytest.mark.overpass_answers_empty
def test_failed_plan_save_stops_the_run_with_its_cause_before_any_encode(tmp_path):
    state, _, _ = _build_state_with_drive_plan(
        tmp_path,
        days_to_lf_indices={"2026-02-20": [1]},
        shorts_per_day=1,
        pre_record_days=False,
    )
    encode = MagicMock(return_value=[])
    plan = _UnsavablePlan(
        regular_lfs=[SimpleNamespace(index=1, total_duration=3600.0, segments=[])]
    )
    phase = phase_bc_day_first.PhaseBCDayFirst(Config())
    with (
        patch.object(phase_bc_day_first, "run_all_days", encode),
        patch.object(phase_bc_day_first, "_load_current_drive_plan", return_value=None),
        patch.object(phase_bc_day_first, "_build_drive_plan_from_state", return_value=plan),
    ):
        with pytest.raises(RuntimeError, match=r"cannot write .*drive_plan\.json.*No space left"):
            phase.run(state, probe_duration_fn=lambda p: 100.0)
    encode.assert_not_called()
