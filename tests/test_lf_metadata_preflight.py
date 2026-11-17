"""LF metadata gates run at launch, before any encode starts.

A description-redundancy failure depends only on launch-time inputs (location
string, footer/region config, recording date), so it must fail the drive in
seconds, not after Phase D follows a multi-hour render ("robust > upstream
gate > late gate"). ``assert_lf_metadata_gates_before_encode`` reuses the
generator and gate call Phase D uses.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from _state_helpers import make_state
from gomoveshift.pipeline.drive_phases import phase_bc_day_first
from gomoveshift.pipeline.drive_phases.phase_d_metadata import (
    assert_lf_metadata_gates_before_encode,
)
from test_phase_bc_shorts_per_lf import _build_state_with_drive_plan

_GATE = "gomoveshift.pipeline.validation.content_gates.check_description_redundancy"
_FAIL = SimpleNamespace(passed=False, message="location 'Ireland' appears 4x")
_PASS = SimpleNamespace(passed=True, message="")


def _plan(*indices):
    return SimpleNamespace(
        regular_lfs=[SimpleNamespace(index=i, total_duration=3600.0, segments=[]) for i in indices]
    )


class TestPreflightFunction:
    def test_failing_gate_raises_with_preflight_prefix_and_phase_d_text(self):
        state = make_state(per_lf_locations={1: "Ireland"})
        with patch(_GATE, return_value=_FAIL):
            with pytest.raises(
                RuntimeError,
                match=r"^LF-METADATA-PREFLIGHT.*description redundancy gate failed for LF 1: "
                r"location 'Ireland' appears 4x$",
            ):
                assert_lf_metadata_gates_before_encode(_plan(1), state, "Ireland")

    def test_valid_location_passes_with_the_real_gate(self):
        state = make_state(
            per_lf_locations={1: "County Kerry, Ireland"},
            per_lf_recording_date={1: "2026-05-01"},
        )
        assert_lf_metadata_gates_before_encode(_plan(1), state, "County Kerry, Ireland")

    def test_names_the_failing_lf(self):
        state = make_state(per_lf_locations={1: "A", 2: "B"})
        with patch(_GATE, side_effect=[_PASS, _FAIL]):
            with pytest.raises(RuntimeError, match=r"for LF 2:"):
                assert_lf_metadata_gates_before_encode(_plan(1, 2), state, "A")


class TestRunsBeforeEncode:
    def test_encode_never_entered_when_the_gate_fails(self, tmp_path):
        state, _, _ = _build_state_with_drive_plan(
            tmp_path,
            days_to_lf_indices={"2026-02-20": [1]},
            shorts_per_day=1,
            pre_record_days=False,
        )
        geocoder = MagicMock()
        geocoder.reverse_geocode.return_value = None
        encode = MagicMock(return_value=[])
        from gomoveshift.config import Config

        phase = phase_bc_day_first.PhaseBCDayFirst(Config())
        with (
            patch(
                "gomoveshift.pipeline.gps_enrichment._build_shared_geocoder",
                return_value=geocoder,
            ),
            patch.object(phase_bc_day_first, "run_all_days", encode),
            patch(_GATE, return_value=_FAIL),
            # The fixture has no drive scores, so the real plan build returns
            # None and skips the pre-encode block; hand it a plan instead.
            patch.object(phase_bc_day_first, "_build_drive_plan_from_state", return_value=_plan(1)),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_d_metadata.assert_chapter_floor_before_encode"
            ),
        ):
            with pytest.raises(RuntimeError, match=r"LF-METADATA-PREFLIGHT"):
                phase.run(state, probe_duration_fn=lambda p: 100.0)
        encode.assert_not_called()

    def test_wired_before_run_all_days_in_phase_bc(self):
        src = inspect.getsource(phase_bc_day_first.PhaseBCDayFirst.run)
        assert "assert_lf_metadata_gates_before_encode(" in src
        assert src.index("assert_lf_metadata_gates_before_encode(") < src.index("run_all_days(")
