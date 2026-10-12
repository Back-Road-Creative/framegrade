"""An operator ``--per-lf-location`` label is gated like any other label.

There are no bypasses: the operator chooses the TEXT;
every location gate still checks it against the trace.
"""

from __future__ import annotations

import pytest

from gomoveshift.models import ResolvedLocation
from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    _assert_pre_encode_location_invariant,
    _build_ladder_evidence,
)
from gomoveshift.pipeline.validation.content_gates import check_specificity_ladder

MONT = "Montgomery County, Virginia"
ROAN = "Roanoke County, Virginia"
_LF5 = "Wytheville to Fort Chiswell via I-81, VA"
_SPARSE = [{"lat": 36.95 + i * 0.0001, "lon": -81.08, "time": float(i)} for i in range(3)]


def _radford() -> ResolvedLocation:
    return ResolvedLocation(
        dominant_county="Montgomery County",
        dominant_state="Virginia",
        regions=[MONT, ROAN],
        vote={MONT: 0.562, ROAN: 0.437},
        endpoint_regions={"start": MONT, "end": ROAN},
    )


def test_authoritative_flag_does_not_bypass_ladder():
    ev = {"gps_points": _SPARSE, "window": (0.0, 100.0), "authoritative": True}
    res = check_specificity_ladder({"location": _LF5}, ev)
    assert res.passed is False
    assert "LADDER-OVERRIDE" not in res.message


def test_ladder_evidence_carries_no_authoritative_flag():
    ev = _build_ladder_evidence({1: MONT}, {1: _SPARSE}, {1: MONT}, resolved_by_lf={1: _radford()})
    assert "authoritative" not in ev[1]


def test_per_lf_location_override_fails_coverage():
    rl = _radford()
    evidence = _build_ladder_evidence({1: MONT}, {1: _SPARSE}, {1: MONT}, resolved_by_lf={1: rl})
    with pytest.raises(RuntimeError, match="Roanoke County"):
        _assert_pre_encode_location_invariant({1: MONT}, {1: rl}, evidence)


def test_correct_operator_label_passes():
    rl = _radford()
    label = "Christiansburg to Salem via US-11, VA"
    _assert_pre_encode_location_invariant({1: label}, {1: rl}, None)
