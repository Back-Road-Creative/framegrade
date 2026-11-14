"""A GPS-less long-form's label is proven before the encode, not after it.

2026-11-13, nctopa-2026-03-05-rec0063 failed four times after hours of rendering:
Phase D's title evidence check and DoD #20 (places_verified) refused text that was
already fixed before the encode started. ``_approve_waypost_titles_before_encode``
skipped every LF with no retained GPS ("GPS-less titles keep their existing Phase D
path"), so the drive's bracket and placeless labels were first checked after the
render. These tests hold the same two checks at the pre-encode gate.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    _approve_waypost_titles_before_encode,
)
from gomoveshift.pipeline.drive_state_model import PipelineState
from gomoveshift.pipeline.title_decision import PreRenderTitleError

ROANOKE_VA = {"county": "Roanoke County", "state": "Virginia", "country": "United States"}
FRANKLIN_PA = {"county": "Franklin County", "state": "Pennsylvania", "country": "United States"}
LEG = "Virginia to Pennsylvania"


def _bracket(gap_h=17.535):
    return {
        "label": LEG,
        "level": "states",
        "gap_h": gap_h,
        "before_place": ROANOKE_VA,
        "after_place": FRANKLIN_PA,
    }


def _approve(ost_by_lf, *, bracket=None, placeless=None, parts=1, gpx=None):
    state = PipelineState()
    state.bracket_route = bracket
    state.placeless_label = placeless
    state.per_lf_titles = dict(ost_by_lf)
    # Phase D's own "this LF has no GPS" record (rec0063: {"1": null, ...}).
    state.per_lf_gpx_path = {str(i): (gpx or {}).get(i) for i in ost_by_lf}
    plans = [
        SimpleNamespace(index=i, part_count=parts, part_index=n)
        for n, i in enumerate(sorted(ost_by_lf), 1)
    ]
    _approve_waypost_titles_before_encode(state, dict(ost_by_lf), plans, {}, None, {})
    return state


def test_the_bracket_leg_label_passes_before_the_encode():
    _approve({1: LEG, 2: LEG, 3: LEG}, bracket=_bracket(), parts=3)


def test_a_claim_free_placeless_label_passes_before_the_encode():
    _approve({1: "Somewhere On The Road"}, placeless="Somewhere On The Road")


def test_a_label_the_bracket_did_not_measure_stops_before_the_encode():
    # Its shorts would burn this text with no clip GPS and fail DoD #20 after the render.
    with pytest.raises(PreRenderTitleError, match=r"LF 1 .*Virginia to Maryland"):
        _approve({1: "Virginia to Maryland"}, bracket=_bracket())


def test_an_a_to_b_label_on_a_placeless_drive_stops_before_the_encode():
    # No bracket backs the endpoint claim, and it is not the placeless label either.
    with pytest.raises(PreRenderTitleError, match="LF 1 title .*no location evidence record"):
        _approve({1: LEG}, placeless="Somewhere On The Road")


def test_a_bracket_past_its_gap_cap_backs_nothing_before_the_encode():
    with pytest.raises(PreRenderTitleError, match="LF 1"):
        _approve({1: LEG}, bracket=_bracket(gap_h=40.0))


def test_every_unproven_lf_is_named_at_once():
    with pytest.raises(PreRenderTitleError) as exc:
        _approve({1: LEG, 2: "Cork", 3: "Ohio to Iowa"}, bracket=_bracket())
    text = str(exc.value)
    assert "LF 2" in text and "LF 3" in text and "LF 1 " not in text


def test_the_rec0063_part_labels_pass_before_the_encode():
    state = PipelineState()
    state.bracket_route = _bracket()
    state.per_lf_gpx_path = {"1": None, "2": None, "3": None}
    state.per_lf_titles = {i: f"{LEG} (Part {i} of 3)" for i in (1, 2, 3)}
    plans = [SimpleNamespace(index=i, part_count=3, part_index=i) for i in (1, 2, 3)]
    _approve_waypost_titles_before_encode(state, {i: LEG for i in (1, 2, 3)}, plans, {}, None, {})


def test_a_resumed_lf_with_a_recorded_gpx_is_not_judged_gps_less():
    # A resume arrives with no GPS points even for a GPS drive; its GPX record says otherwise.
    _approve({1: "Charleston County, SC"}, gpx={1: "/stages/lf_01.gpx"})
