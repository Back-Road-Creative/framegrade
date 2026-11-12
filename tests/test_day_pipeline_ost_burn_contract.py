"""The OST-burn contract assertion in ``run_day_pipeline``.

The assertion refuses a per-LF label that equals the drive-wide
``--location`` verbatim, because that used to mean the drive-wide fallback
had leaked into a per-LF slot. A label the specificity ladder awarded on its
own evidence can legitimately be the same words: a GPS-less drive whose
bracket route only resolves to the country (rec0063, 2026-11-12: "United
States" from fixes 17.5 h apart, drive-wide ``--region usa`` = "United
States") or a placeless operator label. Those LFs carry their award in
``per_lf_tier_evidence``; the assertion must let them burn and keep refusing
every LF without that evidence.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

from gomoveshift.pipeline import day_pipeline
from gomoveshift.pipeline.bracket_route import BRACKET_TIER_REASON
from gomoveshift.pipeline.drive_phases.phase_bc_day_first import PLACELESS_TIER_REASON
from tests.test_day_pipeline_ost_audit_gate import (
    _make_day_batch,
    _make_lf_plan,
    _make_scores,
    _make_seg,
    _patch_heavy,
)

DATE = "2025-09-14"
LABEL = "United States"


def _state(tier_evidence=None, key=1):
    state = {
        "per_lf_ost_text": {1: LABEL},
        "per_lf_titles": {1: f"Driving — {LABEL} (Day {DATE})"},
        "per_lf_recording_date": {1: DATE},
        "per_lf_locations": {},
        "ost_audit_confirmed_this_run": True,
    }
    if tier_evidence is not None:
        state["per_lf_tier_evidence"] = {key: tier_evidence}
    return state


def _burn(tmp_path, state):
    """Run the day with drive-wide ``location`` == the LF's label; return apply_ost mock."""
    output_base = tmp_path / "out"
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / f"shorts_ost_audit_{DATE}.confirmed").write_text("ok")
    _, patches = _patch_heavy(tmp_path)
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        apply_mock = stack.enter_context(
            patch(
                "gomoveshift.pipeline.day_pipeline.apply_ost",
                side_effect=lambda input_path, output_path, location, layout, **kw: (
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    or Path(output_path).write_bytes(b"ost")
                    or output_path
                ),
            )
        )
        day_pipeline.run_day_pipeline(
            day_batch=_make_day_batch(DATE, 1, per_ep=600.0),
            day_scores=_make_scores(600, 0.6),
            output_dir=output_base,
            location=LABEL,
            config=None,
            lf_plans=[_make_lf_plan(1, [_make_seg(0, 100, 200)], [])],
            per_lf_ost_text=state["per_lf_ost_text"],
            per_lf_locations=state["per_lf_locations"],
            state=state,
        )
    return apply_mock


@pytest.mark.parametrize("reason", [BRACKET_TIER_REASON, PLACELESS_TIER_REASON])
def test_ladder_awarded_label_equal_to_drive_wide_burns(tmp_path, reason):
    apply_mock = _burn(tmp_path, _state({"tier_reason": reason}))
    assert apply_mock.call_count == 1
    assert apply_mock.call_args.kwargs["location"] == LABEL


def test_str_keyed_tier_evidence_from_state_json_burns(tmp_path):
    """pipeline_state.json round-trips the LF index as a string key ("1")."""
    apply_mock = _burn(tmp_path, _state({"tier_reason": BRACKET_TIER_REASON}, key="1"))
    assert apply_mock.call_count == 1


def test_label_equal_to_drive_wide_without_tier_evidence_still_refused(tmp_path):
    with pytest.raises(RuntimeError, match="OST-burn contract violation"):
        _burn(tmp_path, _state())


def test_label_equal_to_drive_wide_with_other_tier_reason_still_refused(tmp_path):
    with pytest.raises(RuntimeError, match="OST-burn contract violation"):
        _burn(tmp_path, _state({"tier_reason": "county_corroborated"}))
