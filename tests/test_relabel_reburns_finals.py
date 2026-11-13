"""A label changed on resume re-burns the day's finals instead of reusing them.

2026-11-13, nctopa-2026-03-05-rec0063: the bracket relabel ("United States" ->
"Virginia to Pennsylvania") cleared ``per_day_completed``, and seconds later the
recovery hook re-added the day from the finals on disk, so the day was skipped and
3 long-forms and 21 Shorts kept the old burned text (DoD check 20 refused all 21).
The relabel now marks each such day in ``state.reburn_days``; the mark outlives a
crash and is dropped only when the day really completes.
"""

from __future__ import annotations

from types import SimpleNamespace

from gomoveshift.pipeline import preflight_plan
from gomoveshift.pipeline.drive_orchestrator import (
    auto_populate_per_day_completed,
    relabel_bracket_on_resume,
)
from gomoveshift.pipeline.drive_phases import phase_bc_day_first as bc
from gomoveshift.pipeline.drive_state_model import PipelineState

from _state_helpers import make_state

OLD = "United States"
NEW = "Virginia to Pennsylvania"
PHASES = ["phase_a", "phase_bc_day_first", "phase_d"]
DAY = "2026-03-05"


def _record():
    return {
        "label": OLD,
        "level": "country",
        "gap_h": 17.535,
        "before": {"time": "2026-03-04T23:28:05Z"},
        "after": {"time": "2026-03-05T17:00:12Z"},
        "before_place": {"county": "Roanoke County", "state": "Virginia", "country": OLD},
        "after_place": {"county": "Franklin County", "state": "Pennsylvania", "country": OLD},
    }


def _saved_state():
    mirror = {1: OLD, 2: OLD, 3: OLD}
    return make_state(
        bracket_route=_record(),
        location=OLD,
        per_day_locations={DAY: OLD},
        per_lf_overrides=dict(mirror),
        per_lf_locations=dict(mirror),
        per_lf_ost_text=dict(mirror),
        per_lf_titles=dict(mirror),
        per_day_completed={DAY: {"master_path": None}},
        shorts_ost_audit_passed={DAY: True},
        shorts_ost_audit_hash={DAY: "d" * 64},
        phases_completed=list(PHASES),
    )


def _stages(tmp_path):
    stages = tmp_path / "out" / "_stages"
    day_dir = stages / f"day_{DAY}"
    (day_dir / "regular_lfs").mkdir(parents=True)
    (day_dir / "shorts").mkdir()
    (day_dir / "regular_lfs" / "lf_01.mp4").write_bytes(b"x")
    (day_dir / "shorts" / "short_01.mp4").write_bytes(b"y")
    return stages


def _relabel(state, tmp_path):
    stages = _stages(tmp_path)
    relabel_bracket_on_resume(
        state, list(PHASES), list(PHASES), output_base=tmp_path / "out", stages_dir=stages
    )
    return stages


def test_the_recovery_hook_does_not_bring_the_old_finals_back(tmp_path):
    state = _saved_state()
    stages = _relabel(state, tmp_path)
    auto_populate_per_day_completed(state, stages, fresh=False)
    assert state.per_day_completed == {}
    assert DAY in state.reburn_days


def test_a_day_on_disk_but_not_in_the_record_is_marked_too(tmp_path):
    state = _saved_state()
    state.per_day_completed = {}
    stages = _relabel(state, tmp_path)
    auto_populate_per_day_completed(state, stages, fresh=False)
    assert state.per_day_completed == {}
    assert DAY in state.reburn_days


def test_the_shorts_confirmation_of_the_old_text_is_dropped(tmp_path):
    state = _saved_state()
    stages = _stages(tmp_path)
    sidecar = stages / f"shorts_ost_audit_{DAY}.json"
    review = stages / f"shorts_ost_system_review_{DAY}.json"
    sidecar.write_text("{}")
    review.write_text("{}")
    relabel_bracket_on_resume(
        state, list(PHASES), list(PHASES), output_base=tmp_path / "out", stages_dir=stages
    )
    assert state.shorts_ost_audit_passed == {} and state.shorts_ost_audit_hash == {}
    assert not sidecar.exists() and not review.exists()


def test_a_crash_marker_does_not_let_the_day_skip(tmp_path):
    # A crash mid-re-burn writes a recovery marker; the day still must re-run.
    state = _saved_state()
    _relabel(state, tmp_path)
    state.per_day_completed = {DAY: {"master_path": None}}
    assert bc.day_needs_reburn(state, DAY) is True
    assert bc.day_needs_reburn(state, "2026-03-06") is False


def test_the_plan_shows_the_day_running(tmp_path):
    state = _saved_state()
    stages = _relabel(state, tmp_path)
    state.per_day_completed = {DAY: {"master_path": None}}
    row = preflight_plan._day_row(state, stages, DAY)
    assert row.status == "RUN"
    assert "re-burn" in row.reason


def test_only_a_really_finished_day_drops_the_mark(tmp_path):
    state = _saved_state()
    _relabel(state, tmp_path)
    crashed = SimpleNamespace(date=DAY, master_path=None, regular_lf_paths=[], short_paths=[])
    bc._record_day_and_save(crashed, state, None)
    assert DAY in state.reburn_days
    master = tmp_path / "master.mp4"
    master.write_bytes(b"m")
    done = SimpleNamespace(date=DAY, master_path=master, regular_lf_paths=[], short_paths=[])
    bc._record_day_and_save(done, state, None)
    assert DAY not in (state.reburn_days or {})
    assert state.per_day_completed[DAY]["master_path"] == str(master)


def test_the_mark_survives_a_state_round_trip(tmp_path):
    state = _saved_state()
    _relabel(state, tmp_path)
    again = PipelineState.from_dict(state.to_dict())
    assert DAY in again.reburn_days


def test_a_state_without_the_mark_behaves_as_before(tmp_path):
    stages = _stages(tmp_path)
    state = make_state(per_day_completed={})
    auto_populate_per_day_completed(state, stages, fresh=False)
    assert state.per_day_completed == {DAY: {"master_path": None}}
    assert bc.day_needs_reburn(state, DAY) is False
