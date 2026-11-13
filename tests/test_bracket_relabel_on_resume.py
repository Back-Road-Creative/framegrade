"""A bracket label the rules now word differently replaces the old text on resume.

2026-11-13, nctopa-2026-03-05-rec0063: the drive was staged while a two-state bracket
earned only the country, so its checkpoint, 3 long-forms and 21 Shorts carry "United
States". The record keeps both geocoded fixes, so the label is recomputed from the
record itself (no geocoder, no fix files) and, when it differs, the old text is
dropped the way a corrected ``--placeless-label`` is.
"""

from __future__ import annotations

from gomoveshift.pipeline import bracket_route as br
from gomoveshift.pipeline.drive_orchestrator import relabel_bracket_on_resume
from gomoveshift.pipeline.drive_state_model import PER_LF_OVERRIDE_MIRROR_FIELDS

from _state_helpers import make_state

OLD = "United States"
NEW = "Virginia to Pennsylvania"
PHASES = ["phase_a", "phase_bc_day_first", "phase_d"]
DAY = "2026-03-05"


def _record(label=OLD, level="country", gap_h=17.535):
    return {
        "label": label,
        "level": level,
        "gap_h": gap_h,
        "before": {"time": "2026-03-04T23:28:05Z"},
        "after": {"time": "2026-03-05T17:00:12Z"},
        "before_place": {"county": "Roanoke County", "state": "Virginia", "country": OLD},
        "after_place": {"county": "Franklin County", "state": "Pennsylvania", "country": OLD},
    }


def _saved_state(record=None):
    label = (record or _record())["label"]
    mirror = {1: label, 2: label, 3: label}
    return make_state(
        bracket_route=record or _record(),
        location=label,
        per_day_locations={DAY: label},
        per_lf_overrides=dict(mirror),
        per_lf_locations=dict(mirror),
        per_lf_ost_text=dict(mirror),
        per_lf_titles=dict(mirror),
        per_day_completed={DAY: {"master_path": "/x/m.mp4"}},
        ost_audit_passed={DAY: True},
        ost_audit_hash={DAY: "a" * 64},
        shorts_ost_system_review={DAY: {"text_hash": "b"}},
        superlong_ost_system_review={DAY: {"text_hash": "c"}},
        phases_completed=list(PHASES),
    )


def _run(state, tmp_path):
    stages_completed = list(PHASES)
    out = relabel_bracket_on_resume(
        state,
        list(stages_completed),
        stages_completed,
        output_base=tmp_path / "out",
        stages_dir=tmp_path / "out" / "_stages",
    )
    return out, stages_completed


def test_the_record_recomputes_its_own_label():
    assert br.recomputed_region(_record()) == ("states", NEW)
    assert br.recomputed_region(_record(gap_h=40.0)) == ("country", OLD)
    assert br.recomputed_region(_record(gap_h=60.0)) is None
    assert br.recomputed_region({"label": OLD}) is None
    assert br.recomputed_region(None) is None


def test_the_record_and_every_mirror_take_the_new_label(tmp_path):
    state = _saved_state()
    _run(state, tmp_path)
    assert (state.bracket_route["level"], state.bracket_route["label"]) == ("states", NEW)
    assert br.bracket_record_valid(state.bracket_route)
    assert state.per_lf_overrides == {1: NEW, 2: NEW, 3: NEW}
    for field in PER_LF_OVERRIDE_MIRROR_FIELDS:
        assert getattr(state, field) == {1: NEW, 2: NEW, 3: NEW}, field
    assert state.location == NEW
    assert state.per_day_locations == {DAY: NEW}


def test_burned_phases_and_old_confirmations_are_dropped(tmp_path):
    state = _saved_state()
    out = tmp_path / "out"
    (out / "_stages").mkdir(parents=True)
    sidecar = out / "_stages" / f"ost_audit_{DAY}.json"
    marker = out / f"{DAY}.confirmed"
    sidecar.write_text("{}")
    marker.write_text("")
    resumed, stages_completed = _run(state, tmp_path)
    assert resumed == ["phase_a"] == stages_completed == state.phases_completed
    assert state.per_day_completed == {} and state.ost_audit_passed == {}
    assert state.ost_audit_hash == {} and state.shorts_ost_system_review == {}
    assert not sidecar.exists() and not marker.exists()


def test_an_operator_location_is_left_alone(tmp_path):
    state = _saved_state()
    state.location = "Shenandoah Valley"
    _run(state, tmp_path)
    assert state.location == "Shenandoah Valley"


def test_an_unchanged_label_is_a_no_op(tmp_path):
    state = _saved_state(_record(label=NEW, level="states"))
    before = state.to_dict()
    resumed, stages_completed = _run(state, tmp_path)
    assert resumed == PHASES == stages_completed
    assert state.to_dict() == before


def test_no_record_or_an_unreadable_one_is_a_no_op(tmp_path):
    for record in (None, {"label": OLD, "level": "country"}):
        state = _saved_state()
        state.bracket_route = record
        resumed, _ = _run(state, tmp_path)
        assert resumed == PHASES
        assert state.per_lf_overrides == {1: OLD, 2: OLD, 3: OLD}
