"""A corrected ``--placeless-label`` replaces the old text on resume.

"if we cant verify something then it cant be said": a drive staged under a
claiming label ("Through The Valleys And Ridges") and healed with a claim-free
one must render the NEW text everywhere. ``placeless_label`` is CLI-wins on
resume, but the checkpoint brings back the per-LF mirrors with the OLD text and
``--per-lf-location`` (the only thing that re-mirrors) is refused beside it.
"""

from __future__ import annotations

from pathlib import Path

from gomoveshift.config import Config
from gomoveshift.pipeline.drive_orchestrator import relabel_placeless_on_resume
from gomoveshift.pipeline.drive_state_model import PER_LF_OVERRIDE_MIRROR_FIELDS
from gomoveshift.pipeline.flag_registry import merge_on_resume

from _state_helpers import make_state

OLD = "Through The Valleys And Ridges"
NEW = "Somewhere On The Road"
PHASES = ["phase_a", "phase_bc_day_first", "phase_d"]


def _saved_state(label=OLD):
    """The checkpoint a placeless drive left: label mirrored onto both LFs."""
    mirror = {1: label, 2: label}
    return make_state(
        placeless_label=label,
        per_lf_overrides=dict(mirror),
        per_lf_locations=dict(mirror),
        per_lf_ost_text=dict(mirror),
        per_lf_titles=dict(mirror),
        per_day_completed={"2026-03-05": {"master_path": "/x/m.mp4"}},
        ost_audit_passed={"2026-03-05": True},
        ost_audit_hash={"2026-03-05": "a" * 64},
        shorts_ost_system_review={"2026-03-05": {"text_hash": "b"}},
        superlong_ost_system_review={"2026-03-05": {"text_hash": "c"}},
        phases_completed=list(PHASES),
    )


def _run(state, tmp_path, phases=None):
    stages_completed = list(phases if phases is not None else PHASES)
    out = relabel_placeless_on_resume(
        state,
        list(stages_completed),
        stages_completed,
        output_base=tmp_path / "out",
        stages_dir=tmp_path / "out" / "_stages",
    )
    return out, stages_completed


def test_new_label_replaces_every_mirror_for_every_lf(tmp_path):
    state = _saved_state()
    state.placeless_label = NEW
    _run(state, tmp_path)
    assert state.per_lf_overrides == {1: NEW, 2: NEW}
    for field in PER_LF_OVERRIDE_MIRROR_FIELDS:
        assert getattr(state, field) == {1: NEW, 2: NEW}, field


def test_no_old_text_survives_anywhere_on_state(tmp_path):
    state = _saved_state()
    state.placeless_label = NEW
    _run(state, tmp_path)
    assert OLD not in repr(state.to_dict())


def test_burned_phases_are_dropped_and_phase_a_kept(tmp_path):
    state = _saved_state()
    state.placeless_label = NEW
    resumed, stages_completed = _run(state, tmp_path)
    assert resumed == ["phase_a"]
    assert stages_completed == ["phase_a"]
    assert state.phases_completed == ["phase_a"]


def test_cached_finals_and_old_confirmations_do_not_carry_over(tmp_path):
    state = _saved_state()
    state.placeless_label = NEW
    _run(state, tmp_path)
    assert not state.per_day_completed
    assert not state.ost_audit_passed
    assert not state.ost_audit_hash
    assert not state.shorts_ost_system_review
    assert not state.superlong_ost_system_review


def test_old_sidecar_and_confirmed_markers_are_removed(tmp_path):
    out = tmp_path / "out"
    stages = out / "_stages"
    stages.mkdir(parents=True)
    keep = out / "unrelated.json"
    keep.write_text("{}")
    gone = [
        out / "ost_audit_2026-03-05.json",
        out / "ost_audit_2026-03-05.confirmed",
        out / "shorts_ost_audit_2026-03-05.confirmed",
        stages / "ost_audit_2026-03-05.confirmed",
    ]
    for p in gone:
        p.write_text(OLD)
    state = _saved_state()
    state.placeless_label = NEW
    _run(state, tmp_path)
    assert [p for p in gone if p.exists()] == []
    assert keep.exists()


def test_unchanged_label_is_a_noop(tmp_path):
    state = _saved_state(NEW)
    before = state.to_dict()
    resumed, stages_completed = _run(state, tmp_path)
    assert resumed == PHASES and stages_completed == PHASES
    assert state.to_dict() == before


def test_unchanged_label_keeps_the_old_confirmation(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    marker = out / "ost_audit_2026-03-05.confirmed"
    marker.write_text("x")
    state = _saved_state(NEW)
    _run(state, tmp_path)
    assert marker.exists()


def test_a_drive_without_a_placeless_label_is_untouched(tmp_path):
    state = _saved_state()
    state.placeless_label = None
    before = state.to_dict()
    resumed, _ = _run(state, tmp_path)
    assert resumed == PHASES
    assert state.to_dict() == before


def test_through_the_real_merge_the_new_cli_label_wins(tmp_path):
    saved = _saved_state().to_dict()
    state = make_state(placeless_label=NEW)  # this invocation's CLI-built state
    merge_on_resume(state, saved, config=Config())
    assert state.placeless_label == NEW
    assert state.per_lf_ost_text == {1: OLD, 2: OLD}  # the bug: old text restored
    resumed, _ = _run(state, tmp_path)
    assert state.per_lf_ost_text == {1: NEW, 2: NEW}
    assert resumed == ["phase_a"]
    assert OLD not in repr(state.to_dict())


def test_run_wires_the_relabel_after_the_override_preserve():
    import inspect

    from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

    src = inspect.getsource(DrivePipelineOrchestrator.run)
    assert src.index("preserve_per_lf_overrides_on_resume(") < src.index(
        "relabel_placeless_on_resume("
    )
    assert src.index("relabel_placeless_on_resume(") < src.index(
        "invalidate_phase_d_if_fingerprint_stale("
    )
