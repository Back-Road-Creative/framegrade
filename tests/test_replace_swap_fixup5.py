"""Fixup 5: dispatch holds a duplicate-index or undated Short itself, and the replace gate's swap
projection covers only the replace run's own plan. Real queue, sidecars and ``drain_queue``; only
the probe and the mask gate are injected."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from test_replace_rebuild import NOW, PROJECT, World
from test_replace_swap_fixup import (
    LATER,
    REAL,
    _attest,
    _confirmed_fields,
    _drain,
    _open,
    _reasons,
    _state,
)
from test_replace_swap_fixup2 import LEASE

from gomoveshift.cli.handlers.dispatch_queue import SidecarRecordLoader
from gomoveshift.cli.handlers.enqueue_plan import GateNotApprovedError, assert_gate_cd_approved
from gomoveshift.pipeline.drive_state_model import PipelineState
from gomoveshift.pipeline.review_gate import compute_gate_cd_hash, record_gate_cd_confirmation
from gomoveshift.publishing.asset_plan import review_hash
from gomoveshift.publishing.dispatch import TransientMediaError


@pytest.fixture
def w(tmp_path):
    world = World(tmp_path)
    yield world
    world.conn.close()


def _rebuilt(r):
    return [x.short_index for x in r.rebuilt]


def _row(w, did):
    return w.conn.execute("SELECT * FROM deliveries WHERE id=?", (did,)).fetchone()


# ---- MAJOR-1: a duplicate short_index never uploads another Short's video ----


def _dup_world(w, identical_second_render):
    """The old plan lists Short 2's record a second time under index 1 (no day)."""
    raw = json.loads(w.old_sidecar.read_text())
    dup = dict(raw["shorts"][1])
    dup["short_index"] = 1
    dup.pop("recording_date")
    raw["shorts"].append(dup)
    w.old_sidecar.write_text(json.dumps(raw))
    if identical_second_render:  # Short 2's re-render came out byte-identical
        (w.out / "shorts" / "lf_01" / "short_02.mp4").write_bytes(b"old-bytes-2")
    return raw


def _undup(w, raw):
    raw["shorts"] = raw["shorts"][:2]
    w.old_sidecar.write_text(json.dumps(raw))


def test_dispatch_never_resolves_a_swapped_short_to_a_duplicate_record(w):
    _dup_world(w, identical_second_render=False)
    w.row(1)
    w.run()
    with pytest.raises(TransientMediaError, match="more than once"):
        SidecarRecordLoader(w.out, conn=w.conn)(PROJECT, 1)
    with pytest.raises(TransientMediaError):  # the whole project is held, not just index 1
        SidecarRecordLoader(w.out, conn=w.conn)(PROJECT, 2)


def test_every_reader_of_the_plan_refuses_a_duplicate_index(w):
    from gomoveshift.publishing.plan_io import (
        DuplicateShortIndexError,
        PublishPlanFormatError,
        load_publish_plan,
    )

    raw = json.loads(_dup_world(w, identical_second_render=False) and w.old_sidecar.read_text())
    for rec in raw["shorts"]:
        rec.setdefault("recording_date", None)
    w.old_sidecar.write_text(json.dumps(raw))
    with pytest.raises(DuplicateShortIndexError, match="more than once"):
        load_publish_plan(w.old_sidecar)
    assert issubclass(DuplicateShortIndexError, PublishPlanFormatError)  # callers already catch it
    assert load_publish_plan(
        w.old_sidecar, allow_duplicate_index=True
    ).shorts  # writers' check only


def test_the_rebuild_refuses_every_position_of_a_duplicate_index_with_one_incident(w):
    raw = _dup_world(w, identical_second_render=False)
    did = w.row(1)
    old = w.hash_of(did)
    r = w.run()
    assert 1 not in _rebuilt(r)
    assert _reasons(r)[1].count("short_index 1") >= 1 and "more than once" in _reasons(r)[1]
    assert [x.short_index for x in r.refused].count(1) == 1  # one refusal for the index
    assert len(_open(w, 1)) == 1 and w.hash_of(did) == old  # one incident, row held
    on_disk = json.loads(w.sidecar.read_text())["shorts"]
    ones = [s for s in on_disk if s["short_index"] == 1]
    # old records, untouched (the plan parser's missing day goes back as null)
    assert ones == [{"recording_date": None, **s} for s in raw["shorts"] if s["short_index"] == 1]


def test_a_duplicate_index_never_uploads_another_shorts_video_and_converges(w):
    _attest(w.stages)
    raw = _dup_world(w, identical_second_render=True)
    did = w.row(1)
    w.run(privacy_fn=REAL)
    calls, res = _drain(w, LATER)
    assert calls == [] and res.succeeded == 0 and res.errors  # held, with a reason
    row = _row(w, did)
    assert row["status"] == "queued" and row["error_class"] == "RecordHeldError"
    assert _open(w, 1)
    _undup(w, raw)  # the duplicate is fixed
    r = w.run(privacy_fn=REAL, now=LATER)
    assert 1 in _rebuilt(r) and _open(w, 1) == []
    calls, _ = _drain(w, LATER + timedelta(hours=3))  # no hand step
    assert [c.name for c in calls] == ["short_01.mp4"]
    assert calls[0].read_bytes() == b"new-bytes-1"


def test_a_duplicate_index_row_is_never_burned_however_often_it_is_drained(w):
    _attest(w.stages)
    _dup_world(w, identical_second_render=True)
    did = w.row(1)
    w.run(privacy_fn=REAL)
    for n in range(12):
        _drain(w, LATER + timedelta(days=n))
    assert _row(w, did)["status"] == "queued"


# ---- MAJOR-2: an undated Short refused "rows held" is really held ----


def _undated_world(w):
    _attest(w.stages)
    raw = json.loads(w.old_sidecar.read_text())
    del raw["shorts"][0]["recording_date"]
    w.old_sidecar.write_text(json.dumps(raw))
    (w.out / "shorts" / "lf_01" / "short_01.mp4").write_bytes(b"old-bytes-1")  # identical render
    return raw


def test_an_undated_short_with_unchanged_bytes_is_held_in_dispatch(w):
    raw = _undated_world(w)
    did1, did2 = w.row(1), w.row(2)
    r = w.run(privacy_fn=REAL)
    assert "no recording_date" in _reasons(r)[1] and "rows held" in json.dumps(_open(w, 1))
    calls, res = _drain(w, LATER)
    assert not any(c.name == "short_01.mp4" for c in calls)  # the incident told the truth
    row = _row(w, did1)
    assert row["status"] == "queued" and row["error_class"] == "RecordHeldError"
    assert "recording_date" in row["error_detail"]
    assert _open(w, 1)
    assert _row(w, did2)["status"] != "failed_permanent"
    raw["shorts"][0]["recording_date"] = w.records[0].recording_date  # the day is supplied
    w.old_sidecar.write_text(json.dumps(raw))
    assert 1 in _rebuilt(w.run(privacy_fn=REAL, now=LATER))
    calls, _ = _drain(w, LATER + timedelta(hours=3))
    assert any(c.name == "short_01.mp4" for c in calls)  # converged with no hand step


def test_an_undated_row_is_never_burned_however_often_it_is_drained(w):
    _undated_world(w)
    did = w.row(1)
    w.run(privacy_fn=REAL)
    for n in range(12):
        _drain(w, LATER + timedelta(days=n))
    assert _row(w, did)["status"] == "queued"


def test_the_loader_holds_a_null_dated_record_and_serves_a_dated_one(w):
    raw = json.loads(w.old_sidecar.read_text())
    raw["shorts"][0]["recording_date"] = None
    w.old_sidecar.write_text(json.dumps(raw))
    w.sidecar.write_text(w.old_sidecar.read_text())
    w.row(1)
    loader = SidecarRecordLoader(w.out, conn=w.conn)
    with pytest.raises(TransientMediaError, match="recording_date"):
        loader(PROJECT, 1)
    assert loader(PROJECT, 2).recording_date  # a dated Short of the same plan is served


# ---- MINOR-3: the swap projection covers only the replace run's own plan ----

SIB = {
    "short_index": 1,
    "media": {"path": "/sib/short_01.mp4", "sha256": "a" * 64, "size_bytes": 10},
    "reviewed_plan_hash": review_hash("t", "d", ["a"], "a" * 64),
    "consent_hash": "consent-v1",
    "payloads": {"youtube": {"title": "t", "description": "d", "tags": ["a"]}},
}
SIB_EDITS = {
    "changed_consent": {"consent_hash": "consent-v2-someone-else"},
    "repointed_media": {
        "media": {"path": "/sib/other.mp4", "sha256": "b" * 64, "size_bytes": 99},
        "reviewed_plan_hash": review_hash("t", "d", ["a"], "b" * 64),
    },
}


def _sibling_world(w):
    sib = w.out / "sibling-project_publish_plan.json"
    sib.write_text(json.dumps({"shorts": [SIB]}))
    w.row(1)
    w.row(2, lease_owner="x", lease_expires_at=LEASE())
    assert _rebuilt(w.run()) == [1]
    _state(w, **_confirmed_fields(w.out))  # confirmed with the own sidecar and a sibling present
    return sib


@pytest.mark.parametrize("edit", sorted(SIB_EDITS))
def test_a_sibling_plans_swap_field_edit_is_refused_by_the_replace_gate(w, edit):
    sib = _sibling_world(w)
    sib.write_text(json.dumps({"shorts": [{**SIB, **SIB_EDITS[edit]}]}))
    state = PipelineState.from_dict(json.loads((w.stages / "pipeline_state.json").read_text()))
    with pytest.raises(GateNotApprovedError):  # enqueue-plan refuses the real change
        assert_gate_cd_approved(state, w.out)
    r = w.run(now=NOW + timedelta(minutes=10))
    assert r.rebuilt == [] and "re-confirmed" in _reasons(r)[2]
    assert "re-confirmed" in json.dumps(_open(w))


def test_an_untouched_sibling_plan_still_lets_the_own_swap_through(w):
    _sibling_world(w)
    r = w.run(now=NOW + timedelta(minutes=10))
    assert _rebuilt(r) == [2] and not r.refused and _open(w) == []


def test_a_sibling_plans_title_edit_is_refused_by_the_replace_gate(w):
    sib = _sibling_world(w)
    edited = {**SIB, "payloads": {"youtube": {"title": "T2", "description": "d", "tags": ["a"]}}}
    sib.write_text(json.dumps({"shorts": [edited]}))
    r = w.run(now=NOW + timedelta(minutes=10))
    assert r.rebuilt == [] and "re-confirmed" in json.dumps(_open(w))


def test_the_hash_projects_only_the_named_plans_and_the_confirmation_records_one_per_plan(w):
    for name in ("a", "b"):
        (w.out / f"{name}_publish_plan.json").write_text(json.dumps({"shorts": [SIB]}))
    state = PipelineState.from_dict({"gate_cd_confirmed": True})

    def h(**kw):
        return compute_gate_cd_hash(state=state, output_dir=w.out, **kw)

    whole = h()
    a_only = h(project_plans=("a_publish_plan.json",))
    assert a_only != whole and h(project_plans=("b_publish_plan.json",)) != a_only
    (w.out / "b_publish_plan.json").write_text(
        json.dumps({"shorts": [{**SIB, **SIB_EDITS["changed_consent"]}]})
    )
    assert h(project_plans=("a_publish_plan.json",)) != a_only  # b is hashed whole, so it moved
    assert h(project_plans=("a_publish_plan.json", "b_publish_plan.json")) == h(
        project_plans=("a_publish_plan.json", "b_publish_plan.json")
    )
    record_gate_cd_confirmation(state, w.out)
    assert {"a_publish_plan.json", "b_publish_plan.json"} <= set(
        state.gate_cd_confirmation_swap_hash
    )  # one entry per plan in the dir (the world's own plan too)
    assert state.gate_cd_confirmation_swap_hash["a_publish_plan.json"] == h(
        project_plans=("a_publish_plan.json",)
    )


def test_a_confirmation_with_no_entry_for_the_plan_refuses_the_swap_projection(w):
    own = "x_publish_plan.json"
    (w.out / own).write_text(json.dumps({"shorts": [SIB]}))
    state = PipelineState.from_dict({"gate_cd_confirmed": True})
    record_gate_cd_confirmation(state, w.out)
    (w.out / own).write_text(
        json.dumps({"shorts": [{**SIB, "media": {**SIB["media"], "sha256": "c" * 64}}]})
    )
    assert_gate_cd_approved(state, w.out, swap_plan=own)  # the recorded plan: accepted
    with pytest.raises(GateNotApprovedError):
        assert_gate_cd_approved(state, w.out, swap_plan="other_publish_plan.json")
    state.gate_cd_confirmation_swap_hash = "a-legacy-single-string"
    with pytest.raises(GateNotApprovedError, match="re-confirm Gate C/D"):
        assert_gate_cd_approved(state, w.out, swap_plan=own)  # legacy: visible, never a crash


# ---- MINOR-4: a fingerprint error on a settled carried Short is refused, never settled ----


def test_a_fingerprint_error_on_a_settled_carried_short_is_refused_with_an_incident(w):
    w.row(1)
    assert 1 in _rebuilt(w.run())
    assert _open(w, 1) == []

    def boom(path):
        raise OSError("drive I/O error")

    r = w.run(fingerprint_fn=boom)
    assert "cannot fingerprint" in _reasons(r).get(1, "")
    assert _open(w, 1)


def test_reslot_names_a_duplicate_index_plan_instead_of_crashing(w):
    from gomoveshift.cli.handlers.queue_reslot import _record_for, _RefusedError

    raw = _dup_world(w, identical_second_render=False)
    for rec in raw["shorts"]:
        rec.setdefault("recording_date", "2026-03-05")
    w.old_sidecar.write_text(json.dumps(raw))
    row = {"sidecar_path": str(w.old_sidecar), "short_index": 1}
    with pytest.raises(_RefusedError, match="more than once"):
        _record_for(row)
