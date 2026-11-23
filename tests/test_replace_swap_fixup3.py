"""A finished swap is judged by all of the Short's rows, a carried record is proven to be the swap,
and a Gate C/D confirmation is accepted whichever order it was taken in. Real queue, sidecars,
privacy gate and ``drain_queue``; only the probe and mask gate are injected."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from test_replace_rebuild import DATE, NOW, PROJECT, World, _no
from test_replace_swap_fixup import LATER, REAL, _attest, _drain, _open, _reasons, _state
from test_replace_swap_fixup2 import D2, LEASE, _crash, _rebuilt, _two_days

from gomoveshift.cli.handlers.enqueue_plan import GateNotApprovedError, assert_gate_cd_approved
from gomoveshift.pipeline.drive_state_model import PipelineState
from gomoveshift.pipeline.review_gate import compute_gate_cd_hash
from gomoveshift.publishing import dispatch, replace_rebuild
from gomoveshift.publishing.plan_io import load_publish_plan


@pytest.fixture
def w(tmp_path):
    world = World(tmp_path)
    yield world
    world.conn.close()


def _plan_hash(w, pos=0):
    return load_publish_plan(w.sidecar).shorts[pos].reviewed_plan_hash


def _record(w, pos=0):
    return json.loads(w.sidecar.read_text())["shorts"][pos]


def _confirm_with_plan(w):
    """A human confirmed Gate C/D while the swap's sidecar already existed (any run phase)."""
    state = {"gate_cd_confirmed": True}
    state["gate_cd_confirmation_hash"] = compute_gate_cd_hash(
        state=PipelineState.from_dict(state), output_dir=w.out
    )
    _state(w, **state)


def _edit(w, fn):
    raw = json.loads(w.sidecar.read_text())
    fn(raw["shorts"][0])
    w.sidecar.write_text(json.dumps(raw))


# ---- F1: a finished swap survives a sibling left at the old hash ----


@pytest.mark.parametrize("status", ["cancelled", "failed_permanent"])
def test_a_terminal_sibling_at_the_old_hash_never_reverts_a_finished_swap(w, status):
    _attest(w.stages)
    yt = w.row(1)
    w.row(1, "instagram", status=status)
    old = w.records[0]
    assert 1 in _rebuilt(w.run(privacy_fn=REAL))
    new = w.hash_of(yt)
    assert new != old.reviewed_plan_hash
    for over in ({}, {"mask_fn": _no}, {}):  # plain reruns and a refusing one
        r = w.run(privacy_fn=REAL, **over)
        assert 1 not in _reasons(r) and _plan_hash(w) == new and w.hash_of(yt) == new
    assert load_publish_plan(w.sidecar).shorts[0].media.sha256 != old.media.sha256
    assert _open(w, 1) == []
    calls, _ = _drain(w, LATER)
    assert len(calls) == 1  # the moved row drains on the new record


def test_an_awaiting_manual_sibling_and_a_later_refusal_keep_a_finished_swap(w):
    yt = w.row(1)
    w.row(1, "instagram", status="awaiting_manual_publish")
    assert 1 in _rebuilt(w.run())
    new = w.hash_of(yt)
    for _ in range(2):
        r = w.run(mask_fn=_no)
        assert 1 not in _reasons(r) and w.hash_of(yt) == new and _plan_hash(w) == new


def test_a_queued_straggler_at_the_old_hash_moves_to_the_finished_swaps_hash(w):
    yt = w.row(1)
    assert 1 in _rebuilt(w.run())
    new = w.hash_of(yt)
    ig = w.row(1, "instagram")  # enqueued at the old hash after the swap
    r = w.run()
    assert 1 in _rebuilt(r) and w.hash_of(ig) == new and w.hash_of(yt) == new
    assert _plan_hash(w) == new and _open(w, 1) == []


def test_a_refusing_run_leaves_the_moved_row_and_a_straggler_where_they_are(w):
    yt = w.row(1)
    old = w.records[0].reviewed_plan_hash
    assert 1 in _rebuilt(w.run())
    new = w.hash_of(yt)
    ig = w.row(1, "instagram")
    r = w.run(mask_fn=_no)
    assert "mask" in _reasons(r)[1]
    assert w.hash_of(yt) == new and w.hash_of(ig) == old and _plan_hash(w) == new
    assert _open(w, 1)  # the straggler's refusal stays visible
    again = w.run()  # the gate clears: the straggler follows with no hand step
    assert w.hash_of(ig) == new and 1 in _rebuilt(again) and _open(w, 1) == []


def test_an_uploaded_straggler_is_refused_visibly_and_the_finished_swap_stays(w):
    yt = w.row(1)
    assert 1 in _rebuilt(w.run())
    new = w.hash_of(yt)
    w.row(1, "instagram", platform_upload_id="ig1", status="uploaded")
    r = w.run()
    assert "half old" in _reasons(r)[1] and w.hash_of(yt) == new and _plan_hash(w) == new
    assert _open(w, 1)


def test_a_crash_with_every_other_row_terminal_completes_the_swap(w, monkeypatch):
    yt = w.row(1)
    w.row(1, "instagram", status="cancelled")
    old = w.hash_of(yt)
    _crash(w, monkeypatch)
    moved = w.hash_of(yt)
    assert moved != old and "shorts" not in json.loads(w.sidecar.read_text())  # ahead of the file
    r = w.run()
    assert 1 not in _reasons(r) and _plan_hash(w) == moved and w.hash_of(yt) == moved


def test_a_crash_with_every_other_row_terminal_is_restored_by_a_refusing_run(w, monkeypatch):
    yt = w.row(1)
    w.row(1, "instagram", status="cancelled")
    old = w.hash_of(yt)
    _crash(w, monkeypatch)
    w.run(mask_fn=_no)
    assert w.hash_of(yt) == old and _plan_hash(w) == old


def test_a_terminal_sibling_does_not_settle_a_short_that_still_has_a_movable_row(w):
    yt = w.row(1)
    w.row(1, "instagram", status="cancelled")
    old = w.hash_of(yt)
    assert 1 in _rebuilt(w.run()) and w.hash_of(yt) != old


# ---- F2: a carried record must be the deterministic swap of the old one ----

TAMPERS = {
    "title": lambda s: s["payloads"]["youtube"].__setitem__("title", "NEVER REVIEWED"),
    "description": lambda s: s["payloads"]["youtube"].__setitem__("description", "edited"),
    "tags": lambda s: s["payloads"]["youtube"].__setitem__("tags", ["x"]),
    "lf_index": lambda s: s.__setitem__("lf_index", 9),
    "schedule": lambda s: s["schedule"].__setitem__("timezone", "UTC"),
    "hash_not_derived": lambda s: s.__setitem__("reviewed_plan_hash", "f" * 64),
    "media_sha": lambda s: s["media"].__setitem__("sha256", "a" * 64),
    "media_path": lambda s: s["media"].__setitem__("path", "/elsewhere/short_01.mp4"),
    "consent": lambda s: s.__setitem__("consent_hash", "forged"),
}


@pytest.mark.parametrize("what", sorted(TAMPERS))
def test_an_edited_carried_record_is_refused_discarded_and_the_next_run_converges(w, what):
    yt = w.row(1)
    old = w.records[0].reviewed_plan_hash
    assert 1 in _rebuilt(w.run())
    new = w.hash_of(yt)
    _edit(w, TAMPERS[what])
    r = w.run()
    assert 1 in _reasons(r) and _open(w, 1)  # visible, never carried
    rec = _record(w)
    assert rec["payloads"]["youtube"]["title"] == "t1" and rec["reviewed_plan_hash"] == old
    assert rec["media"]["sha256"] == w.records[0].media.sha256
    assert w.hash_of(yt) == old  # the row follows the file back: it never serves unreviewed text
    again = w.run()  # nothing edited this time: the swap converges by itself
    assert 1 in _rebuilt(again) and w.hash_of(yt) == new and _plan_hash(w) == new
    assert _open(w, 1) == []


def test_an_edited_carried_record_is_refused_even_under_a_confirmed_gate(w):
    yt = w.row(1)
    assert 1 in _rebuilt(w.run())
    _confirm_with_plan(w)
    _edit(w, TAMPERS["title"])
    r = w.run()
    assert 1 in _reasons(r)
    assert _record(w)["payloads"]["youtube"]["title"] == "t1"
    assert w.hash_of(yt) == w.records[0].reviewed_plan_hash


def test_a_carried_record_whose_consent_followed_the_hash_passes(w):
    raw = json.loads(w.old_sidecar.read_text())
    raw["shorts"][0]["consent_hash"] = raw["shorts"][0]["reviewed_plan_hash"]
    w.old_sidecar.write_text(json.dumps(raw))
    yt = w.row(1)
    assert 1 in _rebuilt(w.run())
    new = w.hash_of(yt)
    assert _record(w)["consent_hash"] == new
    first = w.sidecar.read_bytes()
    r = w.run(mask_fn=_no)
    assert 1 not in _reasons(r) and _plan_hash(w) == new
    assert json.loads(first)["shorts"][0] == _record(w)


# ---- F3: a confirmation is accepted whichever order it was taken in ----


def test_a_confirmation_taken_with_the_sidecar_present_survives_every_rerun(w):
    w.row(1)
    w.row(2, lease_owner="x", lease_expires_at=LEASE())
    assert _rebuilt(w.run()) == [1]
    _confirm_with_plan(w)
    r = w.run(now=NOW + timedelta(minutes=10))
    assert 2 in _rebuilt(r) and not r.refused
    for k in (20, 30):
        r = w.run(now=NOW + timedelta(minutes=k))
        assert not r.refused and _open(w) == []


def test_a_confirmation_taken_before_any_swap_survives_every_rerun(w):
    w.sidecar.unlink()  # a replace run emits no plan: the confirmation covers none
    state = {"gate_cd_confirmed": True}
    state["gate_cd_confirmation_hash"] = compute_gate_cd_hash(
        state=PipelineState.from_dict(state), output_dir=w.out
    )
    _state(w, **state)
    w.row(1)
    w.row(2, lease_owner="x", lease_expires_at=LEASE())
    assert _rebuilt(w.run()) == [1]
    r = w.run(now=NOW + timedelta(minutes=10))
    assert 2 in _rebuilt(r) and not r.refused


@pytest.mark.parametrize("name", ["lf_youtube.json", "other_publish_plan.json"])
def test_a_content_change_after_a_confirmation_with_the_plan_present_still_refuses(w, name):
    (w.out / name).write_text(json.dumps({"title": "a"}))
    w.row(1)
    w.row(2, lease_owner="x", lease_expires_at=LEASE())
    assert _rebuilt(w.run()) == [1]
    _confirm_with_plan(w)
    (w.out / name).write_text(json.dumps({"title": "edited after the review"}))
    r = w.run(now=NOW + timedelta(minutes=10))
    assert r.rebuilt == [] and "re-confirmed" in _reasons(r)[2]


def _plan(w, **short):
    base = {
        "short_index": 1,
        "media": {"sha256": "a"},
        "reviewed_plan_hash": "h",
        "consent_hash": "c",
        "payloads": {"youtube": {"title": "t"}},
    }
    (w.out / "x_publish_plan.json").write_text(json.dumps({"shorts": [base | short]}))


def _gate_hash(w, **kw):
    return compute_gate_cd_hash(state=PipelineState.from_dict({}), output_dir=w.out, **kw)


def test_the_gate_hash_ignores_the_swapped_fields_but_not_the_text(w):
    _plan(w)
    base = _gate_hash(w)
    _plan(w, media={"sha256": "b"}, reviewed_plan_hash="h2", consent_hash="c2")
    assert _gate_hash(w) == base
    _plan(w, payloads={"youtube": {"title": "t2"}})
    assert _gate_hash(w) != base
    _plan(w, short_index=2)
    assert _gate_hash(w) != base


def test_a_confirmation_recorded_with_the_full_plan_hash_is_still_approved(w):
    _plan(w)
    legacy = _gate_hash(w, full_plans=True)
    assert legacy != _gate_hash(w)
    state = PipelineState.from_dict(
        {"gate_cd_confirmed": True, "gate_cd_confirmation_hash": legacy}
    )
    assert_gate_cd_approved(state, w.out)
    _plan(w, payloads={"youtube": {"title": "t2"}})
    with pytest.raises(GateNotApprovedError):
        assert_gate_cd_approved(state, w.out)


# ---- F4: a reclaimed stage copy of a finished day never holds the other Shorts ----


def test_a_finished_days_reclaimed_stage_copy_never_holds_the_other_day(w):
    _two_days(w)
    _attest(w.stages, DATE)
    _attest(w.stages, D2)
    w.row(1, lease_owner="x", lease_expires_at=LEASE())
    w.row(2)
    assert _rebuilt(w.run(privacy_fn=REAL)) == [2]
    (w.stages / f"day_{D2}" / f"day_{D2}_privacy_attestation.json").unlink()  # reclaimed
    r = w.run(now=NOW + timedelta(minutes=10), privacy_fn=REAL)
    assert _rebuilt(r) == [1] and r.deferred == []
    w.conn.execute("UPDATE deliveries SET lease_owner=NULL, lease_expires_at=NULL")
    w.conn.commit()
    first, _ = _drain(w, LATER)
    second, _ = _drain(w, LATER + timedelta(days=1))  # the daily cap spaces the two uploads
    assert len({*first, *second}) == 2  # both drain: the verified copy beside the plan covers D2


def test_a_missing_beside_copy_of_a_finished_day_still_holds_the_other_day(w):
    _two_days(w)
    _attest(w.stages, DATE)
    _attest(w.stages, D2)
    w.row(1, lease_owner="x", lease_expires_at=LEASE())
    w.row(2)
    assert _rebuilt(w.run(privacy_fn=REAL)) == [2]
    (w.stages / f"day_{D2}" / f"day_{D2}_privacy_attestation.json").unlink()
    (w.out / f"day_{D2}_privacy_attestation.json").unlink()  # nothing verifies D2 anywhere
    r = w.run(now=NOW + timedelta(minutes=10), privacy_fn=REAL)
    assert [d.short_index for d in r.deferred] == [1] and D2 in r.deferred[0].reason


def test_a_day_with_nothing_to_move_still_holds_a_staged_short_when_unattested(w):
    _two_days(w)
    _attest(w.stages, DATE)  # D2 is never attested
    w.row(1)
    w.row(2, lease_owner="x", lease_expires_at=LEASE())  # Short 2 waits: its own gate never runs
    r = w.run(privacy_fn=REAL)
    assert r.rebuilt == [] and D2 in {d.reason for d in r.deferred if d.short_index == 1}.pop()


def test_a_verified_beside_copy_survives_a_run_that_stages_another_day(w):
    _two_days(w)
    _attest(w.stages, DATE)
    _attest(w.stages, D2)
    w.row(1)
    w.row(2, lease_owner="x", lease_expires_at=LEASE())
    assert _rebuilt(w.run(privacy_fn=REAL)) == [1]
    beside = w.out / f"day_{DATE}_privacy_attestation.json"
    (w.stages / f"day_{DATE}" / f"day_{DATE}_privacy_attestation.json").unlink()
    r = w.run(now=NOW + timedelta(minutes=10), privacy_fn=REAL)  # Short 2 stages and places
    assert _rebuilt(r) == [2]
    classes = json.loads(beside.read_text())["classes"]
    assert all(c["state"] == "verified" for c in classes.values()) and classes


# ---- F5-F8 ----


def test_an_attestation_with_no_date_field_is_refused_with_that_cause(w):
    _attest(w.stages, DATE)
    p = w.stages / f"day_{DATE}" / f"day_{DATE}_privacy_attestation.json"
    doc = json.loads(p.read_text())
    del doc["date"]
    p.write_text(json.dumps(doc))
    assert "date" in replace_rebuild.privacy_refusal(w.stages, DATE)
    w.row(1)
    assert w.run(privacy_fn=REAL).rebuilt == []


def test_a_null_path_row_is_not_served_from_a_same_named_plan_in_another_dir(w, tmp_path):
    did = w.row(1, sidecar_path=None)
    other = tmp_path / "elsewhere"
    other.mkdir()
    r = w.run(new_sidecar=other / f"{PROJECT}_publish_plan.json")
    assert 1 in _reasons(r) and w.hash_of(did) == w.records[0].reviewed_plan_hash


def test_a_null_path_row_follows_the_sidecar_dir_dispatch_reads(w, tmp_path):
    did = w.row(1, sidecar_path=None)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    assert 1 in _reasons(w.run(sidecar_dir=elsewhere))
    assert w.hash_of(did) == w.records[0].reviewed_plan_hash
    assert (
        1 in _rebuilt(w.run(sidecar_dir=w.out))
        and w.hash_of(did) != w.records[0].reviewed_plan_hash
    )


def test_a_null_path_row_follows_the_newest_recorded_sidecar_not_the_oldest(w):
    w.row(1, "instagram", status="cancelled")  # oldest recorded path: this plan
    did = w.row(1, sidecar_path=None)
    w.row(2, "instagram", status="cancelled", sidecar_path="/elsewhere/other_publish_plan.json")
    r = w.run()
    assert 1 in _reasons(r) and w.hash_of(did) == w.records[0].reviewed_plan_hash


def test_a_zero_backoff_base_is_one_attempt_and_never_escapes_the_drain(w):
    assert dispatch.media_stall_attempts(0, 60) == 1
    assert dispatch.media_stall_attempts(-5, 60) == 1
    res = dispatch.drain_queue(
        w.conn, {}, now=LATER.isoformat(), incidents_dir=w.incidents, backoff_base_seconds=0
    )
    assert res.errors == []


def test_one_short_missing_its_date_key_is_refused_and_its_sibling_still_moves(w):
    raw = json.loads(w.old_sidecar.read_text())
    del raw["shorts"][0]["recording_date"]
    w.old_sidecar.write_text(json.dumps(raw))
    a, b = w.row(1), w.row(2)
    r = w.run()
    assert "no recording_date" in _reasons(r)[1] and _rebuilt(r) == [2]
    assert w.hash_of(a) == w.records[0].reviewed_plan_hash
    assert w.hash_of(b) == _plan_hash(w) and _open(w, 1)  # the dropped Short leaves pos 0
    load_publish_plan(w.sidecar)  # the written file still parses


# ---- the restore and the classifier keep their own filters ----


def _leftover(w, monkeypatch):
    did = w.row(1)
    _crash(w, monkeypatch)
    return did, w.hash_of(did)


@pytest.mark.parametrize(
    "sql",
    [
        "lease_owner='x', lease_expires_at='{lease}'",
        "platform_upload_id='y'",
        "sidecar_path='/elsewhere/p_publish_plan.json'",
        "recovers_delivery_id=999",
    ],
)
def test_a_crash_leftover_that_is_leased_uploaded_foreign_or_a_recovery_is_never_restored(
    w, monkeypatch, sql
):
    did, moved = _leftover(w, monkeypatch)
    w.conn.execute(
        f"UPDATE deliveries SET {sql.format(lease=LEASE())} WHERE id=?",  # noqa: S608
        (did,),
    )
    w.conn.commit()
    w.run(mask_fn=_no)
    assert w.hash_of(did) == moved


def test_a_crash_leftover_that_is_plain_is_restored(w, monkeypatch):
    did, moved = _leftover(w, monkeypatch)
    w.run(mask_fn=_no)
    assert w.hash_of(did) == w.records[0].reviewed_plan_hash != moved


def test_a_published_sibling_with_no_upload_id_does_not_refuse_the_short(w):
    yt = w.row(1)
    w.row(1, "instagram", status="published")
    old = w.hash_of(yt)
    r = w.run()
    assert 1 in _rebuilt(r) and w.hash_of(yt) != old


def test_a_finished_recovery_row_is_refused_visibly_never_settled_silently(w):
    w.row(1, status="cancelled", recovers_delivery_id=999)
    old = w.records[0].reviewed_plan_hash
    r = w.run()
    assert "recovery row" in _reasons(r)[1] and _plan_hash(w) == old
    assert _open(w, 1)
