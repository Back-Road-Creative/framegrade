"""The replace swap proves it is the same clip, covers the Short's own day and never goes silent.
Real queue, sidecars, privacy gate and ``drain_queue``; only the probe and mask gate are injected."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import pytest
from test_replace_rebuild import DATE, FAR, NOW, PROJECT, World, _no

from gomoveshift.cli.handlers import enqueue_plan
from gomoveshift.cli.handlers.dispatch_queue import (
    PrivacyAttestationLoader,
    SidecarRecordLoader,
    _verified_path,
)
from gomoveshift.health.store import load_all
from gomoveshift.pipeline.drive_state_model import PipelineState
from gomoveshift.pipeline.review_gate import record_gate_cd_confirmation
from gomoveshift.publishing import dispatch, replace_rebuild
from gomoveshift.publishing.dispatch import PRIVACY_REQUIRED_CLASSES, TransientMediaError
from gomoveshift.publishing.plan_io import load_publish_plan
from gomoveshift.publishing.queue import enqueue, open_queue_db
from gomoveshift.publishing.queue_short_text import LeaseRaceError

REAL = replace_rebuild.privacy_refusal
LATER = NOW + timedelta(days=3, minutes=1)


@pytest.fixture
def w(tmp_path):
    world = World(tmp_path)
    yield world
    world.conn.close()


def _attest(stages, date=DATE, state="verified"):
    classes = {c: {"state": state} for c in PRIVACY_REQUIRED_CLASSES}
    doc = {"schema": "gms.privacy_attestation/v1", "date": date, "classes": classes}
    (stages / f"day_{date}").mkdir(parents=True, exist_ok=True)
    (stages / f"day_{date}" / f"day_{date}_privacy_attestation.json").write_text(json.dumps(doc))


def _audit(stages, entries):
    (stages / f"shorts_ost_audit_{DATE}.json").write_text(json.dumps({"shorts": entries}))


def _entry(rank, start, length=21.5, **extra):
    return {
        "rank": rank,
        "parent_lf_index": 1,
        "start_sec": start,
        "end_sec": start + length,
    } | extra


def _reasons(result):
    return {r.short_index: r.reason for r in result.refused}


def _open(w, index=None):
    return [i for i in w.incidents_open() if index in (None, i.get("short_index"))]


def _drain(w, at):
    calls = []
    loader = SidecarRecordLoader(w.out, conn=w.conn)

    def handler(row):
        calls.append(_verified_path(loader(row["project_name"], row["short_index"])))

    res = dispatch.drain_queue(
        w.conn,
        {"youtube": handler},
        now=at.isoformat(),
        privacy_attestations=PrivacyAttestationLoader(w.out),
        incidents_dir=w.incidents,
    )
    return calls, res


def test_a_different_segment_at_the_same_rank_is_refused(w):
    a = w.row(1)
    old = w.hash_of(a)
    _audit(w.stages, [_entry(1, 300.0), _entry(2, 900.0)])  # re-planned: old run had 0.0
    result = w.run()
    assert (
        result.rebuilt == []
        and "segment" in _reasons(result)[1]
        and set(_reasons(result)) == {1, 2}
    )
    assert w.hash_of(a) == old and _open(w, 1)
    assert load_publish_plan(w.sidecar).shorts[0].media.sha256 == w.records[0].media.sha256


def test_no_old_audit_is_refused_with_its_reason(w):
    (w.old_stages / f"shorts_ost_audit_{DATE}.json").unlink()
    assert "old run" in _reasons(w.run())[1]


def test_same_segment_within_the_tolerance_is_swapped(w):
    w.row(1)
    _audit(w.stages, [_entry(1, 0.4, 21.5), _entry(2, 0.0, 22.2)])  # encode rounding only
    assert [r.short_index for r in w.run().rebuilt] == [1, 2]


def test_an_attestation_for_another_day_does_not_cover_this_short(w):
    w.row(1)
    _attest(w.stages, "2026-03-09")
    result = w.run(privacy_fn=REAL)
    assert result.rebuilt == [] and DATE in _reasons(result)[1]


@pytest.mark.parametrize("state", ["inconclusive", "failed"])
def test_an_unverified_attestation_for_the_day_is_refused(w, state):
    w.row(1)
    _attest(w.stages, DATE, state)
    assert w.run(privacy_fn=REAL).rebuilt == []


def test_swapped_row_uploads_the_new_bytes_with_no_hand_step(w):
    w.row(1)
    _attest(w.stages)
    assert 1 in [r.short_index for r in w.run(privacy_fn=REAL).rebuilt]
    calls, _ = _drain(w, LATER)
    assert len(calls) == 1 and calls[0].read_bytes() == b"new-bytes-1"


def test_refused_short_never_uploads(w):
    w.row(1)
    _attest(w.stages)
    w.run(privacy_fn=REAL, mask_fn=_no)
    calls, res = _drain(w, LATER)
    assert calls == [] and res.succeeded == 0 and res.errors


def test_an_unplaceable_attestation_refuses_the_short(w, monkeypatch):
    from gomoveshift.pipeline.drive_phases import phase_d

    def boom(**_k):
        raise OSError("read-only")

    monkeypatch.setattr(phase_d, "_sync_privacy_attestations", boom)
    a = w.row(1)
    old = w.hash_of(a)
    result = w.run()
    assert result.rebuilt == [] and "attestation" in _reasons(result)[1]
    assert w.hash_of(a) == old
    assert load_publish_plan(w.sidecar).shorts[0].media.sha256 == w.records[0].media.sha256


def test_an_uploaded_sibling_refuses_the_short_for_good_with_an_incident(w):
    yt = w.row(1)
    w.row(1, platform="instagram", platform_upload_id="u1", status="uploaded")
    old = w.hash_of(yt)
    result = w.run()
    assert "half old" in _reasons(result)[1] and 1 not in [d.short_index for d in result.deferred]
    assert w.hash_of(yt) == old
    assert "half old" in json.dumps(_open(w, 1))


def test_an_in_flight_row_defers(w):
    w.row(1, status="uploading")
    assert w.run().deferred[0].short_index == 1


def test_a_live_lease_defers_visibly_and_the_next_run_swaps(w):
    w.row(1, lease_owner="x", lease_expires_at=(NOW + timedelta(minutes=5)).isoformat())
    result = w.run()
    assert [d.short_index for d in result.deferred] == [1] and 1 not in _reasons(result)
    assert "retried" in json.dumps(_open(w, 1))
    again = w.run(now=NOW + timedelta(minutes=10))
    assert 1 in [r.short_index for r in again.rebuilt] and not _open(w, 1)


def test_a_future_slot_inside_the_lead_time_defers_with_its_retry_time_then_moves(w):
    slot = NOW + timedelta(hours=2)
    a = w.row(1, scheduled_at=slot.isoformat())
    old = w.hash_of(a)
    d = [x for x in w.run().deferred if x.short_index == 1][0]
    assert slot.isoformat() in d.reason and w.hash_of(a) == old and _open(w, 1)
    later = w.run(now=NOW + timedelta(days=2))  # the slot passed, the row is still queued
    assert 1 in [r.short_index for r in later.rebuilt] and w.hash_of(a) != old


@pytest.mark.parametrize("slot", [None, (NOW - timedelta(hours=3)).isoformat()])
def test_a_missing_or_past_slot_is_due_and_moves(w, slot):
    a = w.row(1, scheduled_at=slot)
    old = w.hash_of(a)
    result = w.run()
    assert 1 in [r.short_index for r in result.rebuilt] and w.hash_of(a) != old
    assert not any("lead time" in x.reason for x in result.refused + result.deferred)


def _state(w, **fields):
    (w.stages / "pipeline_state.json").write_text(json.dumps(fields))


def _confirmed_fields(out):
    """What a human Gate C/D confirmation records over ``out`` as it is (full and swap hash)."""
    state = PipelineState.from_dict({"gate_cd_confirmed": True})
    record_gate_cd_confirmation(state, out)
    return {
        "gate_cd_confirmed": True,
        "gate_cd_confirmation_hash": state.gate_cd_confirmation_hash,
        "gate_cd_confirmation_swap_hash": state.gate_cd_confirmation_swap_hash,
    }


def test_a_confirmed_gate_whose_content_changed_refuses_every_short_once(w):
    a = w.row(1)
    old = w.hash_of(a)
    _state(w, gate_cd_confirmed=True, gate_cd_confirmation_hash="0" * 64)
    result = w.run()
    assert result.rebuilt == [] and set(_reasons(result)) == {1, 2}
    assert all("re-confirmed" in r for r in _reasons(result).values())
    assert w.hash_of(a) == old
    inc = _open(w)
    assert len(inc) == 1 and "Gate C/D must be re-confirmed" in json.dumps(inc)
    assert load_publish_plan(w.sidecar).shorts[0].media.sha256 == w.records[0].media.sha256
    _state(w)  # the gate clears: the next passing run resolves the incident
    w.run()
    assert _open(w) == []


def test_an_unreadable_state_refuses(w):
    (w.stages / "pipeline_state.json").unlink()
    assert w.run().rebuilt == []


def test_the_gate_check_is_called_with_the_run_state_and_the_plan_dir(w, monkeypatch):
    seen = []

    def spy(state, output_dir, **_k):
        seen.append(output_dir)
        raise enqueue_plan.GateNotApprovedError("spy refuses")

    monkeypatch.setattr(enqueue_plan, "assert_gate_cd_approved", spy)
    result = w.run()
    # Exactly two checks on a refusing gate: the plan-skipping full hash, then the projected hash
    # the proven swap may use; one more or fewer means a hash path was added or lost.
    assert seen == [w.out, w.out]
    assert result.rebuilt == [] and "spy refuses" in json.dumps(_open(w))


def test_an_incident_write_failure_is_logged_at_error_level(w, tmp_path, caplog):
    w.row(1)
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    caplog.set_level(logging.DEBUG)
    assert w.run(mask_fn=_no, incidents_dir=blocker).refused
    assert [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_an_open_refusal_incident_is_not_rewritten_or_repushed(w):
    w.row(1)
    w.run(mask_fn=_no)
    for p in w.incidents.glob("*.json"):
        p.write_text(json.dumps(json.loads(p.read_text()) | {"pushed": True}))
    before = {p.name: p.read_bytes() for p in w.incidents.glob("*.json")}
    w.run(mask_fn=_no)
    assert {p.name: p.read_bytes() for p in w.incidents.glob("*.json")} == before


def test_rows_match_by_resolved_path(w):
    did = w.row(1, sidecar_path=f"{w.out}/./{w.sidecar.name}")
    result = w.run()
    assert 1 in [r.short_index for r in result.rebuilt]
    assert w.hash_of(did) == load_publish_plan(w.sidecar).shorts[0].reviewed_plan_hash
    assert w.conn.execute("SELECT COUNT(*) FROM response_log").fetchone()[0] == 1


def test_a_row_on_another_sidecar_file_refuses_the_short(w, tmp_path):
    did = w.row(1, sidecar_path=str(tmp_path / "elsewhere" / "plan.json"))
    old = w.hash_of(did)
    result = w.run()
    assert 1 in _reasons(result) and "sidecar" in _reasons(result)[1]
    assert w.hash_of(did) == old and _open(w, 1)


def test_crash_then_a_refusing_rerun_restores_the_rows_to_the_files_hash(w, monkeypatch):
    import os

    did = w.row(1)
    old = w.hash_of(did)

    def boom(*_a, **_k):
        raise OSError("disk gone")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        w.run()
    assert w.hash_of(did) != old  # committed before the crashed replace
    monkeypatch.undo()
    result = w.run(mask_fn=_no)
    assert result.refused[0].delivery_ids == (did,)
    assert load_publish_plan(w.sidecar).shorts[0].reviewed_plan_hash == old
    assert w.hash_of(did) == old
    q = "SELECT COUNT(*) FROM response_log WHERE action='replace_rebuild_restore'"
    assert w.conn.execute(q).fetchone()[0] == 1


def test_a_malformed_audit_entry_refuses_that_short_and_the_run_continues(w):
    w.row(1)
    _audit(w.stages, [{"rank": 1, "parent_lf_index": 1, "start_sec": 0.0}, _entry(2, 0.0)])
    result = w.run()
    assert "malformed" in _reasons(result)[1]
    assert [r.short_index for r in result.rebuilt] == [2] and _open(w, 1)


def test_a_recovery_row_refuses_the_short(w):
    w.row(1, platform="instagram", recovers_delivery_id=w.row(1))
    assert "recovery row" in _reasons(w.run())[1]


def _race(w, monkeypatch, sql, *args):
    real = replace_rebuild._write

    def racing(conn, *a, **k):
        conn.execute(sql, args)
        conn.commit()
        return real(conn, *a, **k)

    monkeypatch.setattr(replace_rebuild, "_write", racing)


@pytest.mark.parametrize(
    "sql,args",
    [
        ("UPDATE deliveries SET status='awaiting_manual_publish' WHERE id=?", ()),
        ("UPDATE deliveries SET status='published' WHERE id=?", ()),
        ("UPDATE deliveries SET platform_upload_id='u9' WHERE id=?", ()),
        ("UPDATE deliveries SET lease_owner='z', lease_expires_at=? WHERE id=?", ("L",)),
    ],
)
def test_a_row_that_changes_mid_run_rolls_the_whole_write_back(w, monkeypatch, sql, args):
    a, b = w.row(1), w.row(2)
    lease = (NOW + timedelta(minutes=5)).isoformat()
    args = tuple(lease if x == "L" else x for x in args) + (b,)
    _race(w, monkeypatch, sql, *args)
    before, ha = w.sidecar.read_bytes(), w.hash_of(a)
    with pytest.raises(LeaseRaceError):
        w.run()
    assert w.hash_of(a) == ha and w.sidecar.read_bytes() == before


def test_the_refused_incident_resolves_when_a_later_run_swaps(w):
    w.row(1)
    w.run(mask_fn=_no)
    assert _open(w, 1)
    w.run()
    assert _open(w) == []


def test_consent_hash_moves_only_when_it_equalled_the_old_hash(tmp_path):
    world = World(tmp_path, n=2)
    raw = json.loads(world.old_sidecar.read_text())
    raw["shorts"][1]["consent_hash"] = raw["shorts"][1]["reviewed_plan_hash"]
    world.old_sidecar.write_text(json.dumps(raw))
    world.run()
    one, two = load_publish_plan(world.sidecar).shorts
    assert one.consent_hash == "c" and two.consent_hash == two.reviewed_plan_hash
    world.conn.close()


def test_the_rebuilt_file_must_parse_before_it_replaces_anything(w, monkeypatch):
    import gomoveshift.publishing.plan_io as plan_io

    a = w.row(1)
    before, old = w.sidecar.read_bytes(), w.hash_of(a)

    def bad(_p):
        raise ValueError("does not parse")

    monkeypatch.setattr(plan_io, "load_publish_plan", bad)
    with pytest.raises(ValueError):
        w.run()
    assert w.sidecar.read_bytes() == before and w.hash_of(a) == old
    assert not list(w.out.glob("*.rebuild-tmp"))


def test_awaiting_manual_rows_are_listed_and_never_changed(w):
    a = w.row(1, status="awaiting_manual_publish")
    old = w.hash_of(a)
    result = w.run()
    assert [u.delivery_id for u in result.untouched] == [a] and w.hash_of(a) == old
    assert "awaiting manual" in result.untouched[0].why


def test_missing_rerendered_media_is_refused(w):
    (w.out / "shorts" / "lf_01" / "short_01.mp4").unlink()
    assert "missing" in _reasons(w.run())[1]


def test_each_move_writes_an_audit_row(w):
    a = w.row(1)
    old = w.hash_of(a)
    w.run()
    rows = w.conn.execute("SELECT reason FROM response_log WHERE action='replace_rebuild'")
    docs = [json.loads(r[0]) for r in rows]
    assert [d["delivery_id"] for d in docs] == [a] and docs[0]["old_plan_hash"] == old
    assert docs[0]["new_plan_hash"] == w.hash_of(a)


def test_the_stall_incident_opens_from_a_real_drain_once_and_closes_on_upload(tmp_path):
    conn = open_queue_db(tmp_path / "q")
    inc = tmp_path / "inc"
    did = enqueue(conn, PROJECT, 1, "youtube", "h", scheduled_at=None)
    conn.commit()
    t = datetime(2026, 11, 22, tzinfo=timezone.utc)
    fail = {"on": True}

    def handler(row):
        if fail["on"]:
            raise TransientMediaError("sidecar not readable")

    def drain():
        dispatch.drain_queue(conn, {"youtube": handler}, now=t.isoformat(), incidents_dir=inc)

    for _ in range(dispatch.MEDIA_STALL_ATTEMPTS):
        drain()
        t += timedelta(hours=2)
    (path,) = list(inc.glob("*.json"))
    path.write_text(json.dumps(json.loads(path.read_text()) | {"pushed": True}))
    before = path.read_bytes()
    drain()  # a later stalled drain does not rewrite or re-push the open incident
    assert path.read_bytes() == before
    fail["on"] = False
    t += timedelta(hours=2)
    drain()
    row = conn.execute("SELECT status FROM deliveries WHERE id=?", (did,)).fetchone()
    assert row[0] == "published" and all(r.get("resolved_at") for r in load_all(inc)[0])
    conn.close()
