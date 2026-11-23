"""The replace swap keeps a finished swap and a confirmed Gate C/D across reruns, reports an
upload another day holds as deferred, and resolves what no longer needs it. Real queue, sidecars
and ``drain_queue``; only the probe and mask gate are injected."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import timedelta

import pytest
from test_replace_rebuild import DATE, FAR, NOW, PROJECT, World, _no
from test_replace_swap_fixup import (
    LATER,
    REAL,
    _attest,
    _confirmed_fields,
    _drain,
    _entry,
    _open,
    _reasons,
    _state,
)

from gomoveshift.health.store import load_all
from gomoveshift.publishing import dispatch
from gomoveshift.publishing.dispatch import PRIVACY_REQUIRED_CLASSES
from gomoveshift.publishing.plan_io import PublishPlanFormatError, load_publish_plan
from gomoveshift.publishing.queue import enqueue, open_queue_db

D2 = "2026-03-06"
LEASE = lambda: (NOW + timedelta(minutes=5)).isoformat()  # noqa: E731


@pytest.fixture
def w(tmp_path):
    world = World(tmp_path)
    yield world
    world.conn.close()


def _rebuilt(result):
    return [r.short_index for r in result.rebuilt]


def _two_days(w):
    """Short 2 is recorded on D2 (its own audit in both runs); Short 1 stays on DATE."""
    raw = json.loads(w.old_sidecar.read_text())
    raw["shorts"][1]["recording_date"] = D2
    w.old_sidecar.write_text(json.dumps(raw))
    for st in (w.stages, w.old_stages):
        (st / f"shorts_ost_audit_{D2}.json").write_text(json.dumps({"shorts": [_entry(2, 0.0)]}))


def _crash(w, monkeypatch):
    """A run whose rows moved (committed) but whose ``os.replace`` never happened."""
    monkeypatch.setattr(os, "replace", lambda *_a, **_k: (_ for _ in ()).throw(OSError("gone")))
    with pytest.raises(OSError):
        w.run()
    monkeypatch.undo()


def _confirm(w):
    """A human confirmed Gate C/D over the content as it is; a replace run wrote no plan yet."""
    w.sidecar.unlink()
    _state(w, **_confirmed_fields(w.out))


# ---- the restore undoes only an unfinished move ----


def test_a_finished_swap_survives_a_later_refusal(w):
    a = w.row(1)
    assert 1 in _rebuilt(w.run())
    new = w.hash_of(a)
    again = w.run(mask_fn=_no)  # a rerun now refuses (the mask gate): the swap stays
    assert 1 not in _reasons(again) and w.hash_of(a) == new
    assert load_publish_plan(w.sidecar).shorts[0].reviewed_plan_hash == new
    (w.old_stages / f"shorts_ost_audit_{DATE}.json").unlink()  # the set-aside run is reclaimed
    w.run()
    assert w.hash_of(a) == new
    assert load_publish_plan(w.sidecar).shorts[0].media.sha256 != w.records[0].media.sha256


def test_a_deferred_sibling_moves_on_the_next_run_after_the_first_published(w):
    a, b = w.row(1), w.row(2, lease_owner="x", lease_expires_at=LEASE())
    r1 = w.run()
    assert _rebuilt(r1) == [1] and [d.short_index for d in r1.deferred] == [2]
    w.conn.execute(
        "UPDATE deliveries SET status='published', platform_upload_id='y' WHERE id=?", (a,)
    )
    w.conn.commit()
    new_a, old_b = w.hash_of(a), w.hash_of(b)
    r2 = w.run(now=NOW + timedelta(minutes=10))  # no raise: Short 2's lease ran out
    assert 2 in _rebuilt(r2) and w.hash_of(b) != old_b and w.hash_of(a) == new_a


def test_a_crash_leftover_that_is_no_longer_queued_is_listed_never_restored(w, monkeypatch):
    did = w.row(1)
    _crash(w, monkeypatch)
    moved = w.hash_of(did)
    w.conn.execute(
        "UPDATE deliveries SET status='published', platform_upload_id='y' WHERE id=?", (did,)
    )
    w.conn.commit()
    result = w.run(mask_fn=_no)  # must not raise
    assert w.hash_of(did) == moved and did in {u.delivery_id for u in result.untouched}
    assert str(did) in _reasons(result)[1]


def test_the_restore_only_follows_audit_rows_of_its_own_old_hash(w, monkeypatch):
    did = w.row(1)
    _crash(w, monkeypatch)
    moved = w.hash_of(did)
    w.conn.execute("UPDATE response_log SET reason = REPLACE(reason, 'old_plan_hash', 'x_hash')")
    w.conn.commit()
    w.run(mask_fn=_no)
    assert w.hash_of(did) == moved


# ---- a confirmed Gate C/D stays approved across reruns ----


def test_a_confirmed_gate_whose_hash_still_matches_swaps(w):
    a = w.row(1)
    _confirm(w)
    old = w.hash_of(a)
    assert _rebuilt(w.run()) == [1, 2] and w.hash_of(a) != old


def test_a_confirmed_gate_survives_the_swaps_own_write_and_the_sibling_swaps_next(w):
    a = w.row(1)
    b = w.row(2, lease_owner="x", lease_expires_at=LEASE())
    _confirm(w)
    r1 = w.run()
    assert _rebuilt(r1) == [1] and not r1.refused
    new_a, old_b = w.hash_of(a), w.hash_of(b)
    r2 = w.run(now=NOW + timedelta(minutes=10))
    assert 2 in _rebuilt(r2) and not r2.refused and w.hash_of(b) != old_b
    assert w.hash_of(a) == new_a and _open(w) == []


@pytest.mark.parametrize("change", ["lf_json", "another_plan"])
def test_a_real_content_change_between_runs_still_refuses(w, change):
    name = "lf_youtube.json" if change == "lf_json" else "other_publish_plan.json"
    (w.out / name).write_text(json.dumps({"title": "a"}))
    a = w.row(1)
    w.row(2, lease_owner="x", lease_expires_at=LEASE())
    _confirm(w)
    assert 1 in _rebuilt(w.run())
    (w.out / name).write_text(json.dumps({"title": "edited after the review"}))
    r2 = w.run(now=NOW + timedelta(minutes=10))
    assert r2.rebuilt == [] and all("re-confirmed" in r for r in _reasons(r2).values())
    assert w.hash_of(a) == load_publish_plan(w.sidecar).shorts[0].reviewed_plan_hash  # kept


def test_a_cleared_gate_incident_resolves_on_the_next_passing_run(w):
    w.row(1)
    _state(w, gate_cd_confirmed=True, gate_cd_confirmation_hash="0" * 64)
    w.run()
    assert [i for i in _open(w) if i.get("short_index") is None]
    _state(w)
    w.run()
    assert _open(w) == []


def test_the_swap_changes_only_media_hash_and_consent(w):
    w.row(1)
    w.run()
    old = json.loads(w.old_sidecar.read_text())["shorts"][0]
    new = json.loads(w.sidecar.read_text())["shorts"][0]
    assert {k for k in old if old[k] != new[k]} <= {"media", "reviewed_plan_hash", "consent_hash"}


# ---- restored and mutant-killing tests ----


def test_a_verified_attestation_for_the_day_passes(w):
    w.row(1)
    _attest(w.stages, DATE)
    assert _rebuilt(w.run(privacy_fn=REAL)) == [1, 2]


def test_privacy_is_evaluated_per_day(w):
    _two_days(w)
    w.row(1)
    days = []

    def privacy(_stages, day):
        days.append(day)
        return "day not attested" if day == D2 else None

    result = w.run(privacy_fn=privacy)
    assert "day not attested" in _reasons(result)[2] and set(days) == {DATE, D2}
    (held,) = result.deferred  # Short 1's own day passes; the other day holds the run in dispatch
    assert held.short_index == 1 and D2 in held.reason and result.rebuilt == []


def test_rows_match_by_resolved_path_through_a_parent_segment(w):
    did = w.row(1, sidecar_path=f"{w.out}/../{w.out.name}/{w.sidecar.name}")
    assert 1 in _rebuilt(w.run()) and w.hash_of(did) != w.records[0].reviewed_plan_hash


def test_an_attestation_naming_another_day_is_refused(w):
    w.row(1)
    _attest(w.stages, DATE)
    p = w.stages / f"day_{DATE}" / f"day_{DATE}_privacy_attestation.json"
    p.write_text(json.dumps(json.loads(p.read_text()) | {"date": D2}))
    assert w.run(privacy_fn=REAL).rebuilt == []


def test_the_deferral_carries_the_exact_retry_time(w):
    slot = (NOW + timedelta(hours=2)).isoformat()
    w.row(1, scheduled_at=slot)
    (d,) = [x for x in w.run().deferred if x.short_index == 1]
    assert d.retry_at == slot
    assert f"earliest retry {slot}" in json.dumps(_open(w))


# ---- placement and the result stay honest ----


def test_a_verified_placed_copy_is_never_overwritten_by_an_inconclusive_one(w):
    _two_days(w)
    w.row(1)
    w.row(2)
    p1 = w.stages / f"day_{DATE}" / f"day_{DATE}_privacy_attestation.json"
    _attest(w.stages, DATE)
    _attest(w.stages, D2)
    assert _rebuilt(w.run(privacy_fn=REAL)) == [1, 2]
    p1.unlink()  # a cache cleaner drops the stage copy
    w.run(privacy_fn=REAL)
    beside = json.loads((w.out / f"day_{DATE}_privacy_attestation.json").read_text())
    assert beside["classes"][PRIVACY_REQUIRED_CLASSES[0]]["state"] == "verified"


def test_a_short_held_by_another_days_missing_attestation_is_deferred_not_rebuilt(w):
    _two_days(w)
    a = w.row(1)
    w.row(2)
    _attest(w.stages, DATE)
    result = w.run(privacy_fn=REAL)
    assert 1 not in _rebuilt(result) and 2 in _reasons(result)
    (d,) = [x for x in result.deferred if x.short_index == 1]
    assert D2 in d.reason and _open(w, 1)
    calls, _ = _drain(w, LATER)
    assert calls == []  # the drain does hold it
    _attest(w.stages, D2)  # the missing day is attested: the next run clears the hold
    again = w.run(privacy_fn=REAL, now=NOW + timedelta(minutes=1))
    assert again.deferred == [] and _open(w) == []
    calls, _ = _drain(w, LATER)
    assert (
        len(calls) >= 1
        and w.hash_of(a) == load_publish_plan(w.sidecar).shorts[0].reviewed_plan_hash
    )


# ---- a published or uploaded sibling refuses the Short for good ----


def test_a_published_sibling_with_an_upload_id_refuses_the_short(w):
    yt = w.row(1)
    w.row(1, platform="instagram", platform_upload_id="ig1", status="published")
    old = w.hash_of(yt)
    assert "half old" in _reasons(w.run())[1] and w.hash_of(yt) == old


def test_an_uploaded_sibling_stays_refused_after_it_publishes(w):
    yt = w.row(1)
    ig = w.row(1, platform="instagram", platform_upload_id="ig1", status="uploaded")
    assert "half old" in _reasons(w.run())[1]
    w.conn.execute("UPDATE deliveries SET status='published' WHERE id=?", (ig,))
    w.conn.commit()
    old = w.hash_of(yt)
    assert "half old" in _reasons(w.run())[1] and w.hash_of(yt) == old


# ---- identity inputs and legacy rows ----


def test_a_short_with_a_null_recording_date_is_refused_never_a_key_error(w):
    raw = json.loads(w.old_sidecar.read_text())
    raw["shorts"][0]["recording_date"] = None
    w.old_sidecar.write_text(json.dumps(raw))
    w.row(1)
    assert "no recording_date" in _reasons(w.run())[1]


def test_a_short_missing_its_recording_date_key_is_refused_with_its_rows_held(w):
    raw = json.loads(w.old_sidecar.read_text())
    del raw["shorts"][0]["recording_date"]
    w.old_sidecar.write_text(json.dumps(raw))
    did = w.row(1)
    r = w.run()  # refused, not raised; the old record is written back so the plan keeps its index
    assert "no recording_date" in _reasons(r)[1] and _open(w, 1)
    assert w.hash_of(did) == w.records[0].reviewed_plan_hash
    for k in range(12):  # dispatch holds the stale bytes; the row is never burned by this cause
        _drain(w, LATER + timedelta(minutes=k))
    status = w.conn.execute("SELECT status FROM deliveries WHERE id=?", (did,)).fetchone()[0]
    assert status in ("queued", "awaiting_manual") and _open(w, 1)


def test_a_legacy_row_without_a_sidecar_path_is_served_from_the_plan_dispatch_reads(w):
    did = w.row(1, sidecar_path=None)
    result = w.run(sidecar_dir=w.out)
    assert 1 in _rebuilt(result) and w.hash_of(did) != w.records[0].reviewed_plan_hash


def test_a_missing_old_audit_names_the_path_it_expected(w):
    audit = w.old_stages / f"shorts_ost_audit_{DATE}.json"
    audit.unlink()
    reason = _reasons(w.run())[1]
    assert str(audit) in reason and "_superseded" in reason


# ---- an incident resolves when nothing is left to swap; a stall sync failure is visible ----


@pytest.mark.parametrize("status", ["cancelled", "published"])
def test_a_refusal_incident_resolves_once_the_only_row_is_finished(w, status):
    did = w.row(1)
    w.run(mask_fn=_no)
    assert _open(w, 1)
    w.conn.execute(
        "UPDATE deliveries SET status=?, platform_upload_id='y' WHERE id=?", (status, did)
    )
    w.conn.commit()
    w.run(mask_fn=_no)
    assert _open(w, 1) == [] and w.hash_of(did) == w.records[0].reviewed_plan_hash


def test_a_stall_sync_sqlite_error_is_logged_and_reported_not_raised(tmp_path, caplog):
    conn = open_queue_db(tmp_path / "q")

    class Locked:
        def execute(self, sql, *a):
            raise sqlite3.OperationalError("database is locked")

    caplog.set_level(logging.ERROR)
    msg = dispatch.sync_media_stall_incidents(Locked(), NOW.isoformat(), tmp_path / "inc")
    assert "database is locked" in msg and [r for r in caplog.records if r.levelno >= 40]
    conn.close()


def test_the_drain_reports_a_failed_stall_sync(tmp_path, monkeypatch):
    conn = open_queue_db(tmp_path / "q")
    monkeypatch.setattr(
        dispatch, "sync_media_stall_incidents", lambda *_a, **_k: "stall sync failed: x"
    )
    res = dispatch.drain_queue(conn, {}, now=NOW.isoformat())
    assert "stall sync failed: x" in res.errors
    conn.close()


# ---- the stall threshold follows the drain's backoff; the incident closes only when it is done ----


def _stalled_row(conn, attempts, **cols):
    did = enqueue(conn, PROJECT, 1, "youtube", "h", scheduled_at=FAR)
    cols = {"error_class": "TransientMediaError", "attempt_count": attempts} | cols
    for col, val in cols.items():
        conn.execute(f"UPDATE deliveries SET {col}=? WHERE id=?", (val, did))  # noqa: S608
    conn.commit()
    return did


def _open_stalls(inc):
    return [r for r in load_all(inc)[0] if not r.get("resolved_at")]


def test_the_stall_threshold_follows_the_backoff_arguments(tmp_path):
    conn = open_queue_db(tmp_path / "q")
    inc = tmp_path / "inc"
    _stalled_row(conn, 3)  # 10 s doubling to a 40 s cap reaches its ceiling at attempt 3
    assert dispatch.sync_media_stall_incidents(conn, NOW.isoformat(), inc) is None
    assert _open_stalls(inc) == []  # the default backoff (ceiling at 7) has not reached it
    dispatch.sync_media_stall_incidents(
        conn, NOW.isoformat(), inc, backoff_base_seconds=10, backoff_cap_seconds=40
    )
    assert len(_open_stalls(inc)) == 1
    conn.close()


def test_the_drain_passes_its_backoff_arguments_to_the_stall_sync(tmp_path, monkeypatch):
    conn = open_queue_db(tmp_path / "q")
    seen = {}

    def spy(_conn, _now, _dir=None, **kw):
        seen.update(kw)

    monkeypatch.setattr(dispatch, "sync_media_stall_incidents", spy)
    dispatch.drain_queue(
        conn, {}, now=NOW.isoformat(), backoff_base_seconds=10, backoff_cap_seconds=40
    )
    assert seen == {"backoff_base_seconds": 10, "backoff_cap_seconds": 40}
    conn.close()


def test_a_stall_incident_stays_open_while_the_row_is_still_waiting(tmp_path):
    conn = open_queue_db(tmp_path / "q")
    inc = tmp_path / "inc"
    did = _stalled_row(conn, dispatch.MEDIA_STALL_ATTEMPTS)
    dispatch.sync_media_stall_incidents(conn, NOW.isoformat(), inc)
    assert len(_open_stalls(inc)) == 1
    conn.execute("UPDATE deliveries SET error_class='RetryableError' WHERE id=?", (did,))
    conn.commit()  # another error replaced the media error: the row has still not gone out
    dispatch.sync_media_stall_incidents(conn, NOW.isoformat(), inc)
    assert len(_open_stalls(inc)) == 1
    conn.execute(
        "UPDATE deliveries SET status='uploaded', platform_upload_id='u' WHERE id=?", (did,)
    )
    conn.commit()
    dispatch.sync_media_stall_incidents(conn, NOW.isoformat(), inc)
    assert _open_stalls(inc) == []
    conn.close()
