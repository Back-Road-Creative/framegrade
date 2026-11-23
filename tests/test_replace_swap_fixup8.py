"""Fixup 8: a record hold closes only on a positive pass -- the loader actually returned the Short's
record (or parsed the project's plan with no duplicate). A loader failure of any other kind leaves
the incident open; the resolve never aborts a drain; an incident whose rows left the queue closes."""

from __future__ import annotations

import json
import logging
import os
from datetime import timedelta
from pathlib import Path

import pytest
from test_replace_rebuild import World
from test_replace_swap_fixup import LATER, REAL, _attest, _drain
from test_replace_swap_fixup5 import _dup_world, _undated_world

from gomoveshift.cli.handlers.dispatch_queue import PrivacyAttestationLoader, SidecarRecordLoader
from gomoveshift.health.store import safe_name
from gomoveshift.publishing import dispatch
from gomoveshift.publishing.queue import enqueue, open_queue_db

KIND = "delivery_record_held"


@pytest.fixture
def w(tmp_path):
    world = World(tmp_path)
    yield world
    world.conn.close()


def _row(conn, did):
    return conn.execute("SELECT * FROM deliveries WHERE id=?", (did,)).fetchone()


def _held(w):
    return [i for i in w.incidents_open() if i.get("kind") == KIND]


def _all_held(w):
    docs = [json.loads(p.read_text()) for p in sorted(w.incidents.glob("*.json"))]
    return [d for d in docs if d.get("kind") == KIND]


def _loader_sidecar(w):
    row = w.conn.execute(
        "SELECT sidecar_path FROM deliveries WHERE sidecar_path IS NOT NULL ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return Path(row["sidecar_path"])


def _mark_pushed(w):
    first = _held(w)[0]
    path = w.incidents / first["file"]
    doc = json.loads(path.read_text())
    doc["pushed"] = True
    path.write_text(json.dumps(doc))
    return first


def _drain_with(w, at, handler):
    return dispatch.drain_queue(
        w.conn,
        {"youtube": handler},
        now=at.isoformat(),
        privacy_attestations=PrivacyAttestationLoader(w.out),
        incidents_dir=w.incidents,
    )


# ---- MAJOR-1: a loader failure that is not a pass never closes a hold in force ----------------


def test_unreadable_sidecar_keeps_a_still_undated_shorts_incident_open_and_unpushed_again(w):
    _undated_world(w)
    did = w.row(1)
    w.run(privacy_fn=REAL)
    _drain(w, LATER)
    first = _mark_pushed(w)
    side = _loader_sidecar(w)
    os.chmod(side, 0)
    try:
        _, res = _drain(w, LATER + timedelta(hours=1))
    finally:
        os.chmod(side, 0o644)
    assert _row(w.conn, did)["status"] == "queued"
    still = _held(w)
    assert len(still) == 1, res.errors
    assert still[0]["pushed"] is True and still[0]["detected_at"] == first["detected_at"]
    _, res2 = _drain(w, LATER + timedelta(days=2))
    assert res2.held_record == 1, res2
    only = _held(w)
    assert len(only) == 1 and only[0]["pushed"] is True  # no second push for one unchanged hold
    assert only[0]["detected_at"] == first["detected_at"]


def test_missing_sidecar_keeps_a_still_duplicated_projects_incident_open(w):
    _attest(w.stages)
    _dup_world(w, identical_second_render=True)
    w.row(1)
    w.row(2)
    w.run(privacy_fn=REAL)
    _drain(w, LATER)
    assert len(_held(w)) == 1
    first = _mark_pushed(w)
    side = _loader_sidecar(w)
    hidden = side.with_suffix(".hidden")
    side.rename(hidden)
    try:
        _, res = _drain(w, LATER + timedelta(hours=1))
    finally:
        hidden.rename(side)
    assert len(_held(w)) == 1, res.errors
    _, res2 = _drain(w, LATER + timedelta(days=2))
    assert res2.held_record == 2
    only = _held(w)
    assert len(only) == 1 and only[0]["pushed"] is True
    assert only[0]["detected_at"] == first["detected_at"]


def test_an_error_raised_after_the_loader_returned_the_record_still_counts_as_a_pass(w):
    raw = _undated_world(w)
    did = w.row(1)
    w.run(privacy_fn=REAL)
    _drain(w, LATER)
    assert len(_held(w)) == 1
    raw["shorts"][0]["recording_date"] = w.records[0].recording_date
    w.old_sidecar.write_text(json.dumps(raw))
    w.run(privacy_fn=REAL, now=LATER)
    loader = SidecarRecordLoader(w.out, conn=w.conn)

    def handler(row):
        loader(row["project_name"], row["short_index"])  # the record loaded ...
        raise RuntimeError("upload blew up")  # ... then the handler failed

    _drain_with(w, LATER + timedelta(hours=3), handler)
    assert _row(w.conn, did)["status"] == "queued"
    assert _held(w) == []
    assert _all_held(w)[0]["resolved_at"] == (LATER + timedelta(hours=3)).isoformat()


def test_a_handler_error_that_never_reached_a_record_closes_nothing(w):
    _undated_world(w)
    w.row(1)
    w.run(privacy_fn=REAL)
    _drain(w, LATER)
    assert len(_held(w)) == 1

    def handler(row):
        raise RuntimeError("no record was ever loaded")

    _drain_with(w, LATER + timedelta(hours=1), handler)
    assert len(_held(w)) == 1


def test_a_normal_return_without_a_loaded_record_closes_nothing(w, monkeypatch):
    # the row publishes (so the sweep would close it); isolate the per-row resolve
    monkeypatch.setattr(dispatch, "sync_record_held_incidents", lambda *a, **k: None)
    _undated_world(w)
    w.row(1)
    w.run(privacy_fn=REAL)
    _drain(w, LATER)
    assert len(_held(w)) == 1
    _drain_with(w, LATER + timedelta(hours=1), lambda row: None)
    assert len(_held(w)) == 1


def test_hold_on_index_1_is_not_resolved_by_index_2_passing(w):
    _undated_world(w)
    d1, d2 = w.row(1), w.row(2)
    w.run(privacy_fn=REAL)
    _drain(w, LATER)
    _drain(w, LATER + timedelta(hours=1))
    assert _row(w.conn, d2)["status"] == "published"
    assert _row(w.conn, d1)["status"] == "queued"
    held = _held(w)
    assert len(held) == 1 and held[0]["short_index"] == 1


def test_duplicate_project_hold_every_index_holds_so_no_false_resolve(w):
    _attest(w.stages)
    _dup_world(w, identical_second_render=True)
    w.row(1)
    w.row(2)
    w.run(privacy_fn=REAL)
    for h in (0, 1, 2):
        _, res = _drain(w, LATER + timedelta(hours=h))
        assert res.held_record == 2 and res.attempted == 0
    held = _held(w)
    assert len(held) == 1 and held[0]["key"] == f"{KIND}:{held[0]['project']}"


def test_a_resolve_keeps_pushed_and_detected_at(w):
    raw = _undated_world(w)
    w.row(1)
    w.run(privacy_fn=REAL)
    _drain(w, LATER)
    first = _mark_pushed(w)
    raw["shorts"][0]["recording_date"] = w.records[0].recording_date
    w.old_sidecar.write_text(json.dumps(raw))
    w.run(privacy_fn=REAL, now=LATER)
    _drain(w, LATER + timedelta(hours=3))
    doc = _all_held(w)[0]
    assert doc["resolved_at"] == (LATER + timedelta(hours=3)).isoformat()
    assert doc["pushed"] is True and doc["detected_at"] == first["detected_at"]


def test_the_incident_resolve_reads_the_file_health_reads(w):
    from gomoveshift.health.store import load_all

    _undated_world(w)
    w.row(1)
    w.run(privacy_fn=REAL)
    _drain(w, LATER)
    rows, bad = load_all(w.incidents)
    held = [r for r in rows if r["kind"] == KIND]
    assert bad == 0 and len(held) == 1
    assert held[0]["file"] == f"{safe_name(held[0]['key'])}.json" and held[0]["key"].endswith(":1")
    assert "every platform held" in held[0]["detail"] and "short 1" in held[0]["detail"]
    assert "youtube#" not in held[0]["detail"]


# ---- MINOR-2: the resolve never aborts a drain after the irreversible publish -----------------


@pytest.mark.parametrize("body", ["[]", "null", '{"kind": "delivery_record_held"}'])
def test_a_malformed_incident_file_never_aborts_the_drain_after_a_publish(tmp_path, body, caplog):
    conn = open_queue_db(tmp_path / "q")
    did = enqueue(conn, "p", 1, "youtube", "H")
    conn.commit()
    inc = tmp_path / "i"
    inc.mkdir()
    name = f"{safe_name(f'{KIND}:p:1')}.json"
    (inc / name).write_text(body)

    def handler(row):
        dispatch.note_record_pass(row["project_name"], row["short_index"])

    with caplog.at_level(logging.WARNING, logger=dispatch.logger.name):
        res = dispatch.drain_queue(
            conn, {"youtube": handler}, now="2030-01-01T00:00:00+00:00", incidents_dir=inc
        )
    assert _row(conn, did)["status"] == "published" and res.succeeded == 1
    assert any(f"{KIND}:p:1" in r.getMessage() for r in caplog.records if r.levelname == "WARNING")
    conn.close()


def test_the_resolve_runs_after_the_published_transition(tmp_path, monkeypatch):
    conn = open_queue_db(tmp_path / "q")
    did = enqueue(conn, "p", 1, "youtube", "H")
    conn.commit()
    seen = []
    monkeypatch.setattr(
        dispatch,
        "_resolve_record_held_incidents",
        lambda row, now, inc, *a, **kw: seen.append(_row(conn, row["id"])["status"]),
    )

    def handler(row):
        dispatch.note_record_pass(row["project_name"], row["short_index"])

    dispatch.drain_queue(
        conn, {"youtube": handler}, now="2030-01-01T00:00:00+00:00", incidents_dir=tmp_path
    )
    assert seen == ["published"] and _row(conn, did)["status"] == "published"
    conn.close()


# ---- MINOR-3: an incident whose rows stop being selected closes -------------------------------


def test_a_held_shorts_cancelled_rows_close_its_incident_on_the_next_pass(w):
    _undated_world(w)
    did = w.row(1)
    w.run(privacy_fn=REAL)
    _drain(w, LATER)
    assert len(_held(w)) == 1
    first = _mark_pushed(w)
    w.conn.execute("UPDATE deliveries SET status='cancelled' WHERE id=?", (did,))
    w.conn.commit()
    _drain(w, LATER + timedelta(hours=1))
    assert _held(w) == []
    doc = _all_held(w)[0]
    assert doc["resolved_at"] == (LATER + timedelta(hours=1)).isoformat()
    assert doc["pushed"] is True and doc["detected_at"] == first["detected_at"]


def test_a_duplicated_projects_incident_closes_when_every_row_is_terminal(w):
    _attest(w.stages)
    _dup_world(w, identical_second_render=True)
    d1, d2 = w.row(1), w.row(2)
    w.run(privacy_fn=REAL)
    _drain(w, LATER)
    assert len(_held(w)) == 1
    w.conn.execute("UPDATE deliveries SET status='cancelled' WHERE id=?", (d1,))
    w.conn.commit()
    _drain(w, LATER + timedelta(hours=1))
    assert len(_held(w)) == 1  # the other row of the project is still held
    w.conn.execute("UPDATE deliveries SET status='cancelled' WHERE id=?", (d2,))
    w.conn.commit()
    _drain(w, LATER + timedelta(hours=2))
    assert _held(w) == []


def test_the_sweep_leaves_an_incident_open_while_a_row_of_the_short_is_live(w):
    _undated_world(w)
    w.row(1)
    w.run(privacy_fn=REAL)
    for h in (0, 1, 2):
        _drain(w, LATER + timedelta(hours=h))
    assert len(_held(w)) == 1
