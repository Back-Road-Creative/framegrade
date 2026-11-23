"""Fixup 7: a record hold's incident closes when the hold clears, is keyed per Short (not per
platform row), a later hold files a fresh pushed-able incident, and the cron line counts it."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from test_dispatch_queue_summary_last_line import _run
from test_replace_rebuild import World
from test_replace_swap_fixup import LATER, REAL, _attest, _drain
from test_replace_swap_fixup5 import _dup_world, _undated_world, _undup

from gomoveshift.cli.handlers.dispatch_queue import PrivacyAttestationLoader, SidecarRecordLoader
from gomoveshift.publishing import dispatch
from gomoveshift.publishing.dispatch import DrainResult
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
    return [
        json.loads(p.read_text())
        for p in sorted(w.incidents.glob("*.json"))
        if json.loads(p.read_text()).get("kind") == KIND
    ]


def test_undated_hold_incident_resolves_when_the_short_publishes(w):
    raw = _undated_world(w)
    did = w.row(1)
    w.run(privacy_fn=REAL)
    _drain(w, LATER)
    assert len(_held(w)) == 1
    raw["shorts"][0]["recording_date"] = w.records[0].recording_date
    w.old_sidecar.write_text(json.dumps(raw))
    w.run(privacy_fn=REAL, now=LATER)
    calls, _ = _drain(w, LATER + timedelta(hours=3))
    assert any(c.name == "short_01.mp4" for c in calls)
    assert _row(w.conn, did)["status"] == "published"
    assert _held(w) == []
    assert _all_held(w)[0]["resolved_at"] is not None


def test_duplicate_hold_incident_resolves_once_the_rows_publish(w):
    _attest(w.stages)
    raw = _dup_world(w, identical_second_render=True)
    d1, d2 = w.row(1), w.row(2)
    w.run(privacy_fn=REAL)
    _drain(w, LATER)
    assert len(_held(w)) == 1
    _undup(w, raw)
    w.run(privacy_fn=REAL, now=LATER)
    _drain(w, LATER + timedelta(hours=3))
    _drain(w, LATER + timedelta(hours=6))
    assert {_row(w.conn, d)["status"] for d in (d1, d2)} == {"published"}
    assert _held(w) == []


def test_a_new_duplicate_after_a_resolve_files_a_fresh_unpushed_incident(w):
    _attest(w.stages)
    raw = _dup_world(w, identical_second_render=True)
    w.row(1)
    w.row(2)
    w.run(privacy_fn=REAL)
    _drain(w, LATER)
    first = _held(w)[0]
    path = w.incidents / first["file"]
    doc = json.loads(path.read_text())
    doc["pushed"] = True  # health check already pushed it
    path.write_text(json.dumps(doc))
    _undup(w, raw)
    w.run(privacy_fn=REAL, now=LATER)
    _drain(w, LATER + timedelta(hours=3))
    _drain(w, LATER + timedelta(hours=6))
    assert _held(w) == []
    raw2 = json.loads(w.sidecar.read_text())
    dup = dict(raw2["shorts"][0])
    dup["short_index"] = 2
    raw2["shorts"].append(dup)
    w.sidecar.write_text(json.dumps(raw2))
    w.conn.execute(
        "UPDATE deliveries SET status='queued', lease_owner=NULL, lease_expires_at=NULL, "
        "next_attempt_at=NULL WHERE status='published'"
    )
    w.conn.commit()
    _, res = _drain(w, LATER + timedelta(hours=9))
    now_open = _held(w)
    assert len(now_open) == 1, res
    assert "[2]" in now_open[0]["detail"]
    assert now_open[0]["pushed"] is False
    assert now_open[0]["resolved_at"] is None


def test_repeated_holds_keep_one_incident_unchanged(w):
    _undated_world(w)
    w.row(1)
    w.run(privacy_fn=REAL)
    _, res = _drain(w, LATER)
    assert res.attempted == 0 and res.held_record == 1
    first = _held(w)[0]
    _drain(w, LATER + timedelta(hours=1))
    _drain(w, LATER + timedelta(hours=2))
    again = _held(w)
    assert len(again) == 1
    assert again[0]["detected_at"] == first["detected_at"]
    assert again[0]["pushed"] is False


def test_one_undated_short_on_two_platforms_opens_one_incident(w):
    _undated_world(w)
    w.row(1, platform="youtube")
    w.row(1, platform="instagram")
    w.run(privacy_fn=REAL)
    loader = SidecarRecordLoader(w.out, conn=w.conn)

    def handler(row):
        loader(row["project_name"], row["short_index"])

    dispatch.drain_queue(
        w.conn,
        {"youtube": handler, "instagram": handler},
        now=LATER.isoformat(),
        privacy_attestations=PrivacyAttestationLoader(w.out),
        incidents_dir=w.incidents,
    )
    assert len(_held(w)) == 1


def test_cron_summary_line_counts_held_record(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("GMS_TIKTOK_MODE", "inbox")
    monkeypatch.setenv("GMS_TIKTOK_PACK_DIR", str(tmp_path / "pack"))
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "_console"))
    lines = _run(tmp_path, capsys, DrainResult(held_record=5, errors=["youtube#1: record hold: x"]))
    assert "held_record=5" in lines[-1]


def test_cron_summary_line_omits_held_record_when_zero(tmp_path, capsys):
    lines = _run(tmp_path, capsys, DrainResult(attempted=1, succeeded=1))
    assert "held_record" not in lines[-1]


def test_a_pure_hold_pass_counts_no_attempt(tmp_path):
    conn = open_queue_db(tmp_path / "q")
    enqueue(conn, "p", 1, "youtube", "H")
    conn.commit()

    def handler(row):
        raise dispatch.RecordHeldError("held")

    res = dispatch.drain_queue(
        conn, {"youtube": handler}, now="2030-01-01T00:00:00+00:00", incidents_dir=tmp_path / "i"
    )
    assert res.attempted == 0 and res.held_record == 1
