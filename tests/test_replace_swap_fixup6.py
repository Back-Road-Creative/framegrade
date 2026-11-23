"""Fixup 6: a held duplicate-index or undated Short is HELD, never charged an attempt, and the hold
opens an incident. Real queue, sidecars and ``drain_queue``; only the probe and the mask gate are
injected."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from test_replace_rebuild import World
from test_replace_swap_fixup import LATER, REAL, _attest, _drain
from test_replace_swap_fixup5 import _dup_world, _row, _undated_world, _undup

from gomoveshift.cli.handlers.dispatch_queue import PrivacyAttestationLoader, SidecarRecordLoader
from gomoveshift.publishing import dispatch

RECORD_HELD_KIND = "delivery_record_held"


@pytest.fixture
def w(tmp_path):
    world = World(tmp_path)
    yield world
    world.conn.close()


def _blip_drain(w, at):
    """One drain whose handler loads the record (as every adapter does), then hits an ordinary
    retryable platform error."""
    loader = SidecarRecordLoader(w.out, conn=w.conn)

    def handler(row):
        loader(row["project_name"], row["short_index"])
        raise ConnectionError("network blip")

    return dispatch.drain_queue(
        w.conn,
        {"youtube": handler},
        now=at.isoformat(),
        privacy_attestations=PrivacyAttestationLoader(w.out),
        incidents_dir=w.incidents,
    )


def _hold_many(w, n=10):
    for k in range(n):
        _drain(w, LATER + timedelta(hours=2 * k))
    return LATER + timedelta(hours=2 * n + 2)


def _fix_date(w, raw):
    raw["shorts"][0]["recording_date"] = w.records[0].recording_date
    w.old_sidecar.write_text(json.dumps(raw))


def _held(w):
    return [i for i in w.incidents_open() if i.get("kind") == RECORD_HELD_KIND]


def _assert_untouched(w, did, before):
    row = _row(w, did)
    assert row["attempt_count"] == before["attempt_count"]
    assert row["status"] == before["status"]
    assert row["lease_owner"] is None
    assert row["error_class"] == "RecordHeldError"
    assert row["error_detail"]


def test_a_duplicate_index_hold_counts_no_attempt_and_stays_visible(w):
    _attest(w.stages)
    _dup_world(w, identical_second_render=True)
    did = w.row(1)
    w.run(privacy_fn=REAL)
    before = dict(_row(w, did))
    _hold_many(w, 10)
    _assert_untouched(w, did, before)
    assert "more than once" in _row(w, did)["error_detail"]


def test_an_undated_hold_counts_no_attempt_and_stays_visible(w):
    _undated_world(w)
    did = w.row(1)
    w.run(privacy_fn=REAL)
    before = dict(_row(w, did))
    _hold_many(w, 10)
    _assert_untouched(w, did, before)
    assert "recording_date" in _row(w, did)["error_detail"]


def test_a_released_duplicate_hold_keeps_the_whole_retry_budget(w):
    _attest(w.stages)
    raw = _dup_world(w, identical_second_render=True)
    did = w.row(1)
    w.run(privacy_fn=REAL)
    t = _hold_many(w, 10)
    _undup(w, raw)
    r = w.run(privacy_fn=REAL, now=t)
    assert 1 in [x.short_index for x in r.rebuilt]
    _blip_drain(w, t + timedelta(hours=2))
    row = _row(w, did)
    assert row["status"] == "queued" and row["attempt_count"] == 1


def test_a_released_undated_hold_keeps_the_whole_retry_budget(w):
    raw = _undated_world(w)
    did = w.row(1)
    w.run(privacy_fn=REAL)
    t = _hold_many(w, 10)
    _fix_date(w, raw)
    r = w.run(privacy_fn=REAL, now=t)
    assert 1 in [x.short_index for x in r.rebuilt]
    _blip_drain(w, t + timedelta(hours=2))
    row = _row(w, did)
    assert row["status"] == "queued" and row["attempt_count"] == 1


def test_an_ordinary_failure_still_burns_after_eight_attempts(w):
    _attest(w.stages)
    did = w.row(1)
    w.run(privacy_fn=REAL)
    for k in range(8):
        _blip_drain(w, LATER + timedelta(hours=3 * k))
    assert _row(w, did)["status"] == "failed_permanent"


def test_a_duplicate_hold_opens_one_project_incident_for_every_held_row(w):
    _attest(w.stages)
    _dup_world(w, identical_second_render=False)
    w.row(1)
    w.row(2)
    w.run(privacy_fn=REAL)
    _hold_many(w, 3)
    found = _held(w)
    assert len(found) == 1
    detail = found[0]["detail"]
    assert found[0]["project"] and found[0]["project"] in detail
    assert "every row" in detail and "more than once" in detail


def test_an_undated_hold_opens_one_incident_naming_the_short(w):
    _undated_world(w)
    did = w.row(1)
    w.run(privacy_fn=REAL)
    _hold_many(w, 3)
    found = _held(w)
    assert len(found) == 1
    assert found[0]["delivery_id"] == did and found[0]["short_index"] == 1
    assert "recording_date" in found[0]["detail"]
