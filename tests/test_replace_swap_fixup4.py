"""Fixup 4: the review gate covers media and consent again outside a proven replace swap, a Short
refused for a missing day keeps its rows held, and a finished swap is bound to the file on disk.
Real queue, sidecars and ``drain_queue``; only the probe and the mask gate are injected."""

from __future__ import annotations

import dataclasses
import json
import os
from datetime import timedelta
from unittest.mock import patch

import pytest
from test_replace_rebuild import NOW, World, _probe
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
from test_replace_swap_fixup2 import D2, LEASE, _two_days

from gomoveshift.cli.handlers.enqueue_plan import GateNotApprovedError, assert_gate_cd_approved
from gomoveshift.pipeline.drive_state_model import PipelineState
from gomoveshift.pipeline.review_gate import (
    ReviewFailedError,
    ReviewGate,
    record_gate_cd_confirmation,
)
from gomoveshift.publishing.asset_plan import review_hash
from gomoveshift.publishing.plan_io import load_publish_plan


@pytest.fixture
def w(tmp_path):
    world = World(tmp_path)
    yield world
    world.conn.close()


def _rebuilt(r):
    return [x.short_index for x in r.rebuilt]


def _status(w, did):
    return w.conn.execute("SELECT status FROM deliveries WHERE id=?", (did,)).fetchone()[0]


# ---- MAJOR-1: only the replace gate accepts the swap-invariant hash ----

SHORT = {
    "short_index": 1,
    "media": {"path": "/out/shorts/lf_01/short_01.mp4", "sha256": "a" * 64, "size_bytes": 10},
    "reviewed_plan_hash": review_hash("t", "d", ["a"], "a" * 64),
    "consent_hash": "consent-v1",
    "payloads": {"youtube": {"title": "t", "description": "d", "tags": ["a"]}},
}
SHA2 = "b" * 64
EDITS = {
    "repointed_media": {
        "media": {"path": "/out/shorts/lf_02/short_07.mp4", "sha256": SHA2, "size_bytes": 99},
        "reviewed_plan_hash": review_hash("t", "d", ["a"], SHA2),
    },
    "rerendered_bytes": {
        "media": {**SHORT["media"], "sha256": SHA2},
        "reviewed_plan_hash": review_hash("t", "d", ["a"], SHA2),
    },
    "changed_consent": {"consent_hash": "consent-v2-someone-else"},
    "title": {"payloads": {"youtube": {"title": "T2", "description": "d", "tags": ["a"]}}},
    "platforms": {"routes": {"selected": ["youtube", "tiktok"]}},
    "schedule": {"schedule": {"timezone": "UTC"}},
}


def _confirmed_then_edited(w, edit):
    plan = w.out / "other_publish_plan.json"
    plan.write_text(json.dumps({"shorts": [SHORT]}))
    state = PipelineState.from_dict({"gate_cd_confirmed": True})
    record_gate_cd_confirmation(state, w.out)
    plan.write_text(json.dumps({"shorts": [{**SHORT, **edit}]}))
    return state


@pytest.mark.parametrize("edit", sorted(EDITS))
def test_ordinary_enqueue_plan_refuses_every_edit_after_the_confirmation(w, edit):
    state = _confirmed_then_edited(w, EDITS[edit])
    with pytest.raises(GateNotApprovedError):
        assert_gate_cd_approved(state, w.out)


@pytest.mark.parametrize("edit", sorted(EDITS))
def test_the_auto_pass_prompts_again_after_every_edit(w, edit, monkeypatch):
    monkeypatch.setattr(ReviewGate, "_stream_count", lambda self, p: 2)
    monkeypatch.setattr(ReviewGate, "_av_drift_pct", lambda self, p: 0.5)
    monkeypatch.setattr(ReviewGate, "_probe_av_durations", lambda self, p: (600.0, 600.0))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    state = _confirmed_then_edited(w, EDITS[edit])
    with patch("builtins.input", return_value="n") as ask:
        with pytest.raises(ReviewFailedError):
            ReviewGate(interactive=True).gate_cd(state=state, output_dir=w.out, date_key="d")
    assert ask.called


def test_the_projected_hash_is_accepted_only_when_the_caller_names_a_proven_swap(w):
    plan = w.out / "other_publish_plan.json"
    plan.write_text(json.dumps({"shorts": [SHORT]}))
    state = PipelineState.from_dict({"gate_cd_confirmed": True})
    record_gate_cd_confirmation(state, w.out)
    assert state.gate_cd_confirmation_hash != state.gate_cd_confirmation_swap_hash
    swapped = {**SHORT, **EDITS["rerendered_bytes"]}
    plan.write_text(json.dumps({"shorts": [swapped]}))
    assert_gate_cd_approved(state, w.out, swap_plan="other_publish_plan.json")
    with pytest.raises(GateNotApprovedError):
        assert_gate_cd_approved(state, w.out)
    plan.write_text(json.dumps({"shorts": [{**swapped, **EDITS["title"]}]}))
    with pytest.raises(GateNotApprovedError):
        assert_gate_cd_approved(state, w.out, swap_plan="other_publish_plan.json")


def test_the_replace_gate_refuses_an_unproven_plan_even_under_a_confirmation(w):
    w.row(1)
    assert 1 in _rebuilt(w.run())
    _state(w, **_confirmed_fields(w.out))
    raw = json.loads(w.sidecar.read_text())
    raw["shorts"][0]["media"]["path"] = str(w.out / "shorts" / "lf_01" / "elsewhere.mp4")
    w.sidecar.write_text(json.dumps(raw))
    r = w.run(now=NOW + timedelta(minutes=10))
    assert r.rebuilt == [] and 1 in _reasons(r)
    assert "re-confirmed" in json.dumps(_open(w))  # the run-level Gate C/D incident


def test_a_confirmation_over_the_swapped_plan_survives_the_next_rerun(w):
    w.row(1)
    w.row(2, lease_owner="x", lease_expires_at=LEASE())
    assert _rebuilt(w.run()) == [1]
    _state(w, **_confirmed_fields(w.out))  # taken with the sidecar present
    r = w.run(now=NOW + timedelta(minutes=10))
    assert _rebuilt(r) == [2] and not r.refused and _open(w) == []


# ---- MAJOR-2: a refused Short's rows are held, never burned ----


def _drop_day(w, restore=False):
    raw = json.loads(w.old_sidecar.read_text())
    if restore:
        raw["shorts"][0]["recording_date"] = "2026-03-05"
    else:
        del raw["shorts"][0]["recording_date"]
    w.old_sidecar.write_text(json.dumps(raw))


def test_a_refused_shorts_row_is_held_through_twelve_drains_and_the_incident_stays_open(w):
    _attest(w.stages)
    _drop_day(w)
    did = w.row(1)
    w.row(2)
    r = w.run(privacy_fn=REAL)
    assert "no recording_date" in _reasons(r)[1] and _open(w, 1)
    at = LATER
    for _ in range(12):
        _drain(w, at)
        at += timedelta(hours=2)
    assert _status(w, did) not in ("failed_permanent", "cancelled")
    assert _open(w, 1)
    load_publish_plan(w.sidecar)  # the written plan keeps the Short's index and parses


def test_a_fixed_rerun_moves_the_held_short_and_resolves_its_incident(w):
    _attest(w.stages)
    _drop_day(w)
    did = w.row(1)
    w.row(2)
    old = w.hash_of(did)
    w.run(privacy_fn=REAL)
    assert _open(w, 1)
    _drop_day(w, restore=True)
    r = w.run(privacy_fn=REAL)
    assert 1 in _rebuilt(r) and w.hash_of(did) != old and _open(w, 1) == []


def test_a_short_whose_rows_already_burned_is_not_settled_while_its_day_is_missing(w):
    _drop_day(w)
    did = w.row(1, status="failed_permanent")
    r = w.run()
    assert "no recording_date" in _reasons(r)[1] and _open(w, 1)
    _drop_day(w, restore=True)
    w.run()
    assert _open(w, 1) == [] and _status(w, did) == "failed_permanent"


def test_a_duplicate_short_index_never_resolves_the_incident_it_just_raised(w):
    raw = json.loads(w.old_sidecar.read_text())
    dup = dict(raw["shorts"][1])
    dup["short_index"] = 1
    dup.pop("recording_date")
    raw["shorts"].append(dup)
    w.old_sidecar.write_text(json.dumps(raw))
    w.row(1)
    r = w.run()
    # fixup 5: a duplicated index is refused at every position, never rebuilt from a guess
    assert 1 not in _rebuilt(r) and 1 in _reasons(r) and _open(w, 1)


# ---- MINOR-1: a finished swap is bound to the file on disk ----


def _forge(w, sha):
    raw = json.loads(w.sidecar.read_text())
    short = raw["shorts"][0]
    yt = short["payloads"]["youtube"]
    short["media"]["sha256"] = sha
    short["reviewed_plan_hash"] = review_hash(yt["title"], yt["description"], yt["tags"], sha)
    w.sidecar.write_text(json.dumps(raw))


def _media_path(w):
    return json.loads(w.sidecar.read_text())["shorts"][0]["media"]["path"]


def _third_render(w):
    with open(_media_path(w), "wb") as fh:
        fh.write(b"third-render-bytes")


def _bigger(path):
    return dataclasses.replace(_probe(path), size_bytes=_probe(path).size_bytes + 1)


def test_a_self_consistent_forged_sha_on_a_settled_short_is_refused_with_an_incident(w):
    w.row(1)
    assert 1 in _rebuilt(w.run())
    _forge(w, "e" * 64)
    r = w.run()
    assert "changed again" in _reasons(r)[1] and _open(w, 1)


def test_a_re_render_after_a_settled_swap_is_flagged_with_no_movable_row(w):
    w.row(1)
    assert 1 in _rebuilt(w.run())
    _third_render(w)
    r = w.run()
    assert "changed again" in _reasons(r)[1] and _open(w, 1)


def test_a_re_render_after_a_swap_is_refused_while_a_straggler_can_still_move(w):
    yt = w.row(1)
    assert 1 in _rebuilt(w.run())
    w.row(1, "instagram")
    _third_render(w)
    r = w.run()
    assert "changed again" in _reasons(r)[1]
    assert w.hash_of(yt) != w.records[0].reviewed_plan_hash  # still on the finished swap


def test_a_finished_swap_whose_file_is_gone_still_settles(w):
    w.row(1)
    assert 1 in _rebuilt(w.run())
    os.unlink(_media_path(w))
    r = w.run()
    assert 1 not in _reasons(r) and _open(w, 1) == []


def test_a_carried_record_is_kept_byte_for_byte_when_a_straggler_moves(w):
    w.row(1)
    assert 1 in _rebuilt(w.run())
    carried = json.loads(w.sidecar.read_text())["shorts"][0]
    straggler = w.row(1, "instagram")
    r = w.run(fingerprint_fn=_bigger)  # a fresh probe says one byte more
    assert 1 in _rebuilt(r) and w.hash_of(straggler) == carried["reviewed_plan_hash"]
    assert json.loads(w.sidecar.read_text())["shorts"][0] == carried


def test_a_swap_with_an_in_flight_sibling_is_deferred_never_settled(w):
    w.row(1)
    assert 1 in _rebuilt(w.run())
    w.row(1, "instagram", lease_owner="x", lease_expires_at=LEASE())
    r = w.run()
    assert [d.short_index for d in r.deferred] == [1] and _open(w, 1)


# ---- null-path rows follow the newest recorded sidecar ----


def test_a_null_path_row_follows_the_newest_recorded_sidecar_even_when_a_sidecar_dir_is_named(w):
    w.row(1, "instagram", status="cancelled")
    did = w.row(1, sidecar_path=None)
    w.row(2, "instagram", status="cancelled", sidecar_path="/elsewhere/other_publish_plan.json")
    r = w.run(sidecar_dir=w.out)
    assert 1 in _reasons(r) and w.hash_of(did) == w.records[0].reviewed_plan_hash


# ---- MINOR-2: a verified attestation beside the plan outlives an unverified stage copy ----


def _two_day_run(w):
    _two_days(w)
    _attest(w.stages, "2026-03-05")
    _attest(w.stages, D2)
    w.row(1, lease_owner="x", lease_expires_at=LEASE())
    w.row(2)
    assert _rebuilt(w.run(privacy_fn=REAL)) == [2]
    return w.out / f"day_{D2}_privacy_attestation.json"


def _states(path):
    return {c["state"] for c in json.loads(path.read_text())["classes"].values()}


def test_an_unverified_stage_copy_never_replaces_a_verified_copy_beside_the_plan(w, caplog):
    beside = _two_day_run(w)
    assert _states(beside) == {"verified"}
    _attest(w.stages, D2, state="inconclusive")
    r = w.run(now=NOW + timedelta(minutes=10), privacy_fn=REAL)
    assert _rebuilt(r) == [1] and r.attestations_kept == [D2]
    assert _states(beside) == {"verified"}
    assert any(D2 in m and "kept" in m for m in caplog.messages)


def test_a_verified_newer_stage_copy_still_replaces_the_copy_beside_the_plan(w):
    beside = _two_day_run(w)
    stage = w.stages / f"day_{D2}" / beside.name
    doc = json.loads(stage.read_text())
    doc["note"] = "fresh"
    stage.write_text(json.dumps(doc))
    r = w.run(now=NOW + timedelta(minutes=10), privacy_fn=REAL)
    assert _rebuilt(r) == [1] and r.attestations_kept == []
    assert json.loads(beside.read_text()).get("note") == "fresh"
