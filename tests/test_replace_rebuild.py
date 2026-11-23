"""``rebuild_replaced_sidecar``: a replace re-render rebuilds the shorts plan onto the new media.

Real sqlite queue and real sidecar files under ``tmp_path``; the media probe and the two
gates are injected (ffprobe is unavailable here) except where a test names the real gate.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from gomoveshift.health.store import load_all
from gomoveshift.pipeline.drive_phases.phase_d import _serialize_publish_plan
from gomoveshift.publishing import dispatch, replace_rebuild
from gomoveshift.publishing.asset_plan import review_hash
from gomoveshift.publishing.media_fingerprint import verify_media
from gomoveshift.publishing.models import (
    MediaFingerprint,
    PlatformPayloads,
    PublishPlan,
    RouteSelection,
    ScheduleWindow,
    ShortPublishRecord,
    YouTubePayload,
)
from gomoveshift.publishing.plan_io import load_publish_plan
from gomoveshift.publishing.queue import enqueue, open_queue_db
from gomoveshift.publishing.replace_rebuild import rebuild_replaced_sidecar

PROJECT = "nctopa-2026-03-05-rec0063"
DATE = "2026-03-05"
NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
FAR = (NOW + timedelta(days=3)).isoformat()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fp(path, data, duration=21.5):
    return MediaFingerprint(str(path), len(data), duration, 1080, 1920, "h264", _sha(data))


def _probe(path):
    data = path.read_bytes()
    return _fp(path, data)


def _ok(*_a, **_k):
    return None


def _no(*_a, **_k):
    return "refused by test gate"


class World:
    """An old (set-aside) sidecar + a new run's media at the SAME output paths."""

    def __init__(self, tmp_path, n=2):
        self.out = tmp_path / "out"
        self.old_dir = tmp_path / "_superseded" / "p-1"
        self.stages = tmp_path / "stages"
        self.old_stages = tmp_path / "old_stages"  # the set-aside run's stage dir
        self.incidents = tmp_path / "incidents"
        for d in (self.out, self.old_dir, self.stages, self.old_stages):
            d.mkdir(parents=True)
        self.sidecar = self.out / f"{PROJECT}_publish_plan.json"
        self.old_sidecar = self.old_dir / f"{PROJECT}_publish_plan.json"
        shorts, entries = [], []
        for i in range(1, n + 1):
            media = self.out / "shorts" / "lf_01" / f"short_{i:02d}.mp4"
            media.parent.mkdir(parents=True, exist_ok=True)
            old = f"old-bytes-{i}".encode()
            title, desc, tags = f"t{i}", f"d{i}", ("a",)
            shorts.append(
                ShortPublishRecord(
                    project_name=PROJECT,
                    lf_index=1,
                    short_index=i,
                    media=_fp(media, old),
                    schedule=ScheduleWindow(NOW + timedelta(days=3), "America/Chicago"),
                    recording_date=DATE,
                    sampled_regions=("driftless",),
                    parent_lf_video_id="oLd",
                    parent_lf_publish_at=NOW,
                    payloads=PlatformPayloads(youtube=YouTubePayload(title, desc, tags, "2")),
                    consent_hash="c",
                    reviewed_plan_hash=review_hash(title, desc, tags, _sha(old)),
                    routes=RouteSelection(selected=("youtube",)),
                )
            )
            media.write_bytes(f"new-bytes-{i}".encode())  # the re-render, same path
            entries.append(
                {
                    "rank": i,
                    "parent_lf_index": 1,
                    "start_sec": 0.0,
                    "end_sec": 21.5,
                    "route_record": None,
                }
            )
        self.old_sidecar.write_text(_serialize_publish_plan(PublishPlan(shorts=tuple(shorts))))
        self.sidecar.write_text("{}")  # the new run emitted none: a replace plan skips it
        for stages in (self.stages, self.old_stages):  # same footage in both runs
            (stages / f"shorts_ost_audit_{DATE}.json").write_text(json.dumps({"shorts": entries}))
        (self.stages / "pipeline_state.json").write_text("{}")  # an unreviewed run (Gate C/D)
        self.conn = open_queue_db(tmp_path / "q")
        self.records = shorts

    def row(self, i, platform="youtube", slot=FAR, **cols):
        did = enqueue(
            self.conn,
            PROJECT,
            i,
            platform,
            self.records[i - 1].reviewed_plan_hash,
            sidecar_path=str(self.sidecar),
            scheduled_at=slot,
        )
        for col, val in cols.items():
            self.conn.execute(f"UPDATE deliveries SET {col}=? WHERE id=?", (val, did))  # noqa: S608
        self.conn.commit()
        return did

    def hash_of(self, did):
        return self.conn.execute("SELECT plan_hash FROM deliveries WHERE id=?", (did,)).fetchone()[
            0
        ]

    def run(self, **over):
        kw = dict(
            old_sidecar=self.old_sidecar,
            new_sidecar=self.sidecar,
            project=PROJECT,
            stages_dir=self.stages,
            old_stages_dir=self.old_stages,
            now=NOW,
            incidents_dir=self.incidents,
            fingerprint_fn=_probe,
            privacy_fn=_ok,
            mask_fn=_ok,
        )
        kw.update(over)
        return rebuild_replaced_sidecar(self.conn, **kw)

    def incidents_open(self):
        rows, _ = load_all(self.incidents)
        return [r for r in rows if not r.get("resolved_at")]


@pytest.fixture
def w(tmp_path):
    world = World(tmp_path)
    yield world
    world.conn.close()


def test_rebuild_swaps_fingerprint_moves_hash_and_dispatch_verifies_new_bytes(w):
    a, b = w.row(1), w.row(2)
    with pytest.raises(Exception):  # the old plan on the new bytes: today's stall
        verify_media(load_publish_plan(w.old_sidecar).shorts[0])
    result = w.run()
    plan = load_publish_plan(w.sidecar)
    assert [r.short_index for r in result.rebuilt] == [1, 2]
    for rec in plan.shorts:
        verify_media(rec)  # the dispatcher's own check passes on the re-render
        assert rec.parent_lf_video_id == "oLd"  # relink rewrites it later
        assert rec.reviewed_plan_hash == review_hash(
            rec.payloads.youtube.title,
            rec.payloads.youtube.description,
            rec.payloads.youtube.tags,
            rec.media.sha256,
        )
    assert w.hash_of(a) == plan.shorts[0].reviewed_plan_hash
    assert w.hash_of(b) == plan.shorts[1].reviewed_plan_hash
    assert result.refused == [] and w.incidents_open() == []


def test_idempotent_second_run_changes_nothing(w):
    a = w.row(1)
    w.run()
    first = w.sidecar.read_bytes()
    again = w.run()
    assert w.sidecar.read_bytes() == first
    assert (
        again.refused == []
        and w.hash_of(a) == load_publish_plan(w.sidecar).shorts[0].reviewed_plan_hash
    )


@pytest.mark.parametrize("gate", ["privacy_fn", "mask_fn"])
def test_gate_refusal_holds_rows_keeps_old_fingerprint_and_opens_incident(w, gate):
    a = w.row(1)
    old_hash = w.hash_of(a)
    result = w.run(**{gate: _no})
    assert [r.short_index for r in result.refused] == [1, 2]
    assert result.rebuilt == []
    rec = load_publish_plan(w.sidecar).shorts[0]
    assert (
        rec.media.sha256 == w.records[0].media.sha256
    )  # old fingerprint: never old bytes silently
    assert w.hash_of(a) == old_hash
    kinds = {i["kind"] for i in w.incidents_open()}
    assert replace_rebuild.INCIDENT_KIND in kinds
    assert "refused by test gate" in json.dumps(w.incidents_open())


def test_identity_mismatch_refused(w):
    w.row(1)
    audit = w.stages / f"shorts_ost_audit_{DATE}.json"
    audit.write_text(
        json.dumps({"shorts": [{"rank": 1, "parent_lf_index": 9, "start_sec": 0, "end_sec": 21.5}]})
    )
    result = w.run()
    reasons = {r.short_index: r.reason for r in result.refused}
    assert "parent LF" in reasons[1]
    assert 2 in reasons  # no audit entry for rank 2: cannot confirm identity
    assert w.incidents_open()


def test_duration_drift_refused(w):
    w.row(1)
    result = w.run(fingerprint_fn=lambda p: _fp(p, p.read_bytes(), duration=40.0))
    assert len(result.refused) == 2 and "duration" in result.refused[0].reason


def test_real_privacy_gate_refuses_unattested_run(w):
    w.row(1)
    result = w.run(privacy_fn=replace_rebuild.privacy_refusal)  # no attestation on disk
    assert len(result.refused) == 2
    assert "privacy" in result.refused[0].reason


def test_non_derived_review_hash_refused_visibly(tmp_path):
    world = World(tmp_path, n=1)
    raw = json.loads(world.old_sidecar.read_text())
    raw["shorts"][0]["reviewed_plan_hash"] = "hand-set-hash"
    world.old_sidecar.write_text(json.dumps(raw))
    world.records[0] = load_publish_plan(world.old_sidecar).shorts[0]
    world.row(1)
    result = world.run()
    assert "re-review" in result.refused[0].reason
    assert world.incidents_open()
    world.conn.close()


def test_untouched_rows_listed_and_never_changed(w):
    leased = w.row(1, lease_owner="x", lease_expires_at=(NOW + timedelta(minutes=5)).isoformat())
    published = w.row(2, status="published")
    uploaded = w.row(2, platform="instagram", platform_upload_id="u1")
    before = {d: w.hash_of(d) for d in (leased, published, uploaded)}
    result = w.run()
    assert {u.delivery_id for u in result.untouched} == {leased, published, uploaded}
    assert {d: w.hash_of(d) for d in before} == before


def test_lead_time_defers_the_row_visibly(w):
    near = w.row(1, scheduled_at=(NOW + timedelta(hours=2)).isoformat())
    old_hash = w.hash_of(near)
    result = w.run()
    assert w.hash_of(near) == old_hash
    assert any("lead time" in d.reason for d in result.deferred)
    assert not any("lead time" in r.reason for r in result.refused)
    assert w.incidents_open()


def test_crash_mid_write_leaves_old_file(w, monkeypatch):
    w.row(1)
    before = w.sidecar.read_bytes()

    def boom(*_a, **_k):
        raise OSError("disk gone")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        w.run()
    assert w.sidecar.read_bytes() == before
    assert not list(w.sidecar.parent.glob("*.rebuild-tmp"))
    monkeypatch.undo()
    w.run()  # a rerun converges: sidecar and queue agree again
    assert w.hash_of(1) == load_publish_plan(w.sidecar).shorts[0].reviewed_plan_hash


def test_stall_incident_after_backoff_ceiling_then_closes(tmp_path):
    conn = open_queue_db(tmp_path / "q")
    inc = tmp_path / "inc"
    did = enqueue(conn, PROJECT, 1, "youtube", "h", scheduled_at=FAR)
    bound = dispatch.MEDIA_STALL_ATTEMPTS
    # the first attempt whose backoff delay is already the cap (60 s doubling to 3600 s)
    assert bound == 7
    conn.execute(
        "UPDATE deliveries SET error_class='TransientMediaError', attempt_count=? WHERE id=?",
        (bound - 1, did),
    )
    conn.commit()
    dispatch.sync_media_stall_incidents(conn, NOW.isoformat(), inc)
    assert load_all(inc)[0] == []
    conn.execute("UPDATE deliveries SET attempt_count=? WHERE id=?", (bound, did))
    conn.commit()
    dispatch.sync_media_stall_incidents(conn, NOW.isoformat(), inc)
    rows = [r for r in load_all(inc)[0] if not r.get("resolved_at")]
    assert [r["kind"] for r in rows] == [dispatch.MEDIA_STALL_KIND]
    assert (
        conn.execute("SELECT status FROM deliveries WHERE id=?", (did,)).fetchone()[0] == "queued"
    )
    conn.execute("UPDATE deliveries SET error_class=NULL, status='uploaded' WHERE id=?", (did,))
    conn.commit()
    dispatch.sync_media_stall_incidents(conn, NOW.isoformat(), inc)
    assert all(r.get("resolved_at") for r in load_all(inc)[0])
    conn.close()
