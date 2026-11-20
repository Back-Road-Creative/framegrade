"""A queued Short the final-text review refuses is repaired before its slot.

Dispatch runs ``short_review_refusal``; a refusal burns an attempt and ends
``failed_permanent``. ``repair_queued_shorts`` (called by ``dispatch-queue``) applies the
existing tag filter and route re-measure to a scratch copy and replaces the sidecar only when
the review then passes. The health rule ``queued_short_refused`` reads the same review
read-only. Synthetic data, ``tmp_path`` only.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from test_publishing_plan_io import _record, _write

from gomoveshift.cli.handlers import dispatch_queue as dq
from gomoveshift.health import check as check_mod
from gomoveshift.publishing.models import (
    MediaFingerprint,
    PlatformPayloads,
    PublishPlan,
    YouTubePayload,
)
from gomoveshift.publishing.plan_io import load_publish_plan
from gomoveshift.publishing.queue import enqueue, open_queue_db
from gomoveshift.publishing.queue_repair import repair_queued_shorts
from gomoveshift.publishing.youtube_adapter import short_review_refusal

DATE = "2026-08-01"
REGION = "Vernon County, Wisconsin"
NOW = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)


class _Geo:
    def reverse_geocode_detailed(self, lat, lon):
        return {"county": "Vernon County", "state": "Wisconsin", "town": "Viroqua"}


def _stages(base, project="proj", *, points=True):
    d = base / "_stages_root" / project / "_stages"
    d.mkdir(parents=True)
    if points:
        (d / f"gps_points_{DATE}.json").write_text(
            json.dumps([[float(t), 43.5, -90.9] for t in range(0, 400)])
        )
    entry = {"rank": 2, "parent_lf_index": 2, "start_sec": 100.0, "end_sec": 121.5}
    (d / f"shorts_ost_audit_{DATE}.json").write_text(json.dumps({"shorts": [entry]}))
    return d


def _short(short_index=3, tags=("Ireland road trip", "Vernon County")):
    media = MediaFingerprint("/v/short_02.mp4", 1234, 21.5, 1080, 1920, "h264", "a" * 64)
    yt = YouTubePayload(
        "Aug 1, 3rd stretch: Vernon County",
        "Drive through Vernon County. Filmed August 2026.",
        tuple(tags),
        "2",
    )
    return dataclasses.replace(
        _record(),
        media=media,
        short_index=short_index,
        sampled_regions=(REGION,),
        payloads=PlatformPayloads(youtube=yt),
    )


@pytest.fixture
def env(tmp_path):
    conn = open_queue_db(tmp_path / "q")
    stages = _stages(tmp_path)

    def add(*shorts, status="queued", slot=NOW + timedelta(hours=10)):
        out = tmp_path / "out"
        out.mkdir(exist_ok=True)
        sidecar = _write(out, PublishPlan(shorts=tuple(shorts)))
        ids = []
        for s in shorts:
            i = enqueue(
                conn,
                "proj",
                s.short_index,
                "youtube",
                "H",
                sidecar_path=str(sidecar),
                scheduled_at=slot.isoformat(),
            )
            conn.execute("UPDATE deliveries SET status=? WHERE id=?", (status, i))
            ids.append(i)
        conn.commit()
        return sidecar, ids

    yield SimpleNamespace(conn=conn, add=add, stages=stages, tmp=tmp_path)
    conn.close()


def _repair(env, *, apply, geocoder=lambda: _Geo(), stages=None):
    return repair_queued_shorts(
        env.conn,
        apply=apply,
        stages_dir_for=lambda sidecar, project: stages or env.stages,
        geocoder_factory=geocoder,
    )


def _plan_short(sidecar, idx):
    return next(s for s in load_publish_plan(sidecar).shorts if s.short_index == idx)


def test_the_fixture_is_refused_for_the_reason_the_real_rows_are(env):
    sidecar, _ = env.add(_short(3))
    refusal = short_review_refusal(_plan_short(sidecar, 3))
    assert refusal and "tag_unreviewable" in refusal


def test_refused_row_is_repaired_and_written(env):
    sidecar, (rid,) = env.add(_short(3))
    before = json.loads(sidecar.read_text())
    [r] = _repair(env, apply=True)
    assert (r.delivery_id, r.outcome, r.written) == (rid, "repaired", True)
    assert r.line() == f"QUEUE_REPAIR {rid} proj repaired"
    assert short_review_refusal(_plan_short(sidecar, 3)) is None
    short = json.loads(sidecar.read_text())["shorts"][0]
    old = before["shorts"][0]
    assert short["payloads"]["youtube"]["tags"] == ["Vernon County"]
    assert short["route_record"]["bins"]
    # nothing else moved: stamp prefix, schedule, date, arms, plan hash
    for key in ("schedule", "recording_date", "experiment_arms", "reviewed_plan_hash"):
        assert short.get(key) == old.get(key)
    assert short["payloads"]["youtube"]["title"] == old["payloads"]["youtube"]["title"]
    logged = env.conn.execute("SELECT count(*) FROM response_log WHERE action='queue_repair'")
    assert logged.fetchone()[0] == 1


def test_dry_run_reports_repaired_and_writes_nothing(env):
    sidecar, _ = env.add(_short(3))
    raw = sidecar.read_bytes()
    [r] = _repair(env, apply=False)
    assert r.outcome == "repaired" and r.written is False
    assert sidecar.read_bytes() == raw
    assert env.conn.execute("SELECT count(*) FROM response_log").fetchone()[0] == 0


def test_unrepairable_row_is_not_written_and_says_why(env, tmp_path):
    sidecar, (rid,) = env.add(_short(3))
    raw = sidecar.read_bytes()
    [r] = _repair(env, apply=True, stages=_stages(tmp_path, "gone", points=False))
    assert r.outcome == "still_refused" and r.written is False
    assert r.line().startswith(f"QUEUE_REPAIR {rid} proj still_refused:")
    assert "gps_points" in r.line() and "\n" not in r.line()
    assert sidecar.read_bytes() == raw  # the tag filter alone must not land


def test_no_geocoder_is_still_refused_not_written(env):
    sidecar, _ = env.add(_short(3))
    raw = sidecar.read_bytes()
    [r] = _repair(env, apply=True, geocoder=lambda: None)
    assert r.outcome == "still_refused" and "geocoder" in r.reason
    assert sidecar.read_bytes() == raw


def test_a_passing_row_is_left_untouched(env):
    sidecar, (rid,) = env.add(_short(3))
    _repair(env, apply=True)  # now repaired
    raw = sidecar.read_bytes()
    [r] = _repair(env, apply=True)
    assert (r.delivery_id, r.outcome) == (rid, "ok")
    assert sidecar.read_bytes() == raw


def test_a_published_row_is_never_touched(env):
    sidecar, _ = env.add(_short(3), status="published")
    raw = sidecar.read_bytes()
    assert _repair(env, apply=True) == []
    assert sidecar.read_bytes() == raw


def test_two_refused_rows_of_a_shared_sidecar_are_both_repaired(env):
    sidecar, (a, b) = env.add(_short(3), _short(4, tags=("Vernon County",)))
    results = {r.delivery_id: r.outcome for r in _repair(env, apply=True)}
    assert results == {a: "repaired", b: "repaired"}
    assert short_review_refusal(_plan_short(sidecar, 3)) is None
    assert short_review_refusal(_plan_short(sidecar, 4)) is None


def test_unreadable_sidecar_is_reported_not_fatal(env):
    sidecar, _ = env.add(_short(3))
    sidecar.unlink()
    [r] = _repair(env, apply=True)
    assert r.outcome == "still_refused" and "unreadable" in r.reason


def test_dispatch_queue_dry_run_prints_repair_lines_and_writes_nothing(env, capsys, monkeypatch):
    sidecar, (rid,) = env.add(_short(3))
    cfg = SimpleNamespace(
        seo=SimpleNamespace(storage_path=str(env.tmp / "q")),
        paths=SimpleNamespace(intermediate_folder=""),
    )
    monkeypatch.setattr(dq, "_routed", lambda config, platform: platform == "youtube")
    monkeypatch.setattr("gomoveshift.paths.gms_intermediate_dir", lambda *a, **k: env.stages)
    monkeypatch.setattr(
        "gomoveshift.pipeline.gps_enrichment._build_shared_geocoder", lambda *a, **k: _Geo()
    )
    raw = sidecar.read_bytes()
    args = SimpleNamespace(apply=False, sidecar_dir=None, ignore_validate=False, reason=None)
    dq.cmd_dispatch_queue(args, cfg)
    assert f"QUEUE_REPAIR {rid} proj repaired" in capsys.readouterr().out
    assert sidecar.read_bytes() == raw


# --- health rule ---------------------------------------------------------------------


def _incidents(env, now=NOW):
    notes: list = []
    db = env.tmp / "q" / "social_queue.sqlite3"
    return check_mod.queued_short_refused_incidents(db, now, now.isoformat(), notes), notes


def test_health_raises_for_a_refused_row_inside_48h(env):
    _, (rid,) = env.add(_short(3), slot=NOW + timedelta(hours=47))
    rows, _ = _incidents(env)
    assert [r["key"] for r in rows] == [f"queued_short_refused:{rid}"]
    assert "tag_unreviewable" in rows[0]["reason"]
    assert "refuses it" in check_mod._line(rows[0])


def test_health_ignores_a_refused_row_beyond_48h(env):
    env.add(_short(3), slot=NOW + timedelta(hours=49))
    assert _incidents(env)[0] == []


def test_health_raises_for_an_overdue_refused_row(env):
    env.add(_short(3), slot=NOW - timedelta(hours=2))
    assert len(_incidents(env)[0]) == 1


def test_health_is_quiet_for_a_repaired_row(env):
    env.add(_short(3), slot=NOW + timedelta(hours=1))
    _repair(env, apply=True)
    assert _incidents(env)[0] == []


def test_health_never_writes_and_notes_an_unreadable_sidecar(env):
    sidecar, _ = env.add(_short(3), slot=NOW + timedelta(hours=1))
    sidecar.unlink()
    db = env.tmp / "q" / "social_queue.sqlite3"
    before = db.read_bytes()
    rows, notes = _incidents(env)
    assert rows == [] and any("not reviewed" in n for n in notes)
    assert db.read_bytes() == before


def test_health_skips_a_published_row(env):
    env.add(_short(3), status="published", slot=NOW + timedelta(hours=1))
    assert _incidents(env)[0] == []
