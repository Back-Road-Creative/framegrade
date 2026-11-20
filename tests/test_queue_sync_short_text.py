"""``gomoveshift queue sync-short-text`` re-writes queued Shorts with the current generator.

Plan unit C5: a Short queued before a generator fix or an arm decision keeps its old text.
The sync regenerates title, description and tags of every queued YouTube Short whose slot
is more than 24 h away, keeps the stamp, date, slot and arm tags, re-runs the Short review,
moves the plan hash and keeps the old one in ``response_log``. All data is synthetic."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from test_place_gate import Geo, _win

from gomoveshift.cli.handlers.queue_recover import cmd_queue
from gomoveshift.pipeline import longform_metadata as lm
from gomoveshift.pipeline.route_claims import measure_route
from gomoveshift.publishing import queue_short_text as qst
from gomoveshift.publishing.plan_io import load_publish_plan
from gomoveshift.publishing.queue import enqueue, open_queue_db
from gomoveshift.publishing.queue_short_text import SYNC_LEAD_TIME, sync_short_text
from gomoveshift.publishing.youtube_adapter import short_review_refusal

PROJECT = "nctopa-2026-03-05-rec0070"
REGION = "Ashby County, Westmark"
NOW = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)
OLD_HASH = "c" * 64
ORDINALS = ("1st", "2nd", "3rd")


@pytest.fixture(autouse=True)
def _no_published_titles(monkeypatch):
    monkeypatch.setattr(lm, "load_published_titles", lambda: set())


def _route():
    """A real route record: 300 retained seconds, all in Harrow, Ashby County."""
    return measure_route([_win("lf1", 0.0, 300.0, lambda t: 9.5)], Geo()).record()


def _short(n, hours_ahead=72, arms=None, **over):
    slot = (NOW + timedelta(hours=hours_ahead)).isoformat()
    short = {
        "additional_media": [],
        "consent_hash": "",
        "experiment_arms": {"yt_title_body": "A"} if arms is None else arms,
        "format": "short",
        "lf_index": 1,
        "media": {
            "codec": "hevc",
            "duration_seconds": 37.97,
            "height": 1920,
            "path": f"/nonexistent/short_{n:02d}.mp4",
            "sha256": f"{n:064x}",
            "size_bytes": 1000,
            "width": 1080,
        },
        "parent_lf_publish_at": "2026-11-20T10:00:00+00:00",
        "parent_lf_video_id": "t32KI85p5yU",
        "payloads": {
            "facebook": {"description": "Another mile in Ashby County. Filmed March 2026."},
            "instagram": {"caption": "Driving through Ashby County. Filmed March 2026."},
            "tiktok": {
                "caption": "Taking you through Ashby County. Filmed March 2026.",
                "disclose_branded_content": False,
                "disclose_your_brand": False,
                "privacy_level": "SELF_ONLY",
                "title": f"Mar 5, {ORDINALS[n - 1]} stretch: 38 seconds, Ashby County",
            },
            "youtube": {
                "category": "19",
                "description": "Front-seat view driving through Ashby County "
                "#Shorts\n\nhttps://youtu.be/t32KI85p5yU",
                "tags": ["POV driving", "drive", "road trip"],
                "title": f"Mar 5, {ORDINALS[n - 1]} stretch: old words #Shorts",
            },
        },
        "project_name": PROJECT,
        "recording_date": "2026-03-05",
        "reviewed_plan_hash": OLD_HASH,
        "route_record": _route(),
        "routes": {"selected": ["youtube", "instagram", "facebook"]},
        "sampled_regions": [REGION],
        "sampled_roads": [],
        "schedule": {
            "display_timezone": "America/New_York",
            "scheduled_at_utc": slot,
            "scheduled_at_utc_by_platform": {},
        },
        "short_index": n,
    }
    short.update(over)
    return short


def _world(tmp_path, shorts, platforms=("youtube",)):
    """A plan + LF sidecar and a queue holding one row per Short and platform."""
    (tmp_path / f"{PROJECT}_lf01_youtube.json").write_text(json.dumps({"location": REGION}))
    plan = tmp_path / f"{PROJECT}_publish_plan.json"
    plan.write_text(json.dumps({"shorts": shorts}, sort_keys=True, indent=2) + "\n")
    conn = open_queue_db(tmp_path)
    for s in shorts:
        for platform in platforms:
            enqueue(
                conn,
                PROJECT,
                s["short_index"],
                platform,
                s["reviewed_plan_hash"],
                sidecar_path=str(plan),
                scheduled_at=s["schedule"]["scheduled_at_utc"],
            )
    return plan, conn


def _youtube(plan, n):
    return next(s for s in load_publish_plan(plan).shorts if s.short_index == n)


def _hashes(conn):
    return {
        (r["short_index"], r["platform"]): r["plan_hash"]
        for r in conn.execute("SELECT * FROM deliveries")
    }


def test_a_changed_short_is_rewritten_and_passes_the_review(tmp_path):
    plan, conn = _world(tmp_path, [_short(1), _short(2)])
    changes = sync_short_text(conn, {}, apply=True, now=NOW)
    assert [c.status for c in changes] == ["changed", "changed"]
    for c in changes:
        rec = _youtube(plan, c.short_index)
        assert rec.payloads.youtube.title == c.new_title != c.old_title
        assert rec.payloads.youtube.title.startswith(
            f"Mar 5, {ORDINALS[c.short_index - 1]} stretch:"
        )
        assert short_review_refusal(rec) is None
        assert c.written


def test_a_short_inside_the_lead_time_is_skipped(tmp_path):
    hours = SYNC_LEAD_TIME.total_seconds() / 3600
    plan, conn = _world(tmp_path, [_short(1, hours_ahead=hours - 1), _short(2)])
    before = json.loads(plan.read_text())["shorts"][0]
    changes = sync_short_text(conn, {}, apply=True, now=NOW)
    assert [c.status for c in changes] == ["inside_lead_time", "changed"]
    assert json.loads(plan.read_text())["shorts"][0] == before


def test_a_short_the_review_refuses_is_left_untouched(tmp_path, monkeypatch):
    from gomoveshift.publishing import youtube_adapter

    real = youtube_adapter.short_review_refusal
    monkeypatch.setattr(
        youtube_adapter,
        "short_review_refusal",
        lambda record: "title_unreviewable" if record.short_index == 1 else real(record),
    )
    plan, conn = _world(tmp_path, [_short(1), _short(2)])
    before = json.loads(plan.read_text())["shorts"][0]
    changes = sync_short_text(conn, {}, apply=True, now=NOW)
    assert [c.status for c in changes] == ["refused", "changed"]
    assert "title_unreviewable" in changes[0].reason
    assert json.loads(plan.read_text())["shorts"][0] == before
    assert _hashes(conn)[(1, "youtube")] == OLD_HASH


def test_the_title_obeys_the_no_repeated_place_rule(tmp_path):
    plan, conn = _world(tmp_path, [_short(1), _short(2)])
    sync_short_text(conn, {}, apply=True, now=NOW)
    t1, t2 = (_youtube(plan, n).payloads.youtube.title for n in (1, 2))
    body = [t.split(": ", 1)[1] for t in (t1, t2)]
    assert body[0] != body[1]


def test_a_demoted_arm_is_not_chosen_and_the_tag_follows(tmp_path):
    arms = {"yt_title_body": "A"}
    decisions = {"yt_title_body": {"A": "demote", "B": "hold", "C": "pause"}}
    plan, conn = _world(tmp_path, [_short(1, arms=arms)])
    changes = sync_short_text(conn, decisions, apply=True, now=NOW)
    assert changes[0].status == "changed"
    assert "experiment_arms" in changes[0].fields
    assert _youtube(plan, 1).experiment_arms == {"yt_title_body": "B"}


def test_every_arm_barred_leaves_the_short_unchanged(tmp_path):
    decisions = {"yt_title_body": {"A": "demote", "B": "pause"}}
    plan, conn = _world(tmp_path, [_short(1)])
    before = plan.read_text()
    changes = sync_short_text(conn, decisions, apply=True, now=NOW)
    assert changes[0].status == "refused"
    assert "no other arm is open" in changes[0].reason
    assert plan.read_text() == before


def test_no_decision_keeps_the_current_arm(tmp_path):
    plan, conn = _world(tmp_path, [_short(1, arms={"yt_title_body": "A"})])
    sync_short_text(conn, {"yt_title_body": {"B": "hold"}}, apply=True, now=NOW)
    assert _youtube(plan, 1).experiment_arms == {"yt_title_body": "A"}


def test_a_facebook_row_sharing_the_record_blocks_the_write_when_refused(tmp_path, monkeypatch):
    from gomoveshift.publishing import youtube_adapter

    plan, conn = _world(tmp_path, [_short(1)], platforms=("youtube", "facebook", "instagram"))
    monkeypatch.setattr(
        youtube_adapter, "short_review_refusal", lambda record: "place_unreviewable"
    )
    before = plan.read_text()
    changes = sync_short_text(conn, {}, apply=True, now=NOW)
    assert [c.status for c in changes] == ["refused"]
    assert plan.read_text() == before
    assert set(_hashes(conn).values()) == {OLD_HASH}


def test_the_sharing_rows_move_to_the_new_hash_together(tmp_path):
    plan, conn = _world(tmp_path, [_short(1)], platforms=("youtube", "facebook", "instagram"))
    changes = sync_short_text(conn, {}, apply=True, now=NOW)
    new = _youtube(plan, 1).reviewed_plan_hash
    assert new != OLD_HASH == changes[0].old_hash and changes[0].new_hash == new
    assert set(_hashes(conn).values()) == {new}


def test_dry_run_writes_nothing(tmp_path):
    plan, conn = _world(tmp_path, [_short(1), _short(2)])
    before = plan.read_text()
    changes = sync_short_text(conn, {}, apply=False, now=NOW)
    assert [c.status for c in changes] == ["changed", "changed"]
    assert not any(c.written for c in changes)
    assert plan.read_text() == before
    assert not list(tmp_path.glob("*.sync-tmp"))
    assert set(_hashes(conn).values()) == {OLD_HASH}
    assert conn.execute("SELECT COUNT(*) FROM response_log").fetchone()[0] == 0


def test_a_second_run_is_a_no_op(tmp_path):
    plan, conn = _world(tmp_path, [_short(1), _short(2)])
    sync_short_text(conn, {}, apply=True, now=NOW)
    after_first = plan.read_text()
    again = sync_short_text(conn, {}, apply=True, now=NOW)
    assert [c.status for c in again] == ["unchanged", "unchanged"]
    assert sum(c.status == "changed" for c in again) == 0
    assert plan.read_text() == after_first
    assert conn.execute("SELECT COUNT(*) FROM response_log").fetchone()[0] == 2


def test_stamp_date_slot_and_arm_tags_are_preserved(tmp_path):
    plan, conn = _world(tmp_path, [_short(1, arms={"yt_title_body": "A", "fb_hashtags": "B"})])
    old = json.loads(plan.read_text())["shorts"][0]
    sync_short_text(conn, {}, apply=True, now=NOW)
    new = json.loads(plan.read_text())["shorts"][0]
    assert new["recording_date"] == old["recording_date"]
    assert new["schedule"] == old["schedule"]
    assert new["experiment_arms"] == old["experiment_arms"]
    assert new["media"] == old["media"]
    assert new["payloads"]["youtube"]["title"].startswith("Mar 5, 1st stretch:")
    assert new["payloads"]["tiktok"] == old["payloads"]["tiktok"]


def test_the_old_plan_hash_is_kept_in_an_audit_record(tmp_path):
    plan, conn = _world(tmp_path, [_short(1)])
    sync_short_text(conn, {}, apply=True, now=NOW)
    row = conn.execute("SELECT * FROM response_log WHERE action = 'sync_short_text'").fetchone()
    audit = json.loads(row["reason"])
    assert audit["old_plan_hash"] == OLD_HASH
    assert audit["new_plan_hash"] == _youtube(plan, 1).reviewed_plan_hash != OLD_HASH
    assert audit["rows_moved"] == 1


def test_a_published_row_is_never_touched(tmp_path):
    plan, conn = _world(tmp_path, [_short(1), _short(2)])
    conn.execute("UPDATE deliveries SET status = 'published' WHERE short_index = 1")
    conn.commit()
    before = json.loads(plan.read_text())["shorts"][0]
    changes = sync_short_text(conn, {}, apply=True, now=NOW)
    assert [c.short_index for c in changes] == [2]
    assert json.loads(plan.read_text())["shorts"][0] == before


def test_the_cli_action_is_dry_run_by_default(tmp_path, capsys, monkeypatch):
    soon = datetime.now(timezone.utc) + timedelta(days=3)
    schedule = {
        "display_timezone": "America/New_York",
        "scheduled_at_utc": soon.isoformat(),
        "scheduled_at_utc_by_platform": {},
    }
    plan, conn = _world(tmp_path, [_short(1, schedule=schedule)])
    conn.close()
    from gomoveshift.cli.handlers import metrics

    monkeypatch.setattr(metrics, "shorts_arm_decisions", lambda config, brand: ({}, None))
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
    args = SimpleNamespace(queue_action="sync-short-text", apply=False)
    before = plan.read_text()
    assert cmd_queue(args, config) == 0
    assert "dry run: would change" in capsys.readouterr().out
    assert plan.read_text() == before
    args.apply = True
    assert cmd_queue(args, config) == 0
    assert plan.read_text() != before
