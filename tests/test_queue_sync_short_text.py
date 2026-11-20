"""``sync_short_text`` re-writes queued Shorts with the current generator (plan unit C5).
Slot more than 24 h away, stamp, date, slot and arms kept, Short review re-run, plan hash moved
with the old one in ``response_log``. All data is synthetic."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from test_place_gate import Geo, _win

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
    changes = sync_short_text(conn, apply=True, now=NOW)
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
    changes = sync_short_text(conn, apply=True, now=NOW)
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
    changes = sync_short_text(conn, apply=True, now=NOW)
    assert [c.status for c in changes] == ["refused", "changed"]
    assert "title_unreviewable" in changes[0].reason
    assert json.loads(plan.read_text())["shorts"][0] == before
    assert _hashes(conn)[(1, "youtube")] == OLD_HASH


def test_the_title_obeys_the_no_repeated_place_rule(tmp_path):
    plan, conn = _world(tmp_path, [_short(1), _short(2)])
    sync_short_text(conn, apply=True, now=NOW)
    t1, t2 = (_youtube(plan, n).payloads.youtube.title for n in (1, 2))
    body = [t.split(": ", 1)[1] for t in (t1, t2)]
    assert body[0] != body[1]


def test_a_facebook_row_sharing_the_record_blocks_the_write_when_refused(tmp_path, monkeypatch):
    from gomoveshift.publishing import youtube_adapter

    plan, conn = _world(tmp_path, [_short(1)], platforms=("youtube", "facebook", "instagram"))
    monkeypatch.setattr(
        youtube_adapter, "short_review_refusal", lambda record: "place_unreviewable"
    )
    before = plan.read_text()
    changes = sync_short_text(conn, apply=True, now=NOW)
    assert [c.status for c in changes] == ["refused"]
    assert plan.read_text() == before
    assert set(_hashes(conn).values()) == {OLD_HASH}


def test_the_sharing_rows_move_to_the_new_hash_together(tmp_path):
    plan, conn = _world(tmp_path, [_short(1)], platforms=("youtube", "facebook", "instagram"))
    changes = sync_short_text(conn, apply=True, now=NOW)
    new = _youtube(plan, 1).reviewed_plan_hash
    assert new != OLD_HASH == changes[0].old_hash and changes[0].new_hash == new
    assert set(_hashes(conn).values()) == {new}


def test_dry_run_writes_nothing(tmp_path):
    plan, conn = _world(tmp_path, [_short(1), _short(2)])
    before = plan.read_text()
    changes = sync_short_text(conn, apply=False, now=NOW)
    assert [c.status for c in changes] == ["changed", "changed"]
    assert not any(c.written for c in changes)
    assert plan.read_text() == before
    assert not list(tmp_path.glob("*.sync-tmp"))
    assert set(_hashes(conn).values()) == {OLD_HASH}
    assert conn.execute("SELECT COUNT(*) FROM response_log").fetchone()[0] == 0


def test_a_second_run_is_a_no_op(tmp_path):
    plan, conn = _world(tmp_path, [_short(1), _short(2)])
    sync_short_text(conn, apply=True, now=NOW)
    after_first = plan.read_text()
    again = sync_short_text(conn, apply=True, now=NOW)
    assert [c.status for c in again] == ["unchanged", "unchanged"]
    assert sum(c.status == "changed" for c in again) == 0
    assert plan.read_text() == after_first
    assert conn.execute("SELECT COUNT(*) FROM response_log").fetchone()[0] == 2


def test_stamp_date_slot_and_arm_tags_are_preserved(tmp_path):
    plan, conn = _world(tmp_path, [_short(1, arms={"yt_title_body": "A", "fb_hashtags": "B"})])
    old = json.loads(plan.read_text())["shorts"][0]
    sync_short_text(conn, apply=True, now=NOW)
    new = json.loads(plan.read_text())["shorts"][0]
    assert new["recording_date"] == old["recording_date"]
    assert new["schedule"] == old["schedule"]
    assert new["experiment_arms"] == old["experiment_arms"]
    assert new["media"] == old["media"]
    assert new["payloads"]["youtube"]["title"].startswith("Mar 5, 1st stretch:")
    assert new["payloads"]["tiktok"] == old["payloads"]["tiktok"]


def test_the_old_plan_hash_is_kept_in_an_audit_record(tmp_path):
    plan, conn = _world(tmp_path, [_short(1)])
    sync_short_text(conn, apply=True, now=NOW)
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
    changes = sync_short_text(conn, apply=True, now=NOW)
    assert [c.short_index for c in changes] == [2]
    assert json.loads(plan.read_text())["shorts"][0] == before


def _where(conn, short_index, platform, **cols):
    sets = ", ".join(f"{k} = ?" for k in cols)
    conn.execute(
        f"UPDATE deliveries SET {sets} WHERE short_index = ? AND platform = ?",  # noqa: S608
        (*cols.values(), short_index, platform),
    )
    conn.commit()


def _named_first(n=1):
    """A Short whose STORED title and description name the county the generator opens with."""
    short = _short(n)
    yt = short["payloads"]["youtube"]
    yt["title"] = f"Mar 5, {ORDINALS[n - 1]} stretch: Ashby County #Shorts"
    return short


def _two(tmp_path, platforms=("youtube",)):
    return _world(tmp_path, [_short(1), _short(2)], platforms=platforms)


def _untouched(world, mutate):
    """Run the sync after ``mutate(conn)``; assert nothing moved; return the changes."""
    plan, conn = world
    mutate(conn)
    before = plan.read_text()
    changes = sync_short_text(conn, apply=True, now=NOW)
    assert plan.read_text() == before
    assert set(_hashes(conn).values()) == {OLD_HASH}
    return changes


@pytest.mark.parametrize("status", ["published", "uploading"])
def test_the_place_chain_is_seeded_from_the_stored_text_of_a_sibling(tmp_path, status):
    plan, conn = _world(tmp_path, [_named_first(), _short(2)])
    _where(conn, 1, "youtube", status=status)  # published / in flight: not rewritten
    sync_short_text(conn, apply=True, now=NOW)
    yt = _youtube(plan, 2).payloads.youtube
    assert "Harrow" in yt.title and "Ashby County" not in yt.title.split(" #")[0]
    assert "through Harrow" in yt.description
    assert _youtube(plan, 1).payloads.youtube.title.endswith("Ashby County #Shorts")


def test_the_sync_pins_the_generator_date_to_the_plan(tmp_path, monkeypatch):
    seen = []
    real = qst._generate_lf
    monkeypatch.setattr(
        qst, "_generate_lf", lambda *a, **kw: seen.append(kw["experiment_today"]) or real(*a, **kw)
    )
    # a plan date that is not today, so a generator left on date.today() cannot pass
    stale = "2026-11-18T10:00:00+00:00"
    _, conn = _world(tmp_path, [_short(1, parent_lf_publish_at=stale)])
    sync_short_text(conn, apply=False, now=NOW)
    assert seen == [datetime(2026, 11, 18).date()]


@pytest.mark.parametrize(
    "cols",
    [
        {"status": "awaiting_facebook_reconcile"},
        {"lease_owner": "w1", "lease_expires_at": (NOW + timedelta(hours=1)).isoformat()},
        {"scheduled_at": None},
        {"scheduled_at": (NOW + timedelta(hours=2)).isoformat()},
        {"recovers_delivery_id": 1},
    ],
    ids=["in_flight", "leased", "null_slot", "slot_in_lead_time", "recovery_row"],
)
def test_a_sibling_row_that_is_not_safe_refuses_the_whole_short(tmp_path, cols):
    world = _world(tmp_path, [_short(1)], platforms=("youtube", "facebook"))
    changes = _untouched(world, lambda c: _where(c, 1, "facebook", **cols))
    assert [c.status for c in changes] == ["refused"]
    assert "facebook row" in changes[0].reason


def test_a_leased_youtube_row_is_not_selected_and_a_lapsed_lease_is(tmp_path):
    _, conn = _two(tmp_path)
    live = (NOW + timedelta(hours=1)).isoformat()
    _where(conn, 1, "youtube", lease_owner="w1", lease_expires_at=live)
    _where(
        conn,
        2,
        "youtube",
        lease_owner="w1",
        lease_expires_at=(NOW - timedelta(hours=1)).isoformat(),
    )
    assert [c.short_index for c in sync_short_text(conn, apply=False, now=NOW)] == [2]


def test_the_update_leaves_updated_at_alone(tmp_path):
    plan, conn = _world(tmp_path, [_short(1)], platforms=("youtube", "facebook"))
    before = {r["platform"]: r["updated_at"] for r in conn.execute("SELECT * FROM deliveries")}
    sync_short_text(conn, apply=True, now=NOW)
    after = {r["platform"]: r["updated_at"] for r in conn.execute("SELECT * FROM deliveries")}
    assert before == after


def test_a_lease_taken_after_planning_keeps_that_row_out_of_the_update(tmp_path, monkeypatch):
    plan, conn = _world(tmp_path, [_short(1)], platforms=("youtube", "facebook"))
    real = qst._sidecar_changes

    def leased_late(*a):
        out = real(*a)
        live = (NOW + timedelta(hours=1)).isoformat()
        _where(conn, 1, "facebook", lease_owner="w1", lease_expires_at=live)
        return out

    monkeypatch.setattr(qst, "_sidecar_changes", leased_late)
    sync_short_text(conn, apply=True, now=NOW)
    assert _hashes(conn)[(1, "facebook")] == OLD_HASH != _hashes(conn)[(1, "youtube")]
    audit = json.loads(conn.execute("SELECT reason FROM response_log").fetchone()[0])
    assert audit["rows_moved"] == 1


def test_experiment_arms_are_kept_exactly_as_stored(tmp_path):
    arms = {"yt_title_body": "Z", "yt_hashtag_set": "Q"}
    plan, conn = _world(tmp_path, [_short(1, arms=arms)])
    sync_short_text(conn, apply=True, now=NOW)
    assert _youtube(plan, 1).experiment_arms == arms


def _converged(plan, conn):
    side = {s.short_index: s.reviewed_plan_hash for s in load_publish_plan(plan).shorts}
    return all(
        side[r["short_index"]] == r["plan_hash"] for r in conn.execute("SELECT * FROM deliveries")
    )


def test_a_dry_run_writes_no_file_at_all(tmp_path):
    _, conn = _two(tmp_path)
    seen = set(os.listdir(tmp_path))
    sync_short_text(conn, apply=False, now=NOW)
    assert set(os.listdir(tmp_path)) == seen


def test_the_queue_update_matches_the_rows_raw_sidecar_path(tmp_path):
    plan, conn = _world(tmp_path, [_short(1)])
    raw = f"{plan.parent}/./{plan.name}"
    conn.execute("UPDATE deliveries SET sidecar_path = ?", (raw,))
    conn.commit()
    sync_short_text(conn, apply=True, now=NOW)
    assert _converged(plan, conn)


def test_a_stored_title_arm_is_rendered_not_replaced_by_legacy_text(tmp_path):
    plan, conn = _world(tmp_path, [_short(1, arms={"yt_title_body": "clip_led"})])
    [change] = sync_short_text(conn, apply=True, now=NOW)
    assert change.status == "changed"
    assert _youtube(plan, 1).payloads.youtube.title.startswith("Mar 5, 1st stretch: 38 seconds, ")
    assert _youtube(plan, 1).experiment_arms == {"yt_title_body": "clip_led"}
