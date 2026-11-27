"""``queue sync-short-text`` re-writes queued Shorts with the current generator (plan unit C5).
Slot more than 24 h away, stamp, date, slot and arms kept, Short review re-run, plan hash moved
with the old one in ``response_log``. All data is synthetic."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from test_place_gate import Geo, _win

from gomoveshift.cli.handlers.queue_recover import cmd_queue
from gomoveshift.pipeline import longform_metadata as lm
from gomoveshift.pipeline.route_claims import measure_route
from gomoveshift.publishing import anti_spam_gate, models
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
        # the measured place (the town, then the county), else the stamped title
        title = rec.payloads.youtube.title
        assert title.split(" #", 1)[0] in {"Harrow", "Ashby County"} or title.startswith(
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
    body = [t.split(" #", 1)[0].split(": ", 1)[-1] for t in (t1, t2)]
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
    assert new["payloads"]["youtube"]["title"].split(" #", 1)[0] == "Harrow"
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
    # the description opens with the Short's own title place, never the sibling's
    assert yt.description.startswith("Harrow \u2014 POV drive.")
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


def test_a_row_leased_between_the_read_and_the_write_rolls_the_sidecar_back(tmp_path, monkeypatch):
    plan, conn = _world(tmp_path, [_short(1)], platforms=("youtube", "facebook"))
    real = qst._sidecar_changes

    def leased_late(*a):
        out = real(*a)
        live = (NOW + timedelta(hours=1)).isoformat()
        _where(conn, 1, "facebook", lease_owner="w1", lease_expires_at=live)
        return out

    monkeypatch.setattr(qst, "_sidecar_changes", leased_late)
    before = plan.read_text()
    changes = sync_short_text(conn, apply=True, now=NOW)
    assert [c.status for c in changes] == ["refused"]
    assert "a row was leased during the write" in changes[0].reason
    assert plan.read_text() == before and set(_hashes(conn).values()) == {OLD_HASH}
    assert conn.execute("SELECT COUNT(*) FROM response_log").fetchone()[0] == 0


def test_experiment_arms_are_kept_exactly_as_stored(tmp_path):
    arms = {"yt_title_body": "Z", "yt_hashtag_set": "Q"}
    plan, conn = _world(tmp_path, [_short(1, arms=arms)])
    sync_short_text(conn, apply=True, now=NOW)
    assert _youtube(plan, 1).experiment_arms == arms


class _CommitFails:
    """A connection whose ``with conn:`` rolls back and then fails like a full disk."""

    def __init__(self, conn):
        self._c = conn

    def __getattr__(self, name):
        return getattr(self._c, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._c.rollback()
        raise sqlite3.OperationalError("disk I/O error")


def _converged(plan, conn):
    side = {s.short_index: s.reviewed_plan_hash for s in load_publish_plan(plan).shorts}
    return all(
        side[r["short_index"]] == r["plan_hash"] for r in conn.execute("SELECT * FROM deliveries")
    )


def test_a_commit_failure_leaves_both_sides_alone_and_the_rerun_converges(tmp_path):
    plan, conn = _two(tmp_path)
    before = plan.read_text()
    changes = sync_short_text(_CommitFails(conn), apply=True, now=NOW)
    assert {c.status for c in changes} == {"refused"} and not any(c.written for c in changes)
    assert plan.read_text() == before and set(_hashes(conn).values()) == {OLD_HASH}
    assert conn.execute("SELECT COUNT(*) FROM response_log").fetchone()[0] == 0
    assert not list(tmp_path.glob("*.sync-short-tmp"))
    again = sync_short_text(conn, apply=True, now=NOW)
    assert [c.status for c in again] == ["changed", "changed"] and _converged(plan, conn)


class _Crash(BaseException):
    pass


def test_a_crash_after_commit_before_the_replace_converges_on_the_next_run(tmp_path, monkeypatch):
    plan, conn = _two(tmp_path)
    before = plan.read_text()

    def die(*a):
        raise _Crash

    monkeypatch.setattr(qst.os, "replace", die)
    with pytest.raises(_Crash):
        sync_short_text(conn, apply=True, now=NOW)
    monkeypatch.undo()
    assert plan.read_text() == before and not _converged(plan, conn)
    assert conn.execute("SELECT COUNT(*) FROM response_log").fetchone()[0] == 2  # audited
    again = sync_short_text(conn, apply=True, now=NOW)
    assert [c.status for c in again] == ["changed", "changed"] and _converged(plan, conn)
    assert [c.status for c in sync_short_text(conn, apply=True, now=NOW)] == ["unchanged"] * 2


def test_a_queue_row_behind_a_synced_sidecar_is_moved_with_an_audit_row(tmp_path):
    plan, conn = _two(tmp_path)
    sync_short_text(conn, apply=True, now=NOW)
    conn.execute("UPDATE deliveries SET plan_hash = ?", (OLD_HASH,))
    conn.commit()
    changes = sync_short_text(conn, apply=True, now=NOW)
    assert [c.status for c in changes] == ["reconciled", "reconciled"] and _converged(plan, conn)
    audits = [json.loads(r["reason"]) for r in conn.execute("SELECT * FROM response_log")]
    assert sum(a["reconcile"] for a in audits) == 2


def test_a_row_with_an_unknown_plan_hash_refuses_the_short(tmp_path):
    plan, conn = _world(tmp_path, [_short(1)])
    conn.execute("UPDATE deliveries SET plan_hash = ?", ("e" * 64,))
    conn.commit()
    before = plan.read_text()
    assert sync_short_text(conn, apply=True, now=NOW)[0].status == "refused"
    assert plan.read_text() == before


def test_a_legacy_pool_title_is_refused_by_the_anti_spam_gate(tmp_path, monkeypatch):
    plan, conn = _two(tmp_path)
    title = sync_short_text(conn, apply=False, now=NOW)[0].new_title
    pool = anti_spam_gate.LEGACY_TITLE_POOL | {
        anti_spam_gate._normalize_exact(anti_spam_gate._bare_title(title))
    }
    monkeypatch.setattr(anti_spam_gate, "LEGACY_TITLE_POOL", pool)
    before = plan.read_text()
    changes = sync_short_text(conn, apply=True, now=NOW)
    assert changes[0].status == "refused" and "legacy-title" in changes[0].reason
    assert (
        _youtube(plan, 1).payloads.youtube.title
        == json.loads(before)["shorts"][0]["payloads"]["youtube"]["title"]
    )


def test_a_payload_over_a_platform_limit_is_refused(tmp_path, monkeypatch):
    def over(payloads):
        raise models.PayloadLimitError("youtube title: too long")

    monkeypatch.setattr(models, "validate_payloads", over)
    plan, conn = _two(tmp_path)
    before = plan.read_text()
    changes = sync_short_text(conn, apply=True, now=NOW)
    assert {c.status for c in changes} == {"refused"} and "payload limit" in changes[0].reason
    assert plan.read_text() == before


PROJECT2 = "nctopa-2026-03-06-rec0071"


def _other(conn, tmp_path, published=False, day=6):
    """A second plan (own directory and project) queued in the same database."""
    d = tmp_path / "other"
    d.mkdir()
    (d / f"{PROJECT2}_lf01_youtube.json").write_text(json.dumps({"location": REGION}))
    short = _short(1, project_name=PROJECT2, recording_date=f"2026-03-0{day}")
    short["payloads"]["youtube"]["title"] = f"Mar {day}, 1st stretch: zebra words #Shorts"
    plan = d / f"{PROJECT2}_publish_plan.json"
    plan.write_text(json.dumps({"shorts": [short]}, sort_keys=True, indent=2) + "\n")
    enqueue(
        conn,
        PROJECT2,
        1,
        "youtube",
        short["reviewed_plan_hash"],
        sidecar_path=str(plan),
        scheduled_at=short["schedule"]["scheduled_at_utc"],
    )
    if published:  # not in the sync set, so its text is the queue's RECENT text
        conn.execute(
            "UPDATE deliveries SET status = 'published' WHERE project_name = ?", (PROJECT2,)
        )
        conn.commit()
    return plan


def test_the_gates_see_the_queues_recent_text(tmp_path, monkeypatch):
    seen = {}
    real = anti_spam_gate.check_batch
    monkeypatch.setattr(
        anti_spam_gate, "check_batch", lambda r, **kw: seen.update(kw) or real(r, **kw)
    )
    _, conn = _two(tmp_path)
    _other(conn, tmp_path, published=True)
    sync_short_text(conn, apply=False, now=NOW, per_location_cap=1)
    assert seen["per_location_cap"] == 1
    assert any("zebra words" in text for text in seen["recent_texts"]["youtube"])


def test_two_sidecars_dry_run_and_apply_agree_on_one_snapshot_per_run(tmp_path, monkeypatch):
    _, conn = _two(tmp_path)
    _other(conn, tmp_path)
    reads = []
    real = anti_spam_gate.recent_text_entries
    monkeypatch.setattr(
        anti_spam_gate,
        "recent_text_entries",
        lambda c, **kw: reads.append(kw) or real(c, **kw),
    )

    def key(changes):
        return [(c.delivery_id, c.status, c.new_title, c.new_hash) for c in changes]

    dry = sync_short_text(conn, apply=False, now=NOW)
    applied = sync_short_text(conn, apply=True, now=NOW)
    assert key(dry) == key(applied) and {c.status for c in applied} == {"changed"}
    # one pre-run read per run (filtered per sidecar afterwards), never one per sidecar
    assert reads == [{"now": NOW}, {"now": NOW}]


def test_a_hand_titled_short_is_left_alone(tmp_path):
    plan, conn = _two(tmp_path)
    (tmp_path / "social").mkdir()
    (tmp_path / "social" / "titles.yaml").write_text('1:\n  title: "Mar 5, 1st stretch: by hand"\n')
    changes = sync_short_text(conn, apply=True, now=NOW)
    assert [(c.status, c.reason) for c in changes] == [
        ("unchanged", "hand_title"),
        ("changed", ""),
    ]
    assert "old words" in _youtube(plan, 1).payloads.youtube.title


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


def test_a_stored_arm_is_rendered_not_replaced_by_legacy_text(tmp_path):
    # The title-body arms shape only the stamped fallback and are never assigned now, so the
    # surviving stored arm is the hashtag set: it is rendered, not dropped.
    plan, conn = _world(tmp_path, [_short(1, arms={"yt_hashtag_set": "road"})])
    [change] = sync_short_text(conn, apply=True, now=NOW)
    assert change.status == "changed"
    assert _youtube(plan, 1).payloads.youtube.title.split(" #", 1)[0] == "Harrow"
    assert "#RoadTrip" in _youtube(plan, 1).payloads.youtube.description
    assert _youtube(plan, 1).experiment_arms == {"yt_hashtag_set": "road"}


def test_a_stored_title_body_arm_is_kept_as_stored_not_refused(tmp_path):
    # A fact title carries no title-body variant, so a plan holding one is left as it was: it
    # posts its stored text, and a refusal would only file the same incident every night.
    plan, conn = _world(tmp_path, [_short(1, arms={"yt_title_body": "clip_led"})])
    before = plan.read_bytes()
    [change] = sync_short_text(conn, apply=True, now=NOW)
    assert change.status == "unchanged" and "no longer renders" in change.reason
    assert not qst.is_actionable_refusal(change)
    assert plan.read_bytes() == before


def test_the_unrendered_factors_are_the_ones_the_generator_never_carries():
    from gomoveshift.pipeline.social_generator import SocialGenerator

    gen = SocialGenerator(location=REGION, project_name=PROJECT)
    carried = set(gen._shorts_arm_factors(1, None))
    assert set(qst.UNRENDERED_FACTORS) == set(qst.SHORTS_ARMS) - carried


def _recorded(n, title, generator_title):
    """A Short the plan builder recorded ``generator_title`` for, now titled ``title``."""
    short = _short(n)
    short["payloads"]["youtube"]["title"] = title
    short["generator_title"] = generator_title
    return short


def test_a_hand_edited_title_is_never_regenerated(tmp_path):
    # Both titles LOOK generated (a valid stamp; exactly a measured place), but neither is the
    # text the generator recorded: they were set by hand, so the sidecar stays byte for byte.
    plan, conn = _world(
        tmp_path,
        [
            _recorded(1, "Mar 5, 1st stretch: my own words #Shorts", "Mar 5, 1st stretch: x"),
            _recorded(2, "Harrow #Shorts", "Mar 5, 2nd stretch: old words #Shorts"),
        ],
    )
    before = plan.read_bytes()
    changes = sync_short_text(conn, apply=True, now=NOW)
    assert [(c.status, c.reason) for c in changes] == [
        ("unchanged", "hand_title (edited in the plan)")
    ] * 2
    assert plan.read_bytes() == before
    assert set(_hashes(conn).values()) == {OLD_HASH}


def test_a_generator_written_title_still_syncs_and_records_the_new_one(tmp_path):
    stored = "Mar 5, 1st stretch: old words #Shorts"
    plan, conn = _world(tmp_path, [_recorded(1, stored, stored)])
    [change] = sync_short_text(conn, apply=True, now=NOW)
    assert change.status == "changed" and change.new_title != stored
    rec = _youtube(plan, 1)
    assert rec.generator_title == rec.payloads.youtube.title == change.new_title
    assert [c.status for c in sync_short_text(conn, apply=True, now=NOW)] == ["unchanged"]


def test_a_sidecar_edited_during_the_run_is_refused(tmp_path, monkeypatch):
    plan, conn = _two(tmp_path)
    real = qst._sidecar_changes

    def edited(*a):
        out = real(*a)
        plan.write_text(plan.read_text() + "\n")
        return out

    monkeypatch.setattr(qst, "_sidecar_changes", edited)
    changes = sync_short_text(conn, apply=True, now=NOW)
    assert {c.status for c in changes} == {"refused"} and "changed while" in changes[0].reason
    assert set(_hashes(conn).values()) == {OLD_HASH}


def _live_schedule():
    """A slot 3 days past the REAL clock: the CLI reads datetime.now, not NOW."""
    soon = datetime.now(timezone.utc) + timedelta(days=3)
    return {
        "display_timezone": "America/New_York",
        "scheduled_at_utc": soon.isoformat(),
        "scheduled_at_utc_by_platform": {},
    }


def test_the_cli_action_is_dry_run_by_default(tmp_path, capsys, monkeypatch):
    plan, conn = _world(tmp_path, [_short(1, schedule=_live_schedule())])
    conn.close()
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
    args = SimpleNamespace(queue_action="sync-short-text", apply=False)
    before = plan.read_text()
    assert cmd_queue(args, config) == 0
    assert "dry run: would change" in capsys.readouterr().out
    assert plan.read_text() == before
    args.apply = True
    assert cmd_queue(args, config) == 0
    assert plan.read_text() != before


def test_the_cli_exits_non_zero_when_a_short_is_refused(tmp_path):
    soon = datetime.now(timezone.utc) + timedelta(days=3)
    schedule = {**_short(1)["schedule"], "scheduled_at_utc": soon.isoformat()}
    plan, conn = _world(tmp_path, [_short(1, schedule=schedule)])
    conn.close()
    (tmp_path / f"{PROJECT}_lf01_youtube.json").unlink()  # the LF sidecar is gone
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
    args = SimpleNamespace(queue_action="sync-short-text", apply=True)
    assert cmd_queue(args, config) == 1


def test_a_row_on_a_hash_this_sync_never_wrote_is_refused_not_moved(tmp_path):
    plan, conn = _world(tmp_path, [_short(1)])
    sync_short_text(conn, apply=True, now=NOW)
    foreign = "e" * 64  # a Phase D re-plan: no audit row of ours names it
    conn.execute("UPDATE deliveries SET plan_hash = ?", (foreign,))
    conn.commit()
    before = plan.read_text()
    changes = sync_short_text(conn, apply=True, now=NOW)
    assert [c.status for c in changes] == ["refused"] and "unknown" in changes[0].reason
    assert plan.read_text() == before and set(_hashes(conn).values()) == {foreign}


def test_a_rewritten_short_does_not_repeat_the_place_of_a_kept_successor(tmp_path):
    successor = _named_first(2)
    # kept as stored: names Harrow, the place the generator opens the rewritten Short with
    successor["payloads"]["youtube"]["title"] = "Mar 5, 2nd stretch: Harrow #Shorts"
    successor["payloads"]["youtube"]["description"] = "Harrow \u2014 POV drive. #Shorts"
    plan, conn = _world(tmp_path, [_short(1), successor])
    _where(conn, 2, "youtube", status="published")
    changes = sync_short_text(conn, apply=True, now=NOW)
    # the kept successor never changes, so refusing the Short for its place would repeat
    # every night: the generator is steered off that place and names the clip's other one
    assert [c.status for c in changes] == ["changed"], changes[0].reason
    assert _youtube(plan, 1).payloads.youtube.title.split(" #", 1)[0] == "Ashby County"


def test_a_kept_hand_titled_successor_never_refuses_a_short_with_no_other_place(
    tmp_path, monkeypatch
):
    # The 20th/21st case: the 21st's hand title names the 20th's only measured place. The 20th
    # cannot vary, so the repeat is sanctioned, not refused night after night.
    from gomoveshift.pipeline import social_generator

    monkeypatch.setattr(
        social_generator, "measured_places", lambda *a, **k: [("county", "Ashby County")]
    )
    successor = _short(2)
    successor["payloads"]["youtube"]["title"] = "Ashby County back roads #Shorts"
    plan, conn = _world(tmp_path, [_short(1), successor])
    (tmp_path / "social").mkdir()
    (tmp_path / "social" / "titles.yaml").write_text('2:\n  title: "Ashby County back roads"\n')
    changes = sync_short_text(conn, apply=True, now=NOW)
    assert [(c.status, c.reason) for c in changes] == [("changed", ""), ("unchanged", "hand_title")]
    assert "Ashby County" in _youtube(plan, 1).payloads.youtube.title


@pytest.mark.parametrize("gate", ["assert_plan_text_grammar", "assert_shorts_metadata_in_region"])
def test_a_plan_text_gate_that_raises_refuses_the_short(tmp_path, monkeypatch, gate):
    from gomoveshift.publishing import plan_gates

    def boom(by_lf):
        raise AssertionError(f"{gate} says no")

    monkeypatch.setattr(plan_gates, gate, boom)
    plan, conn = _two(tmp_path)
    before = plan.read_text()
    changes = sync_short_text(conn, apply=True, now=NOW)
    assert {c.status for c in changes} == {"refused"} and f"{gate} says no" in changes[0].reason
    assert plan.read_text() == before


def test_a_generator_failure_for_one_short_refuses_only_that_short(tmp_path, monkeypatch):
    real = qst._generate_lf

    def flaky(view, *a, **kw):
        live = [s["short_index"] for s in view if not s.get("_discard")]
        if len(live) > 1 or live == [2]:
            raise ValueError("no title for this Short")
        return real(view, *a, **kw)

    monkeypatch.setattr(qst, "_generate_lf", flaky)
    second = _short(2)
    second["payloads"]["youtube"]["description"] = "A quiet road. #Shorts"  # names no place
    plan, conn = _world(tmp_path, [_short(1), second])
    changes = sync_short_text(conn, apply=True, now=NOW)
    assert [c.status for c in changes] == ["changed", "refused"]
    assert "no title for this Short" in changes[1].reason
    assert "old words" in _youtube(plan, 2).payloads.youtube.title


def test_a_terminal_row_already_on_the_new_hash_refuses_the_short(tmp_path):
    plan, conn = _world(tmp_path, [_short(1)])
    new_hash = sync_short_text(conn, apply=False, now=NOW)[0].new_hash
    row = dict(conn.execute("SELECT * FROM deliveries").fetchone())
    row.update(id=None, plan_hash=new_hash, status="failed_permanent")
    conn.execute(
        f"INSERT INTO deliveries ({', '.join(row)}) VALUES ({', '.join('?' for _ in row)})",  # noqa: S608
        tuple(row.values()),
    )
    conn.commit()
    before = plan.read_text()
    changes = sync_short_text(conn, apply=True, now=NOW)
    assert [c.status for c in changes] == ["refused"] and "already holds" in changes[0].reason
    assert plan.read_text() == before


def test_the_location_cap_is_counted_across_the_sidecar(tmp_path):
    plan, conn = _two(tmp_path)  # two Shorts, one location, one sidecar
    changes = sync_short_text(conn, apply=False, now=NOW, per_location_cap=1)
    assert any("location-cap" in c.reason for c in changes if c.status == "refused")


def test_the_cli_passes_the_brand_location_cap(tmp_path, monkeypatch):
    # Real-clock slots: a NOW-relative slot fell inside the lead time on 2026-11-22.
    live = _live_schedule()
    plan, conn = _world(tmp_path, [_short(1, schedule=live), _short(2, schedule=live)])
    conn.close()
    monkeypatch.setattr(
        "gomoveshift.publishing.brand_policy.load_brand_policy",
        lambda brand: SimpleNamespace(per_location_cap=1),
    )
    monkeypatch.setattr("gomoveshift.console.settings.current_brand", lambda: "brand")
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
    args = SimpleNamespace(queue_action="sync-short-text", apply=False)
    before = plan.read_text()
    assert cmd_queue(args, config) == 1  # the second Short is refused for location-cap
    assert plan.read_text() == before


def test_two_queued_projects_cannot_both_take_the_same_text(tmp_path, monkeypatch):
    # Both places are already live titles, so each Short falls back to the stamped title
    # ("<date>, Nth stretch: ..."), the only title the 3-word shared-prefix rule can see.
    monkeypatch.setattr(lm, "load_published_titles", lambda: {"harrow", "ashby county"})
    plan, conn = _world(tmp_path, [_short(1)])
    _other(conn, tmp_path, day=5)  # same day and place, so the same text
    changes = sync_short_text(conn, apply=True, now=NOW)
    # The second sidecar's text equals the first's, so it is refused; the first's new text in
    # turn clashes with the second's text as STORED, so it is refused too (checked against later
    # sidecars): nothing is written and the pair stays stable night to night.
    assert [c.status for c in changes] == ["refused", "refused"]
    assert all("anti-spam gate" in c.reason for c in changes)
    assert "later sidecar" in changes[0].reason
    assert not any(c.written for c in changes)


def test_a_reconcile_onto_a_hash_a_terminal_row_holds_is_unchanged(tmp_path):
    plan, conn = _world(tmp_path, [_short(1)])
    sync_short_text(conn, apply=True, now=NOW)
    new_hash = _hashes(conn)[(1, "youtube")]
    row = dict(conn.execute("SELECT * FROM deliveries").fetchone())
    conn.execute("UPDATE deliveries SET plan_hash = ?", (OLD_HASH,))
    row.update(id=None, plan_hash=new_hash, status="failed_permanent")
    conn.execute(
        f"INSERT INTO deliveries ({', '.join(row)}) VALUES ({', '.join('?' for _ in row)})",  # noqa: S608
        tuple(row.values()),
    )
    conn.commit()
    changes = sync_short_text(conn, apply=True, now=NOW)
    # nothing for the sync to write: by design `unchanged`, not a refusal (TRANSIENT_REFUSALS note)
    assert [c.status for c in changes] == ["unchanged"] and "already holds" in changes[0].reason


def test_a_crash_then_a_generator_change_still_converges(tmp_path, monkeypatch):
    plan, conn = _world(tmp_path, [_short(1)])

    def die(*a):
        raise _Crash

    monkeypatch.setattr(qst.os, "replace", die)
    with pytest.raises(_Crash):
        sync_short_text(conn, apply=True, now=NOW)
    monkeypatch.undo()
    real = qst._youtube_text
    monkeypatch.setattr(
        qst,
        "_youtube_text",
        lambda s, p: {**real(s, p), "tags": real(s, p)["tags"][::-1]},
    )  # the generator moved on after the crash
    changes = sync_short_text(conn, apply=True, now=NOW)
    assert [c.status for c in changes] == ["changed"] and _converged(plan, conn)


def test_a_skipped_short_that_cannot_compose_does_not_refuse_the_live_one(tmp_path, monkeypatch):
    from gomoveshift.pipeline.social_generator import SocialGenerator

    real = SocialGenerator.generate_post

    def post(self, i, weather=None, **kw):
        if kw.get("title_facts") is None:
            raise ValueError("legacy composition raised")
        return real(self, i, weather, **kw)

    monkeypatch.setattr(SocialGenerator, "generate_post", post)
    plan, conn = _world(tmp_path, [_short(1), _short(2)])
    conn.execute("UPDATE deliveries SET status = 'published' WHERE short_index = 1")
    conn.commit()
    changes = sync_short_text(conn, apply=False, now=NOW)
    assert [c.status for c in changes] == ["changed"] and changes[0].short_index == 2


def test_two_rewritten_neighbours_cannot_name_the_same_place(tmp_path, monkeypatch):
    real = qst._generate_lf

    def only_alone(view, *a, **kw):
        if len([s for s in view if not s.get("_discard")]) > 1:
            raise ValueError("ask each Short alone")
        return real(view, *a, **kw)

    monkeypatch.setattr(qst, "_generate_lf", only_alone)
    second = _short(2)
    second["payloads"]["youtube"]["description"] = "A quiet road. #Shorts"  # names no place
    plan, conn = _world(tmp_path, [_short(1), second])
    changes = sync_short_text(conn, apply=False, now=NOW)
    # asked alone, the second still sees the title the first just took, so it names its other
    # place instead of being refused for the repeat
    assert [c.status for c in changes] == ["changed", "changed"]
    assert changes[0].new_title.split(" #")[0] != changes[1].new_title.split(" #")[0]


def test_a_short_asked_alone_in_the_fallback_gets_no_no_alternative_exemption(
    tmp_path, monkeypatch
):
    real = qst._generate_lf

    def only_alone(view, *a, **kw):
        if len([s for s in view if not s.get("_discard")]) > 1:
            raise ValueError("ask each Short alone")
        made = real(view, *a, **kw)
        for post in made:  # the generator "sanctions" every repeat
            post.place_variety = {
                k: {"reason": "no_alternative", "body": "Ashby County"}
                for k in ("title", "description")
            }
        return made

    monkeypatch.setattr(qst, "_generate_lf", only_alone)
    from gomoveshift.pipeline import social_generator

    # one measured place, so the second cannot steer off the title the first just took
    monkeypatch.setattr(
        social_generator, "measured_places", lambda *a, **k: [("county", "Ashby County")]
    )
    second = _short(2)
    second["payloads"]["youtube"]["description"] = "A quiet road. #Shorts"  # names no place
    plan, conn = _world(tmp_path, [_short(1), second])
    changes = sync_short_text(conn, apply=False, now=NOW)
    # its seed was the predecessor's STORED places, so the sanction says nothing about the
    # predecessor's NEW ones: the clash stands
    assert [c.status for c in changes] == ["changed", "refused"]
    assert "place repeats" in changes[1].reason


def _only_alone_sanctioned(monkeypatch):
    """The generator refuses a multi-Short ask (forcing the alone fallback) and sanctions
    every repeat with ``no_alternative``."""
    real = qst._generate_lf

    def only_alone(view, *a, **kw):
        if len([s for s in view if not s.get("_discard")]) > 1:
            raise ValueError("ask each Short alone")
        made = real(view, *a, **kw)
        for post in made:
            post.place_variety = {
                k: {"reason": "no_alternative", "body": "Ashby County"}
                for k in ("title", "description")
            }
        return made

    monkeypatch.setattr(qst, "_generate_lf", only_alone)


def test_a_short_asked_alone_keeps_the_exemption_where_its_predecessor_keeps_its_place(
    tmp_path, monkeypatch
):
    _only_alone_sanctioned(monkeypatch)
    from gomoveshift.pipeline import social_generator

    # a single-place route: the only place the generator (and the stored-text reader) can name
    monkeypatch.setattr(
        social_generator, "measured_places", lambda *a, **k: [("county", "Ashby County")]
    )
    first, second = _short(1), _short(2)
    # Short 1's STORED title and description already name the place its new text names, so
    # Short 2's alone seed (the predecessor's STORED places) was the predecessor's final ones
    # too: the sanctioned repeat is valid for each such key.
    first["payloads"]["youtube"]["title"] = "Mar 5, 1st stretch: Ashby County #Shorts"
    plan, conn = _world(tmp_path, [first, second])
    changes = sync_short_text(conn, apply=False, now=NOW)
    assert [c.status for c in changes] == ["changed", "changed"], [c.reason for c in changes]


def test_a_placeholder_media_length_is_refused_not_titled_one_second(tmp_path):
    # A 0.0 duration is a placeholder, not a measurement: the Short is refused, never given
    # a stand-in length (plan_bracket_sync used to round it up to "1 seconds").
    short = _short(1)
    short["media"]["duration_seconds"] = 0.0
    plan, conn = _world(tmp_path, [short])
    before = plan.read_bytes()
    [change] = sync_short_text(conn, apply=True, now=NOW)
    assert change.status == "refused" and "measured secs" in change.reason
    assert plan.read_bytes() == before


def test_two_long_forms_of_one_plan_cannot_both_take_the_same_title(tmp_path, monkeypatch):
    # One title ledger across the plan's LFs, as Phase D keeps one per drive: each LF's Short
    # measured only Ashby County, and only one of them may be titled with it.
    from gomoveshift.pipeline import social_generator

    monkeypatch.setattr(
        social_generator, "measured_places", lambda *a, **k: [("county", "Ashby County")]
    )
    second = _short(2, hours_ahead=96, lf_index=2)
    (tmp_path / f"{PROJECT}_lf02_youtube.json").write_text(json.dumps({"location": REGION}))
    plan, conn = _world(tmp_path, [_short(1), second])
    changes = sync_short_text(conn, apply=True, now=NOW)
    titles = [_youtube(plan, n).payloads.youtube.title.split(" #", 1)[0] for n in (1, 2)]
    assert len(set(titles)) == 2, (titles, [c.reason for c in changes])
