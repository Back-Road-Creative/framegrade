"""``relink_replaced_lf``: the NEW id reaches queued first comments and the drive sidecars."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from gomoveshift.pipeline.drive_phases.phase_d import _serialize_publish_plan
from gomoveshift.pipeline.pending_journal import PendingEntry, load_pending, save_pending
from gomoveshift.publishing import replace_relink
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
from gomoveshift.publishing.queue import (
    FIRST_COMMENT_MAX_ATTEMPTS,
    enqueue,
    open_queue_db,
)
from gomoveshift.publishing.replace_relink import relink_replaced_lf

PROJECT = "nctopa-2026-03-05-rec0063"
OLD, NEW = "oLdVideo_01", "nEwVideo-02"
OTHER = "otherVid_99"
NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
MAP = {OLD: NEW}


def _comment(vid):
    return f"\U0001f3ac Watch the full drive: https://youtu.be/{vid}"


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "q")
    yield c
    c.close()


def _row(conn, idx, vid=OLD, status="queued", posted=None, platform="youtube", project=PROJECT):
    did = enqueue(
        conn, project, idx, platform, f"h{idx}{platform}", first_comment_text=_comment(vid)
    )
    conn.execute(
        "UPDATE deliveries SET status=?, first_comment_posted_at=? WHERE id=?",
        (status, posted, did),
    )
    conn.commit()
    return did


def _text(conn, did):
    return conn.execute("SELECT first_comment_text FROM deliveries WHERE id=?", (did,)).fetchone()[
        0
    ]


def _short(index, parent, description, media="/v/x.mp4"):
    return ShortPublishRecord(
        project_name=PROJECT,
        lf_index=1,
        short_index=index,
        media=MediaFingerprint(media, 1234, 21.5, 1080, 1920, "h264", "a" * 64),
        schedule=ScheduleWindow(NOW, "America/Chicago"),
        recording_date="2026-03-05",
        sampled_regions=("driftless",),
        parent_lf_video_id=parent,
        parent_lf_publish_at=NOW,
        payloads=PlatformPayloads(youtube=YouTubePayload("t", description, ("a",), "2")),
        consent_hash="c",
        reviewed_plan_hash=f"HASH{index}",
        routes=RouteSelection(selected=("youtube",)),
    )


def _plan_path(out):
    return out / f"{PROJECT}_publish_plan.json"


def _drive(tmp_path):
    out = tmp_path / "drive"
    out.mkdir()
    (out / "related_video_worklist_state.json").write_text(
        json.dumps([f'RELATED VIDEO: "A" → https://youtu.be/{OLD}', f"x https://youtu.be/{OTHER}"])
    )
    plan = PublishPlan(
        shorts=(
            _short(1, OLD, f"See https://youtu.be/{OLD} now", media=f"/v/{OLD}.mp4"),
            _short(2, OTHER, "No link here"),
        )
    )
    _plan_path(out).write_text(_serialize_publish_plan(plan), encoding="utf-8")
    save_pending(
        out / "pending_journal.json",
        [
            PendingEntry(OLD, "2026-12-01T00:00:00Z", "t", "10:00", "2026-03-05", "/s/j.md", 1),
            PendingEntry(OTHER, "2026-12-02T00:00:00Z", "t2", "10:00", "2026-03-06", "/s/j.md", 2),
        ],
    )
    return out


def test_queued_rows_rewritten(conn, tmp_path):
    a, b = _row(conn, 1), _row(conn, 2, platform="instagram")
    keep = _row(conn, 3, vid=OTHER)
    report = relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=NOW)
    assert [r.delivery_id for r in report.queue_rewritten] == [a, b]
    assert _text(conn, a) == _comment(NEW) == _text(conn, b)
    assert _text(conn, keep) == _comment(OTHER)
    assert report.queue_skipped == []


def test_other_project_untouched(conn, tmp_path):
    other = _row(conn, 1, project="someone-else")
    relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=NOW)
    assert _text(conn, other) == _comment(OLD)


@pytest.mark.parametrize(
    "status,posted,reason",
    [
        ("queued", "2026-11-21T00:00:00+00:00", "comment already live"),
        ("published", "2026-11-21T00:00:00+00:00", "comment already live"),
        ("cancelled", None, "status cancelled"),
        ("failed_permanent", None, "status failed_permanent"),
        ("missing_remote", None, "status missing_remote"),
    ],
)
def test_frozen_rows_skipped_and_reported(conn, tmp_path, status, posted, reason):
    did = _row(conn, 1, status=status, posted=posted)
    report = relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=NOW)
    assert _text(conn, did) == _comment(OLD)
    assert report.queue_rewritten == []
    (skip,) = report.queue_skipped
    assert skip.delivery_id == did and reason in skip.reason and skip.status == status
    if posted:
        assert any(
            "links OLD" in item and f"delivery {did}" in item for item in report.operator_items
        )


@pytest.mark.parametrize("status", ["published", "awaiting_manual_publish"])
def test_unposted_rows_are_rewritten_whatever_their_live_status(conn, tmp_path, status):
    did = _row(conn, 1, status=status)
    report = relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=NOW)
    assert _text(conn, did) == _comment(NEW)
    assert [r.delivery_id for r in report.queue_rewritten] == [did]


@pytest.mark.parametrize(
    "sets,reason",
    [
        (f"first_comment_attempts={FIRST_COMMENT_MAX_ATTEMPTS}", "attempts"),
        ("lease_owner='w', lease_expires_at='2026-11-22T12:05:00+00:00'", "lease"),
        ("lease_owner='w', lease_expires_at='2026-11-22T11:55:00+00:00'", None),  # expired
    ],
)
def test_an_exhausted_or_live_leased_row_is_frozen(conn, tmp_path, sets, reason):
    did = _row(conn, 1)
    conn.execute(f"UPDATE deliveries SET {sets} WHERE id=?", (did,))  # noqa: S608
    conn.commit()
    report = relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=NOW)
    assert _text(conn, did) == _comment(OLD if reason else NEW)
    assert not reason or reason in report.queue_skipped[0].reason


def test_a_row_that_moved_queued_to_published_mid_run_is_rewritten_on_rerun(
    conn, tmp_path, monkeypatch
):
    did = _row(conn, 1)
    real = replace_relink._queue.rewrite_unposted_first_comment

    def racing(c, delivery_id, old, new, now):
        c.execute("UPDATE deliveries SET status='published' WHERE id=?", (delivery_id,))
        c.execute("UPDATE deliveries SET first_comment_text=? WHERE id=?", (old + " ", delivery_id))
        c.commit()
        return real(c, delivery_id, old, new, now)

    monkeypatch.setattr(replace_relink._queue, "rewrite_unposted_first_comment", racing)
    first = relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=NOW)
    assert "rerun" in first.queue_skipped[0].reason
    monkeypatch.undo()
    conn.execute("UPDATE deliveries SET first_comment_text=? WHERE id=?", (_comment(OLD), did))
    conn.commit()
    second = relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=NOW)
    assert [r.status for r in second.queue_rewritten] == ["published"]
    assert _text(conn, did) == _comment(NEW)


def test_id_inside_a_longer_token_is_not_touched(conn, tmp_path):
    did = _row(conn, 1, vid=OLD + "X")
    report = relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=NOW)
    assert _text(conn, did) == _comment(OLD + "X")
    assert report.queue_rewritten == [] and report.queue_skipped == []


def test_sidecars_rewritten(conn, tmp_path):
    out = _drive(tmp_path)
    report = relink_replaced_lf(PROJECT, out, MAP, conn=conn, now=NOW)
    # M2: the dedup record is never rewritten; the NEW line reaches the operator checklist.
    state = json.loads((out / "related_video_worklist_state.json").read_text())
    assert f'RELATED VIDEO: "A" → https://youtu.be/{OLD}' in state
    assert f'RELATED VIDEO: "A" → https://youtu.be/{NEW}' in state
    checklist = (out / "operator_checklist.txt").read_text().splitlines()
    assert f'RELATED VIDEO: "A" → https://youtu.be/{NEW}' in checklist
    assert f"SUPERSEDED: {OLD} replaced by {NEW}" in checklist
    plan = json.loads(_plan_path(out).read_text())
    assert plan["shorts"][0]["parent_lf_video_id"] == NEW
    assert (
        plan["shorts"][0]["payloads"]["youtube"]["description"] == f"See https://youtu.be/{NEW} now"
    )
    assert plan["shorts"][0]["media"]["path"] == f"/v/{OLD}.mp4"  # a media path is never touched
    assert plan["shorts"][1]["parent_lf_video_id"] == OTHER
    assert load_publish_plan(_plan_path(out)).shorts[0].parent_lf_video_id == NEW
    ids = [e.youtube_id for e in load_pending(out / "pending_journal.json")]
    assert ids == [NEW, OTHER]
    counts = {f.path.rsplit("/", 1)[1]: f.replacements for f in report.files}
    assert counts == {
        "related_video_worklist_state.json": 1,
        f"{PROJECT}_publish_plan.json": 2,
        "pending_journal.json": 1,
    }
    pending = [f for f in report.files if f.path.endswith("pending_journal.json")][0]
    assert "no sent flag" in pending.note
    assert not list(out.glob("*.tmp")) and not list(out.glob(".*.tmp"))
    assert not list(out.glob("*.relink-tmp"))


def test_a_url_context_is_required_for_a_text_rewrite(conn, tmp_path):
    out = _drive(tmp_path)
    raw = json.loads(_plan_path(out).read_text())
    raw["shorts"][1]["payloads"]["youtube"]["description"] = (
        f"{OLD} watch?v={OLD} /shorts/{OLD} https://youtu.be/{OLD}.mp4x notes {OLD}"
    )
    _plan_path(out).write_text(json.dumps(raw))
    relink_replaced_lf(PROJECT, out, MAP, conn=conn, now=NOW)
    desc = json.loads(_plan_path(out).read_text())["shorts"][1]["payloads"]["youtube"][
        "description"
    ]
    assert desc == f"{OLD} watch?v={NEW} /shorts/{NEW} https://youtu.be/{NEW}.mp4x notes {OLD}"


def test_a_plan_edited_between_read_and_replace_is_not_overwritten(conn, tmp_path, monkeypatch):
    out = _drive(tmp_path)
    real = replace_relink.load_publish_plan
    edited = {}

    def racing(path):
        text = _plan_path(out).read_text().replace("No link here", "Edited by hand")
        _plan_path(out).write_text(text)
        edited["bytes"] = text.encode()
        return real(path)

    monkeypatch.setattr(replace_relink, "load_publish_plan", racing)
    did = _row(conn, 1)
    report = relink_replaced_lf(PROJECT, out, MAP, conn=conn, now=NOW)
    assert _plan_path(out).read_bytes() == edited["bytes"]
    assert any("changed" in e for e in report.errors)
    assert not list(out.glob("*.relink-tmp"))
    # a partial report: the other stores still ran
    assert _text(conn, did) == _comment(NEW)
    assert [e.youtube_id for e in load_pending(out / "pending_journal.json")] == [NEW, OTHER]


def test_a_plan_that_fails_validation_is_never_written(conn, tmp_path):
    out = _drive(tmp_path)
    raw = json.loads(_plan_path(out).read_text())
    raw["shorts"][0]["payloads"]["youtube"]["title"] = "t #Tag"  # hashtag absent from description
    _plan_path(out).write_text(json.dumps(raw))
    before = _plan_path(out).read_bytes()
    report = relink_replaced_lf(PROJECT, out, MAP, conn=conn, now=NOW)
    assert _plan_path(out).read_bytes() == before
    assert any("publish_plan" in e for e in report.errors)


def test_one_store_failing_leaves_a_partial_report(conn, tmp_path):
    out = _drive(tmp_path)
    (out / "related_video_worklist_state.json").write_text("{not json")
    did = _row(conn, 1)
    report = relink_replaced_lf(PROJECT, out, MAP, conn=conn, now=NOW)
    assert any("related_video_worklist_state" in e for e in report.errors)
    assert _text(conn, did) == _comment(NEW)
    assert json.loads(_plan_path(out).read_text())["shorts"][0]["parent_lf_video_id"] == NEW


def test_pending_entries_collapse_onto_an_existing_new_entry(conn, tmp_path):
    out = _drive(tmp_path)
    save_pending(
        out / "pending_journal.json",
        [
            PendingEntry(OLD, "2026-12-01T00:00:00Z", "t", "10:00", "2026-03-05", "/s/j.md", 1),
            PendingEntry(NEW, "2026-12-01T00:00:00Z", "t", "10:00", "2026-03-05", "/s/j.md", 1),
            PendingEntry(OLD, "2026-12-01T00:00:00Z", "t", "10:00", "2026-03-05", "/s/k.md", 1),
        ],
    )
    relink_replaced_lf(PROJECT, out, MAP, conn=conn, now=NOW)
    got = [(e.journey_path, e.youtube_id) for e in load_pending(out / "pending_journal.json")]
    assert got == [("/s/j.md", NEW), ("/s/k.md", NEW)]


def test_missing_sidecars_are_reported_absent(conn, tmp_path):
    report = relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=NOW)
    assert [f.note for f in report.files] == ["absent"] * 3 and report.changes == 0


def test_idempotent(conn, tmp_path):
    out = _drive(tmp_path)
    _row(conn, 1)
    first = relink_replaced_lf(PROJECT, out, MAP, conn=conn, now=NOW)
    assert first.changes == 5
    snap = {p.name: p.read_bytes() for p in out.iterdir()}
    second = relink_replaced_lf(PROJECT, out, MAP, conn=conn, now=NOW)
    assert second.changes == 0 and second.queue_skipped == []
    assert {p.name: p.read_bytes() for p in out.iterdir()} == snap


def test_dry_run_reports_but_writes_nothing(conn, tmp_path):
    out = _drive(tmp_path)
    did = _row(conn, 1)
    snap = {p.name: p.read_bytes() for p in out.iterdir()}
    report = relink_replaced_lf(PROJECT, out, MAP, conn=conn, now=NOW, dry_run=True)
    assert report.dry_run and report.changes == 5
    assert all(not f.written for f in report.files) and not report.queue_rewritten[0].written
    assert _text(conn, did) == _comment(OLD)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == snap


@pytest.mark.parametrize(
    "bad",
    [{}, {OLD: OLD}, {"": NEW}, {OLD: ""}, {OLD: NEW, NEW: "zzz"}],
    ids=["empty", "same", "empty-old", "empty-new", "chain"],
)
def test_refusals_write_nothing(conn, tmp_path, bad):
    out = _drive(tmp_path)
    did = _row(conn, 1)
    snap = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(ValueError):
        relink_replaced_lf(PROJECT, out, bad, conn=conn, now=NOW)
    assert _text(conn, did) == _comment(OLD)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == snap


def test_two_ids_swap_independently(conn, tmp_path):
    a, b = _row(conn, 1), _row(conn, 2, vid=OTHER)
    relink_replaced_lf(PROJECT, tmp_path, {OLD: NEW, OTHER: "thirdVid_03"}, conn=conn, now=NOW)
    assert _text(conn, a) == _comment(NEW) and _text(conn, b) == _comment("thirdVid_03")
