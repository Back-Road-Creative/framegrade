"""``relink_replaced_lf``: the NEW id reaches queued first comments and the drive sidecars."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

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
NEW_AT = "2026-11-22T12:00:00+00:00"  # the replacement goes public at NOW
SLOT = "2026-11-22T13:00:00+00:00"  # a short's slot after it: the parent is public by then
MAP = {OLD: (NEW, NEW_AT)}


def _comment(vid):
    return f"\U0001f3ac Watch the full drive: https://youtu.be/{vid}"


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "q")
    yield c
    c.close()


def _row(
    conn, idx, vid=OLD, status="queued", posted=None, platform="youtube", project=PROJECT, slot=SLOT
):
    did = enqueue(
        conn,
        project,
        idx,
        platform,
        f"h{idx}{platform}",
        scheduled_at=slot,
        first_comment_text=_comment(vid),
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
        ("lease_owner='w', lease_expires_at='{+1h}'", "lease"),
        ("lease_owner='w', lease_expires_at='{-1h}'", None),  # expired
    ],
)
def test_an_exhausted_or_live_leased_row_is_frozen(conn, tmp_path, sets, reason):
    # A lease is live by this process's clock (not the caller's ``now``): the live one runs out
    # in 9999, the expired one in 2000, so no wall clock is read.
    sets = sets.replace("{+1h}", datetime(9999, 1, 1, tzinfo=timezone.utc).isoformat()).replace(
        "{-1h}", datetime(2000, 1, 1, tzinfo=timezone.utc).isoformat()
    )
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
    [
        {},
        {OLD: (OLD, NEW_AT)},
        {"": (NEW, NEW_AT)},
        {OLD: ("", NEW_AT)},
        {OLD: (NEW, NEW_AT), NEW: ("zzz", NEW_AT)},
        {OLD: (NEW, "")},
        {OLD: (NEW, "not a time")},
        {OLD: (NEW, "2026-11-22T12:00:00")},
    ],
    ids=["empty", "same", "empty-old", "empty-new", "chain", "no-time", "bad-time", "unzoned"],
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
    relink_replaced_lf(
        PROJECT, tmp_path, {OLD: (NEW, NEW_AT), OTHER: ("thirdVid_03", NEW_AT)}, conn=conn, now=NOW
    )
    assert _text(conn, a) == _comment(NEW) and _text(conn, b) == _comment("thirdVid_03")


def test_a_live_comment_left_by_a_mid_post_rewrite_is_an_operator_item(conn, tmp_path):
    # The drain recorded the OLD text as live (posted) with the stale error; relink reports it.
    did = _row(conn, 1, posted="2026-11-22T11:00:00+00:00")
    conn.execute(
        "UPDATE deliveries SET first_comment_error=?, first_comment_correction=? WHERE id=?",
        (
            f"stale comment live: links {OLD}",
            json.dumps([{"live_id": "c1", "live_text": _comment(OLD), "action": "edit",
                         "to": _comment(NEW)}]),
            did,
        ),
    )  # fmt: skip
    conn.commit()
    report = relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=NOW)
    assert _text(conn, did) == _comment(OLD)
    (item,) = [i for i in report.operator_items if f"delivery {did}" in i]
    assert "links OLD" in item and "(stale comment live: links " in item
    assert "the drain corrects it" in item and "by hand" not in item


def test_a_posted_comment_with_no_stale_note_is_still_a_hand_fix(conn, tmp_path):
    did = _row(conn, 1, posted="2026-11-22T11:00:00+00:00")
    report = relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=NOW)
    (item,) = [i for i in report.operator_items if f"delivery {did}" in i]
    assert "fix it by hand" in item and "(stale" not in item


def test_a_rewritten_published_row_is_listed_for_the_drain_to_verify(conn, tmp_path):
    queued = _row(conn, 1)
    published = _row(conn, 2, status="published", platform="instagram")
    conn.execute("UPDATE deliveries SET platform_post_id='media1' WHERE id=?", (published,))
    conn.commit()
    report = relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=NOW)
    (note,) = report.drain_will_verify
    assert f"delivery {published}" in note and "drain" in note
    assert not any(f"delivery {queued}" in n for n in report.drain_will_verify)
    dry = relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=NOW, dry_run=True)
    assert dry.drain_will_verify == []  # nothing left to rewrite on the rerun


def test_a_rewritten_plan_asks_for_gate_c_d_again(conn, tmp_path):
    out = _drive(tmp_path)
    report = relink_replaced_lf(PROJECT, out, MAP, conn=conn, now=NOW)
    assert any("re-confirm Gate C/D" in i for i in report.operator_items)
    (tmp_path / "d2").mkdir()
    dry_out = _drive(tmp_path / "d2")
    dry = relink_replaced_lf(PROJECT, dry_out, MAP, conn=conn, now=NOW, dry_run=True)
    assert any("re-confirm Gate C/D" in i for i in dry.operator_items)


def test_no_plan_change_means_no_gate_c_d_item(conn, tmp_path):
    report = relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=NOW)
    assert not any("Gate C/D" in i for i in report.operator_items)


def test_rows_of_other_projects_naming_the_old_id_are_reported_not_rewritten(conn, tmp_path):
    mine = _row(conn, 1)
    other = _row(conn, 2, project="other-project")
    longer = _row(conn, 3, vid=OLD + "X", project="other-project")
    report = relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=NOW)
    assert _text(conn, mine) == _comment(NEW)
    assert _text(conn, other) == _comment(OLD)
    items = [i for i in report.operator_items if "other-project" in i]
    assert len(items) == 1 and f"delivery {other}" in items[0]
    assert not any(f"delivery {longer}" in i for i in report.operator_items)
    assert not any(f"delivery {mine}" in i for i in report.operator_items)  # own project excluded


def test_a_failing_other_project_report_is_recorded_not_raised(tmp_path, monkeypatch):
    from gomoveshift.publishing import replace_relink

    conn = open_queue_db(tmp_path / "q")

    def boom(*a):
        raise RuntimeError("boom")

    monkeypatch.setattr(replace_relink, "_report_other_projects", boom)
    report = relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=NOW)
    assert any("other projects" in e and "boom" in e for e in report.errors)


def test_the_lease_precheck_uses_this_process_clock_not_the_callers_now(conn, tmp_path):
    from datetime import datetime, timezone

    did = _row(conn, 1)
    conn.execute(
        "UPDATE deliveries SET lease_owner='w', lease_expires_at='2099-01-01T00:00:00+00:00' "
        "WHERE id=?",
        (did,),
    )
    conn.commit()
    future = datetime(2100, 1, 1, tzinfo=timezone.utc)
    report = relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=future)
    assert any("lease held" in s.reason for s in report.queue_skipped)


# --- #517 parent-public gate: a relink never links a video that is still private -------------

EARLY = "2026-11-22T11:00:00+00:00"  # a slot before the replacement goes public (NEW_AT)
PLATFORMS = ["youtube", "facebook", "instagram"]


def _row_state(conn, did):
    return conn.execute("SELECT * FROM deliveries WHERE id=?", (did,)).fetchone()


def test_a_bare_string_id_map_value_is_a_type_error(conn, tmp_path):
    did = _row(conn, 1)
    with pytest.raises(TypeError):
        relink_replaced_lf(PROJECT, tmp_path, {OLD: NEW}, conn=conn, now=NOW)
    assert _text(conn, did) == _comment(OLD)


@pytest.mark.parametrize("platform", PLATFORMS)
def test_an_unposted_row_is_not_relinked_to_a_private_video(conn, tmp_path, platform):
    did = _row(conn, 1, platform=platform, slot=EARLY)
    report = relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=NOW)
    row = _row_state(conn, did)
    assert row["first_comment_text"] is None  # dropped, the way enqueue drops a private link
    assert row["first_comment_posted_at"] is None
    assert [r.delivery_id for r in report.queue_rewritten] == [did]
    (item,) = [i for i in report.operator_items if f"delivery {did}" in i]
    assert "not public" in item and EARLY in item


def test_the_gate_is_per_row_slot(conn, tmp_path):
    early = _row(conn, 1, slot=EARLY)
    late = _row(conn, 2, slot=SLOT)
    unknown = _row(conn, 3, slot=None)
    relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=NOW)
    assert _text(conn, early) is None
    assert _text(conn, late) == _comment(NEW)
    assert _text(conn, unknown) is None  # an unknown slot cannot be ordered: never links


@pytest.mark.parametrize("platform", PLATFORMS)
def test_a_passing_gate_still_relinks(conn, tmp_path, platform):
    did = _row(conn, 1, platform=platform, slot=NEW_AT)  # public exactly at the slot
    report = relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=NOW)
    assert _text(conn, did) == _comment(NEW)
    assert not any("not public" in i for i in report.operator_items)


@pytest.mark.parametrize("platform", PLATFORMS)
def test_a_live_comment_with_a_pending_edit_gets_a_delete_not_an_edit(conn, tmp_path, platform):
    did = _row(conn, 1, platform=platform, posted="2026-11-22T10:00:00+00:00", slot=EARLY)
    conn.execute(
        "UPDATE deliveries SET first_comment_error=?, first_comment_correction=? WHERE id=?",
        (
            "stale comment live: links zeroVid_00",
            json.dumps([{"live_id": "c1", "live_text": _comment("zeroVid_00"), "action": "edit",
                         "to": _comment(OLD)}]),
            did,
        ),
    )  # fmt: skip
    conn.commit()
    report = relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=NOW)
    row = _row_state(conn, did)
    (fix,) = json.loads(row["first_comment_correction"])
    assert fix == {"live_id": "c1", "live_text": _comment("zeroVid_00"), "action": "delete"}
    assert row["first_comment_text"] is None  # no intended text: the delete is allowed to run
    assert any("not public" in i and f"delivery {did}" in i for i in report.operator_items)


def test_a_pending_edit_is_rewritten_with_the_new_parent_time_when_the_gate_passes(conn, tmp_path):
    did = _row(conn, 1, posted="2026-11-22T10:00:00+00:00")
    conn.execute(
        "UPDATE deliveries SET first_comment_correction=? WHERE id=?",
        (
            json.dumps([{"live_id": "c1", "live_text": _comment("zeroVid_00"), "action": "edit",
                         "to": _comment(OLD)}]),
            did,
        ),
    )  # fmt: skip
    conn.commit()
    relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=NOW)
    (fix,) = json.loads(_row_state(conn, did)["first_comment_correction"])
    assert fix["action"] == "edit" and fix["to"] == _comment(NEW) and fix["parent_at"] == NEW_AT


def test_the_plan_gets_the_new_parent_publish_time(conn, tmp_path):
    out = _drive(tmp_path)
    relink_replaced_lf(PROJECT, out, MAP, conn=conn, now=NOW)
    plan = json.loads(_plan_path(out).read_text())
    assert plan["shorts"][0]["parent_lf_video_id"] == NEW
    assert plan["shorts"][0]["parent_lf_publish_at"] == NEW_AT
    assert plan["shorts"][1]["parent_lf_video_id"] == OTHER  # untouched
    assert (
        plan["shorts"][1]["parent_lf_publish_at"]
        == json.loads(_serialize_publish_plan(PublishPlan(shorts=(_short(2, OTHER, "x"),))))[
            "shorts"
        ][0]["parent_lf_publish_at"]
    )
