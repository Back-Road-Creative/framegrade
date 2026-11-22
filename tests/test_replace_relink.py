"""``relink_replaced_lf``: a replaced long-form upload's NEW id reaches the queued first comments
and the drive's sidecars; posted / terminal / manual rows are skipped and reported. Synthetic data."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from gomoveshift.pipeline.pending_journal import PendingEntry, load_pending, save_pending
from gomoveshift.publishing.queue import enqueue, open_queue_db
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


def _drive(tmp_path):
    out = tmp_path / "drive"
    out.mkdir()
    (out / "related_video_worklist_state.json").write_text(
        json.dumps([f'RELATED VIDEO: "A" → https://youtu.be/{OLD}', f"x https://youtu.be/{OTHER}"])
    )
    plan = {
        "shorts": [
            {
                "short_index": 1,
                "parent_lf_video_id": OLD,
                "payloads": {"youtube": {"description": f"See https://youtu.be/{OLD} now"}},
            },
            {"short_index": 2, "parent_lf_video_id": OTHER, "payloads": {}},
        ]
    }
    (out / f"{PROJECT}_publish_plan.json").write_text(json.dumps(plan))
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
        ("queued", "2026-11-21T00:00:00+00:00", "already posted"),
        ("published", None, "status published"),
        ("awaiting_manual_publish", None, "status awaiting_manual_publish"),
        ("cancelled", None, "status cancelled"),
        ("failed_permanent", None, "status failed_permanent"),
    ],
)
def test_frozen_rows_skipped_and_reported(conn, tmp_path, status, posted, reason):
    did = _row(conn, 1, status=status, posted=posted)
    report = relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=NOW)
    assert _text(conn, did) == _comment(OLD)
    assert report.queue_rewritten == []
    (skip,) = report.queue_skipped
    assert skip.delivery_id == did and reason in skip.reason and skip.status == status


def test_id_inside_a_longer_token_is_not_touched(conn, tmp_path):
    did = _row(conn, 1, vid=OLD + "X")
    report = relink_replaced_lf(PROJECT, tmp_path, MAP, conn=conn, now=NOW)
    assert _text(conn, did) == _comment(OLD + "X")
    assert report.queue_rewritten == [] and report.queue_skipped == []


def test_sidecars_rewritten(conn, tmp_path):
    out = _drive(tmp_path)
    report = relink_replaced_lf(PROJECT, out, MAP, conn=conn, now=NOW)
    state = json.loads((out / "related_video_worklist_state.json").read_text())
    assert state == [f'RELATED VIDEO: "A" → https://youtu.be/{NEW}', f"x https://youtu.be/{OTHER}"]
    plan = json.loads((out / f"{PROJECT}_publish_plan.json").read_text())
    assert plan["shorts"][0]["parent_lf_video_id"] == NEW
    assert (
        plan["shorts"][0]["payloads"]["youtube"]["description"] == f"See https://youtu.be/{NEW} now"
    )
    assert plan["shorts"][1]["parent_lf_video_id"] == OTHER
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
