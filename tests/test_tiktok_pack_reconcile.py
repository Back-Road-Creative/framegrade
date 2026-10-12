"""Tests for ``tiktok_pack.reconcile_packed``: closing pack-parked rows from
TikTok's read-only ``video.list``.

A packed row has no ``platform_upload_id``, so ``reconcile_awaiting_manual``
cannot poll it. A row reads published only when exactly one listed video
matches its slot and caption; anything less leaves it parked. Fakes only.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from test_tiktok_pack import NOW, SLOT, _enqueue, _rec

from gomoveshift.publishing.queue import (
    AWAITING_MANUAL_PUBLISH_NEXT_ATTEMPT,
    NEXT_ATTEMPT_AWAITING_MANUAL,
    open_queue_db,
    transition,
)
from gomoveshift.publishing.tiktok_pack import RECONCILE_WINDOW, reconcile_packed

CAPTION = "Full caption #POVDriving"
LATER = SLOT + timedelta(days=1)  # "now" once the slot is well behind us


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _parked(conn, *, short_index=3, scheduled=SLOT, upload_id=None):
    delivery = _enqueue(conn, short_index=short_index, scheduled=scheduled)
    assert transition(
        conn,
        delivery,
        "queued",
        "awaiting_manual_publish",
        next_attempt_at=AWAITING_MANUAL_PUBLISH_NEXT_ATTEMPT,
        next_attempt_reason=NEXT_ATTEMPT_AWAITING_MANUAL,
        platform_upload_id=upload_id,
    )
    return delivery


def _row(conn, delivery):
    return dict(conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery,)).fetchone())


def _video(vid, *, at=SLOT, desc=CAPTION):
    return {"id": vid, "create_time": int(at.timestamp()), "video_description": desc}


class _Pages:
    """A fake ``list_videos(cursor)`` serving newest-first pages."""

    def __init__(self, *pages):
        self.pages = list(pages)
        self.cursors = []

    def __call__(self, cursor=None):
        self.cursors.append(cursor)
        index = len(self.cursors) - 1
        return self.pages[index], str(index + 1), index + 1 < len(self.pages)


def _run(conn, pages, *, records=None, now=LATER):
    records = records or {}
    return reconcile_packed(
        conn,
        list_videos=pages,
        permalink_for=lambda vid: f"https://www.tiktok.com/@gms/video/{vid}",
        record_loader=lambda project, index: records.get(index) or _rec(short_index=index),
        now=now,
    )


def test_one_matching_video_publishes_the_row_with_its_id_and_permalink(conn):
    delivery = _parked(conn)
    result = _run(conn, _Pages([_video("111")]))
    row = _row(conn, delivery)
    assert row["status"] == "published"
    assert row["platform_post_id"] == "111"
    assert row["permalink"] == "https://www.tiktok.com/@gms/video/111"
    assert result.published == 1 and result.ambiguous == 0


def test_a_description_trimmed_after_the_first_line_still_matches(conn):
    delivery = _parked(conn)
    rec = _rec(caption="First line here\n\nSecond paragraph #tags")
    _run(
        conn,
        _Pages([_video("111", desc="  FIRST   line here  ")]),
        records={3: rec},
    )
    assert _row(conn, delivery)["status"] == "published"


def test_a_video_outside_the_window_leaves_the_row_parked(conn):
    delivery = _parked(conn)
    far = SLOT + RECONCILE_WINDOW + timedelta(minutes=1)
    result = _run(conn, _Pages([_video("111", at=far)]), now=far + timedelta(days=1))
    assert _row(conn, delivery)["status"] == "awaiting_manual_publish"
    assert result.published == 0 and result.ambiguous == 0


def test_a_caption_mismatch_leaves_the_row_parked(conn):
    delivery = _parked(conn)
    _run(conn, _Pages([_video("111", desc="Something else entirely")]))
    row = _row(conn, delivery)
    assert row["status"] == "awaiting_manual_publish"
    assert row["platform_post_id"] is None


def test_two_candidate_videos_leave_the_row_parked_and_ambiguous(conn):
    delivery = _parked(conn)
    result = _run(conn, _Pages([_video("111"), _video("222")]))
    assert _row(conn, delivery)["status"] == "awaiting_manual_publish"
    assert result.published == 0
    assert result.ambiguous == 1
    assert any(str(delivery) in line for line in result.ambiguous_rows)


def test_one_video_matching_two_rows_leaves_both_parked(conn):
    a = _parked(conn, short_index=3)
    b = _parked(conn, short_index=4)
    result = _run(conn, _Pages([_video("111")]))
    assert _row(conn, a)["status"] == "awaiting_manual_publish"
    assert _row(conn, b)["status"] == "awaiting_manual_publish"
    assert result.published == 0 and result.ambiguous == 2


def test_a_video_id_already_on_another_delivery_is_never_reused(conn):
    taken = _parked(conn, short_index=8)
    assert transition(
        conn,
        taken,
        "awaiting_manual_publish",
        "published",
        platform_post_id="111",
        permalink="u",
        published_at_source="platform_reconciled",
    )
    delivery = _parked(conn)
    result = _run(conn, _Pages([_video("111")]))
    assert _row(conn, delivery)["status"] == "awaiting_manual_publish"
    assert result.published == 0


def test_paging_stops_once_a_page_reaches_back_past_the_window(conn):
    delivery = _parked(conn)
    old = SLOT - RECONCILE_WINDOW - timedelta(hours=1)
    newer = LATER - timedelta(hours=1)
    pages = _Pages([_video("900", at=newer, desc="x")], [_video("111"), _video("5", at=old)], [])
    _run(conn, pages)
    assert _row(conn, delivery)["status"] == "published"
    assert pages.cursors == [None, "1"]  # the third page is never requested


def test_inbox_rows_with_an_upload_id_are_untouched(conn):
    delivery = _parked(conn, upload_id="v_pub_file~abc")
    pages = _Pages([_video("111")])
    result = _run(conn, pages)
    assert _row(conn, delivery)["status"] == "awaiting_manual_publish"
    assert pages.cursors == []  # nothing pack-shaped: TikTok is never asked
    assert result.considered == 0


def test_no_parked_rows_means_no_api_call(conn):
    pages = _Pages([_video("111")])
    _run(conn, pages)
    assert pages.cursors == []


def test_a_slot_that_has_not_arrived_is_not_listed_for(conn):
    _parked(conn)
    pages = _Pages([_video("111")])
    _run(conn, pages, now=NOW)  # slot is a day and more ahead
    assert pages.cursors == []


def test_an_unloadable_record_is_an_error_not_a_crash(conn):
    delivery = _parked(conn)

    def loader(project, index):
        raise FileNotFoundError("sidecar gone")

    result = reconcile_packed(
        conn,
        list_videos=_Pages([_video("111")]),
        permalink_for=lambda v: f"u/{v}",
        record_loader=loader,
        now=LATER,
    )
    assert _row(conn, delivery)["status"] == "awaiting_manual_publish"
    assert len(result.errors) == 1
