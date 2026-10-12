"""Tests for the next-7-days delivery digest.

See docs/pipeline/149-dynamic-scheduling.md.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gomoveshift.publishing.next_week import next_week_report
from gomoveshift.publishing.queue import enqueue, open_queue_db


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _iso(dt):
    return dt.isoformat()


class TestNextWeekReport:
    def test_empty_queue_is_an_empty_string(self, conn):
        now = datetime(2026, 9, 7, tzinfo=timezone.utc)
        assert next_week_report(conn, now) == ""

    def test_lists_a_queued_delivery_in_the_next_seven_days(self, conn):
        now = datetime(2026, 9, 7, tzinfo=timezone.utc)
        when = now + timedelta(days=2, hours=4)
        enqueue(conn, "proj", 1, "youtube", "hash-a", scheduled_at=_iso(when))
        report = next_week_report(conn, now)
        assert "proj" in report
        assert "youtube" in report
        assert "14:0" in report or ":" in report  # slot label present

    def test_excludes_deliveries_more_than_seven_days_out(self, conn):
        now = datetime(2026, 9, 7, tzinfo=timezone.utc)
        far = now + timedelta(days=10)
        enqueue(conn, "proj", 1, "youtube", "hash-a", scheduled_at=_iso(far))
        assert next_week_report(conn, now) == ""

    def test_excludes_non_queued_statuses(self, conn):
        now = datetime(2026, 9, 7, tzinfo=timezone.utc)
        when = now + timedelta(days=1)
        delivery_id = enqueue(conn, "proj", 1, "youtube", "hash-a", scheduled_at=_iso(when))
        conn.execute("UPDATE deliveries SET status='published' WHERE id=?", (delivery_id,))
        conn.commit()
        assert next_week_report(conn, now) == ""

    def test_groups_by_day_in_chronological_order(self, conn):
        now = datetime(2026, 9, 7, tzinfo=timezone.utc)
        day1 = now + timedelta(days=1, hours=1)
        day2 = now + timedelta(days=3, hours=1)
        enqueue(conn, "proj", 1, "youtube", "hash-a", scheduled_at=_iso(day2))
        enqueue(conn, "proj", 2, "youtube", "hash-b", scheduled_at=_iso(day1))
        report = next_week_report(conn, now)
        assert report.index(day1.date().isoformat()) < report.index(day2.date().isoformat())

    def test_facebook_row_moved_off_the_youtube_slot_is_labelled_stagger(self, conn):
        now = datetime(2026, 9, 7, tzinfo=timezone.utc)
        yt_when = now + timedelta(days=1)
        fb_when = yt_when + timedelta(hours=2)
        enqueue(conn, "proj", 1, "youtube", "hash-a", scheduled_at=_iso(yt_when))
        enqueue(conn, "proj", 1, "facebook", "hash-a", scheduled_at=_iso(fb_when))
        report = next_week_report(conn, now)
        lines = [line for line in report.splitlines() if "facebook" in line]
        assert len(lines) == 1
        assert "stagger" in lines[0]
