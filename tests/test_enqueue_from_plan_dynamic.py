"""Wiring tests: enqueue_from_plan_sidecar's order=/stagger_hours=
keywords, and the no-double-publish acceptance test.
"""

from __future__ import annotations

import json

import pytest

from gomoveshift.publishing.enqueue_from_plan import enqueue_from_plan_sidecar
from gomoveshift.publishing.queue import open_queue_db


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _record(
    *,
    project_name="p",
    short_index=10,
    selected=("youtube", "instagram", "facebook"),
    plan_hash="hash-a",
    scheduled_at="2026-09-07T16:00:00+00:00",
    by_platform=None,
):
    schedule = {"scheduled_at_utc": scheduled_at, "display_timezone": "America/New_York"}
    if by_platform:
        schedule["scheduled_at_utc_by_platform"] = by_platform
    return {
        "project_name": project_name,
        "lf_index": 1,
        "short_index": short_index,
        "reviewed_plan_hash": plan_hash,
        "routes": {"selected": list(selected)},
        "schedule": schedule,
    }


def _write_sidecar(tmp_path, records, name="p_publish_plan.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"shorts": records}), encoding="utf-8")
    return path


class TestOrderKeyword:
    def test_order_reverses_insertion_order(self, tmp_path, conn):
        records = [_record(short_index=1), _record(short_index=2)]
        sidecar = _write_sidecar(tmp_path, records)

        def reverse(capped):
            return list(reversed(capped))

        enqueue_from_plan_sidecar(conn, sidecar, order=reverse)
        rows = conn.execute(
            "SELECT short_index FROM deliveries WHERE platform='youtube' ORDER BY id"
        ).fetchall()
        assert [r["short_index"] for r in rows] == [2, 1]

    def test_order_does_not_change_which_shorts_are_selected(self, tmp_path, conn):
        records = [_record(short_index=1), _record(short_index=2), _record(short_index=3)]
        sidecar = _write_sidecar(tmp_path, records)

        def reverse(capped):
            return list(reversed(capped))

        result = enqueue_from_plan_sidecar(conn, sidecar, max_shorts=2, order=reverse)
        selected = {key[1] for key in result.enqueued}
        assert selected == {1, 2}


class TestStaggerHoursKeyword:
    def test_fills_facebook_and_instagram_when_absent(self, tmp_path, conn):
        sidecar = _write_sidecar(tmp_path, [_record(scheduled_at="2026-09-07T16:00:00+00:00")])
        enqueue_from_plan_sidecar(conn, sidecar, stagger_hours=2)
        fb = conn.execute(
            "SELECT scheduled_at FROM deliveries WHERE platform='facebook'"
        ).fetchone()
        yt = conn.execute("SELECT scheduled_at FROM deliveries WHERE platform='youtube'").fetchone()
        assert fb["scheduled_at"] != yt["scheduled_at"]
        assert fb["scheduled_at"] == "2026-09-07T18:00:00+00:00"

    def test_sidecar_explicit_by_platform_wins_over_computed_stagger(self, tmp_path, conn):
        sidecar = _write_sidecar(
            tmp_path,
            [
                _record(
                    scheduled_at="2026-09-07T16:00:00+00:00",
                    by_platform={"facebook": "2026-09-07T20:00:00+00:00"},
                )
            ],
        )
        enqueue_from_plan_sidecar(conn, sidecar, stagger_hours=2)
        fb = conn.execute(
            "SELECT scheduled_at FROM deliveries WHERE platform='facebook'"
        ).fetchone()
        assert fb["scheduled_at"] == "2026-09-07T20:00:00+00:00"


class TestNoDoublePublish:
    def test_reenqueue_of_a_published_short_under_a_new_plan_hash_adds_no_row(self, tmp_path, conn):
        """Re-enqueuing a (project, short_index, platform) already
        published under a NEW plan_hash must not create a second row.

        Grounded against ``queue.enqueue`` (queue.py ~352-380): a published
        row for the 3-tuple is found and its id returned BEFORE any INSERT
        is attempted, regardless of ``plan_hash`` -- so this test pins
        existing behaviour, it does not newly gate it (queue.py is not
        touched here).
        """
        first = _write_sidecar(
            tmp_path, [_record(selected=("youtube",), plan_hash="hash-a")], name="first.json"
        )
        enqueue_from_plan_sidecar(conn, first)
        conn.execute(
            "UPDATE deliveries SET status='published' WHERE project_name='p' "
            "AND short_index=10 AND platform='youtube'"
        )
        conn.commit()

        second = _write_sidecar(
            tmp_path, [_record(selected=("youtube",), plan_hash="hash-b")], name="second.json"
        )
        enqueue_from_plan_sidecar(conn, second)

        rows = conn.execute(
            "SELECT COUNT(*) FROM deliveries WHERE project_name='p' AND short_index=10 "
            "AND platform='youtube'"
        ).fetchone()[0]
        assert rows == 1
