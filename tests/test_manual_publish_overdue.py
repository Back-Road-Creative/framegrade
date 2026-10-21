"""A TikTok inbox draft whose slot passed long ago is named, not just listed.

An inbox upload parks ``awaiting_manual_publish`` until a human taps Post in
the app. The digest listed every waiting row, but a row three days past its
slot read exactly like one due this afternoon, so nothing said "this one was
missed". ``publishing-status --validate`` now names a parked row once its slot
plus a grace window has passed, and the digest counts them.

It is a reminder for the human, not a pipeline defect: the publishing gate
must not hold other shorts because the operator has not tapped Post yet. All data is
synthetic.
"""

from __future__ import annotations

from gomoveshift.cli.handlers.pub_status import (
    MANUAL_PUBLISH_GRACE_HOURS,
    gating_problems,
    validate_restored_state,
)
from gomoveshift.publishing.queue import enqueue, open_queue_db

NOW = "2026-10-15T12:00:00+00:00"


def _parked(conn, scheduled_at, *, project="synth-coast", short_index=2):
    delivery_id = enqueue(conn, project, short_index, "tiktok", "HASH0")
    conn.execute(
        "UPDATE deliveries SET status='awaiting_manual_publish', scheduled_at=? WHERE id=?",
        (scheduled_at, delivery_id),
    )
    conn.commit()
    return delivery_id


def _overdue(conn):
    return [
        p
        for p in validate_restored_state(conn, None, now=NOW)
        if p.kind == "manual_publish_overdue"
    ]


def test_a_draft_past_its_slot_and_grace_is_named(tmp_path):
    conn = open_queue_db(tmp_path)
    late = _parked(conn, "2026-10-12T16:00:00+00:00")

    problems = _overdue(conn)
    conn.close()

    assert [(p.delivery_id, p.platform) for p in problems] == [(late, "tiktok")]
    assert "2026-10-12T16:00:00" in problems[0].detail


def test_a_draft_inside_the_grace_window_is_not_named(tmp_path):
    conn = open_queue_db(tmp_path)
    _parked(conn, "2026-10-15T09:00:00+00:00")
    _parked(conn, "2026-10-16T09:00:00+00:00", short_index=3)
    assert MANUAL_PUBLISH_GRACE_HOURS >= 12

    assert _overdue(conn) == []
    conn.close()


def test_an_unscheduled_draft_is_not_named(tmp_path):
    conn = open_queue_db(tmp_path)
    _parked(conn, None)

    assert _overdue(conn) == []
    conn.close()


def test_an_overdue_draft_never_gates_publishing(tmp_path):
    """The operator not tapping Post is not a reason to stop every other short."""
    conn = open_queue_db(tmp_path)
    _parked(conn, "2026-10-01T16:00:00+00:00")

    assert [
        p for p in gating_problems(conn, None, now=NOW) if p.kind == "manual_publish_overdue"
    ] == []
    conn.close()
