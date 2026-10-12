"""A platform pause (``queue.pause_platform``) is enforced in exactly one
query today -- the normal drain's ``_select_due``. The two reconcile paths
select and act on deliveries independently of that query and, before this
change, ignored the pause entirely: a crashed-worker reconcile could still
charge an attempt (and eventually fail a row permanently) against a platform
an operator just paused, and the TikTok inbox reconcile could still poll a
paused platform's API and close a row out.

These tests pin what "honor the pause" means for each path -- see the
docstrings on ``reconcile_interrupted_attempts`` and
``reconcile_awaiting_manual`` in ``dispatch.py`` for why the two answers
differ."""

from datetime import datetime, timedelta

import pytest

from gomoveshift.publishing.dispatch import (
    reconcile_awaiting_manual,
    reconcile_interrupted_attempts,
)
from gomoveshift.publishing.queue import enqueue, open_queue_db, pause_platform, acquire_lease

NOW = "2026-08-23T00:00:00+00:00"


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _row(conn, delivery_id):
    return dict(conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone())


def _expired_lease(conn, delivery_id, *, owner="worker-a:1:dead", now=NOW, seconds=300):
    acquired_at = datetime.fromisoformat(now) - timedelta(seconds=seconds + 1)
    acquire_lease(
        conn,
        delivery_id,
        owner,
        (acquired_at + timedelta(seconds=seconds)).isoformat(),
        now=acquired_at.isoformat(),
    )


def test_a_paused_platforms_interrupted_attempt_is_not_charged(conn):
    delivery_id = enqueue(conn, "proj", 0, "facebook", "hash-a")
    _expired_lease(conn, delivery_id)
    assert pause_platform(conn, "facebook") is True

    reconciled = reconcile_interrupted_attempts(conn, now=NOW)

    row = _row(conn, delivery_id)
    assert row["attempt_count"] == 0
    assert row["status"] == "queued"
    assert reconciled == 0


def test_a_paused_platforms_stale_lease_is_still_released(conn):
    """Releasing the lease is bookkeeping, not a platform touch -- it costs
    the platform nothing and unblocks selection the moment the pause lifts."""
    delivery_id = enqueue(conn, "proj", 0, "facebook", "hash-a")
    _expired_lease(conn, delivery_id)
    assert pause_platform(conn, "facebook") is True

    reconcile_interrupted_attempts(conn, now=NOW)

    row = _row(conn, delivery_id)
    assert row["lease_owner"] is None
    assert row["lease_expires_at"] is None


def test_a_paused_platform_never_reaches_failed_permanent_via_reconcile(conn):
    """Before this fix, a platform paused because it was misbehaving could
    still be driven to failed_permanent by a crash-loop reconcile while
    nobody was watching."""
    delivery_id = enqueue(conn, "proj", 0, "facebook", "hash-a")
    _expired_lease(conn, delivery_id)
    assert pause_platform(conn, "facebook") is True

    for _ in range(5):
        reconcile_interrupted_attempts(conn, now=NOW, max_attempts=1)
        _expired_lease(conn, delivery_id)

    assert _row(conn, delivery_id)["status"] == "queued"


def test_an_unpaused_platforms_interrupted_attempt_is_unaffected(conn):
    """The existing behavior for a live platform must not regress."""
    delivery_id = enqueue(conn, "proj", 0, "facebook", "hash-a")
    _expired_lease(conn, delivery_id)

    reconciled = reconcile_interrupted_attempts(conn, now=NOW)

    row = _row(conn, delivery_id)
    assert row["attempt_count"] == 1
    assert reconciled == 1


AWAITING = "awaiting_manual_publish"
PARKED_FOREVER = "9999-01-01T00:00:00+00:00"


def _parked(conn, publish_id="v_inbox_file~v2.1"):
    delivery_id = enqueue(conn, "proj", 0, "tiktok", "hash-a")
    conn.execute(
        "UPDATE deliveries SET status=?, next_attempt_at=?, platform_upload_id=? WHERE id=?",
        (AWAITING, PARKED_FOREVER, publish_id, delivery_id),
    )
    conn.commit()
    return delivery_id


def _permalink_for(post_id):
    return f"https://www.tiktok.com/@example_account/video/{post_id}"


def test_a_paused_platforms_awaiting_manual_row_is_never_polled(conn):
    delivery_id = _parked(conn)
    assert pause_platform(conn, "tiktok") is True
    calls = []

    result = reconcile_awaiting_manual(
        conn, poll=lambda p: calls.append(p), permalink_for=_permalink_for
    )

    assert calls == []
    assert result.polled == 0
    assert _row(conn, delivery_id)["status"] == AWAITING


def test_an_unpaused_platforms_awaiting_manual_row_is_still_polled(conn):
    delivery_id = _parked(conn)

    result = reconcile_awaiting_manual(
        conn,
        poll=lambda _p: {"status": "PUBLISH_COMPLETE", "publicaly_available_post_id": [1]},
        permalink_for=_permalink_for,
    )

    assert result.published == 1
    assert _row(conn, delivery_id)["status"] == "published"
