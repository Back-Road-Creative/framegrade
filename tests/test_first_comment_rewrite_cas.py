"""``rewrite_unposted_first_comment``'s own SQL guards, with no caller pre-check in front.

``replace_relink`` checks the same conditions before it calls, which hides these clauses from
a relink test: drop one from the UPDATE and every relink test still passes. Called directly,
each guard is the only thing standing between a frozen row and a rewrite."""

from datetime import datetime, timedelta, timezone

import pytest

from gomoveshift.publishing.queue import (
    FIRST_COMMENT_FROZEN_STATUSES,
    FIRST_COMMENT_MAX_ATTEMPTS,
    enqueue,
    open_queue_db,
    rewrite_unposted_first_comment,
)

OLD, NEW = "Watch: https://youtu.be/old", "Watch: https://youtu.be/new"


NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


# A lease is judged by the process clock, not the caller's ``now``: a live one runs out in 9999,
# an expired one in 2000, so no wall clock is read.
LIVE_LEASE = datetime(9999, 1, 1, tzinfo=timezone.utc).isoformat()
EXPIRED_LEASE = datetime(2000, 1, 1, tzinfo=timezone.utc).isoformat()


def _iso(delta, now=NOW):
    return (now + delta).isoformat()


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "q")
    yield c
    c.close()


def _delivery(conn, sets="", args=()):
    did = enqueue(conn, "proj", 1, "youtube", "h1", first_comment_text=OLD)
    if sets:
        conn.execute(f"UPDATE deliveries SET {sets} WHERE id=?", (*args, did))  # noqa: S608
        conn.commit()
    return did


def _text(conn, did):
    return conn.execute("SELECT first_comment_text FROM deliveries WHERE id=?", (did,)).fetchone()[
        0
    ]


def _rewrite(conn, did):
    return rewrite_unposted_first_comment(conn, did, OLD, NEW, _iso(timedelta(0)))


def test_an_open_row_is_rewritten_and_remembers_the_old_text(conn):
    did = _delivery(conn)
    assert _rewrite(conn, did) is True
    row = conn.execute(
        "SELECT first_comment_text, first_comment_prior_text FROM deliveries WHERE id=?", (did,)
    ).fetchone()
    assert row[0] == NEW and OLD in row[1]


@pytest.mark.parametrize(
    "sets,args",
    [
        ("first_comment_posted_at=?", ("2026-08-23T00:00:00+00:00",)),
        ("first_comment_attempts=?", (FIRST_COMMENT_MAX_ATTEMPTS,)),
        *[("status=?", (s,)) for s in FIRST_COMMENT_FROZEN_STATUSES],
        ("lease_owner='w', lease_expires_at=?", (LIVE_LEASE,)),
    ],
    ids=["posted", "attempts-at-max", *FIRST_COMMENT_FROZEN_STATUSES, "live-lease"],
)
def test_a_frozen_row_is_refused_and_unchanged(conn, sets, args):
    did = _delivery(conn, sets, args)
    assert _rewrite(conn, did) is False
    assert _text(conn, did) == OLD


def test_an_expired_lease_does_not_block(conn):
    did = _delivery(conn, "lease_owner='w', lease_expires_at=?", (EXPIRED_LEASE,))
    assert _rewrite(conn, did) is True and _text(conn, did) == NEW


def test_a_lease_is_judged_by_the_clock_not_the_callers_now(conn):
    # A caller passing a wrong (future) `now` must not make a live lease look expired.
    did = _delivery(conn, "lease_owner='w', lease_expires_at=?", (LIVE_LEASE,))
    assert rewrite_unposted_first_comment(conn, did, OLD, NEW, "2100-01-01T00:00:00+00:00") is False
    assert _text(conn, did) == OLD


def test_a_changed_text_is_refused(conn):
    did = _delivery(conn)
    assert rewrite_unposted_first_comment(conn, did, OLD + "x", NEW, _iso(timedelta(0))) is False
