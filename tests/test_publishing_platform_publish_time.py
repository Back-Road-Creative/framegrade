"""A scheduled upload is not a publication confirmed to the second.

`published_at` is always the local moment the queue reached "published",
and `published_at_source` is a promise about how close that is to the real
thing -- `platform_confirmed` means "the publish call returned success, so
this is within seconds". The YouTube path uploads with
`publish_at=<scheduled time>` and `privacy="private"`: the call returning
success means the platform ACCEPTED it and will make it public later,
usually hours later. Labelling that `platform_confirmed` overstates the
evidence by exactly the length of the schedule, and the real publish time --
which the platform was just told, and which the queue already has -- was
thrown away.
"""

import pytest

from gomoveshift.publishing.queue import (
    PUBLISHED_AT_PLATFORM_CONFIRMED as CONFIRMED,
    PUBLISHED_AT_PLATFORM_SCHEDULED as SCHEDULED,
    enqueue,
    open_queue_db,
    transition,
)

WHEN = "2026-09-20T14:00:00+00:00"


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _queued(conn) -> int:
    return enqueue(conn, "youtube", "vid-1", "short-1", "/tmp/plan.json")


def test_a_scheduled_publication_records_the_time_the_platform_will_publish(conn):
    """The one fact that makes the receipt honest: the queue says when it
    handed the video over AND when the platform said it goes live."""
    delivery_id = _queued(conn)

    transition(
        conn,
        delivery_id,
        "queued",
        "published",
        published_at_source=SCHEDULED,
        platform_published_at=WHEN,
    )

    row = conn.execute(
        "SELECT published_at, published_at_source, platform_published_at "
        "FROM deliveries WHERE id=?",
        (delivery_id,),
    ).fetchone()
    assert row["platform_published_at"] == WHEN
    assert row["published_at_source"] == SCHEDULED
    # published_at stays the local hand-over moment, not the platform's time.
    assert row["published_at"] != WHEN


def test_a_platform_time_that_is_not_a_timestamp_is_refused(conn):
    """An unparseable value in the evidence chain is worse than no value --
    it reads as a real platform receipt on every surface downstream."""
    delivery_id = _queued(conn)

    with pytest.raises(ValueError):
        transition(
            conn,
            delivery_id,
            "queued",
            "published",
            published_at_source=SCHEDULED,
            platform_published_at="soon",
        )


def test_a_scheduled_source_without_a_time_is_refused(conn):
    """ "Scheduled" is a claim that a specific later time is known. Without
    one it is indistinguishable from `platform_confirmed`, and quieter."""
    delivery_id = _queued(conn)

    with pytest.raises(ValueError):
        transition(conn, delivery_id, "queued", "published", published_at_source=SCHEDULED)


def test_a_platform_time_on_a_non_published_transition_is_refused(conn):
    """There is no platform publish time for a delivery that has not been
    handed over yet."""
    delivery_id = _queued(conn)

    with pytest.raises(ValueError):
        transition(conn, delivery_id, "queued", "submitting", platform_published_at=WHEN)


def test_a_confirmed_publication_may_not_claim_a_future_platform_time(conn):
    """`platform_confirmed` already means "within seconds". Carrying a
    separate platform time alongside it would be two different claims about
    the same event with nothing to say which one a reader should believe."""
    delivery_id = _queued(conn)

    with pytest.raises(ValueError):
        transition(
            conn,
            delivery_id,
            "queued",
            "published",
            published_at_source=CONFIRMED,
            platform_published_at=WHEN,
        )


def test_an_existing_queue_db_gains_the_column(tmp_path):
    """An existing queue DB can predate the column, so it has to ALTER onto a
    table that predates it -- and legacy rows stay NULL, an honest unknown,
    never backfilled from a timestamp that means something else.

    The "old" DB is the current schema with this one column dropped, so the
    fixture is a real queue in every other respect rather than a hand-written
    guess at what the table used to look like."""
    directory = tmp_path / "seo"
    first = open_queue_db(directory)
    try:
        enqueue(first, "youtube", "vid-1", "short-1", "/tmp/plan.json")
        first.execute("ALTER TABLE deliveries DROP COLUMN platform_published_at")
        first.commit()
        cols = {r["name"] for r in first.execute("PRAGMA table_info(deliveries)")}
        assert "platform_published_at" not in cols
    finally:
        first.close()

    conn = open_queue_db(directory)
    try:
        row = conn.execute("SELECT platform_published_at FROM deliveries").fetchone()
        assert row["platform_published_at"] is None
    finally:
        conn.close()


# --- accept-cases: the existing receipts are unchanged ----------------------


def test_a_confirmed_publication_still_works_with_no_platform_time(conn):
    """The guard must not break the paths that publish immediately."""
    delivery_id = _queued(conn)

    assert transition(conn, delivery_id, "queued", "published", published_at_source=CONFIRMED)

    row = conn.execute(
        "SELECT published_at, platform_published_at FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    assert row["published_at"]
    assert row["platform_published_at"] is None


def test_an_ordinary_transition_is_unaffected(conn):
    delivery_id = _queued(conn)

    assert transition(conn, delivery_id, "queued", "submitting")
