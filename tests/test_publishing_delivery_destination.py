"""A delivery receipt names the account it landed on.

`deliveries` records WHAT was posted (`platform_post_id`, `permalink`), WHEN
(`published_at`) and how trustworthy that time is (`published_at_source`). It
never recorded WHERE. `platform` says "facebook"; it does not say which Page.

The adapters know. `facebook_adapter` holds `self._page_id = credentials
["page_id"]` and `instagram_adapter` holds `self._ig_user_id`, both taken from
the brand's credentials at dispatch, and both thrown away once the post
returns. So a published row proves a post exists and cannot say whose feed it
is in.

That was tolerable while one operator ran one set of credentials. Waypost is
multi-tenant: brands are a tenant boundary, each tenant self-hosts its own
credentials, and credentials get rotated and mistyped. A delivery that went to
the wrong Page is indistinguishable on every surface from one that went to the
right one, and the only place the truth existed was a local variable.

The value is recorded from the adapter's OWN identifier at publish time, never
re-derived later from whatever credentials are loaded now -- re-deriving would
rewrite history to match the current configuration, which is precisely the
question being asked. `NULL` means the row predates this column or its adapter
has no account identity, an honest unknown and never a guess.
"""

from __future__ import annotations

import pytest

from gomoveshift.publishing.queue import (
    PUBLISHED_AT_PLATFORM_CONFIRMED,
    enqueue,
    open_queue_db,
    transition,
)


@pytest.fixture
def conn(tmp_path):
    connection = open_queue_db(tmp_path)
    yield connection
    connection.close()


def _row(conn, delivery_id):
    return conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone()


def test_a_published_delivery_records_the_account_it_went_to(conn):
    """The defect: `platform` said "facebook" and nothing said which Page."""
    delivery = enqueue(conn, "proj", 1, "facebook", "hash-a")

    assert transition(
        conn,
        delivery,
        "queued",
        "published",
        platform_post_id="fb_1",
        published_at_source=PUBLISHED_AT_PLATFORM_CONFIRMED,
        destination_account="1234567890",
    )

    assert _row(conn, delivery)["destination_account"] == "1234567890"


def test_a_delivery_that_never_published_has_no_destination(conn):
    delivery = enqueue(conn, "proj", 1, "facebook", "hash-a")

    assert _row(conn, delivery)["destination_account"] is None


def test_the_destination_is_only_valid_on_reaching_published(conn):
    """Same rule as `published_at_source`: an account name on a queued or
    failed row is a claim that something landed there when nothing did."""
    delivery = enqueue(conn, "proj", 1, "facebook", "hash-a")

    with pytest.raises(ValueError) as excinfo:
        transition(conn, delivery, "queued", "submitting", destination_account="1234567890")

    assert "destination_account" in str(excinfo.value)


def test_a_published_row_may_omit_it(conn):
    """An adapter with no account identity of its own must still be able to
    publish. The column is an honest unknown, not a new required field."""
    delivery = enqueue(conn, "proj", 1, "youtube", "hash-a")

    assert transition(
        conn,
        delivery,
        "queued",
        "published",
        platform_upload_id="yt_1",
        published_at_source=PUBLISHED_AT_PLATFORM_CONFIRMED,
    )

    assert _row(conn, delivery)["destination_account"] is None


def test_an_empty_destination_is_refused_rather_than_stored(conn):
    """`""` is not an unknown -- it is a stored answer that reads as one on
    every surface, while being indistinguishable from a bug that dropped the
    real value. If there is nothing to say, say nothing."""
    delivery = enqueue(conn, "proj", 1, "facebook", "hash-a")

    with pytest.raises(ValueError) as excinfo:
        transition(
            conn,
            delivery,
            "queued",
            "published",
            platform_post_id="fb_1",
            published_at_source=PUBLISHED_AT_PLATFORM_CONFIRMED,
            destination_account="   ",
        )

    assert "destination_account" in str(excinfo.value)


def test_an_existing_queue_file_gains_the_column(tmp_path):
    """The migration pattern the other late columns use: a live queue picks
    this up on the next `open_queue_db`, with NULL for every existing row."""
    first = open_queue_db(tmp_path)
    delivery = enqueue(first, "proj", 1, "facebook", "hash-a")
    first.execute("ALTER TABLE deliveries DROP COLUMN destination_account")
    first.commit()
    first.close()

    second = open_queue_db(tmp_path)
    try:
        assert (
            second.execute(
                "SELECT destination_account FROM deliveries WHERE id=?", (delivery,)
            ).fetchone()[0]
            is None
        )
    finally:
        second.close()


def test_the_recorded_account_is_never_overwritten_by_a_later_touch(conn):
    """`published_at`'s guarantee, for the same reason: the CAS already gates
    the UPDATE on a non-terminal status, so a reconcile that runs after a
    credential rotation cannot rewrite where the post actually went."""
    delivery = enqueue(conn, "proj", 1, "facebook", "hash-a")
    transition(
        conn,
        delivery,
        "queued",
        "published",
        platform_post_id="fb_1",
        published_at_source=PUBLISHED_AT_PLATFORM_CONFIRMED,
        destination_account="the-original-page",
    )

    assert not transition(
        conn,
        delivery,
        "published",
        "published",
        published_at_source=PUBLISHED_AT_PLATFORM_CONFIRMED,
        destination_account="a-rotated-page",
    )
    assert _row(conn, delivery)["destination_account"] == "the-original-page"


# ── The adapters' end ────────────────────────────────────────────────────
#
# The account id lives on the HTTP CLIENT, not the adapter above it: the
# client is the object holding the credential that was actually used. Both
# publish paths read it through `_account_of`, which tolerates a client with
# no such identity (an injected test double) by recording the honest unknown.
# That tolerance is why these two tests exist: without them a rename of
# `account_id` would turn every real receipt into `NULL` and nothing would
# fail.


def test_the_facebook_client_exposes_the_page_it_posts_to():
    from gomoveshift.publishing.facebook_adapter import FacebookReelsAdapter

    client = FacebookReelsAdapter(
        {"page_id": "page-42", "page_access_token": "t"}, session=object()
    )

    assert client.account_id == "page-42"


def test_the_instagram_client_exposes_the_user_it_posts_to():
    from gomoveshift.publishing.instagram_adapter import InstagramReelsAdapter

    client = InstagramReelsAdapter(
        {"ig_user_id": "ig-42", "page_access_token": "t"}, session=object()
    )

    assert client.account_id == "ig-42"


def test_a_client_with_no_account_identity_records_the_unknown():
    """A test double, or a future client for a platform with no such notion.
    Publishing must not break, and the column must not be given a guess."""
    from gomoveshift.publishing import facebook_adapter, instagram_adapter

    assert facebook_adapter._account_of(object()) is None
    assert instagram_adapter._account_of(object()) is None
