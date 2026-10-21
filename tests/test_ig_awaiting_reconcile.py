"""U-R5(b): an Instagram publish refused with "Media ID is not available" parks
at ``awaiting_ig_reconcile`` instead of failing permanently, and only the
account media scan (the one ``queue recover-terminal`` gate (c) uses) moves it:
found -> published, proven absent -> a bounded retry, unprovable -> parked.

The error text can mean the post went live (``test_queue_recover_terminal``),
so neither a blind retry nor a terminal failure is the right answer. All data
is synthetic."""

from __future__ import annotations


import pytest
from test_publishing_models import _record

from gomoveshift.cli.handlers.pub_status import validate_restored_state
from gomoveshift.publishing.dispatch import (
    PermanentDeliveryError,
    _select_due,
    drain_queue,
    reconcile_awaiting_ig,
    reconcile_interrupted_attempts,
)
from gomoveshift.publishing.handlers import make_instagram_handler
from gomoveshift.publishing.ig_media_scan import SCAN_LIMIT, ScanUnprovableError, scan_for_post
from gomoveshift.publishing.instagram_adapter import InstagramPermanentError
from gomoveshift.publishing.queue import (
    STATUS_AWAITING_IG_RECONCILE,
    enqueue,
    open_queue_db,
    pause_platform,
    retry_delivery,
)

NOW = "2026-10-15T12:00:00+00:00"
MEDIA_GONE = (
    "[graph http=400 code=9007 subcode=2207006 fbtrace=SYNTH01] Graph API permanently "
    "rejected the request: Media ID is not available"
)
CAPTION = "Golden hour on the coast road"


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _row(conn, delivery_id):
    return dict(conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone())


def _parked(conn, *, created_at="2026-10-14T12:00:00+00:00", updated_at=None, attempts=0):
    delivery_id = enqueue(conn, "proj", 0, "instagram", "hash-a")
    conn.execute(
        "UPDATE deliveries SET status=?, next_attempt_at='9999-01-01T00:00:00+00:00', "
        "next_attempt_reason='awaiting_ig_reconcile', platform_upload_id='SYNTHcontainer1', "
        "attempt_count=?, created_at=?, updated_at=? WHERE id=?",
        (STATUS_AWAITING_IG_RECONCILE, attempts, created_at, updated_at or created_at, delivery_id),
    )
    conn.commit()
    return delivery_id


class _Raises:
    def __init__(self, exc):
        self.exc = exc

    def publish(self, record, *, conn, delivery_id, **kwargs):
        raise self.exc


def _loader(record=None):
    return lambda project, index: record or _record()


# --- the handler parks instead of failing --------------------------------


def test_media_id_not_available_parks_the_row_not_failed_permanent(conn):
    delivery_id = enqueue(conn, "proj", 0, "instagram", "hash-a")
    handler = make_instagram_handler(
        _Raises(InstagramPermanentError(MEDIA_GONE)), conn, _loader(), lambda r: b"v"
    )
    handler(_row(conn, delivery_id))  # returns: parked, not raised
    row = _row(conn, delivery_id)
    assert row["status"] == STATUS_AWAITING_IG_RECONCILE
    assert row["next_attempt_reason"] == "awaiting_ig_reconcile"
    assert row["next_attempt_at"].startswith("9999")
    assert "Media ID is not available" in row["error_detail"]
    assert row["error_detail"].startswith("[graph http=400")


def test_any_other_permanent_instagram_error_is_still_terminal(conn):
    delivery_id = enqueue(conn, "proj", 0, "instagram", "hash-a")
    handler = make_instagram_handler(
        _Raises(InstagramPermanentError("content policy rejection")),
        conn,
        _loader(),
        lambda r: b"v",
    )
    with pytest.raises(PermanentDeliveryError):
        handler(_row(conn, delivery_id))


def test_drain_does_not_count_a_parked_row_as_published(conn):
    delivery_id = enqueue(conn, "proj", 0, "instagram", "hash-a")
    handler = make_instagram_handler(
        _Raises(InstagramPermanentError(MEDIA_GONE)), conn, _loader(), lambda r: b"v"
    )
    result = drain_queue(conn, {"instagram": handler}, now=NOW)
    row = _row(conn, delivery_id)
    assert result.succeeded == 0 and result.failed_permanent == 0
    assert row["status"] == STATUS_AWAITING_IG_RECONCILE
    assert row["lease_owner"] is None


def test_a_parked_row_is_never_selected_even_when_its_wait_is_cleared(conn):
    delivery_id = _parked(conn)
    conn.execute("UPDATE deliveries SET next_attempt_at=NULL WHERE id=?", (delivery_id,))
    conn.commit()
    assert [r["id"] for r in _select_due(conn, NOW)] == []


def test_operator_retry_now_cannot_blind_retry_a_parked_row(conn):
    delivery_id = _parked(conn)
    assert retry_delivery(conn, delivery_id, now=NOW) is False
    assert _row(conn, delivery_id)["next_attempt_at"].startswith("9999")


def test_an_expired_lease_on_a_parked_row_is_released_without_an_attempt(conn):
    delivery_id = _parked(conn)
    conn.execute(
        "UPDATE deliveries SET lease_owner='w:1:dead', lease_expires_at=? WHERE id=?",
        ("2026-10-15T11:00:00+00:00", delivery_id),
    )
    conn.commit()
    reconcile_interrupted_attempts(conn, now=NOW)
    row = _row(conn, delivery_id)
    assert row["attempt_count"] == 0 and row["lease_owner"] is None
    assert row["status"] == STATUS_AWAITING_IG_RECONCILE


# --- the reconcile -------------------------------------------------------


def _media(*items):
    return lambda limit: list(items)


def test_a_post_found_on_the_account_closes_the_row_as_published(conn):
    delivery_id = _parked(conn)
    found = {
        "id": "SYNTHmedia9",
        "caption": _record().payloads.instagram.caption,
        "permalink": "https://www.instagram.com/reel/SYNTH9/",
        "timestamp": "2026-10-14T12:05:00+0000",
    }
    result = reconcile_awaiting_ig(
        conn, recent_media=_media(found), record_loader=_loader(), now=NOW
    )
    row = _row(conn, delivery_id)
    assert result.published == 1
    assert row["status"] == "published"
    assert row["platform_post_id"] == "SYNTHmedia9"
    assert row["permalink"] == "https://www.instagram.com/reel/SYNTH9/"
    assert row["published_at_source"] == "platform_reconciled"


def test_proven_absent_retries_with_a_fresh_container_and_one_attempt(conn):
    delivery_id = _parked(conn)
    other = {
        "id": "SYNTHother",
        "caption": "Something else",
        "timestamp": "2026-10-14T10:00:00+0000",
    }
    result = reconcile_awaiting_ig(
        conn, recent_media=_media(other), record_loader=_loader(), now=NOW
    )
    row = _row(conn, delivery_id)
    assert result.retried == 1
    assert row["status"] == "queued"
    assert row["attempt_count"] == 1
    assert row["platform_upload_id"] is None  # the next pass builds a new container
    assert row["next_attempt_reason"] == "retry_backoff"
    assert row["next_attempt_at"] > NOW
    assert "SYNTHcontainer1" in row["error_detail"]


def test_proven_absent_at_the_attempt_ceiling_fails_permanently(conn):
    delivery_id = _parked(conn, attempts=7)
    result = reconcile_awaiting_ig(
        conn,
        recent_media=_media(
            {"id": "x", "caption": "other", "timestamp": "2026-10-14T10:00:00+0000"}
        ),
        record_loader=_loader(),
        now=NOW,
        max_attempts=8,
    )
    row = _row(conn, delivery_id)
    assert result.failed == 1
    assert row["status"] == "failed_permanent"
    assert row["error_class"].startswith("MaxAttemptsExceeded")


def test_an_empty_scan_cannot_prove_absence_and_the_row_stays_parked(conn):
    delivery_id = _parked(conn)
    result = reconcile_awaiting_ig(conn, recent_media=_media(), record_loader=_loader(), now=NOW)
    assert result.unprovable == 1
    row = _row(conn, delivery_id)
    assert row["status"] == STATUS_AWAITING_IG_RECONCILE and row["attempt_count"] == 0


def test_a_full_page_that_does_not_reach_back_stays_parked(conn):
    delivery_id = _parked(conn, created_at="2026-10-01T00:00:00+00:00")
    page = [
        {"id": f"m{i}", "caption": "other", "timestamp": "2026-10-14T00:00:00+0000"}
        for i in range(SCAN_LIMIT)
    ]
    result = reconcile_awaiting_ig(
        conn, recent_media=_media(*page), record_loader=_loader(), now=NOW
    )
    assert result.unprovable == 1
    assert _row(conn, delivery_id)["status"] == STATUS_AWAITING_IG_RECONCILE


def test_a_scan_error_leaves_the_row_parked_and_is_reported(conn):
    delivery_id = _parked(conn)

    def boom(limit):
        raise RuntimeError("graph down")

    result = reconcile_awaiting_ig(conn, recent_media=boom, record_loader=_loader(), now=NOW)
    assert result.errors and "graph down" in result.errors[0]
    assert _row(conn, delivery_id)["status"] == STATUS_AWAITING_IG_RECONCILE


def test_a_paused_instagram_is_not_scanned(conn):
    _parked(conn)
    pause_platform(conn, "instagram", reason="operator")
    calls = []
    reconcile_awaiting_ig(
        conn, recent_media=lambda limit: calls.append(limit) or [], record_loader=_loader(), now=NOW
    )
    assert calls == []


def test_the_scan_is_one_helper_shared_with_recover_terminal():
    from gomoveshift.cli.handlers import queue_recover

    assert queue_recover.scan_for_post is scan_for_post
    assert issubclass(ScanUnprovableError, Exception)


# --- validate surfaces a row parked too long -----------------------------


def test_validate_names_a_row_parked_past_the_age_bound(conn, tmp_path):
    delivery_id = _parked(conn, updated_at="2026-10-13T12:00:00+00:00")
    problems = validate_restored_state(conn, str(tmp_path / "credentials"), now=NOW)
    parked = [p for p in problems if p.kind == "awaiting_ig_reconcile"]
    assert [p.delivery_id for p in parked] == [delivery_id]
    assert "48h" in parked[0].detail


def test_validate_leaves_a_freshly_parked_row_alone(conn, tmp_path):
    _parked(conn, updated_at="2026-10-15T06:00:00+00:00")
    problems = validate_restored_state(conn, str(tmp_path / "credentials"), now=NOW)
    assert "awaiting_ig_reconcile" not in [p.kind for p in problems]


def test_scan_for_post_matches_by_caption_prefix_or_fingerprint():
    record = _record()
    row_created = "2026-10-14T12:00:00+00:00"
    by_caption = scan_for_post(
        [{"id": "a", "caption": CAPTION.upper(), "timestamp": "2026-10-14T13:00:00+0000"}],
        created_at=row_created,
        caption=CAPTION,
        sha256=None,
    )
    assert [m["id"] for m in by_caption.matches] == ["a"]
    by_sha = scan_for_post(
        [
            {
                "id": "b",
                "caption": "x",
                "sha256": "SYNTHSHA",
                "timestamp": "2026-10-14T13:00:00+0000",
            }
        ],
        created_at=row_created,
        caption=CAPTION,
        sha256="SYNTHSHA",
    )
    assert [m["id"] for m in by_sha.matches] == ["b"]
    with pytest.raises(ScanUnprovableError):
        scan_for_post(
            [{"id": "c", "caption": "x"}], created_at=row_created, caption="", sha256=None
        )
    assert record.payloads.instagram.caption  # the fixture carries a caption
