"""A row parked at ``needs_operator`` can be closed as published by a person,
and a row parked off a dead worker's lease is counted in the drain summary.

Before: the console's mark-published took ``awaiting_manual_publish`` only, so
an operator who found a parked post live (or posted it by hand) could cancel or
retry it, never record it; and ``reconcile_interrupted_attempts`` dropped the
"parked" outcome, so ``DrainResult.needs_operator`` read 0 for a row it parked."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from gomoveshift.console import pubqueue
from gomoveshift.console.app import create_app
from gomoveshift.publishing import queue
from gomoveshift.publishing.dispatch import UNCOUNTED_RETRY_BOUND, drain_queue
from gomoveshift.publishing.queue import acquire_lease, enqueue, open_queue_db, transition

T0 = datetime.fromisoformat("2026-11-10T12:00:00+00:00")


def _row(tmp_path, delivery_id):
    conn = open_queue_db(tmp_path)
    try:
        return dict(conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone())
    finally:
        conn.close()


def _parked_at_needs_operator(tmp_path, platform="instagram"):
    conn = open_queue_db(tmp_path)
    try:
        delivery_id = enqueue(conn, "proj", 0, platform, "hash-a")
        assert transition(
            conn,
            delivery_id,
            "queued",
            queue.STATUS_NEEDS_OPERATOR,
            next_attempt_at=queue.AWAITING_MANUAL_PUBLISH_NEXT_ATTEMPT,
            next_attempt_reason=queue.NEXT_ATTEMPT_NEEDS_OPERATOR,
            platform_upload_id="c-1",
            upload_account="acct-1",
        )
    finally:
        conn.close()
    return delivery_id


def test_a_needs_operator_row_can_be_marked_published_by_a_person(tmp_path):
    delivery_id = _parked_at_needs_operator(tmp_path)

    pubqueue.mark_published(
        tmp_path / "social_queue.sqlite3",
        delivery_id,
        permalink="https://www.instagram.com/reel/abc/",
        platform_post_id="p-1",
    )

    row = _row(tmp_path, delivery_id)
    assert row["status"] == "published"
    assert row["published_at_source"] == queue.PUBLISHED_AT_OPERATOR_CONFIRMED
    assert (row["platform_post_id"], row["platform_upload_id"]) == ("p-1", "c-1")


def test_mark_published_still_refuses_a_row_no_person_is_waiting_on(tmp_path):
    conn = open_queue_db(tmp_path)
    delivery_id = enqueue(conn, "proj", 0, "instagram", "hash-a")
    conn.close()
    with pytest.raises(ValueError):
        pubqueue.mark_published(tmp_path / "social_queue.sqlite3", delivery_id)
    assert _row(tmp_path, delivery_id)["status"] == "queued"


def test_the_publishing_page_offers_mark_published_on_a_needs_operator_row(tmp_path, monkeypatch):
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: config))
    delivery_id = _parked_at_needs_operator(tmp_path)

    response = TestClient(create_app()).get("/publishing")

    assert response.status_code == 200
    assert f"/publishing/mark-published/{delivery_id}" in response.text


def test_a_row_parked_off_a_dead_workers_lease_counts_in_needs_operator(tmp_path):
    conn = open_queue_db(tmp_path / "seo")
    try:
        delivery_id = enqueue(conn, "proj", 0, "facebook", "hash-a")
        since = (T0 - UNCOUNTED_RETRY_BOUND - timedelta(hours=1)).isoformat()
        transition(
            conn,
            delivery_id,
            "queued",
            "awaiting_facebook_reconcile",
            platform_upload_id="v-1",
            upload_account="page-1",
            uncounted_since=since,
        )
        lease_at = (T0 - timedelta(hours=1)).isoformat()
        assert acquire_lease(conn, delivery_id, "dead-worker", T0.isoformat(), now=lease_at)

        result = drain_queue(conn, {}, now=T0.isoformat(), incidents_dir=tmp_path / "inc")

        row = dict(conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone())
    finally:
        conn.close()
    assert row["status"] == queue.STATUS_NEEDS_OPERATOR
    assert result.reconciled_interrupted == 1
    assert result.needs_operator == 1
    assert any(e.startswith(f"facebook#{delivery_id}: needs_operator") for e in result.errors)
