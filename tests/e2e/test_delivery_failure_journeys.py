"""Delivery failure journeys: failures never read as posts, and a
lost response never becomes a second post. Manifest:
tests/fixtures/acceptance/journeys.json."""

from __future__ import annotations

import json
from pathlib import Path

from .acceptance_harness import (
    NOW,
    PROJECT,
    RecordingInstagramAdapter,
    asset_loader,
    loader_for,
    make_record,
    row_of,
)

from gomoveshift.publishing.dispatch import drain_queue
from gomoveshift.publishing.handlers import build_handlers
from gomoveshift.publishing.queue import (
    NEXT_ATTEMPT_AWAITING_MANUAL,
    acquire_lease,
    enqueue,
    transition,
)

LATER = "2026-10-03T12:00:30+00:00"
AFTER_BACKOFF = "2026-10-04T12:00:00+00:00"
STALE_LEASE = "2026-10-03T11:00:00+00:00"
STALE_LEASE_NOW = "2026-10-03T10:00:00+00:00"


def _setup(conn, asset, script):
    adapter = RecordingInstagramAdapter(script)
    handlers = build_handlers(
        conn,
        loader_for({(PROJECT, 0): make_record(asset)}),
        instagram=(adapter, asset_loader),
    )
    return adapter, handlers, enqueue(conn, PROJECT, 0, "instagram", "hash-1")


def test_j08_01_transient_failure_schedules_retry_not_published(conn, asset):
    adapter, handlers, did = _setup(conn, asset, ["transient", "ok"])

    result = drain_queue(conn, handlers, now=NOW)

    assert result.retried == 1 and result.succeeded == 0
    row = row_of(conn, did)
    assert row["status"] == "queued"
    assert row["attempt_count"] == 1
    assert row["next_attempt_at"] > NOW
    assert row["next_attempt_reason"] == "retry_backoff"
    assert row["platform_post_id"] is None and row["published_at"] is None
    # Not due yet: an immediate second pass makes no second call.
    assert drain_queue(conn, handlers, now=LATER).attempted == 0
    assert len(adapter.calls) == 1
    # Once due, it recovers and publishes exactly once.
    assert drain_queue(conn, handlers, now=AFTER_BACKOFF).succeeded == 1
    assert len(adapter.remote_posts) == 1


def test_j08_02_auth_expiry_is_blocked_with_visible_reason(conn, asset):
    adapter, handlers, did = _setup(conn, asset, ["auth", "ok"])

    result = drain_queue(conn, handlers, now=NOW)

    assert result.failed_permanent == 1 and result.succeeded == 0
    row = row_of(conn, did)
    assert row["status"] == "failed_permanent"
    assert "access token expired" in row["error_detail"]
    assert row["platform_post_id"] is None and row["published_at"] is None
    assert adapter.remote_posts == []
    assert any("access token expired" in e for e in result.errors)
    # A blocked row is not retried by later drains.
    drain_queue(conn, handlers, now=AFTER_BACKOFF)
    assert len(adapter.calls) == 1


def test_j08_03_lost_response_never_creates_a_second_post(conn, asset):
    adapter, handlers, did = _setup(conn, asset, ["accept_then_lose", "ok"])

    first = drain_queue(conn, handlers, now=NOW)

    assert first.retried == 1 and first.succeeded == 0
    row = row_of(conn, did)
    assert row["status"] == "uploaded"  # remote accepted; id persisted
    assert row["platform_upload_id"] == "ig-post-1"
    assert row["published_at"] is None

    second = drain_queue(conn, handlers, now=AFTER_BACKOFF)

    assert len(adapter.calls) == 2
    assert [p["id"] for p in adapter.remote_posts] == ["ig-post-1"]  # no second post
    assert second.attempted == 1
    row = row_of(conn, did)
    assert row["status"] == "published"
    assert row["platform_post_id"] == "ig-post-1"
    # And a published row is never picked up again.
    assert drain_queue(conn, handlers, now="2026-10-05T12:00:00+00:00").attempted == 0
    assert len(adapter.remote_posts) == 1


def test_j08_04_expired_lease_on_uploaded_never_published_without_post_id(conn, asset):
    adapter, handlers, did = _setup(conn, asset, ["transient", "transient"])
    assert acquire_lease(conn, did, "dead-worker", STALE_LEASE, now=STALE_LEASE_NOW)
    assert transition(conn, did, "queued", "uploaded", platform_upload_id="ig-post-9")

    result = drain_queue(conn, handlers, now=NOW)

    assert result.reconciled_interrupted == 1
    row = row_of(conn, did)
    assert row["status"] != "published"
    assert row["platform_post_id"] is None
    assert row["attempt_count"] >= 1  # the dead worker's attempt was charged
    # Invariant across the whole queue: published implies a post id.
    bad = conn.execute(
        "SELECT id FROM deliveries WHERE status='published' AND platform_post_id IS NULL"
    ).fetchall()
    assert bad == []


def test_j08_05_expired_lease_on_parked_row_stays_parked(conn, asset):
    adapter, handlers, did = _setup(conn, asset, [])
    assert acquire_lease(conn, did, "dead-worker", STALE_LEASE, now=STALE_LEASE_NOW)
    assert transition(
        conn,
        did,
        "queued",
        "awaiting_manual_publish",
        platform_upload_id="pub-7",
        next_attempt_at="9999-01-01T00:00:00+00:00",
        next_attempt_reason=NEXT_ATTEMPT_AWAITING_MANUAL,
    )

    drain_queue(conn, handlers, now=NOW)

    row = row_of(conn, did)
    assert row["status"] == "awaiting_manual_publish"
    assert row["next_attempt_at"] == "9999-01-01T00:00:00+00:00"
    assert row["lease_owner"] is None
    assert row["platform_post_id"] is None and row["published_at"] is None
    assert adapter.calls == []


def test_manifest_entries_map_to_existing_tests():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "tests/fixtures/acceptance/journeys.json").read_text())
    ids = [j["id"] for j in manifest["journeys"]]
    assert len(ids) == len(set(ids))
    for journey in manifest["journeys"]:
        rel, _, func = journey["test"].partition("::")
        source = (root / rel).read_text()
        assert f"def {func}(" in source, f"{journey['id']} -> missing {journey['test']}"
    for built in manifest["built"]:
        assert any(i.startswith(built + "-") for i in ids), f"{built} has no journeys"
