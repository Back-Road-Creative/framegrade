"""Two brand deployments must never see, dispatch, or confirm each
other's deliveries even when they enqueue the identical
``(project_name, short_index, platform, plan_hash)`` key -- the queue's
uniqueness key carries no brand column (``publishing/queue.py``), so
isolation has to come from each brand resolving its OWN storage (a
separate sqlite file) rather than from any in-table distinction. This
locks in that console/pubqueue.py's listing, dispatch.drain_queue's
selection, and pubqueue.mark_published all stay correctly scoped to
whichever brand's config/db they were given -- the invariant the
per-brand SEO storage dir (``GMS_SEO_STORAGE_DIR``) depends on.
"""

from __future__ import annotations

from types import SimpleNamespace

from gomoveshift.console import pubqueue
from gomoveshift.publishing import dispatch
from gomoveshift.publishing.queue import enqueue, open_queue_db, transition


def _brand_config(tmp_path, name):
    storage = tmp_path / name
    storage.mkdir()
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(storage)))


class _RecordingResolver:
    """Sentinel: records every row it was called for. Never returns a
    plausible-looking value that could pass a weaker ``Mock``-style check --
    the assertion is "never invoked", not "invoked with the wrong thing"."""

    def __init__(self, brand):
        self.brand = brand
        self.calls = []

    def __call__(self, row):
        self.calls.append((row["project_name"], row["short_index"], row["platform"]))


def test_two_brands_with_the_same_delivery_key_get_independent_listings(tmp_path):
    config_a = _brand_config(tmp_path, "brand-a")
    config_b = _brand_config(tmp_path, "brand-b")
    conn_a = open_queue_db(config_a.seo.storage_path)
    conn_b = open_queue_db(config_b.seo.storage_path)
    try:
        enqueue(conn_a, "proj", 3, "youtube", "same-hash")
        enqueue(conn_b, "proj", 3, "youtube", "same-hash")
    finally:
        conn_a.close()
        conn_b.close()

    rows_a = pubqueue.queue_rows(config_a)
    rows_b = pubqueue.queue_rows(config_b)

    assert len(rows_a) == 1 and len(rows_b) == 1
    # Same key tuple in both -- the two rows are genuinely distinct DB rows,
    # not one row read twice, or the isolation claim is vacuous.
    assert rows_a[0]["id"] == 1 and rows_b[0]["id"] == 1
    assert pubqueue.queue_db_path(config_a) != pubqueue.queue_db_path(config_b)


def test_draining_one_brands_queue_never_invokes_the_other_brands_resolver(tmp_path):
    config_a = _brand_config(tmp_path, "brand-a")
    config_b = _brand_config(tmp_path, "brand-b")
    conn_a = open_queue_db(config_a.seo.storage_path)
    conn_b = open_queue_db(config_b.seo.storage_path)
    try:
        enqueue(conn_a, "proj", 3, "youtube", "same-hash")
        enqueue(conn_b, "proj", 3, "youtube", "same-hash")

        resolver_a = _RecordingResolver("a")
        resolver_b = _RecordingResolver("b")

        dispatch.drain_queue(conn_a, {"youtube": resolver_a}, now="2026-09-09T00:00:00+00:00")

        assert resolver_a.calls == [("proj", 3, "youtube")]
        assert resolver_b.calls == []

        dispatch.drain_queue(conn_b, {"youtube": resolver_b}, now="2026-09-09T00:00:00+00:00")

        assert resolver_b.calls == [("proj", 3, "youtube")]
        # Brand A's resolver was never touched by brand B's drain either.
        assert resolver_a.calls == [("proj", 3, "youtube")]
    finally:
        conn_a.close()
        conn_b.close()


def test_mark_published_on_one_brand_never_touches_the_others_row(tmp_path):
    config_a = _brand_config(tmp_path, "brand-a")
    config_b = _brand_config(tmp_path, "brand-b")
    conn_a = open_queue_db(config_a.seo.storage_path)
    conn_b = open_queue_db(config_b.seo.storage_path)
    try:
        delivery_a = enqueue(conn_a, "proj", 3, "tiktok", "same-hash")
        delivery_b = enqueue(conn_b, "proj", 3, "tiktok", "same-hash")
        assert delivery_a == delivery_b == 1  # same id space, different DBs
        transition(conn_a, delivery_a, "queued", "awaiting_manual_publish")
    finally:
        conn_a.close()
        conn_b.close()

    pubqueue.mark_published(pubqueue.queue_db_path(config_a), delivery_a)

    conn_a = open_queue_db(config_a.seo.storage_path)
    conn_b = open_queue_db(config_b.seo.storage_path)
    try:
        row_a = conn_a.execute("SELECT status FROM deliveries WHERE id=?", (delivery_a,)).fetchone()
        row_b = conn_b.execute("SELECT status FROM deliveries WHERE id=?", (delivery_b,)).fetchone()
    finally:
        conn_a.close()
        conn_b.close()

    assert row_a["status"] == "published"
    # Brand B's row (same id, same key tuple, different DB) never left
    # "queued" -- brand A's confirmation resolved against its own file only.
    assert row_b["status"] == "queued"
