"""Upload provenance is a checkpoint, not a claim of delivered publication."""

import pytest

from gomoveshift.publishing.queue import enqueue, find_delivery, open_queue_db, transition


def test_legacy_queue_migration_keeps_upload_account_unknown_and_preserves_rows(tmp_path):
    path = tmp_path / "seo"
    conn = open_queue_db(path)
    delivery = enqueue(conn, "legacy", 1, "facebook", "hash")
    transition(conn, delivery, "queued", "uploaded", platform_upload_id="old-video")
    # Recreate the previous schema, then exercise the real reopening migration.
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(deliveries)")}
    if "upload_account" in columns:
        conn.execute("ALTER TABLE deliveries DROP COLUMN upload_account")
        conn.commit()
    original = find_delivery(conn, "legacy", 1, "facebook")
    conn.close()
    for _ in range(2):
        conn = open_queue_db(path)
        migrated = find_delivery(conn, "legacy", 1, "facebook")
        assert migrated.pop("upload_account") is None
        assert migrated == original
        conn.close()


def test_upload_account_is_write_once_without_claiming_a_published_destination(tmp_path):
    conn = open_queue_db(tmp_path / "seo")
    try:
        delivery = enqueue(conn, "project", 1, "facebook", "hash")
        assert transition(conn, delivery, "queued", "submitting", upload_account="page-1")
        assert transition(
            conn,
            delivery,
            "submitting",
            "uploaded",
            upload_account="page-1",
            platform_upload_id="video-1",
        )
        assert not transition(conn, delivery, "uploaded", "uploaded", upload_account="page-2")
        row = find_delivery(conn, "project", 1, "facebook")
        assert row["upload_account"] == "page-1" and row["platform_upload_id"] == "video-1"
        assert row["destination_account"] is None and row["published_at"] is None
    finally:
        conn.close()


@pytest.mark.parametrize("account", [None, "", "  ", 123])
def test_upload_account_rejects_missing_or_non_string_replacements(tmp_path, account):
    conn = open_queue_db(tmp_path / "seo")
    try:
        delivery = enqueue(conn, "project", 1, "facebook", "hash")
        with pytest.raises(ValueError, match="upload_account"):
            transition(conn, delivery, "queued", "submitting", upload_account=account)
        assert find_delivery(conn, "project", 1, "facebook")["status"] == "queued"
    finally:
        conn.close()
