"""Tests for the durable per-delivery social publishing queue.

Deliberately independent of ``publishing.models``/``plan`` (see the
``queue.py`` module docstring)."""

import os
import sqlite3

import pytest

from gomoveshift.publishing.queue import (
    NEXT_ATTEMPT_AWAITING_MANUAL as AWAITING_MANUAL,
    NEXT_ATTEMPT_PLATFORM_THROTTLE as THROTTLE,
    NEXT_ATTEMPT_RETRY_BACKOFF as RETRY_BACKOFF,
    PUBLISHED_AT_OPERATOR_CONFIRMED as OPERATOR,
    PUBLISHED_AT_PLATFORM_RECONCILED as PLATFORM,
    acquire_lease,
    cancel_delivery,
    enqueue,
    find_delivery,
    get_platform_daily_cap,
    is_platform_paused,
    log_response,
    open_queue_db,
    pause_platform,
    reschedule_delivery,
    resume_platform,
    retry_delivery,
    select_due_first_comments,
    set_platform_daily_cap,
    transition,
)


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def test_fresh_db_creates_schema_with_wal(tmp_path, conn):
    db_path = tmp_path / "seo" / "social_queue.sqlite3"
    assert db_path.exists()
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "deliveries" in tables


def test_enqueue_is_idempotent_for_same_four_tuple(conn):
    first = enqueue(conn, "proj", 0, "instagram", "hash-a")
    second = enqueue(conn, "proj", 0, "instagram", "hash-a")
    assert first == second
    rows = conn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0]
    assert rows == 1


def test_db_and_lock_file_are_group_writable_regardless_of_umask(tmp_path):
    # A restrictive umask must not leave the spool openable only by the
    # user who happened to create it -- two OS users (host and container)
    # share this spool.
    old_umask = os.umask(0o022)
    try:
        open_queue_db(tmp_path).close()
    finally:
        os.umask(old_umask)
    db_path = tmp_path / "social_queue.sqlite3"
    lock_path = tmp_path / "social_queue.sqlite3.schema.lock"
    assert os.stat(db_path).st_mode & 0o020
    assert os.stat(lock_path).st_mode & 0o020


def test_second_active_delivery_rejected_by_db(conn):
    enqueue(conn, "proj", 0, "instagram", "hash-a")
    with pytest.raises(sqlite3.IntegrityError):
        enqueue(conn, "proj", 0, "instagram", "hash-b")


def test_republish_after_published_returns_existing_row_never_inserts(conn):
    # A short already delivered for this (project, short, platform) must
    # never be spooled again under a new plan_hash -- the active-delivery
    # index excludes terminal statuses, so nothing else stops this.
    first_id = enqueue(conn, "proj", 0, "instagram", "hash-a")
    assert transition(conn, first_id, "queued", "published", published_at_source=OPERATOR)

    second_id = enqueue(conn, "proj", 0, "instagram", "hash-b")

    assert second_id == first_id
    rows = conn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0]
    assert rows == 1


def test_opening_db_created_without_published_index_gets_it(tmp_path):
    db_dir = tmp_path / "seo"
    db_dir.mkdir()
    db_path = db_dir / "social_queue.sqlite3"
    old_conn = sqlite3.connect(str(db_path))
    old_conn.executescript(
        """
        CREATE TABLE deliveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT NOT NULL, short_index INTEGER NOT NULL,
            platform TEXT NOT NULL, plan_hash TEXT NOT NULL, status TEXT NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 0, next_attempt_at TEXT,
            platform_upload_id TEXT, platform_post_id TEXT, error_class TEXT,
            lease_owner TEXT, lease_expires_at TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE (project_name, short_index, platform, plan_hash)
        );
        """
    )
    old_conn.commit()
    old_conn.close()

    conn = open_queue_db(db_dir)
    try:
        names = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        }
        assert "ux_deliveries_published" in names
    finally:
        conn.close()


def test_cas_transition_from_wrong_state_fails_and_changes_nothing(conn):
    delivery_id = enqueue(conn, "proj", 0, "tiktok", "hash-a")
    ok = transition(conn, delivery_id, "staging", "staged")
    assert ok is False
    row = conn.execute("SELECT status FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    assert row[0] == "queued"


def test_awaiting_manual_publish_is_a_valid_non_terminal_state(conn):
    # TikTok inbox mode (handlers.make_tiktok_handler) lands here --
    # not one of the two hardcoded terminal statuses, so it still accepts a
    # further transition (a human marking it published).
    delivery_id = enqueue(conn, "proj", 0, "tiktok", "hash-a")
    assert transition(conn, delivery_id, "queued", "awaiting_manual_publish")
    assert transition(
        conn, delivery_id, "awaiting_manual_publish", "published", published_at_source=OPERATOR
    )
    row = conn.execute("SELECT status FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    assert row[0] == "published"


def test_bogus_transition_into_awaiting_manual_publish_is_refused(conn):
    delivery_id = enqueue(conn, "proj", 0, "tiktok", "hash-a")
    ok = transition(conn, delivery_id, "uploaded", "awaiting_manual_publish")
    assert ok is False
    row = conn.execute("SELECT status FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    assert row[0] == "queued"


def test_terminal_state_rejects_further_transitions(conn):
    delivery_id = enqueue(conn, "proj", 1, "tiktok", "hash-a")
    assert transition(conn, delivery_id, "queued", "failed_permanent")
    assert transition(conn, delivery_id, "failed_permanent", "queued") is False
    row = conn.execute("SELECT status FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    assert row[0] == "failed_permanent"


T0, T1, T2 = "2026-08-23T00:00:00+00:00", "2026-08-23T01:00:00+00:00", "2026-08-23T02:00:00+00:00"
T3 = "2026-08-23T03:00:00+00:00"


def test_expired_lease_is_reclaimable(conn):
    delivery_id = enqueue(conn, "proj", 2, "youtube", "hash-a")
    assert acquire_lease(conn, delivery_id, "worker-1", T0, now=T0)
    assert acquire_lease(conn, delivery_id, "worker-2", T2, now=T1) is True


def test_live_lease_is_not_reclaimable(conn):
    delivery_id = enqueue(conn, "proj", 3, "youtube", "hash-a")
    assert acquire_lease(conn, delivery_id, "worker-1", T2, now=T0)
    assert acquire_lease(conn, delivery_id, "worker-2", T2, now=T1) is False


def test_holder_can_renew_its_own_live_lease(conn):
    """The HOLDER re-acquiring is a renewal, not a collision.

    drain_queue takes the lease, then hands the row to an adapter that takes
    it again before the irreversible API call. With a live-lease test of
    "unheld or expired" only, that second call always fails and every adapter
    raises RuntimeError("... leased by another worker") -- against itself.
    All four due deliveries (youtube, instagram, facebook, tiktok) would
    fail that way on the first real drain, and nothing could ever publish.
    """
    delivery_id = enqueue(conn, "proj", 4, "youtube", "hash-a")
    # LIVE at T1 (expires T2), so this is a renewal, not an expiry reclaim.
    assert acquire_lease(conn, delivery_id, "worker-1", T2, now=T0)
    assert acquire_lease(conn, delivery_id, "worker-1", T3, now=T1) is True
    row = conn.execute(
        "SELECT lease_owner, lease_expires_at FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    assert row[0] == "worker-1"
    assert row[1] == T3, "a renewal must extend the expiry"


def test_renewal_still_refuses_a_terminal_delivery(conn):
    """Renewal must not resurrect a delivery that already published."""
    delivery_id = enqueue(conn, "proj", 5, "youtube", "hash-a")
    assert acquire_lease(conn, delivery_id, "worker-1", T2, now=T0)
    transition(conn, delivery_id, "queued", "published", published_at_source=PLATFORM)
    assert acquire_lease(conn, delivery_id, "worker-1", T2, now=T1) is False


def test_find_delivery_returns_existing_for_reconciliation(conn):
    delivery_id = enqueue(conn, "proj", 4, "instagram", "hash-a")
    transition(conn, delivery_id, "queued", "staging")
    found = find_delivery(conn, "proj", 4, "instagram")
    assert found is not None
    assert found["id"] == delivery_id
    assert found["status"] == "staging"


def test_fresh_db_has_dedicated_permalink_column(conn):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(deliveries)").fetchall()}
    assert "permalink" in cols


def test_opening_db_created_without_permalink_column_adds_it(tmp_path):
    db_dir = tmp_path / "seo"
    db_dir.mkdir()
    db_path = db_dir / "social_queue.sqlite3"
    old_conn = sqlite3.connect(str(db_path))
    old_conn.executescript(
        """
        CREATE TABLE deliveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT NOT NULL, short_index INTEGER NOT NULL,
            platform TEXT NOT NULL, plan_hash TEXT NOT NULL, status TEXT NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 0, next_attempt_at TEXT,
            platform_upload_id TEXT, platform_post_id TEXT, error_class TEXT,
            lease_owner TEXT, lease_expires_at TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE (project_name, short_index, platform, plan_hash)
        );
        """
    )
    old_conn.commit()
    old_conn.close()

    conn = open_queue_db(db_dir)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(deliveries)").fetchall()}
        assert "permalink" in cols
    finally:
        conn.close()


def test_transition_can_persist_permalink_alongside_bare_post_id(conn):
    delivery_id = enqueue(conn, "proj", 5, "instagram", "hash-a")
    ok = transition(
        conn,
        delivery_id,
        "queued",
        "published",
        platform_post_id="m-1",
        permalink="https://instagram.com/reel/m-1",
        published_at_source=PLATFORM,
    )
    assert ok
    row = conn.execute(
        "SELECT platform_post_id, permalink FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    assert row["platform_post_id"] == "m-1"
    assert row["permalink"] == "https://instagram.com/reel/m-1"


def test_deliveries_default_to_live_not_shadow(conn):
    delivery_id = enqueue(conn, "proj", 1, "youtube", "hash")
    row = conn.execute("SELECT shadow FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    assert row["shadow"] == 0


def test_shadow_deliveries_are_marked_as_such(conn):
    delivery_id = enqueue(conn, "proj", 1, "youtube", "hash", shadow=True)
    row = conn.execute("SELECT shadow FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    assert row["shadow"] == 1


def test_fresh_db_has_dedicated_sidecar_path_column(conn):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(deliveries)").fetchall()}
    assert "sidecar_path" in cols


def test_enqueue_records_the_sidecar_path_on_the_row(conn):
    delivery_id = enqueue(conn, "proj", 6, "youtube", "hash-a", sidecar_path="/d/A/proj.json")
    row = conn.execute("SELECT sidecar_path FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    assert row["sidecar_path"] == "/d/A/proj.json"


def test_enqueue_without_sidecar_path_leaves_it_null(conn):
    delivery_id = enqueue(conn, "proj", 7, "youtube", "hash-a")
    row = conn.execute("SELECT sidecar_path FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    assert row["sidecar_path"] is None


def test_opening_db_created_without_sidecar_path_column_adds_it(tmp_path):
    db_dir = tmp_path / "seo"
    db_dir.mkdir()
    db_path = db_dir / "social_queue.sqlite3"
    old_conn = sqlite3.connect(str(db_path))
    old_conn.executescript(
        """
        CREATE TABLE deliveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT NOT NULL, short_index INTEGER NOT NULL,
            platform TEXT NOT NULL, plan_hash TEXT NOT NULL, status TEXT NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 0, next_attempt_at TEXT,
            platform_upload_id TEXT, platform_post_id TEXT, permalink TEXT,
            error_class TEXT, lease_owner TEXT, lease_expires_at TEXT,
            shadow INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE (project_name, short_index, platform, plan_hash)
        );
        """
    )
    old_conn.execute(
        "INSERT INTO deliveries (project_name, short_index, platform, plan_hash, status, "
        "created_at, updated_at) VALUES ('p', 1, 'youtube', 'h', 'queued', 'x', 'x')"
    )
    old_conn.commit()
    old_conn.close()

    conn = open_queue_db(db_dir)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(deliveries)").fetchall()}
        assert "sidecar_path" in cols
        # A pre-existing row predates this column, so it falls back to the
        # directory convention rather than carrying a bogus path.
        row = conn.execute("SELECT sidecar_path FROM deliveries WHERE short_index=1").fetchone()
        assert row["sidecar_path"] is None
    finally:
        conn.close()


def test_a_db_created_before_the_shadow_column_gains_it(tmp_path):
    # Same upgrade path as `permalink`: CREATE TABLE IF NOT EXISTS never adds
    # a column, so an existing spool would otherwise be missing it and every
    # query naming `shadow` would raise OperationalError.
    db_dir = tmp_path / "seo"
    db_dir.mkdir()
    old = sqlite3.connect(str(db_dir / "social_queue.sqlite3"))
    old.executescript(
        """
        CREATE TABLE deliveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT NOT NULL, short_index INTEGER NOT NULL,
            platform TEXT NOT NULL, plan_hash TEXT NOT NULL, status TEXT NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 0, next_attempt_at TEXT,
            platform_upload_id TEXT, platform_post_id TEXT, permalink TEXT,
            error_class TEXT, lease_owner TEXT, lease_expires_at TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE (project_name, short_index, platform, plan_hash)
        );
        """
    )
    old.execute(
        "INSERT INTO deliveries (project_name, short_index, platform, plan_hash, status, "
        "created_at, updated_at) VALUES ('p', 1, 'youtube', 'h', 'queued', 'x', 'x')"
    )
    old.commit()
    old.close()

    upgraded = open_queue_db(db_dir)
    try:
        row = upgraded.execute("SELECT shadow FROM deliveries WHERE short_index=1").fetchone()
        # A pre-existing row predates shadow mode, so it is live ammunition.
        assert row["shadow"] == 0
    finally:
        upgraded.close()


def test_schema_lock_file_is_group_writable_regardless_of_umask(tmp_path):
    """The .schema.lock beside the spool is shared by two OS users (both write
    the same spool from different processes). ``os.open(..., 0o664)`` is only
    a ceiling -- the process umask (022 for both users) lowered it to 644, so
    the second user could not even take the flock (PermissionError on
    the schema lock file). The mode must be
    set explicitly after creation."""
    import os
    import stat

    from gomoveshift.publishing import queue as q

    old = os.umask(0o022)
    try:
        q.open_queue_db(tmp_path).close()
    finally:
        os.umask(old)
    lock = tmp_path / "social_queue.sqlite3.schema.lock"
    assert lock.exists()
    assert stat.S_IMODE(lock.stat().st_mode) == 0o664


def test_error_detail_column_exists_and_is_writable(conn):
    """``error_class`` alone cannot diagnose a failure.

    Without it, a delivery that failed with error_class=RuntimeError left
    nothing else on record: the message ("delivery N is leased by another
    worker") lived only in the drain's stdout, and a cron wrapper that logs
    `tail -1` loses all but the last. Root-causing it needed reading the
    adapters instead of the record.
    """
    delivery_id = enqueue(conn, "proj", 6, "youtube", "hash-a")
    assert transition(
        conn,
        delivery_id,
        "queued",
        "queued",
        error_class="RuntimeError",
        error_detail="delivery 8 is leased by another worker",
    )
    row = conn.execute(
        "SELECT error_class, error_detail FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    assert row[0] == "RuntimeError"
    assert row[1] == "delivery 8 is leased by another worker"


def test_publish_registers_platform_id_with_tracking_store(conn, monkeypatch):
    """Reaching ``published`` registers the delivered platform id against
    the project's tracking record, the first import edge from publishing/
    into tracking/."""
    registered = {}

    class _FakeStore:
        def update_platform_id(self, project, short_index, platform, video_id, **kw):
            registered["call"] = (project, short_index, platform, video_id)
            registered["provenance"] = kw

    monkeypatch.setattr("gomoveshift.tracking.TrackingStore", _FakeStore)
    delivery_id = enqueue(conn, "proj", 8, "instagram", "hash-a")
    assert transition(
        conn,
        delivery_id,
        "queued",
        "published",
        platform_post_id="ig_123",
        published_at_source=PLATFORM,
    )
    assert registered["call"] == ("proj", 8, "instagram", "ig_123")
    # `transition` derives `published_at` and requires the source, so the
    # hook has a complete pair and must hand both over, not the id alone.
    assert registered["provenance"]["posted_at_source"] == PLATFORM
    assert registered["provenance"]["posted_at"]


def test_publish_registers_youtube_upload_id_not_playlist_id(conn, monkeypatch):
    """YouTube stores the video id in ``platform_upload_id`` and the
    playlist id in ``platform_post_id`` -- the video id is the trackable
    content id, not the playlist."""
    registered = {}

    class _FakeStore:
        def update_platform_id(self, project, short_index, platform, video_id, **kw):
            registered["call"] = (project, short_index, platform, video_id)
            registered["provenance"] = kw

    monkeypatch.setattr("gomoveshift.tracking.TrackingStore", _FakeStore)
    delivery_id = enqueue(conn, "proj", 9, "youtube", "hash-a")
    assert transition(
        conn,
        delivery_id,
        "queued",
        "published",
        platform_upload_id="yt_video_1",
        platform_post_id="yt_playlist_1",
        published_at_source=PLATFORM,
    )
    assert registered["call"] == ("proj", 9, "youtube", "yt_video_1")


def test_publish_swallows_unregistered_short_without_failing(conn, monkeypatch):
    """The Phase-D hook's contract: a short unknown to the tracking store
    raises ``ValueError`` there, which must never fail the publish."""

    class _RaisingStore:
        def update_platform_id(self, project, short_index, platform, video_id):
            raise ValueError(f"Short #{short_index} not found in project {project}")

    monkeypatch.setattr("gomoveshift.tracking.TrackingStore", _RaisingStore)
    delivery_id = enqueue(conn, "proj", 10, "tiktok", "hash-a")
    assert transition(
        conn,
        delivery_id,
        "queued",
        "published",
        platform_post_id="tt_1",
        published_at_source=PLATFORM,
    )


def test_publish_without_any_id_never_touches_tracking_store(conn, monkeypatch):
    """No id delivered (both columns NULL) must never even attempt the call."""

    class _ExplodingStore:
        def __init__(self, *a, **k):
            raise AssertionError("TrackingStore must not be constructed with no id to register")

    monkeypatch.setattr("gomoveshift.tracking.TrackingStore", _ExplodingStore)
    delivery_id = enqueue(conn, "proj", 11, "facebook", "hash-a")
    assert transition(conn, delivery_id, "queued", "published", published_at_source=PLATFORM)


def test_transition_to_published_sets_published_at(conn):
    delivery_id = enqueue(conn, "proj", 12, "instagram", "hash-a")
    assert transition(conn, delivery_id, "queued", "published", published_at_source=PLATFORM)
    row = conn.execute("SELECT published_at FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    assert row["published_at"] is not None


def test_transition_to_non_published_status_leaves_published_at_null(conn):
    delivery_id = enqueue(conn, "proj", 13, "instagram", "hash-a")
    assert transition(conn, delivery_id, "queued", "staging")
    row = conn.execute("SELECT published_at FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    assert row["published_at"] is None


def test_caller_cannot_set_published_at_explicitly(conn):
    """published_at is derived internally on the published transition only --
    an explicit caller value would defeat the once-only guarantee."""
    delivery_id = enqueue(conn, "proj", 14, "instagram", "hash-a")
    with pytest.raises(ValueError):
        transition(
            conn, delivery_id, "queued", "published", published_at="2020-01-01T00:00:00+00:00"
        )


def test_published_at_is_not_rewritten_by_a_later_transition(conn):
    """updated_at moves on every transition; published_at only moves once,
    on the transition into ``published`` -- never on any transition before
    or (were it possible) after it."""
    delivery_id = enqueue(conn, "proj", 15, "instagram", "hash-a")
    assert transition(conn, delivery_id, "queued", "staging")
    row0 = conn.execute(
        "SELECT published_at, updated_at FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    assert row0["published_at"] is None
    updated_at_before_publish = row0["updated_at"]

    assert transition(conn, delivery_id, "staging", "published", published_at_source=PLATFORM)
    row1 = conn.execute(
        "SELECT published_at, updated_at FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    assert row1["published_at"] is not None
    assert row1["updated_at"] != updated_at_before_publish

    # A further transition attempt is refused by the CAS guard (published is
    # terminal), so nothing on the row moves -- including published_at.
    assert transition(conn, delivery_id, "published", "queued") is False
    row2 = conn.execute(
        "SELECT published_at, updated_at FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    assert row2["published_at"] == row1["published_at"]
    assert row2["updated_at"] == row1["updated_at"]


def test_opening_db_created_without_published_at_column_adds_it(tmp_path):
    db_dir = tmp_path / "seo"
    db_dir.mkdir()
    db_path = db_dir / "social_queue.sqlite3"
    old_conn = sqlite3.connect(str(db_path))
    old_conn.executescript(
        """
        CREATE TABLE deliveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT NOT NULL, short_index INTEGER NOT NULL,
            platform TEXT NOT NULL, plan_hash TEXT NOT NULL, status TEXT NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 0, next_attempt_at TEXT,
            platform_upload_id TEXT, platform_post_id TEXT, permalink TEXT,
            error_class TEXT, error_detail TEXT, lease_owner TEXT, lease_expires_at TEXT,
            shadow INTEGER NOT NULL DEFAULT 0, sidecar_path TEXT, scheduled_at TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE (project_name, short_index, platform, plan_hash)
        );
        """
    )
    old_conn.execute(
        "INSERT INTO deliveries (project_name, short_index, platform, plan_hash, status, "
        "created_at, updated_at) VALUES ('p', 1, 'youtube', 'h', 'published', 'x', 'x')"
    )
    old_conn.commit()
    old_conn.close()

    conn = open_queue_db(db_dir)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(deliveries)").fetchall()}
        assert "published_at" in cols
        # A pre-existing published row predates this column -- NULL, never a
        # fabricated backfill from updated_at.
        row = conn.execute("SELECT published_at FROM deliveries WHERE short_index=1").fetchone()
        assert row["published_at"] is None
    finally:
        conn.close()


def test_success_clears_the_previous_failure(conn):
    """A published row must not keep advertising the error of an earlier
    attempt. A row published after a failed attempt must not still read
    ``error_class=RuntimeError`` from the attempt before."""
    delivery_id = enqueue(conn, "proj", 7, "youtube", "hash-a")
    transition(
        conn, delivery_id, "queued", "queued", error_class="RuntimeError", error_detail="boom"
    )
    assert transition(conn, delivery_id, "queued", "published", published_at_source=PLATFORM)
    row = conn.execute(
        "SELECT status, error_class, error_detail FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    assert row[0] == "published"
    assert row[1] is None and row[2] is None, "a success must clear the prior failure"


def test_transition_to_published_requires_a_recognized_source(conn):
    """A caller reaching ``published`` without naming who confirmed it (or
    naming a bogus source) is a bug -- raising makes an unlabelled receipt
    impossible to create."""
    delivery_id = enqueue(conn, "proj", 16, "instagram", "hash-a")
    with pytest.raises(ValueError):
        transition(conn, delivery_id, "queued", "published")
    with pytest.raises(ValueError):
        transition(conn, delivery_id, "queued", "published", published_at_source="operator_vibes")
    row = conn.execute("SELECT status FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    assert row["status"] == "queued"


def test_opening_db_created_without_published_at_source_column_adds_it(tmp_path):
    """A pre-existing row with ``published_at`` set but no source column
    (the real upgrade path from a database created before the source column existed) reads unknown -- NULL,
    never backfilled as an operator confirmation it cannot prove."""
    db_dir = tmp_path / "seo"
    db_dir.mkdir()
    db_path = db_dir / "social_queue.sqlite3"
    old_conn = sqlite3.connect(str(db_path))
    old_conn.executescript(
        """
        CREATE TABLE deliveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT NOT NULL, short_index INTEGER NOT NULL,
            platform TEXT NOT NULL, plan_hash TEXT NOT NULL, status TEXT NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 0, published_at TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE (project_name, short_index, platform, plan_hash)
        );
        """
    )
    old_conn.execute(
        "INSERT INTO deliveries (project_name, short_index, platform, plan_hash, status, "
        "published_at, created_at, updated_at) VALUES "
        "('p', 1, 'youtube', 'h', 'published', '2020-01-01T00:00:00+00:00', 'x', 'x')"
    )
    old_conn.commit()
    old_conn.close()

    conn = open_queue_db(db_dir)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(deliveries)").fetchall()}
        assert "published_at_source" in cols
        row = conn.execute(
            "SELECT published_at, published_at_source FROM deliveries WHERE short_index=1"
        ).fetchone()
        assert row["published_at"] == "2020-01-01T00:00:00+00:00"
        assert row["published_at_source"] is None
    finally:
        conn.close()


def test_cancel_succeeds_from_queued(conn):
    delivery_id = enqueue(conn, "proj", 20, "instagram", "hash-a")
    assert cancel_delivery(conn, delivery_id) is True
    row = conn.execute("SELECT status FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    assert row["status"] == "cancelled"


def test_cancel_refuses_a_published_row(conn):
    delivery_id = enqueue(conn, "proj", 21, "instagram", "hash-a")
    assert transition(conn, delivery_id, "queued", "published", published_at_source=PLATFORM)
    assert cancel_delivery(conn, delivery_id) is False
    row = conn.execute("SELECT status FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    assert row["status"] == "published"


def test_cancel_refuses_a_live_leased_row(conn):
    delivery_id = enqueue(conn, "proj", 22, "instagram", "hash-a")
    assert acquire_lease(conn, delivery_id, "worker-1", T2, now=T0)
    assert cancel_delivery(conn, delivery_id, now=T1) is False
    row = conn.execute("SELECT status FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    assert row["status"] == "queued"


def test_cancel_succeeds_on_an_expired_lease(conn):
    delivery_id = enqueue(conn, "proj", 23, "instagram", "hash-a")
    assert acquire_lease(conn, delivery_id, "worker-1", T0, now=T0)
    assert cancel_delivery(conn, delivery_id, now=T2) is True
    row = conn.execute("SELECT status FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    assert row["status"] == "cancelled"


def test_cancelled_row_not_returned_by_select_due(conn):
    from gomoveshift.publishing.dispatch import _select_due

    delivery_id = enqueue(conn, "proj", 24, "instagram", "hash-a")
    assert cancel_delivery(conn, delivery_id) is True
    now = "2026-09-11T00:00:00+00:00"
    ids = [r["id"] for r in _select_due(conn, now)]
    assert delivery_id not in ids


def test_acquire_lease_refuses_a_cancelled_row(conn):
    delivery_id = enqueue(conn, "proj", 25, "instagram", "hash-a")
    assert cancel_delivery(conn, delivery_id) is True
    assert acquire_lease(conn, delivery_id, "worker-1", T2, now=T0) is False


def test_transition_cannot_move_a_cancelled_row(conn):
    delivery_id = enqueue(conn, "proj", 26, "instagram", "hash-a")
    assert cancel_delivery(conn, delivery_id) is True
    assert transition(conn, delivery_id, "cancelled", "queued") is False
    row = conn.execute("SELECT status FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    assert row["status"] == "cancelled"


def test_transition_refuses_next_attempt_at_without_a_reason(conn):
    delivery_id = enqueue(conn, "proj", 32, "instagram", "hash-a")
    with pytest.raises(ValueError, match="next_attempt_reason"):
        transition(conn, delivery_id, "queued", "queued", next_attempt_at=T3)


def test_transition_clears_next_attempt_reason_when_next_attempt_at_is_cleared(conn):
    delivery_id = enqueue(conn, "proj", 33, "instagram", "hash-a")
    assert transition(
        conn, delivery_id, "queued", "queued", next_attempt_at=T3, next_attempt_reason=RETRY_BACKOFF
    )
    assert transition(conn, delivery_id, "queued", "queued", next_attempt_at=None)
    row = conn.execute(
        "SELECT next_attempt_at, next_attempt_reason FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    assert row["next_attempt_at"] is None
    assert row["next_attempt_reason"] is None


def test_reschedule_changes_scheduled_at(conn):
    delivery_id = enqueue(conn, "proj", 27, "instagram", "hash-a", scheduled_at=T0)
    assert reschedule_delivery(conn, delivery_id, T2) is True
    row = conn.execute("SELECT scheduled_at FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    assert row["scheduled_at"] == T2


def test_reschedule_refuses_a_terminal_row(conn):
    delivery_id = enqueue(conn, "proj", 28, "instagram", "hash-a")
    assert transition(conn, delivery_id, "queued", "published", published_at_source=PLATFORM)
    assert reschedule_delivery(conn, delivery_id, T2) is False


def test_reschedule_refuses_a_live_leased_row(conn):
    delivery_id = enqueue(conn, "proj", 29, "instagram", "hash-a")
    assert acquire_lease(conn, delivery_id, "worker-1", T2, now=T0)
    assert reschedule_delivery(conn, delivery_id, T3, now=T1) is False


def test_reschedule_forward_clears_a_stale_past_next_attempt_at(conn):
    """A row that failed and is waiting on retry backoff has both
    ``next_attempt_at`` and ``scheduled_at`` set. Rescheduling to a time in
    the FUTURE while ``next_attempt_at`` still sits in the past must not
    leave the row due immediately at the new schedule's arrival by the old
    backoff -- an operator moving a slot forward means "not before this new
    time", so a past ``next_attempt_at`` is cleared."""
    delivery_id = enqueue(conn, "proj", 30, "instagram", "hash-a", scheduled_at=T0)
    assert transition(
        conn, delivery_id, "queued", "queued", next_attempt_at=T0, next_attempt_reason=RETRY_BACKOFF
    )
    assert reschedule_delivery(conn, delivery_id, T3, now=T1) is True
    row = conn.execute(
        "SELECT scheduled_at, next_attempt_at, next_attempt_reason FROM deliveries WHERE id=?",
        (delivery_id,),
    ).fetchone()
    assert row["scheduled_at"] == T3
    assert row["next_attempt_at"] is None
    assert row["next_attempt_reason"] is None


def test_reschedule_earlier_leaves_a_future_next_attempt_at_alone(conn):
    """Rescheduling EARLIER while ``next_attempt_at`` is later than the new
    schedule must not make the row due before its retry backoff expires --
    the backoff is a rate limit against the platform, not a display
    artifact, and only a stale (already-past) ``next_attempt_at`` is
    considered no-longer-meaningful."""
    delivery_id = enqueue(conn, "proj", 31, "instagram", "hash-a", scheduled_at=T3)
    assert transition(
        conn, delivery_id, "queued", "queued", next_attempt_at=T2, next_attempt_reason=RETRY_BACKOFF
    )
    assert reschedule_delivery(conn, delivery_id, T0, now=T1) is True
    row = conn.execute(
        "SELECT scheduled_at, next_attempt_at FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    assert row["scheduled_at"] == T0
    assert row["next_attempt_at"] == T2


def test_a_db_whose_active_index_predates_cancelled_is_rebuilt(tmp_path):
    """``CREATE UNIQUE INDEX IF NOT EXISTS`` silently keeps an index that
    already exists under an older predicate, so a queue file created before
    ``cancelled`` was terminal would go on counting a cancelled row as active
    -- and the re-enqueue that cancelling exists to allow would fail on an
    existing install while passing on a fresh one."""
    db_dir = tmp_path / "seo"
    open_queue_db(db_dir).close()
    stale = sqlite3.connect(str(db_dir / "social_queue.sqlite3"))
    stale.executescript(
        "DROP INDEX ux_deliveries_active;"
        "CREATE UNIQUE INDEX ux_deliveries_active ON deliveries"
        " (project_name, short_index, platform)"
        " WHERE status NOT IN ('published', 'failed_permanent');"
    )
    stale.commit()
    stale.close()

    conn = open_queue_db(db_dir)
    try:
        stored = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='ux_deliveries_active'"
        ).fetchone()["sql"]
        assert "'cancelled'" in stored

        first = enqueue(conn, project_name="p", short_index=0, platform="youtube", plan_hash="h1")
        assert cancel_delivery(conn, first) is True
        # The slot is free again: the same short may be re-enqueued.
        second = enqueue(conn, project_name="p", short_index=0, platform="youtube", plan_hash="h2")
        assert second != first
    finally:
        conn.close()


def test_pause_platform_succeeds_for_a_known_platform(conn):
    assert pause_platform(conn, "instagram") is True
    assert is_platform_paused(conn, "instagram") is True


def test_pause_platform_refuses_an_unknown_platform(conn):
    assert pause_platform(conn, "myspace") is False
    assert is_platform_paused(conn, "myspace") is False


def test_resume_platform_clears_the_pause(conn):
    assert pause_platform(conn, "instagram") is True
    assert resume_platform(conn, "instagram") is True
    assert is_platform_paused(conn, "instagram") is False


def test_resume_platform_not_paused_is_a_no_op_false(conn):
    assert resume_platform(conn, "instagram") is False


def test_pause_state_survives_reopening_the_db(tmp_path):
    db_dir = tmp_path / "seo"
    conn = open_queue_db(db_dir)
    try:
        assert pause_platform(conn, "tiktok") is True
    finally:
        conn.close()

    reopened = open_queue_db(db_dir)
    try:
        assert is_platform_paused(reopened, "tiktok") is True
    finally:
        reopened.close()


def test_pause_table_is_added_to_a_db_created_before_it_existed(tmp_path):
    """An already-deployed queue file predates ``platform_pauses`` --
    ``CREATE TABLE IF NOT EXISTS`` on a brand-new table (never an ALTER on
    an existing one) means simply reopening picks it up, unlike the
    active-index predicate trap above."""
    db_dir = tmp_path / "seo"
    open_queue_db(db_dir).close()
    pre_existing = sqlite3.connect(str(db_dir / "social_queue.sqlite3"))
    pre_existing.execute("DROP TABLE IF EXISTS platform_pauses")
    pre_existing.commit()
    pre_existing.close()

    conn = open_queue_db(db_dir)
    try:
        assert pause_platform(conn, "facebook") is True
        assert is_platform_paused(conn, "facebook") is True
    finally:
        conn.close()


def test_retry_moves_a_backing_off_row_without_changing_attempt_count(conn):
    delivery_id = enqueue(conn, "proj", 40, "instagram", "hash-a")
    assert transition(
        conn,
        delivery_id,
        "queued",
        "queued",
        attempt_count=2,
        next_attempt_at=T3,
        next_attempt_reason=RETRY_BACKOFF,
    )
    assert retry_delivery(conn, delivery_id, now=T1) is True
    row = conn.execute(
        "SELECT next_attempt_at, next_attempt_reason, attempt_count FROM deliveries WHERE id=?",
        (delivery_id,),
    ).fetchone()
    assert row["next_attempt_at"] is None
    assert row["next_attempt_reason"] is None
    assert row["attempt_count"] == 2


def test_retry_refuses_a_platform_throttle_and_leaves_the_row_unchanged(conn):
    delivery_id = enqueue(conn, "proj", 45, "tiktok", "hash-a")
    assert transition(
        conn, delivery_id, "queued", "queued", next_attempt_at=T3, next_attempt_reason=THROTTLE
    )
    assert retry_delivery(conn, delivery_id, now=T1) is False
    row = conn.execute(
        "SELECT next_attempt_at, next_attempt_reason FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    assert row["next_attempt_at"] == T3
    assert row["next_attempt_reason"] == THROTTLE


def test_retry_refuses_an_awaiting_manual_row_and_leaves_it_unchanged(conn):
    delivery_id = enqueue(conn, "proj", 46, "tiktok", "hash-a")
    assert transition(
        conn,
        delivery_id,
        "queued",
        "awaiting_manual_publish",
        next_attempt_at="9999-01-01T00:00:00+00:00",
        next_attempt_reason=AWAITING_MANUAL,
    )
    assert retry_delivery(conn, delivery_id, now=T1) is False
    row = conn.execute(
        "SELECT next_attempt_at, next_attempt_reason FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    assert row["next_attempt_at"] == "9999-01-01T00:00:00+00:00"
    assert row["next_attempt_reason"] == AWAITING_MANUAL


def test_retry_treats_a_pre_migration_null_reason_as_ordinary_backoff(conn):
    """A row written before this column existed carries a time with no
    reason. ``retry_delivery`` must not refuse it -- an unlabelled wait is
    the common case (ordinary backoff), never a throttle or a manual park,
    both of which are always written with their reason."""
    delivery_id = enqueue(conn, "proj", 47, "instagram", "hash-a")
    conn.execute("UPDATE deliveries SET next_attempt_at=? WHERE id=?", (T3, delivery_id))
    conn.commit()
    assert retry_delivery(conn, delivery_id, now=T1) is True
    row = conn.execute(
        "SELECT next_attempt_at, next_attempt_reason FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    assert row["next_attempt_at"] is None
    assert row["next_attempt_reason"] is None


def test_retry_refuses_a_terminal_row(conn):
    delivery_id = enqueue(conn, "proj", 41, "instagram", "hash-a")
    assert transition(conn, delivery_id, "queued", "published", published_at_source=PLATFORM)
    assert retry_delivery(conn, delivery_id) is False


def test_retry_refuses_a_cancelled_row(conn):
    delivery_id = enqueue(conn, "proj", 42, "instagram", "hash-a")
    assert cancel_delivery(conn, delivery_id) is True
    assert retry_delivery(conn, delivery_id) is False


def test_retry_refuses_a_live_leased_row(conn):
    delivery_id = enqueue(conn, "proj", 43, "instagram", "hash-a")
    assert acquire_lease(conn, delivery_id, "worker-1", T2, now=T0)
    assert retry_delivery(conn, delivery_id, now=T1) is False


def test_retry_succeeds_on_an_expired_lease(conn):
    delivery_id = enqueue(conn, "proj", 44, "instagram", "hash-a")
    assert acquire_lease(conn, delivery_id, "worker-1", T0, now=T0)
    assert transition(
        conn, delivery_id, "queued", "queued", next_attempt_at=T3, next_attempt_reason=RETRY_BACKOFF
    )
    assert retry_delivery(conn, delivery_id, now=T2) is True


# ---------------------------------------------------------------------------
# YouTube first comment (schema + selector)
# ---------------------------------------------------------------------------

_PUB_AT = "2026-08-23T00:00:00+00:00"
_NOW_25H_LATER = "2026-08-23T00:06:00+00:00"  # 6min after _PUB_AT: due (delay is 5min)
_NOW_23H_LATER = "2026-08-23T00:04:00+00:00"  # 4min after _PUB_AT: not yet due


def _make_published_youtube_row(conn, short_index, first_comment_text="great drive!"):
    delivery_id = enqueue(
        conn,
        "proj",
        short_index,
        "youtube",
        f"hash-{short_index}",
        first_comment_text=first_comment_text,
    )
    assert transition(
        conn,
        delivery_id,
        "queued",
        "published",
        platform_upload_id=f"yt-post-{short_index}",
        published_at_source=PLATFORM,
    )
    conn.execute("UPDATE deliveries SET published_at=? WHERE id=?", (_PUB_AT, delivery_id))
    conn.commit()
    return delivery_id


def test_opening_a_pre_c1_db_adds_first_comment_columns(tmp_path):
    from gomoveshift.publishing.queue import select_due_first_comments

    db_dir = tmp_path / "seo"
    db_dir.mkdir()
    db_path = db_dir / "social_queue.sqlite3"
    old_conn = sqlite3.connect(str(db_path))
    old_conn.executescript(
        """
        CREATE TABLE deliveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT NOT NULL, short_index INTEGER NOT NULL,
            platform TEXT NOT NULL, plan_hash TEXT NOT NULL, status TEXT NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 0, next_attempt_at TEXT,
            platform_upload_id TEXT, platform_post_id TEXT, error_class TEXT,
            lease_owner TEXT, lease_expires_at TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE (project_name, short_index, platform, plan_hash)
        );
        """
    )
    old_conn.commit()
    old_conn.close()

    conn = open_queue_db(db_dir)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(deliveries)").fetchall()}
        assert {
            "first_comment_text",
            "first_comment_posted_at",
            "first_comment_error",
            "first_comment_attempts",
        } <= cols
        # A legacy row has no columns to be due on; must not raise.
        assert select_due_first_comments(conn, now=_NOW_25H_LATER) == []
    finally:
        conn.close()


def test_select_due_first_comments_returns_row_past_the_delay(conn):
    delivery_id = _make_published_youtube_row(conn, 50)
    due = select_due_first_comments(conn, now=_NOW_25H_LATER)
    assert [row["id"] for row in due] == [delivery_id]


def test_select_due_first_comments_excludes_row_before_the_delay(conn):
    _make_published_youtube_row(conn, 51)
    assert select_due_first_comments(conn, now=_NOW_23H_LATER) == []


def test_select_due_first_comments_excludes_already_posted(conn):
    delivery_id = _make_published_youtube_row(conn, 52)
    conn.execute(
        "UPDATE deliveries SET first_comment_posted_at=? WHERE id=?",
        (_NOW_25H_LATER, delivery_id),
    )
    conn.commit()
    assert select_due_first_comments(conn, now=_NOW_25H_LATER) == []


def test_select_due_first_comments_excludes_row_with_no_comment_text(conn):
    _make_published_youtube_row(conn, 53, first_comment_text=None)
    assert select_due_first_comments(conn, now=_NOW_25H_LATER) == []


def test_select_due_first_comments_excludes_row_with_no_post_id(conn):
    delivery_id = enqueue(conn, "proj", 54, "youtube", "hash-54", first_comment_text="hi")
    assert transition(conn, delivery_id, "queued", "published", published_at_source=PLATFORM)
    conn.execute("UPDATE deliveries SET published_at=? WHERE id=?", (_PUB_AT, delivery_id))
    conn.commit()
    assert select_due_first_comments(conn, now=_NOW_25H_LATER) == []


def test_select_due_first_comments_excludes_tiktok_platform(conn):
    delivery_id = enqueue(conn, "proj", 55, "tiktok", "hash-55", first_comment_text="hi")
    assert transition(
        conn,
        delivery_id,
        "queued",
        "published",
        platform_post_id="tt-post-55",
        published_at_source=PLATFORM,
    )
    conn.execute("UPDATE deliveries SET published_at=? WHERE id=?", (_PUB_AT, delivery_id))
    conn.commit()
    assert select_due_first_comments(conn, now=_NOW_25H_LATER) == []


def test_pause_platform_stores_reason(conn):
    assert pause_platform(conn, "tiktok", now="2026-09-17T00:00:00+00:00", reason="auto:suppressed")
    row = conn.execute("SELECT reason FROM platform_pauses WHERE platform='tiktok'").fetchone()
    assert row["reason"] == "auto:suppressed"


def test_pause_platform_reason_defaults_to_none(conn):
    assert pause_platform(conn, "facebook")
    row = conn.execute("SELECT reason FROM platform_pauses WHERE platform='facebook'").fetchone()
    assert row["reason"] is None


def test_platform_pauses_reason_column_is_added_to_a_db_created_before_it_existed(tmp_path):
    db_dir = tmp_path / "seo"
    db_dir.mkdir()
    old = sqlite3.connect(str(db_dir / "social_queue.sqlite3"))
    old.executescript(
        """
        CREATE TABLE platform_pauses (
            platform TEXT PRIMARY KEY,
            paused_at TEXT NOT NULL
        );
        """
    )
    old.execute("INSERT INTO platform_pauses (platform, paused_at) VALUES ('youtube', 'x')")
    old.commit()
    old.close()

    upgraded = open_queue_db(db_dir)
    try:
        row = upgraded.execute(
            "SELECT reason FROM platform_pauses WHERE platform='youtube'"
        ).fetchone()
        assert row["reason"] is None
        assert pause_platform(upgraded, "tiktok", reason="auto:suppressed")
    finally:
        upgraded.close()


def test_platform_daily_cap_round_trip(conn):
    assert get_platform_daily_cap(conn, "youtube") is None
    set_platform_daily_cap(
        conn,
        "youtube",
        cap=2,
        state="RECOVERING",
        mode="shadow",
        reason="w2 ladder step",
        now="2026-09-17T00:00:00+00:00",
    )
    row = get_platform_daily_cap(conn, "youtube")
    assert row["cap"] == 2
    assert row["state"] == "RECOVERING"
    assert row["mode"] == "shadow"
    assert row["reason"] == "w2 ladder step"


def test_platform_daily_cap_upsert_keeps_one_row_per_platform(conn):
    set_platform_daily_cap(conn, "youtube", cap=1, state="SUPPRESSED", mode="live")
    set_platform_daily_cap(conn, "youtube", cap=2, state="RECOVERING", mode="live")
    row = get_platform_daily_cap(conn, "youtube")
    assert row["cap"] == 2
    assert row["state"] == "RECOVERING"
    count = conn.execute(
        "SELECT COUNT(*) FROM platform_daily_caps WHERE platform='youtube'"
    ).fetchone()[0]
    assert count == 1


def test_log_response_appends_an_audit_row(conn):
    row_id = log_response(
        conn,
        "youtube",
        state="SUPPRESSED",
        cap=1,
        action="cap_set",
        mode="shadow",
        reason="w2",
        now="2026-09-17T00:00:00+00:00",
    )
    row = conn.execute("SELECT * FROM response_log WHERE id=?", (row_id,)).fetchone()
    assert row["platform"] == "youtube"
    assert row["action"] == "cap_set"
    assert row["mode"] == "shadow"


def test_response_log_keeps_every_row_never_upserts(conn):
    log_response(conn, "youtube", state="SUPPRESSED", cap=1, action="cap_set", mode="shadow")
    log_response(conn, "youtube", state="SUPPRESSED", cap=1, action="cap_set", mode="shadow")
    count = conn.execute("SELECT COUNT(*) FROM response_log WHERE platform='youtube'").fetchone()[0]
    assert count == 2


def test_w2_tables_are_added_to_a_db_created_before_they_existed(tmp_path):
    db_dir = tmp_path / "seo"
    db_dir.mkdir()
    old = sqlite3.connect(str(db_dir / "social_queue.sqlite3"))
    old.executescript(
        """
        CREATE TABLE platform_pauses (
            platform TEXT PRIMARY KEY,
            paused_at TEXT NOT NULL
        );
        """
    )
    old.commit()
    old.close()

    upgraded = open_queue_db(db_dir)
    try:
        set_platform_daily_cap(upgraded, "youtube", cap=1, state="SUPPRESSED", mode="shadow")
        assert get_platform_daily_cap(upgraded, "youtube")["cap"] == 1
        log_response(
            upgraded, "youtube", state="SUPPRESSED", cap=1, action="cap_set", mode="shadow"
        )
    finally:
        upgraded.close()
