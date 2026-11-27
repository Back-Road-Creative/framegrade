"""Durable per-delivery social publishing queue (SQLite spool).

One row per ``(project_name, short_index, platform, plan_hash)``.
Intentionally independent of ``publishing.models``/``plan`` — ``plan_hash``
is a plain ``str``, so storage carries no import edge onto planning.

Robust-by-construction uniqueness: a partial unique index over non-terminal
statuses enforces "at most one ACTIVE delivery per
``(project_name, short_index, platform)``" in the DB — a second enqueue
while one is in flight raises ``sqlite3.IntegrityError`` rather than
relying on a check-then-insert a concurrent writer could race.

No network call may occur under a write transaction: every function here is
one short statement inside its own ``with conn:`` block with no callback
seam, so callers make platform HTTP calls *between* calls into this module.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

from gomoveshift.pipeline._file_lock import lock_exclusive, unlock

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS deliveries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_name TEXT NOT NULL, short_index INTEGER NOT NULL,
    platform TEXT NOT NULL, plan_hash TEXT NOT NULL, status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0, next_attempt_at TEXT,
    next_attempt_reason TEXT,
    platform_upload_id TEXT, platform_post_id TEXT, permalink TEXT,
    upload_account TEXT, upload_kind TEXT,
    error_class TEXT,
    error_detail TEXT,
    lease_owner TEXT, lease_expires_at TEXT,
    shadow INTEGER NOT NULL DEFAULT 0,
    sidecar_path TEXT,
    scheduled_at TEXT,
    published_at TEXT,
    published_at_source TEXT,
    destination_account TEXT,
    recovers_delivery_id INTEGER,
    recovery_authorized_by TEXT,
    recovery_evidence TEXT,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    UNIQUE (project_name, short_index, platform, plan_hash)
);
CREATE TABLE IF NOT EXISTS platform_pauses (
    platform TEXT PRIMARY KEY,
    paused_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS platform_daily_caps (
    platform TEXT PRIMARY KEY,
    cap INTEGER NOT NULL,
    state TEXT NOT NULL,
    mode TEXT NOT NULL,
    set_at TEXT NOT NULL,
    reason TEXT
);
CREATE TABLE IF NOT EXISTS response_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    platform TEXT NOT NULL,
    state TEXT NOT NULL,
    cap INTEGER NOT NULL,
    action TEXT NOT NULL,
    mode TEXT NOT NULL,
    reason TEXT
);
CREATE TABLE IF NOT EXISTS facebook_photo_attempts (
    delivery_id INTEGER PRIMARY KEY REFERENCES deliveries(id),
    plan_sha256 TEXT NOT NULL, item_index INTEGER NOT NULL,
    request_json TEXT NOT NULL, write_started_at TEXT NOT NULL,
    receipt_json TEXT, story_id TEXT,
    UNIQUE(plan_sha256, item_index)
);
"""

FACEBOOK_PHOTO_KIND = "facebook_page_photo"
STATUS_PHOTO_AWAITING_ID = "awaiting_facebook_photo_id"
STATUS_PHOTO_RECONCILE = "awaiting_facebook_photo_reconcile"
#: ``platform_pauses``, ``platform_daily_caps`` and ``response_log`` are all
#: brand-new tables, unlike the ``ALTER TABLE`` columns below and the
#: active-delivery index further down: ``CREATE TABLE IF NOT EXISTS`` never
#: leaves a stale definition behind for a table that never existed, so a
#: queue file deployed before any of these tables shipped picks them up on
#: the very next ``open_queue_db`` -- no migration flag needed.
#:
#: Bounded suppression response: ``platform_daily_caps`` holds the
#: LATEST cap decision per platform (upserted; one row), read by
#: ``dispatch._select_due`` to enforce (``mode='live'``) or merely log
#: (``mode='shadow'``) a limit. ``response_log`` is the append-only audit
#: trail of every decision, shadow or live -- "a row in the queue DB" the
#: rollout plan requires for comparing the shadow log against what an
#: operator would have done.

#: Every terminal status, in one place. ``transition``, ``acquire_lease``
#: and ``dispatch._select_due`` each guard on "not one of these" -- a status
#: added here without touching all three would let a drain still pick up a
#: row an operator believed was done, which is worse than no guard at all.
#: ``missing_remote`` (U-R1): a YouTube row whose video the channel no longer
#: returns. Terminal so nothing re-uploads it unasked; ``mark_missing_remote``.
STATUS_MISSING_REMOTE = "missing_remote"
TERMINAL_STATUSES = ("published", "failed_permanent", "cancelled", STATUS_MISSING_REMOTE)
_TERMINAL_SQL = "(" + ", ".join(f"'{s}'" for s in TERMINAL_STATUSES) + ")"

#: The active-delivery uniqueness index. Deliberately NOT part of
#: ``_SCHEMA``: ``CREATE UNIQUE INDEX IF NOT EXISTS`` silently keeps an index
#: that already exists under an OLDER predicate, so a queue file created
#: before ``cancelled`` was terminal would go on counting a cancelled row as
#: active and refuse the very re-enqueue that cancelling exists to allow.
#: ``_ensure_active_index`` recreates it whenever the stored predicate is not
#: the current one, which is also how a fresh DB first gets it.
_ACTIVE_INDEX_NAME = "ux_deliveries_active"
_ACTIVE_INDEX_SQL = (
    f"CREATE UNIQUE INDEX {_ACTIVE_INDEX_NAME} ON deliveries "
    "(project_name, short_index, platform) "
    f"WHERE status NOT IN {_TERMINAL_SQL}"
)


def _ensure_active_index(conn: sqlite3.Connection) -> None:
    """Create the active-delivery index, replacing a stale-predicate one.

    Narrowing the predicate only ever indexes FEWER rows, so the recreate
    cannot fail on rows that already exist."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
        (_ACTIVE_INDEX_NAME,),
    ).fetchone()
    stored = (row["sql"] or "") if row is not None else ""
    if stored and all(f"'{status}'" in stored for status in TERMINAL_STATUSES):
        return
    conn.execute(f"DROP INDEX IF EXISTS {_ACTIVE_INDEX_NAME}")  # noqa: S608
    conn.execute(_ACTIVE_INDEX_SQL)


#: A second partial unique index, over PUBLISHED rows only. Its
#: predicate (``status='published'``) never changes over time the way the
#: active index's terminal-status list has, so unlike ``_ensure_active_index``
#: this one never needs to detect and rebuild a stale predicate --
#: ``CREATE UNIQUE INDEX IF NOT EXISTS`` is safe here and applied on every
#: open, which is how a queue file created before this index shipped picks
#: it up. Belt-and-suspenders alongside ``enqueue``'s own published-row
#: check below: this index guards against a write path that skips
#: ``enqueue`` entirely (a raw INSERT would still raise), not just a
#: caller that goes through it.
_PUBLISHED_INDEX_NAME = "ux_deliveries_published"
_PUBLISHED_INDEX_SQL = (
    f"CREATE UNIQUE INDEX IF NOT EXISTS {_PUBLISHED_INDEX_NAME} ON deliveries "
    "(project_name, short_index, platform) WHERE status='published'"
)


#: ``platform_post_id`` is ALWAYS a bare platform id, never an encoded
#: blob — see ``docs/social-publishing-queue.md``. ``permalink`` is the
#: platform's public URL for the delivery when one is returned.
#:
#: ``status`` is not an enumerated column (no CHECK constraint) — any
#: string is a valid state, and ``transition``'s compare-and-set is what
#: "refuses a bogus transition" (the current status must match
#: ``expected_status``, and neither terminal status accepts a further
#: transition). TikTok's inbox mode (``handlers.make_tiktok_handler``)
#: adds ``"awaiting_manual_publish"``: not terminal in this table (a human
#: still moves it to ``"published"`` via the console), but excluded from
#: ``dispatch._select_due`` re-selection by parking its own
#: ``next_attempt_at`` far in the future rather than teaching dispatch a
#: third terminal-ish state.


#: Three provenances for ``published_at``, each a promise about closeness to real publish time:
PUBLISHED_AT_OPERATOR_CONFIRMED = "operator_confirmed"  # human click, possibly hours late
PUBLISHED_AT_PLATFORM_RECONCILED = "platform_reconciled"  # delayed poll noticed it -- unknown lag
PUBLISHED_AT_PLATFORM_CONFIRMED = "platform_confirmed"  # publish call returned success -- seconds
#: The platform ACCEPTED it and named a later time to make it public. The
#: YouTube path uploads with ``publish_at=<scheduled>`` and ``privacy=private``
#: (a Short already due goes up ``public``, its send time recorded here), so
#: its call returning success is not a publication -- calling that
#: ``platform_confirmed`` overstates the evidence by the whole length of the
#: schedule. This source requires ``platform_published_at``: the claim it makes
#: is that a specific later time is KNOWN, and without one it says less than
#: ``platform_confirmed`` while sounding like more.
PUBLISHED_AT_PLATFORM_SCHEDULED = "platform_scheduled"
#: The platform CONFIRMED the video is public and ``platform_published_at`` holds
#: the time it went public (YouTube ``snippet.publishedAt`` once
#: ``status.privacyStatus`` reads ``public``). Reached only from
#: ``platform_scheduled`` through ``record_live_publish``, never by ``transition``:
#: a scheduled slot is a promise, this is the receipt, and a reader tells them
#: apart by this source.
PUBLISHED_AT_PLATFORM_LIVE = "platform_live"
_PUBLISHED_AT_SOURCES = (
    PUBLISHED_AT_OPERATOR_CONFIRMED,
    PUBLISHED_AT_PLATFORM_RECONCILED,
    PUBLISHED_AT_PLATFORM_CONFIRMED,
    PUBLISHED_AT_PLATFORM_SCHEDULED,
    PUBLISHED_AT_PLATFORM_LIVE,
)


#: Four reasons ``next_attempt_at`` may be set, each answering a different
#: "can an operator's retry-now override this wait?" question.
#: ``dispatch.py``'s three ordinary retryable-failure branches (interrupted
#: attempt, transient media, generic retryable exception) -- an operator
#: retry clears both fields, same as today.
NEXT_ATTEMPT_RETRY_BACKOFF = "retry_backoff"
#: ``dispatch.py``'s ``ThrottledDeliveryError`` branch -- a platform-side
#: rate limit only the platform clears; retry-now must not shorten it.
NEXT_ATTEMPT_PLATFORM_THROTTLE = "platform_throttle"
#: ``youtube_adapter.py``'s daily-upload-budget deferral -- a limit THIS
#: queue imposed on itself (``config.drive.youtube_video_insert_daily_uploads``),
#: not something YouTube refused; an operator retry may lift it.
NEXT_ATTEMPT_PLATFORM_DEFERRAL = "platform_deferral"
#: ``handlers.py``'s TikTok inbox park -- not a wait at all, a human has to
#: act in the TikTok app; retry-now cannot substitute for that.
NEXT_ATTEMPT_AWAITING_MANUAL = "awaiting_manual"

#: A delivery a human must publish by hand (the TikTok Studio pack, or the
#: legacy inbox draft) -- never reached by any automated transition to
#: "published". Lives here so ``handlers`` and ``tiktok_pack`` share one
#: definition without importing each other.
STATUS_AWAITING_MANUAL_PUBLISH = "awaiting_manual_publish"

#: ``dispatch._select_due`` reselects anything not terminal whose
#: ``next_attempt_at`` has arrived; a manual-publish row parks its own
#: ``next_attempt_at`` far enough out that it is never due again on its own.
AWAITING_MANUAL_PUBLISH_NEXT_ATTEMPT = "9999-01-01T00:00:00+00:00"

#: An Instagram publish Graph refused with "Media ID is not available". The
#: text can mean the post went live, so the row is neither retried blind nor
#: failed: it parks (same 9999 wait) until ``dispatch.reconcile_awaiting_ig``
#: scans the account. Not terminal; ``_select_due`` excludes it by status.
STATUS_AWAITING_IG_RECONCILE = "awaiting_ig_reconcile"
#: Its wait reason: only the account scan may end it, never retry-now.
NEXT_ATTEMPT_AWAITING_IG_RECONCILE = "awaiting_ig_reconcile"
#: A delivery whose retries never counted against ``max_attempts`` (unknown
#: Facebook outcome, a container still processing, unreadable media, a
#: platform throttle) and that kept retrying past ``dispatch.UNCOUNTED_RETRY_BOUND``.
#: Not terminal -- its platform ids stay, so nothing re-uploads it -- and never
#: selected by the drain: a person decides it (``dispatch`` raises an incident).
STATUS_NEEDS_OPERATOR = "needs_operator"
#: Its wait reason: only a person ends it, never retry-now.
NEXT_ATTEMPT_NEEDS_OPERATOR = "needs_operator"
_NEXT_ATTEMPT_REASONS = (
    NEXT_ATTEMPT_NEEDS_OPERATOR,
    NEXT_ATTEMPT_AWAITING_IG_RECONCILE,
    NEXT_ATTEMPT_RETRY_BACKOFF,
    NEXT_ATTEMPT_PLATFORM_THROTTLE,
    NEXT_ATTEMPT_PLATFORM_DEFERRAL,
    NEXT_ATTEMPT_AWAITING_MANUAL,
)


def _chmod_group_writable(path: Path) -> None:
    """Force ``0o664`` regardless of the creating process's umask.

    The spool is shared by two OS users (host/CI and the dev container);
    an ``os.open``/``sqlite3.connect`` mode request is masked by the
    creator's umask, so a restrictive umask silently locks the other user
    out. Never raises: a file this process doesn't own (a legacy file from
    before this fix) can't be chmod'd by it either -- that sweep is a
    host job, not this one's to force.
    """
    try:
        os.chmod(path, 0o664)
    except PermissionError:
        logger.debug("cannot chmod %s (not owned by this process)", path)


@contextlib.contextmanager
def _schema_lock(db_path: Path):
    """Blocking file lock guarding one-time schema creation (the lock call
    is ``pipeline._file_lock``'s portable one)."""
    lock_path = db_path.parent / (db_path.name + ".schema.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o664)
    # The mode argument above is only a ceiling: the umask (022 is common)
    # lowers it to 644, and then a DIFFERENT user cannot open the
    # lock for writing and the flock fails. Set it outright;
    # fchmod on a file we do not own raises, and that is the one case where
    # the file already belongs to the other user, so ignore it.
    # Windows Python before 3.13 has no fchmod, and has no group bits to set.
    if hasattr(os, "fchmod"):
        with contextlib.suppress(OSError):
            os.fchmod(fd, 0o664)
    try:
        lock_exclusive(fd)
        yield
    finally:
        with contextlib.suppress(OSError):
            unlock(fd)
        os.close(fd)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def facebook_photo_attempt(conn, delivery_id: int) -> Optional[Dict[str, Any]]:
    """Read an immutable request snapshot and its separately retained receipt."""
    row = conn.execute(
        "SELECT * FROM facebook_photo_attempts WHERE delivery_id=?", (delivery_id,)
    ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["request"] = json.loads(result.pop("request_json"))
    receipt = result.pop("receipt_json")
    result["receipt"] = json.loads(receipt) if receipt else None
    return result


def begin_facebook_photo_attempt(conn, delivery_id, owner, request, *, now):
    """Atomic write-ahead checkpoint, never a permission or safe-retry proof.

    The caller has verified authority and immutable bytes. A duplicate approved
    item raises IntegrityError and rolls back the delivery change. No nested
    caller transaction is accepted: returning must mean this is committed.
    """
    if conn.in_transaction:
        raise ValueError("Photo checkpoint requires its own transaction")
    if request["delivery_id"] != delivery_id or request["published"] is not True:
        raise ValueError("Photo checkpoint request binding invalid")
    if request["valid_until"] <= now:
        return False
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":"))
    with conn:
        changed = conn.execute(
            "UPDATE deliveries SET status=?, upload_kind=?, upload_account=?, updated_at=? "
            "WHERE id=? AND platform='facebook' AND status='queued' AND shadow=0 "
            "AND plan_hash=? AND lease_owner=? AND lease_expires_at>? "
            "AND (scheduled_at IS NULL OR scheduled_at<=?) "
            "AND (next_attempt_at IS NULL OR next_attempt_at<=?) "
            "AND platform NOT IN (SELECT platform FROM platform_pauses) "
            "AND upload_kind IS NULL AND upload_account IS NULL "
            "AND platform_upload_id IS NULL AND platform_post_id IS NULL "
            "AND NOT EXISTS (SELECT 1 FROM facebook_photo_attempts WHERE delivery_id=deliveries.id)",
            (
                STATUS_PHOTO_AWAITING_ID,
                FACEBOOK_PHOTO_KIND,
                request["page_id"],
                now,
                delivery_id,
                request["queue_plan_hash"],
                owner,
                now,
                now,
                now,
            ),
        ).rowcount
        if changed != 1:
            return False
        conn.execute(
            "INSERT INTO facebook_photo_attempts "
            "(delivery_id,plan_sha256,item_index,request_json,write_started_at) VALUES (?,?,?,?,?)",
            (delivery_id, request["plan_sha256"], request["item_index"], encoded, now),
        )
    return True


def retain_facebook_photo_receipt(conn, delivery_id, receipt):
    """Persist the returned ID and receipt together; never replace a retained ID."""
    if conn.in_transaction:
        raise ValueError("Photo receipt requires its own transaction")
    with conn:
        changed = conn.execute(
            "UPDATE deliveries SET status=?, platform_upload_id=?, updated_at=? "
            "WHERE id=? AND status=? AND upload_kind=? AND upload_account=? "
            "AND platform_upload_id IS NULL AND platform_post_id IS NULL "
            "AND EXISTS (SELECT 1 FROM facebook_photo_attempts WHERE delivery_id=deliveries.id)",
            (
                STATUS_PHOTO_RECONCILE,
                receipt["photo_id"],
                _now_iso(),
                delivery_id,
                STATUS_PHOTO_AWAITING_ID,
                FACEBOOK_PHOTO_KIND,
                receipt["page_id"],
            ),
        ).rowcount
        if changed != 1:
            return False
        if (
            conn.execute(
                "UPDATE facebook_photo_attempts SET receipt_json=? "
                "WHERE delivery_id=? AND receipt_json IS NULL",
                (json.dumps(receipt, sort_keys=True), delivery_id),
            ).rowcount
            != 1
        ):
            raise sqlite3.IntegrityError("Photo receipt checkpoint conflict")
    return True


def retain_facebook_photo_story(conn, delivery_id, story_id):
    with conn:
        return (
            conn.execute(
                "UPDATE facebook_photo_attempts SET story_id=? WHERE delivery_id=? "
                "AND receipt_json IS NOT NULL AND (story_id IS NULL OR story_id=?)",
                (story_id, delivery_id, story_id),
            ).rowcount
            == 1
        )


def confirm_facebook_photo(conn, delivery_id, photo_id, story_id, account):
    """Close only a retained identity after the consumer verifies readback."""
    with conn:
        changed = (
            conn.execute(
                "UPDATE deliveries SET status='published', platform_post_id=?, destination_account=?, "
                "published_at=?, published_at_source=?, updated_at=?, error_class=NULL, error_detail=NULL, "
                "lease_owner=NULL, lease_expires_at=NULL, next_attempt_at=NULL, next_attempt_reason=NULL "
                "WHERE id=? AND status=? AND upload_kind=? AND upload_account=? AND platform_upload_id=? "
                "AND EXISTS (SELECT 1 FROM facebook_photo_attempts WHERE delivery_id=deliveries.id "
                "AND story_id=? AND receipt_json IS NOT NULL)",
                (
                    story_id,
                    account,
                    _now_iso(),
                    PUBLISHED_AT_PLATFORM_RECONCILED,
                    _now_iso(),
                    delivery_id,
                    STATUS_PHOTO_RECONCILE,
                    FACEBOOK_PHOTO_KIND,
                    account,
                    photo_id,
                    story_id,
                ),
            ).rowcount
            == 1
        )
    # General photos have no driving-short tracking record. Their confirmed
    # platform identity belongs to this delivery and its durable attempt only.
    return changed


def open_queue_db(storage_path: Union[str, Path]) -> sqlite3.Connection:
    """Open/create the spool at ``<storage_path>/social_queue.sqlite3``,
    WAL-enabled, schema applied (``storage_path`` = caller-resolved
    ``config.seo.storage_path``, see ``content_ledger._default_ledger_path``)."""
    db_path = Path(storage_path) / "social_queue.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_existed = db_path.exists()
    with _schema_lock(db_path):
        conn = sqlite3.connect(str(db_path))
        if not db_existed:
            _chmod_group_writable(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        _ensure_active_index(conn)
        conn.execute(_PUBLISHED_INDEX_SQL)
        # ``CREATE TABLE IF NOT EXISTS`` never adds a column to a
        # pre-existing table — an older DB file (created before the
        # ``permalink`` column existed) needs an explicit, idempotent
        # upgrade rather than silently missing the column.
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(deliveries)").fetchall()}
        if "permalink" not in cols:
            conn.execute("ALTER TABLE deliveries ADD COLUMN permalink TEXT")
        if "shadow" not in cols:
            # Rows that predate shadow mode are live ammunition, hence 0.
            conn.execute("ALTER TABLE deliveries ADD COLUMN shadow INTEGER NOT NULL DEFAULT 0")
        if "scheduled_at" not in cols:
            # NULL for rows that predate this column. ``dispatch._select_due``
            # treats NULL as due: a row with no recorded time would otherwise
            # be stranded silently, which is the failure this column exists
            # to prevent.
            conn.execute("ALTER TABLE deliveries ADD COLUMN scheduled_at TEXT")
        if "error_detail" not in cols:
            # error_class alone cannot diagnose a failure: a run of
            # deliveries can record RuntimeError and nothing else while the one
            # message that explains them ("... leased by another worker")
            # survives only as the last line a cron wrapper happened to log.
            conn.execute("ALTER TABLE deliveries ADD COLUMN error_detail TEXT")
        if "sidecar_path" not in cols:
            # NULL for rows that predate this column -- the reader falls
            # back to the directory convention for those, see
            # ``dispatch_queue.SidecarRecordLoader``.
            conn.execute("ALTER TABLE deliveries ADD COLUMN sidecar_path TEXT")
        if "published_at" not in cols:
            # NULL for rows that predate this column. Never backfilled from
            # ``updated_at`` -- ``updated_at`` is rewritten by every
            # transition (retries, lease changes, reconcile), so it is not
            # the actual publish moment and a guessed value here would be a
            # fabricated timestamp, worse than an honest null.
            conn.execute("ALTER TABLE deliveries ADD COLUMN published_at TEXT")
        if "published_at_source" not in cols:
            # NULL for rows that predate this column -- an honest "unknown",
            # never backfilled as an operator confirmation it cannot prove.
            conn.execute("ALTER TABLE deliveries ADD COLUMN published_at_source TEXT")
        if "platform_published_at" not in cols:
            # When the platform goes live with it, as the platform itself
            # stated -- distinct from ``published_at``, which is when THIS
            # queue handed it over. NULL for every row that predates the
            # column and for every path that publishes immediately: an
            # honest unknown, never derived from ``published_at``, which
            # would be a fabricated platform receipt.
            conn.execute("ALTER TABLE deliveries ADD COLUMN platform_published_at TEXT")
        if "destination_account" not in cols:
            # WHICH account the delivery landed on, as the adapter that
            # published it held the identifier -- ``platform`` says
            # "facebook", not which Page. NULL for every row that predates
            # the column and for every adapter with no account identity of
            # its own. Never backfilled from current credentials: those can
            # have been rotated or corrected since, so a derived value would
            # rewrite history to agree with the present configuration, which
            # is exactly the question a reader is asking.
            conn.execute("ALTER TABLE deliveries ADD COLUMN destination_account TEXT")
        if "upload_account" not in cols:
            # Provenance captured before an upload starts, not a publication
            # receipt. Legacy IDs have no proven account; never backfill from
            # present-day credentials or a destination inferred later.
            conn.execute("ALTER TABLE deliveries ADD COLUMN upload_account TEXT")
        if "upload_kind" not in cols:
            conn.execute("ALTER TABLE deliveries ADD COLUMN upload_kind TEXT")
        if "first_submitted_at" not in cols:
            # When the row FIRST entered ``submitting`` (set by ``transition``,
            # never moved). A YouTube retry looks for its own lost upload from
            # here, not from "now". NULL for rows that predate it: never guessed.
            conn.execute("ALTER TABLE deliveries ADD COLUMN first_submitted_at TEXT")
        if "uploaded_at" not in cols:
            # When ``platform_upload_id`` was first written (``transition``): the
            # upload moment the YouTube daily budget counts by. NULL for older rows.
            conn.execute("ALTER TABLE deliveries ADD COLUMN uploaded_at TEXT")
        if "next_attempt_reason" not in cols:
            # NULL for rows written before this column existed. ``retry_delivery``
            # treats a non-NULL next_attempt_at with a NULL reason as the common
            # case, ``retry_backoff`` -- an unlabelled wait predates the reasons
            # this column distinguishes, and most waits are ordinary backoff.
            conn.execute("ALTER TABLE deliveries ADD COLUMN next_attempt_reason TEXT")
        if "first_comment_text" not in cols:
            # The answer text supplied at enqueue time (owned by the
            # social plan, not this module) -- NULL means no first comment is
            # scheduled for this delivery at all.
            conn.execute("ALTER TABLE deliveries ADD COLUMN first_comment_text TEXT")
        if "first_comment_posted_at" not in cols:
            # NULL until ``youtube_comments.drain_first_comments`` posts it.
            # Never derived from ``published_at`` -- an honest unknown until
            # a real API call succeeds.
            conn.execute("ALTER TABLE deliveries ADD COLUMN first_comment_posted_at TEXT")
        if "first_comment_error" not in cols:
            # The last first-comment failure (scope/permission or otherwise),
            # so an operator sees WHY a scheduled comment never posted
            # instead of guessing from a bare "still NULL".
            conn.execute("ALTER TABLE deliveries ADD COLUMN first_comment_error TEXT")
        if "first_comment_attempts" not in cols:
            # First-comment tries the platform refused because of the post
            # itself; at FIRST_COMMENT_MAX_ATTEMPTS the row is never selected
            # again. NOT NULL DEFAULT 0: rows already in the queue read 0,
            # never NULL, which is never under the cap and would stop every
            # comment.
            conn.execute(
                "ALTER TABLE deliveries ADD COLUMN first_comment_attempts "
                "INTEGER NOT NULL DEFAULT 0"
            )
        if "first_comment_prior_text" not in cols:
            # JSON list of texts a relink replaced (``rewrite_unposted_first_comment``);
            # the YouTube drain's dedup accepts a live comment saying any of them.
            conn.execute("ALTER TABLE deliveries ADD COLUMN first_comment_prior_text TEXT")
        if "first_comment_live_id" not in cols:
            # The id of the comment the drain recorded as the row's own when it marked it posted;
            # NULL when unknown (older rows, or no id). A second own comment saying the same text
            # is told apart from it by this.
            conn.execute("ALTER TABLE deliveries ADD COLUMN first_comment_live_id TEXT")
        if "first_comment_removed_at" not in cols:
            # Set when a delete of the row's live comment left it with no text to repost; only a
            # new live id (the drain reposts, or adopts our own copy) clears it.
            conn.execute("ALTER TABLE deliveries ADD COLUMN first_comment_removed_at TEXT")
        if "first_comment_correction" not in cols:
            # JSON list ``{live_id, live_text, action: edit|delete, to}`` for the drain to fix.
            conn.execute("ALTER TABLE deliveries ADD COLUMN first_comment_correction TEXT")
        if "first_comment_correction_attempts" not in cols:
            # Refusals of a stale-comment correction (apart from ``first_comment_attempts``).
            conn.execute(
                "ALTER TABLE deliveries ADD COLUMN first_comment_correction_attempts "
                "INTEGER NOT NULL DEFAULT 0"
            )
        for identity_col in ("content_sha256", "content_title_key"):
            # What the row publishes, so ``content_twin`` can see the same
            # footage queued under another project or index. NULL on every
            # row queued before these columns existed, and never backfilled:
            # such a row is simply not compared.
            if identity_col not in cols:
                conn.execute(f"ALTER TABLE deliveries ADD COLUMN {identity_col} TEXT")
        for recovery_col, recovery_type in (
            # ``recover_terminal_delivery``: NULL on every row that is not a
            # recovery row -- an honest "not a recovery", never backfilled.
            ("recovers_delivery_id", "INTEGER"),
            ("recovery_authorized_by", "TEXT"),
            ("recovery_evidence", "TEXT"),
        ):
            if recovery_col not in cols:
                conn.execute(f"ALTER TABLE deliveries ADD COLUMN {recovery_col} {recovery_type}")
        if "remote_checked_at" not in cols:
            # When ``youtube_adapter`` last saw this row's video in ``videos.list``;
            # NULL until it first looks, so an older row is checked first.
            conn.execute("ALTER TABLE deliveries ADD COLUMN remote_checked_at TEXT")
        if "uncounted_since" not in cols:
            # When the row's current run of retries that never count against
            # ``max_attempts`` began; NULL when it is not in one. Bounds that run
            # (``dispatch.UNCOUNTED_RETRY_BOUND``). NULL for older rows: their run
            # starts at the next such retry, never at a guessed earlier time.
            conn.execute("ALTER TABLE deliveries ADD COLUMN uncounted_since TEXT")
        pause_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(platform_pauses)").fetchall()
        }
        if "reason" not in pause_cols:
            # NULL for a pause set before this column existed, or set by
            # hand from the console -- an honest "no reason recorded", never
            # backfilled as an auto-pause it wasn't.
            conn.execute("ALTER TABLE platform_pauses ADD COLUMN reason TEXT")
        conn.commit()
    return conn


def enqueue(
    conn: sqlite3.Connection,
    project_name: str,
    short_index: int,
    platform: str,
    plan_hash: str,
    shadow: bool = False,
    sidecar_path: Optional[str] = None,
    scheduled_at: Optional[str] = None,
    first_comment_text: Optional[str] = None,
    content_sha256: Optional[str] = None,
    content_title_key: Optional[str] = None,
) -> int:
    """Insert a delivery, or return the existing id for the same 4-tuple
    (idempotent). Raises ``sqlite3.IntegrityError`` if a different
    ``plan_hash`` is already ACTIVE for this 3-tuple — DB-level, not app-level.

    A short already delivered for this 3-tuple is never republished.
    ``TERMINAL_STATUSES`` excludes ``published`` from the active-delivery
    index's purview by design (a cancelled row must be re-enqueueable), so
    nothing else stops a caller from spooling a second, different
    ``plan_hash`` after the first already published. If a published row
    exists for the 3-tuple, its id is returned and nothing new is written —
    same idempotent shape as the exact-4-tuple case above, never raises.
    A ``missing_remote`` row counts the same way (U-R1): its video was
    uploaded once, and re-uploading it is a human decision, never a re-plan.

    ``shadow=True`` marks the row as a rehearsal that must never be
    published. ``dispatch._select_due`` cannot see such a row, so no drain
    path can fire it; see ``docs/social-publishing-queue.md``. Defaults to
    False so a caller that omits it spools real, publishable ammunition —
    the safe default is the explicit one at the CLI, not here.

    ``sidecar_path`` records the exact publish-plan sidecar file this
    delivery was enqueued from, so a drain pass reads back the file that
    actually produced the row instead of guessing a directory convention
    that may not match the run's real output location; see
    ``docs/social-publishing-queue.md``. Left ``NULL`` when the caller
    doesn't know it (or predates this column).

    ``scheduled_at`` is the plan's publish time (ISO 8601, UTC). It lived
    only in the sidecar, and ``dispatch._select_due`` gated solely on
    ``next_attempt_at`` -- which ``enqueue`` leaves NULL -- so every row was
    due the instant it was spooled. Only the YouTube adapter passes
    ``publish_at``; Instagram, TikTok and Facebook publish immediately, so a
    single drain pass would fire a whole week of shorts at once. Recording it
    here puts the gate in the SQL selection, where no caller can take a
    different path around it. ``NULL`` means a pre-migration row and stays
    due.

    ``first_comment_text`` is the first comment a YouTube / Instagram /
    Facebook short carries (``first_comment.full_drive_comment_text``),
    posted ``FIRST_COMMENT_DELAY_S`` after publish -- see
    ``select_due_first_comments``. The TEXT is generated elsewhere; this
    module only stores and schedules it. ``NULL`` means no first comment is scheduled for this delivery.

    ``content_sha256`` / ``content_title_key`` are what the row publishes (the
    plan's media sha256, and on YouTube its normalised title), stored for
    ``content_twin``. Neither is checked here: the caller refuses a twin first."""
    now = _now_iso()
    with conn:
        existing = conn.execute(
            "SELECT id FROM deliveries WHERE project_name=? AND short_index=? "
            "AND platform=? AND plan_hash=?",
            (project_name, short_index, platform, plan_hash),
        ).fetchone()
        if existing is not None:
            return int(existing["id"])
        published = conn.execute(
            "SELECT id FROM deliveries WHERE project_name=? AND short_index=? "
            "AND platform=? AND status IN ('published', ?)",
            (project_name, short_index, platform, STATUS_MISSING_REMOTE),
        ).fetchone()
        if published is not None:
            return int(published["id"])
        cur = conn.execute(
            "INSERT INTO deliveries "
            "(project_name, short_index, platform, plan_hash, status, "
            " attempt_count, shadow, sidecar_path, scheduled_at, first_comment_text, "
            " content_sha256, content_title_key, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'queued', 0, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                project_name,
                short_index,
                platform,
                plan_hash,
                1 if shadow else 0,
                sidecar_path,
                scheduled_at,
                first_comment_text,
                content_sha256,
                content_title_key,
                now,
                now,
            ),
        )
        return int(cur.lastrowid)


def content_twin(
    conn: sqlite3.Connection,
    project_name: str,
    short_index: int,
    platform: str,
    *,
    content_sha256: Optional[str],
    content_title_key: Optional[str],
) -> Optional[sqlite3.Row]:
    """Another live row on ``platform`` that publishes the same content, or ``None``.

    The queue's own uniqueness knows a row only by ``(project_name, short_index,
    platform)``, so the same footage queued under another name -- a day reel
    (``day-reel-<drive file id>``) and a drive Short of the same clip, or two reels
    from two Drive files -- never collided and both published. A twin is a row with
    a DIFFERENT ``(project_name, short_index)`` holding the same media sha256, or the
    same normalised title, whose status is active, ``published`` or
    ``missing_remote``: a ``cancelled`` or ``failed_permanent`` row never went out,
    so it blocks nothing. A shadow row is a rehearsal and blocks nothing either.
    Rows queued before the identity columns existed hold NULL and are not compared."""
    clauses, params = [], []
    if content_sha256:
        clauses.append("content_sha256 = ?")
        params.append(content_sha256)
    if content_title_key:
        clauses.append("content_title_key = ?")
        params.append(content_title_key)
    if not clauses:
        return None
    return conn.execute(
        "SELECT * FROM deliveries WHERE platform = ? AND shadow = 0 "  # noqa: S608
        "AND status NOT IN ('cancelled', 'failed_permanent') "
        "AND NOT (project_name = ? AND short_index = ?) "
        f"AND ({' OR '.join(clauses)}) ORDER BY id LIMIT 1",
        (platform, project_name, short_index, *params),
    ).fetchone()


def registered_platform_id(
    platform: str,
    platform_upload_id: Optional[str],
    platform_post_id: Optional[str],
) -> Optional[str]:
    """The id metrics/tracking registers for a published delivery.

    YouTube's adapter stores the video id in ``platform_upload_id`` and the
    playlist id in ``platform_post_id`` -- the video id is the trackable
    content id. Every other platform's post id IS its trackable id.
    """
    return platform_upload_id if platform == "youtube" else platform_post_id


def recorded_publish_time(
    published_at: Optional[str],
    published_at_source: Optional[str],
    platform_published_at: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """The posting time tracking may record, and which moment it is.

    a time is only worth handing across if it says which moment it is --
    an ``operator_confirmed`` catch-up click and a ``platform_confirmed``
    publish second are hours apart, and tracking anchors age-based judgements
    to whatever it is given. So the pair travels intact or not at all.

    A row with a time and no source predates ``published_at_source`` (or was
    written by a path that never set it). Handing the bare time over would
    make an unknown look like a measurement, so both come back ``None`` and
    the platform id still registers -- an honest unknown, not a guess.

    ``platform_live`` carries the time the platform confirmed the video public in
    ``platform_published_at`` (``published_at`` is still the upload moment), so
    that is the time tracking takes; without it the row says nothing.
    """
    if published_at_source == PUBLISHED_AT_PLATFORM_LIVE:
        if platform_published_at:
            return platform_published_at, published_at_source
        return None, None
    if published_at and published_at_source:
        return published_at, published_at_source
    return None, None


#: How long after a publish its first comment is due. One constant, one
#: place -- ``select_due_first_comments`` is the only reader.
FIRST_COMMENT_DELAY_S = 300

#: Platforms whose shorts get a first comment. TikTok's Content Posting API
#: has no comment endpoint, so it is deliberately absent.
FIRST_COMMENT_PLATFORMS = ("youtube", "instagram", "facebook")

#: Posts with no recorded publish time (posts imported from before the column existed)
#: comment at most this many per platform per pass. All at once they would put
#: the same link under every one of them within a minute; the hourly pass
#: spreads them out instead.
FIRST_COMMENT_BACKLOG_PER_PASS = 1

#: A first comment the platform refuses this many times is never tried again:
#: its post is gone (a short taken down by hand) or closed to comments, and
#: the hourly pass would otherwise call the platform about it forever. Only a
#: refusal because of the post counts -- an outage, a 5xx, a rate limit or
#: spent quota, a lapsed token or a missing scope never does, so a long
#: outage can't use up a live short's tries. Each drain decides which is
#: which. The row keeps its last ``first_comment_error``.
FIRST_COMMENT_MAX_ATTEMPTS = 8


def select_due_first_comments(
    conn: sqlite3.Connection,
    now: Optional[str] = None,
    platforms: Optional[Iterable[str]] = None,
) -> list:
    """YouTube / Instagram / Facebook deliveries whose first comment is due.

    A row qualifies when: a comment text was supplied (``first_comment_text``
    not NULL or empty), none has posted yet (``first_comment_posted_at`` IS
    NULL), fewer than ``FIRST_COMMENT_MAX_ATTEMPTS`` tries have failed, the
    platform is not paused, the platform accepted the post (a
    comment target id exists), and the publish is at least
    ``FIRST_COMMENT_DELAY_S`` in the past. That is proven one of two ways:

    - ``recorded_publish_time`` would return a real, labelled publish moment
      -- ``published_at`` AND ``published_at_source`` both set -- that old.
    - The row is ``published`` with no source (it published before
      ``published_at_source`` existed)
      and its last write (``updated_at``) and its slot (``scheduled_at``, when
      set) are both that old. ``transition()`` stamps ``updated_at`` on the
      move to published and every later write only moves it forward, so it
      bounds the publish from above; a scheduled YouTube upload goes public
      at its slot. Without this those shorts could never get a comment. The
      bare, unlabelled ``published_at`` is never trusted. At most
      ``FIRST_COMMENT_BACKLOG_PER_PASS`` of these per platform per call,
      untried rows first, so one that keeps failing never starves the rest.

    ``platforms`` narrows the set (default: all three). Durable Page-photo
    attempts are excluded: their exact approval contains no first-comment
    authorization, including text inherited from enqueue or later backfill.

    Each returned dict carries ``comment_target_id``: the YouTube VIDEO id
    (``platform_upload_id`` -- ``platform_post_id`` is the playlist there) or
    the IG media / FB reel id (``platform_post_id``), so no drain ever picks
    the column itself. Read-only; same shape as ``dispatch._select_due``."""
    now = now or _now_iso()
    wanted = [p for p in (platforms or FIRST_COMMENT_PLATFORMS) if p in FIRST_COMMENT_PLATFORMS]
    if not wanted:
        return []
    marks = ",".join("?" for _ in wanted)
    delay = f"-{FIRST_COMMENT_DELAY_S} seconds"
    rows = conn.execute(
        f"SELECT * FROM deliveries WHERE platform IN ({marks}) "  # noqa: S608
        "AND (upload_kind IS NULL OR upload_kind <> ?) "
        "AND first_comment_text IS NOT NULL AND first_comment_text <> '' "
        "AND first_comment_posted_at IS NULL AND first_comment_attempts < ? "
        "AND ((published_at IS NOT NULL AND published_at_source IS NOT NULL "
        "AND datetime(published_at) <= datetime(?, ?)) "
        "OR (status = 'published' AND published_at_source IS NULL "
        "AND datetime(updated_at) <= datetime(?, ?) "
        "AND (scheduled_at IS NULL OR datetime(scheduled_at) <= datetime(?, ?)))) "
        "ORDER BY id",
        (
            *wanted,
            FACEBOOK_PHOTO_KIND,
            FIRST_COMMENT_MAX_ATTEMPTS,
            now,
            delay,
            now,
            delay,
            now,
            delay,
        ),
    ).fetchall()
    paused: dict = {}
    due = []
    backlog: dict = {}
    for row in rows:
        platform = row["platform"]
        if platform not in paused:
            paused[platform] = is_platform_paused(conn, platform)
        if paused[platform]:
            continue
        target = registered_platform_id(
            platform, row["platform_upload_id"], row["platform_post_id"]
        )
        if not target:
            continue
        entry = dict(row)
        entry["comment_target_id"] = target
        if row["published_at_source"] is None:
            backlog.setdefault(platform, []).append(entry)
        else:
            due.append(entry)
    for entries in backlog.values():
        entries.sort(key=lambda e: (e["first_comment_error"] is not None, e["id"]))
        due.extend(entries[:FIRST_COMMENT_BACKLOG_PER_PASS])
    return sorted(due, key=lambda e: e["id"])


def _register_platform_id_best_effort(conn: sqlite3.Connection, delivery_id: int) -> None:
    """Register the delivered platform id with the tracking store.

    First (and only) import edge from ``publishing/`` into ``tracking/`` --
    imported lazily so a module-level cycle can never form. Non-fatal: a
    publish must never fail because tracking bookkeeping did; ``ValueError``
    (short not yet registered) is the documented contract shared with the
    Phase-D post-upload hook (``phase_d._collect_metrics_best_effort``),
    anything else is swallowed too and logged.
    """
    row = conn.execute(
        "SELECT project_name, short_index, platform, platform_upload_id, "
        "platform_post_id, published_at, published_at_source, platform_published_at "
        "FROM deliveries WHERE id=?",
        (delivery_id,),
    ).fetchone()
    if row is None:
        return
    video_id = registered_platform_id(
        row["platform"], row["platform_upload_id"], row["platform_post_id"]
    )
    if not video_id:
        return
    try:
        from ..tracking import TrackingStore

        posted_at, posted_at_source = recorded_publish_time(
            row["published_at"], row["published_at_source"], row["platform_published_at"]
        )
        TrackingStore().update_platform_id(
            row["project_name"],
            row["short_index"],
            row["platform"],
            video_id,
            posted_at=posted_at,
            posted_at_source=posted_at_source,
        )
    except ValueError:
        logger.debug(
            "tracking: short #%s not registered for %s, skipping id registration",
            row["short_index"],
            row["project_name"],
        )
    except Exception as exc:  # noqa: BLE001 -- a publish must never fail on this
        logger.warning("tracking: failed to register platform id (non-fatal): %s", exc)


def transition(
    conn: sqlite3.Connection,
    delivery_id: int,
    expected_status: str,
    new_status: str,
    **fields: Any,
) -> bool:
    """Compare-and-set: succeeds only when current status equals
    ``expected_status`` and is not terminal; returns ``True`` iff changed."""
    # Photo attempts have dedicated atomic identity/receipt operations below.
    # Generic retries may update bookkeeping, never release an uncertain write.
    photo_identity_change = new_status != expected_status or set(fields) & {
        "upload_account",
        "upload_kind",
        "platform_upload_id",
        "platform_post_id",
        "destination_account",
        "published_at_source",
        "platform_published_at",
        "permalink",
    }
    if facebook_photo_attempt(conn, delivery_id) is not None and photo_identity_change:
        return False
    allowed = {
        "attempt_count",
        "next_attempt_at",
        "next_attempt_reason",
        "platform_upload_id",
        "upload_account",
        "platform_post_id",
        "permalink",
        "error_class",
        "error_detail",
        "uncounted_since",
        # Lease fields: a retryable transition releases the lease it held so
        # next_attempt_at alone governs when the row is retried.
        "lease_owner",
        "lease_expires_at",
    }
    # Popped before the unknown-field check below, like ``published_at``
    # itself: never a generic transition field, handled explicitly here.
    published_at_source = fields.pop("published_at_source", None)
    platform_published_at = fields.pop("platform_published_at", None)
    destination_account = fields.pop("destination_account", None)
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"unknown transition field(s): {sorted(unknown)}")
    if "upload_account" in fields:
        account = fields["upload_account"]
        if not isinstance(account, str) or not account.strip():
            raise ValueError("upload_account must name an account; omit it when unknown")
        fields["upload_account"] = account.strip()
    if published_at_source is not None and new_status != "published":
        raise ValueError("published_at_source is only valid on a transition to 'published'")
    if platform_published_at is not None and new_status != "published":
        raise ValueError("platform_published_at is only valid on a transition to 'published'")
    if destination_account is not None and new_status != "published":
        # An account name on a queued or failed row claims something landed
        # there when nothing did.
        raise ValueError("destination_account is only valid on a transition to 'published'")
    # ``next_attempt_reason`` is what lets ``retry_delivery`` tell an ordinary
    # backoff from a platform throttle or an awaiting-manual park -- a time
    # with no reason would leave every writer's wait indistinguishable again.
    # ``next_attempt_at=None`` clears the reason with it in the same UPDATE:
    # a cleared wait has no reason to remember.
    if "next_attempt_at" in fields:
        if fields["next_attempt_at"] is None:
            fields["next_attempt_reason"] = None
        elif fields.get("next_attempt_reason") not in _NEXT_ATTEMPT_REASONS:
            raise ValueError(
                "next_attempt_at requires next_attempt_reason to be one of "
                f"{_NEXT_ATTEMPT_REASONS}; got {fields.get('next_attempt_reason')!r}"
            )
    # Reaching "published" clears the failure of an earlier attempt, unless the
    # caller is deliberately writing one. Otherwise a live, successful delivery
    # keeps advertising an error it recovered from -- a published row would
    # still read error_class=RuntimeError from the attempt before.
    if new_status == "published":
        fields.setdefault("error_class", None)
        fields.setdefault("error_detail", None)
        # A caller reaching "published" without naming a valid source is a
        # bug -- raising makes an unlabelled receipt impossible to create.
        if published_at_source not in _PUBLISHED_AT_SOURCES:
            raise ValueError(
                "published_at_source must be one of "
                f"{_PUBLISHED_AT_SOURCES}; got {published_at_source!r}"
            )
        # published_at is derived here, never caller-supplied (it is
        # deliberately absent from ``allowed`` above, so passing it raises
        # ValueError before this point). The UPDATE below is additionally
        # gated on "status NOT IN ('published', ...)", so a row can reach
        # this branch and have it stick at most once -- a retry, lease
        # renewal or reconcile that touches an already-published row never
        # matches the WHERE clause and this value is never overwritten.
        # A scheduled hand-over is only meaningful with the time it was
        # scheduled FOR, and the two other platform sources already claim to
        # be the publish moment themselves -- carrying a second, different
        # time beside one of those would be two claims about one event with
        # nothing to say which a reader should believe.
        if published_at_source == PUBLISHED_AT_PLATFORM_SCHEDULED:
            if platform_published_at is None:
                raise ValueError(
                    f"{PUBLISHED_AT_PLATFORM_SCHEDULED!r} requires platform_published_at -- "
                    "without it the claim is weaker than 'platform_confirmed', not stronger"
                )
        elif platform_published_at is not None:
            raise ValueError(
                "platform_published_at is only valid with published_at_source="
                f"{PUBLISHED_AT_PLATFORM_SCHEDULED!r}; got {published_at_source!r}"
            )
        if platform_published_at is not None:
            # Parsed, never stored raw: an unparseable value reads as a real
            # platform receipt on every surface downstream.
            try:
                datetime.fromisoformat(str(platform_published_at).replace("Z", "+00:00"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"platform_published_at must be an ISO-8601 timestamp; "
                    f"got {platform_published_at!r}"
                ) from exc
            fields["platform_published_at"] = platform_published_at
        if destination_account is not None:
            # `""` and `"   "` are not unknowns: they are stored answers that
            # READ as unknowns on every surface while being indistinguishable
            # from a bug that dropped the real value. If there is nothing to
            # say, the column stays NULL.
            if not str(destination_account).strip():
                raise ValueError(
                    "destination_account must name an account or be omitted; "
                    f"got {destination_account!r}"
                )
            fields["destination_account"] = str(destination_account).strip()
        fields["published_at"] = _now_iso()
        fields["published_at_source"] = published_at_source
    set_cols = ", ".join(f"{k}=?" for k in fields)
    set_clause = f"status=?, updated_at=?{', ' + set_cols if set_cols else ''}"
    params = [new_status, _now_iso(), *fields.values()]
    if new_status == "submitting":
        # The first submit only: every later attempt keeps it (COALESCE).
        set_clause += ", first_submitted_at=COALESCE(first_submitted_at, ?)"
        params.append(_now_iso())
    if fields.get("platform_upload_id"):
        set_clause += ", uploaded_at=COALESCE(uploaded_at, ?)"
        params.append(_now_iso())
    params += [delivery_id, expected_status]
    account_guard = ""
    photo_guard = (
        " AND NOT EXISTS (SELECT 1 FROM facebook_photo_attempts WHERE delivery_id=deliveries.id)"
        if photo_identity_change
        else ""
    )
    if "upload_account" in fields:
        # A checkpoint may be repeated but never rebound to rotated credentials.
        account_guard = " AND (upload_account IS NULL OR upload_account=?)"
        params.append(fields["upload_account"])
    with conn:  # set_clause: built only from the ``allowed`` allowlist above
        cur = conn.execute(
            f"UPDATE deliveries SET {set_clause} "  # noqa: S608
            f"WHERE id=? AND status=? AND status NOT IN {_TERMINAL_SQL}{account_guard}{photo_guard}",
            params,
        )
        changed = cur.rowcount == 1
    if changed and new_status == "published":
        _register_platform_id_best_effort(conn, delivery_id)
    return changed


def restart_delivery(
    conn: sqlite3.Connection, delivery_id: int, expected_status: str, **fields: Any
) -> bool:
    """Start a delivery over: back to ``queued`` with its upload id cleared and
    ``attempt_count`` 0. Every adapter path that abandons an upload it has
    PROVEN can never publish (an expired Instagram container, a Facebook reel
    whose finish never ran) goes through here.

    The count resets because the tries it holds were spent on the abandoned
    upload, often uncounted reads that still bumped it: kept, the fresh upload
    failed permanently on its first ordinary error. ``uncounted_since`` is kept:
    a fresh upload that is again left undecided is the same stuck episode, so
    the bound (``dispatch.UNCOUNTED_RETRY_BOUND``) still parks it rather than a
    restart every 24 h looping for good. Never use this for a try that simply
    failed; that try is counted like any other. Compare-and-set like
    ``transition``; ``fields`` are extra columns to write (an error note)."""
    return transition(
        conn, delivery_id, expected_status, "queued", platform_upload_id=None, attempt_count=0,
        **fields,
    )  # fmt: skip


def acquire_lease(
    conn: sqlite3.Connection,
    delivery_id: int,
    owner: str,
    expires_at: str,
    now: Optional[str] = None,
) -> bool:
    """Acquire/reclaim/renew the worker lease: succeeds when unheld, when the
    held lease's ``lease_expires_at <= now``, or when ``owner`` already holds
    it. Never on a terminal delivery or another worker's live lease. ``now``
    defaults to current UTC time.

    The owner clause is load-bearing, not a convenience. ``drain_queue`` takes
    the lease and then hands the row to an adapter that takes it AGAIN before
    the irreversible API call. Without it that second call can never succeed,
    and every adapter raises RuntimeError("... leased by another worker") --
    against its own pass. The first real drain failed every due delivery
    exactly so, and no delivery could ever have published.
    """
    now = now or _now_iso()
    with conn:
        cur = conn.execute(
            "UPDATE deliveries SET lease_owner=?, "  # noqa: S608 -- fixed literal, not input
            "lease_expires_at=?, updated_at=? "
            "WHERE id=? "
            f"AND status NOT IN {_TERMINAL_SQL} "
            "AND (lease_owner IS NULL OR lease_expires_at <= ? OR lease_owner = ?)",
            (owner, expires_at, _now_iso(), delivery_id, now, owner),
        )
        return cur.rowcount == 1


def _has_live_foreign_lease(conn: sqlite3.Connection, delivery_id: int, now: str) -> bool:
    """True iff the row is held by a lease that has not yet expired.

    An operator action (cancel/reschedule) must never race a worker that may
    already have made an irreversible platform API call; an expired lease is
    not "live" and does not block."""
    row = conn.execute(
        "SELECT lease_owner, lease_expires_at FROM deliveries WHERE id=?",
        (delivery_id,),
    ).fetchone()
    if row is None:
        return False
    return row["lease_owner"] is not None and (row["lease_expires_at"] or "") > now


def cancel_delivery(conn: sqlite3.Connection, delivery_id: int, now: Optional[str] = None) -> bool:
    """Move a non-terminal, non-live-leased row to the terminal ``cancelled``
    status. Returns ``False`` (never raises) on an ordinary refusal --
    already terminal, or held by another worker's live lease, which may
    already be mid platform-API-call. An expired lease may be cancelled unless
    a durable photo attempt exists; its uncertain write cannot be replaced."""
    now = now or _now_iso()
    if _has_live_foreign_lease(conn, delivery_id, now) or facebook_photo_attempt(conn, delivery_id):
        return False
    with conn:
        cur = conn.execute(
            "UPDATE deliveries SET status='cancelled', updated_at=? "  # noqa: S608 -- fixed literal
            f"WHERE id=? AND status NOT IN {_TERMINAL_SQL} "
            "AND NOT EXISTS (SELECT 1 FROM facebook_photo_attempts WHERE delivery_id=deliveries.id)",
            (_now_iso(), delivery_id),
        )
        return cur.rowcount == 1


def record_live_publish(
    conn: sqlite3.Connection,
    delivery_id: int,
    live_at: str,
    now: Optional[str] = None,
) -> bool:
    """Replace a scheduled row's slot with the time the platform confirmed it public.

    Only a ``published`` row whose source is ``platform_scheduled`` moves; it
    becomes ``platform_live`` with ``platform_published_at=live_at`` and tracking
    ``posted_at`` follows. ``published_at`` (the upload moment) is untouched.
    ``False`` when the row is not in that state, so a repeat is a no-op."""
    now = now or _now_iso()
    try:
        datetime.fromisoformat(str(live_at).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"live_at must be an ISO-8601 timestamp; got {live_at!r}") from exc
    with conn:
        cur = conn.execute(
            "UPDATE deliveries SET platform_published_at=?, published_at_source=?, "
            "updated_at=? WHERE id=? AND status='published' AND published_at_source=?",
            (
                live_at,
                PUBLISHED_AT_PLATFORM_LIVE,
                now,
                delivery_id,
                PUBLISHED_AT_PLATFORM_SCHEDULED,
            ),
        )
    if cur.rowcount != 1:
        return False
    _register_platform_id_best_effort(conn, delivery_id)
    return True


def parse_slot(value: Any) -> datetime:
    """``value`` read as an ISO-8601 time (a trailing ``Z`` is UTC), or
    ``ValueError`` naming it. A time with no zone comes back without one:
    ``normalise_slot`` refuses it, and the drain reads it as UTC."""
    try:
        return datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"scheduled_at must be an ISO-8601 timestamp; got {value!r}") from exc


def normalise_slot(value: Any) -> str:
    """``value`` as a UTC ISO-8601 string, the one form ``scheduled_at`` holds.

    ``dispatch._select_due`` compares ``scheduled_at`` with its own UTC ``now``
    as TEXT, so "09:00-05:00", "14:00Z" or "14:00" with a space would all sort
    wrong against it even where they parse. A time with no zone is refused, not
    guessed: it could be Eastern or UTC, and the drain's arithmetic on it raised
    ``TypeError``."""
    parsed = parse_slot(value)
    if parsed.tzinfo is None:
        raise ValueError(
            f"scheduled_at must name its time zone (for example +00:00 or -05:00); "
            f"got {value!r}, which could be any zone"
        )
    return parsed.astimezone(timezone.utc).isoformat()


def reschedule_delivery(
    conn: sqlite3.Connection,
    delivery_id: int,
    scheduled_at: str,
    now: Optional[str] = None,
) -> bool:
    """Set a non-terminal, non-live-leased row's ``scheduled_at``.

    A durable photo attempt is immutable even after its lease expires.

    Not a ``transition`` field (rescheduling changes no status), so this is
    its own guarded ``UPDATE``. A row waiting on retry backoff carries both
    ``next_attempt_at`` and ``scheduled_at``; ``dispatch._select_due`` gates
    on both. Moving the schedule FORWARD past a stale (already-past)
    ``next_attempt_at`` clears it -- an operator pushing a slot out means
    "not before this new time", and an old backoff timestamp already in the
    past would otherwise leave the row due immediately once the new schedule
    arrives. A ``next_attempt_at`` still in the FUTURE is left alone whatever
    its ``next_attempt_reason`` -- ``dispatch._select_due`` gates on both
    times, so shortening either one here would let an unrelated reschedule
    fire the row before its own wait (backoff, throttle, or deferral) has
    actually elapsed.

    ``scheduled_at`` must be a readable timestamp with a zone, and is stored
    as UTC (``normalise_slot``). The column is TEXT and
    ``dispatch._select_due`` compares it as a string, so an unparseable or
    differently written value does not fail loudly -- it makes the row never
    due, or due for ever. Refusing here rather than at any one caller means no
    caller can write one; ``transition`` guards ``platform_published_at`` the
    same way. This writes the row only: ``plan_title_sync.move_delivery_slot``
    moves the publish plan's slot with it."""
    now = now or _now_iso()
    scheduled_at = normalise_slot(scheduled_at)
    if _has_live_foreign_lease(conn, delivery_id, now) or facebook_photo_attempt(conn, delivery_id):
        return False
    with conn:
        row = conn.execute(
            "SELECT next_attempt_at FROM deliveries WHERE id=?", (delivery_id,)
        ).fetchone()
        clear_next_attempt = (
            row is not None and row["next_attempt_at"] is not None and row["next_attempt_at"] < now
        )
        if clear_next_attempt:
            cur = conn.execute(
                "UPDATE deliveries SET scheduled_at=?, "  # noqa: S608 -- fixed literal
                "next_attempt_at=NULL, next_attempt_reason=NULL, updated_at=? "
                f"WHERE id=? AND status NOT IN {_TERMINAL_SQL} "
                "AND NOT EXISTS (SELECT 1 FROM facebook_photo_attempts WHERE delivery_id=deliveries.id)",
                (scheduled_at, _now_iso(), delivery_id),
            )
        else:
            cur = conn.execute(
                "UPDATE deliveries SET scheduled_at=?, updated_at=? "  # noqa: S608 -- fixed literal
                f"WHERE id=? AND status NOT IN {_TERMINAL_SQL} "
                "AND NOT EXISTS (SELECT 1 FROM facebook_photo_attempts WHERE delivery_id=deliveries.id)",
                (scheduled_at, _now_iso(), delivery_id),
            )
        return cur.rowcount == 1


#: ``platform_pauses.reason`` of a pause taken because the platform refused its
#: access token (``handlers._pause_for_auth``); the refusal follows after a
#: space. ``health check`` raises one ``platform_auth_blocked`` incident for each
#: platform paused this way and resolves it once the pause is lifted.
AUTH_BLOCKED_PAUSE_REASON = "auth:blocked"


def pause_platform(
    conn: sqlite3.Connection,
    platform: str,
    now: Optional[str] = None,
    reason: Optional[str] = None,
) -> bool:
    """Pause a platform: ``dispatch._select_due`` stops selecting its rows
    for the NEXT drain pass. Stored in this DB, not console settings --
    ``dispatch-queue`` runs as a separate process from the console, and
    only the queue file is visible to both. Governs pickup only: a row
    already leased and in flight when the pause takes effect is left
    alone, never mutated, because the platform API call it's mid-flight on
    cannot be un-sent. Refuses (``False``, never raises) an unrecognized
    platform name rather than creating a pause row nothing will ever match.

    ``reason`` is optional and free-form (``None`` for a manual console
    pause, ``"auto:suppressed"`` for the auto-pause path,
    ``AUTH_BLOCKED_PAUSE_REASON`` for a refused access token)."""
    # Lazy import: queue.py is deliberately independent of publishing.models
    # (see module docstring) except for this one shared vocabulary check.
    from .models import KNOWN_PLATFORMS

    if platform not in KNOWN_PLATFORMS:
        return False
    now = now or _now_iso()
    with conn:
        conn.execute(
            "INSERT INTO platform_pauses (platform, paused_at, reason) VALUES (?, ?, ?) "
            "ON CONFLICT(platform) DO UPDATE SET paused_at=excluded.paused_at, "
            "reason=excluded.reason",
            (platform, now, reason),
        )
    return True


def resume_platform(conn: sqlite3.Connection, platform: str) -> bool:
    """Undo ``pause_platform``: selection resumes with no other side
    effect -- no backfill, no catch-up burst throttle. Rows that piled up
    while paused simply become due again on the next drain pass; if that
    pass is large, that is a real, visible consequence of how long the
    platform was paused, not a bug to mask here. Returns ``False`` (not an
    error) if the platform wasn't paused."""
    with conn:
        cur = conn.execute("DELETE FROM platform_pauses WHERE platform=?", (platform,))
        return cur.rowcount == 1


def is_platform_paused(conn: sqlite3.Connection, platform: str) -> bool:
    """True iff ``platform`` currently has an active pause row."""
    row = conn.execute("SELECT 1 FROM platform_pauses WHERE platform=?", (platform,)).fetchone()
    return row is not None


def set_platform_daily_cap(
    conn: sqlite3.Connection,
    platform: str,
    cap: int,
    state: str,
    mode: str,
    reason: Optional[str] = None,
    now: Optional[str] = None,
) -> None:
    """Record the latest bounded-response cap decision for
    ``platform`` (upserted -- one row per platform holds only the CURRENT
    decision; the full history is ``response_log``). ``mode`` is
    ``"shadow"`` or ``"live"`` -- ``dispatch._select_due`` only ENFORCES a
    ``"live"`` row; a ``"shadow"`` row is logged, never enforced."""
    now = now or _now_iso()
    with conn:
        conn.execute(
            "INSERT INTO platform_daily_caps (platform, cap, state, mode, set_at, reason) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(platform) DO UPDATE SET cap=excluded.cap, state=excluded.state, "
            "mode=excluded.mode, set_at=excluded.set_at, reason=excluded.reason",
            (platform, cap, state, mode, now, reason),
        )


def get_platform_daily_cap(conn: sqlite3.Connection, platform: str) -> Optional[Dict[str, Any]]:
    """The latest bounded-response cap decision for ``platform``, or
    ``None`` if no cap has ever been decided."""
    row = conn.execute("SELECT * FROM platform_daily_caps WHERE platform=?", (platform,)).fetchone()
    return dict(row) if row is not None else None


def clear_platform_daily_cap(conn: sqlite3.Connection, platform: str) -> bool:
    """Remove ``platform``'s cap decision entirely -- how a manual
    console override is lifted. Returns ``False`` (never raises) if there
    was no row to clear."""
    with conn:
        cur = conn.execute("DELETE FROM platform_daily_caps WHERE platform=?", (platform,))
        return cur.rowcount == 1


def log_response(
    conn: sqlite3.Connection,
    platform: str,
    state: str,
    cap: int,
    action: str,
    mode: str,
    reason: Optional[str] = None,
    now: Optional[str] = None,
) -> int:
    """Append one audit row to ``response_log`` -- shadow or live,
    every decision, so the 14-day shadow log can be compared against what
    an operator would have done. Returns the new row's id."""
    now = now or _now_iso()
    with conn:
        cur = conn.execute(
            "INSERT INTO response_log (at, platform, state, cap, action, mode, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (now, platform, state, cap, action, mode, reason),
        )
        return cur.lastrowid


#: ``next_attempt_reason`` values a manual retry refuses to override: a
#: platform-imposed throttle only the platform clears, and an
#: awaiting-manual park that is not a wait at all -- a human must act in the
#: platform's own app. Every other reason (an ordinary retry backoff, this
#: queue's own upload-budget deferral, or a pre-migration NULL, which is
#: treated as the common case, an ordinary backoff) is an eligible retry.
_NEXT_ATTEMPT_REASONS_RETRY_REFUSES = frozenset(
    {
        NEXT_ATTEMPT_PLATFORM_THROTTLE,
        NEXT_ATTEMPT_AWAITING_MANUAL,
        NEXT_ATTEMPT_AWAITING_IG_RECONCILE,
        NEXT_ATTEMPT_NEEDS_OPERATOR,
    }
)


def retry_delivery(conn: sqlite3.Connection, delivery_id: int, now: Optional[str] = None) -> bool:
    """Clear a backing-off row's ``next_attempt_at`` so the next drain pass
    treats it as due immediately -- the eligible manual retry. Same refusal
    shape as ``cancel_delivery``: returns ``False`` (never raises) on a
    terminal row or one held by another worker's live lease, which may
    already be mid platform-API-call. Also refuses -- leaving the row
    untouched -- when ``next_attempt_reason`` says the wait is not this
    queue's to end: see ``_NEXT_ATTEMPT_REASONS_RETRY_REFUSES``. Durable photo
    attempts cannot be manually retried; retained-ID reads use normal backoff.

    Deliberately does NOT touch ``attempt_count``: resetting it would let a
    delivery dodge ``dispatch.drain_queue``'s max-attempts ceiling forever
    via repeated manual retries, and incrementing it would burn an attempt
    against a call that hasn't actually happened yet. The count only moves
    when the drain loop itself makes (and grades) the next real attempt."""
    now = now or _now_iso()
    if _has_live_foreign_lease(conn, delivery_id, now) or facebook_photo_attempt(conn, delivery_id):
        return False
    row = conn.execute(
        "SELECT next_attempt_reason FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    if row is not None and row["next_attempt_reason"] in _NEXT_ATTEMPT_REASONS_RETRY_REFUSES:
        return False
    with conn:
        cur = conn.execute(
            "UPDATE deliveries SET next_attempt_at=NULL, next_attempt_reason=NULL, "  # noqa: S608
            "updated_at=? "
            f"WHERE id=? AND status NOT IN {_TERMINAL_SQL} "
            "AND NOT EXISTS (SELECT 1 FROM facebook_photo_attempts WHERE delivery_id=deliveries.id)",
            (_now_iso(), delivery_id),
        )
        return cur.rowcount == 1


#: The only rows ``mark_missing_remote`` moves: a YouTube upload that went
#: through (``uploaded``) or was recorded as done (``published``).
_MISSING_REMOTE_FROM = ("uploaded", "published")


def mark_missing_remote(
    conn: sqlite3.Connection,
    delivery_id: int,
    *,
    evidence: Dict[str, Any],
    now: Optional[str] = None,
) -> bool:
    """U-R1: record that YouTube no longer returns this row's video.

    Moves an ``uploaded``/``published`` YouTube row that carries a
    ``platform_upload_id`` to the terminal ``missing_remote`` and appends a
    ``response_log`` audit row; returns ``False`` (writing nothing) for any
    other row. A direct UPDATE because ``transition`` refuses to move a
    ``published`` row. The upload id stays on the row as the receipt of what
    went missing. Nothing remote is touched: no delete, no re-upload."""
    now = now or _now_iso()
    with conn:
        row = conn.execute(
            "SELECT platform, platform_upload_id FROM deliveries WHERE id=?", (delivery_id,)
        ).fetchone()
        if row is None or row["platform"] != "youtube" or not row["platform_upload_id"]:
            return False
        video_id = row["platform_upload_id"]
        cur = conn.execute(
            "UPDATE deliveries SET status=?, error_class='MissingRemote', error_detail=?, "
            "updated_at=? WHERE id=? AND status IN (?, ?)",
            (
                STATUS_MISSING_REMOTE,
                f"YouTube returns no video {video_id} for this delivery",
                now,
                delivery_id,
                *_MISSING_REMOTE_FROM,
            ),
        )
        if cur.rowcount != 1:
            return False
        conn.execute(
            "INSERT INTO response_log (at, platform, state, cap, action, mode, reason) "
            "VALUES (?, 'youtube', ?, 0, 'mark_missing_remote', 'live', ?)",
            (
                now,
                STATUS_MISSING_REMOTE,
                json.dumps(
                    {"delivery_id": delivery_id, "video_id": video_id, "evidence": evidence},
                    sort_keys=True,
                    default=str,
                ),
            ),
        )
    return True


def mark_remote_checked(conn: sqlite3.Connection, delivery_ids: Iterable[int], now: str) -> None:
    """Stamp ``remote_checked_at`` on rows whose video ``videos.list`` just returned.

    ``updated_at`` is deliberately left alone: for a legacy published row it is the
    upper bound ``select_due_first_comments`` uses for the publish moment, so a daily
    check moving it would keep pushing that row's first comment back."""
    ids = [int(i) for i in delivery_ids]
    if not ids:
        return
    marks = ",".join("?" for _ in ids)
    with conn:
        conn.execute(
            f"UPDATE deliveries SET remote_checked_at=? WHERE id IN ({marks})",  # noqa: S608
            (now, *ids),
        )


def find_delivery(
    conn: sqlite3.Connection,
    project_name: str,
    short_index: int,
    platform: str,
) -> Optional[Dict[str, Any]]:
    """Idempotency lookup for reconciliation: most recent delivery (if any)
    for this 3-tuple, including platform upload/post IDs."""
    row = conn.execute(
        "SELECT * FROM deliveries WHERE project_name=? AND short_index=? AND platform=? "
        "ORDER BY id DESC LIMIT 1",
        (project_name, short_index, platform),
    ).fetchone()
    return dict(row) if row is not None else None


def first_comments_naming(
    conn: sqlite3.Connection, project_name: str, needles: Iterable[str]
) -> list:
    """Deliveries of ``project_name`` whose ``first_comment_text`` or pending correction JSON
    contains a needle (plain substring), in id order, whatever their state."""
    needles = [n for n in needles if n]
    if not needles:
        return []
    clause = " OR ".join(
        "instr(first_comment_text, ?) > 0 OR instr(COALESCE(first_comment_correction, ''), ?) > 0"
        for _ in needles
    )
    return conn.execute(
        "SELECT id, short_index, platform, status, first_comment_text, "  # noqa: S608
        "first_comment_posted_at, first_comment_attempts, first_comment_error, lease_owner, "
        "lease_expires_at, scheduled_at, first_comment_correction, first_comment_correction_attempts, "
        "platform_upload_id, platform_post_id, upload_kind "
        f"FROM deliveries WHERE project_name=? AND ({clause}) ORDER BY id",
        (project_name, *[n for n in needles for _ in (0, 1)]),
    ).fetchall()


def first_comments_naming_elsewhere(
    conn: sqlite3.Connection, project_name: str, needles: Iterable[str]
) -> list:
    """Deliveries of every project EXCEPT ``project_name`` whose ``first_comment_text``
    contains a needle (plain substring), in id order. Report-only: a relink never rewrites
    another project's row, it only names it."""
    needles = [n for n in needles if n]
    if not needles:
        return []
    clause = " OR ".join("instr(first_comment_text, ?) > 0" for _ in needles)
    return conn.execute(
        "SELECT id, project_name, short_index, platform, status, first_comment_text, "  # noqa: S608
        "first_comment_posted_at FROM deliveries "
        f"WHERE project_name<>? AND ({clause}) ORDER BY id",
        (project_name, *needles),
    ).fetchall()


_URL_ID = re.compile(r"(?:youtu\.be/|[?&]v=|/shorts/)([A-Za-z0-9_-]+)")


def stale_live_error(live_text: str, current_text: str) -> str:
    """The ``first_comment_error`` for a comment that is live with a text the row no longer
    holds: names each video id the live text links that the current text does not."""
    ids = [i for i in dict.fromkeys(_URL_ID.findall(live_text)) if i not in current_text]
    return "stale comment live: links " + (", ".join(ids) if ids else "an id the relink replaced")


def first_comment_corrections(row: Any) -> list:
    """The row's pending corrections, as dicts."""
    try:
        raw = row["first_comment_correction"]
        items = json.loads(raw) if raw else []
    except (KeyError, IndexError, TypeError, ValueError):
        return []
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []


def _same_correction(a: dict, b: dict) -> bool:
    return a.get("live_id") == b.get("live_id") and a.get("live_text") == b.get("live_text")


def _same_applied(a: dict, b: dict) -> bool:
    """``a`` is ``b`` as it was read: same comment, text, action AND intended ``to``."""
    return (
        _same_correction(a, b) and a.get("action") == b.get("action") and a.get("to") == b.get("to")
    )


def first_comment_intended_texts(row: Any) -> set:
    """The texts the row means to have live: every pending edit's ``to``, plus its current text
    unless a pending edit is still fixing a comment that says it (the row text is then the stale
    one a drain recorded, not an intended one)."""
    current = (row["first_comment_text"] or "").strip()
    items = first_comment_corrections(row)
    texts = {(i.get("to") or "").strip() for i in items}
    if not any(
        i.get("action") == "edit" and (i.get("live_text") or "").strip() == current for i in items
    ):
        texts.add(current)
    return texts - {""}


def record_stale_live_comment(
    conn: sqlite3.Connection, delivery_id: int, live_text: str, now: str, live_id: str = ""
) -> Optional[str]:
    """Comment ``live_id`` saying ``live_text`` is live but the row holds other text. Unposted
    row: store the live text, mark it posted, queue an ``edit`` to the intended text. Posted row
    (concurrent drain, both texts live): queue a ``delete``, never of an intended text; a
    live pending edit's ``to`` is ADOPTED (row text, its edits become deletes). Returns the error
    set, or ``None`` when nothing changed."""
    with conn:
        row = conn.execute(
            "SELECT first_comment_text, first_comment_posted_at, first_comment_correction, "
            "first_comment_live_id FROM deliveries WHERE id=?",
            (delivery_id,),
        ).fetchone()
        if row is None:
            return None
        current = row["first_comment_text"] or ""
        items = first_comment_corrections(row)
        error = stale_live_error(live_text, current)
        item = {"live_id": live_id, "live_text": live_text}
        if any(_same_correction(i, item) for i in items):
            return None
        if row["first_comment_posted_at"] is not None:
            said = live_text.strip()
            intended = first_comment_intended_texts(row)
            if said == current.strip() and said in intended:
                known = row["first_comment_live_id"]
                if live_id and not known:
                    # An old row: the comment it recorded as posted is unknown, so which copy is
                    # ours to keep cannot be told. Nothing is deleted; the operator is told once.
                    error = (
                        f"a second copy of the comment may be live (comment {live_id}); this "
                        "row has no recorded comment id, so check and delete one by hand"
                    )
                    done = conn.execute(
                        "UPDATE deliveries SET first_comment_error=?, updated_at=? WHERE id=? "
                        "AND first_comment_posted_at IS NOT NULL AND first_comment_live_id IS NULL "
                        "AND (first_comment_error IS NULL OR first_comment_error!=?)",
                        (error, now, delivery_id, error),
                    ).rowcount
                    return error if done == 1 else None
                if not (live_id and known and live_id != known):
                    return None
                # A second comment of ours saying the intended text beside the recorded one:
                # delete it (``extra``: the delete's own text is an intended one, still to go).
                error = f"a second copy of the comment is live (comment {live_id}); deleting it"
                done = conn.execute(
                    "UPDATE deliveries SET first_comment_error=?, first_comment_correction=?, "
                    "updated_at=? WHERE id=? AND first_comment_posted_at IS NOT NULL "
                    "AND first_comment_live_id=?",
                    (
                        error,
                        json.dumps(
                            [*items, {**item, "action": "delete", "extra": True, "ours": True}]
                        ),
                        now,
                        delivery_id,
                        known,
                    ),
                ).rowcount
                return error if done == 1 else None
            if said in intended:
                items = [
                    {**i, "action": "delete"}
                    if i.get("action") == "edit" and (i.get("to") or "").strip() == said
                    else i
                    for i in items
                    if not (live_id and i.get("live_id") == live_id)
                ]
                error = stale_live_error(current, live_text) if items else None
                done = conn.execute(
                    "UPDATE deliveries SET first_comment_text=?, first_comment_live_id=?, "
                    "first_comment_error=?, first_comment_correction=?, updated_at=? WHERE id=? "
                    "AND first_comment_posted_at IS NOT NULL AND first_comment_text=?",
                    (
                        live_text,
                        live_id or None,
                        error,
                        json.dumps(items) if items else None,
                        now,
                        delivery_id,
                        current,
                    ),
                ).rowcount
                return error if done == 1 else None
            done = conn.execute(
                "UPDATE deliveries SET first_comment_error=?, first_comment_correction=?, "
                "updated_at=? WHERE id=? AND first_comment_posted_at IS NOT NULL",
                (error, json.dumps([*items, {**item, "action": "delete"}]), now, delivery_id),
            ).rowcount
        else:
            if current == live_text:
                return None
            # An empty current text (a private relink dropped it) never becomes an edit to "":
            # the live comment goes, and the row keeps no text (a delete of an intended text
            # never runs, and none is intended).
            fix = (
                {**item, "action": "edit", "to": current}
                if current
                else {**item, "action": "delete"}
            )
            done = conn.execute(
                "UPDATE deliveries SET first_comment_text=?, first_comment_posted_at=?, "
                "first_comment_live_id=?, first_comment_error=?, first_comment_correction=?, "
                "updated_at=? WHERE id=? AND first_comment_posted_at IS NULL",
                (
                    live_text if current else None,
                    now,
                    live_id or None,
                    error,
                    json.dumps([*items, fix]),
                    now,
                    delivery_id,
                ),
            ).rowcount
    return error if done == 1 else None


def first_comment_correction_ids(conn: sqlite3.Connection, platforms: Iterable[str]) -> set:
    """Ids of deliveries on ``platforms`` with a pending correction."""
    wanted = list(platforms)
    marks = ",".join("?" for _ in wanted)
    return {
        r[0]
        for r in conn.execute(
            "SELECT id FROM deliveries WHERE first_comment_correction IS NOT NULL "  # noqa: S608
            f"AND platform IN ({marks})",
            wanted,
        )
    }


def pending_first_comment_corrections(conn: sqlite3.Connection, platforms: Iterable[str]) -> list:
    """Posted rows with a correction to make: under the attempt cap, platform not paused."""
    wanted = list(platforms)
    marks = ",".join("?" for _ in wanted)
    rows = conn.execute(
        "SELECT id, project_name, short_index, platform, first_comment_text, platform_upload_id, "  # noqa: S608
        "platform_post_id, first_comment_error, first_comment_correction, scheduled_at, "
        "first_comment_live_id "
        f"FROM deliveries WHERE platform IN ({marks}) AND first_comment_correction IS NOT NULL "
        "AND first_comment_posted_at IS NOT NULL AND first_comment_correction_attempts < ? "
        "ORDER BY id",
        (*wanted, FIRST_COMMENT_MAX_ATTEMPTS),
    ).fetchall()
    paused: dict = {}
    for row in rows:
        if row["platform"] not in paused:
            paused[row["platform"]] = is_platform_paused(conn, row["platform"])
    return [r for r in rows if not paused[r["platform"]]]


def capped_first_comment_corrections(conn: sqlite3.Connection, platforms: Iterable[str]) -> list:
    """Posted rows whose correction hit ``FIRST_COMMENT_MAX_ATTEMPTS`` refusals (a human acts)."""
    wanted = list(platforms)
    marks = ",".join("?" for _ in wanted)
    return conn.execute(
        "SELECT id, project_name, short_index, platform, first_comment_text, "  # noqa: S608
        "first_comment_correction FROM deliveries "
        f"WHERE platform IN ({marks}) AND first_comment_correction IS NOT NULL "
        "AND first_comment_posted_at IS NOT NULL AND first_comment_correction_attempts >= ? "
        "ORDER BY id",
        (*wanted, FIRST_COMMENT_MAX_ATTEMPTS),
    ).fetchall()


def unposted_first_comment_ids(conn: sqlite3.Connection, platforms: Iterable[str]) -> set:
    """Ids of deliveries on ``platforms`` whose first comment is not posted."""
    wanted = list(platforms)
    marks = ",".join("?" for _ in wanted)
    return {
        r[0]
        for r in conn.execute(
            "SELECT id FROM deliveries WHERE first_comment_posted_at IS NULL "  # noqa: S608
            f"AND platform IN ({marks})",
            wanted,
        )
    }


def removed_first_comment_rows(conn: sqlite3.Connection, platforms: Iterable[str]) -> list:
    """Posted rows on ``platforms`` whose live comment was deleted with no text to replace it
    (``mark_first_comment_removed``); the incident stays open until a NEW live id is recorded."""
    wanted = list(platforms)
    marks = ",".join("?" for _ in wanted)
    return conn.execute(
        "SELECT id, project_name, short_index, sidecar_path, scheduled_at, first_comment_text "  # noqa: S608
        "FROM deliveries WHERE first_comment_posted_at IS NOT NULL "
        f"AND first_comment_removed_at IS NOT NULL AND platform IN ({marks}) ORDER BY id",
        wanted,
    ).fetchall()


def removed_first_comment_ids(conn: sqlite3.Connection, platforms: Iterable[str]) -> set:
    """Ids of ``removed_first_comment_rows``."""
    return {r["id"] for r in removed_first_comment_rows(conn, platforms)}


def mark_first_comment_removed(conn: sqlite3.Connection, delivery_id: int, now: str) -> bool:
    """A delete landed: when the posted row has no text, mark it removed. ``posted_at`` stays."""
    with conn:
        return bool(
            conn.execute(
                "UPDATE deliveries SET first_comment_removed_at=COALESCE(first_comment_removed_at, ?) "
                "WHERE id=? AND first_comment_posted_at IS NOT NULL "
                "AND (first_comment_text IS NULL OR first_comment_text='')",
                (now, delivery_id),
            ).rowcount
        )


def unpost_removed_first_comment(
    conn: sqlite3.Connection, delivery_id: int, text: str, now: str
) -> bool:
    """A removed row has comment text again: unpost it with that text, the way a gone edit does
    (``unpost_first_comment_for_repost``), so the drain reposts it or adopts our own copy."""
    with conn:
        return bool(
            conn.execute(
                "UPDATE deliveries SET first_comment_text=?, first_comment_posted_at=NULL, "
                "first_comment_live_id=NULL, first_comment_error=NULL, "
                "first_comment_removed_at=NULL, updated_at=? WHERE id=? "
                "AND first_comment_removed_at IS NOT NULL AND first_comment_posted_at IS NOT NULL "
                "AND first_comment_correction IS NULL",
                (text, now, delivery_id),
            ).rowcount
        )


def unpost_first_comment_for_repost(
    conn: sqlite3.Connection, delivery_id: int, item: dict, now: str, *, count_attempt: bool = False
) -> None:
    """An ``edit`` found its comment gone (404): unpost the row with the edit's ``to`` as its
    text, so the normal drain posts it. Items for the same dead comment go."""
    with conn:
        row = conn.execute(
            "SELECT first_comment_correction FROM deliveries WHERE id=?", (delivery_id,)
        ).fetchone()
        if row is None:
            return
        items = first_comment_corrections(row)
        if not any(_same_applied(i, item) for i in items):
            return  # a relink moved its ``to`` meanwhile: the next pass edits to the new one
        left = [i for i in items if i.get("live_id") != item.get("live_id")]
        conn.execute(
            "UPDATE deliveries SET first_comment_text=?, first_comment_posted_at=NULL, "
            "first_comment_live_id=NULL, first_comment_error=NULL, first_comment_correction=?, "
            "first_comment_attempts=first_comment_attempts+?, updated_at=? "
            "WHERE id=? AND first_comment_correction=?",
            (
                item.get("to") or "",
                json.dumps(left) if left else None,
                int(count_attempt),
                now,
                delivery_id,
                row["first_comment_correction"],
            ),
        )


def rewrite_first_comment_corrections(
    conn: sqlite3.Connection,
    delivery_id: int,
    rewrite: Callable[[str], str],
    now: str,
    gate: Optional[Callable[[str], Tuple[Optional[str], bool]]] = None,
) -> Optional[bool]:
    """Compare-and-set ``to`` -> ``rewrite(to)`` on every pending ``edit`` (a second replace: the
    edit must write the newest link). With ``gate`` (``to`` -> ``(parent_publish_at, public)``)
    an edit whose replacement is public keeps ``parent_at`` for the drain's own re-check, and one
    that is not becomes a ``delete`` of the live comment and the row's intended text is dropped
    (a delete of the intended text never runs). ``None`` when no edit changes, ``True`` when
    written, ``False`` when the JSON moved since it was read (rerun)."""
    with conn:
        row = conn.execute(
            "SELECT first_comment_correction FROM deliveries WHERE id=?", (delivery_id,)
        ).fetchone()
        if row is None or not row["first_comment_correction"]:
            return None
        raw = row["first_comment_correction"]
        items = first_comment_corrections(row)
        dropped = False

        def fix(i: dict) -> dict:
            nonlocal dropped
            if i.get("action") != "edit" or not isinstance(i.get("to"), str):
                return i
            new_to = rewrite(i["to"])
            if gate is None:
                return {**i, "to": new_to}
            if new_to == i["to"]:
                return i
            at, public = gate(i["to"])
            if public:
                return {**i, "to": new_to, "parent_at": at}
            dropped = True
            return {
                "live_id": i.get("live_id"),
                "live_text": i.get("live_text"),
                "action": "delete",
            }

        out = [fix(i) for i in items]
        if out == items:
            return None
        done = conn.execute(
            "UPDATE deliveries SET first_comment_correction=?, updated_at=?"  # noqa: S608
            + (", first_comment_text=NULL" if dropped else "")
            + " WHERE id=? AND first_comment_correction=?",
            (json.dumps(out), now, delivery_id, raw),
        ).rowcount
        return done == 1


_CAS_RETRIES = 5


def drop_private_edit(conn: sqlite3.Connection, delivery_id: int, item: dict, now: str) -> None:
    """``item`` (an ``edit``) would link a video that is still private at the row's slot: it
    becomes a ``delete`` of the live comment and the row's intended text goes, so the delete
    (never run against the intended text) can run. A compare-and-set on the JSON, retried when a
    concurrent drain moved it, so an item it appended is never lost."""
    for _ in range(_CAS_RETRIES):
        with conn:
            row = conn.execute(
                "SELECT first_comment_correction FROM deliveries WHERE id=?", (delivery_id,)
            ).fetchone()
            if row is None:
                return
            out = [
                {"live_id": i.get("live_id"), "live_text": i.get("live_text"), "action": "delete"}
                if _same_correction(i, item) and i.get("action") == "edit"
                else i
                for i in first_comment_corrections(row)
            ]
            done = conn.execute(
                "UPDATE deliveries SET first_comment_correction=?, first_comment_text=NULL, "
                "updated_at=? WHERE id=? AND first_comment_correction IS ?",
                (json.dumps(out), now, delivery_id, row["first_comment_correction"]),
            ).rowcount
        if done == 1:
            return
    raise sqlite3.OperationalError(f"delivery {delivery_id}: the correction JSON kept moving")


def resolve_first_comment_correction(
    conn: sqlite3.Connection,
    delivery_id: int,
    item: dict,
    now: str,
    new_live_id: Optional[str] = None,
) -> None:
    """``item`` was applied: drop it (an ``edit`` also sets the row's intended text); the last
    one clears the error. ``new_live_id``: the replace (Instagram, or a Facebook edit Graph
    refused) posted a new comment; it becomes the row's recorded live id when the replaced
    comment was the recorded one, so a later scan never takes it for a second copy."""
    with conn:
        row = conn.execute(
            "SELECT first_comment_text, first_comment_correction, first_comment_live_id "
            "FROM deliveries WHERE id=?",
            (delivery_id,),
        ).fetchone()
        if row is None:
            return
        items = first_comment_corrections(row)
        live = row["first_comment_live_id"]
        if new_live_id and item.get("action") == "edit" and live in (None, item.get("live_id")):
            live = new_live_id
        left = [i for i in items if not _same_applied(i, item)]
        text = row["first_comment_text"]
        if (
            len(left) < len(items)
            and item.get("action") == "edit"
            and text == item.get("live_text")
        ):
            text = item.get("to") or text
        conn.execute(
            "UPDATE deliveries SET first_comment_text=?, first_comment_live_id=?, "
            "first_comment_correction=?, "
            "first_comment_error=CASE WHEN ? THEN first_comment_error ELSE NULL END, "
            "updated_at=? WHERE id=? AND first_comment_correction=?",
            (
                text,
                live,
                json.dumps(left) if left else None,
                int(bool(left)),
                now,
                delivery_id,
                row["first_comment_correction"],
            ),
        )


def fail_first_comment_correction(
    conn: sqlite3.Connection, delivery_id: int, detail: str, refused: bool
) -> None:
    """Keep the correction, say why on the row, count a platform refusal."""
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_error=?, "
            "first_comment_correction_attempts=first_comment_correction_attempts+? WHERE id=?",
            (f"stale comment live: correction failed: {detail}"[:500], int(refused), delivery_id),
        )


#: Statuses whose first comment can never post. ``published`` is NOT here: a scheduled
#: YouTube Short is ``published`` from upload and its comment still posts later.
FIRST_COMMENT_FROZEN_STATUSES = ("failed_permanent", "cancelled", STATUS_MISSING_REMOTE)


def first_comment_prior_texts(row: Any) -> Tuple[str, ...]:
    """The texts a relink replaced on this delivery row (``first_comment_prior_text``)."""
    try:
        raw = row["first_comment_prior_text"]
        texts = json.loads(raw) if raw else []
    except (KeyError, IndexError, TypeError, ValueError):
        return ()
    return tuple(t for t in texts if isinstance(t, str)) if isinstance(texts, list) else ()


def rewrite_unposted_first_comment(
    conn: sqlite3.Connection, delivery_id: int, old_text: str, new_text: Optional[str], now: str
) -> bool:
    """Compare-and-set ``first_comment_text`` ``old_text`` -> ``new_text`` (``None`` drops the
    link: the parent is private at the row's slot) on one delivery,
    only while the comment can still post: unposted, under ``FIRST_COMMENT_MAX_ATTEMPTS``,
    status not in ``FIRST_COMMENT_FROZEN_STATUSES``, no live lease. ``False`` (nothing written)
    when the row moved since it was read. ``old_text`` joins ``first_comment_prior_text`` for
    the drain's dedup; no other column changes. A lease is live or expired by this process's
    clock, never by the caller's ``now`` (stored only as ``updated_at``)."""
    frozen = ", ".join("?" for _ in FIRST_COMMENT_FROZEN_STATUSES)
    clock = _now_iso()
    with conn:
        row = conn.execute(
            "SELECT first_comment_prior_text FROM deliveries WHERE id=?", (delivery_id,)
        ).fetchone()
        if row is None:
            return False
        prior = list(first_comment_prior_texts(row))
        if old_text not in prior:
            prior.append(old_text)
        return (
            conn.execute(
                "UPDATE deliveries SET first_comment_text=?, first_comment_prior_text=?, "  # noqa: S608
                "updated_at=? WHERE id=? AND first_comment_text=? "
                "AND first_comment_posted_at IS NULL AND first_comment_attempts < ? "
                f"AND status NOT IN ({frozen}) "
                "AND (lease_owner IS NULL OR lease_expires_at <= ?)",
                (
                    new_text,
                    json.dumps(prior),
                    now,
                    delivery_id,
                    old_text,
                    FIRST_COMMENT_MAX_ATTEMPTS,
                    *FIRST_COMMENT_FROZEN_STATUSES,
                    clock,
                ),
            ).rowcount
            == 1
        )


def first_comment_still_current(conn: sqlite3.Connection, delivery_id: int) -> Optional[Any]:
    """The row's current text columns, or None once its comment is posted or gone: the drains
    re-read this just before posting so a relink after selection is honoured."""
    row = conn.execute(
        "SELECT first_comment_text, first_comment_prior_text FROM deliveries "
        "WHERE id=? AND first_comment_posted_at IS NULL "
        "AND first_comment_text IS NOT NULL AND first_comment_text <> ''",
        (delivery_id,),
    ).fetchone()
    return row


class RecoveryRefusedError(Exception):
    """A ``recover_terminal_delivery`` precondition did not hold; nothing was
    written. The message is the one-line reason the CLI prints."""


#: A recovery row's ``plan_hash`` is derived from the terminal row's, which
#: keeps ``UNIQUE (project_name, short_index, platform, plan_hash)`` satisfied
#: AND makes a second recovery of the same row impossible by construction:
#: the derived hash is identical, so the second INSERT is a constraint
#: violation, not an application-level race. Nothing reads ``plan_hash`` back
#: to compare against a sidecar (the drain loads records by project/index).
RECOVERY_PLAN_HASH_TAG = "#recovers-"

#: Terminal statuses ``check_recoverable`` lets through; the caller pairs each
#: with its platform (``cli/handlers/queue_recover.py``).
_RECOVERABLE_STATUSES = ("failed_permanent", STATUS_MISSING_REMOTE)


def check_recoverable(conn: sqlite3.Connection, delivery_id: int) -> sqlite3.Row:
    """The terminal row, or ``RecoveryRefusedError``: only a ``failed_permanent``
    or ``missing_remote`` row not already recovered qualifies. ``retry_delivery``
    refuses terminal rows by design; this is the one audited way past that, and
    it never mutates the row it recovers."""
    if facebook_photo_attempt(conn, delivery_id):
        raise RecoveryRefusedError("A recorded photo attempt cannot authorize a replacement upload")
    row = conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    if row is None:
        raise RecoveryRefusedError(f"no delivery with id {delivery_id}")
    if row["status"] not in _RECOVERABLE_STATUSES:
        raise RecoveryRefusedError(
            f"delivery {delivery_id} is {row['status']!r}; only failed_permanent or "
            "missing_remote can be recovered"
        )
    prior = conn.execute(
        "SELECT id FROM deliveries WHERE recovers_delivery_id=?", (delivery_id,)
    ).fetchone()
    if prior is not None:
        raise RecoveryRefusedError(
            f"delivery {delivery_id} was already recovered by row {prior['id']}"
        )
    return row


def recover_terminal_delivery(
    conn: sqlite3.Connection,
    delivery_id: int,
    *,
    authorized_by: str,
    evidence: Dict[str, Any],
    now: Optional[str] = None,
    scheduled_at: Optional[str] = None,
) -> int:
    """Insert ONE new ``queued`` delivery for the same short/platform, linked
    to the terminal row by ``recovers_delivery_id``; returns its
    id. The terminal row (status, receipts, ids, ``attempt_count``) is not
    touched. The caller is responsible for the gates (see
    ``cli/handlers/queue_recover.py``); this re-checks status and
    single-recovery inside the write so a stale caller cannot double-insert.
    Also appends an audit row to ``response_log``. The new row is due ``now`` unless
    ``scheduled_at`` names its slot (a re-slotted YouTube Short books its new time)."""
    if not (authorized_by or "").strip():
        raise RecoveryRefusedError("--authorized-by must be a non-empty statement")
    now = now or _now_iso()
    evidence_json = json.dumps(evidence, sort_keys=True, default=str)
    try:
        with conn:
            old = check_recoverable(conn, delivery_id)
            cur = conn.execute(
                "INSERT INTO deliveries "
                "(project_name, short_index, platform, plan_hash, status, attempt_count, "
                " shadow, sidecar_path, scheduled_at, first_comment_text, "
                " recovers_delivery_id, recovery_authorized_by, recovery_evidence, "
                " created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'queued', 0, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    old["project_name"],
                    old["short_index"],
                    old["platform"],
                    f"{old['plan_hash']}{RECOVERY_PLAN_HASH_TAG}{delivery_id}",
                    old["shadow"],
                    old["sidecar_path"],
                    scheduled_at or now,
                    old["first_comment_text"],
                    delivery_id,
                    authorized_by.strip(),
                    evidence_json,
                    now,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO response_log (at, platform, state, cap, action, mode, reason) "
                "VALUES (?, ?, 'recovered', 0, 'recover_terminal', 'live', ?)",
                (
                    now,
                    old["platform"],
                    json.dumps(
                        {
                            "recovered": delivery_id,
                            "new": cur.lastrowid,
                            "authorized_by": authorized_by.strip(),
                            "evidence": evidence,
                        },
                        sort_keys=True,
                        default=str,
                    ),
                ),
            )
            return int(cur.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise RecoveryRefusedError(
            f"delivery {delivery_id} cannot be recovered: an active delivery exists for the "
            f"same short/platform or it was already recovered ({exc})"
        ) from exc
