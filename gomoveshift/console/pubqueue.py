"""Console Publishing tab: read-only view over the delivery queue
plus job-runner launches of ``enqueue-plan``/``dispatch-queue``. Never
re-implements enqueue/drain; argv mirrors ``cli/parser.py`` exactly.
Per-platform pause IS in scope: the rule and the storage stay in
``publishing/queue.py``, which ``dispatch-queue`` opens too -- this module
only reads it back and turns a refusal into a sentence.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_args
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from ..config import Config
from ..publishing.brand_policy import load_brand_policy
from ..publishing.dispatch import parse_response_log_at, platform_quiet_alert_at
from ..publishing.models import KNOWN_PLATFORMS
from ..publishing.plan_io import PublishPlanFormatError, load_publish_plan
from ..publishing.queue import (
    PUBLISHED_AT_OPERATOR_CONFIRMED,
    PUBLISHED_AT_PLATFORM_SCHEDULED,
    TERMINAL_STATUSES,
    cancel_delivery,
    clear_platform_daily_cap,
    is_platform_paused,
    open_queue_db,
    pause_platform,
    reschedule_delivery,
    resume_platform,
    retry_delivery,
    set_platform_daily_cap,
    transition,
)
from ..publishing.response_policy import Status as ResponseStatus
from ..publishing.response_policy import daily_cap_for
from .library import privacy_status_for_media
from .settings import current_brand

logger = logging.getLogger(__name__)

#: The only state ``mark_published`` accepts as a starting point --
#: see ``publishing.handlers.STATUS_AWAITING_MANUAL_PUBLISH``, not imported
#: here to avoid a console -> handlers edge for one literal.
_AWAITING_MANUAL_PUBLISH = "awaiting_manual_publish"

#: Lifecycle order for grouping the queue view. ``awaiting_manual_publish``
#: sorts FIRST: the only state a drain can never advance, so the only
#: row that strictly needs a person. ``failed_permanent`` stays LAST: it is
#: terminal, its error already inline in the row -- unlike
#: ``awaiting_manual_publish``, stuck open until "mark published" is clicked
#: right here. Any status not listed (future/unknown) sorts after these,
#: alphabetically -- safe only because every *known* attention state is
#: listed above it.
_STATE_ORDER = (
    _AWAITING_MANUAL_PUBLISH,
    "queued",
    "submitting",
    "uploaded",
    "published",
    "cancelled",
    "failed_permanent",
)


def _config(config: Optional[Any] = None) -> Any:
    return config if config is not None else Config.load()


def queue_db_path(config: Optional[Any] = None) -> Path:
    """The same ``<storage_path>/social_queue.sqlite3`` the CLI opens."""
    return Path(_config(config).seo.storage_path) / "social_queue.sqlite3"


def queue_rows(config: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Every delivery row, read-only. A missing DB file is an empty queue,
    never created just by viewing the tab.

    Deliberately unbounded, and no longer what the tab reads -- that is
    ``queue_page`` below. The callers left here are the observers in
    ``notify.py``, whose job is to look at every row; a LIMIT here would stop
    the failure webhook at row ``PER_PAGE`` and call the rest healthy."""
    db_path = queue_db_path(config)
    if not db_path.exists():
        return []
    conn = open_queue_db(db_path.parent)
    try:
        rows = conn.execute("SELECT * FROM deliveries ORDER BY status, id").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


#: Rows per page on the Publishing tab. The queue only grows -- every short,
#: every platform, every week -- and nothing ever deletes a ``published`` row,
#: so handing the whole read to the template built a table without bound. Same
#: value and same reasoning as the Media library's ``PER_PAGE``.
PER_PAGE = 100

#: ``_STATE_ORDER`` as a SQL sort key, GENERATED from it rather than retyped so
#: the two cannot drift. The order has to be in the SQL, not applied to
#: whatever a page came back with: ``ORDER BY status`` is alphabetical, which
#: would put ``awaiting_manual_publish`` -- the one state waiting on a human --
#: behind a page full of ``cancelled``. An unknown status sorts last, matching
#: ``_state_sort_key``.
_STATE_ORDER_SQL = "CASE status {} ELSE {} END".format(
    " ".join(f"WHEN '{status}' THEN {index}" for index, status in enumerate(_STATE_ORDER)),
    len(_STATE_ORDER),
)

#: The page query, assembled ONCE from the constant above so the statement
#: handed to sqlite is a plain string with bound parameters -- nothing per-call
#: is ever interpolated into it.
_QUEUE_PAGE_SQL = f"SELECT * FROM deliveries ORDER BY {_STATE_ORDER_SQL}, id LIMIT ? OFFSET ?"  # noqa: S608


def _pager(total: int, page: Any) -> Dict[str, Any]:
    """Where in ``total`` rows ``page`` lands, for a ``page`` that came off a
    query string and may be anything at all. Out of range lands on the nearest
    real page: an operator reads an empty table as an empty queue, which is a
    worse lie than a nearest-page answer."""
    pages = max(1, -(-total // PER_PAGE))
    try:
        wanted = int(page)
    except (TypeError, ValueError):
        wanted = 1
    wanted = min(max(wanted, 1), pages)
    return {
        "total": total,
        "per_page": PER_PAGE,
        "page": wanted,
        "pages": pages,
        "has_prev": wanted > 1,
        "has_next": wanted < pages,
    }


def _windows_path(path: Path) -> str:
    """``/mnt/e/Clips`` as ``E:\\Clips``, the form a Windows user types in
    Explorer; any other path is shown as it is."""
    parts = path.parts
    if len(parts) >= 3 and parts[:2] == ("/", "mnt") and len(parts[2]) == 1:
        return parts[2].upper() + ":\\" + "\\".join(parts[3:])
    return str(path)


def awaiting_manual_hint() -> str:
    """What the owner does with a row parked ``awaiting_manual_publish``, which
    depends on ``$GMS_TIKTOK_MODE``: pack mode (the default) leaves a file and
    ``captions.txt`` in the Studio pack folder, inbox/direct mode leaves a draft
    in the TikTok app. An unreadable mode gets a neutral line, never an error:
    the Publishing tab must render whatever the environment says."""
    from ..publishing.handlers import resolve_tiktok_mode
    from ..publishing.tiktok_pack import resolve_pack_dir

    try:
        mode = resolve_tiktok_mode()
    except ValueError:
        return "Post the short in TikTok, then confirm here."
    if mode == "pack":
        return f"Post the file from {_windows_path(resolve_pack_dir())} (captions.txt), then confirm here."
    return "Publish from the TikTok app inbox, then confirm here."


def queue_page(config: Optional[Any] = None, page: Any = 1) -> Dict[str, Any]:
    """One bounded page of the delivery queue, read-only, for the Publishing
    tab. The LIMIT and the lifecycle ORDER BY are both in the SQL -- fetching
    every row and slicing would cost exactly what it did before, and sorting
    after the fetch would bury the state a human is waiting on.

    ``queue_rows`` above stays unbounded on purpose: its other caller scans
    every row by definition, and a bound there would stop the observers at row
    ``PER_PAGE``. A missing DB file is an empty page, never created by viewing.
    """
    db_path = queue_db_path(config)
    if not db_path.exists():
        return {"rows": [], "awaiting_hint": awaiting_manual_hint(), **_pager(0, page)}
    conn = open_queue_db(db_path.parent)
    try:
        total = conn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0]
        pager = _pager(total, page)
        rows = conn.execute(_QUEUE_PAGE_SQL, (PER_PAGE, (pager["page"] - 1) * PER_PAGE)).fetchall()
        return {
            "rows": [dict(row) for row in rows],
            "awaiting_hint": awaiting_manual_hint(),
            **pager,
        }
    finally:
        conn.close()


def _state_sort_key(status: str) -> tuple:
    try:
        return (0, _STATE_ORDER.index(status))
    except ValueError:
        return (1, status)


def _title_for_platform(short: Any, platform: Optional[str]) -> Optional[str]:
    payloads = short.payloads
    if platform == "youtube" and payloads.youtube:
        return payloads.youtube.title
    if platform == "tiktok" and payloads.tiktok:
        return payloads.tiktok.title
    if platform == "instagram" and payloads.instagram:
        return payloads.instagram.caption
    if platform == "facebook" and payloads.facebook:
        return payloads.facebook.description
    return None


def _format_scheduled(moment: datetime, tz_name: Optional[str]) -> str:
    """``moment`` (aware) as ``YYYY-MM-DD HH:MM <tz>``, in ``tz_name`` when
    that name resolves, else plain UTC -- never raises on a bad tz name."""
    if tz_name:
        try:
            local = moment.astimezone(ZoneInfo(tz_name))
            return f"{local.strftime('%Y-%m-%d %H:%M')} {tz_name}"
        except (KeyError, ValueError):
            pass
    return f"{moment.strftime('%Y-%m-%d %H:%M')} UTC"


#: The only two schemes this console will turn into a clickable link. A
#: permalink reaches the queue from a platform's API response and from an
#: operator typing into the manual-confirmation form -- neither is trusted
#: input, and Jinja's autoescaping does not help here: it escapes quotes, and
#: ``javascript:alert(1)`` needs none. Allowlist, never a blocklist: ``data:``
#: and ``vbscript:`` are only the schemes anyone thought to name today.
_LINKABLE_SCHEMES = ("http", "https")


def linkable_permalink(value: "Optional[str]") -> "Optional[str]":
    """``value`` when it is a URL this console is willing to link, else
    ``None``. A schemeless ``example.com/p/1`` is refused too: in an ``href``
    that is a RELATIVE path pointing back into the console, so the link would
    go somewhere other than where it reads."""
    if not value:
        return None
    try:
        parsed = urlparse(value)
    except ValueError:
        return None
    if parsed.scheme.lower() in _LINKABLE_SCHEMES and parsed.netloc:
        return value
    return None


def enrich_rows_with_sidecar(
    rows: List[Dict[str, Any]],
    cache: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """``rows`` plus ``scheduled_at_local``/``scheduled_at_utc``,
    ``scheduled_platforms``, ``title`` and ``format``, read from the row's own
    ``sidecar_path``; and the row's own ``scheduled_at`` column
    (``publishing/queue.py``) as the AUTHORITATIVE effective schedule --
    ``scheduled_at_effective_local``. A sidecar time that disagrees with the
    row's surfaces as ``schedule_discrepancy`` rather than being silently
    resolved either way. Read-only, never raises: a missing/unmatched
    sidecar leaves the sidecar-only fields ``None``; a malformed/unreadable
    one also sets ``sidecar_read_error`` so the tab says "could not read"
    rather than implying "no schedule".

    ``cache`` (path -> parsed ``PublishPlan``) is reused across rows so
    N rows from one project's sidecar read the file once."""
    if cache is None:
        cache = {}
    policy = load_brand_policy(current_brand())
    blackout_dates = set(policy.blackout)
    #: Computed the same way as ``dispatch._exclude_quiet_window`` --
    #: ``platform_quiet_alert_at`` per platform, lazily and cached, so a
    #: page with N rows for the same platform opens the queue DB once per
    #: platform, never once per row. ``None`` means no queries at all, same
    #: contract as the dispatch-side reader.
    quiet_hours = policy.quiet_after_alert_hours
    quiet_alert_at_cache: Dict[str, Optional[str]] = {}
    quiet_conn = None
    if quiet_hours is not None:
        db_path = queue_db_path(config)
        if db_path.exists():
            quiet_conn = open_queue_db(db_path.parent)
    enriched = []
    for row in rows:
        row = dict(row)
        row["scheduled_at_local"] = None
        row["scheduled_at_utc"] = None
        row["scheduled_platforms"] = None
        row["title"] = None
        row["scheduled_at_effective_local"] = None
        row["schedule_discrepancy"] = None
        row["sidecar_read_error"] = False
        row["format"] = None
        #: The recorded privacy report for this delivery's media, from
        #: the same sidecar read below -- ``None`` until a matching short is
        #: found, same as every other sidecar-derived field on this row.
        row["privacy_status"] = None
        #: "blackout" when this row's effective date falls on the
        #: current brand's ``policy.blackout``, matching what
        #: ``dispatch._select_due`` will refuse to select -- ``None`` for a
        #: terminal row (a blackout is a future-pickup gate, not a verdict
        #: on a row already published/failed).
        row["held_reason"] = None
        # Both the queue list and the detail page render this column from these
        # same enriched rows, so the guard lives here rather than in either
        # template -- a per-template copy is a per-template chance to forget it.
        row["permalink_href"] = linkable_permalink(row.get("permalink"))

        try:
            row_scheduled = datetime.fromisoformat(row.get("scheduled_at"))
        except (TypeError, ValueError):
            row_scheduled = None

        if blackout_dates and row.get("status") not in TERMINAL_STATUSES:
            if row_scheduled is not None:
                held_date = row_scheduled.date()
            else:
                held_date = datetime.now(timezone.utc).date()
            if held_date in blackout_dates:
                row["held_reason"] = "blackout"

        #: "quiet" the same way "blackout" above is -- blackout wins
        #: when both apply, matching ``_select_due``, which excludes
        #: blackout rows before the quiet window ever sees them.
        if (
            row["held_reason"] is None
            and quiet_conn is not None
            and row.get("status") not in TERMINAL_STATUSES
        ):
            platform = row.get("platform")
            if platform not in quiet_alert_at_cache:
                quiet_alert_at_cache[platform] = platform_quiet_alert_at(quiet_conn, platform)
            alert_at = quiet_alert_at_cache[platform]
            if alert_at is not None:
                elapsed = (
                    datetime.now(timezone.utc) - parse_response_log_at(alert_at)
                ).total_seconds()
                if elapsed < quiet_hours * 3600:
                    row["held_reason"] = "quiet"

        sidecar_scheduled = None
        tz_name = None
        sidecar_path = row.get("sidecar_path")
        if sidecar_path:
            try:
                if sidecar_path not in cache:
                    cache[sidecar_path] = load_publish_plan(Path(sidecar_path))
                plan = cache[sidecar_path]
                short = next(
                    (
                        s
                        for s in plan.shorts
                        if s.project_name == row.get("project_name")
                        and s.short_index == row.get("short_index")
                    ),
                    None,
                )
                if short is not None:
                    sidecar_scheduled = short.schedule.scheduled_at_utc
                    tz_name = short.schedule.display_timezone
                    row["scheduled_at_utc"] = sidecar_scheduled.isoformat()
                    row["scheduled_at_local"] = _format_scheduled(sidecar_scheduled, tz_name)
                    row["scheduled_platforms"] = ", ".join(short.routes.selected)
                    row["title"] = _title_for_platform(short, row.get("platform"))
                    row["format"] = short.format
                    row["privacy_status"] = privacy_status_for_media(short.media.path, short.format)
            except PublishPlanFormatError:
                row["sidecar_read_error"] = True

        if row_scheduled is not None:
            # Plain UTC, never the sidecar's tz -- avoids blending a
            # disagreement into one confusing number.
            row["scheduled_at_effective_local"] = _format_scheduled(row_scheduled, None)
            if sidecar_scheduled is not None and sidecar_scheduled != row_scheduled:
                row["schedule_discrepancy"] = (
                    f"sidecar says {_format_scheduled(sidecar_scheduled, tz_name)}"
                )
        elif sidecar_scheduled is not None:
            # No authoritative row value -- the sidecar's own time is real
            # information, not nothing.
            row["scheduled_at_effective_local"] = row["scheduled_at_local"]
        enriched.append(row)
    if quiet_conn is not None:
        quiet_conn.close()
    return enriched


#: Human labels for ``deliveries.published_at_source`` (see ``publishing.queue``).
#: ``None`` is a row written before that column existed -- an honest
#: "unknown", never guessed as an operator confirmation it can't prove.
_PUBLISH_CONFIRMATION_LABELS = {
    None: "unknown (legacy)",
    "operator_confirmed": "operator confirmed",
    "platform_reconciled": "platform reconciled (delayed)",
    "platform_confirmed": "platform confirmed",
    # The platform ACCEPTED the delivery and publishes it at the
    # instant in ``platform_published_at``. Without an entry here it fell
    # through to "unknown (legacy)" -- the strongest evidence this queue holds
    # wearing the label meant for rows that carry none.
    PUBLISHED_AT_PLATFORM_SCHEDULED: "platform scheduled (publishes later)",
}


def _publish_confirmation(row: Dict[str, Any]) -> str:
    return _PUBLISH_CONFIRMATION_LABELS.get(row.get("published_at_source"), "unknown (legacy)")


def rows_by_state(rows: List[Dict[str, Any]]) -> List[tuple]:
    """``rows`` grouped into ``(status, [row, ...])`` pairs, lifecycle-ordered,
    each row enriched with its sidecar's schedule/title (see
    ``enrich_rows_with_sidecar``) and a ``publish_confirmation`` label from
    ``published_at_source``, distinctly labelled so a delayed confirmation
    never reads as an accurate one; a legacy row reports "unknown (legacy)"."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in enrich_rows_with_sidecar(rows):
        row["publish_confirmation"] = _publish_confirmation(row)
        grouped.setdefault(row["status"], []).append(row)
    return [(status, grouped[status]) for status in sorted(grouped, key=_state_sort_key)]


#: Human sentences for the ``error_class`` values this queue actually
#: writes. Keyed on the CLASS,
#: never built from ``error_detail`` -- untrusted platform text that can
#: carry a signed URL/account id/hostname. See ``_error_explanation``.
_ERROR_CLASS_EXPLANATIONS = {
    "PermanentDeliveryError": "The platform rejected this outright; it will not retry.",
    "ThrottledDeliveryError": "The platform rate-limited this; retrying after a fixed delay.",
    "TransientMediaError": "A transient media problem; retrying with backoff.",
    "quota": "YouTube's daily upload quota was exhausted; this retries once it resets.",
    "tiktok_publish_failed": "TikTok reported this manually-published draft failed to publish.",
    # Written by ``dispatch.reconcile_interrupted_attempts``, not by any
    # exception -- the worker was killed holding the lease, so nothing in that
    # process ever saw a failure. Unmapped, it read "no description yet" and
    # logged a WARNING on every view of a row the queue itself had labelled.
    "InterruptedAttempt": (
        "The worker handling this was killed mid-attempt; nobody knows whether the "
        "platform received it. It was charged one attempt and retries with backoff."
    ),
}

_UNMAPPED_ERROR_EXPLANATION = "This failure type has no description yet; see the server log."


def _error_explanation(row: Dict[str, Any]) -> Optional[str]:
    """A human sentence for ``row['error_class']``, or ``None`` with no
    error. Never touches ``error_detail`` -- an unmapped class logs it at
    warning level (recoverable server-side) and reads honestly instead."""
    error_class = row.get("error_class")
    if not error_class:
        return None
    if error_class in _ERROR_CLASS_EXPLANATIONS:
        return _ERROR_CLASS_EXPLANATIONS[error_class]
    if error_class.startswith("MaxAttemptsExceeded("):
        return "Retries exhausted; the platform never confirmed this."
    logger.warning(
        "delivery %s: no explanation for error_class=%r; detail=%r",
        row.get("id"),
        error_class,
        row.get("error_detail"),
    )
    return _UNMAPPED_ERROR_EXPLANATION


def _lease_state(row: Dict[str, Any], now: Optional[datetime] = None) -> str:
    """``"none"`` | ``"live"`` | ``"expired"``: an expired lease means no
    worker actually holds this row right now, whatever ``lease_owner``
    still says -- never collapsed into one "in progress" claim."""
    owner = row.get("lease_owner")
    expires_at = row.get("lease_expires_at")
    if not owner or not expires_at:
        return "none"
    try:
        expires = datetime.fromisoformat(expires_at)
    except (TypeError, ValueError):
        return "none"
    now = now or datetime.now(timezone.utc)
    return "live" if expires > now else "expired"


def delivery_by_id(delivery_id: int, config: Optional[Any] = None) -> Dict[str, Any]:
    """One delivery row for the detail page: enriched like ``rows_by_state``
    plus ``lease_state``/``error_explanation``/``platform_paused``. Raises
    ``KeyError`` for a missing DB or unknown id; the route maps that to a 404.

    ``platform_paused`` is read on the connection this already opens: while
    its platform is paused the next drain pass skips this row entirely, so
    the page must not tell an operator it is waiting on a schedule."""
    db_path = queue_db_path(config)
    if not db_path.exists():
        raise KeyError(delivery_id)
    conn = open_queue_db(db_path.parent)
    try:
        row = conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
        if row is None:
            raise KeyError(delivery_id)
        row = dict(row)
        paused = is_platform_paused(conn, row["platform"])
    finally:
        conn.close()
    enriched = enrich_rows_with_sidecar([row])[0]
    enriched["platform_paused"] = paused
    enriched["publish_confirmation"] = _publish_confirmation(enriched)
    enriched["lease_state"] = _lease_state(enriched)
    enriched["error_explanation"] = _error_explanation(enriched)
    enriched["operator_actions"] = operator_actions_for(enriched)
    return enriched


def mark_published(
    db_path: Union[str, Path],
    delivery_id: int,
    *,
    permalink: Optional[str] = None,
    platform_post_id: Optional[str] = None,
) -> None:
    """Confirm a human published this TikTok short (a pack-folder file
    or an inbox draft), transitioning the row ``awaiting_manual_publish`` -> ``published`` via
    ``queue.transition`` (never raw SQL) -- the same compare-and-set every
    other status change in this queue goes through, so this can't move a
    row out of any other state. ``permalink``/``platform_post_id``, when
    given, are written to the existing ``deliveries`` columns of the same
    name -- the only evidence a manual confirmation can offer when there is
    no readable platform API to verify against.

    A row already ``published`` is treated as an idempotent repeat
    confirmation (no-op, no error, no second publish) rather than a
    conflict -- confirming twice is an operator mistake, not a state bug.
    Any other status is a genuine conflict, whether read that way to start
    with or discovered because ``transition``'s compare-and-set lost a race
    against a concurrent change: ``transition`` returning ``False`` is
    never logged as success.

    Raises ``KeyError`` for a missing db file or unknown ``delivery_id``,
    ``ValueError`` for a conflicting status; the console route maps these
    to 404/409."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise KeyError(delivery_id)
    conn = open_queue_db(db_path.parent)
    try:
        row = conn.execute("SELECT status FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
        if row is None:
            raise KeyError(delivery_id)
        status = row["status"]
        if status == "published":
            logger.info("delivery %s already published, confirmation is a no-op", delivery_id)
            return
        if status != _AWAITING_MANUAL_PUBLISH:
            raise ValueError(
                f"delivery {delivery_id} is {status!r}, not {_AWAITING_MANUAL_PUBLISH!r}"
            )
        fields: Dict[str, Any] = {}
        if permalink is not None:
            fields["permalink"] = permalink
        if platform_post_id is not None:
            fields["platform_post_id"] = platform_post_id
        fields["published_at_source"] = PUBLISHED_AT_OPERATOR_CONFIRMED
        changed = transition(conn, delivery_id, _AWAITING_MANUAL_PUBLISH, "published", **fields)
        if not changed:
            # Lost the compare-and-set: the row moved between the read above
            # and this call. Re-read rather than assume -- if it landed on
            # "published" (e.g. a concurrent confirmation won), that is the
            # same idempotent no-op as above; anything else is a genuine
            # conflict, never silently reported as success.
            now = conn.execute(
                "SELECT status FROM deliveries WHERE id=?", (delivery_id,)
            ).fetchone()
            now_status = now["status"] if now else "missing"
            if now_status == "published":
                logger.info(
                    "delivery %s already published by a concurrent confirmation", delivery_id
                )
                return
            raise ValueError(f"delivery {delivery_id} is {now_status!r}, confirmation lost a race")
        logger.info("delivery %s confirmed published by hand (console mark-published)", delivery_id)
    finally:
        conn.close()


#: What an operator action is called on the page, for the refusal message. The
#: queue functions return a bare ``False`` for every ordinary refusal, so the
#: verb has to come from the caller.
_OPERATOR_ACTIONS = {"cancel": cancel_delivery, "retry": retry_delivery}


#: Past participle per control, spelled out rather than built by suffixing the
#: action name -- "reschedule" + "led" was the sentence an operator would have
#: read. A control with no entry here cannot reach ``_refusal_reason``: the
#: route allowlists the action first.
_ACTION_PAST = {"cancel": "cancelled", "retry": "retried", "reschedule": "rescheduled"}


def _refusal_reason(row: Dict[str, Any], action: str) -> str:
    """Why the queue just refused ``action`` on ``row``, read back from the
    row itself rather than guessed. Both refusals -- a terminal status and
    another worker's live lease -- return the same ``False``, and between the
    attempt and this read the row may have moved again, so the honest answer
    is the state it is in NOW."""
    status = row["status"]
    done = _ACTION_PAST[action]
    if status in TERMINAL_STATUSES:
        return f"delivery {row['id']} is already {status!r}; there is nothing to {action}"
    if row["lease_owner"]:
        return (
            f"delivery {row['id']} is held by {row['lease_owner']} until "
            f"{row['lease_expires_at']}; that worker may already be mid-publish, "
            f"so it cannot be {done} from here"
        )
    return f"delivery {row['id']} could not be {done} (status {status!r})"


def operator_action(
    db_path: Union[str, Path],
    delivery_id: int,
    action: str,
) -> None:
    """Run one operator control -- ``"cancel"`` or ``"retry"`` -- against the
    queue, or say why it was refused.

    The queue owns both rules (terminal rows are final; a live foreign lease
    may already have made an irreversible platform call) and signals a refusal
    by returning ``False``. Turning that into a redirect to an unchanged page
    would be indistinguishable from success, so a refusal becomes a
    ``ValueError`` carrying the row's current state. ``KeyError`` for a missing
    db or unknown id; the route maps these to 409/404."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise KeyError(delivery_id)
    conn = open_queue_db(db_path.parent)
    try:
        row = conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
        if row is None:
            raise KeyError(delivery_id)
        if not _OPERATOR_ACTIONS[action](conn, delivery_id):
            fresh = conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
            raise ValueError(_refusal_reason(dict(fresh or row), action))
        logger.info("delivery %s %s from the console", delivery_id, _ACTION_PAST[action])
    finally:
        conn.close()


def reschedule(db_path: Union[str, Path], delivery_id: int, scheduled_at: str) -> None:
    """Move one delivery's slot, or say why not.

    Same shape as ``operator_action``: every rule is the queue's -- a terminal
    row is final, a live foreign lease may already be mid-publish, and a slot
    the drain could not compare is refused outright -- and a refusal becomes a
    ``ValueError`` the route renders on the page rather than a redirect to an
    unchanged one. ``KeyError`` for a missing db or unknown id.

    The queue's own ``ValueError`` (an unreadable timestamp) is allowed
    through unchanged: it already names the value, and rewording it here would
    put a second copy of that rule in the console."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise KeyError(delivery_id)
    conn = open_queue_db(db_path.parent)
    try:
        row = conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
        if row is None:
            raise KeyError(delivery_id)
        if _packed_tiktok(dict(row)):
            raise ValueError(
                "this short is already in the TikTok pack under its slot; change its time "
                "in TikTok Studio, not here"
            )
        if not reschedule_delivery(conn, delivery_id, scheduled_at):
            fresh = conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
            raise ValueError(_refusal_reason(dict(fresh or row), "reschedule"))
        logger.info("delivery %s rescheduled to %s from the console", delivery_id, scheduled_at)
    finally:
        conn.close()


def operator_actions_for(row: Dict[str, Any]) -> List[str]:
    """The controls worth showing for ``row``. A button that always refuses is
    worse than no button: it tells an operator an action is available and then
    409s. A live lease is deliberately NOT filtered here -- it expires on its
    own, so the page would be offering a control that comes back seconds later
    for reasons the operator cannot see; the refusal explains it instead."""
    if row.get("status") in TERMINAL_STATUSES:
        return []
    if _packed_tiktok(row):
        return ["cancel", "retry"]
    return ["cancel", "retry", "reschedule"]


def _packed_tiktok(row: Dict[str, Any]) -> bool:
    """A TikTok row parked by pack mode: its slot is fixed in the pack's file
    name and captions.txt, so the queue's time can no longer move it."""
    if row.get("platform") != "tiktok" or row.get("status") != "awaiting_manual_publish":
        return False
    from ..publishing.handlers import resolve_tiktok_mode

    try:
        return resolve_tiktok_mode() == "pack"
    except ValueError:
        return False


def known_platforms() -> List[str]:
    """What the pause form may offer, taken from the publishing vocabulary
    rather than retyped here. ``pause_platform`` refuses anything outside it,
    so a control the queue would reject must not exist on the page at all."""
    return sorted(KNOWN_PLATFORMS)


def paused_platforms(config: Optional[Any] = None) -> List[str]:
    """Which platforms the next drain pass will skip, read-only. A missing DB
    file is "none paused", never created just by viewing the tab."""
    db_path = queue_db_path(config)
    if not db_path.exists():
        return []
    conn = open_queue_db(db_path.parent)
    try:
        rows = conn.execute("SELECT platform FROM platform_pauses ORDER BY platform").fetchall()
        return [row["platform"] for row in rows]
    finally:
        conn.close()


def set_platform_pause(db_path: Union[str, Path], platform: str, *, paused: bool) -> None:
    """Pause or resume ``platform`` for the next drain pass, or say why not.

    Same shape as ``operator_action``: the queue owns the rule -- an unknown
    platform name has nothing to pause, a platform that was never paused has
    nothing to resume -- and signals both with a bare ``False``. Redirecting
    back to an unchanged page would be indistinguishable from success, so a
    refusal becomes a ``ValueError`` the route renders on the page.

    Writing goes through the queue DB, not console settings, because
    ``dispatch-queue`` runs as a separate process and only the queue file is
    visible to both."""
    conn = open_queue_db(Path(db_path).parent)
    try:
        if paused:
            if not pause_platform(conn, platform):
                raise ValueError(
                    f"{platform!r} is not a platform this queue publishes to; "
                    f"pick one of {', '.join(known_platforms())}"
                )
        elif not resume_platform(conn, platform):
            raise ValueError(f"{platform!r} was not paused, so there is nothing to resume")
        logger.info("platform %s %s from the console", platform, "paused" if paused else "resumed")
    finally:
        conn.close()


#: The states :func:`set_state_override` accepts, taken from
#: ``response_policy.Status`` (via ``typing.get_args``) rather than retyped
#: here -- a state the policy doesn't know is never offered.
_OVERRIDE_STATES = get_args(ResponseStatus)


def platform_daily_caps(config: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Every platform's current bounded-response cap decision, read-only.
    A missing DB file is "no decisions yet", never created just by viewing
    the tab -- same rule as ``paused_platforms``."""
    db_path = queue_db_path(config)
    if not db_path.exists():
        return []
    conn = open_queue_db(db_path.parent)
    try:
        rows = conn.execute("SELECT * FROM platform_daily_caps ORDER BY platform").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def set_state_override(
    db_path: Union[str, Path], platform: str, state: str, *, cap: Optional[int] = None
) -> None:
    """An operator's manual override of one platform's bounded-response
    state -- the sticking half of the response (``dispatch.
    apply_suppression_response`` skips any ``platform_daily_caps`` row whose
    ``reason`` starts with ``"manual:"``, so this is the only writer of that
    prefix). ``state="clear"`` lifts a previously set override instead of
    setting a new one -- the next nightly run's own decision then applies
    normally. A missing ``cap`` defaults to what the ladder floor for
    ``state`` would give a brand-less caller (``daily_cap_for(state, None,
    0)``) -- an explicit, deliberately conservative default rather than
    silently uncapping.

    Raises ``ValueError`` (never persisting anything) on an unknown
    platform, an unrecognized state, or a negative cap -- the same refusal
    shape as ``set_platform_pause``."""
    if platform not in KNOWN_PLATFORMS:
        raise ValueError(
            f"{platform!r} is not a platform this queue publishes to; "
            f"pick one of {', '.join(known_platforms())}"
        )
    conn = open_queue_db(Path(db_path).parent)
    try:
        if state == "clear":
            clear_platform_daily_cap(conn, platform)
            return
        if state not in _OVERRIDE_STATES:
            raise ValueError(
                f"{state!r} is not a recognized bounded-response state; "
                f"pick one of {', '.join(_OVERRIDE_STATES)} or 'clear'"
            )
        if cap is None:
            cap = daily_cap_for(state, None, 0)
        elif cap < 0:
            raise ValueError(f"cap must be >= 0, got {cap!r}")
        set_platform_daily_cap(
            conn, platform, cap=cap, state=state, mode="live", reason=f"manual:{state}"
        )
        logger.info(
            "platform %s manually overridden to %s (cap=%d) from the console", platform, state, cap
        )
    finally:
        conn.close()


def build_enqueue_argv(state_path: str, *, output_dir: str = "", shadow: bool = False) -> List[str]:
    """Argv for ``gomoveshift enqueue-plan <state_path> --apply [...]``,
    matching ``cli/parser.py::_add_enqueue_plan_parser`` exactly."""
    argv = [sys.executable, "-m", "gomoveshift", "enqueue-plan", str(state_path), "--apply"]
    if output_dir:
        argv += ["--output-dir", str(output_dir)]
    if shadow:
        argv.append("--shadow")
    return argv


def build_drain_argv(*, sidecar_dir: str = "") -> List[str]:
    """Argv for ``gomoveshift dispatch-queue --apply [...]``, matching
    ``cli/parser.py::_add_dispatch_queue_parser`` exactly."""
    argv = [sys.executable, "-m", "gomoveshift", "dispatch-queue", "--apply"]
    if sidecar_dir:
        argv += ["--sidecar-dir", str(sidecar_dir)]
    return argv
