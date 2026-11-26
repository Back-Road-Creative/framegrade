"""Drain loop for the social publishing delivery queue (``queue.py``). Selects
due, non-terminal deliveries and, per row, acquires the worker lease, calls
the platform handler, and transitions the row on the result. Full contract:
``docs/social-publishing-dispatch.md``.

Handlers are injected via the ``handlers`` mapping (keyed by platform), never
imported here — wiring the real adapters in is a separate change.

A handler raises :class:`PermanentDeliveryError` to mark its delivery
``failed_permanent``. Any OTHER exception is retryable by default (left
non-terminal, ``attempt_count`` bumped) — an unrecognised exception must
never silently discard a publishable short.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Tuple

from ..console.settings import current_brand
from .brand_policy import load_brand_policy
from .ig_media_scan import SCAN_LIMIT, ScanUnprovableError, scan_for_post
from .plan_title_sync import write_plan_slot
from .queue import (
    _TERMINAL_SQL,
    FACEBOOK_PHOTO_KIND,
    STATUS_AWAITING_IG_RECONCILE,
    STATUS_MISSING_REMOTE,
    STATUS_NEEDS_OPERATOR,
    STATUS_PHOTO_AWAITING_ID,
    STATUS_PHOTO_RECONCILE,
    TERMINAL_STATUSES,
    acquire_lease,
    get_platform_daily_cap,
    is_platform_paused,
    log_response,
    parse_slot,
    pause_platform,
    reschedule_delivery,
    set_platform_daily_cap,
    transition,
)
from .queue import AWAITING_MANUAL_PUBLISH_NEXT_ATTEMPT as PARKED_UNTIL
from .queue import NEXT_ATTEMPT_NEEDS_OPERATOR as NEEDS_OPERATOR
from .queue import NEXT_ATTEMPT_PLATFORM_THROTTLE as PLATFORM_THROTTLE
from .queue import NEXT_ATTEMPT_RETRY_BACKOFF as RETRY_BACKOFF
from .queue import PUBLISHED_AT_PLATFORM_CONFIRMED as CONFIRMED
from .queue import PUBLISHED_AT_PLATFORM_RECONCILED as RECONCILED
from .response_policy import Status, consecutive_healthy_days, daily_cap_for, should_auto_pause

logger = logging.getLogger(__name__)

_DEFAULT_LEASE_SECONDS = 300

#: How far past its ``scheduled_at`` a HELD row may sit before the hold becomes
#: an incident. Two hours: longer than a normal drain cadence plus one retry, so
#: an ordinary pass-to-pass delay never alerts, but short enough that a Short
#: missing its slot is seen the same day. A hold is silent by design (no
#: transition, no attempt charged), so without this a stuck row looks like an
#: idle queue.
HELD_PAST_SLOT_MARGIN = timedelta(hours=2)
HELD_PAST_SLOT_KIND = "delivery_held_past_slot"
#: A due row whose ``scheduled_at`` cannot be read as a time at all. It is held
#: (never sent: nobody knows when it was meant to go) and named once.
UNREADABLE_SLOT_KIND = "delivery_slot_unreadable"
#: Where a row with no ``scheduled_at`` sorts: due now, so oldest of all.
_NO_SLOT = datetime.min.replace(tzinfo=timezone.utc)

#: Backoff base/cap and the retry bound. Instagram's 100-posts/24h quota is
#: account-wide — a stuck delivery must cost
#: that quota almost nothing. Doubling from 60s and capping at 3600s (1h)
#: means a delivery retries at 1, 2, 4, 8, 16, 32, 60, 60... minutes; with
#: the default 8-attempt bound that is at most 8 requests over ~3.7h before
#: it is declared permanently failed — negligible against the 100/24h
#: budget even stacked with TikTok/YouTube's own (looser) per-endpoint
#: limits. Both are parameters so a canary run can tighten them without a
#: code change.
_DEFAULT_BACKOFF_BASE_SECONDS = 60
_DEFAULT_BACKOFF_CAP_SECONDS = 3600
_DEFAULT_MAX_ATTEMPTS = 8
#: A media stall is visible once the backoff has reached its ceiling: the first attempt whose
#: delay (``base * 2**(n-1)``) is already the cap (7 at 60 s doubling to 3600 s). Derived from
#: the existing backoff, not a new bound; the row keeps retrying (``TransientMediaError`` is
#: exempt from ``max_attempts``) and the incident closes when it next uploads.
MEDIA_STALL_ATTEMPTS = next(
    n
    for n in range(1, 64)
    if _DEFAULT_BACKOFF_BASE_SECONDS * 2 ** (n - 1) >= _DEFAULT_BACKOFF_CAP_SECONDS
)
MEDIA_STALL_KIND = "delivery_media_stalled"

#: How long a delivery may keep retrying without those retries counting against
#: ``max_attempts``: from the first such retry (``deliveries.uncounted_since``)
#: to the one that parks it at ``needs_operator`` and raises a
#: ``delivery_needs_operator`` incident. 24 h is an Instagram container's whole
#: life and a full day of hourly drains: an unknown outcome, a drive, or a
#: throttle that has not cleared by then is not going to clear by itself. Before
#: this, an unknown Facebook outcome re-read its status every hour for good,
#: with no alert. Every uncounted branch goes through ``_uncounted_retry``.
UNCOUNTED_RETRY_BOUND = timedelta(hours=24)
NEEDS_OPERATOR_KIND = "delivery_needs_operator"

#: A throttled delivery waits this long regardless of `attempt_count`.
#: Six hours is well inside the daily publishing cadence (slots are
#: 12:00Z and 16:00Z) while being long enough that a throttled queue
#: stops asking TikTok the same question every hour.
_THROTTLE_BACKOFF_SECONDS = 6 * 3600

#: A TikTok inbox draft waits here for a human to publish it from the app.
#: Defined locally rather than imported from ``handlers``: handlers imports
#: this module, so the other direction would be a cycle.
_STATUS_AWAITING_MANUAL_PUBLISH = "awaiting_manual_publish"
#: A YouTube row its adapter deferred past the daily upload budget. Local too:
#: importing ``youtube_adapter`` would pull the uploader into the drain loop.
_STATUS_DEFERRED = "deferred"
# Mirrors the adapter checkpoint without importing any platform client here.
_STATUS_AWAITING_FACEBOOK_RECONCILE = "awaiting_facebook_reconcile"
#: Where a handler that returns WITHOUT publishing leaves its row. The row's
#: own wait governs it from there, never this pass's lease.
_WAITING_STATUSES = (
    STATUS_PHOTO_AWAITING_ID,
    STATUS_PHOTO_RECONCILE,
    _STATUS_AWAITING_MANUAL_PUBLISH,
    _STATUS_DEFERRED,
    STATUS_AWAITING_IG_RECONCILE,
    _STATUS_AWAITING_FACEBOOK_RECONCILE,
)
#: Never selected by the drain whatever their ``next_attempt_at``: each waits
#: on something other than a retry time (a person; the account scan).
_PARKED_STATUSES = (
    _STATUS_AWAITING_MANUAL_PUBLISH,
    STATUS_AWAITING_IG_RECONCILE,
    STATUS_PHOTO_AWAITING_ID,
    STATUS_NEEDS_OPERATOR,
)


#: Every privacy class a publishable short must have VERIFIED.
#: Mirrors ``pipeline.day_master_encoder.REQUIRED_PRIVACY_CLASSES``;
#: deliberately NOT imported from it, because that module pulls the encoder
#: (and ffmpeg) into the drain loop for a tuple of four strings.
#: ``tests/test_privacy_attestation_gate.py`` checks the two agree.
PRIVACY_REQUIRED_CLASSES = ("people", "faces", "plates", "vehicle_occupants")

#: Three states, not two: ``inconclusive`` is "no detector ran, nobody knows",
#: which must never render as a pass and must stay distinguishable from a
#: detector that ran and found the class uncovered.
PRIVACY_VERIFIED = "verified"
PRIVACY_FAILED = "failed"
PRIVACY_INCONCLUSIVE = "inconclusive"

#: Filename glob of the per-day attestations beside a run's publish plan. The
#: day-master encoder writes them into the stage dir; Phase D places them (or
#: an inconclusive record) in the dir it writes the publish-plan sidecar into,
#: which is the bridge from a delivery row to its attestation.
PRIVACY_ATTESTATION_GLOB = "day_*_privacy_attestation.json"

#: Worst-wins ordering when a run's days disagree. An unrecognised state
#: sorts worst of all: a state this code doesn't understand is not a pass.
_PRIVACY_STATE_RANK = {PRIVACY_VERIFIED: 0, PRIVACY_INCONCLUSIVE: 1, PRIVACY_FAILED: 2}


def _state_rank(state: Any) -> int:
    return _PRIVACY_STATE_RANK.get(state, 3)


def merge_privacy_attestations(
    docs: List[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Collapse a run's per-day attestations into one, worst state per class.

    A run's shorts are cut from every day's master, and a delivery row names
    the project, not the day — so one bad day holds the run. Over-holding by a
    day is the safe direction: the alternative is publishing a short whose own
    day nobody can name. ``None`` when there is nothing to merge, which
    ``privacy_hold_reason`` reads as "unattested", not "fine"."""
    merged: Dict[str, Any] = {}
    for doc in docs:
        for name, entry in (doc.get("classes") or {}).items():
            current = merged.get(name)
            if current is None or _state_rank(entry.get("state")) > _state_rank(
                current.get("state")
            ):
                merged[name] = dict(entry)
    if not merged:
        return None
    return {"classes": merged}


def privacy_hold_reason(attestation: Optional[Mapping[str, Any]]) -> Optional[str]:
    """``None`` when every required privacy class is attested ``verified``;
    otherwise the one-line reason this delivery is HELD.

    Fail-closed at both ends: an absent attestation and an absent class are
    holds, not passes, because "nobody wrote it down" and "the detector
    cleared it" are the two claims this gate exists to keep apart. The state
    string is carried through verbatim so ``inconclusive`` (no detector ran)
    never reads like ``failed`` (one ran and the class is uncovered)."""
    if not attestation:
        return "no privacy attestation covers this delivery"
    classes = attestation.get("classes") or {}
    problems = []
    for name in PRIVACY_REQUIRED_CLASSES:
        entry = classes.get(name)
        if entry is None:
            problems.append(f"{name}=unattested")
            continue
        state = entry.get("state")
        if state == PRIVACY_VERIFIED:
            continue
        reason = entry.get("reason")
        problems.append(f"{name}={state}" + (f" ({reason})" if reason else ""))
    if not problems:
        return None
    return "required privacy classes not verified: " + "; ".join(problems)


def compute_next_attempt_at(
    now: str,
    attempt_count: int,
    *,
    base_seconds: int = _DEFAULT_BACKOFF_BASE_SECONDS,
    cap_seconds: int = _DEFAULT_BACKOFF_CAP_SECONDS,
) -> str:
    """Pure function: the retryable delivery's next-due timestamp, ``now``
    plus an exponential-backoff delay (``base_seconds * 2**(attempt_count-1)``,
    capped at ``cap_seconds``) keyed off the row's NEW ``attempt_count``."""
    delay = min(base_seconds * (2 ** (attempt_count - 1)), cap_seconds)
    return (datetime.fromisoformat(now) + timedelta(seconds=delay)).isoformat()


class PermanentDeliveryError(Exception):
    """Raise from a handler to mark its delivery ``failed_permanent`` — any
    other exception is retryable by default, see the module docstring."""


class ThrottledDeliveryError(Exception):
    """Raise when the PLATFORM is refusing more of this work until a human
    acts -- not when a request merely failed.

    Deliberately EXEMPT from ``max_attempts``, like
    :class:`TransientMediaError`, and for the same reason: retrying cannot
    clear it and the delivery is not defective. TikTok's
    ``spam_risk_too_many_pending_share`` arrives once too many uploaded
    drafts sit unpublished in the account's inbox; it lifts when someone
    publishes or dismisses them. Spending the 8-attempt budget on that would
    declare ``failed_permanent`` a short whose media and credentials are
    both fine.

    Backs off on ``_THROTTLE_BACKOFF_SECONDS`` rather than the ordinary
    curve, which caps at an hour -- sized for API flakiness, far too eager
    for a state only a person can change."""


class ReconcilePendingError(Exception):
    """Raise when the platform holds this delivery's upload and has not decided
    it yet (an Instagram container still processing). Polled again on the
    ordinary backoff and EXEMPT from ``max_attempts``: the upload is fine and
    starting over could post twice. Bounded like every exemption, by
    ``UNCOUNTED_RETRY_BOUND``."""


class TransientMediaError(Exception):
    """Raise when a handler can't confirm its media still matches disk
    (missing file, moved run directory, unmounted drive — see
    ``media_fingerprint.verify_media``).

    Deliberately EXEMPT from ``max_attempts``: an unmounted drive is
    operator-fixable (remount and the next drain pass just works) and is a
    property of the local filesystem, not of the content or the platform,
    so applying the same cap sized for API flakiness would burn the last
    surviving copy of a short's delivery record for a problem retrying
    can't make worse. This branch never transitions ``failed_permanent``, so
    `attempt_count` here is for visibility only; ``UNCOUNTED_RETRY_BOUND``
    parks the row at ``needs_operator`` instead."""


@dataclass
class DrainResult:
    """One ``drain_queue`` pass's outcome. Empty/all-skipped is success."""

    attempted: int = 0
    succeeded: int = 0
    retried: int = 0
    failed_permanent: int = 0
    skipped_leased: int = 0
    skipped_no_handler: int = 0
    #: Per-platform breakdown of ``skipped_no_handler`` -- which platforms
    #: were blocked, and how many due deliveries each had, not just how many
    #: total. Always consistent with ``skipped_no_handler``: its values
    #: sum to that count. Only platforms that actually had a skip this pass
    #: appear as keys -- see ``route_inventory.route_states`` for the full
    #: routed/not_routed table, which this is not (it says nothing about a
    #: not-routed platform with no due work this pass).
    skipped_no_handler_by_platform: Dict[str, int] = field(default_factory=dict)
    #: Due deliveries left untouched because their media's required privacy
    #: classes are not all attested ``verified``. NOT a failure count: the row
    #: keeps its status, its lease and its attempt count, so restoring the
    #: model and re-running the day publishes the very same delivery.
    held_privacy_unverified: int = 0
    #: Due deliveries left untouched because ``publishing-status --validate``
    #: names a problem on their (platform, project). Same contract as the
    #: privacy hold: no lease, no transition, no attempt charged.
    held_validate: int = 0
    #: YouTube Shorts moved to the next free slot because another Short already
    #: held that clock hour (this pass or the queue).
    moved_same_hour: int = 0
    #: Due deliveries left queued because their platform already had its one
    #: post this pass. NOT a failure or a hold: no lease, no transition, no
    #: attempt charged; the next pass takes the next one.
    spaced_one_per_pass: int = 0
    #: Due deliveries held because their ``scheduled_at`` is not a readable
    #: time. Same contract as the other holds; each raises one incident.
    held_unreadable_slot: int = 0
    #: Subset of `retried`, counted separately: a throttled pass is a
    #: standing instruction to a human, not queue noise.
    throttled: int = 0
    #: Deliveries whose worker died holding the lease, reconciled at the top
    #: of this pass. NOT a subset of the counters above: none of them was
    #: attempted by THIS pass, and a non-zero value here is a report about a
    #: previous worker, not about this one.
    reconciled_interrupted: int = 0
    #: Rows this pass parked at ``needs_operator``: their uncounted retries ran
    #: past ``UNCOUNTED_RETRY_BOUND``. Not in ``retried``; each has an incident.
    needs_operator: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def blocked_no_route(self) -> bool:
        """True when this pass had due work it could not even attempt because
        no handler is registered for the platform -- distinct from a
        genuinely idle pass, where ``_select_due`` returned nothing at all.

        ``attempted=0 succeeded=0 failed_permanent=0`` reads identically for
        both today: a healthy idle hour and an hour where every due
        delivery was skipped for a missing route look the same on the one
        summary line an operator or a cron log actually reads.

        For WHICH platform(s) were blocked and how many deliveries each, see
        ``skipped_no_handler_by_platform``; for the full routed/not_routed
        table (including platforms with no due work this pass), see
        ``route_inventory.route_states``."""
        return self.skipped_no_handler > 0

    @property
    def held_for_privacy_review(self) -> bool:
        """True when this pass refused to publish media whose required privacy
        classes were not verified. Same blind spot ``blocked_no_route`` covers:
        a held pass and a healthy idle pass both report ``attempted=0
        succeeded=0 failed_permanent=0`` on the one line a cron log keeps."""
        return self.held_privacy_unverified > 0


def _slot_time(value: Any) -> Optional[datetime]:
    """A stored slot as an aware time, the one way dispatch reads one: ``None``
    when unset, a zoneless time read as UTC (rows written before
    ``queue.normalise_slot``). An unreadable one raises ``ValueError``;
    ``_select_due`` holds such a row back, so nothing after it meets one."""
    if not value:
        return None
    parsed = parse_slot(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _slot_order(row: Dict[str, Any]) -> Tuple[datetime, int]:
    """Oldest slot first (no slot is oldest of all), then ``id``."""
    return (_slot_time(row.get("scheduled_at")) or _NO_SLOT, row["id"])


def _select_due(
    conn,
    now: str,
    *,
    platform: Optional[str] = None,
    status: Optional[str] = None,
    scheduled_through: Optional[str] = None,
    unreadable: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Non-terminal, NON-SHADOW rows whose ``next_attempt_at`` AND
    ``scheduled_at`` have both arrived (or are unset). Read-only; no
    lease/status side effects.

    The two times answer different questions: ``next_attempt_at`` is the
    retry backoff, ``scheduled_at`` is the plan's publish time. Gating on
    the schedule here rather than in the drain loop means no caller -- the
    CLI, the cron, a future worker -- can publish a short early by taking a
    different path, exactly as with shadow rows below. Only the YouTube
    adapter defers server-side (``publish_at``); Instagram, TikTok and
    Facebook publish the moment they are called, so without this one
    ``--apply`` fired a whole week of shorts at once. A NULL
    ``scheduled_at`` (a row enqueued before the column existed) stays due:
    stranding it silently is the failure this guard exists to prevent.

    Shadow rows are excluded here rather than skipped inside the drain loop
    so that no caller — the CLI, the cron, a future worker — can publish one
    by taking a different path. A shadow run has to be safe even when
    ``dispatch-queue --apply`` runs by accident.

    A paused platform (``queue.pause_platform``) is excluded the same way,
    for the same reason: this is the one place the CLI, the cron and any
    future worker all funnel through, so a pause enforced only in the drain
    loop or a console route could be routed around. A paused row is simply
    never selected here -- no status change, no attempt bump, no error --
    because leaving it alone, not failing it, is what a pause means. A row
    already leased when the pause took effect is unaffected: this query
    governs what is picked up NEXT, not work already in flight.

    A row whose ``scheduled_at`` date (or ``now``'s date, for a NULL
    ``scheduled_at``) is one of the current brand's ``policy.blackout`` dates
    is excluded the same way -- no status change, no attempt bump -- so it
    is picked up automatically once the blackout date passes. ``deliveries``
    carries no ``brand`` column (this service runs one brand instance at a
    time, selected by ``GMS_CONSOLE_BRAND``), so the policy is resolved once
    per call via ``console.settings.current_brand``, the same resolver
    ``enqueue-plan`` and the day pipeline already use for brand policy.

    A parked TikTok draft (``awaiting_manual_publish``) is excluded by its
    STATUS, not only by its ``9999-01-01`` retry time. Any write that moved
    ``next_attempt_at`` put it back in front of the drain, which then read
    the handler's early return for a parked row as a publish. Only
    ``reconcile_awaiting_manual`` closes a parked row.

    The keyword arguments let a second caller narrow the same funnel instead
    of re-implementing it: ``platform`` and ``status`` restrict the rows (a
    ``status`` replaces the parked-row exclusion with an exact match), and
    ``scheduled_through`` moves the ``scheduled_at`` ceiling past ``now``. The
    TikTok pack sweep (``tiktok_pack.sweep_queued``) packs files up to nine
    days before their slot, yet must still honour the pause, shadow, backoff,
    blackout, quiet-window and daily-cap rules above. ``now`` stays the clock
    for the backoff, the quiet window and the caps.

    A row whose ``scheduled_at`` is not a readable time is left out too, and
    appended to ``unreadable`` when the caller passes a list (the drain raises
    an incident for it). Every later step parses the slot; one bad row there
    used to raise and end the whole pass."""
    conditions = [f"status NOT IN {_TERMINAL_SQL}", "shadow = 0"]
    params: List[Any] = []
    if status is None:
        conditions.append("status NOT IN (" + ",".join("?" for _ in _PARKED_STATUSES) + ")")
        params.extend(_PARKED_STATUSES)
    else:
        conditions.append("status = ?")
        params.append(status)
    if platform is not None:
        conditions.append("platform = ?")
        params.append(platform)
    conditions.append("platform NOT IN (SELECT platform FROM platform_pauses)")
    conditions.append("(next_attempt_at IS NULL OR next_attempt_at <= ?)")
    params.append(now)
    conditions.append("(scheduled_at IS NULL OR scheduled_at <= ?)")
    params.append(scheduled_through or now)
    rows = conn.execute(
        "SELECT * FROM deliveries WHERE " + " AND ".join(conditions) + " ORDER BY id",  # noqa: S608
        params,
    ).fetchall()
    due = []
    for row in map(dict, rows):
        try:
            _slot_time(row.get("scheduled_at"))
        except ValueError:
            if unreadable is not None:
                unreadable.append(row)
            continue
        due.append(row)
    due = _exclude_blackout(due, now)
    due = _exclude_quiet_window(conn, due, now)
    return _apply_daily_caps(conn, due, now)


def _exclude_blackout(due: List[Dict[str, Any]], now: str) -> List[Dict[str, Any]]:
    policy = load_brand_policy(current_brand())
    if not policy.blackout:
        return due
    blackout_dates = set(policy.blackout)
    now_date = datetime.fromisoformat(now).date()
    kept = []
    for row in due:
        slot = _slot_time(row.get("scheduled_at"))
        row_date = slot.date() if slot else now_date
        if row_date in blackout_dates:
            continue
        kept.append(row)
    return kept


def _exclude_quiet_window(conn, due: List[Dict[str, Any]], now: str) -> List[Dict[str, Any]]:
    """Hold a platform's rows for ``policy.quiet_after_alert_hours``
    hours after its most recent SUPPRESSED *alert* -- the ``response_log``
    transition INTO ``SUPPRESSED`` (the preceding row for that platform, by
    ``id``, has a different state, or there is no preceding row). A nightly
    re-log while still SUPPRESSED does not restart the clock: only the
    transition row's ``at`` anchors the window. ``RECOVERING`` and
    ``HEALTHY`` are never alerts. Scoped to the one platform whose alert
    fired -- no status change, no attempt bump, no error for its rows,
    exactly the "leave it alone" contract ``_exclude_blackout`` and the
    pause exclusion already give themselves. Only ``mode='live'`` rows
    count, matching how ``_apply_daily_caps`` treats shadow decisions as
    inert. ``quiet_after_alert_hours`` unset means no hold and no query at
    all -- the common case pays nothing for this feature."""
    policy = load_brand_policy(current_brand())
    hours = policy.quiet_after_alert_hours
    if hours is None:
        return due
    platforms = {row["platform"] for row in due}
    if not platforms:
        return due
    now_dt = parse_response_log_at(now)
    held_platforms = set()
    for platform in platforms:
        alert_at = platform_quiet_alert_at(conn, platform)
        if alert_at is None:
            continue
        elapsed = (now_dt - parse_response_log_at(alert_at)).total_seconds()
        if elapsed < hours * 3600:
            held_platforms.add(platform)
    if not held_platforms:
        return due
    return [row for row in due if row["platform"] not in held_platforms]


def platform_quiet_alert_at(conn, platform: str) -> Optional[str]:
    """The ``at`` of ``platform``'s most recent SUPPRESSED *alert* --
    the ``response_log`` transition INTO ``SUPPRESSED`` (the row's
    immediately preceding row for this platform, by ``id``, has a
    different state, or there is no preceding row). ``None`` when the
    platform has never alerted. Only ``mode='live'`` rows count. Shared by
    ``_exclude_quiet_window`` and the console Publishing tab's
    ``held: quiet`` label so both compute the window identically."""
    rows = conn.execute(
        "SELECT state, at FROM response_log WHERE platform = ? AND mode = 'live' ORDER BY id",
        (platform,),
    ).fetchall()
    alert_at = None
    prev_state = None
    for row in rows:
        state = row["state"]
        if state == "SUPPRESSED" and prev_state != "SUPPRESSED":
            alert_at = row["at"]
        prev_state = state
    return alert_at


def parse_response_log_at(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _apply_daily_caps(conn, due: List[Dict[str, Any]], now: str) -> List[Dict[str, Any]]:
    """Honour ``queue.platform_daily_caps`` on top of the base
    due-selection above. ``mode='live'`` actually truncates a platform's
    due rows to ``cap - <published today>``, keeping the lowest
    (``scheduled_at``, ``id``) rows -- the same priority order the
    "top-scored queued short" ordering uses (no score hook exists yet;
    ``id`` is the tie-break until it does). ``mode='shadow'``
    changes nothing about what is selected, only logs what a live cap
    would have limited, so the 14-day shadow log can be compared against
    what an operator would have done without ever touching a real delivery.

    Enforced HERE, the one funnel `_select_due` already is, rather than in
    the drain loop -- exactly the reasoning the pause exclusion above
    gives for itself."""
    caps = conn.execute("SELECT * FROM platform_daily_caps").fetchall()
    if not caps:
        return due
    today = now[:10]
    for cap_row in caps:
        platform = cap_row["platform"]
        platform_rows = [row for row in due if row["platform"] == platform]
        if not platform_rows:
            continue
        published_today = conn.execute(
            "SELECT COUNT(*) FROM deliveries WHERE platform=? AND status='published' "
            "AND substr(published_at, 1, 10) = ?",
            (platform, today),
        ).fetchone()[0]
        remaining = max(0, cap_row["cap"] - published_today)
        if len(platform_rows) <= remaining:
            continue
        if cap_row["mode"] == "shadow":
            logger.info(
                "shadow: would cap %s at %d (%d due, state=%s)",
                platform,
                cap_row["cap"],
                len(platform_rows),
                cap_row["state"],
            )
            continue
        if cap_row["mode"] != "live":
            continue
        prioritized = sorted(platform_rows, key=lambda row: (row["scheduled_at"] or "", row["id"]))
        keep_ids = {row["id"] for row in prioritized[:remaining]}
        due = [row for row in due if row["platform"] != platform or row["id"] in keep_ids]
    return due


_ERROR_DETAIL_MAX = 2000


def _detail(exc: BaseException) -> str:
    """The exception's message, bounded, for ``deliveries.error_detail``.

    ``error_class`` alone is not diagnosable: a run of deliveries can
    record ``RuntimeError`` and nothing else, with the message that explains
    them surviving only as the single line a cron wrapper happened to keep.
    Bounded because a platform SDK can raise a very large body.
    """
    text = str(exc).strip() or exc.__class__.__name__
    return text[:_ERROR_DETAIL_MAX]


#: What an interrupted attempt records instead of an exception class. There is
#: no exception to name: the worker was killed between ``acquire_lease`` and
#: any ``transition``, so nothing in this process ever saw a failure.
_INTERRUPTED_ERROR_CLASS = "InterruptedAttempt"


def reconcile_interrupted_attempts(
    conn,
    *,
    now: str,
    backoff_base_seconds: int = _DEFAULT_BACKOFF_BASE_SECONDS,
    backoff_cap_seconds: int = _DEFAULT_BACKOFF_CAP_SECONDS,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> int:
    """Charge an attempt to every delivery whose worker died holding it, and
    return how many were reconciled.

    A worker killed between ``acquire_lease`` and any ``transition`` -- OOM,
    a host restart, SIGKILL -- leaves ``attempt_count`` untouched. The lease
    expires, the row becomes due again, and it is retried. That part already
    works. What does not is the counting: every ``max_attempts`` check in
    ``drain_queue`` lives inside an ``except`` clause, and a SIGKILL raises
    nothing, so a delivery that reliably kills its worker is retried forever.
    In the console it reads "queued, 0 attempts" throughout -- the same thing
    a delivery that was never tried reads.

    An EXPIRED lease is the signal, and the only one available: a worker that
    finishes either terminalizes the row or releases the lease, so a lease
    outliving its own expiry means nobody is coming back for it. A LIVE lease
    is left strictly alone -- that worker may be mid platform-API call, and
    clearing its lease would let a second worker publish the same short twice.

    This does NOT decide whether the interrupted attempt reached the platform.
    Retained Facebook upload IDs have an ambiguous outcome and only retry status
    reads, with backoff, beyond the usual attempt limit, for up to
    ``UNCOUNTED_RETRY_BOUND`` (then ``needs_operator``). Other crash loops stop
    at that limit.

    A PAUSED platform's row is selected here (unlike ``_select_due``) because
    the lease itself needs releasing regardless: it is pure bookkeeping about
    a worker that is provably dead, costs the paused platform nothing, and
    leaves the row ready to be picked up the moment the pause lifts. What the
    pause DOES stop is charging the attempt or transitioning the row toward
    ``failed_permanent`` -- either would decide the outcome of a delivery to
    a platform an operator deliberately told this system to leave alone, and
    a crash-looping worker must not be able to burn a paused platform's whole
    retry budget while nobody is watching.

    A parked TikTok draft (``awaiting_manual_publish``) gets the same release
    and no attempt: it waits on a person, and charging it set a retry time
    that un-parked it. ``drain_queue`` now releases its own
    lease when a handler parks or defers a row, so this is the fallback for
    a worker that died in between. A ``deferred`` row is still charged: it
    is selectable, so a worker dying on it every pass is a crash loop.
    """
    rows = conn.execute(
        "SELECT * FROM deliveries "  # noqa: S608 -- _TERMINAL_SQL is a fixed literal, not input
        f"WHERE status NOT IN {_TERMINAL_SQL} "
        "AND lease_owner IS NOT NULL "
        "AND lease_expires_at IS NOT NULL "
        "AND lease_expires_at <= ? ORDER BY id",
        (now,),
    ).fetchall()
    reconciled = 0
    for row in rows:
        delivery_id, status, platform = row["id"], row["status"], row["platform"]
        if is_platform_paused(conn, platform):
            transition(conn, delivery_id, status, status, lease_owner=None, lease_expires_at=None)
            logger.info(
                "dispatch: delivery %s's stale lease was released without charging an "
                "attempt because %s is paused",
                delivery_id,
                platform,
            )
            continue
        if status in _PARKED_STATUSES:
            # A parked draft waits on a person, not on a dead worker. Charging
            # it set a retry time that un-parked it.
            transition(conn, delivery_id, status, status, lease_owner=None, lease_expires_at=None)
            logger.info(
                "dispatch: delivery %s's stale lease was released without charging an "
                "attempt because it is parked (%s)",
                delivery_id,
                status,
            )
            continue
        row = _checkpoint_facebook_upload(conn, row)
        if row is None or row["status"] in TERMINAL_STATUSES:
            continue
        status = row["status"]
        attempt_count = row["attempt_count"] + 1
        detail = (
            f"worker {row['lease_owner']} held this delivery until "
            f"{row['lease_expires_at']} and never reported an outcome; counted "
            "as an attempt; an ambiguous Facebook upload remains read-only"
        )
        if _facebook_unknown(row):
            _uncounted_retry(
                conn,
                row,
                status,
                now,
                next_attempt_at=compute_next_attempt_at(
                    now,
                    attempt_count,
                    base_seconds=backoff_base_seconds,
                    cap_seconds=backoff_cap_seconds,
                ),
                next_attempt_reason=RETRY_BACKOFF,
                attempt_count=attempt_count,
                error_class=_INTERRUPTED_ERROR_CLASS,
                error_detail=detail,
            )
        elif attempt_count >= max_attempts:
            transition(
                conn,
                delivery_id,
                status,
                "failed_permanent",
                attempt_count=attempt_count,
                error_class=f"MaxAttemptsExceeded({_INTERRUPTED_ERROR_CLASS})",
                error_detail=detail,
                lease_owner=None,
                lease_expires_at=None,
            )
        else:
            transition(
                conn,
                delivery_id,
                status,
                status,
                attempt_count=attempt_count,
                # Whatever killed the worker is probably still true a second
                # later, so an interrupted attempt backs off exactly like an
                # ordinary retryable failure rather than spinning at drain
                # frequency.
                next_attempt_at=compute_next_attempt_at(
                    now,
                    attempt_count,
                    base_seconds=backoff_base_seconds,
                    cap_seconds=backoff_cap_seconds,
                ),
                next_attempt_reason=RETRY_BACKOFF,
                error_class=_INTERRUPTED_ERROR_CLASS,
                error_detail=detail,
                lease_owner=None,
                lease_expires_at=None,
            )
        reconciled += 1
        logger.warning(
            "dispatch: delivery %s was interrupted by a worker that never came back (%s)",
            delivery_id,
            row["lease_owner"],
        )
    return reconciled


def _live_row(conn, delivery_id: int):
    """The live receipt/checkpoint NOW, or ``None`` if it is gone."""
    row = conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    return dict(row) if row is not None else None


def _facebook_unknown(row) -> bool:
    return bool(
        row is not None
        and row["platform"] == "facebook"
        and (
            (row["status"] == _STATUS_AWAITING_FACEBOOK_RECONCILE and row["platform_upload_id"])
            or (
                row["upload_kind"] == FACEBOOK_PHOTO_KIND
                and row["status"] in (STATUS_PHOTO_AWAITING_ID, STATUS_PHOTO_RECONCILE)
            )
        )
        and not row["platform_post_id"]
    )


def _checkpoint_facebook_upload(conn, row):
    """Old retained IDs have ambiguous finish state too; never exhaust them."""
    if (
        row is not None
        and row["platform"] == "facebook"
        and row["upload_kind"] != FACEBOOK_PHOTO_KIND
        and row["platform_upload_id"]
        and not row["platform_post_id"]
        and row["status"] not in (*TERMINAL_STATUSES, _STATUS_AWAITING_FACEBOOK_RECONCILE)
    ):
        transition(conn, row["id"], row["status"], _STATUS_AWAITING_FACEBOOK_RECONCILE)
        return _live_row(conn, row["id"])
    return row


def _uncounted_retry(
    conn, row, status: str, now: str, *, next_attempt_at: str, next_attempt_reason: str, **fields
) -> Optional[str]:
    """Record a retry that does not count against ``max_attempts``, bounded.

    The ONLY way any branch skips ``max_attempts``
    (``tests/test_delivery_uncounted_bounds.py`` checks each one). The row's run
    of such retries starts at its first (``uncounted_since``); once it has
    lasted ``UNCOUNTED_RETRY_BOUND`` the row parks at ``needs_operator`` with
    its platform ids kept, so it is never re-uploaded and never selected again,
    and ``sync_media_stall_incidents`` raises one incident for it. A Page-photo
    attempt cannot change status (``transition`` refuses), so it keeps its
    read-only retries and gets the incident all the same. Returns ``"parked"``,
    ``"retried"`` or ``None`` when nothing was written. ``fields`` carry
    ``attempt_count`` and the error."""
    since = row["uncounted_since"] or now
    try:
        elapsed = datetime.fromisoformat(now) - datetime.fromisoformat(since)
    except (TypeError, ValueError):  # unreadable or mixed-naive: the run starts now
        since, elapsed = now, timedelta(0)
    common = dict(fields, uncounted_since=since, lease_owner=None, lease_expires_at=None)
    if elapsed >= UNCOUNTED_RETRY_BOUND:
        hours = elapsed.total_seconds() / 3600
        detail = (
            f"parked from {status} after {hours:.0f} h of retries that do not count against "
            f"max_attempts (since {since}); platform ids kept, never re-uploaded; a person "
            f"decides it: {common.get('error_detail') or ''}"
        )
        parked = dict(common, error_detail=detail[:_ERROR_DETAIL_MAX])
        if transition(
            conn,
            row["id"],
            status,
            STATUS_NEEDS_OPERATOR,
            next_attempt_at=PARKED_UNTIL,
            next_attempt_reason=NEEDS_OPERATOR,
            **parked,
        ):
            logger.warning("dispatch: delivery %s parked at needs_operator", row["id"])
            return "parked"
    if transition(
        conn,
        row["id"],
        status,
        status,
        next_attempt_at=next_attempt_at,
        next_attempt_reason=next_attempt_reason,
        **common,
    ):
        return "retried"
    return None


def _release_own_lease(conn, delivery_id: int, owner: str) -> None:
    """Drop the lease this pass holds when nothing else will: a failure that
    could not be recorded, or a row its handler left waiting
    (``_WAITING_STATUSES``).

    Only the lease fields, only when ``owner`` still holds it: a lease another
    worker has since taken is theirs, and a terminal row's receipt is untouched.
    ``transition`` refuses terminal rows, so it cannot do this."""
    with conn:
        conn.execute(
            "UPDATE deliveries SET lease_owner=NULL, lease_expires_at=NULL "
            "WHERE id=? AND lease_owner=?",
            (delivery_id, owner),
        )


def _unrecorded_failure(
    conn, result: DrainResult, platform: str, delivery_id: int, owner: str, exc: BaseException
) -> None:
    """A handler failed but the row could not be moved to record it: it went
    terminal (or changed) while the handler ran. Never counted as a failure or
    a retry -- nothing was written -- but never silent either."""
    _release_own_lease(conn, delivery_id, owner)
    live = _live_row(conn, delivery_id)
    live_status = live["status"] if live is not None else "missing"
    result.errors.append(
        f"{platform}#{delivery_id}: handler raised {type(exc).__name__} but the delivery is "
        f"now {live_status!r}, so the failure was not recorded: {exc}"
    )
    logger.warning(
        "dispatch: delivery %s raised %s but is now %r; failure not recorded",
        delivery_id,
        type(exc).__name__,
        live_status,
    )


def _floor_hour(moment: datetime) -> datetime:
    return moment.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _raise_held_incident(
    row: Dict[str, Any], now: str, reason: str, incidents_dir: Optional[Path]
) -> None:
    """One incident per delivery held past its slot by more than the margin.

    Never rewritten while open (the row is skipped if its file exists unresolved),
    never fatal: an unwritable incidents dir, or a ``scheduled_at`` written
    before ``queue.normalise_slot`` (unreadable, or with no zone, which made the
    subtraction raise ``TypeError``), must not stop the drain. A zoneless time
    is read as UTC (``_slot_time``). ``pushed`` stays
    False; ``health check``'s generic ``detail`` fallback pushes it."""
    scheduled_at = row.get("scheduled_at")
    try:
        slot = _slot_time(scheduled_at)
    except ValueError:
        logger.warning("dispatch: delivery %s has an unreadable slot %r", row["id"], scheduled_at)
        return
    if slot is None:
        return
    late = parse_response_log_at(now) - slot
    if late <= HELD_PAST_SLOT_MARGIN:
        return
    hours = late.total_seconds() / 3600
    _write_delivery_incident(
        row,
        now,
        HELD_PAST_SLOT_KIND,
        f"{row['platform']}#{row['id']} {row['project_name']} short "
        f"{row['short_index']} is {hours:.1f} h past its {scheduled_at} slot: {reason}",
        incidents_dir,
    )


def _write_delivery_incident(
    row: Dict[str, Any], now: str, kind: str, detail: str, incidents_dir: Optional[Path]
) -> None:
    """Write one ``kind`` incident for ``row`` unless one is already open. Never fatal."""
    from ..health.store import (  # lazy: health imports the pipeline
        default_data_root,
        default_incidents_dir,
        new_incident,
        safe_name,
        write_row,
    )

    key = f"{kind}:{row['id']}"
    directory = incidents_dir or default_incidents_dir(default_data_root())
    filename = f"{safe_name(key)}.json"
    try:
        existing = directory / filename
        if existing.is_file() and json.loads(existing.read_text()).get("resolved_at") is None:
            return
        write_row(
            directory,
            new_incident(
                kind,
                key,
                filename,
                now,
                delivery_id=row["id"],
                platform=row["platform"],
                project=row["project_name"],
                scheduled_at=row.get("scheduled_at"),
                detail=detail,
            ),
        )
    except (OSError, ValueError):
        logger.exception("dispatch: could not write %s incident for %s", kind, row["id"])


def sync_media_stall_incidents(conn, now: str, incidents_dir: Optional[Path] = None) -> None:
    """Keep one open incident per stuck row and resolve the rest. Two kinds:

    - ``delivery_media_stalled``: a non-terminal row whose last failure was
      ``TransientMediaError`` and whose attempts reached :data:`MEDIA_STALL_ATTEMPTS`
      (it keeps retrying);
    - ``delivery_needs_operator``: a row parked at ``needs_operator``, or one whose
      uncounted retries have run past ``UNCOUNTED_RETRY_BOUND`` without parking
      (a Page-photo attempt, which cannot change status).

    An incident whose row is no longer in its state (it uploaded, its media came
    back, a person closed it) is resolved. Never changes a row and never fatal."""
    from ..health.store import (  # lazy: health imports the pipeline
        default_data_root,
        default_incidents_dir,
        load_all,
        new_incident,
        safe_name,
        write_row,
    )

    terminal = ", ".join("?" for _ in TERMINAL_STATUSES)
    try:
        cutoff = (datetime.fromisoformat(now) - UNCOUNTED_RETRY_BOUND).isoformat()
        stuck = conn.execute(
            "SELECT id, platform, project_name, short_index, attempt_count, error_detail, "  # noqa: S608
            "CASE WHEN status = ? OR uncounted_since <= ? THEN ? ELSE ? END AS kind "
            "FROM deliveries WHERE (status = ? OR uncounted_since <= ? OR "
            "(error_class = 'TransientMediaError' AND attempt_count >= ?)) "
            f"AND status NOT IN ({terminal})",
            (
                STATUS_NEEDS_OPERATOR,
                cutoff,
                NEEDS_OPERATOR_KIND,
                MEDIA_STALL_KIND,
                STATUS_NEEDS_OPERATOR,
                cutoff,
                MEDIA_STALL_ATTEMPTS,
                *TERMINAL_STATUSES,
            ),
        ).fetchall()
        directory = incidents_dir or default_incidents_dir(default_data_root())
        live = {(r["id"], r["kind"]) for r in stuck}
        rows, _ = load_all(directory)
        open_keys = set()
        for row in rows:
            kind = row.get("kind")
            if kind not in (MEDIA_STALL_KIND, NEEDS_OPERATOR_KIND) or row.get("resolved_at"):
                continue
            if (row.get("delivery_id"), kind) in live:
                open_keys.add((row.get("delivery_id"), kind))
            else:
                row["resolved_at"] = now
                write_row(directory, row)
        for r in stuck:
            if (r["id"], r["kind"]) in open_keys:
                continue
            if r["kind"] == MEDIA_STALL_KIND:
                what = (
                    f"media unreadable for {r['attempt_count']} attempts "
                    "(backoff at its ceiling); still retrying"
                )
            else:
                what = (
                    f"retried {UNCOUNTED_RETRY_BOUND} without an outcome, outside "
                    "max_attempts; parked for a person, platform ids kept"
                )
            key = f"{r['kind']}:{r['id']}"
            write_row(
                directory,
                new_incident(
                    r["kind"],
                    key,
                    f"{safe_name(key)}.json",
                    now,
                    delivery_id=r["id"],
                    platform=r["platform"],
                    project=r["project_name"],
                    detail=(
                        f"{r['platform']}#{r['id']} {r['project_name']} short {r['short_index']}: "
                        f"{what}: {r['error_detail']}"
                    )[:500],
                ),
            )
    except (OSError, ValueError):
        logger.exception("dispatch: could not sync media-stall incidents")


def _defer_same_hour(conn, row: Dict[str, Any], now: str, released_hours: set) -> bool:
    """Keep one YouTube Short per clock hour. ``True`` when this row was moved.

    Its hour is its ``scheduled_at`` hour (the ``publishAt`` the platform goes
    public at). The first due row in an hour claims it; any later one, or one
    sharing the hour with a Short already uploaded/published for it, moves to the
    next free hour at the same minute, sidecar first (as ``queue-reslot`` does).
    A row whose sidecar cannot be written is left as it was."""
    now_dt = datetime.fromisoformat(now)
    slot = _slot_time(row.get("scheduled_at")) or now_dt
    sibling_hours = set()
    for (value,) in conn.execute(
        "SELECT COALESCE(platform_published_at, scheduled_at) FROM deliveries "
        "WHERE platform='youtube' AND id<>? AND shadow=0 "
        "AND status NOT IN ('cancelled', 'failed_permanent', 'missing_remote') "
        "AND COALESCE(platform_published_at, scheduled_at) IS NOT NULL",
        (row["id"],),
    ):
        try:
            sibling_hours.add(_floor_hour(_slot_time(value)))
        except ValueError:
            continue  # a sibling's unreadable slot holds no hour
    here = _floor_hour(slot)
    # Rows queued for the same hour only conflict once one is released; a
    # published/uploaded sibling already holds the hour.
    held_elsewhere = conn.execute(
        "SELECT 1 FROM deliveries WHERE platform='youtube' AND id<>? AND shadow=0 "
        "AND status IN ('uploaded', 'published') "
        "AND strftime('%Y-%m-%dT%H', COALESCE(platform_published_at, scheduled_at)) = ?",
        (row["id"], here.strftime("%Y-%m-%dT%H")),
    ).fetchone()
    if here not in released_hours and not held_elsewhere:
        released_hours.add(here)
        return False
    taken = sibling_hours | released_hours
    candidate = slot
    for _ in range(24 * 14):
        candidate += timedelta(hours=1)
        if candidate > now_dt and _floor_hour(candidate) not in taken:
            break
    else:
        return False
    if row.get("sidecar_path"):
        try:
            write_plan_slot(
                Path(row["sidecar_path"]), row["project_name"], row["short_index"], candidate
            )
        except (OSError, ValueError, KeyError):
            logger.exception("dispatch: could not move %s to %s", row["id"], candidate)
            return False
    if not reschedule_delivery(conn, row["id"], candidate.isoformat(), now=now):
        return False
    released_hours.add(_floor_hour(candidate))
    logger.info(
        "dispatch: delivery %s moved %s -> %s (one YouTube Short per hour)",
        row["id"],
        row.get("scheduled_at"),
        candidate.isoformat(),
    )
    return True


def drain_queue(
    conn,
    handlers: Mapping[str, Callable[[Dict[str, Any]], None]],
    *,
    owner: Optional[str] = None,
    lease_seconds: int = _DEFAULT_LEASE_SECONDS,
    now: Optional[str] = None,
    backoff_base_seconds: int = _DEFAULT_BACKOFF_BASE_SECONDS,
    backoff_cap_seconds: int = _DEFAULT_BACKOFF_CAP_SECONDS,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    privacy_attestations: Optional[Callable[[Dict[str, Any]], Optional[Mapping[str, Any]]]] = None,
    held_scopes: Optional[Collection[Tuple[str, str]]] = None,
    facebook_photo_reconciler: Optional[Callable[[Dict[str, Any]], None]] = None,
    incidents_dir: Optional[Path] = None,
) -> DrainResult:
    """One drain pass. Never raises on an empty queue, an unavailable lease,
    or a single handler's exception — all recorded on ``DrainResult`` so one
    bad delivery never aborts the pass for the rest.

    ``privacy_attestations`` resolves a delivery row to the privacy attestation
    covering its media; a row whose required classes are not all ``verified``
    is HELD, never attempted. Injected like ``handlers`` rather than read from
    disk here — the writer lives under ``pipeline/`` and importing it would
    drag the encoder into the drain loop. Left ``None``, the gate is inert, so
    wiring the loader in is what arms it.

    ``held_scopes`` is the validate gate: ``(platform, project)`` pairs whose
    rows are HELD the same way, because the validator names a problem on that
    pair. Every other pair drains normally. A row held more than
    ``HELD_PAST_SLOT_MARGIN`` past its ``scheduled_at`` (by either hold) writes one
    ``delivery_held_past_slot`` incident under ``incidents_dir`` (default: the
    health incidents dir).

    Two YouTube Shorts are never released in one clock hour: the second due row
    moves to the next free slot (row and sidecar ``publishAt``).

    At most one delivery per platform is sent in one pass, on every platform,
    oldest slot first.
    After an outage every overdue row is due at once, and Facebook, Instagram
    and TikTok post the moment they are called, so a single pass used to post
    a whole backlog together, which the platforms read as spam. The rest stay
    queued, untouched, for the next pass (``spaced_one_per_pass``). Enforced
    here, at the lease, rather than in ``_select_due``: a held row must not use
    up its platform's turn, and the TikTok pack sweep shares that selection.

    ``facebook_photo_reconciler`` is an explicitly trusted read-only port for
    persisted Page-photo kinds. It runs without current-draft validation or
    privacy loaders. If absent, those rows stay held instead of falling into
    legacy handlers. It must never start an upload or invoke fresh authority.
    """
    now = now or datetime.now(timezone.utc).isoformat()
    owner = owner or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    expires_at = (datetime.fromisoformat(now) + timedelta(seconds=lease_seconds)).isoformat()

    result = DrainResult()
    # Before selecting anything: a worker that died holding a lease never got
    # to record its attempt. Reconciling here rather than in a separate
    # startup hook means the CLI, the cron and any future worker all get it --
    # the same reason _select_due gates on the schedule instead of the loop.
    result.reconciled_interrupted = reconcile_interrupted_attempts(
        conn,
        now=now,
        backoff_base_seconds=backoff_base_seconds,
        backoff_cap_seconds=backoff_cap_seconds,
        max_attempts=max_attempts,
    )
    released_hours: set = set()
    sent_platforms: set = set()
    unreadable: List[Dict[str, Any]] = []
    due = _select_due(conn, now, unreadable=unreadable)
    for row in unreadable:
        result.held_unreadable_slot += 1
        reason = f"held: scheduled_at {row['scheduled_at']!r} is not a readable time"
        result.errors.append(f"{row['platform']}#{row['id']}: {reason}")
        _write_delivery_incident(
            row, now, UNREADABLE_SLOT_KIND, f"{row['platform']}#{row['id']} {reason}", incidents_dir
        )
    # Oldest slot first, so each platform's one turn this pass goes to the row
    # that has waited longest, not to the lowest id.
    for row in sorted(due, key=_slot_order):
        delivery_id, platform, status = row["id"], row["platform"], row["status"]

        # Confirm a handler exists BEFORE acquiring the lease — otherwise a
        # platform with no adapter wired up locks its rows for the full
        # lease duration for no reason.
        # A durable photo identity never falls back to current-record routing.
        # Only this explicit read-only port may bypass pre-publication checks.
        photo_read = platform == "facebook" and row["upload_kind"] == FACEBOOK_PHOTO_KIND
        handler = facebook_photo_reconciler if photo_read else handlers.get(platform)
        if handler is None:
            result.skipped_no_handler += 1
            result.skipped_no_handler_by_platform[platform] = (
                result.skipped_no_handler_by_platform.get(platform, 0) + 1
            )
            logger.warning("dispatch: no handler registered for platform %r", platform)
            continue

        if not photo_read and held_scopes and (platform, row["project_name"]) in held_scopes:
            result.held_validate += 1
            reason = (
                f"validate hold: {platform}/{row['project_name']} "
                "has a publishing-status --validate problem"
            )
            result.errors.append(f"{platform}#{delivery_id}: {reason}")
            _raise_held_incident(row, now, reason, incidents_dir)
            continue

        # Same reasoning, one rung earlier in the same "check before locking"
        # sequence: publication is the irreversible step, so media whose
        # required privacy classes were never verified is refused HERE rather
        # than at the encode, and refused without taking a lease it would only
        # hold for the full duration. A hold is not a failure — no transition,
        # no attempt bump — so the delivery publishes normally once the model
        # is restored and the day re-run.
        if not photo_read and privacy_attestations is not None:
            try:
                hold = privacy_hold_reason(privacy_attestations(row))
            except Exception as exc:  # noqa: BLE001 -- unreadable == unattested
                # Fail closed. An attestation that can't be read is not a
                # privacy pass; the row stays exactly where it is.
                hold = f"privacy attestation unreadable: {_detail(exc)}"
            if hold is not None:
                result.held_privacy_unverified += 1
                result.errors.append(f"{platform}#{delivery_id}: privacy hold: {hold}")
                logger.warning(
                    "dispatch: delivery %s held, privacy not verified: %s", delivery_id, hold
                )
                _raise_held_incident(row, now, f"privacy hold: {hold}", incidents_dir)
                continue

        if platform == "youtube" and not photo_read:
            if _defer_same_hour(conn, row, now, released_hours):
                result.moved_same_hour += 1
                continue

        # One post per platform per pass. The read-only photo reconciler posts
        # nothing, so it never takes or waits for a turn.
        if not photo_read:
            if platform in sent_platforms:
                result.spaced_one_per_pass += 1
                continue
            sent_platforms.add(platform)

        if not acquire_lease(conn, delivery_id, owner, expires_at, now=now):
            result.skipped_leased += 1  # another worker holds a live lease
            continue

        result.attempted += 1
        try:
            row = _checkpoint_facebook_upload(conn, row)
            if row is None or row["status"] in TERMINAL_STATUSES:
                continue
            handler(row)
        except Exception as exc:  # noqa: BLE001 - classified below; default path is retryable
            # Handlers move the row themselves before they can fail (queued ->
            # submitting -> uploaded), so ``status`` -- read when the row was
            # SELECTED -- is stale by now. ``transition`` is a compare-and-set,
            # so expecting it wrote nothing, lost the error text, and still
            # got counted (seen with Instagram). Every failure is
            # recorded from the row's LIVE status instead, and only counted
            # when the write actually happened.
            live = _checkpoint_facebook_upload(conn, _live_row(conn, delivery_id))
            # A terminal (or vanished) row is never moved: ``transition``
            # would refuse it anyway, but a "retry" of a published row asks it
            # to re-enter 'published', which raises before the refusal.
            # ``live_status is None`` below means "nothing to record into".
            live_status = (
                live["status"]
                if live is not None and live["status"] not in TERMINAL_STATUSES
                else None
            )
            new_attempt_count = (live["attempt_count"] if live is not None else 0) + 1
            outcome = None
            if isinstance(exc, PermanentDeliveryError) and not _facebook_unknown(live):
                recorded = live_status is not None and transition(
                    conn,
                    delivery_id,
                    live_status,
                    "failed_permanent",
                    error_class=type(exc).__name__,
                    error_detail=_detail(exc),
                )
                if recorded:
                    result.failed_permanent += 1
                    result.errors.append(f"{platform}#{delivery_id}: permanent: {exc}")
            elif isinstance(exc, ThrottledDeliveryError):
                # Retry mechanics of the ordinary retryable branch, but NEVER
                # checked against max_attempts and on a fixed long delay -- see
                # the class docstring.
                outcome = live_status and _uncounted_retry(
                    conn,
                    live,
                    live_status,
                    now,
                    next_attempt_at=(
                        datetime.fromisoformat(now) + timedelta(seconds=_THROTTLE_BACKOFF_SECONDS)
                    ).isoformat(),
                    next_attempt_reason=PLATFORM_THROTTLE,
                    attempt_count=new_attempt_count,
                    error_class=type(exc).__name__,
                    error_detail=_detail(exc),
                )
                recorded = bool(outcome)
                if outcome == "retried":
                    result.retried += 1
                    result.throttled += 1
                    result.errors.append(f"{platform}#{delivery_id}: throttled: {exc}")
            elif isinstance(exc, (TransientMediaError, ReconcilePendingError)) or _facebook_unknown(
                live
            ):
                # Same retry mechanics as the ordinary retryable branch below
                # (backoff, lease release) but NEVER checked against
                # max_attempts — see the class docstrings. Bounded in time.
                outcome = live_status and _uncounted_retry(
                    conn,
                    live,
                    live_status,
                    now,
                    next_attempt_at=compute_next_attempt_at(
                        now,
                        new_attempt_count,
                        base_seconds=backoff_base_seconds,
                        cap_seconds=backoff_cap_seconds,
                    ),
                    next_attempt_reason=RETRY_BACKOFF,
                    attempt_count=new_attempt_count,
                    error_class=type(exc).__name__,
                    error_detail=_detail(exc),
                )
                recorded = bool(outcome)
                if outcome == "retried":
                    result.retried += 1
                    reason = (
                        "transient media"
                        if isinstance(exc, TransientMediaError)
                        else "reconciliation pending"
                    )
                    result.errors.append(f"{platform}#{delivery_id}: {reason}: {exc}")
            elif new_attempt_count >= max_attempts:
                recorded = live_status is not None and transition(
                    conn,
                    delivery_id,
                    live_status,
                    "failed_permanent",
                    attempt_count=new_attempt_count,
                    error_class=f"MaxAttemptsExceeded({type(exc).__name__})",
                    error_detail=_detail(exc),
                )
                if recorded:
                    result.failed_permanent += 1
                    result.errors.append(
                        f"{platform}#{delivery_id}: max attempts ({max_attempts}) exceeded: {exc}"
                    )
            else:
                recorded = live_status is not None and transition(
                    conn,
                    delivery_id,
                    live_status,
                    live_status,
                    attempt_count=new_attempt_count,
                    next_attempt_at=compute_next_attempt_at(
                        now,
                        new_attempt_count,
                        base_seconds=backoff_base_seconds,
                        cap_seconds=backoff_cap_seconds,
                    ),
                    next_attempt_reason=RETRY_BACKOFF,
                    error_class=type(exc).__name__,
                    error_detail=_detail(exc),
                    uncounted_since=None,  # an ordinary failure ends an uncounted run
                    # Release the lease we just held. next_attempt_at is the
                    # only thing that should govern when this row is retried;
                    # a lingering 300s lease would swallow every backoff
                    # shorter than the lease (the first three, by default),
                    # so tuning backoff_base_seconds would appear to do
                    # nothing.
                    lease_owner=None,
                    lease_expires_at=None,
                )
                if recorded:
                    result.retried += 1
                    result.errors.append(f"{platform}#{delivery_id}: retryable: {exc}")
            if outcome == "parked":
                result.needs_operator += 1
                result.errors.append(
                    f"{platform}#{delivery_id}: needs_operator: its retries never counted "
                    f"against max_attempts and it has retried {UNCOUNTED_RETRY_BOUND} without an "
                    f"outcome; parked with its platform ids for a person: {exc}"
                )
            if not recorded:
                _unrecorded_failure(conn, result, platform, delivery_id, owner, exc)
        else:
            # Returning normally is not always publishing: the TikTok inbox
            # handler parks a draft for a human, the YouTube adapter defers a
            # row past its daily upload budget. The row's own wait governs it
            # now, so this pass's lease goes and nothing is counted. Held, the
            # lease expired into an "interrupted attempt" whose retry time
            # un-parked the draft, and the next pass read the handler's early
            # return for it as a publish.
            live = _live_row(conn, delivery_id)
            if live is not None and live["status"] == STATUS_MISSING_REMOTE:
                # U-R1: the handler uploaded, then found YouTube does not
                # list the video. Terminal and never a success.
                _release_own_lease(conn, delivery_id, owner)
                result.failed_permanent += 1
                result.errors.append(
                    f"{platform}#{delivery_id}: missing_remote: uploaded but YouTube "
                    "returns no such video; not re-uploaded"
                )
                continue
            if live is not None and live["status"] in _WAITING_STATUSES:
                _release_own_lease(conn, delivery_id, owner)
                logger.info(
                    "dispatch: delivery %s is %s; nothing was published this pass",
                    delivery_id,
                    live["status"],
                )
                continue
            transition(conn, delivery_id, status, "published", published_at_source=CONFIRMED)
            result.succeeded += 1

    sync_media_stall_incidents(conn, now, incidents_dir)
    return result


#: The TikTok ``status/fetch`` outcomes ``reconcile_awaiting_manual`` acts on.
_PUBLISH_COMPLETE = "PUBLISH_COMPLETE"
_PUBLISH_FAILED = "FAILED"


@dataclass
class ReconcileResult:
    """One ``reconcile_awaiting_manual`` pass. Nothing to do is success."""

    polled: int = 0
    published: int = 0
    failed: int = 0
    errors: List[str] = field(default_factory=list)


def reconcile_awaiting_manual(
    conn,
    *,
    poll: Callable[[str], Mapping[str, Any]],
    permalink_for: Callable[[str], str],
    platform: str = "tiktok",
) -> ReconcileResult:
    """Close the rows a human has already published from the TikTok app.

    An inbox draft parks at ``awaiting_manual_publish`` with
    ``next_attempt_at=9999-01-01``, a status ``_select_due`` never returns,
    so ``drain_queue`` cannot move it. Before this, the only way
    the row ever closed was someone clicking the console's mark-published
    button; unclicked, a live post read as an outstanding chore forever.

    ``poll(publish_id)`` returns TikTok's ``status/fetch`` data dict. Shape of the live response::

        {'status': 'PUBLISH_COMPLETE',
         'publicaly_available_post_id': [7000000000001]}

    (``publicaly_`` is TikTok's spelling.) The post id trails the tap by a
    minute or two of moderation, so ``PUBLISH_COMPLETE`` without one means
    "not yet", never "published with no id" — storing a NULL post id is the
    ambiguity the YouTube rows already carry.

    Best effort by construction: this runs BEFORE the drain, and any row's
    failure is recorded rather than raised, so a TikTok API blip can never
    stop the shorts that are due today from publishing.

    A PAUSED platform is skipped entirely, before ``poll`` is ever called.
    Unlike the interrupted-lease reconcile there is no bookkeeping half to
    salvage here: every action this function takes -- calling ``poll`` and
    transitioning to ``published``/``failed_permanent`` -- is itself a
    platform-facing decision about the delivery's outcome, which is exactly
    what a pause means to stop touching.
    """
    result = ReconcileResult()
    if is_platform_paused(conn, platform):
        return result
    rows = conn.execute(
        "SELECT id, platform_upload_id FROM deliveries "
        "WHERE status=? AND platform=? AND shadow=0 ORDER BY id",
        (_STATUS_AWAITING_MANUAL_PUBLISH, platform),
    ).fetchall()

    for row in rows:
        delivery_id, publish_id = row["id"], row["platform_upload_id"]
        if not publish_id:
            # Nothing to ask about: the row parked without an upload id.
            continue
        try:
            result.polled += 1
            data = poll(publish_id) or {}
            status = data.get("status")

            if status == _PUBLISH_COMPLETE:
                post_ids = data.get("publicaly_available_post_id") or []
                if not post_ids:
                    continue  # moderation still running; next pass
                post_id = str(post_ids[0])
                # Built BEFORE the transition and passed INTO it. `transition`
                # refuses every write once a row is terminal, so a permalink
                # fetched afterwards is unstorable (a
                # cosmetic post-publish call once classified a successful publish
                # as failed_permanent; only the terminal guard saved it).
                transition(
                    conn,
                    delivery_id,
                    _STATUS_AWAITING_MANUAL_PUBLISH,
                    "published",
                    platform_post_id=post_id,
                    permalink=permalink_for(post_id),
                    published_at_source=RECONCILED,
                )
                result.published += 1
                logger.info(
                    "reconcile: delivery %s published in the app as %s", delivery_id, post_id
                )

            elif status == _PUBLISH_FAILED:
                transition(
                    conn,
                    delivery_id,
                    _STATUS_AWAITING_MANUAL_PUBLISH,
                    "failed_permanent",
                    error_class="tiktok_publish_failed",
                    error_detail=str(data.get("fail_reason") or status)[:_ERROR_DETAIL_MAX],
                )
                result.failed += 1

            # Any other status (SEND_TO_USER_INBOX, PROCESSING_*) is a draft
            # still waiting on a human. Leave it exactly where it is.
        except Exception as exc:  # noqa: BLE001 — one bad row never aborts the pass
            result.errors.append(f"{platform}#{delivery_id}: reconcile: {_detail(exc)}")
            logger.warning("reconcile: delivery %s could not be reconciled", delivery_id)

    return result


def apply_suppression_response(
    conn,
    *,
    brand: str,
    states: Mapping[str, Mapping[str, Status]],
    today: str,
    ladder: Optional[Any],
    mode: str = "shadow",
    now: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Compute and record each platform's bounded response for
    ``today`` from its already-classified daily ``states`` (``tracking.
    suppression.classify_*``'s output, ``{platform: {day: Status}}``).

    Default ``mode="shadow"``: every decision is written to
    ``queue.response_log`` and ``queue.platform_daily_caps`` (both tagged
    ``mode="shadow"``) but nothing is ever paused or actually limits
    ``dispatch._select_due`` -- a shadow-mode row there only makes
    ``_select_due`` LOG what it would have capped. This function never
    flips ``mode`` itself; there is no config switch to live in this
    change -- the shadow window is the operator's decision, made by calling
    this again with ``mode="live"``.

    ``mode="live"`` additionally calls ``queue.pause_platform`` with
    ``reason="auto:suppressed"`` when :func:`response_policy.
    should_auto_pause` is true for that platform's state history.

    ``brand`` is not yet used to scope anything (every known platform in
    ``states`` is processed) -- it is accepted now so the eventual CLI
    wiring doesn't need a signature change; recorded here so a
    future caller and its test both see the parameter is real.

    Returns one summary dict per platform: ``platform``, ``state``,
    ``cap``, ``healthy_days``, ``auto_pause`` (bool), ``mode``."""
    now = now or _now_iso_for_response()
    decisions: List[Dict[str, Any]] = []
    for platform, day_states in states.items():
        state = day_states.get(today, "HEALTHY")
        healthy_days = consecutive_healthy_days(day_states, today)

        existing = get_platform_daily_cap(conn, platform)
        if existing is not None and (existing.get("reason") or "").startswith("manual:"):
            # An operator's console override sticks -- the nightly
            # run must never clobber it back to an automatic decision.
            log_response(
                conn,
                platform,
                state=existing["state"],
                cap=existing["cap"],
                action="skipped-manual",
                mode=mode,
                reason=existing.get("reason"),
                now=now,
            )
            decisions.append(
                {
                    "platform": platform,
                    "state": existing["state"],
                    "cap": existing["cap"],
                    "healthy_days": healthy_days,
                    "auto_pause": False,
                    "mode": mode,
                }
            )
            continue

        cap = daily_cap_for(state, ladder, healthy_days)
        auto_pause = state == "SUPPRESSED" and should_auto_pause(day_states, today)

        set_platform_daily_cap(
            conn,
            platform,
            cap=cap,
            state=state,
            mode=mode,
            reason=f"w2:{brand}:{state.lower()}",
            now=now,
        )
        log_response(
            conn,
            platform,
            state=state,
            cap=cap,
            action="cap_set",
            mode=mode,
            reason=f"w2:{brand}:{state.lower()}",
            now=now,
        )

        if auto_pause:
            if mode == "live":
                pause_platform(conn, platform, now=now, reason="auto:suppressed")
                log_response(
                    conn,
                    platform,
                    state=state,
                    cap=cap,
                    action="auto_pause",
                    mode=mode,
                    reason="auto:suppressed",
                    now=now,
                )
            else:
                log_response(
                    conn,
                    platform,
                    state=state,
                    cap=cap,
                    action="would_pause",
                    mode=mode,
                    reason="auto:suppressed",
                    now=now,
                )

        decisions.append(
            {
                "platform": platform,
                "state": state,
                "cap": cap,
                "healthy_days": healthy_days,
                "auto_pause": auto_pause,
                "mode": mode,
            }
        )
    return decisions


def _now_iso_for_response() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class IgReconcileResult:
    """One ``reconcile_awaiting_ig`` pass. Nothing to do is success."""

    considered: int = 0
    published: int = 0
    retried: int = 0
    failed: int = 0
    unprovable: int = 0
    errors: List[str] = field(default_factory=list)


_IG_ABSENT_ERROR_CLASS = "IgReconcileAbsent"


def reconcile_awaiting_ig(
    conn,
    *,
    recent_media: Callable[[int], List[Dict[str, Any]]],
    record_loader: Callable[[str, int], Any],
    now: str,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    backoff_base_seconds: int = _DEFAULT_BACKOFF_BASE_SECONDS,
    backoff_cap_seconds: int = _DEFAULT_BACKOFF_CAP_SECONDS,
) -> IgReconcileResult:
    """Decide every Instagram row parked at ``awaiting_ig_reconcile`` from a
    scan of the account's recent media -- the same scan ``queue
    recover-terminal`` gate (c) runs (``ig_media_scan.scan_for_post``):

    - a match: the post is live, so the row closes ``published`` with the
      matched media id (``platform_reconciled``);
    - proven absent: back to ``queued`` with one attempt charged, the old
      container cleared so the next pass builds a fresh one, and the ordinary
      backoff; at ``max_attempts`` it fails permanently instead;
    - unprovable (empty page, a page short of the row's creation): the row
      stays parked, untouched. ``publishing-status --validate`` names one
      parked too long.

    ``recent_media(limit)`` is read once per pass, lazily. Best effort like
    ``reconcile_awaiting_manual``: one row's failure is recorded, never
    raised, and a paused Instagram is never scanned."""
    result = IgReconcileResult()
    if is_platform_paused(conn, "instagram"):
        return result
    rows = conn.execute(
        "SELECT * FROM deliveries WHERE status=? AND platform='instagram' AND shadow=0 ORDER BY id",
        (STATUS_AWAITING_IG_RECONCILE,),
    ).fetchall()
    media: Optional[List[Dict[str, Any]]] = None
    for row in rows:
        delivery_id = row["id"]
        result.considered += 1
        try:
            record = record_loader(row["project_name"], row["short_index"])
            if media is None:
                media = list(recent_media(SCAN_LIMIT))
            scan = scan_for_post(
                media,
                created_at=row["created_at"],
                caption=record.payloads.instagram.caption,
                sha256=getattr(record.media, "sha256", None),
            )
        except ScanUnprovableError as exc:
            result.unprovable += 1
            result.errors.append(f"instagram#{delivery_id}: still parked: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 -- one bad row never aborts the pass
            result.errors.append(f"instagram#{delivery_id}: ig reconcile: {_detail(exc)}")
            continue

        if scan.matches:
            found = scan.matches[0]
            if transition(
                conn,
                delivery_id,
                STATUS_AWAITING_IG_RECONCILE,
                "published",
                platform_post_id=str(found["id"]),
                permalink=found.get("permalink"),
                published_at_source=RECONCILED,
            ):
                result.published += 1
            continue

        attempt_count = row["attempt_count"] + 1
        detail = (
            f"account scan of {scan.scanned} media (oldest {scan.oldest_scanned or 'whole account'}) "
            f"found no post; container {row['platform_upload_id']} dropped for a fresh one"
        )
        if attempt_count >= max_attempts:
            moved = transition(
                conn,
                delivery_id,
                STATUS_AWAITING_IG_RECONCILE,
                "failed_permanent",
                attempt_count=attempt_count,
                error_class=f"MaxAttemptsExceeded({_IG_ABSENT_ERROR_CLASS})",
                error_detail=detail,
            )
            result.failed += int(moved)
            continue
        moved = transition(
            conn,
            delivery_id,
            STATUS_AWAITING_IG_RECONCILE,
            "queued",
            attempt_count=attempt_count,
            platform_upload_id=None,
            next_attempt_at=compute_next_attempt_at(
                now,
                attempt_count,
                base_seconds=backoff_base_seconds,
                cap_seconds=backoff_cap_seconds,
            ),
            next_attempt_reason=RETRY_BACKOFF,
            error_class=_IG_ABSENT_ERROR_CLASS,
            error_detail=detail,
        )
        result.retried += int(moved)
    return result
