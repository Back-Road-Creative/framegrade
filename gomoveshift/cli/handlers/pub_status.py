"""Handler for ``gomoveshift publishing-status``.

Ground-truth status of the first-party social publishing stack: per
platform, whether its credential file is PRESENT on disk (existence only,
NEVER contents — no decrypt, no read) and the delivery queue's row counts by
state. See ``docs/social-publishing-status.md`` for the full evidence-cited
table this command's presence check summarizes.

``--validate`` adds the other half: after restoring a backup into an
isolated instance, cross-check what the queue rows DEPEND on rather than only
counting them. Read-only by construction — it opens no adapter, imports
nothing from ``dispatch``, and issues no write — so it is safe to run on a
restore before any dispatcher is allowed near it.

``--live-catalog PATH`` adds the remote half: every ``published`` YouTube
delivery is checked against a ``live_catalog.json`` written by
``gomoveshift catalog-fetch``. A video id the channel no longer returns, or a
live title that differs from the publish plan's, is REPORTED -- never
re-queued, re-uploaded or retitled (GMS accuracy over SEO: a mismatch must
surface for a human, not be auto-fixed). This command itself makes no network
call; the catalog file is the evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ...publishing.plan_io import PublishPlanFormatError, load_publish_plan
from ...publishing.plan_title_sync import applied_retitles
from ...publishing.queue import (
    STATUS_AWAITING_IG_RECONCILE,
    STATUS_AWAITING_MANUAL_PUBLISH,
    STATUS_MISSING_REMOTE,
    TERMINAL_STATUSES,
    log_response,
    open_queue_db,
)

#: platform -> credential filename, resolved against
#: ``Path(seo.credentials_dir).parent`` — the same convention as
#: ``_yt_creds.load_youtube_credentials``, ``meta_auth.load_meta_credentials``,
#: and ``tiktok_auth.load_tiktok_credentials``. Instagram and Facebook share
#: one Meta credential file.
_CREDENTIAL_FILENAMES = {
    "youtube": "youtube_upload_token.json.enc",
    "instagram": "meta_publishing.json.enc",
    "facebook": "meta_publishing.json.enc",
    "tiktok": "tiktok_publishing.json.enc",
}

_PLATFORM_ORDER = ("youtube", "instagram", "facebook", "tiktok")


def credential_presence(credentials_dir) -> Dict[str, bool]:
    """``{platform: file exists}`` for every known platform. Checks existence
    only — never opens, decrypts, or reads the file."""
    base = Path(credentials_dir).parent
    return {platform: (base / name).exists() for platform, name in _CREDENTIAL_FILENAMES.items()}


def queue_state_counts(conn) -> Dict[str, int]:
    """``{status: row count}`` across the whole delivery queue (every
    platform, shadow rows included)."""
    rows = conn.execute("SELECT status, COUNT(*) AS n FROM deliveries GROUP BY status").fetchall()
    return {row["status"]: row["n"] for row in rows}


@dataclass(frozen=True)
class RestoreProblem:
    """One reason a restored delivery cannot do what its row claims it will.

    ``kind`` is a stable machine-readable token so a restore script can grep
    for a class of failure; ``detail`` is the human sentence.
    """

    delivery_id: int
    platform: str
    kind: str
    detail: str


def _sidecar_problem(row, plan_cache: Dict[str, object]) -> "Optional[RestoreProblem]":
    """The queue DB and the publish-plan sidecars are separate files that a
    restore can easily bring back one of. A row whose sidecar is missing, or
    is a different run's, cannot publish -- and nothing says so today until a
    drain pass fails on it, long after the restore was called good."""
    sidecar_path = row["sidecar_path"]
    if not sidecar_path:
        return RestoreProblem(
            row["id"],
            row["platform"],
            "sidecar_path_unset",
            "no publish-plan sidecar recorded, so the delivery cannot be read back",
        )
    if sidecar_path not in plan_cache:
        try:
            plan_cache[sidecar_path] = load_publish_plan(Path(sidecar_path))
        except PublishPlanFormatError as exc:
            plan_cache[sidecar_path] = exc
    plan = plan_cache[sidecar_path]
    if isinstance(plan, PublishPlanFormatError):
        return RestoreProblem(row["id"], row["platform"], "sidecar_unreadable", str(plan))
    match = any(
        short.project_name == row["project_name"] and short.short_index == row["short_index"]
        for short in plan.shorts
    )
    if not match:
        return RestoreProblem(
            row["id"],
            row["platform"],
            "short_not_in_sidecar",
            f"{sidecar_path} carries no short {row['project_name']}#{row['short_index']}",
        )
    return None


def validate_restored_state(conn, credentials_dir, *, now: "Optional[str]" = None) -> List:
    """Everything a NON-TERMINAL delivery depends on, checked against the
    files this instance actually has. Returns the problems, oldest delivery
    first; an empty list means the restore is coherent.

    Terminal rows are skipped deliberately: a published or cancelled delivery
    owes no further work, so a sidecar it no longer needs is not a restore
    defect, and listing it would bury the rows that DO still need one.

    Never writes. That is the "with dispatch disabled" half of the acceptance
    criterion -- a validator that could mutate the queue would be one more
    thing to trust rather than the thing that establishes trust.
    """
    now = now or _utcnow().isoformat()
    # ``None`` = this caller has no credentials dir to check against (the
    # publishing gate on a config without one), so presence is not judged.
    presence = credential_presence(credentials_dir) if credentials_dir is not None else None
    placeholders = ", ".join("?" for _ in TERMINAL_STATUSES)
    rows = conn.execute(
        f"SELECT * FROM deliveries WHERE status NOT IN ({placeholders}) ORDER BY id",  # noqa: S608
        TERMINAL_STATUSES,
    ).fetchall()

    problems: List[RestoreProblem] = []
    # missing_remote is terminal (nothing will re-upload it) but it is never
    # a coherent state: the queue says uploaded, YouTube says no such video.
    # A recovery row (``queue recover-terminal``) is the operator's decision
    # and resolves it; the new row is then checked like any other.
    for gone in conn.execute(
        "SELECT id, platform, platform_upload_id FROM deliveries d WHERE status=? "
        "AND NOT EXISTS (SELECT 1 FROM deliveries r WHERE r.recovers_delivery_id = d.id) "
        "ORDER BY id",
        (STATUS_MISSING_REMOTE,),
    ).fetchall():
        problems.append(
            RestoreProblem(
                gone["id"],
                gone["platform"],
                "missing_remote",
                f"uploaded as {gone['platform_upload_id']} but YouTube returns no such video; "
                "terminal, never re-uploaded unasked -- re-queueing it is the operator's per-row decision: "
                f"`gomoveshift queue recover-terminal --id {gone['id']} --authorized-by "
                "'<who, when>'` dry-runs every gate, then add `--apply` to write",
            )
        )
    plan_cache: Dict[str, object] = {}
    for row in rows:
        sidecar_problem = _sidecar_problem(row, plan_cache)
        if sidecar_problem is not None:
            problems.append(sidecar_problem)
        if presence is not None and not presence.get(row["platform"], False):
            problems.append(
                RestoreProblem(
                    row["id"],
                    row["platform"],
                    "credential_absent",
                    f"no credential file for {row['platform']}; the first drain would fail",
                )
            )
        if row["lease_owner"] and (row["lease_expires_at"] or "") > now:
            # The worker that took this lease died with the instance. Until it
            # expires, queue._has_live_foreign_lease refuses the operator's
            # cancel and reschedule, and the row reads as busy work in flight.
            problems.append(
                RestoreProblem(
                    row["id"],
                    row["platform"],
                    "stale_lease",
                    f"held by {row['lease_owner']} until {row['lease_expires_at']}; "
                    "no such worker is running after a restore",
                )
            )
        parked = _parked_too_long(row, now)
        if parked is not None:
            problems.append(parked)
        overdue = _manual_publish_overdue(row, now)
        if overdue is not None:
            problems.append(overdue)
    return problems


#: Kinds the publishing gate ignores. A live lease on a running box is work in
#: flight, not a defect -- it only means "dead worker" after a restore, which is
#: the question ``--validate`` answers and the gate does not.
#: ``manual_publish_overdue`` is a reminder for the human who taps Post, not a
#: pipeline defect: holding every other short until the operator opens TikTok would turn
#: one missed tap into a stalled queue.
_NOT_GATING_KINDS = frozenset({"stale_lease", "manual_publish_overdue"})

#: Kinds ``publishing-status --validate`` prints without failing: the plan
#: lags a retitle our own tool made, which the sync command closes. Real drift
#: (``published_title_drift``) still fails it.
_REPORT_ONLY_KINDS = frozenset({"plan_stale_retitled"})

#: The exact command an operator runs to see what the gate saw.
VALIDATE_COMMAND = "gomoveshift publishing-status --validate"


@dataclass(frozen=True)
class GateProblem:
    """A ``RestoreProblem`` with the project it belongs to: the publishing
    gate refuses per (platform, project), never the whole queue."""

    delivery_id: int
    kind: str
    platform: str
    project: str
    detail: str

    def line(self) -> str:
        return (
            f"  delivery={self.delivery_id} kind={self.kind} platform={self.platform} "
            f"project={self.project}: {self.detail}"
        )


#: The latest ``catalog-fetch --known-ids-from-queue`` result, beside the queue
#: DB. The gate reads it; only ``catalog-fetch`` writes it.
LIVE_CATALOG_SNAPSHOT = "live_catalog.latest.json"

#: A snapshot older than this is a warning, not evidence: a video deleted after
#: the fetch would be missed, and one restored would still hold its pair.
LIVE_CATALOG_MAX_AGE = timedelta(hours=24)

_REFRESH_COMMAND = "gomoveshift catalog-fetch --known-ids-from-queue --out <path>"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def live_catalog_snapshot_path(storage_path) -> Path:
    return Path(storage_path) / LIVE_CATALOG_SNAPSHOT


def _fresh_live_catalog(storage_path, now: datetime) -> "Optional[dict]":
    """The snapshot when it is at most ``LIVE_CATALOG_MAX_AGE`` old, else None
    after printing a named warning. Never fetches: a drain must not depend on
    the YouTube API being up."""
    path = live_catalog_snapshot_path(storage_path)
    if not path.exists():
        print(f"warning: live_catalog_missing: no {path}; run `{_REFRESH_COMMAND}`")
        return None
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
        fetched_at = catalog.get("fetched_at")
        fetched = datetime.fromisoformat(fetched_at) if fetched_at else None
    except (OSError, ValueError, AttributeError) as exc:
        print(f"warning: live_catalog_unreadable: {path}: {exc}; run `{_REFRESH_COMMAND}`")
        return None
    if fetched is None or now - fetched > LIVE_CATALOG_MAX_AGE:
        print(
            f"warning: live_catalog_stale: {path} fetched {fetched_at or 'at an unknown time'}, "
            f"older than 24 h; run `{_REFRESH_COMMAND}`"
        )
        return None
    return catalog


def gating_problems(
    conn, credentials_dir, *, now: "Optional[str]" = None, storage_path=None
) -> List[GateProblem]:
    """The ``--validate`` problems that stop enqueue-plan and dispatch-queue
    for their (platform, project). Read-only, like the validator it wraps.

    With ``storage_path`` it also reads the live-catalog snapshot there: a
    fresh one adds ``published_video_missing`` / ``published_title_drift``;
    a stale, missing or unreadable one is a printed warning and adds nothing."""
    found = [
        p
        for p in validate_restored_state(conn, credentials_dir, now=now)
        if p.kind not in _NOT_GATING_KINDS
    ]
    if storage_path is not None:
        at = datetime.fromisoformat(now) if now else _utcnow()
        catalog = _fresh_live_catalog(storage_path, at)
        if catalog is not None:
            found += [
                p
                for p in validate_against_live_catalog(conn, catalog)
                if p.kind not in _REPORT_ONLY_KINDS
            ]
    if not found:
        return []
    ids = sorted({p.delivery_id for p in found})
    marks = ", ".join("?" for _ in ids)
    project_of = {
        r["id"]: r["project_name"]
        for r in conn.execute(
            f"SELECT id, project_name FROM deliveries WHERE id IN ({marks})",  # noqa: S608
            ids,
        )
    }
    return [
        GateProblem(p.delivery_id, p.kind, p.platform, project_of[p.delivery_id], p.detail)
        for p in found
    ]


def record_ignore_validate(conn, problems, *, reason: str, command: str) -> None:
    """Audit an operator's ``--ignore-validate``: one ``response_log`` row per
    platform it let through, carrying the reason and the rows it overrode."""
    by_platform: Dict[str, List[GateProblem]] = {}
    for problem in problems:
        by_platform.setdefault(problem.platform, []).append(problem)
    for platform, rows in sorted(by_platform.items()):
        log_response(
            conn,
            platform,
            "validate",
            0,
            "ignore_validate",
            "live",
            json.dumps(
                {
                    "command": command,
                    "reason": reason,
                    "problems": [
                        {"delivery_id": p.delivery_id, "kind": p.kind, "project": p.project}
                        for p in rows
                    ],
                }
            ),
        )


#: A row parked at ``awaiting_ig_reconcile`` longer than this is named: each
#: hourly drain's account scan should have decided it long before.
IG_RECONCILE_STALE_HOURS = 24


def _parked_too_long(row, now: str) -> "Optional[RestoreProblem]":
    if row["status"] != STATUS_AWAITING_IG_RECONCILE:
        return None
    try:
        parked_at = datetime.fromisoformat(row["updated_at"])
        hours = (datetime.fromisoformat(now) - parked_at).total_seconds() / 3600
    except (TypeError, ValueError):
        hours = None
    if hours is not None and hours <= IG_RECONCILE_STALE_HOURS:
        return None
    age = "unknown age" if hours is None else f"{int(hours)}h"
    return RestoreProblem(
        row["id"],
        row["platform"],
        "awaiting_ig_reconcile",
        f"parked {age} since {row['updated_at']} (bound {IG_RECONCILE_STALE_HOURS}h); the "
        "account scan cannot prove whether the post is live -- check the account by hand",
    )


#: An inbox draft is named once its slot is this far behind: long enough that a
#: same-day tap is never nagged, short enough that a missed day surfaces in the
#: next morning's digest.
MANUAL_PUBLISH_GRACE_HOURS = 24


def _manual_publish_overdue(row, now: str) -> "Optional[RestoreProblem]":
    if row["status"] != STATUS_AWAITING_MANUAL_PUBLISH or not row["scheduled_at"]:
        return None
    try:
        late = datetime.fromisoformat(now) - datetime.fromisoformat(row["scheduled_at"])
    except (TypeError, ValueError):
        return None
    hours = late.total_seconds() / 3600
    if hours <= MANUAL_PUBLISH_GRACE_HOURS:
        return None
    return RestoreProblem(
        row["id"],
        row["platform"],
        "manual_publish_overdue",
        f"slot {row['scheduled_at']} passed {int(hours)}h ago (grace "
        f"{MANUAL_PUBLISH_GRACE_HOURS}h) and the draft is still waiting for Post in the app",
    )


def _plan_youtube_title(sidecar_path, project_name, short_index, plan_cache) -> "Optional[str]":
    """The YouTube title the publish plan gave this short, or ``None`` when
    the sidecar is gone/unreadable or carries no such short (unknown, which
    is not the same as a mismatch)."""
    if not sidecar_path:
        return None
    if sidecar_path not in plan_cache:
        try:
            plan_cache[sidecar_path] = load_publish_plan(Path(sidecar_path))
        except PublishPlanFormatError as exc:
            plan_cache[sidecar_path] = exc
    plan = plan_cache[sidecar_path]
    if isinstance(plan, PublishPlanFormatError):
        return None
    for short in plan.shorts:
        if short.project_name == project_name and short.short_index == short_index:
            youtube = short.payloads.youtube
            return youtube.title if youtube else None
    return None


def validate_against_live_catalog(conn, live_catalog: dict, retitles=None) -> List:
    """Every ``published`` YouTube delivery checked against the live channel.

    ``live_catalog`` is ``catalog-fetch`` output: ``{"videos": {id: {title,
    privacy, ...}}}``. Names ``published_video_missing`` when the row's
    ``platform_upload_id`` is not in it, and ``published_title_drift`` when
    the live title differs from the plan's YouTube title. Never writes.

    ``retitles`` (``plan_title_sync.applied_retitles``) names the titles our
    own ``retitle_apply`` pushed. A mismatch whose live title is the newest
    APPLIED retitle is ``plan_stale_retitled`` -- the plan lagging a change we
    made, report-only, with the sync command -- never drift.
    """
    videos = live_catalog.get("videos") or {}
    rows = conn.execute(
        "SELECT * FROM deliveries WHERE platform = 'youtube' AND status = 'published' "
        "AND platform_upload_id IS NOT NULL ORDER BY id"
    ).fetchall()
    problems: List[RestoreProblem] = []
    plan_cache: Dict[str, object] = {}
    for row in rows:
        video_id = row["platform_upload_id"]
        where = f"{row['project_name']}#{row['short_index']}"
        live = videos.get(video_id)
        if live is None:
            problems.append(
                RestoreProblem(
                    row["id"],
                    row["platform"],
                    "published_video_missing",
                    f"{where} recorded as published as {video_id}, but the live "
                    "catalog has no such video (deleted or removed on YouTube?)",
                )
            )
            continue
        planned = _plan_youtube_title(
            row["sidecar_path"], row["project_name"], row["short_index"], plan_cache
        )
        if planned is not None and (live.get("title") or "") != planned:
            retitle = (retitles or {}).get(video_id)
            if retitle is not None and retitle.title == (live.get("title") or ""):
                problems.append(
                    RestoreProblem(
                        row["id"],
                        row["platform"],
                        "plan_stale_retitled",
                        f"{where} {video_id}: retitled {retitle.at} to {retitle.title!r} "
                        f"({retitle.audit_path}); plan {row['sidecar_path']} still says "
                        f"{planned!r} -- run `gomoveshift queue sync-plan-titles --audit "
                        f"{retitle.audit_path} --apply`",
                    )
                )
                continue
            problems.append(
                RestoreProblem(
                    row["id"],
                    row["platform"],
                    "published_title_drift",
                    f"{where} {video_id}: plan title {planned!r}, live title {live.get('title')!r}",
                )
            )
    return problems


def cmd_publishing_status(args, config) -> int:
    presence = credential_presence(config.seo.credentials_dir)
    print("credential files present (existence only, never contents):")
    for platform in _PLATFORM_ORDER:
        print(f"  {platform}: {'present' if presence[platform] else 'ABSENT'}")

    conn = open_queue_db(config.seo.storage_path)
    try:
        counts = queue_state_counts(conn)
    finally:
        conn.close()

    print("\ndelivery queue row counts by state:")
    if counts:
        for status in sorted(counts):
            print(f"  {status}: {counts[status]}")
    else:
        print("  (empty)")

    if not getattr(args, "validate", False):
        # Without the flag this stays a report. A broken restore must not
        # start failing a status check that never claimed to validate.
        return 0

    live_catalog_path = getattr(args, "live_catalog", None)
    conn = open_queue_db(config.seo.storage_path)
    try:
        problems = validate_restored_state(conn, config.seo.credentials_dir)
        if live_catalog_path:
            live_catalog = json.loads(Path(live_catalog_path).read_text(encoding="utf-8"))
            retitles = applied_retitles(
                Path(a) for a in (getattr(args, "retitle_audit", None) or [])
            )
            problems += validate_against_live_catalog(conn, live_catalog, retitles)
    finally:
        conn.close()

    print("\nrestore validation (read-only; nothing is dispatched):")
    if not problems:
        print("  no problems found")
        return 0
    for problem in problems:
        print(
            f"  delivery {problem.delivery_id} ({problem.platform}) {problem.kind}: {problem.detail}"
        )
    return 1 if any(p.kind not in _REPORT_ONLY_KINDS for p in problems) else 0
