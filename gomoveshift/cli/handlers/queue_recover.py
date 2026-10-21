"""Handlers for ``gomoveshift queue recover-terminal`` and ``mark-missing-remote``.

``retry_delivery`` refuses a ``failed_permanent`` row by design, and raw SQL is
not an approved way past that. This is the gated, audited way: it inserts a NEW
delivery linked to the terminal one (``queue.recover_terminal_delivery``) and
only when every gate below holds. Any gate failing prints a one-line reason,
exits non-zero and writes nothing.

Gates, in order (cheap and local first, Graph reads last):

a. the row is ``failed_permanent`` and not already recovered;
b. the old container (``platform_upload_id``) reads EXPIRED or not-found from
   the Graph API -- a NULL id counts as gone. Any other state refuses: a live
   or PUBLISHED container means the post may exist;
c. a FRESH remote scan of the account's recent media finds no caption-prefix
   match and no media-fingerprint (sha256) match. An empty scan is refused:
   it cannot prove absence. A full page must reach back past the row's
   ``created_at``, or a post the failed attempt made could sit beyond it;
d. the sidecar record loads and its media verifies and resolves to the same
   Instagram rendition path the drain uploads (never the raw over-cap master);
e. the privacy attestation permits it, decided by ``dispatch.privacy_hold_reason``
   exactly as the drain decides a hold. An absent attestation refuses.

A YouTube row recovers only from ``missing_remote`` (the channel stopped
returning its video), through its own gates, in order:

a. the row is ``missing_remote`` and not already recovered;
b. a FRESH ``videos.list`` on the old id still returns nothing. A video that is
   back refuses, and so does a lookup that errors: unknown is not gone;
f. the plan's YouTube slot is still ahead of now. A passed slot refuses (a
   private upload with a past ``publishAt`` would never go public);
d. the sidecar record loads and its media verifies, the file the drain uploads;
e. privacy, exactly as above;
c. the channel's recent uploads carry no video with the plan's exact title. An
   empty listing is refused: it cannot prove absence.

A recovery row resolves the ``missing_remote`` validate problem; until one
exists that problem names this command.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ...publishing.dispatch import privacy_hold_reason
from ...publishing.ig_media_scan import SCAN_LIMIT, scan_for_post
from ...publishing.plan_title_sync import applied_retitles, sync_plan_title
from ...publishing.queue import (
    RecoveryRefusedError,
    check_recoverable,
    mark_missing_remote,
    open_queue_db,
    recover_terminal_delivery,
)

#: Container states that mean "gone": it can never publish.
_GONE_STATES = frozenset({"EXPIRED", "NOT_FOUND"})


class InstagramRecoveryGraph:
    """The two read-only Graph calls the gates need, over the existing
    ``InstagramReelsAdapter`` (no hand-rolled HTTP). Tests pass a fake."""

    def __init__(self, reels):
        self._reels = reels

    @property
    def account_id(self) -> Optional[str]:
        return getattr(self._reels, "account_id", None)

    def container_state(self, container_id: str) -> str:
        from ...publishing.instagram_adapter import InstagramPermanentError

        try:
            body = self._reels.check_container_status(container_id)
        except InstagramPermanentError as exc:
            # Graph answers a deleted/expired-away object with "... does not
            # exist ..."; any OTHER permanent rejection is unknown, not gone.
            if "does not exist" in str(exc):
                return "NOT_FOUND"
            raise
        return body.get("status_code") or "UNKNOWN"

    def recent_media(self, limit: int) -> List[Dict[str, Any]]:
        return self._reels.list_recent_media(limit)


#: How many recent channel uploads the YouTube duplicate scan reads.
YOUTUBE_SCAN_LIMIT = 200


class YouTubeRecoveryReader:
    """The two read-only Data API calls the YouTube gates need, over the
    pipeline's own service and ``list_uploads``. Tests pass a fake."""

    def __init__(self, service, lister=None):
        self._service = service
        if lister is None:
            from ...publishing.youtube_uploads_listing import list_uploads as lister
        self._lister = lister

    def video_exists(self, video_id: str) -> bool:
        resp = self._service.videos().list(part="status", id=video_id).execute()
        return bool(resp.get("items"))

    def recent_uploads(self, limit: int) -> List[Dict[str, Any]]:
        return [
            {"id": u.video_id, "title": u.title} for u in self._lister(self._service, limit=limit)
        ]


@dataclass(frozen=True)
class RecoveryDeps:
    # instagram: container_state(id), recent_media(limit);
    # youtube: video_exists(id), recent_uploads(limit)
    graph: Any
    record_loader: Callable[[str, int], Any]
    privacy_loader: Callable[[Any], Optional[Dict]]
    rendition_path: Callable[[Any], Path]


def recover_terminal(
    conn,
    delivery_id: int,
    authorized_by: str,
    deps: RecoveryDeps,
    now: Optional[str] = None,
    *,
    apply: bool = True,
) -> Optional[int]:
    """Run the platform's gates; on pass insert the linked row and return its
    id. Raises ``RecoveryRefusedError`` (nothing written) otherwise. With
    ``apply=False`` every gate still runs (the same reads) and nothing is
    written; returns None -- the CLI's default dry run."""
    if not (authorized_by or "").strip():
        raise RecoveryRefusedError("--authorized-by must be a non-empty statement")
    row = check_recoverable(conn, delivery_id)  # (a)
    platform = row["platform"]
    if platform not in _RECOVERS_FROM:
        raise RecoveryRefusedError(
            f"recover-terminal supports instagram and youtube, not {platform!r}"
        )
    if row["status"] != _RECOVERS_FROM[platform]:
        raise RecoveryRefusedError(
            f"{platform} recovers {_RECOVERS_FROM[platform]} only; "
            f"delivery {delivery_id} is {row['status']!r}"
        )
    gates = _youtube_gates if platform == "youtube" else _instagram_gates
    evidence = gates(row, deps, now or datetime.now(timezone.utc).isoformat())
    if not apply:
        return None
    return recover_terminal_delivery(
        conn, delivery_id, authorized_by=authorized_by, evidence=evidence, now=now
    )


#: The one terminal status each platform recovers from.
_RECOVERS_FROM = {"instagram": "failed_permanent", "youtube": "missing_remote"}


def _loaded_media(row, deps: RecoveryDeps):
    """Gate (d): the sidecar record and the media path the drain uploads."""
    try:
        record = deps.record_loader(row["project_name"], row["short_index"])
    except Exception as exc:  # noqa: BLE001 - any load failure is a refusal
        raise RecoveryRefusedError(f"sidecar record unavailable: {exc}") from exc
    try:
        media = deps.rendition_path(record)
    except Exception as exc:  # noqa: BLE001
        raise RecoveryRefusedError(f"sidecar media not deliverable: {exc}") from exc
    if not Path(media).is_file():
        raise RecoveryRefusedError(f"sidecar media not deliverable: {media} is not a file")
    return record, media


def _privacy_gate(row, deps: RecoveryDeps) -> None:
    """Gate (e), by the drain's own rule."""
    held = privacy_hold_reason(deps.privacy_loader(row))
    if held is not None:
        raise RecoveryRefusedError(f"privacy hold: {held}")


def _youtube_gates(row, deps: RecoveryDeps, now: str) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {"status": row["status"]}
    video_id = row["platform_upload_id"]
    # (b) the old video is still gone
    try:
        back = deps.graph.video_exists(video_id)
    except Exception as exc:  # noqa: BLE001 - unknown is not gone
        raise RecoveryRefusedError(f"could not confirm {video_id} is gone: {exc}") from exc
    if back:
        raise RecoveryRefusedError(
            f"YouTube returns {video_id} again; it is not missing -- nothing to recover"
        )
    evidence["old_video"] = {"id": video_id, "returned": False}

    record, media = _loaded_media(row, deps)  # (d)
    # (f) the slot is still ahead
    slot = record.schedule.scheduled_at_utc
    if _parse_utc(slot) <= _parse_utc(now):
        raise RecoveryRefusedError(
            f"slot {slot} has passed; a re-upload would never go public -- re-slot first"
        )
    evidence["slot"] = slot
    evidence["rendition_path"] = str(media)

    _privacy_gate(row, deps)  # (e)
    evidence["privacy"] = "verified"

    # (c) no same-title video already on the channel
    title = record.payloads.youtube.title
    uploads = list(deps.graph.recent_uploads(YOUTUBE_SCAN_LIMIT))
    if not uploads:
        raise RecoveryRefusedError("channel uploads scan came back empty; absence is unproven")
    dupes = [u["id"] for u in uploads if u.get("title") == title]
    if dupes:
        raise RecoveryRefusedError(f"the channel already carries {title!r}: videos {dupes}")
    evidence["duplicate_scan"] = {"scanned": len(uploads), "matches": 0}
    return evidence


def _parse_utc(stamp: str) -> datetime:
    parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _instagram_gates(row, deps: RecoveryDeps, now: str) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {"status": row["status"]}

    # (b) container
    upload_id = row["platform_upload_id"]
    if upload_id is None:
        state = "NOT_FOUND"
    else:
        state = deps.graph.container_state(upload_id)
    if state not in _GONE_STATES:
        raise RecoveryRefusedError(
            f"old container {upload_id} reads {state}, not EXPIRED/not-found; "
            "it may still publish or already has"
        )
    evidence["container"] = {"id": upload_id, "state": state}

    # (d) record + rendition (before the Graph scan: local failures are free)
    record, rendition = _loaded_media(row, deps)
    evidence["rendition_path"] = str(rendition)

    _privacy_gate(row, deps)  # (e)
    evidence["privacy"] = "verified"

    # (c) fresh remote duplicate scan (the one reconcile_awaiting_ig uses)
    media = list(deps.graph.recent_media(SCAN_LIMIT))
    scan = scan_for_post(
        media,
        created_at=row["created_at"],
        caption=record.payloads.instagram.caption,
        sha256=getattr(record.media, "sha256", None),
    )
    if scan.matches:
        ids = [m.get("id") for m in scan.matches]
        raise RecoveryRefusedError(f"remote duplicate found on the account: media {ids}")
    evidence["duplicate_scan"] = {
        "scanned": scan.scanned,
        "matches": 0,
        "oldest_scanned": scan.oldest_scanned,
        "account": getattr(deps.graph, "account_id", None),
    }
    return evidence


def _build_real_deps(conn, config, sidecar_dir: Path, platform: str) -> RecoveryDeps:
    from .dispatch_queue import PrivacyAttestationLoader, SidecarRecordLoader, _media_path

    if platform == "youtube":
        from ...pipeline.drive_phases.youtube_uploader import YouTubeUploader

        return RecoveryDeps(
            graph=YouTubeRecoveryReader(YouTubeUploader()._build_service()),
            record_loader=SidecarRecordLoader(sidecar_dir, conn=conn),
            privacy_loader=PrivacyAttestationLoader(sidecar_dir),
            rendition_path=_media_path,
        )
    from ...publishing.instagram_adapter import InstagramReelsAdapter
    from ...publishing.meta_auth import load_meta_credentials
    from ...publishing.tiktok_rendition import ensure_instagram_rendition
    from .dispatch_queue import PrivacyAttestationLoader, SidecarRecordLoader, _verified_path

    meta = load_meta_credentials(Path(config.seo.credentials_dir))

    def rendition(record) -> Path:
        # Same resolution as dispatch_queue._read_instagram_media, minus the read.
        source = _verified_path(record)
        m = record.media
        bitrate = int(m.size_bytes * 8 / m.duration_seconds) if m.duration_seconds > 0 else None
        return ensure_instagram_rendition(source, bitrate_bps=bitrate)

    return RecoveryDeps(
        graph=InstagramRecoveryGraph(InstagramReelsAdapter(meta)),
        record_loader=SidecarRecordLoader(sidecar_dir, conn=conn),
        privacy_loader=PrivacyAttestationLoader(sidecar_dir),
        rendition_path=rendition,
    )


@dataclass
class MissingRemotePlan:
    """One ``mark-missing-remote`` pass: ``(delivery_id, video_id)`` pairs."""

    marked: List[tuple] = field(default_factory=list)
    #: Published rows whose id the catalog never asked about -- unproven.
    unchecked: List[tuple] = field(default_factory=list)


def missing_remote_from_catalog(conn, catalog: dict, *, apply: bool, now=None):
    """U-R1: mark ``missing_remote`` every published YouTube row whose id the
    catalog checked by id (``checked_ids``) and did not return. An id the
    catalog never asked about is reported as unchecked and left alone: its
    absence proves nothing. A catalog without ``checked_ids`` is refused."""
    if "checked_ids" not in catalog:
        raise ValueError(
            "live catalog has no checked_ids; re-run `gomoveshift catalog-fetch "
            "--known-ids-from-queue` so absence is provable"
        )
    checked = set(catalog["checked_ids"])
    videos = catalog.get("videos") or {}
    rows = conn.execute(
        "SELECT id, platform_upload_id FROM deliveries WHERE platform='youtube' "
        "AND status='published' AND platform_upload_id IS NOT NULL ORDER BY id"
    ).fetchall()
    plan = MissingRemotePlan()
    for row in rows:
        pair = (row["id"], row["platform_upload_id"])
        if pair[1] in videos:
            continue
        if pair[1] not in checked:
            plan.unchecked.append(pair)
            continue
        evidence = {"source": "catalog_fetch", "checked_by_id": True}
        if not apply or mark_missing_remote(conn, pair[0], evidence=evidence, now=now):
            plan.marked.append(pair)
    return plan


def _cmd_mark_missing_remote(args, config) -> int:
    catalog = json.loads(Path(args.live_catalog).read_text(encoding="utf-8"))
    conn = open_queue_db(config.seo.storage_path)
    try:
        plan = missing_remote_from_catalog(
            conn, catalog, apply=args.apply, now=datetime.now(timezone.utc).isoformat()
        )
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    verb = "marked" if args.apply else "dry run: would mark"
    for delivery_id, video_id in plan.marked:
        print(f"{verb} delivery {delivery_id} missing_remote (YouTube returns no {video_id})")
    for delivery_id, video_id in plan.unchecked:
        print(f"unchecked: delivery {delivery_id} {video_id} not checked by id; left as is")
    if not args.apply:
        print(f"dry run: {len(plan.marked)} row(s); re-run with --apply to write")
    return 0


def _cmd_sync_plan_titles(args, config) -> int:
    """Replay retitle_apply audit logs into the publish-plan sidecars: each
    video's newest APPLIED title becomes its plan title. Local file writes
    only, never a YouTube call; dry run unless ``--apply``."""
    retitles = applied_retitles(Path(a) for a in args.audit)
    conn = open_queue_db(config.seo.storage_path)
    total = refused = 0
    try:
        for video_id, retitle in sorted(retitles.items()):
            # Each video stands alone: one refused title must not stop the rest.
            matched: list = []
            try:
                changes = sync_plan_title(
                    conn, video_id, retitle.title, apply=args.apply, already=matched
                )
            except Exception as exc:
                refused += 1
                print(f"REFUSED {video_id}: {exc}", file=sys.stderr)
                continue
            for delivery_id in matched:
                print(f"already synced {video_id} delivery {delivery_id}: plan title matches")
            for c in changes:
                total += 1
                verb = "synced" if c.written else "dry run: would sync"
                print(
                    f"{verb} delivery {c.delivery_id} {video_id} plan title "
                    f"{c.old!r} -> {c.new!r} ({c.sidecar_path})"
                )
    finally:
        conn.close()
    if refused:
        done = "synced" if args.apply else "would sync"
        print(f"{total} {done}, {refused} refused")
    if not args.apply:
        print(f"dry run: {total} plan title(s); re-run with --apply to write")
    elif not total and not refused:
        print("every plan title already matches its newest applied retitle")
    return 1 if refused else 0


def _cmd_sync_plan_tags(args, config) -> int:
    """Bring each queued YouTube Short's sidecar tags into line with the Short tag rule.
    Local file writes only, never a platform call; dry run unless ``--apply``."""
    from gomoveshift.publishing.plan_title_sync import sync_plan_tags

    conn = open_queue_db(config.seo.storage_path)
    try:
        changes = sync_plan_tags(conn, apply=args.apply)
    finally:
        conn.close()
    verb = "synced" if args.apply else "dry run: would sync"
    for c in changes:
        dropped = [t for t in c.old if t not in c.new]
        print(
            f"{verb} delivery {c.delivery_id} plan tags: drop {dropped!r}, "
            f"keep {list(c.new)!r} ({c.sidecar_path})"
        )
    if not args.apply:
        print(f"dry run: {len(changes)} row(s); re-run with --apply to write")
    elif not changes:
        print("every queued YouTube Short's tags are already reviewable")
    return 0


def cmd_queue(args, config) -> int:
    if args.queue_action not in (
        "recover-terminal",
        "mark-missing-remote",
        "sync-plan-titles",
        "sync-plan-tags",
        "reslot-overdue-manual",
    ):
        print(f"Error: unknown queue action {args.queue_action!r}", file=sys.stderr)
        return 2
    if not Path(config.seo.storage_path).is_absolute():
        print(
            "Error: seo.storage_path is relative; refusing to open a queue that may not be "
            "the real one.",
            file=sys.stderr,
        )
        return 1
    if args.queue_action == "mark-missing-remote":
        return _cmd_mark_missing_remote(args, config)
    if args.queue_action == "sync-plan-titles":
        return _cmd_sync_plan_titles(args, config)
    if args.queue_action == "sync-plan-tags":
        return _cmd_sync_plan_tags(args, config)
    if args.queue_action == "reslot-overdue-manual":
        from .queue_reslot import cmd_reslot_overdue_manual

        return cmd_reslot_overdue_manual(args, config)
    sidecar_dir = Path(args.sidecar_dir) if args.sidecar_dir else Path(config.seo.storage_path)
    conn = open_queue_db(config.seo.storage_path)
    try:
        # Refuse the cheap local gates before any credential is decrypted.
        row = check_recoverable(conn, args.id)
        deps = _build_real_deps(conn, config, sidecar_dir, row["platform"])
        new_id = recover_terminal(
            conn,
            args.id,
            args.authorized_by,
            deps,
            now=datetime.now(timezone.utc).isoformat(),
            apply=args.apply,
        )
    except RecoveryRefusedError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    if new_id is None:
        print(
            f"dry run: delivery {args.id} passes every recovery gate; nothing written. "
            "Re-run with --apply to re-queue it as one new linked delivery."
        )
        return 0
    print(f"recovered delivery {args.id} as new delivery {new_id} (queued, due for the next drain)")
    return 0
