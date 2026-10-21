"""Handler for ``gomoveshift dispatch-queue``.

Drains the social publishing delivery queue: builds the platform handler
registry (``publishing/handlers.build_handlers``) and hands it to
``publishing/dispatch.drain_queue``. This is the command a scheduled drain
(a cron job or timer) invokes.

Dry-run by default, matching ``publish-journal``: without ``--apply`` this
reports what WOULD be drained and calls no platform API. Nothing is
published unless ``--apply`` is passed.

Only platforms actually routed ``first_party`` get a handler. A queued row
for a platform with no handler is left alone and counted as
``skipped_no_handler`` — never failed — because a route flipped back to
``metricool`` must not destroy deliveries that were already spooled.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

from ...config import ConfigError
from ...console import notify
from ...publishing.dispatch import (
    PRIVACY_ATTESTATION_GLOB,
    TransientMediaError,
    drain_queue,
    merge_privacy_attestations,
    reconcile_awaiting_ig,
    reconcile_awaiting_manual,
)
from ...publishing.enqueue_from_plan import backfill_first_comment_text
from ...publishing.handlers import build_handlers, resolve_tiktok_mode
from ...publishing.media_fingerprint import MediaVerificationError, verify_media
from ...publishing.meta_comments import drain_meta_first_comments
from ...publishing.models import ShortPublishRecord
from ...publishing.photo_adapter import PhotoPermanentError
from ...publishing.photo_image_url import PhotoImageUrlError, image_url_for_media_path
from ...publishing.plan_io import PublishPlanFormatError, load_publish_plan
from ...publishing.queue import TERMINAL_STATUSES, open_queue_db
from ...publishing.tiktok_pack import reconcile_packed, resolve_pack_dir, sweep_queued
from ...publishing.tiktok_rendition import ensure_h264_rendition, ensure_instagram_rendition
from ...publishing.youtube_comments import drain_first_comments
from .pub_status import VALIDATE_COMMAND, gating_problems, record_ignore_validate

logger = logging.getLogger(__name__)

#: Bound parameters rather than an interpolated list: the summary counts below
#: must follow ``TERMINAL_STATUSES`` (``publishing/queue.py``) rather than keep
#: their own copy, or a status added there reads as still-pending here.
_TERMINAL_PLACEHOLDERS = ", ".join("?" for _ in TERMINAL_STATUSES)


class SidecarRecordLoader:
    """``(project_name, short_index) -> ShortPublishRecord``, backed by the
    Phase D sidecars.

    The queue row carries only ``(project_name, short_index, platform,
    plan_hash)``, so the full record has to come back from the sidecar that
    produced it. Sidecars are loaded lazily and cached per project: one
    drain pass usually covers many shorts of the same project, and re-reading
    and re-parsing the file per delivery would be pure waste.

    A run's sidecar lives in the run's OWN output dir
    (``phase_d._emit_publish_plan_sidecar``), not in a single flat
    directory, and a queue holding deliveries from two different runs (or
    drives) can't be served by one ``--sidecar-dir``. ``queue.enqueue``
    records each row's own ``sidecar_path``, so when ``conn`` is given this
    loader prefers that recorded path and falls back to the
    ``sidecar_dir``/``<project>_publish_plan.json`` convention only when the
    column is ``NULL`` (a row enqueued before this existed)."""

    def __init__(self, sidecar_dir: Path, conn=None):
        self._dir = Path(sidecar_dir)
        self._conn = conn
        self._cache: Dict[str, Dict[int, ShortPublishRecord]] = {}

    def _resolve_path(self, project_name: str) -> Path:
        if self._conn is not None:
            # Newest enqueue wins. Records are cached per project, so if the
            # same project is re-run and re-enqueued while the earlier run's
            # rows are still non-terminal, both sidecars are live and this
            # picks one for all of them -- the most recent, rather than
            # whichever the query happened to return first. Resolving
            # strictly per delivery would need plan_hash threaded through the
            # loader's (project, short_index) interface; noted rather than
            # done, because the drain is idempotent per row and a stale
            # record surfaces as a plan_hash mismatch, not a wrong upload.
            row = self._conn.execute(
                "SELECT sidecar_path FROM deliveries "
                "WHERE project_name=? AND sidecar_path IS NOT NULL "
                "ORDER BY id DESC LIMIT 1",
                (project_name,),
            ).fetchone()
            if row is not None:
                return Path(row["sidecar_path"])
        return self._dir / f"{project_name}_publish_plan.json"

    def _project(self, project_name: str) -> Dict[int, ShortPublishRecord]:
        if project_name not in self._cache:
            path = self._resolve_path(project_name)
            if not path.exists():
                # Since the sidecar path is recorded per row, the sidecar now
                # lives on the RUN's drive alongside the media and shares its
                # offline exposure. An absent file is a filesystem condition,
                # not a bad delivery, so it must not be counted toward
                # max_attempts any more than missing media is.
                raise TransientMediaError(f"publish plan sidecar not readable: {path}")
            plan = load_publish_plan(path)
            self._cache[project_name] = {s.short_index: s for s in plan.shorts}
        return self._cache[project_name]

    def __call__(self, project_name: str, short_index: int) -> ShortPublishRecord:
        shorts = self._project(project_name)
        try:
            return shorts[short_index]
        except KeyError:
            raise LookupError(
                f"publish plan for project {project_name!r} has no short_index "
                f"{short_index}; the sidecar and the queue disagree"
            ) from None


class PrivacyAttestationLoader:
    """``delivery row -> the privacy attestation covering its media``.

    Linkage: the day-master encoder writes one attestation per DAY beside its
    blur mask in the stage cache; Phase D places the copies beside the run's
    ``*_publish_plan.json`` sidecars — so ``deliveries.sidecar_path``, already
    recorded per row and already the bridge ``SidecarRecordLoader`` uses,
    locates the attestations too. Nothing new has to be threaded through
    enqueue, and a row from a different run reads its OWN run's attestations.
    Rows predating ``sidecar_path`` fall back to ``--sidecar-dir``, exactly as
    the record loader does.

    A run's days are merged worst-state-wins: a delivery names a project, not
    a day, so one unverified day holds the run.
    """

    def __init__(self, sidecar_dir: Path):
        self._dir = Path(sidecar_dir)
        self._cache: Dict[str, Optional[Dict]] = {}

    def __call__(self, row) -> Optional[Dict]:
        sidecar_path = row["sidecar_path"]
        base = Path(sidecar_path).parent if sidecar_path else self._dir
        key = str(base)
        if key not in self._cache:
            docs = []
            for path in sorted(base.glob(PRIVACY_ATTESTATION_GLOB)):
                # A malformed attestation is not a pass: let it raise, and the
                # drain's fail-closed handler turns it into a hold.
                docs.append(json.loads(path.read_text(encoding="utf-8")))
            self._cache[key] = merge_privacy_attestations(docs)
        return self._cache[key]


def _verified_path(record: ShortPublishRecord) -> Path:
    """The record's media path, but only once the file on disk still matches
    the fingerprint taken when the plan was written.

    A delivery fires days to weeks after its plan, by which time the drive can
    be unmounted, the run directory moved, or the short deleted outright. The
    fingerprint is on the record, so a missing/swapped/truncated file is
    detectable — this is where that check is spent, immediately before an
    upload rather than after one.

    Failures raise :class:`TransientMediaError`, so the drain retries WITHOUT
    counting against ``max_attempts``: an unmounted drive is operator-fixable
    and burning the delivery would need a manual re-enqueue.
    """
    try:
        verify_media(record)
    except MediaVerificationError as exc:
        raise TransientMediaError(str(exc)) from exc
    return Path(record.media.path)


def _read_media(record: ShortPublishRecord) -> bytes:
    """Meta adapters upload from memory. The path is on the record already
    (``MediaFingerprint.path``) — no separate media lookup is needed."""
    return _verified_path(record).read_bytes()


def _photo_image_url_loader(site_root: Path):
    """Build ``PhotoCarouselAdapter``'s ``media_image_url_loader``: resolves a bare ``MediaFingerprint`` (a photo/carousel
    record's own ``media`` or one of its ``additional_media`` items) to the
    publicly-reachable ``image_url`` it publishes under — the Meta photo
    container API wants a URL, not raw bytes (see
    ``photo_graph_client.py``'s module docstring). ``site_root`` is bound
    once per call so a second brand's config actually reaches the URL
    deriver instead of falling back to a default site checkout.

    Reuses ``verify_media``'s fingerprint-vs-disk check first, exactly as
    ``_read_media``/the old ``_read_photo_media`` did — an unmounted drive
    still retries via ``TransientMediaError`` without burning
    ``max_attempts``, before any URL is derived.

    A path that isn't under the site's served
    ``static/images/journeys/<slug>/<filename>`` layout (e.g. a resized
    social copy) is a distinct, permanent failure — never a guess — raised
    as ``PhotoPermanentError`` so the handler translates it straight to
    ``PermanentDeliveryError`` instead of leaving it to retry forever."""

    def loader(fingerprint) -> str:
        try:
            verify_media(SimpleNamespace(media=fingerprint))
        except MediaVerificationError as exc:
            raise TransientMediaError(str(exc)) from exc
        try:
            return image_url_for_media_path(fingerprint.path, site_root=site_root)
        except PhotoImageUrlError as exc:
            raise PhotoPermanentError(str(exc)) from exc

    return loader


def _tiktok_rendition_path(record: ShortPublishRecord) -> Path:
    """The file TikTok gets: an H.264 rendition, never the HEVC master -- see
    ``publishing/tiktok_rendition.py`` for the live evidence. The master is
    fingerprint-verified FIRST (the rendition is derived from it and has no
    fingerprint of its own), then transcoded if it is not already H.264. The
    TikTok Studio pack copies this path; the legacy upload modes read it.
    """
    source = _verified_path(record)
    return ensure_h264_rendition(source, codec=record.media.codec)


def _read_tiktok_media(record: ShortPublishRecord) -> bytes:
    return _tiktok_rendition_path(record).read_bytes()


def _read_instagram_media(record: ShortPublishRecord) -> bytes:
    """Instagram caps Reels at 25 Mbps and stalls on over-cap masters — see
    ``publishing/tiktok_rendition.py``. Master verified FIRST, then a
    rendition is built only if the master is over the cap."""
    source = _verified_path(record)
    # The fingerprint's size and duration were just checked against the file,
    # so their ratio IS the container's average bitrate -- no ffprobe needed.
    media = record.media
    bitrate = (
        int(media.size_bytes * 8 / media.duration_seconds) if media.duration_seconds > 0 else None
    )
    return ensure_instagram_rendition(source, bitrate_bps=bitrate).read_bytes()


def _media_path(record: ShortPublishRecord) -> Path:
    return _verified_path(record)


def _routed(config, platform: str) -> bool:
    return getattr(config.drive, f"publishing_route_{platform}", "off") == "first_party"


def _validate_held_scopes(conn, config, ignore: bool, reason: str) -> set:
    """U-R2: the (platform, project) pairs this drain must hold because
    ``publishing-status --validate`` names a problem on them. Computed after
    the reconcilers so a row they just settled no longer holds its pair.
    ``--ignore-validate --reason`` holds nothing and audits the override."""
    credentials_dir = getattr(getattr(config, "seo", None), "credentials_dir", None)
    problems = gating_problems(
        conn,
        credentials_dir,
        storage_path=getattr(getattr(config, "seo", None), "storage_path", None),
    )
    if not problems:
        return set()
    if ignore:
        record_ignore_validate(conn, problems, reason=reason, command="dispatch-queue")
        print(f"warning: --ignore-validate: {len(problems)} validate problem(s) overridden")
        return set()
    print("validate gate: holding these platform/project pairs (other pairs drain normally):")
    for problem in problems:
        print(problem.line())
    print(f"See: {VALIDATE_COMMAND}. Override only with --ignore-validate --reason '<why>'.")
    return {(p.platform, p.project) for p in problems}


def cmd_dispatch_queue(args, config) -> int:
    # A relative storage_path resolves against the CALLER's cwd, and
    # open_queue_db creates what it does not find -- so a drain started from
    # the wrong directory silently opened an EMPTY queue, reported
    # "non-terminal deliveries: none" and exited 0. Refuse before touching
    # any database: publishing nothing must never look like nothing to do.
    if not Path(config.seo.storage_path).is_absolute():
        print(
            f"Error: seo.storage_path is relative ({config.seo.storage_path!r}), so the queue "
            "would resolve against this process's working directory and could be a different, "
            "empty database than the one enqueue-plan wrote to. Set an absolute path in "
            "config.yaml, or run from the service directory where its config.yaml is found."
        )
        return 1

    ignore_validate = getattr(args, "ignore_validate", False)
    ignore_reason = (getattr(args, "reason", None) or "").strip()
    if ignore_validate and not ignore_reason:
        print("Error: --ignore-validate needs --reason '<why>' (written to the audit log).")
        return 1

    sidecar_dir = Path(args.sidecar_dir) if args.sidecar_dir else Path(config.seo.storage_path)
    conn = open_queue_db(config.seo.storage_path)
    try:
        pending = conn.execute(
            "SELECT platform, COUNT(*) AS n FROM deliveries "  # noqa: S608
            f"WHERE status NOT IN ({_TERMINAL_PLACEHOLDERS}) AND shadow = 0 "
            "GROUP BY platform",
            TERMINAL_STATUSES,
        ).fetchall()
        shadow_count = conn.execute(
            "SELECT COUNT(*) AS n FROM deliveries "  # noqa: S608
            f"WHERE status NOT IN ({_TERMINAL_PLACEHOLDERS}) AND shadow = 1",
            TERMINAL_STATUSES,
        ).fetchone()["n"]
        routed = [p for p in ("youtube", "instagram", "facebook", "tiktok") if _routed(config, p)]
        now = datetime.now(timezone.utc).isoformat()
        not_due = conn.execute(
            "SELECT COUNT(*) AS n, MIN(scheduled_at) AS next FROM deliveries "  # noqa: S608
            f"WHERE status NOT IN ({_TERMINAL_PLACEHOLDERS}) AND shadow = 0 "
            "AND scheduled_at IS NOT NULL AND scheduled_at > ?",
            (*TERMINAL_STATUSES, now),
        ).fetchone()

        print(f"queue: {config.seo.storage_path}/social_queue.sqlite3")
        summary = ", ".join(f"{r['platform']}={r['n']}" for r in pending) or "none"
        print(f"non-terminal deliveries: {summary}")
        if not_due["n"]:
            # Otherwise a correctly-holding queue reads as a stuck one: the
            # rows are non-terminal but their publish slot has not arrived.
            print(f"not yet due (waiting for their slot): {not_due['n']}, next {not_due['next']}")
        print(f"platforms routed first_party: {routed or 'none'}")
        if shadow_count:
            # Say so explicitly: a shadow row sits non-terminal forever, and
            # without this line it reads as a stuck delivery.
            print(f"shadow (never published): {shadow_count}")

        if not args.apply:
            print("\nDRY RUN - no platform API was called and nothing was published.")
            print("Re-run with --apply to drain.")
            return 0

        if not routed:
            # Not an error: every route still on metricool means the CSV
            # path is live and there is legitimately nothing to drain.
            print("\nNo platform is routed first_party; nothing to drain.")
            return 0

        # BEFORE the drain: a draft a human published in the app parks at
        # awaiting_manual_publish with next_attempt_at=9999, so the drain
        # cannot see it and the row advertises an outstanding chore that is
        # actually a live post. Never fatal -- a TikTok blip here must not
        # stop today's shorts.
        try:
            reconciler = build_tiktok_reconciler(config, routed)
        except Exception as exc:  # noqa: BLE001
            reconciler = None
            print(f"reconcile: skipped ({exc.__class__.__name__})")
        if reconciler is not None:
            poll, permalink_for = reconciler
            rec = reconcile_awaiting_manual(conn, poll=poll, permalink_for=permalink_for)
            if rec.polled:
                print(
                    f"reconciled: polled={rec.polled} published={rec.published} failed={rec.failed}"
                )
            for err in rec.errors:
                print(f"  error: {err}")

        # Rows enqueued before first comments existed carry no text; fill it
        # from their sidecars once per run, before the comment drains below.
        if any(_routed(config, p) for p in ("youtube", "instagram", "facebook")):
            bf = backfill_first_comment_text(conn)
            if bf.filled or bf.skipped:
                print(f"first_comment_backfill: filled={bf.filled} skipped={bf.skipped}")

        loader = SidecarRecordLoader(sidecar_dir, conn=conn)
        # U-R5(b): Instagram rows parked on "Media ID is not available" are
        # decided by an account scan BEFORE the drain reads the queue, so a
        # proven-absent row re-queued here is drained this same pass. Never a
        # blind retry; never fatal -- a Graph blip must not stop the shorts.
        if "instagram" in routed:
            try:
                recent_media = build_ig_media_scanner(config)
            except Exception as exc:  # noqa: BLE001
                recent_media = None
                print(f"ig reconcile: skipped ({exc.__class__.__name__})")
            if recent_media is not None:
                ig = reconcile_awaiting_ig(
                    conn,
                    recent_media=recent_media,
                    record_loader=loader,
                    now=datetime.now(timezone.utc).isoformat(),
                )
                if ig.considered or ig.errors:
                    print(
                        f"ig reconcile: considered={ig.considered} published={ig.published} "
                        f"retried={ig.retried} failed={ig.failed} unprovable={ig.unprovable}"
                    )
                for err in ig.errors:
                    print(f"  error: {err}")
        # ONE identity for the pass. drain_queue takes the lease and the
        # adapters re-take it before their irreversible call; with two
        # different ids that second acquire can never succeed and every
        # delivery dies as "leased by another worker".
        worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        # ONE attestation loader for the pass: the pack sweep and the drain
        # hold a short back for the same reason, from the same files.
        privacy = PrivacyAttestationLoader(sidecar_dir)
        # In pack mode TikTok's rows are written to the Studio pack
        # folder for the owner to schedule, BEFORE the drain reads the queue.
        # Never fatal -- a missing drive must not stop today's other shorts.
        # The counts ride on the summary line (the only line the cron keeps).
        pack_counts = {"packed": 0, "errors": 0, "already": 0, "held": 0}
        pack_reconcile_errors = 0
        pack_reconcile_ambiguous = 0
        tiktok_mode = None
        tiktok_mode_invalid = False
        pack_mode = False
        if "tiktok" in routed:
            try:
                tiktok_mode = resolve_tiktok_mode()
                pack_mode = tiktok_mode == "pack"
            except ValueError as exc:
                # A bad mode takes TikTok out of this pass, nothing else: the
                # other platforms still drain and the summary says so.
                tiktok_mode_invalid = True
                print(f"tiktok: skipped for this pass ({exc})")
                routed = [p for p in routed if p != "tiktok"]
        if pack_mode:
            # Close the rows the owner already posted from Studio, from
            # TikTok's read-only video list. Best effort: any failure here
            # (credentials, HTTP) costs the reconcile and nothing else.
            try:
                list_page, pack_permalink_for = build_pack_reconciler(config)
                packed_rec = reconcile_packed(
                    conn,
                    list_videos=list_page,
                    permalink_for=pack_permalink_for,
                    record_loader=loader,
                    now=datetime.now(timezone.utc),
                )
            except Exception as exc:  # noqa: BLE001
                pack_reconcile_errors = 1
                print(f"  error: tiktok reconcile: skipped ({exc.__class__.__name__}: {exc})")
            else:
                pack_reconcile_errors = len(packed_rec.errors)
                pack_reconcile_ambiguous = packed_rec.ambiguous
                if packed_rec.considered or packed_rec.errors:
                    print(
                        f"tiktok reconcile: considered={packed_rec.considered} "
                        f"published={packed_rec.published} ambiguous={packed_rec.ambiguous} "
                        f"errors={len(packed_rec.errors)}"
                    )
                for line in (*packed_rec.ambiguous_rows, *packed_rec.errors):
                    print(f"  error: {line}")
            try:
                swept = sweep_queued(
                    conn,
                    loader,
                    _tiktok_rendition_path,
                    worker_id=worker_id,
                    pack_dir=resolve_pack_dir(),
                    privacy_attestations=privacy,
                )
            except Exception as exc:  # noqa: BLE001
                pack_counts["errors"] = 1
                print(f"tiktok pack: failed ({exc.__class__.__name__}: {exc})")
            else:
                pack_counts.update(
                    packed=swept.packed,
                    errors=len(swept.errors),
                    already=swept.already_packed,
                    held=swept.held_privacy,
                )
                if swept.paused:
                    print("tiktok pack: skipped (tiktok is paused)")
                elif swept.considered or swept.errors:
                    print(
                        f"tiktok pack: packed={swept.packed} "
                        f"already_packed={swept.already_packed} held={swept.held_privacy} "
                        f"leased_elsewhere={swept.skipped_leased} errors={len(swept.errors)}"
                        f" dir={resolve_pack_dir()}"
                    )
                for err in (*swept.holds, *swept.errors):
                    print(f"  error: {err}")

        handlers = _build_routed_handlers(conn, loader, config, routed, worker_id=worker_id)
        held_scopes = _validate_held_scopes(conn, config, ignore_validate, ignore_reason)
        result = drain_queue(
            conn,
            handlers,
            owner=worker_id,
            privacy_attestations=privacy,
            held_scopes=held_scopes,
        )

        # (who, result) for each first-comment drain that ran: added up onto
        # the summary line, the only line the cron log keeps (see below).
        comment_results = []

        # A YouTube first comment, scheduled 24h after publish. Only
        # when YouTube is actually routed first_party -- ``service_factory``
        # is lazy (built at most once, only if a row is due), so a run with
        # no other platform routed to YouTube never pays the SOPS-decrypt
        # cost for a client it would never use.
        if "youtube" in routed:
            fc_result = drain_first_comments(
                conn, service_factory=lambda: _build_youtube_comments_service(config)
            )
            if fc_result.posted or fc_result.failed or fc_result.needs_reauth:
                print(
                    f"first_comment: posted={fc_result.posted} failed={fc_result.failed}"
                    + (" needs_reauth=1" if fc_result.needs_reauth else "")
                )
                for err in fc_result.errors:
                    print(f"  error: {err}")
            comment_results.append(("youtube", fc_result))

        # Instagram / Facebook first comments: same lazy-factory rule, each
        # platform only when routed first_party.
        meta_factories = _meta_comment_factories(config, routed)
        if meta_factories:
            mc_result = drain_meta_first_comments(conn, adapter_factories=meta_factories)
            if mc_result.posted or mc_result.failed or mc_result.needs_reauth:
                print(
                    f"first_comment: posted={mc_result.posted} failed={mc_result.failed}"
                    + (" needs_reauth=1" if mc_result.needs_reauth else "")
                )
                for err in mc_result.errors:
                    print(f"  error: {err}")
            comment_results.append(("meta", mc_result))

        # This is the ONLY drain path that actually runs in production
        # (a scheduled drain invokes this command hourly) -- `check_missed_slots`
        # previously fired only from the console's in-process Ticker, which
        # nothing starts here, so a missed slot was never notified. An
        # observer must never fail the drain it rides along with: a broken
        # webhook or a queue read error here must not turn a healthy publish
        # pass into a failed one.
        try:
            notify.check_missed_slots(config=config)
        except Exception:  # noqa: BLE001 -- an observer must never fail the drain
            logger.exception("dispatch-queue: missed-slot observer failed; continuing")

        # Phone push for a row this very pass parked (a TikTok inbox draft) or failed:
        # AFTER the drain so it is announced in the pass that made it. Same rule as
        # above -- an observer, never a reason to fail the run or move its summary line.
        try:
            notify.check_queue_regressions(config=config)
        except Exception:  # noqa: BLE001 -- an observer must never fail the drain
            logger.exception("dispatch-queue: queue-regression observer failed; continuing")

        # ON the summary line, not after it: the scheduled drain logs `tail -1`
        # of this output, so a separate line would displace the summary from
        # the log and a line printed earlier would never reach it. A delivery
        # in `awaiting_manual_publish` is one no drain can ever advance --
        # TikTok's inbox route parks it until a human taps Post in the app --
        # so a pass that reports only what it attempted is silent about every
        # short sitting unpublished.
        awaiting = conn.execute(
            "SELECT COUNT(*) FROM deliveries WHERE status='awaiting_manual_publish'"
        ).fetchone()[0]
        # Per-delivery errors print ABOVE the summary: the scheduled drain logs
        # `tail -1`, so anything after the summary line displaces it.
        for err in result.errors:
            print(f"  error: {err}")
        fc_posted = sum(r.posted for _, r in comment_results)
        fc_failed = sum(r.failed for _, r in comment_results)
        fc_reauth = ",".join(who for who, r in comment_results if r.needs_reauth)
        print(
            f"\nattempted={result.attempted} succeeded={result.succeeded} "
            f"retried={result.retried} failed_permanent={result.failed_permanent} "
            f"skipped_leased={result.skipped_leased} "
            f"skipped_no_handler={result.skipped_no_handler} "
            f"awaiting_manual_publish={awaiting}"
            # first_party names the adapter route, not proof of an API
            # publish. Keep the actual mode and reconciliation uncertainty
            # on the line cron retains; parked rows alone prove no outcome.
            + (f" tiktok_mode={tiktok_mode}" if tiktok_mode else "")
            + (f" tiktok_reconcile_errors={pack_reconcile_errors}" if pack_reconcile_errors else "")
            + (
                f" tiktok_reconcile_ambiguous={pack_reconcile_ambiguous}"
                if pack_reconcile_ambiguous
                else ""
            )
            # Only when non-zero: the cron log is read by eye and a
            # permanent `throttled=0` trains everyone to skip the field.
            + (f" throttled={result.throttled}" if result.throttled else "")
            # Same rule, and this one matters more: a non-zero value means a
            # worker died mid-attempt since the last pass. Printing
            # `reconciled_interrupted=0` on every healthy run is how that
            # stops being read at all.
            + (
                f" reconciled_interrupted={result.reconciled_interrupted}"
                if result.reconciled_interrupted
                else ""
            )
            # The first-comment lines above never reach the cron log: a pass
            # that posted comments read the same there as a pass with none
            # due. Both drains added up; only when non-zero, like the rest.
            + (f" first_comment_posted={fc_posted}" if fc_posted else "")
            + (f" first_comment_failed={fc_failed}" if fc_failed else "")
            + (f" first_comment_needs_reauth={fc_reauth}" if fc_reauth else "")
            # TikTok pack sweep: its own lines never reach the cron log, so a
            # drive that is offline would read like a healthy idle hour.
            + (f" tiktok_packed={pack_counts['packed']}" if pack_counts["packed"] else "")
            + (f" tiktok_pack_errors={pack_counts['errors']}" if pack_counts["errors"] else "")
            + (f" tiktok_already_packed={pack_counts['already']}" if pack_counts["already"] else "")
            + (f" tiktok_pack_held={pack_counts['held']}" if pack_counts["held"] else "")
            + (" tiktok_mode_invalid=1" if tiktok_mode_invalid else "")
            # Same line, not a separate one -- see the comment above. A pass
            # that skipped due work for a missing handler must not read like
            # healthy inactivity just because attempted/succeeded/
            # failed_permanent are all zero.
            + (" Blocked: publishing route unavailable" if result.blocked_no_route else "")
            # Same line, same rule, and the loudest of the three: a held pass
            # means media whose required privacy classes were never verified
            # was refused at the irreversible step. Printed only when
            # non-zero so it never becomes a field nobody reads.
            + (f" held_validate={result.held_validate}" if result.held_validate else "")
            + (
                f" held_privacy_unverified={result.held_privacy_unverified}"
                " Held: privacy classes unverified"
                if result.held_for_privacy_review
                else ""
            )
        )
        # A drain pass that recorded per-delivery errors is still a
        # successful pass: drain_queue never lets one bad delivery abort the
        # rest, and the bad ones are retried or already terminal.
        return 0
    except PublishPlanFormatError as exc:
        logger.error("dispatch-queue: %s", exc)
        return 1
    finally:
        conn.close()


def build_pack_reconciler(config):
    """``(list_videos(cursor), permalink_for)`` for ``reconcile_packed``.

    Pack mode runs the drain with an empty TikTok token on purpose; only this
    best-effort reconcile fetches real credentials, and raises when they are
    unusable so the caller can skip it."""
    from ...publishing.tiktok_adapter import list_videos, query_creator_info
    from ...publishing.tiktok_auth import get_valid_tiktok_credentials

    token = get_valid_tiktok_credentials(Path(config.seo.credentials_dir))["access_token"]
    username = query_creator_info(token).get("creator_username") or ""

    def list_page(cursor=None):
        return list_videos(token, cursor=cursor)

    def permalink_for(post_id: str) -> str:
        # Canonical URL from the handle, never the utm-tagged share_url.
        return f"https://www.tiktok.com/@{username}/video/{post_id}"

    return list_page, permalink_for


def build_tiktok_reconciler(config, routed):
    """``(poll, permalink_for)`` for ``reconcile_awaiting_manual``, or None
    when TikTok is not routed first_party.

    Kept out of ``publishing/dispatch.py`` deliberately: that module knows
    about the queue and nothing about any platform's API, and this is where
    the TikTok credentials are already being decrypted.
    """
    if "tiktok" not in routed:
        return None

    from ...publishing.tiktok_adapter import (
        TikTokPollTimeoutError,
        poll_publish_status,
        query_creator_info,
    )
    from ...publishing.tiktok_auth import get_valid_tiktok_credentials

    token = get_valid_tiktok_credentials(Path(config.seo.credentials_dir))["access_token"]
    # One lookup for the whole pass; the handle is the same for every row.
    username = query_creator_info(token).get("creator_username") or ""

    def poll(publish_id: str):
        # max_attempts=1: the shipped default is 10 with backoff, ~3 minutes
        # of sleeping, and it RAISES when nothing terminal arrives -- which is
        # the normal answer for a draft still sitting in the inbox. One
        # request, and a timeout means "still parked", not an error.
        try:
            return poll_publish_status(token, publish_id, max_attempts=1)
        except TikTokPollTimeoutError:
            return {}

    def permalink_for(post_id: str) -> str:
        # The canonical URL, NOT the share_url `video/list` hands back: that
        # one carries utm_campaign=tt4d_open_api&utm_source=<client key>, and
        # the client key does not belong in a user-visible column.
        return f"https://www.tiktok.com/@{username}/video/{post_id}"

    return poll, permalink_for


def _build_youtube_comments_service(config):
    """An authenticated YouTube Data API v3 client for
    ``youtube_comments.drain_first_comments``. Same credential path as
    ``cli.handlers.comments.run_comments_pull`` -- the upload adapter's
    OAuth token, not a separate one."""
    from googleapiclient.discovery import build

    from ...pipeline._yt_creds import load_youtube_credentials
    from ...pipeline.drive_phases.youtube_uploader import _build_credentials_obj

    creds = _build_credentials_obj(load_youtube_credentials(Path(config.seo.credentials_dir)))
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def build_ig_media_scanner(config):
    """``recent_media(limit)`` for ``reconcile_awaiting_ig``: the Instagram
    adapter's account media listing. Decrypts the Meta credentials, so the
    caller builds it only when Instagram is routed first_party."""
    from ...publishing.instagram_adapter import InstagramReelsAdapter
    from ...publishing.meta_auth import load_meta_credentials

    adapter = InstagramReelsAdapter(load_meta_credentials(Path(config.seo.credentials_dir)))
    return adapter.list_recent_media


def _meta_comment_factories(config, routed):
    """platform -> lazy zero-arg factory building the Graph adapter that posts
    that platform's first comments. Only routed platforms; credentials load
    (SOPS decrypt) only when a comment is actually due."""

    def _meta(config):
        from ...publishing.meta_auth import load_meta_credentials

        return load_meta_credentials(Path(config.seo.credentials_dir))

    factories = {}
    if "instagram" in routed:
        from ...publishing.instagram_adapter import InstagramReelsAdapter

        factories["instagram"] = lambda: InstagramReelsAdapter(_meta(config))
    if "facebook" in routed:
        from ...publishing.facebook_adapter import FacebookReelsAdapter

        factories["facebook"] = lambda: FacebookReelsAdapter(_meta(config))
    return factories


def _build_routed_handlers(conn, loader, config, routed, *, worker_id="adapter"):
    """Construct adapters only for routed platforms — credentials are
    decrypted through SOPS, so an unrouted platform must not pay that cost
    or fail the whole drain when its credentials are absent."""
    from ...publishing.meta_auth import load_meta_credentials

    youtube: Optional[Tuple[object, object]] = None
    instagram: Optional[Tuple[object, object]] = None
    facebook: Optional[Tuple[object, object]] = None
    tiktok: Optional[Tuple[object, object]] = None
    instagram_photo: Optional[Tuple[object, object]] = None
    facebook_photo: Optional[Tuple[object, object]] = None

    if "youtube" in routed:
        from ...pipeline.drive_phases.youtube_uploader import YouTubeUploader
        from ...publishing.youtube_adapter import YouTubeShortsAdapter

        youtube = (
            YouTubeShortsAdapter(YouTubeUploader(), config, worker_id=worker_id),
            _media_path,
        )

    if "instagram" in routed or "facebook" in routed:
        meta = load_meta_credentials(Path(config.seo.credentials_dir))
        # The config's own site_root, not a module-level fallback — so a
        # second brand's config actually reaches the image_url deriver and
        # verifier. With no ``photos.site_root`` configured there is no site
        # to derive public image URLs from: photo/carousel rows then fall to
        # ``skipped_no_handler`` while the video routes work unchanged.
        try:
            site_root = config.photos.resolve_site_root()
        except ConfigError:
            photos_ready = False
        else:
            photos_ready = True
            photo_loader = _photo_image_url_loader(site_root)
        if "instagram" in routed:
            from ...publishing.instagram_adapter import (
                InstagramReelsAdapter,
                InstagramShortsAdapter,
            )
            from ...publishing.photo_adapter import PhotoCarouselAdapter
            from ...publishing.photo_graph_client import MetaPhotoGraphClient

            # The *ShortsAdapter wraps the *ReelsAdapter (the queue-aware
            # layer over the raw Graph client); it does NOT take credentials.
            instagram = (
                InstagramShortsAdapter(InstagramReelsAdapter(meta), worker_id=worker_id),
                _read_instagram_media,
            )
            # A routed platform gets a photo/carousel handler
            # too, so a format=photo/carousel row isn't left with
            # skipped_no_handler just because a video route exists.
            if photos_ready:
                instagram_photo = (
                    PhotoCarouselAdapter(
                        MetaPhotoGraphClient(meta, site_root=site_root),
                        "instagram",
                        worker_id=worker_id,
                    ),
                    photo_loader,
                )
        if "facebook" in routed:
            from ...publishing.facebook_adapter import (
                FacebookReelsAdapter,
                FacebookShortsAdapter,
            )
            from ...publishing.photo_adapter import PhotoCarouselAdapter
            from ...publishing.photo_graph_client import MetaPhotoGraphClient

            facebook = (
                FacebookShortsAdapter(FacebookReelsAdapter(meta), worker_id=worker_id),
                _read_media,
            )
            if photos_ready:
                facebook_photo = (
                    PhotoCarouselAdapter(
                        MetaPhotoGraphClient(meta, site_root=site_root),
                        "facebook",
                        worker_id=worker_id,
                    ),
                    photo_loader,
                )

    if "tiktok" in routed:
        from ...publishing.handlers import TikTokCredentials
        from ...publishing.tiktok_auth import get_valid_tiktok_credentials

        # Pack mode never calls TikTok: a dead TikTok token must not stop the
        # pass that drains every other platform.
        if resolve_tiktok_mode() == "pack":
            creds = {"access_token": ""}
        else:
            creds = get_valid_tiktok_credentials(Path(config.seo.credentials_dir))
        tiktok = (
            TikTokCredentials(
                access_token=creds["access_token"],
                audit_approved=bool(
                    getattr(config.drive, "publishing_tiktok_audit_approved", False)
                ),
            ),
            _read_tiktok_media,
        )

    return build_handlers(
        conn,
        loader,
        worker_id=worker_id,
        youtube=youtube,
        instagram=instagram,
        facebook=facebook,
        tiktok=tiktok,
        tiktok_rendition_path_loader=_tiktok_rendition_path,
        instagram_photo=instagram_photo,
        facebook_photo=facebook_photo,
    )
