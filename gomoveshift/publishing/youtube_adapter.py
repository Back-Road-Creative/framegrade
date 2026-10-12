"""YouTube Shorts publishing adapter. See ``docs/social-publishing-youtube.md``.
Orchestrates ``YouTubeUploader`` only — never re-implements its upload/retry/token/quota handling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..pipeline.drive_phases.youtube_uploader import YouTubeUploader
from ..pipeline.exceptions import YouTubeQuotaError
from ..pipeline.playlist_resolver import resolve_playlist, short_region
from .models import ShortPublishRecord, YouTubePayload
from .queue import NEXT_ATTEMPT_PLATFORM_DEFERRAL as PLATFORM_DEFERRAL
from .queue import PUBLISHED_AT_PLATFORM_CONFIRMED as PLATFORM_CONFIRMED
from .queue import PUBLISHED_AT_PLATFORM_SCHEDULED as PLATFORM_SCHEDULED
from .queue import acquire_lease, find_delivery, transition
from .youtube_uploads_listing import list_uploads

logger = logging.getLogger(__name__)

PLATFORM = "youtube"
STATUS_QUEUED, STATUS_SUBMITTING, STATUS_UPLOADED = "queued", "submitting", "uploaded"
STATUS_PUBLISHED, STATUS_DEFERRED = "published", "deferred"
STATUS_RETRY_WAIT = "retry_wait"

# Ceiling on a single insert attempt's worker lease — generous for a
# resumable upload; a worker that dies mid-attempt frees this window for
# another worker (or this one, on retry) to reclaim.
LEASE_SEC = 900

# Bound on the channel-scan fallback (see ``_find_existing_upload``): only
# uploads within this window of "now" are candidates, and at most this many
# of the channel's most recent uploads are inspected.
_ADOPT_RECENCY = timedelta(hours=2)
_ADOPT_SCAN_LIMIT = 100  # explicit bound passed to the shared lister


class AmbiguousUploadMatchError(RuntimeError):
    """Two or more recent channel uploads match a retrying short by title
    and recency — adopting either risks stealing the wrong video. A human
    must resolve this; a duplicate is a lesser evil than a wrong adoption."""


@dataclass(frozen=True, slots=True)
class ShortsPublishResult:
    video_id: str
    playlist_id: Optional[str]
    deferred: bool = False


def _utc_midnight(*, plus_days: int = 0) -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=plus_days)


def _uploads_today(conn, platform: str = PLATFORM) -> int:
    # Read from the queue, not an in-memory counter, so this survives a restart.
    row = conn.execute(
        "SELECT COUNT(*) FROM deliveries WHERE platform=? "
        "AND platform_upload_id IS NOT NULL AND created_at >= ?",
        (platform, _utc_midnight().isoformat()),
    ).fetchone()
    return int(row[0]) if row is not None else 0


class YouTubeShortsAdapter:
    """Publish a single ``ShortPublishRecord`` short to YouTube."""

    def __init__(self, uploader: YouTubeUploader, config: Any, worker_id: str = "adapter"):
        self._uploader = uploader
        self._config = config
        self._worker_id = worker_id

    def publish(
        self, record: ShortPublishRecord, *, conn, delivery_id: int, video_path: Path
    ) -> ShortsPublishResult:
        payload = record.payloads.youtube
        if payload is None:
            raise ValueError(f"short_index={record.short_index} carries no YouTube payload")

        existing = find_delivery(conn, record.project_name, record.short_index, PLATFORM)
        current_status = existing["status"] if existing else STATUS_QUEUED
        video_id = (
            self._reconcile(existing, conn=conn, delivery_id=delivery_id, payload=payload)
            if existing
            else None
        )

        if not video_id:
            deferred = self._defer_if_over_budget(conn, delivery_id, record, current_status)
            if deferred is not None:
                return deferred
            lease_expires = (datetime.now(timezone.utc) + timedelta(seconds=LEASE_SEC)).isoformat()
            if not acquire_lease(conn, delivery_id, self._worker_id, lease_expires):
                raise RuntimeError(
                    f"delivery {delivery_id} is leased by another worker — refusing to "
                    "risk a duplicate videos.insert"
                )
            transition(conn, delivery_id, current_status, STATUS_SUBMITTING)
            try:
                video_id = self._uploader.upload_video(
                    video_path,
                    title=payload.title,
                    description=payload.description,
                    tags=list(payload.tags),
                    category_id=payload.category or "2",
                    publish_at=record.schedule.scheduled_at_utc,
                    privacy="private",
                )
            except YouTubeQuotaError:
                # Retryable, not a failure — the daily quota window resets.
                transition(
                    conn, delivery_id, STATUS_SUBMITTING, STATUS_RETRY_WAIT, error_class="quota"
                )
                raise
            # Any other exception (timeout, lost response, network) leaves the
            # delivery in SUBMITTING — an unknown outcome, never a failure.
            # The next attempt reconciles via ``_reconcile`` before ever
            # retrying ``videos.insert`` (see docs).

            # Persist BEFORE the playlist call —
            # the upload above is irreversible, this is not.
            transition(
                conn, delivery_id, STATUS_SUBMITTING, STATUS_UPLOADED, platform_upload_id=video_id
            )

        try:
            playlist_id = self._assign_playlist(video_id, record)
        except Exception:
            logger.exception(
                "playlist assign failed for video_id=%s — kept; resume retries it", video_id
            )
            return ShortsPublishResult(video_id=video_id, playlist_id=None)

        if playlist_id is None:
            # Country not derivable: no playlist insert was made (see
            # ``_assign_playlist``). Fall through to ``published`` with no
            # ``platform_post_id`` — parking this in ``uploaded`` would make
            # every resume retry the same unresolvable assignment forever.
            logger.warning(
                "short video_id=%s published WITHOUT a playlist: no country derivable from "
                "sampled_regions=%r — left unfiled for the nightly reconciler",
                video_id,
                record.sampled_regions,
            )

        # The upload above set ``status.publishAt`` and ``privacy=private``,
        # so the call returning success means YouTube ACCEPTED this video and
        # will make it public at the scheduled instant -- usually hours from
        # now. Calling that ``platform_confirmed`` ("within seconds of the
        # real publish time") overstated the evidence by the whole length of
        # the schedule. Record the instant YouTube was actually told, taken
        # from the same value handed to ``upload_video`` rather than
        # re-derived, so the two can never disagree.
        scheduled_at = record.schedule.scheduled_at_utc
        fields = {
            "platform_upload_id": video_id,
            "platform_post_id": playlist_id,
            "published_at_source": PLATFORM_SCHEDULED if scheduled_at else PLATFORM_CONFIRMED,
        }
        if scheduled_at:
            fields["platform_published_at"] = scheduled_at.isoformat()
        transition(conn, delivery_id, STATUS_UPLOADED, STATUS_PUBLISHED, **fields)
        return ShortsPublishResult(video_id=video_id, playlist_id=playlist_id)

    def _defer_if_over_budget(
        self, conn, delivery_id: int, record: ShortPublishRecord, current_status: str
    ) -> Optional[ShortsPublishResult]:
        budget = getattr(self._config.drive, "youtube_video_insert_daily_uploads", None)
        if not isinstance(budget, int) or _uploads_today(conn) < budget:
            return None
        next_attempt = _utc_midnight(plus_days=1).isoformat()
        transition(
            conn,
            delivery_id,
            current_status,
            STATUS_DEFERRED,
            next_attempt_at=next_attempt,
            next_attempt_reason=PLATFORM_DEFERRAL,
        )
        logger.info("upload budget (%d) reached — deferring to %s", budget, next_attempt)
        return ShortsPublishResult(video_id="", playlist_id=None, deferred=True)

    def _reconcile(
        self, existing: Dict[str, Any], *, conn, delivery_id: int, payload: YouTubePayload
    ) -> Optional[str]:
        """A ``videos.insert`` timeout is an unknown outcome, never a failure.

        Before any new create call, check whether a video_id was already
        persisted for this delivery and confirm with ``videos.list`` that it
        is real. Returns the id to adopt, or ``None`` when nothing was
        created (a fresh insert is permitted).

        ``videos.insert`` has no create-idempotency key: if the insert
        succeeded on YouTube's side but its response was lost before
        anything was persisted, this delivery is stuck in ``submitting``
        with no id to confirm above. In that case — and ONLY in that case,
        never on a first, clean attempt (``queued``, nothing ever
        submitted) — fall back to scanning the channel's recent uploads for
        a video that matches this short and adopt it instead of inserting a
        duplicate."""
        video_id = existing.get("platform_upload_id")
        if video_id:
            service = self._uploader._build_service()
            resp = service.videos().list(part="status,snippet", id=video_id).execute()
            return video_id if resp.get("items") else None

        if existing.get("status") != STATUS_SUBMITTING:
            return None

        adopted = self._find_existing_upload(payload)
        if adopted:
            transition(
                conn, delivery_id, STATUS_SUBMITTING, STATUS_UPLOADED, platform_upload_id=adopted
            )
        return adopted

    def _find_existing_upload(self, payload: YouTubePayload) -> Optional[str]:
        """Scan the channel's most recent uploads (pattern: see
        ``youtube_uploads_listing.list_uploads``) for a video matching
        ``payload`` by exact title and upload recency. Read-only; costs
        ``playlistItems.list``/``videos.list`` units (the combined
        10,000/day bucket), never ``videos.insert`` quota. Never called
        outside the lost-response retry path (see ``_reconcile``)."""
        service = self._uploader._build_service()
        cutoff = datetime.now(timezone.utc) - _ADOPT_RECENCY
        candidates: List[str] = []
        for upload in list_uploads(service, limit=_ADOPT_SCAN_LIMIT):
            if upload.title != payload.title or not upload.published_at:
                continue
            try:
                published_dt = datetime.fromisoformat(upload.published_at.replace("Z", "+00:00"))
            except ValueError:
                continue
            if published_dt >= cutoff:
                candidates.append(upload.video_id)

        if len(candidates) > 1:
            raise AmbiguousUploadMatchError(
                f"{len(candidates)} recent uploads match title={payload.title!r} within "
                f"{_ADOPT_RECENCY} — refusing to guess which is this short's video"
            )
        return candidates[0] if candidates else None

    def _assign_playlist(self, video_id: str, record: ShortPublishRecord) -> Optional[str]:
        """Assign the short to its country's Shorts playlist; ``None`` when skipped.

        ``sampled_regions`` are PLACE labels ("West Cork", "Exampleton, Tennessee"),
        never a country, so the country is derived (``short_region``) and passed
        as the resolver's ``region``; the place rides along as ``location``.
        A short with no derivable country, or one the config has no Shorts rule
        for, is NOT assigned: filing it under ``youtube_playlists.default``
        ("Unsorted (fallback)") is the defect this guards.
        """
        country = short_region(record.sampled_regions)
        if not country:
            return None
        metadata = {"format": "shorts", "region": country}
        if record.sampled_regions:
            metadata["location"] = record.sampled_regions[0]
        if resolve_playlist(metadata, self._config) == self._config.drive.youtube_playlists.default:
            return None
        return self._uploader.assign_video_to_resolved_playlist(
            video_id, metadata=metadata, config=self._config
        )
