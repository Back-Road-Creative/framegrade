"""YouTube Shorts publishing adapter. See ``docs/social-publishing-youtube.md``.
Orchestrates ``YouTubeUploader`` only — never re-implements its upload/retry/token/quota handling.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from seo_manager.analytics.final_text_surfaces import compose_description

from ..pipeline.drive_phases import youtube_uploader as youtube_uploader_module
from ..pipeline.drive_phases.youtube_uploader import YouTubeUploader
from ..pipeline.exceptions import YouTubeQuotaError
from ..pipeline.final_text_record import (
    GMS_TEXT_POLICY,
    FinalTextReviewError,
    PublicationGateError,
    check_final_text,
)
from ..pipeline.location_evidence import _evidence_id
from ..pipeline.playlist_resolver import resolve_playlist, short_region
from ..pipeline.title_decision import decide_title, short_place_facts
from .models import ShortPublishRecord, YouTubePayload
from .queue import NEXT_ATTEMPT_PLATFORM_DEFERRAL as PLATFORM_DEFERRAL
from .queue import PUBLISHED_AT_PLATFORM_CONFIRMED as PLATFORM_CONFIRMED
from .queue import PUBLISHED_AT_PLATFORM_SCHEDULED as PLATFORM_SCHEDULED
from .queue import acquire_lease, find_delivery, mark_missing_remote, transition
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

# U-R1: after a fresh insert, ``videos.list`` must return the new id before
# the row is written ``published``. An empty answer is looked at once more
# after this pause (the insert's own response just named the id), never in a
# loop; empty twice is ``missing_remote``. Module-level so tests skip it.
VERIFY_RECHECK_SEC = 10
_sleep = time.sleep


#: The one surface a Short's review covers: the title as published.
SHORT_SURFACE = "short_title"
_PLACE_SOURCE = "clip_gps_vote"
_ROAD_SOURCE = "clip_gps_road"


def _digest(prefix: str, payload) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return prefix + hashlib.sha256(body.encode("utf-8")).hexdigest()


def _clip_places(record: ShortPublishRecord, segment_id: str) -> list[dict]:
    """The clip's own GPS region votes (``"County, State"``) as place evidence."""
    places = []
    for label in record.sampled_regions:
        county, _, state = (part.strip() for part in label.partition(","))
        if county:
            places.append(
                {
                    "evidence_id": _evidence_id(segment_id, county, county, state, _PLACE_SOURCE),
                    "kind": "location",
                    "value": county,
                    "source": _PLACE_SOURCE,
                }
            )
    return places


def _clip_roads(record: ShortPublishRecord, segment_id: str) -> list[dict]:
    """The clip's own GPS road names as ``kind=road`` evidence. A clip
    with no GPS road yields none, so no road tag can be reviewed for it."""
    return [
        {
            "evidence_id": _evidence_id(segment_id, road, road, "", _ROAD_SOURCE),
            "kind": "road",
            "value": road,
            "source": _ROAD_SOURCE,
        }
        for road in dict.fromkeys(r.strip() for r in record.sampled_roads if r.strip())
    ]


def _short_text_entries(
    record: ShortPublishRecord, places: list, roads: list, named, segment: str
) -> list:
    """The Short's description (one place_line) and tag entries for the review.

    Waypost reviews a description only as filler words plus cited places, so
    the Short sends this one line, not the generated prose (nor its parent link).

    A tag is reviewable only as an exact Waypost tag template or as a name of
    one of the clip's own GPS places or GPS roads; any other tag refuses the Short
    (``tag_unreviewable:<tag>``): no silent drop, no bypass."""
    from seo_manager.analytics.final_text_surfaces import all_templates

    payload = record.payloads.youtube
    if not named:
        raise FinalTextReviewError(
            record.short_index,
            ["description_no_clip_place"],
            "the Short's title names no clip place for its description to cite",
        )
    segment_places = [p["evidence_id"] for p in places]
    entries = [
        {
            "surface": "description",
            "kind": "place_line",
            "text": f"Drive through {named['value']}",
            "evidence_ids": [named["evidence_id"]],
            "scope": "segment",
            "segment_places": segment_places,
        }
    ]
    templates = {text: tid for tid, text in all_templates().items() if tid.startswith("tag.")}
    by_place = {p["value"]: p for p in [*places, *roads]}
    for i, tag in enumerate(payload.tags):
        entry = {"surface": f"tags.{i}", "kind": "tag", "text": tag}
        if tag in templates:
            entry.update(evidence_ids=[], template_id=templates[tag])
        elif tag in by_place:
            entry.update(evidence_ids=[by_place[tag]["evidence_id"]])
        else:
            raise FinalTextReviewError(
                record.short_index,
                [f"tag_unreviewable:{tag}"],
                f"Short tag {tag!r} is neither a Waypost tag template nor a clip place or road",
            )
        entries.append(entry)
    return entries


def short_final_text_context(record: ShortPublishRecord) -> dict:
    """What the Short will send, recomputed from ``record`` now (never read from a
    review), for :func:`check_final_text`. Refuses a Short with no description."""
    payload = record.payloads.youtube
    idx = record.short_index
    if payload is None:
        raise ValueError(f"short_index={idx} carries no YouTube payload")
    if not (payload.description or "").strip():
        raise FinalTextReviewError(idx, ["description_missing"], "the Short has no description")
    segment = f"{record.project_name}-short{idx:03d}"
    decision = decide_title("short", payload.title, None, suffix="")
    return {
        "lf_index": idx,
        "visible": {SHORT_SURFACE: payload.title},
        # A Short's review covers its title only: no drive description layout.
        "description_layout": None,
        # Zero tolerance: this GPS-bearing Short's title, description and tags are
        # checked against its measured route; none persisted = unverifiable, refused.
        "route_record": record.route_record,
        "place_record_required": True,
        "source_fingerprint": _digest("src-", [record.media.sha256]),
        "plan_fingerprint": _digest(
            "plan-",
            [
                segment,
                round(float(record.media.duration_seconds), 3),
                sorted(record.sampled_regions),
                sorted(record.sampled_roads),
            ],
        ),
        "policy_version": str(decision["policy_version"]),
    }


def build_short_final_text_review(record: ShortPublishRecord) -> dict:
    """Waypost's review of the Short's title, description and tags against its own
    clip's GPS votes.

    The place the title names must be one of the clip's region votes; a title
    naming anything else, or no place, comes back not ``agree``. The approval
    marker is the plan's ``reviewed_plan_hash``; an empty one is refused."""
    from seo_manager.analytics.final_text_review import (
        RECORD_VERSION,
        review_final_text,
        review_to_dict,
    )

    context = short_final_text_context(record)
    payload = record.payloads.youtube
    segment = f"{record.project_name}-short{record.short_index:03d}"
    places = _clip_places(record, segment)
    roads = _clip_roads(record, segment)
    named = next(
        (
            p
            for p in places
            if p["value"].lower() in payload.title.lower()
            and short_place_facts(
                payload.title, p["value"], record.sampled_regions, segment_id=segment
            )
        ),
        None,
    )
    facts = (
        short_place_facts(payload.title, named["value"], record.sampled_regions, segment_id=segment)
        if named
        else None
    )
    decision = decide_title("short", payload.title, None, suffix="", facts=facts)
    date_id = f"date-{record.recording_date}"
    seconds = float(record.media.duration_seconds)
    approved = {
        "record_version": RECORD_VERSION,
        "approval_hash": record.reviewed_plan_hash,
        "title": payload.title,
        "recording_date": record.recording_date,
        "date_evidence_id": date_id,
        "source_fingerprint": context["source_fingerprint"],
        "plan_fingerprint": context["plan_fingerprint"],
        "route_places": [
            {
                "evidence_id": p["evidence_id"],
                "retained_seconds": seconds,
                "required": seconds >= GMS_TEXT_POLICY.min_place_seconds,
            }
            for p in places
        ],
        "region_claims": [],
        "visible": [
            {
                "surface": SHORT_SURFACE,
                "kind": "place_line",
                "text": payload.title,
                "evidence_ids": [named["evidence_id"]] if named else [],
                "scope": "segment",
                "segment_places": [p["evidence_id"] for p in places],
            },
            *_short_text_entries(record, places, roads, named, segment),
        ],
    }
    # Road evidence backs a tag only: Waypost's route holds locations alone.
    evidence = (
        places
        + roads
        + [
            {
                "evidence_id": date_id,
                "kind": "date",
                "value": record.recording_date,
                "source": "recording_date",
            }
        ]
    )
    review = review_final_text(approved, decision, evidence, GMS_TEXT_POLICY)
    return review_to_dict(review)


class AmbiguousUploadMatchError(RuntimeError):
    """Two or more recent channel uploads match a retrying short by title
    and recency — adopting either risks stealing the wrong video. A human
    must resolve this; a duplicate is a lesser evil than a wrong adoption."""


@dataclass(frozen=True, slots=True)
class ShortsPublishResult:
    video_id: str
    playlist_id: Optional[str]
    deferred: bool = False
    #: U-R1: the insert returned an id YouTube then did not list -- the row
    #: is ``missing_remote``, not ``published``.
    missing_remote: bool = False


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
        self,
        record: ShortPublishRecord,
        *,
        conn,
        delivery_id: int,
        video_path: Path,
        final_text_review: Optional[dict] = None,
    ) -> ShortsPublishResult:
        """Publish one Short. ``final_text_review`` is a review made earlier
        (carried); without one it is built from ``record`` now. Either way the
        context is recomputed from ``record`` now, so a title, evidence or
        media change since the review is refused before the delivery leaves
        ``queued``."""
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
            # Refuse before the lease and the SUBMITTING mark, so a refused
            # Short stays queued instead of reading as an unknown outcome.
            # Publication gate: bind off (or absent) holds the Short here, still queued,
            # so a held insert never reads as an unknown SUBMITTING outcome.
            if not getattr(youtube_uploader_module, "FINAL_TEXT_BIND_ON", False):
                raise PublicationGateError(youtube_uploader_module.PUBLICATION_GATE_MESSAGE)
            # A GPS-less Short (no clip region votes) has no place evidence to
            # review, as a GPS-less LF; every other Short ships only reviewed.
            ft_context = ft_review = None
            description = payload.description
            if record.sampled_regions:
                ft_context = short_final_text_context(record)
                ft_review = final_text_review or build_short_final_text_review(record)
                check_final_text(
                    ft_review,
                    ft_context,
                    payload.title,
                    description=compose_description(ft_review["surfaces"]),
                    tags=list(payload.tags),
                )
            if ft_review is not None:
                # Send the text Waypost reviewed, not a second build of it.
                description = compose_description(ft_review["surfaces"])
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
                    description=description,
                    tags=list(payload.tags),
                    category_id=payload.category or "2",
                    publish_at=record.schedule.scheduled_at_utc,
                    privacy="private",
                    final_text_review=ft_review,
                    final_text_context=ft_context,
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
            if self._channel_lacks(video_id):
                mark_missing_remote(
                    conn,
                    delivery_id,
                    evidence={"source": "post_upload_videos_list", "lookups": 2},
                )
                logger.error(
                    "short video_id=%s: insert succeeded but videos.list returns no such "
                    "video -- delivery %s is missing_remote, not published; never re-uploaded",
                    video_id,
                    delivery_id,
                )
                return ShortsPublishResult(video_id=video_id, playlist_id=None, missing_remote=True)

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

    def _channel_lacks(self, video_id: str) -> bool:
        """``True`` only when ``videos.list`` twice returns no item for
        ``video_id``. A lookup that raises proves nothing, so it is logged and
        the row publishes; the nightly catalog pass checks it again."""
        try:
            service = self._uploader._build_service()
            for attempt in range(2):
                if attempt:
                    _sleep(VERIFY_RECHECK_SEC)
                resp = service.videos().list(part="status", id=video_id).execute()
                if resp.get("items"):
                    return False
        except Exception:  # noqa: BLE001 -- unknown is not missing
            logger.exception("post-upload videos.list failed for video_id=%s", video_id)
            return False
        return True

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
