"""Build a typed, classified manifest of the GoMoveShift YouTube channel.

Enumerates every upload via the YouTube Data API uploads playlist (unioned with
its format-split siblings, see ``enumerate_uploads``), classifies
each video (``short`` / ``stream`` / ``real_lf`` / ``held``), and returns a
stable dict consumed by the sync-site command and the coverage audit. The
``service`` and ``now`` are injected so the builder is unit-testable without
live credentials and ``generated_at`` is deterministic.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from googleapiclient.errors import HttpError

from gomoveshift.pipeline._atomic_write import atomic_write_json
from gomoveshift.pipeline.site_sync.known_ids import FROZEN_KNOWN_IDS

logger = logging.getLogger(__name__)

# YouTube derives per-format siblings of the uploads playlist (``UU<channel>``) by
# swapping the ``UU`` prefix: shorts, long-form, live. The plain ``UU`` listing
# repeats some ids and omits others, so every consumer that must see the whole channel
# unions the three splits into it.
UPLOADS_SPLIT_PREFIXES = ("UUSH", "UULF", "UULV")

# Classification thresholds (seconds).
SHORT_MAX_SECONDS = 90  # <= this is a Short.
STREAM_MIN_SECONDS = 21600  # > this (6h) is a livestream/raw, not a real LF.

_ISO8601_DURATION = re.compile(
    r"^P(?:(?P<days>\d+)D)?"
    r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)

# Live-cam / livestream titles must NOT reach the site as real long-forms: the
# channel has sub-6h "… Cam" clips (e.g. 49 min, 227 min) that the duration
# rule alone would mis-tag. The rule only applies above Short length (or at 0s,
# which live/upcoming broadcasts report as ``P0D``): a <=90s clip is a Short
# whatever its title says ("Car Cam: …" shorts, a 46s "… is Live!" promo).
_CAM_TITLE = re.compile(r"\bcam\b|\blive\b", re.IGNORECASE)


def parse_iso8601_duration(s: str) -> int:
    """Convert a YouTube ``contentDetails.duration`` to total seconds.

    Handles hours/minutes/seconds and missing components (``PT44M``,
    ``PT90S``). Returns 0 for ``P0D`` / live edge cases and any unparseable
    input (empty string, garbage).
    """
    if not s:
        return 0
    match = _ISO8601_DURATION.match(s.strip())
    if not match:
        return 0
    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def classify(duration_seconds: int, privacy_status: str, title: str = "") -> str:
    """Classify a video as ``held`` / ``short`` / ``stream`` / ``real_lf``.

    Rules applied in order: non-public is ``held``; a live-cam title (``cam``
    or ``live`` as a whole word) is a ``stream`` only when the duration is 0
    (live/upcoming) or above Short length; then a short (<=90s, whatever the
    title); then a livestream/raw (>6h); otherwise a real long-form.
    """
    if privacy_status != "public":
        return "held"
    if (duration_seconds == 0 or duration_seconds > SHORT_MAX_SECONDS) and _CAM_TITLE.search(title):
        return "stream"
    if duration_seconds <= SHORT_MAX_SECONDS:
        return "short"
    if duration_seconds > STREAM_MIN_SECONDS:
        return "stream"
    return "real_lf"


def _list_playlist(service, playlist_id: str) -> dict[str, dict]:
    """Every video of one playlist as ``{video_id: playlist snippet}``, deduped.

    Keeps the first snippet per id and preserves order. A listing that returns a
    repeated id logs one WARNING (the symptom of the uploads-listing blind spot).
    """
    found: dict[str, dict] = {}
    entries = 0
    page_token = None
    while True:
        response = (
            service.playlistItems()
            .list(part="snippet", playlistId=playlist_id, maxResults=50, pageToken=page_token)
            .execute()
        )
        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            video_id = snippet.get("resourceId", {}).get("videoId")
            if not video_id:
                continue
            entries += 1
            found.setdefault(video_id, snippet)
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    if entries != len(found):
        logger.warning(
            "uploads enumeration: playlist %s returned %d entries but %d unique ids (%d repeated)",
            playlist_id,
            entries,
            len(found),
            entries - len(found),
        )
    return found


def _recover_known_ids(
    service, channel_id: str | None, known_ids: Iterable[str], found: dict[str, dict]
) -> None:
    """Add every known id no listing returned that ``videos.list`` places on this channel.

    The listings can omit private/unlisted videos entirely, so ids remembered
    locally are asked for directly (50 per call, ``part=snippet``). Only a video
    whose ``snippet.channelId`` is ``channel_id`` is kept; an id ``videos.list``
    does not return (deleted) or that belongs to another channel is ignored. The
    snippet is stored under the id with the keys the playlist snippet carries
    (``title``, ``publishedAt``). Logs one INFO line: recovered vs unresolved.
    """
    wanted = [i for i in dict.fromkeys(known_ids) if i and i not in found]
    if not wanted:
        return
    if not channel_id:
        logger.warning(
            "uploads enumeration: %d known ids not listed but the channel id is unavailable;"
            " not verifying ownership, skipped",
            len(wanted),
        )
        return
    recovered = 0
    for start in range(0, len(wanted), 50):
        response = (
            service.videos().list(part="snippet", id=",".join(wanted[start : start + 50])).execute()
        )
        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            if item.get("id") and snippet.get("channelId") == channel_id:
                found.setdefault(item["id"], snippet)
                recovered += 1
    logger.info(
        "uploads enumeration: known ids recovered %d via videos.list, %d unresolved"
        " (deleted or not on this channel)",
        recovered,
        len(wanted) - recovered,
    )


def enumerate_uploads(service, *, known_ids: Iterable[str] = ()) -> dict[str, dict]:
    """Every own-channel video id the Data API will list, as ``{id: snippet}``.

    The one shared enumeration (site manifest, library-hygiene): the uploads
    playlist unioned with its format-split siblings (``UUSH`` shorts, ``UULF``
    long-form, ``UULV`` live), deduped, plain listing first. A split playlist
    that 404s (the channel has none of that format) is skipped; any other error,
    and any error on the plain listing, propagates. ``known_ids`` (ids the caller
    remembers locally) that no listing returned are then fetched by id and kept
    when they belong to this channel -- the listings never return private or
    unlisted videos reliably. Read-only; costs one ``channels.list``,
    ``ceil(n/50)`` ``playlistItems.list`` units per playlist and
    ``ceil(missing/50)`` ``videos.list`` units (none without ``known_ids``).
    """
    channels_response = service.channels().list(part="id,contentDetails", mine=True).execute()
    items = channels_response.get("items") or []
    if not items:
        return {}
    uploads_playlist = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    found = _list_playlist(service, uploads_playlist)
    if uploads_playlist.startswith("UU"):
        for prefix in UPLOADS_SPLIT_PREFIXES:
            split_id = prefix + uploads_playlist[2:]
            try:
                listed = _list_playlist(service, split_id)
            except HttpError as exc:
                if getattr(exc.resp, "status", None) != 404:
                    raise
                logger.info("uploads enumeration: split playlist %s not found; skipped", split_id)
                continue
            for video_id, snippet in listed.items():
                found.setdefault(video_id, snippet)
    _recover_known_ids(service, items[0].get("id"), known_ids, found)
    return found


def build_channel_manifest(
    service, *, now: datetime, known_ids: Iterable[str] | None = None
) -> dict:
    """Enumerate the channel's uploads and return a classified manifest dict.

    ``service`` is an already-built YouTube Data API client; ``now`` is a
    tz-aware timestamp stamped into ``generated_at``; ``known_ids`` are locally
    remembered ids forwarded to ``enumerate_uploads`` (private/unlisted videos),
    defaulting to the frozen seed ``FROZEN_KNOWN_IDS`` (an explicit value, even
    empty, replaces it).
    """
    if known_ids is None:
        known_ids = FROZEN_KNOWN_IDS
    by_id: dict[str, dict] = {
        video_id: {
            "id": video_id,
            "title": snippet.get("title", ""),
            "published_at": snippet.get("publishedAt", ""),
            "duration_seconds": 0,
            "privacy_status": "",
        }
        for video_id, snippet in enumerate_uploads(service, known_ids=known_ids).items()
    }

    # Fetch duration + privacy in batches of 50 and merge.
    ids = list(by_id.keys())
    for start in range(0, len(ids), 50):
        batch = ids[start : start + 50]
        response = (
            service.videos()
            .list(part="contentDetails,status,snippet", id=",".join(batch))
            .execute()
        )
        for video in response.get("items", []):
            entry = by_id.get(video.get("id"))
            if entry is None:
                continue
            duration = video.get("contentDetails", {}).get("duration", "")
            entry["duration_seconds"] = parse_iso8601_duration(duration)
            entry["privacy_status"] = video.get("status", {}).get("privacyStatus", "")

    counts = {"total": 0, "short": 0, "stream": 0, "real_lf": 0, "held": 0}
    videos = []
    for entry in by_id.values():
        category = classify(entry["duration_seconds"], entry["privacy_status"], entry["title"])
        videos.append({**entry, "category": category})
        counts["total"] += 1
        counts[category] += 1

    videos.sort(key=lambda v: v["published_at"], reverse=True)

    return {
        "generated_at": now.isoformat(),
        "source": "youtube-data-api/uploads-playlist",
        "counts": counts,
        "videos": videos,
    }


def write_manifest(manifest: dict, path: Path) -> None:
    """Atomically write the manifest as 2-space-indented JSON."""
    atomic_write_json(Path(path), manifest, indent=2, sort_keys=False)
