"""One paged lister for the channel's uploads playlist (read-only).

Replaces three hand-rolled, capped copies of the same walk. ``limit=None``
walks every page; a caller that only needs the newest N passes ``limit``
explicitly. Costs ``channels.list`` + ``playlistItems.list`` pages +
``videos.list`` batches of 50 -- never ``videos.insert`` quota.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional

_BATCH = 50
_DURATION_RE = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?$")


@dataclass(frozen=True, slots=True)
class Upload:
    video_id: str
    title: str
    published_at: Optional[str]  # snippet.publishedAt
    publish_at: Optional[str]  # status.publishAt (scheduled), else None
    privacy: Optional[str]  # status.privacyStatus
    duration_sec: Optional[float]  # None when contentDetails.duration is absent


def parse_iso_duration(value: Any) -> Optional[float]:
    """``PT1M5S`` -> 65.0; ``None`` for anything unparseable."""
    m = _DURATION_RE.match(str(value or ""))
    if not m or not any(m.groups()):
        return None
    d, h, mi, s = m.groups()
    return int(d or 0) * 86400 + int(h or 0) * 3600 + int(mi or 0) * 60 + float(s or 0)


def list_uploads(service: Any, limit: Optional[int] = None) -> List[Upload]:
    """The channel's uploads, newest first; all of them unless ``limit`` is set."""
    resp = service.channels().list(part="contentDetails", mine=True).execute()
    items = resp.get("items") or []
    if not items:
        return []
    uploads_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    video_ids: List[str] = []
    page_token = None
    while limit is None or len(video_ids) < limit:
        resp = (
            service.playlistItems()
            .list(
                part="contentDetails",
                playlistId=uploads_id,
                maxResults=_BATCH,
                pageToken=page_token,
            )
            .execute()
        )
        video_ids += [i["contentDetails"]["videoId"] for i in resp.get("items", []) or []]
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    if limit is not None:
        video_ids = video_ids[:limit]

    out: List[Upload] = []
    for i in range(0, len(video_ids), _BATCH):
        resp = (
            service.videos()
            .list(
                part="status,snippet,contentDetails",
                id=",".join(video_ids[i : i + _BATCH]),
                maxResults=_BATCH,
            )
            .execute()
        )
        for item in resp.get("items", []) or []:
            snippet = item.get("snippet") or {}
            status = item.get("status") or {}
            out.append(
                Upload(
                    video_id=item["id"],
                    title=snippet.get("title") or "",
                    published_at=snippet.get("publishedAt"),
                    publish_at=status.get("publishAt"),
                    privacy=status.get("privacyStatus"),
                    duration_sec=parse_iso_duration(
                        (item.get("contentDetails") or {}).get("duration")
                    ),
                )
            )
    return out
