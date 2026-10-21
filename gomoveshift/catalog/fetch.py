"""Live channel pull for the title-accuracy inventory (read-only).

``fetch_catalog`` lists videos by id in batches of 50 and returns the
``live_catalog.json`` shape ``catalog.inventory.build_inventory`` consumes.
It only reads from the Data API; nothing is ever written to YouTube.

``checked_ids`` is every id YouTube was asked about by id -- the listed ids
plus the caller's ``known_ids`` (which ``enumerate_uploads`` fetched by id).
Only an id in it and absent from ``videos`` is proof the channel does not
return that video; ``queue mark-missing-remote`` acts on nothing else.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..pipeline.site_sync.channel_manifest import parse_iso8601_duration

BATCH_SIZE = 50
_PARTS = "snippet,contentDetails,status,recordingDetails"


def fetch_catalog(
    service: Any, ids: Iterable[str], *, known_ids: Iterable[str] = ()
) -> dict[str, Any]:
    ids = sorted(set(ids))
    rows: dict[str, dict[str, Any]] = {}
    for i in range(0, len(ids), BATCH_SIZE):
        resp = (
            service.videos()
            .list(part=_PARTS, id=",".join(ids[i : i + BATCH_SIZE]), maxResults=BATCH_SIZE)
            .execute()
        )
        for it in resp.get("items", []):
            sn, st = it["snippet"], it["status"]
            rows[it["id"]] = {
                "title": sn.get("title"),
                "description": sn.get("description"),
                "tags": sn.get("tags", []),
                "published_at": sn.get("publishedAt"),
                "duration_s": parse_iso8601_duration(it["contentDetails"].get("duration", "")),
                "privacy": st.get("privacyStatus"),
                "publish_at": st.get("publishAt"),
                "recording_date": (it.get("recordingDetails") or {}).get("recordingDate"),
            }
    return {
        "enumerated": len(ids),
        "listed": len(rows),
        "videos": rows,
        "checked_ids": sorted(set(ids) | set(known_ids)),
    }
