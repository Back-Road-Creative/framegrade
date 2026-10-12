"""Enrol every upload on the channel into the tracking store.

The store only knew uploads the pipeline made; the channel also holds older,
private and unlisted ones. Volume/cadence evidence must see all of them, so
this walks the full uploads playlist (``list_uploads``, uncapped) and
registers the ones no tracking record already holds. Idempotent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, List, Optional

from ..publishing.youtube_uploads_listing import Upload, list_uploads
from ..video_types import SHORTS_MAX_DURATION
from .store import TrackingStore

ENROL_PROJECT = "channel-import"
PLATFORM = "youtube"


@dataclass(frozen=True)
class EnrolCoverage:
    shorts_total: int
    long_total: int
    oldest: str
    newest: str
    shorts_added: int = 0
    long_added: int = 0

    def line(self) -> str:
        return (
            f"EVIDENCE_COVERAGE shorts_total={self.shorts_total} "
            f"long_total={self.long_total} oldest={self.oldest} newest={self.newest}"
        )


def _is_short(u: Upload) -> bool:
    return u.duration_sec is not None and u.duration_sec <= SHORTS_MAX_DURATION


def _when(u: Upload) -> Optional[str]:
    return u.publish_at or u.published_at


def _source(u: Upload, now: datetime) -> str:
    """A future ``publishAt`` is a schedule, never a confirmed publish."""
    if u.publish_at:
        try:
            if datetime.fromisoformat(u.publish_at.replace("Z", "+00:00")) > now:
                return "platform_scheduled"
        except ValueError:
            pass
    return "platform_confirmed"


def _known_ids(store: TrackingStore) -> set:
    known = set()
    for rec in store.load_all_records():
        known.update(s.platform_ids.get(PLATFORM) for s in rec.shorts)
        known.update(lf.video_id for lf in rec.long_forms)
    known.discard(None)
    return known


def enrol_channel(service: Any, store: TrackingStore, *, now: Optional[datetime] = None):
    """Register unknown uploads; returns ``EnrolCoverage`` for the whole channel."""
    now = now or datetime.now(timezone.utc)
    uploads: List[Upload] = list_uploads(service)
    known = _known_ids(store)
    shorts = [u for u in uploads if _is_short(u)]
    longs = [u for u in uploads if not _is_short(u)]

    new_shorts = [u for u in shorts if u.video_id not in known]
    if new_shorts:
        base = (
            max((s.index for s in store.load(ENROL_PROJECT).shorts), default=0)
            if store.exists(ENROL_PROJECT)
            else 0
        )
        store.register_shorts_batch(
            ENROL_PROJECT,
            [
                {
                    "index": base + n,
                    "file": u.title,
                    "duration_sec": u.duration_sec or 0.0,
                }
                for n, u in enumerate(new_shorts, start=1)
            ],
        )
        for n, u in enumerate(new_shorts, start=1):
            when = _when(u)
            store.update_platform_id(
                ENROL_PROJECT,
                base + n,
                PLATFORM,
                u.video_id,
                posted_at=when,
                posted_at_source=_source(u, now) if when else None,
            )

    entries = [
        SimpleNamespace(
            content_type="longform",
            video_id=u.video_id,
            project_name=ENROL_PROJECT,
            title=u.title,
            date=(_when(u) or "")[:10],
            time=(_when(u) or "")[11:19],
        )
        for u in longs
        if u.video_id not in known
    ]
    long_added = store.sync_long_forms_from_ledger(entries)

    dates = sorted(d[:10] for u in uploads if (d := _when(u)))
    return EnrolCoverage(
        shorts_total=len(shorts),
        long_total=len(longs),
        oldest=dates[0] if dates else "none",
        newest=dates[-1] if dates else "none",
        shorts_added=len(new_shorts),
        long_added=long_added,
    )
