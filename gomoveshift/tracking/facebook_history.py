"""History ledger of Facebook Page Reels posted before first-party publishing.

First-party Facebook publishing began 2026-09-10; every earlier Reel went out
through Metricool, so the tracking store has no record of them. This module
lists them once (`FacebookCollector.list_page_reels`), tags each
``source=metricool`` (or ``first_party`` from the cutover on), stores lifetime
metrics read once through `collect_short_metrics`, and marks the Reels whose
captions were hand-edited on 2026-10-02 so matched-cohort audits can exclude
them (`in_matched_cohort`).

The cutover is judged on the Meta (Pacific) day of ``created_time``, the same
day boundary the Page insights use: a Reel created 2026-09-10T05:00Z is still
Sept 9 in Pacific and counts as Metricool.

Storage: ``<root>/<brand>/tracking/facebook_history.json`` beside
``channel_daily.json``. Stored entries are never deleted or rewritten; a
re-run only adds Reels not yet stored (and the caption-edit mark when a
stored entry lacks it).
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from ..paths import gms_output_dir
from .channel_daily import FACEBOOK_BACKFILL_START

logger = logging.getLogger(__name__)

#: First Pacific day Reels went out first-party; earlier ones are Metricool's.
FIRST_PARTY_START = "2026-09-10"
_PACIFIC = ZoneInfo("America/Los_Angeles")
#: Hand-edited captions (2026-10-02 browser run): reel ids, tracked data file.
CAPTION_EDITS_PATH = Path(__file__).parent / "data" / "facebook_caption_edits.json"


def _store_path(brand: str, root: Optional[Path] = None) -> Path:
    """``<root>/<brand>/tracking/facebook_history.json``."""
    base = Path(root) if root else gms_output_dir()
    return base / brand / "tracking" / "facebook_history.json"


def load_history(brand: str, root: Optional[Path] = None) -> Dict[str, Any]:
    """``{"listing": {...}|None, "entries": [...]}``; a missing or corrupt
    ledger loads as empty so a caller never aborts on it."""
    empty: Dict[str, Any] = {"listing": None, "entries": []}
    path = _store_path(brand, root=root)
    if not path.exists():
        return empty
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load facebook_history store at %s: %s", path, exc)
        return empty
    if not isinstance(data, dict):
        return empty
    entries = data.get("entries")
    listing = data.get("listing")
    return {
        "listing": listing if isinstance(listing, dict) else None,
        "entries": [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else [],
    }


def _save_history(brand: str, data: Dict[str, Any], root: Optional[Path] = None) -> Path:
    path = _store_path(brand, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"brand": brand, **data}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def is_complete(brand: str, root: Optional[Path] = None) -> bool:
    """True once one listing succeeded (even an empty one) and was recorded."""
    listing = load_history(brand, root)["listing"]
    return bool(listing and listing.get("completed_at"))


def _pacific_day(created_time: str) -> str:
    parsed = datetime.strptime(created_time, "%Y-%m-%dT%H:%M:%S%z")
    return parsed.astimezone(_PACIFIC).date().isoformat()


def source_for(created_time: str) -> str:
    """``metricool`` for a Reel created before 2026-09-10 (Pacific day)."""
    return "metricool" if _pacific_day(created_time) < FIRST_PARTY_START else "first_party"


def _caption_edits() -> Tuple[str, set]:
    try:
        data = json.loads(CAPTION_EDITS_PATH.read_text(encoding="utf-8"))
        return str(data["edited_on"]), {str(i) for i in data["reel_ids"]}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Facebook caption-edit list unreadable: %s", exc)
        return "", set()


def in_matched_cohort(entry: Dict[str, Any]) -> bool:
    """False for a Reel whose caption was hand-edited (not like-for-like)."""
    return not entry.get("caption_edited_on")


def backfill_facebook_history(
    facebook_collector,
    brand: str,
    root: Optional[Path] = None,
    until: str = FIRST_PARTY_START,
) -> Tuple[Optional[int], int]:
    """List the Page's Reels from `FACEBOOK_BACKFILL_START` to ``until`` and
    add the ones not yet stored. Returns ``(listed, stored)``: ``listed`` is
    ``None`` when the listing was refused (nothing is written, the ledger
    stays incomplete); ``stored`` is the ledger's entry count afterwards.
    """
    reels = facebook_collector.list_page_reels(FACEBOOK_BACKFILL_START, until)
    if reels is None:
        return None, len(load_history(brand, root)["entries"])
    data = load_history(brand, root)
    entries: List[Dict[str, Any]] = data["entries"]
    edited_on, edited_ids = _caption_edits()
    stored_ids = {e.get("id") for e in entries}
    for entry in entries:  # additive mark only; nothing else is rewritten
        if entry.get("id") in edited_ids and "caption_edited_on" not in entry:
            entry["caption_edited_on"] = edited_on
    for reel in reels:
        rid, created = str(reel.get("id") or ""), reel.get("created_time")
        if not rid or not created or rid in stored_ids:
            continue
        stored_ids.add(rid)
        entry = {
            "id": rid,
            "created_time": created,
            "pacific_day": _pacific_day(created),
            "caption": reel.get("description"),
            "source": source_for(created),
            "metrics": facebook_collector.collect_short_metrics(rid).to_dict(),
        }
        if rid in edited_ids:
            entry["caption_edited_on"] = edited_on
        entries.append(entry)
    entries.sort(key=lambda e: (e.get("created_time") or "", e.get("id") or ""))
    data["listing"] = {
        "completed_at": date.today().isoformat(),
        "since": FACEBOOK_BACKFILL_START,
        "until": until,
        "listed": len(reels),
    }
    _save_history(brand, data, root)
    return len(reels), len(entries)
