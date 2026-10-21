"""Did a post for this Instagram delivery land on the account?

One implementation, two callers. ``queue recover-terminal`` gate (c) refuses a
recovery when the scan finds a match; ``dispatch.reconcile_awaiting_ig``
closes a parked row as published when it finds one and retries it only when
the scan PROVES absence. Pure data logic over the account's recent media --
the Graph read itself stays with the caller."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .queue import RecoveryRefusedError

#: How many of the account's most recent media the scan reads.
SCAN_LIMIT = 50
#: Caption characters compared (whitespace-collapsed, case-folded).
CAPTION_PREFIX_LEN = 40


class ScanUnprovableError(RecoveryRefusedError):
    """The scan cannot prove whether the post exists (empty page, a page that
    stops short of the row's creation, no caption to compare)."""


@dataclass(frozen=True)
class ScanResult:
    matches: List[Dict[str, Any]]
    scanned: int
    oldest_scanned: Optional[str]


def parse_ts(value: Optional[str]) -> Optional[datetime]:
    """Graph timestamps read ``2026-10-01T12:00:00+0000``; queue rows are isoformat."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S%z")
        except ValueError:
            return None


def check_scan_reaches_back(media: List[Dict[str, Any]], created_at: str) -> Optional[str]:
    """A short page is the whole account; a full page must reach back to the
    row's creation, or a post made by the failed attempt could sit past it."""
    if len(media) < SCAN_LIMIT:
        return None
    stamps = [parse_ts(m.get("timestamp")) for m in media]
    if any(t is None for t in stamps):
        raise ScanUnprovableError("remote scan has media without a timestamp; cannot bound it")
    created = parse_ts(created_at)
    if created is None:
        raise ScanUnprovableError(f"delivery created_at {created_at!r} is unreadable")
    oldest = min(stamps)
    if oldest > created:
        raise ScanUnprovableError(
            f"remote scan of {len(media)} media does not reach back to the row's creation "
            f"({oldest.isoformat()} > {created.isoformat()}); cannot prove no duplicate"
        )
    return oldest.isoformat()


def caption_prefix(text: Optional[str]) -> str:
    return " ".join((text or "").split())[:CAPTION_PREFIX_LEN].casefold()


def scan_for_post(
    media: List[Dict[str, Any]],
    *,
    created_at: str,
    caption: Optional[str],
    sha256: Optional[str],
) -> ScanResult:
    """The media matching this delivery by caption prefix or sha256. Raises
    ``ScanUnprovableError`` when an empty result would not prove absence."""
    if not media:
        raise ScanUnprovableError("remote scan returned no media; cannot prove no duplicate")
    oldest = check_scan_reaches_back(media, created_at)
    prefix = caption_prefix(caption)
    if not prefix:
        raise ScanUnprovableError("record has an empty Instagram caption; cannot scan by caption")
    matches = [
        m
        for m in media
        if caption_prefix(m.get("caption")) == prefix or (sha256 and m.get("sha256") == sha256)
    ]
    return ScanResult(matches=matches, scanned=len(media), oldest_scanned=oldest)
