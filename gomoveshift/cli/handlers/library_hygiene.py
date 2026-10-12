"""``library-hygiene`` — list, and on ``--apply`` set private, the public low-view
templated shorts on the GMS channel (a weekly, suggest-first digest the operator confirms).

Source of truth is the LIVE uploads listing, not the tracking store: the store
carries no privacy status and almost no YouTube ids, so it cannot name the
list. Series = the ``{location}`` a title's template captured
(``publishing.anti_spam_gate.PRE_POOL_TITLE_SHAPES`` / ``LEGACY_TITLE_POOL_RAW``,
``pipeline.social_generator.TITLE_FRAME_TEMPLATES``); a title matching no
template has no series and is skipped, never guessed. The selection itself is
``tracking.volume.select_library_hygiene_candidates`` unchanged: public,
strictly older than 30 days, strictly under 100 views, series of more than 5,
heaviest series first.

Dry-run by default. ``--apply`` privates at most ``--max`` (50) per run and
appends one ledger line (prior status) BEFORE each API call, so ``--undo
<ledger>`` restores exactly what a run changed. ``--ids`` privates an explicit
list instead of the selection (e.g. the unlisted test renders)."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ...paths import gms_output_dir
from ...pipeline.drive_phases.youtube_uploader import YouTubeUploader
from ...pipeline.site_sync.channel_manifest import enumerate_uploads, parse_iso8601_duration
from ...pipeline.site_sync.known_ids import FROZEN_KNOWN_IDS
from ...pipeline.social_generator import TERRAIN_DESCRIPTORS, TITLE_FRAME_TEMPLATES
from ...publishing.anti_spam_gate import LEGACY_TITLE_POOL_RAW, PRE_POOL_TITLE_SHAPES
from ...tracking.volume import (
    DEFAULT_HYGIENE_BATCH_SIZE,
    TrackedShort,
    select_library_hygiene_candidates,
)
from ..logging_setup import configure_logging

#: YouTube's Shorts ceiling; anything longer is a long-form and never a hygiene target.
MAX_SHORT_SECONDS = 180

_PLACEHOLDER = re.compile(r"\{(\w+)\}")

#: ``{terrain_desc}`` only ever renders one of the generator's descriptors
#: (any case), so the slot is bounded — otherwise ``"{location} {terrain_desc}"``
#: would match every two-word title and name a series for it.
_TERRAIN_ALT = (
    "(?:"
    + "|".join(sorted({re.escape(v) for v in TERRAIN_DESCRIPTORS.values()}, key=len, reverse=True))
    + ")"
)
_SLOT_RES = {"location": "(?P<location>.+?)", "terrain_desc": _TERRAIN_ALT}


def _template_regex(template: str) -> "re.Pattern[str]":
    """``"Open Road: {location} 🚗"`` → a full-match regex capturing ``location``.

    Literal parts are escaped; ``{location}`` becomes the named group,
    ``{terrain_desc}`` the bounded vocabulary, every other placeholder
    (``road``, ``county``, ``moment``) a lazy wildcard."""
    parts: List[str] = []
    pos = 0
    for m in _PLACEHOLDER.finditer(template):
        parts.append(re.escape(template[pos : m.start()]))
        parts.append(_SLOT_RES.get(m.group(1), ".+?"))
        pos = m.end()
    parts.append(re.escape(template[pos:]))
    return re.compile("".join(parts), re.IGNORECASE)


#: Every pool the channel's shorts were ever titled from: the four pre-pool shapes,
#: the retired 100 and the current frames.
_TEMPLATE_RES = tuple(
    _template_regex(t)
    for t in (*PRE_POOL_TITLE_SHAPES, *LEGACY_TITLE_POOL_RAW, *TITLE_FRAME_TEMPLATES)
    if "{location}" in t
)


def series_from_title(title: str) -> Optional[str]:
    """The location a retired template captured, or ``None`` for any other title.

    When several templates match, the SHORTEST capture wins: the most specific
    template ate the most literal text, so its location carries the least
    template noise."""
    captures = [
        m.group("location").strip()
        for r in _TEMPLATE_RES
        for m in [r.fullmatch(title.strip())]
        if m
    ]
    return min(captures, key=len) if captures else None


@dataclass(frozen=True)
class LiveVideo:
    id: str
    title: str
    published_at: date
    privacy: str
    views: int
    seconds: int


def ledger_video_ids() -> List[str]:
    """Every ``video_id`` any hygiene ledger under the output dir recorded, deduped.

    A video this command once set private is still on the channel, but no uploads
    listing returns it; the ledgers are the local memory of those ids. Unreadable
    files and malformed rows are skipped."""
    ids: Dict[str, None] = {}
    for path in sorted((gms_output_dir() / "library-hygiene").glob("*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            video_id = row.get("video_id") or row.get("id") if isinstance(row, dict) else None
            if isinstance(video_id, str) and video_id:
                ids.setdefault(video_id)
    return list(ids)


def list_uploads(service: Any, known_ids: Optional[Iterable[str]] = None) -> List[LiveVideo]:
    """Every upload on the authorised channel, read-only (3 list endpoints).

    Ids come from the shared ``enumerate_uploads`` (uploads playlist unioned with
    its format-split siblings), so a video the plain listing omits is still seen,
    plus ``known_ids`` -- default the ids in the hygiene ledgers unioned with the
    frozen seed ``FROZEN_KNOWN_IDS`` -- so a private
    video no listing returns still resolves (``--ids`` on it would otherwise
    refuse the whole batch as "not on this channel")."""
    if known_ids is None:
        known_ids = list(dict.fromkeys([*ledger_video_ids(), *FROZEN_KNOWN_IDS]))
    ids = list(enumerate_uploads(service, known_ids=known_ids))
    videos: List[LiveVideo] = []
    for i in range(0, len(ids), 50):
        page = (
            service.videos()
            .list(part="snippet,status,statistics,contentDetails", id=",".join(ids[i : i + 50]))
            .execute()
        )
        for v in page.get("items", []):
            videos.append(
                LiveVideo(
                    id=v["id"],
                    title=v["snippet"]["title"],
                    published_at=date.fromisoformat(v["snippet"]["publishedAt"][:10]),
                    privacy=v["status"]["privacyStatus"],
                    views=int(v.get("statistics", {}).get("viewCount", 0)),
                    seconds=parse_iso8601_duration(v.get("contentDetails", {}).get("duration", "")),
                )
            )
    return videos


def tracked_shorts(videos: Iterable[LiveVideo]) -> List[TrackedShort]:
    """Selector rows for the templated shorts only; long-forms and untemplated titles drop."""
    rows: List[TrackedShort] = []
    for v in videos:
        if v.seconds > MAX_SHORT_SECONDS:
            continue
        series = series_from_title(v.title)
        if series is None:
            continue
        rows.append(
            TrackedShort(
                id=v.id,
                published_at=v.published_at,
                privacy=v.privacy,
                views=v.views,
                series=series,
            )
        )
    return rows


def _today() -> date:
    return date.today()


def default_ledger_path() -> Path:
    return gms_output_dir() / "library-hygiene" / f"{_today().isoformat()}-ledger.jsonl"


def _set_private(uploader: Any, targets: Sequence[LiveVideo], ledger: Path) -> int:
    """Ledger line first, then the call, per video; returns how many were changed."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    changed = 0
    with ledger.open("a", encoding="utf-8") as fh:
        for v in targets:
            row = {
                **asdict(v),
                "video_id": v.id,
                "series": series_from_title(v.title),
                "prior_privacy": v.privacy,
                "new_privacy": "private",
                "at": datetime.now(timezone.utc).isoformat(),
            }
            row["published_at"] = v.published_at.isoformat()
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            uploader.update_privacy(v.id, privacy="private")
            changed += 1
            print(f"LIBRARY_HYGIENE_APPLIED id={v.id} prior={v.privacy} title={v.title!r}")
    print(f"LIBRARY_HYGIENE: {changed} set private; ledger {ledger} (restore with --undo {ledger})")
    return changed


def _undo(uploader: Any, ledger: Path) -> int:
    rows = [
        json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    for row in rows:
        uploader.update_privacy(row["video_id"], privacy=row["prior_privacy"])
        print(f"LIBRARY_HYGIENE_RESTORED id={row['video_id']} privacy={row['prior_privacy']}")
    print(f"LIBRARY_HYGIENE: {len(rows)} restored from {ledger}")
    return 0


def cmd_library_hygiene(args: Any, config: Any) -> int:
    configure_logging()
    uploader = YouTubeUploader()
    ledger = Path(args.ledger) if getattr(args, "ledger", None) else default_ledger_path()
    if getattr(args, "undo", None):
        return _undo(uploader, Path(args.undo))

    videos = list_uploads(uploader._build_service())
    by_id: Dict[str, LiveVideo] = {v.id: v for v in videos}
    apply = bool(getattr(args, "apply", False))

    ids = getattr(args, "ids", None)
    if ids:
        missing = [i for i in ids if i not in by_id]
        if missing:
            print(f"library-hygiene: not on this channel: {', '.join(missing)}", file=sys.stderr)
            return 2
        targets: List[LiveVideo] = []
        for i in ids:
            v = by_id[i]
            if v.privacy == "private":
                print(f"LIBRARY_HYGIENE_SKIP id={v.id} already private title={v.title!r}")
            else:
                print(f"LIBRARY_HYGIENE_TARGET id={v.id} privacy={v.privacy} title={v.title!r}")
                targets.append(v)
        if not apply:
            print(f"LIBRARY_HYGIENE: dry run — {len(targets)} would be set private; pass --apply")
            return 0
        _set_private(uploader, targets, ledger)
        return 0

    shorts = tracked_shorts(videos)
    batches = select_library_hygiene_candidates(shorts, _today())
    candidates = [c for batch in batches for c in batch]
    for c in candidates:
        print(c.log_line())
    series = sorted({c.series for c in candidates})
    print(
        f"LIBRARY_HYGIENE: {len(candidates)} candidates in {len(series)} series ({len(shorts)} templated shorts seen)"
    )
    cap = int(getattr(args, "max", DEFAULT_HYGIENE_BATCH_SIZE) or DEFAULT_HYGIENE_BATCH_SIZE)
    if not apply:
        print(
            f"LIBRARY_HYGIENE: dry run — pass --apply to set the first {min(cap, len(candidates))} private"
        )
        return 0
    _set_private(uploader, [by_id[c.id] for c in candidates[:cap]], ledger)
    return 0
