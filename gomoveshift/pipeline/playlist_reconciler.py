"""Daily YouTube playlist reconciler — desired-state diff
engine plus the apply glue.

The playlist titles and the per-state playlist roster below are the
brand's defaults (the ``PL_*`` constants and ``STATE_PLAYLISTS``); a channel
with a different layout edits those constants. Live playlist titles are
canonical, so a title that does not exist on the channel fails loudly.

Desired membership = f(classification): reuses ``channel_manifest.classify``
for held/short/stream/real_lf and ``region_resolver.derive_region`` for
region — this module owns none of that logic, only the diff. ``compute_plan``
is pure and dry-run by construction: it takes the already-fetched live
membership and returns planned inserts, calling no API at all. Insert-only,
quota-capped (``max_inserts``); removals exist only behind ``--prune``.

A short's region is never defaulted: title, then overrides, then
the one Shorts playlist it is live in, else it is reported unrouted -- see
``_route``. Superlong parts are the one exception to classification-driven
routing: a part carries no distinct ``kind`` (it classifies as plain
``real_lf``, and a multi-part title's deliberate ``— Part N of M`` suffix
even breaks title-based region derivation outright, never mind a single-part
day that carries no suffix at all). The uploader puts every part straight
into its region's Superlong playlist at upload time, so live membership of
``PL_IRELAND_SUPERLONG``/``PL_USA_SUPERLONG`` is checked FIRST and, when a
video is already a member of one, is authoritative over classification —
never also routed into the region Drives playlist. Those two playlists are
resolved as OPTIONAL (``resolve_playlist_ids(..., optional=...)``): a channel
that has never shipped superlong content has neither yet, and that must not
hard-fail the nightly run.

``reconcile`` is the runnable glue: it fetches the channel + live target-
playlist membership, calls ``compute_plan``, and either reports the plan
(dry-run) or executes it via ``apply_inserts`` — ``playlistItems().insert``
plus, only on ``prune=True``, ledgered ``playlistItems().delete`` inside the
managed playlists are the ONLY mutations -- never a video, never a playlist.
Target playlist titles are resolved to ids via ONE
``playlists.list(mine=True)`` scan and FAIL LOUDLY on any missing title —
live playlist titles are canonical (a typo forks a new one), so this module
never creates a playlist.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..paths import gms_config_home
from .region_resolver import derive_region, derive_short_title_region
from .site_sync.channel_manifest import build_channel_manifest, classify
from .title_builder import state_full_name

logger = logging.getLogger(__name__)

# Per-run cap on named unrouted videos (log lines and the CLI summary ids).
UNROUTED_LOG_CAP = 25

PL_IRELAND_DRIVES = "Ireland Drives"
PL_USA_DRIVES = "USA Drives"
PL_IRELAND_HIGHLIGHTS = "Ireland Highlights"
PL_USA_HIGHLIGHTS = "USA Highlights"
PL_IRELAND_SHORTS = "Ireland Shorts"
PL_USA_SHORTS = "USA Shorts"
PL_MOUNTAINS_CAM = "Mountains Cam Archive"
# Own playlists for superlong (whole drive-day) parts -- exact titles the
# uploader creates (drive_phases/youtube_uploader._get_or_create_playlist),
# auto-created by exact title on first superlong upload. Kept OUT of a
# region's regular Drives playlist (autoplay: a 4h part vs a 20-minute LF
# are different watch commitments).
PL_IRELAND_SUPERLONG = "Ireland Superlong"
PL_USA_SUPERLONG = "USA Superlong"

# US states carrying their own Drives playlist.
STATE_PLAYLISTS = {
    "north carolina": "North Carolina Drives",
    "south carolina": "South Carolina Drives",
    "colorado": "Colorado Drives",
    "tennessee": "Tennessee Drives",
    "virginia": "Virginia Drives",
}

HIGHLIGHT_PLAYLISTS = (PL_IRELAND_HIGHLIGHTS, PL_USA_HIGHLIGHTS)
# Live-membership families used as the last-resort region signal (never shorts,
# never Superlong -- that is checked before anything else).
_IRELAND_FAMILY = (PL_IRELAND_DRIVES, PL_IRELAND_HIGHLIGHTS)
_USA_FAMILY = (PL_USA_DRIVES, PL_USA_HIGHLIGHTS, *STATE_PLAYLISTS.values())

# Fallback playlist the reconciler never routes into; prune empties it.
PL_UNSORTED = "Unsorted (fallback)"

# Playlists that may legitimately not exist yet -- see
# ``resolve_playlist_ids``'s ``optional`` param.
SUPERLONG_PLAYLISTS = frozenset({PL_IRELAND_SUPERLONG, PL_USA_SUPERLONG})

# Every title ``target_playlists`` can return, plus the superlong pair —
# resolved once per run so a missing REQUIRED playlist fails loudly before
# any diff is computed (the superlong pair is resolved too, for membership,
# but is OPTIONAL -- see ``reconcile``).
ALL_TARGET_PLAYLISTS = frozenset(
    {
        PL_IRELAND_DRIVES,
        PL_USA_DRIVES,
        PL_IRELAND_HIGHLIGHTS,
        PL_USA_HIGHLIGHTS,
        PL_IRELAND_SHORTS,
        PL_USA_SHORTS,
        PL_MOUNTAINS_CAM,
        *SUPERLONG_PLAYLISTS,
        *STATE_PLAYLISTS.values(),
    }
)

# Prune scope: routing targets + the (optional) fallback; nothing else is ever touched.
MANAGED_PLAYLISTS = ALL_TARGET_PLAYLISTS | {PL_UNSORTED}
_OPTIONAL_MANAGED = SUPERLONG_PLAYLISTS | {PL_UNSORTED}
_STATE_TITLES = frozenset(STATE_PLAYLISTS.values())
_USA_TITLES = frozenset({*_USA_FAMILY, PL_USA_SHORTS, PL_USA_SUPERLONG})
_NONPUBLIC = ("private", "unlisted")
# Precedence when one entry matches several classes: it is removed once, under the first.
_PRUNE_ORDER = ("deleted", "unsorted", "misfiled", "nonpublic")
_VIDEO_BATCH = 50


# Trailing quality decorations a title carries after its place: " in 4K" and a
# final "| 4K ..." segment. Stripped before the region/state parse -- they sit
# where the trailing comma segment (the state) would otherwise be read.
_QUALITY_SUFFIXES = (
    re.compile(r"\s+in\s+(?:4k|8k|uhd|hd|1080p|2160p)$", re.IGNORECASE),
    re.compile(r"\s*\|\s*(?:4k|8k|uhd|hd)\b[^|]*$", re.IGNORECASE),
)


def _strip_quality_suffix(title: str) -> str:
    title = title.strip()
    changed = True
    while changed:
        changed = False
        for pattern in _QUALITY_SUFFIXES:
            stripped = pattern.sub("", title).rstrip()
            if stripped != title and stripped:
                title, changed = stripped, True
    return title


def derive_title_region_state(title: str) -> tuple[Optional[str], Optional[str]]:
    """(region, state) off title text via ``derive_region`` (trailing comma
    segment, as for structured LF labels) + ``state_full_name`` on that same
    tail, after a trailing quality suffix (" in 4K", "| 4K ...") is dropped.
    Never guesses: unrecognized -> ``(None, None)``."""
    title = _strip_quality_suffix(title)
    region = derive_region(title)
    region = region if region in ("ireland", "usa") else None
    state = None
    if region == "usa" and "," in title:
        full = state_full_name(title.rsplit(",", 1)[-1].strip()).lower()
        state = full if full in STATE_PLAYLISTS else None
    return region, state


def target_playlists(
    kind: str, region: Optional[str], state: Optional[str], title: str
) -> list[str]:
    """Desired membership for one video (``kind`` short/stream/real_lf; the
    caller filters ``held`` out first). Unroutable real_lf/highlight ->
    ``[]`` (reported by the caller) -- never "Unsorted (fallback)". A short
    with no region is likewise ``[]``, never defaulted to USA Shorts. A USA
    highlight keeps its state playlist but never Drives."""
    if kind == "stream":
        return [PL_MOUNTAINS_CAM]
    if kind == "short":
        return {"ireland": [PL_IRELAND_SHORTS], "usa": [PL_USA_SHORTS]}.get(region or "", [])
    highlight = title.startswith("Highlights from")
    if region == "ireland":
        return [PL_IRELAND_HIGHLIGHTS] if highlight else [PL_IRELAND_DRIVES]
    if region == "usa":
        state_pl = STATE_PLAYLISTS.get(state or "")
        base = [PL_USA_HIGHLIGHTS] if highlight else [PL_USA_DRIVES]
        return [*base, state_pl] if state_pl else base
    return []


def resolve_overrides_path(path) -> Optional[Path]:
    """A configured overrides path -> absolute ``Path``. A relative path
    resolves against the config home (``gms_config_home()``, where
    ``config.yaml`` lives), never the cwd, so a scheduled job and an
    interactive shell read the same file. ``""``/None -> None."""
    if not path:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = gms_config_home() / resolved
    return resolved


def load_overrides(path) -> dict:
    """videoId -> {region, state} manual routing; ``_``-prefixed keys are comments.
    A configured path that does not exist is a loud warning, not a silent ``{}``
    (a silently-missing file would leave its videos unrouted on every run)."""
    if not path:
        return {}
    if not Path(path).exists():
        logger.warning("playlist reconciler overrides file not found: %s (using none)", path)
        return {}
    return {k: v for k, v in json.loads(Path(path).read_text()).items() if not k.startswith("_")}


def _live_in(live_membership, video_id, titles) -> list[str]:
    return [t for t in titles if video_id in live_membership.get(t, set())]


def _route(video_id, kind, title, live_membership, overrides) -> list[str]:
    """Desired playlist titles for one public video; ``[]`` = unrouted.

    Live membership is consulted first where the uploader/operator already
    decided: Superlong, then region Highlights (both authoritative over the
    title and never also routed into Drives). Otherwise the title (quality
    suffix stripped) and the overrides decide the region, and live membership
    of one region family is the last resort for a real_lf the title can't place.
    A short takes its region from the title (``derive_short_title_region``), then
    the overrides, then the ONE Shorts playlist it is live in; it is never
    defaulted, so a title region that contradicts live membership wins.
    A usa video also keeps every state playlist it is live in (hand-curated
    multi-state entries must stay desired)."""
    for superlong in (PL_IRELAND_SUPERLONG, PL_USA_SUPERLONG):
        if video_id in live_membership.get(superlong, set()):
            return [superlong]
    region, state = derive_title_region_state(title)
    if kind == "short":
        region = derive_short_title_region(title)
    override = overrides.get(video_id) or {}
    live_states = _live_in(live_membership, video_id, STATE_PLAYLISTS.values())

    highlights = _live_in(live_membership, video_id, HIGHLIGHT_PLAYLISTS)
    if highlights:
        if len(highlights) > 1:
            return []  # in both regions' Highlights: ambiguous, never guessed
        if region is None and override.get("region") == "usa":
            state = override.get("state")
        state_pl = STATE_PLAYLISTS.get(state or "") if highlights[0] == PL_USA_HIGHLIGHTS else None
        extra = live_states if highlights[0] == PL_USA_HIGHLIGHTS else []
        return list(dict.fromkeys([highlights[0], *([state_pl] if state_pl else []), *extra]))

    if region is None and override:
        region, state = override.get("region"), override.get("state")
    if region is None and kind == "short":
        in_shorts = _live_in(live_membership, video_id, (PL_IRELAND_SHORTS, PL_USA_SHORTS))
        if len(in_shorts) == 1:
            region = "ireland" if in_shorts[0] == PL_IRELAND_SHORTS else "usa"
    if region is None and kind == "real_lf":
        in_ireland = bool(_live_in(live_membership, video_id, _IRELAND_FAMILY))
        in_usa = bool(_live_in(live_membership, video_id, _USA_FAMILY))
        if in_ireland != in_usa:
            region = "ireland" if in_ireland else "usa"
    targets = target_playlists(kind, region, state, title)
    if region == "usa" and kind == "real_lf":
        targets = list(dict.fromkeys([*targets, *live_states]))
    return targets


def compute_plan(
    videos,
    live_membership,
    overrides,
    *,
    max_inserts,
    newest_first,
    backfill_since,
    unscannable_titles=frozenset(),
) -> dict:
    """Desired-state diff: missing memberships, newest/oldest-first, capped
    at ``max_inserts``. ``backfill_since`` bounds shorts only.

    Returns ``{inserts, unrouted, desired}``; ``desired`` maps every routed
    public video id to its sorted desired playlist titles (including videos the
    backfill window or the insert cap leaves un-planned) so a prune step can diff
    live membership against it.

    Live membership of Superlong or region Highlights is authoritative and is
    checked before classification -- see ``_route``. A highlight (or superlong
    part) is therefore never also planned into region Drives.

    ``unscannable_titles`` names playlists whose live membership is unknown
    (a legacy-id mid-pagination 404, see ``fetch_live_membership``) -- those
    targets are dropped entirely rather than treated as empty, since an
    empty default would plan a duplicate insert into a playlist that may
    already carry the video."""
    candidates, unrouted, desired = [], [], {}
    for video in videos:
        if video.get("privacy_status") != "public":
            continue
        video_id, title = video["id"], video.get("title", "")
        kind = video.get("category") or classify(
            video.get("duration_seconds", 0), video.get("privacy_status", ""), title
        )
        targets = _route(video_id, kind, title, live_membership, overrides)
        if not targets:
            unrouted.append({"video_id": video_id, "title": title, "kind": kind})
            continue
        desired[video_id] = sorted(targets)
        if kind == "short" and video.get("published_at", "") < backfill_since:
            continue
        for playlist_title in targets:
            if playlist_title in unscannable_titles:
                continue
            if video_id in live_membership.get(playlist_title, set()):
                continue
            candidates.append(
                {
                    "video_id": video_id,
                    "playlist_title": playlist_title,
                    "published_at": video.get("published_at", ""),
                    "kind": kind,
                }
            )
    candidates.sort(key=lambda c: c["published_at"], reverse=newest_first)
    return {"inserts": candidates[:max_inserts], "unrouted": unrouted, "desired": desired}


def resolve_playlist_ids(service, titles, *, optional=frozenset()) -> dict[str, str]:
    """title -> id for every ``titles``, via one paginated ``playlists.list(mine=True)``
    scan. Raises ``RuntimeError`` naming any REQUIRED title not found — this module
    never falls back to ``playlists.insert``. A title in ``optional`` (e.g. the
    superlong pair, only auto-created by the uploader on its first superlong
    upload) is simply omitted from the result when absent instead of raising, so a
    channel with no superlong content yet still reconciles cleanly."""
    titles = set(titles)
    cache: dict[str, str] = {}
    page_token = None
    while True:
        resp = (
            service.playlists()
            .list(part="snippet", mine=True, maxResults=50, pageToken=page_token)
            .execute()
        )
        for item in resp.get("items", []) or []:
            title = (item.get("snippet") or {}).get("title")
            pid = item.get("id")
            if isinstance(title, str) and isinstance(pid, str):
                cache[title] = pid
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    missing = sorted(titles - set(optional) - cache.keys())
    if missing:
        raise RuntimeError(
            f"reconciler target playlist(s) not found on channel (never auto-created): {missing}"
        )
    return {t: cache[t] for t in titles if t in cache}


def fetch_live_items(service, playlist_ids) -> tuple[dict[str, list[dict]], set[str]]:
    """({playlistId: [{item_id, video_id, position}]}, {unscannable playlistId}), via one
    paginated ``playlistItems.list`` scan per id. Read-only.

    Legacy (13-char) playlist ids are known to accept page 1 of
    ``playlistItems.list`` but raise ``HttpError`` 404 ``playlistNotFound`` on any
    subsequent page -- a real, still-existing playlist whose pagination alone is
    broken. A 404 on the FIRST page still means the playlist genuinely doesn't
    exist and is left to propagate unchanged. A mid-pagination 404 instead logs a
    loud warning and records the id as unscannable rather than membership, since a
    partially-scanned playlist would under-count membership and cause duplicate
    ``playlistItems.insert`` calls if planned against."""
    from googleapiclient.errors import HttpError

    items: dict[str, list[dict]] = {}
    unscannable: set[str] = set()
    for playlist_id in playlist_ids:
        rows: list[dict] = []
        page_token = None
        scannable = True
        while True:
            try:
                resp = (
                    service.playlistItems()
                    .list(
                        part="snippet", playlistId=playlist_id, maxResults=50, pageToken=page_token
                    )
                    .execute()
                )
            except HttpError:
                if page_token is None:
                    raise
                logger.warning(
                    "playlist %s: legacy pagination 404 -- membership unknown, skipping",
                    playlist_id,
                )
                unscannable.add(playlist_id)
                scannable = False
                break
            for item in resp.get("items", []) or []:
                snippet = item.get("snippet") or {}
                vid = (snippet.get("resourceId") or {}).get("videoId")
                if vid:
                    rows.append(
                        {
                            "item_id": item.get("id"),
                            "video_id": vid,
                            "position": snippet.get("position"),
                        }
                    )
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        if scannable:
            items[playlist_id] = rows
    return items, unscannable


def _members(items: dict[str, list[dict]]) -> dict[str, set[str]]:
    return {pid: {row["video_id"] for row in rows} for pid, rows in items.items()}


def fetch_live_membership(service, playlist_ids) -> tuple[dict[str, set[str]], set[str]]:
    """``fetch_live_items`` reduced to ({playlistId: set(videoId)}, {unscannable id})."""
    items, unscannable = fetch_live_items(service, playlist_ids)
    return _members(items), unscannable


def _is_scheduled(status: dict, now: datetime) -> bool:
    """True for a private video whose ``publishAt`` is still in the future."""
    raw = status.get("publishAt")
    if status.get("privacyStatus") != "private" or not raw:
        return False
    try:
        when = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when > now


def fetch_video_privacy(service, video_ids, *, now: Optional[datetime] = None) -> dict[str, str]:
    """{videoId: privacyStatus} via ``videos.list`` (batches of 50, read-only); an id
    the API does not return is deleted and absent. A private video with a future
    ``publishAt`` (a scheduled upload, already filed by the uploader) reports
    ``"scheduled"`` instead -- never nonpublic, never pruned. Resolving NOTHING for a
    non-empty request is a broken lookup, not a channel of deleted videos: raises."""
    now = now or datetime.now(tz=timezone.utc)
    ids = list(dict.fromkeys(video_ids))
    status: dict[str, str] = {}
    for i in range(0, len(ids), _VIDEO_BATCH):
        resp = (
            service.videos()
            .list(part="status", id=",".join(ids[i : i + _VIDEO_BATCH]), maxResults=_VIDEO_BATCH)
            .execute()
        )
        for item in resp.get("items", []) or []:
            st = item.get("status") or {}
            privacy = st.get("privacyStatus")
            if item.get("id") and privacy:
                status[item["id"]] = "scheduled" if _is_scheduled(st, now) else privacy
    if ids and not status:
        raise RuntimeError(f"videos.list resolved none of {len(ids)} ids -- refusing to prune")
    return status


def _entry_reasons(title, video_id, desired, status) -> list[str]:
    """Every prune class one playlist entry falls in (``_PRUNE_ORDER`` order)."""
    state, want = status.get(video_id), desired.get(video_id)
    found = {
        "deleted": state is None,
        "unsorted": title == PL_UNSORTED,
        "misfiled": bool(
            state == "public"
            and want
            and title not in want
            and title != PL_UNSORTED
            and title not in SUPERLONG_PLAYLISTS
            # a usa video keeps every state playlist it is live in
            and not (title in _STATE_TITLES and _USA_TITLES.intersection(want))
        ),
        "nonpublic": state in _NONPUBLIC,
    }
    return [reason for reason in _PRUNE_ORDER if found[reason]]


def compute_prune_plan(
    items_by_title, desired, status, *, inserts=(), include_nonpublic=False
) -> dict:
    """Pure removal planner over live ``{item_id, video_id, position}`` rows per playlist
    title (scannable only), ``compute_plan``'s ``desired``, ``fetch_video_privacy``'s
    ``status`` and this run's planned ``inserts``. Returns ``{deletes, skipped}``; a skip's
    ``why`` is ``pending_insert`` (a desired membership is neither live nor inserted this
    run) or ``last_managed_playlist`` (a public video would be filed nowhere; Unsorted
    does not count as filed) or ``scheduled`` (a scheduled upload's entries are never
    removed, whatever class they fall in)."""
    filed: dict[str, set[str]] = {}
    for title, rows in items_by_title.items():
        for row in rows:
            filed.setdefault(row["video_id"], set()).add(title)
    for ins in inserts:
        filed.setdefault(ins["video_id"], set()).add(ins["playlist_title"])
    candidates = []
    for title, rows in items_by_title.items():
        if title not in MANAGED_PLAYLISTS:
            continue
        for row in rows:
            reasons = _entry_reasons(title, row["video_id"], desired, status)
            reasons = [r for r in reasons if r != "nonpublic" or include_nonpublic]
            if reasons:
                candidates.append((_PRUNE_ORDER.index(reasons[0]), title, row, reasons[0]))
    candidates.sort(key=lambda c: (c[0], c[1], c[2].get("position") or 0))
    deletes, skipped, removed = [], [], {}
    for _, title, row, reason in candidates:
        vid = row["video_id"]
        entry = {
            "playlist_title": title,
            "item_id": row["item_id"],
            "video_id": vid,
            "position": row.get("position"),
            "reason": reason,
        }
        why = None
        if status.get(vid) == "scheduled":
            why = "scheduled"
        elif reason == "misfiled" and not set(desired[vid]) <= filed.get(vid, set()):
            why = "pending_insert"
        elif status.get(vid) == "public":
            left = filed.get(vid, set()) - {PL_UNSORTED, title} - removed.get(vid, set())
            why = None if left else "last_managed_playlist"
        if why:
            skipped.append({**entry, "why": why})
        else:
            deletes.append(entry)
            removed.setdefault(vid, set()).add(title)
    return {"deletes": deletes, "skipped": skipped}


def compute_audit(items_by_title, desired, status, unrouted, manifest_videos, inserts, prune):
    """Read-only health counts, each independent of prune precedence."""
    counts = dict.fromkeys(_PRUNE_ORDER, 0)
    for title, rows in items_by_title.items():
        for row in rows:
            for reason in _entry_reasons(title, row["video_id"], desired, status):
                counts[reason] += 1
    live = {t: {r["video_id"] for r in rows} for t, rows in items_by_title.items()}

    def both(a, b):
        return live.get(a, set()) & live.get(b, set())

    filed = set().union(*(v for t, v in live.items() if t != PL_UNSORTED))
    return {
        "unsorted_items": len(items_by_title.get(PL_UNSORTED, [])),
        "deleted_entries": counts["deleted"],
        "nonpublic_entries": counts["nonpublic"],
        "misfiled_entries": counts["misfiled"],
        "dual_shorts": len(both(PL_IRELAND_SHORTS, PL_USA_SHORTS)),
        "highlights_in_drives": len(both(PL_IRELAND_HIGHLIGHTS, PL_IRELAND_DRIVES))
        + len(both(PL_USA_HIGHLIGHTS, PL_USA_DRIVES)),
        "unrouted": len(unrouted),
        "public_orphans": sum(
            1
            for v in manifest_videos
            if v.get("privacy_status") == "public" and v["id"] not in filed
        ),
        "planned_inserts": len(inserts),
        "planned_deletes": len(prune["deletes"]),
    }


def apply_inserts(service, plan: dict, *, max_inserts: int) -> dict:
    """Execute up to ``max_inserts`` of ``plan["inserts"]`` as ``playlistItems.insert``
    calls — the only mutation this module ever performs. Each candidate carries a
    resolved ``playlist_id`` (``reconcile`` enriches it there); ``plan["live_by_id"]``
    is the id-keyed membership already fetched by the caller, re-checked here
    immediately before each insert so a re-run is idempotent: anything already a
    member is skipped, never re-inserted."""
    candidates = plan.get("inserts", [])[:max_inserts]
    live_by_id = plan.get("live_by_id", {})
    applied = skipped = 0
    for candidate in candidates:
        playlist_id, video_id = candidate["playlist_id"], candidate["video_id"]
        if video_id in live_by_id.get(playlist_id, set()):
            skipped += 1
            continue
        service.playlistItems().insert(
            part="snippet",
            body={
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {"kind": "youtube#video", "videoId": video_id},
                }
            },
        ).execute()
        applied += 1
    return {"applied": applied, "skipped": skipped}


def apply_deletes(service, deletes, ledger: Path) -> dict:
    """``playlistItems().delete`` per row, its ledger line flushed BEFORE the call so
    ``undo_deletes`` can restore a crashed run; a failed call is logged, the run goes on."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    applied = errors = 0
    with ledger.open("a", encoding="utf-8") as fh:
        for row in deletes:
            stamped = {**row, "at": datetime.now(timezone.utc).isoformat()}
            fh.write(json.dumps(stamped, ensure_ascii=False) + "\n")
            fh.flush()
            try:
                service.playlistItems().delete(id=row["item_id"]).execute()
            except Exception as exc:
                errors += 1
                logger.error(
                    "prune: delete of item %s (video %s, %s) failed: %s",
                    row["item_id"],
                    row["video_id"],
                    row["playlist_title"],
                    exc,
                )
                continue
            applied += 1
    return {"applied": applied, "errors": errors}


def undo_deletes(service, ledger: Path) -> dict:
    """Re-insert every ledgered entry at its recorded position (ascending, so later
    positions land correctly). A video already back in its playlist -- a second undo,
    or a ledgered delete that never went through -- is skipped, never duplicated."""
    rows = [
        json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    live, _ = fetch_live_membership(service, sorted({r["playlist_id"] for r in rows}))
    restored = skipped = errors = 0
    for row in sorted(rows, key=lambda r: (r["playlist_id"], r.get("position") or 0)):
        if row["video_id"] in live.get(row["playlist_id"], set()):
            skipped += 1
            continue
        snippet = {
            "playlistId": row["playlist_id"],
            "resourceId": {"kind": "youtube#video", "videoId": row["video_id"]},
        }
        if row.get("position") is not None:
            snippet["position"] = row["position"]
        try:
            service.playlistItems().insert(part="snippet", body={"snippet": snippet}).execute()
        except Exception as exc:
            errors += 1
            logger.error(
                "undo: re-insert of %s into %s failed: %s",
                row["video_id"],
                row["playlist_title"],
                exc,
            )
            continue
        restored += 1
    return {"restored": restored, "skipped": skipped, "errors": errors}


def reconcile(
    service,
    config,
    *,
    apply: bool,
    now: Optional[datetime] = None,
    prune: bool = False,
    audit: bool = False,
    max_deletes: int = 150,
    include_nonpublic: Optional[bool] = None,
    ledger: Optional[Path] = None,
) -> dict:
    """Glue: fetch the channel + live target-playlist membership (ONE
    ``playlists.list`` + one paginated ``playlistItems.list`` scan per target,
    never repeated), ``compute_plan``, then either report it (dry-run) or execute
    it via ``apply_inserts``. Returns ``{planned, applied, skipped, unroutable, unrouted, ...}``.

    ``prune``/``audit`` also scan Unsorted and add one ``videos.list`` pass (the nightly
    call makes neither); ``audit`` is read-only. On ``apply`` inserts run first, then at
    most ``max_deletes`` deletes, each ledgered to ``ledger`` (required) first.
    ``include_nonpublic=None`` defers to ``config.prune_include_nonpublic``."""
    if prune and apply and not audit and ledger is None:
        raise ValueError("prune with apply needs a ledger path")
    now = now or datetime.now(tz=timezone.utc)
    if include_nonpublic is None:
        include_nonpublic = bool(getattr(config, "prune_include_nonpublic", False))
    manifest = build_channel_manifest(service, now=now)
    overrides = load_overrides(resolve_overrides_path(getattr(config, "overrides_path", "")))

    wide = prune or audit
    title_to_id = resolve_playlist_ids(
        service,
        MANAGED_PLAYLISTS if wide else ALL_TARGET_PLAYLISTS,
        optional=_OPTIONAL_MANAGED if wide else SUPERLONG_PLAYLISTS,
    )
    items_by_id, unscannable_ids = fetch_live_items(service, title_to_id.values())
    live_by_id = _members(items_by_id)
    live_membership = {t: live_by_id.get(i, set()) for t, i in title_to_id.items()}
    # A playlist whose id landed in ``unscannable_ids`` had a mid-pagination 404 --
    # its membership is unknown, not empty, so it must be excluded from planning
    # entirely rather than treated as having no members (that would insert
    # duplicates of videos already live on it).
    unscannable_titles = {t for t, i in title_to_id.items() if i in unscannable_ids}

    plan = compute_plan(
        manifest["videos"],
        live_membership,
        overrides,
        max_inserts=config.max_inserts_per_run,
        newest_first=config.newest_first,
        backfill_since=config.backfill_since,
        unscannable_titles=unscannable_titles,
    )
    summary = {
        "planned": len(plan["inserts"]),
        "applied": 0,
        "skipped": 0,
        "unroutable": len(plan["unrouted"]),
        "unrouted": plan["unrouted"],
        "skipped_unscannable": len(unscannable_ids),
    }
    for row in plan["unrouted"][:UNROUTED_LOG_CAP]:
        logger.warning(
            "UNROUTED: %s kind=%s title=%r \u2014 no playlist route (title region "
            "unresolved and not live in any Shorts/region playlist)",
            row["video_id"],
            row["kind"],
            row["title"],
        )
    if len(plan["unrouted"]) > UNROUTED_LOG_CAP:
        logger.warning("\u2026 and %d more unrouted", len(plan["unrouted"]) - UNROUTED_LOG_CAP)
    prune_plan = None
    if wide:
        items_by_title = {t: items_by_id[i] for t, i in title_to_id.items() if i in items_by_id}
        status = fetch_video_privacy(
            service, [r["video_id"] for rows in items_by_title.values() for r in rows], now=now
        )
        prune_plan = compute_prune_plan(
            items_by_title,
            plan["desired"],
            status,
            inserts=plan["inserts"],
            include_nonpublic=include_nonpublic,
        )
    if audit:
        summary["audit"] = compute_audit(
            items_by_title,
            plan["desired"],
            status,
            plan["unrouted"],
            manifest["videos"],
            plan["inserts"],
            prune_plan,
        )
        return summary
    if apply:
        for candidate in plan["inserts"]:
            candidate["playlist_id"] = title_to_id[candidate["playlist_title"]]
        plan["live_by_id"] = live_by_id
        result = apply_inserts(service, plan, max_inserts=config.max_inserts_per_run)
        summary["applied"], summary["skipped"] = result["applied"], result["skipped"]
    if prune:
        rows = prune_plan["deletes"][: max(0, max_deletes)]
        for row in rows:
            row["playlist_id"] = title_to_id[row["playlist_title"]]
        done = apply_deletes(service, rows, ledger) if apply else {"applied": 0, "errors": 0}
        summary["prune"] = {
            "planned": len(rows),
            "applied": done["applied"],
            "skipped": len(prune_plan["skipped"]),
            "deferred": len(prune_plan["deletes"]) - len(rows),
            "errors": done["errors"],
            "rows": rows,
        }
    return summary
