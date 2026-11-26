"""Batch anti-spam gate, upstream of the spool loop.

``check_batch`` is a pure function over the records a run is about to spool
-- no I/O, no queue access -- so it can see cross-record properties no
single-record generator call can.
``cli.handlers.enqueue_plan.cmd_enqueue_plan`` calls it once, right after
the rolling-ceiling gate and before anything is written to the queue: a
violating batch is refused whole, nothing spooled, the same "nothing was
written to the queue" shape the media/privacy/ceiling gates already use.

Two scopes: every rule but two is within-batch. ``cross-batch-repeat`` compares
each record's platform text against ``recent_texts`` -- the captions/titles
already spooled in the last 60 days, loaded by :func:`recent_platform_texts`
-- and refuses a text that opens with the same 6 words as one of them.
``neighbour-place`` lines the batch's YouTube Shorts up with ``queued_youtube`` (the
queue's other active YouTube rows, :func:`queued_youtube_records`) by scheduled slot and
refuses a Short whose title names the same place as the YouTube Short just before or
after it, whatever long-form or project either came from (the generator only sees the
Short before it in one long-form). ``repeated-title``, in the same pass, refuses the very
same title twice in a row or one title body (stamp aside) over three Shorts in a row.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import AbstractSet, Dict, List, Mapping, Optional, Sequence, Tuple

from .models import PublishRecord

logger = logging.getLogger(__name__)

#: Words two texts must share, from the start, to count as the same opener.
CROSS_BATCH_PREFIX_WORDS = 6
#: How far back spooled text counts as "already posted".
CROSS_BATCH_WINDOW_DAYS = 60

#: The single source of truth for the 100 retired legacy title templates
#: (``TITLE_TEMPLATES`` / ``EXTENDED_TITLE_TEMPLATES`` / ``_pick_tiktok_title``).
#: ``tests/fixtures/legacy_title_pool.py`` re-exports this tuple as a list --
#: production code must not import from ``tests/``, so the pool lives here and
#: the fixture is a thin handle over it.
LEGACY_TITLE_POOL_RAW: Tuple[str, ...] = (
    "POV: You're the Driver — {location} 🚗",
    "Rolling Through {location} 🚗",
    "POV: Driving Through {location}",
    "The {terrain_desc} of {location}",
    "Would You Drive This Road? — {location}",
    "POV: {terrain_desc} of {location} 🛤️",
    "Exploring {location} by Car 🚗",
    "Just Driving — {location}",
    "This Is Why I Drive — {location}",
    "Driving {location} 🚗",
    "{location} Drive",
    "On the Road: {location}",
    "Dashcam: {location} 🚗",
    "POV Drive — {location}",
    "{location} — POV 🚗",
    "Drive With Me: {location}",
    "{terrain_desc} of {location} 🚗",
    "Passing Through {location} 🚗",
    "Leaving {location}",
    "Arriving in {location}",
    "Driving Into {location}",
    "Headed Through {location}",
    "{location} 🚗",
    "Just Driving {location}",
    "{location} — Dashcam",
    "On the Road — {location}",
    "{location} — No Commentary",
    "Somewhere in {location} 🚗",
    "Where Does This Road Go? — {location}",
    "Have You Driven {location}?",
    "Have You Been to {location}?",
    "Would You Take This Route? — {location}",
    "Know This Road? — {location}",
    "{terrain_desc} Drive — {location}",
    "{terrain_desc} — {location} 🚗",
    "Driving the {terrain_desc} of {location}",
    "The {terrain_desc} — {location} 🛤️",
    "Road Trip: {location} 🚗",
    "Cruising Through {location}",
    "No Talking, Just Driving — {location}",
    "Real Roads — {location} 🚗",
    "Windshield View — {location}",
    "Dashcam POV — {location}",
    "{location} From Behind the Wheel",
    "Driving {location} — No Music",
    "Along the {terrain_desc} — {location}",
    "Heading Out of {location} 🚗",
    "{location} Road Trip 🚗",
    "Riding Through {location}",
    "The Drive Through {location}",
    "Dashcam Diaries — {location}",
    "POV: {location} by Car 🚗",
    "Next Stop: {location}",
    "Behind the Wheel — {location} 🚗",
    "{location} — Full Drive",
    "A Drive Through {location} 🛤️",
    "This Road in {location}",
    "{terrain_desc} POV — {location}",
    "Come Drive {location} With Me",
    "{location} — Raw Dashcam",
    "Windows Down — {location} 🚗",
    "POV: On the {terrain_desc} of {location} 🚗",
    "Camera Rolling — {location}",
    "Nothing but Road — {location}",
    "This Is {location}",
    "Driving Solo — {location}",
    "Open Road: {location} 🚗",
    "Let the Road Talk — {location}",
    "4K POV — {location}",
    "The {terrain_desc} Near {location}",
    "POV From {location} 🚗",
    "Slow Drive Through {location}",
    "Another Day, Another Drive — {location}",
    "Exploring the {terrain_desc} of {location} 🚗",
    "Rolling Down the {terrain_desc} — {location}",
    "On a Drive Through {location}",
    "Car Cam: {location} 🚗",
    "See {location} From the Car",
    "The Road Ahead — {location}",
    "Driving {road} — {location}",
    "{road}: {location} 🚗",
    "{county} Drive — {location}",
    "Taking {road} Through {location}",
    "{road} — {location} 🚗",
    "Following {road} — {location}",
    "{road} POV — {location}",
    "POV: {location} 🚗",
    "This road in {location}",
    "Driving through {location}",
    "{location} {terrain_desc}",
    "Would you drive here?",
    "POV drive 🚗",
    "Driving {location}",
    "POV: the {terrain_desc}",
    "No talking, just driving",
    "Cruising {location} 🚗",
    "This is {location}",
    "Road trip {location}",
    "{terrain_desc} of {location}",
    "Just driving {location}",
)

#: Four title shapes the channel used BEFORE the pool above existed (some
#: already-public low-view shorts carry one of them). Declared beside
#: the pool, never inside it: ``LEGACY_TITLE_POOL`` feeds the anti-spam gate's
#: exact-title check and must keep meaning "the retired 100". These exist so
#: ``library-hygiene`` can name a series for those shorts instead of skipping them.
PRE_POOL_TITLE_SHAPES: Tuple[str, ...] = (
    "4K Dashcam: {location}",
    "4K Road Trip — {location}",
    "First Time Driving {location}",
    "Through {location} in 4K",
)


def _normalize_exact(text: str) -> str:
    return text.strip().lower()


LEGACY_TITLE_POOL: frozenset = frozenset(_normalize_exact(t) for t in LEGACY_TITLE_POOL_RAW)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize_words(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _bare_title(title: str) -> str:
    """The title without the hashtags ``plan_builder`` appends to it -- a
    legacy pool title must not pass the pool by wearing them, and they must
    not lift a two-word title over the prefix rule's floor. Function-local
    import: ``pipeline`` imports from ``publishing`` (see models.py)."""
    from ..pipeline.text_format import split_trailing_hashtags

    return split_trailing_hashtags(title)[0]


def title_prefix_key(title: str) -> Optional[Tuple[str, ...]]:
    """The shared-prefix key: the title's first 3 words (hashtags stripped,
    lowercased ``[a-z0-9]+`` tokens), or ``None`` when it has fewer than 3."""
    words = _normalize_words(_bare_title(title or ""))
    return tuple(words[:3]) if len(words) >= 3 else None


def shared_prefix_groups(titles: Sequence[str]) -> Dict[Tuple[str, ...], List[int]]:
    """Group title positions by :func:`title_prefix_key`; only groups of 2+.

    The single definition of "shares a prefix" -- ``check_batch`` and the title
    composer both use it, so the generator and the gate cannot drift apart."""
    groups: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for index, title in enumerate(titles):
        key = title_prefix_key(title)
        if key is not None:
            groups[key].append(index)
    return {key: indices for key, indices in groups.items() if len(indices) >= 2}


@dataclass(frozen=True)
class Violation:
    index: int
    platform: Optional[str]
    rule: str
    detail: str


def _platform_texts(record: PublishRecord) -> Dict[str, str]:
    """Non-empty, stripped text per platform for the caption-collision rule.

    YouTube's comparable text is ``description`` (its title is checked
    separately by the legacy-title/shared-prefix rules, and its
    description -- not its title -- is the field that plays the same role
    IG's caption/FB's description/TikTok's caption text play)."""
    payloads = record.payloads
    texts: Dict[str, str] = {}
    if payloads.youtube and payloads.youtube.description.strip():
        texts["youtube"] = payloads.youtube.description.strip()
    if payloads.instagram and payloads.instagram.caption.strip():
        texts["instagram"] = payloads.instagram.caption.strip()
    if payloads.facebook and payloads.facebook.description.strip():
        texts["facebook"] = payloads.facebook.description.strip()
    if payloads.tiktok and payloads.tiktok.caption.strip():
        texts["tiktok"] = payloads.tiktok.caption.strip()
    return texts


def _location_key(record: PublishRecord) -> Optional[str]:
    """The per-location grouping key for the location-cap rule -- the first
    of ``sampled_regions``, the same field
    ``_register_enqueued_shorts_in_tracking`` already treats as the
    record's location label. ``None`` (no sampled regions -- a GPS-less/
    legacy record) is excluded from the count, matching the region
    invariants' own no-op rule for GPS-less/legacy content."""
    regions = record.sampled_regions
    return regions[0] if regions else None


def text_opener_key(text: str) -> Tuple[str, ...]:
    """The cross-batch comparison key: the text's first 6 normalized words
    (hashtags are trailing, so they never reach it)."""
    return tuple(_normalize_words(text)[:CROSS_BATCH_PREFIX_WORDS])


def _record_texts(record: PublishRecord) -> Dict[str, List[str]]:
    """Every comparable text per platform; YouTube carries its title too."""
    texts = {platform: [text] for platform, text in _platform_texts(record).items()}
    youtube = record.payloads.youtube
    if youtube and _bare_title(youtube.title).strip():
        texts.setdefault("youtube", []).insert(0, _bare_title(youtube.title).strip())
    return texts


def recent_text_entries(
    conn: sqlite3.Connection,
    *,
    since_days: int = CROSS_BATCH_WINDOW_DAYS,
    now: Optional[datetime] = None,
    exclude_projects: AbstractSet[str] = frozenset(),
    exclude_shorts: AbstractSet[Tuple[str, int]] = frozenset(),
) -> List[Tuple[str, int, str, str]]:
    """``(project, short_index, platform, text)`` for every text already spooled in the
    last ``since_days``, in queue order.

    The queue stores no text itself; each delivery row names the sidecar it
    was enqueued from, so the text is read back from that plan file. Shadow
    and cancelled rows never publish and are skipped, as are rows of
    ``exclude_projects`` (the batch being checked: re-running the same plan
    must not repeat against itself), rows of the ``exclude_shorts``
    ``(project, short_index)`` pairs and sidecars that are gone or unreadable.
    """
    from .plan_io import load_publish_plan

    cutoff = ((now or datetime.now(timezone.utc)) - timedelta(days=since_days)).isoformat()
    rows = conn.execute(
        "SELECT DISTINCT project_name, short_index, platform, sidecar_path FROM deliveries "
        "WHERE sidecar_path IS NOT NULL AND shadow=0 AND status != 'cancelled' "
        "AND created_at >= ?",
        (cutoff,),
    ).fetchall()
    plans: Dict[str, Optional[Mapping[Tuple[str, int], PublishRecord]]] = {}
    out: List[Tuple[str, int, str, str]] = []
    for row in rows:
        if row["project_name"] in exclude_projects:
            continue
        if (row["project_name"], row["short_index"]) in exclude_shorts:
            continue
        path = row["sidecar_path"]
        if path not in plans:
            try:
                plan = load_publish_plan(path)
                plans[path] = {(r.project_name, r.short_index): r for r in plan.shorts}
            except Exception as exc:  # missing/malformed sidecar: nothing to compare
                logger.warning("anti-spam: skipping unreadable sidecar %s: %s", path, exc)
                plans[path] = None
        by_key = plans[path]
        record = by_key.get((row["project_name"], row["short_index"])) if by_key else None
        if record is None:
            continue
        for text in _record_texts(record).get(row["platform"], []):
            out.append((row["project_name"], row["short_index"], row["platform"], text))
    return out


def recent_platform_texts(
    conn: sqlite3.Connection,
    *,
    since_days: int = CROSS_BATCH_WINDOW_DAYS,
    now: Optional[datetime] = None,
    exclude_projects: AbstractSet[str] = frozenset(),
    exclude_shorts: AbstractSet[Tuple[str, int]] = frozenset(),
) -> Dict[str, List[str]]:
    """Platform -> captions/titles already spooled in the last ``since_days``
    (:func:`recent_text_entries`, which says what is skipped)."""
    out: Dict[str, List[str]] = defaultdict(list)
    for _project, _index, platform, text in recent_text_entries(
        conn,
        since_days=since_days,
        now=now,
        exclude_projects=exclude_projects,
        exclude_shorts=exclude_shorts,
    ):
        if text not in out[platform]:
            out[platform].append(text)
    return dict(out)


def _utc(at: datetime) -> datetime:
    return at if at.tzinfo else at.replace(tzinfo=timezone.utc)


def queued_youtube_records(
    conn: sqlite3.Connection,
    *,
    exclude_shorts: AbstractSet[Tuple[str, int]] = frozenset(),
) -> List[Tuple[datetime, PublishRecord]]:
    """``(slot, record)`` for every active (not terminal, not shadow) YouTube delivery
    with a slot, read back from the sidecar its row names (gone or unreadable sidecars
    and the ``exclude_shorts`` pairs are skipped): the queue's side of ``neighbour-place``."""
    from .plan_io import load_publish_plan
    from .queue import TERMINAL_STATUSES

    terminal = ", ".join("?" for _ in TERMINAL_STATUSES)
    rows = conn.execute(
        "SELECT project_name, short_index, sidecar_path, scheduled_at FROM deliveries "  # noqa: S608
        f"WHERE platform = 'youtube' AND shadow = 0 AND status NOT IN ({terminal}) "
        "AND sidecar_path IS NOT NULL AND scheduled_at IS NOT NULL",
        TERMINAL_STATUSES,
    ).fetchall()
    plans: Dict[str, Optional[Mapping[Tuple[str, int], PublishRecord]]] = {}
    out: List[Tuple[datetime, PublishRecord]] = []
    for row in rows:
        key = (row["project_name"], int(row["short_index"]))
        if key in exclude_shorts:
            continue
        path = row["sidecar_path"]
        if path not in plans:
            try:
                plans[path] = {
                    (r.project_name, r.short_index): r for r in load_publish_plan(path).shorts
                }
            except Exception as exc:  # missing/malformed sidecar: nothing to compare
                logger.warning("anti-spam: skipping unreadable sidecar %s: %s", path, exc)
                plans[path] = None
        record = (plans[path] or {}).get(key)
        try:
            at = _utc(datetime.fromisoformat(str(row["scheduled_at"]).replace("Z", "+00:00")))
        except ValueError:
            continue
        if record is not None:
            out.append((at, record))
    return out


def _youtube_slot(record: PublishRecord) -> Optional[datetime]:
    if record.payloads.youtube is None or "youtube" not in record.routes.selected:
        return None
    sched = record.schedule
    return _utc(sched.scheduled_at_utc_by_platform.get("youtube") or sched.scheduled_at_utc)


def _place_candidates(record: PublishRecord) -> List[str]:
    """The places a record's title may name: its recorded title place, its measured
    towns and counties (``measured_places``), its regions and the bracket label."""
    from ..pipeline.social_generator import measured_places

    title = ((record.place_variety or {}).get("title") or {}).get("body") or ""
    measured = measured_places(record.sampled_regions, record.sampled_roads, record.route_record)
    regions = [r for region in record.sampled_regions for r in (region, region.split(",")[0])]
    label = str((record.bracket_route or {}).get("label") or "")
    return [title, *(b for k, b in measured if k != "road"), *regions, label]


def title_place(record: PublishRecord) -> str:
    """The place a YouTube Short's title names (lowercased, spaces folded), ``""`` when
    its title names none of the record's own places."""
    youtube = record.payloads.youtube
    text = " ".join(_normalize_words(_bare_title(youtube.title))) if youtube else ""
    for place in _place_candidates(record):
        words = " ".join(_normalize_words(place))
        if words and f" {words} " in f" {text} ":
            return words
    return ""


def _no_alternative(record: PublishRecord, taken: AbstractSet[str]) -> bool:
    """The generator could not have varied this Short's title place: it says so
    (``place_variety`` reason ``no_alternative``), or every measured title place of the
    clip is one of ``taken`` (its neighbours' places)."""
    from ..pipeline.social_generator import REASON_NO_ALTERNATIVE, measured_places

    if ((record.place_variety or {}).get("title") or {}).get("reason") == REASON_NO_ALTERNATIVE:
        return True
    measured = measured_places(record.sampled_regions, record.sampled_roads, record.route_record)
    return all(" ".join(_normalize_words(b)) in taken for k, b in measured if k != "road")


#: The ``"Sep 14, 3rd stretch: "`` stamp a stamped title opens with (``title_stamp``).
_STAMP_RE = re.compile(r"^[a-z]{3} \d{1,2} \d+(?:st|nd|rd|th) stretch ")


def title_body(record: PublishRecord) -> str:
    """A YouTube Short's bare title without its stamp (lowercased words, hashtags off)."""
    youtube = record.payloads.youtube
    words = " ".join(_normalize_words(_bare_title(youtube.title))) if youtube else ""
    return _STAMP_RE.sub("", words + " ").strip()


def _neighbour_place_violations(
    records: Sequence[PublishRecord],
    queued_youtube: Sequence[Tuple[datetime, PublishRecord]],
    now: Optional[datetime],
) -> List[Violation]:
    """``neighbour-place``: the batch's YouTube Shorts slotted after ``now`` and the queue's
    other active YouTube rows, in slot order; a batch Short whose title place equals the
    Short's just before or after it is refused, unless the generator could not vary it
    (:func:`_no_alternative`). Only a Short whose generator recorded its title place
    (``place_variety["title"]``) is judged: a hand title, or a plan written before the
    record, is the operator's or unknown, but still counts as a neighbour."""
    batch_keys = {(r.project_name, r.short_index) for r in records}
    line = []  # (slot, project, short_index, record, batch index or None)
    for index, record in enumerate(records):
        at = _youtube_slot(record)
        if at is not None and (now is None or at > _utc(now)):
            line.append((at, record.project_name, record.short_index, record, index))
    for at, record in queued_youtube:
        if (record.project_name, record.short_index) not in batch_keys:
            line.append((_utc(at), record.project_name, record.short_index, record, None))
    line.sort(key=lambda e: e[:3])
    places = [title_place(e[3]) for e in line]
    bodies = [title_body(e[3]) for e in line]
    titles = [" ".join(_normalize_words(_bare_title(e[3].payloads.youtube.title))) for e in line]
    out: List[Violation] = []
    for n, (_at, _p, _i, record, index) in enumerate(line):
        if index is None or not (record.place_variety or {}).get("title"):
            continue
        near = [m for m in (n - 1, n + 1) if 0 <= m < len(line)]
        # a run of one title body over three Shorts in a row, or the very same title twice: a
        # stamped body rotates and a fact title is unique, so the generator can always vary it
        same = [m for m in near if titles[m] == titles[n]]
        if titles[n] and (same or (len(near) == 2 and all(bodies[m] == bodies[n] for m in near))):
            m = same[0] if same else near[0]
            out.append(
                Violation(
                    index,
                    "youtube",
                    "repeated-title",
                    f"title {bodies[n]!r} repeats the YouTube Short(s) beside it by slot "
                    f"({line[m][1]} #{line[m][2]} at {line[m][0].isoformat()})",
                )
            )
        if not places[n]:
            continue
        sides = [
            (m, word) for m, word in ((n - 1, "previous"), (n + 1, "next")) if 0 <= m < len(line)
        ]
        clash = [(m, word) for m, word in sides if places[m] == places[n]]
        if not clash or _no_alternative(record, {places[m] for m, _ in sides}):
            continue
        m, word = clash[0]
        out.append(
            Violation(
                index,
                "youtube",
                "neighbour-place",
                f"title place {places[n]!r} repeats the {word} YouTube Short by slot "
                f"({line[m][1]} #{line[m][2]} at {line[m][0].isoformat()})",
            )
        )
    return out


def check_batch(
    records: Sequence[PublishRecord],
    *,
    per_location_cap: Optional[int],
    recent_texts: Optional[Mapping[str, Sequence[str]]] = None,
    queued_youtube: Optional[Sequence[Tuple[datetime, PublishRecord]]] = None,
    now: Optional[datetime] = None,
) -> List[Violation]:
    """Every anti-spam violation across the whole batch, all at once -- the
    same refuse-whole shape as ``_rolling_ceiling_violations``: a batch that
    fails any rule spools nothing, and every offending index is reported in
    one pass rather than fixed one at a time. ``neighbour-place`` runs only when
    the caller passes ``queued_youtube`` (``[]`` = an empty queue); batch Shorts
    slotted at or before ``now`` (already out) are left out of it."""
    violations: List[Violation] = []

    for index, record in enumerate(records):
        youtube = record.payloads.youtube
        if youtube and _normalize_exact(_bare_title(youtube.title)) in LEGACY_TITLE_POOL:
            violations.append(
                Violation(
                    index,
                    "youtube",
                    "legacy-title",
                    f"title matches the legacy template pool: {youtube.title!r}",
                )
            )

    youtube_titles = [
        record.payloads.youtube.title if record.payloads.youtube else "" for record in records
    ]
    for prefix, indices in shared_prefix_groups(youtube_titles).items():
        for index in indices:
            others = [i for i in indices if i != index]
            violations.append(
                Violation(
                    index,
                    "youtube",
                    "shared-prefix",
                    f"title shares first-3-word prefix {' '.join(prefix)!r} with "
                    f"index(es) {others}",
                )
            )

    for index, record in enumerate(records):
        texts = _platform_texts(record)
        seen: Dict[str, str] = {}
        for platform, text in texts.items():
            match = next((p for p, t in seen.items() if t == text), None)
            if match is not None:
                violations.append(
                    Violation(
                        index,
                        platform,
                        "caption-collision",
                        f"text is identical to the {match} text: {text[:60]!r}",
                    )
                )
            seen[platform] = text

    if recent_texts:
        recent_keys = {
            platform: {text_opener_key(t): t for t in texts if text_opener_key(t)}
            for platform, texts in recent_texts.items()
        }
        for index, record in enumerate(records):
            for platform, texts in _record_texts(record).items():
                known = recent_keys.get(platform, {})
                for text in texts:
                    match = known.get(text_opener_key(text))
                    if match is not None:
                        violations.append(
                            Violation(
                                index,
                                platform,
                                "cross-batch-repeat",
                                f"text opens with the same {CROSS_BATCH_PREFIX_WORDS} words "
                                f"as one already spooled: {text[:60]!r} ~ {match[:60]!r}",
                            )
                        )
                        break

    if per_location_cap is not None:
        by_location: Dict[str, List[int]] = defaultdict(list)
        for index, record in enumerate(records):
            location = _location_key(record)
            if location is None:
                continue
            by_location[location].append(index)
        for location, indices in by_location.items():
            if len(indices) <= per_location_cap:
                continue
            for index in indices[per_location_cap:]:
                violations.append(
                    Violation(
                        index,
                        None,
                        "location-cap",
                        f"location {location!r} has {len(indices)} short(s) in this "
                        f"batch, cap is {per_location_cap}",
                    )
                )

    if queued_youtube is not None:
        violations.extend(_neighbour_place_violations(records, queued_youtube, now))

    return violations
