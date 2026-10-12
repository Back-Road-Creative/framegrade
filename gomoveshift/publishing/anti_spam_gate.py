"""Batch anti-spam gate, upstream of the spool loop.

``check_batch`` is a pure function over the records a run is about to spool
-- no I/O, no queue access -- so it can see cross-record properties no
single-record generator call can.
``cli.handlers.enqueue_plan.cmd_enqueue_plan`` calls it once, right after
the rolling-ceiling gate and before anything is written to the queue: a
violating batch is refused whole, nothing spooled, the same "nothing was
written to the queue" shape the media/privacy/ceiling gates already use.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .models import PublishRecord

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


def check_batch(
    records: Sequence[PublishRecord], *, per_location_cap: Optional[int]
) -> List[Violation]:
    """Every anti-spam violation across the whole batch, all at once -- the
    same refuse-whole shape as ``_rolling_ceiling_violations``: a batch that
    fails any rule spools nothing, and every offending index is reported in
    one pass rather than fixed one at a time."""
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

    return violations
