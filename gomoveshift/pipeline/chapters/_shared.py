"""Shared constants and utilities for the chapters package."""

import re

from ..geo_constants import TOURISM_POI_KEYWORDS

# Canonical display names for well-known scenic routes (global)
_KNOWN_DISPLAY_NAMES: dict[str, str] = {
    "ring of kerry": "Ring of Kerry",
    "wild atlantic way": "Wild Atlantic Way",
    "healy pass": "Healy Pass",
    "caha pass": "Caha Pass",
    "conor pass": "Conor Pass",
    "sky road": "Sky Road",
    "slea head": "Slea Head",
    "vee road": "Vee Road",
    "blue ridge parkway": "Blue Ridge Parkway",
    "natchez trace parkway": "Natchez Trace Parkway",
    "pacific coast highway": "Pacific Coast Highway",
    "overseas highway": "Overseas Highway",
    "scenic highway 17": "Scenic Highway 17",
}

# Real settlements whose whole name collides with a tourism keyword as a
# WHOLE WORD (so the word-boundary test alone can't save them). Checked before
# the keyword loop so a true endpoint is never evicted. Keys are the exact
# lowercased settlement name (state suffix stripped by the caller-side split).
_SETTLEMENT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "red bank",  # NJ, TN — "bank"
        "center",  # TX — "center"
        "center point",  # AL — "center"
        "centre",  # AL — "centre"
        "golf",  # IL — "golf"
        "dollar bay",  # MI — "dollar"
        "bar harbor",  # ME — "bar"
        "bar nunn",  # WY — "bar"
    }
)


def is_valid_chapter_placename(name: str) -> bool:
    """Check if a name is suitable for a YouTube chapter.

    Real towns/villages pass. Tourism businesses and attractions are rejected.
    Also rejects possessive names (e.g., "Sam's Diner", "Fat Patties",
    "Murphy's", "O'Connor's") that look like business names rather than places.
    """
    # Normalize a curly apostrophe (U+2019) to ASCII so possessive detection is
    # uniform ("Murphy’s" -> "Murphy's").
    name = name.replace("’", "'")
    name_lower = name.lower()
    # Known settlements whose name embeds a keyword as a whole word pass before
    # the keyword loop can evict them. Match the exact name, dropping a trailing
    # ", ST" / ", County X" so a geocoder-qualified label still resolves.
    if name_lower.split(",")[0].strip() in _SETTLEMENT_ALLOWLIST:
        return True
    # Reject if any tourism keyword appears as a whole word/token. A bare
    # substring test wrongly rejected settlements that merely embed a keyword
    # ("Centerville" -> "center", "Farmington" -> "farm", "Banks" -> "bank").
    # The lookarounds treat alphanumerics as word chars, so a keyword ending in
    # punctuation (e.g. "inc.") still matches at end-of-string, and multi-word
    # keywords ("ice cream", "bar & grill") match as whole phrases.
    for keyword in TOURISM_POI_KEYWORDS:
        if re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", name_lower):
            return False
    # Reject possessive names that look like businesses. A possessive is an
    # embedded "X's " token OR a name-final "'s" (e.g. "Murphy's", "O'Connor's").
    # Real possessive places end with a geographic feature (Mountain, Ferry, …).
    if "'s " in name or name.endswith("'s"):
        _geo_suffixes = {
            "mountain",
            "hill",
            "creek",
            "river",
            "valley",
            "ferry",
            "landing",
            "point",
            "head",
            "bay",
            "harbor",
            "harbour",
            "gap",
            "pass",
            "ridge",
            "falls",
            "lake",
            "island",
            "fork",
            "springs",
            "crossing",
            "corner",
            "corners",
            "bluff",
        }
        _last_word = name.split()[-1].lower()
        if _last_word not in _geo_suffixes:
            return False
    return True
