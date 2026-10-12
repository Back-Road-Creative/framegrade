"""Single source of truth for series-suffix stripping.

The ``" — Part N"`` series suffix belongs in titles only — it must never
bleed into discovery surfaces (tags, hashtags) or region/country parsing.
Both the regex and the helper previously lived byte-for-byte in
``longform_metadata``, ``social_generator``, and ``region_resolver``; they
are consolidated here so the three call sites can never drift.

Only the *trailing* form is stripped; mid-string occurrences of "Part"
are preserved (e.g. ``"Some Part of Town"``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Both shapes: the bare ``" — Part N"`` and the ``" — Part N of M"`` that
# ``title_builder.build_superlong_part_title`` emits. The ``of M`` tail is
# NOT optional decoration — without it the anchor lands right after the
# digits, the suffix survives, and every consumer treats it as part of the
# location (region derivation off a multi-part title then matches no
# country/state/county tail at all).
_SERIES_SUFFIX_RE = re.compile(r"\s*—\s*Part\s+\d+(?:\s+of\s+\d+)?\s*$")


def _strip_series_suffix(s: str) -> str:
    """Strip a trailing ``" — Part N"`` / ``" — Part N of M"`` series suffix."""
    return _SERIES_SUFFIX_RE.split(s, maxsplit=1)[0]


# An operator ``--per-lf-location`` value like ``"To Clonakilty, County
# Cork"`` is an OST-burn TITLE, not a place noun. This leading title word is
# NOT part of the label's grammar (unlike the curated verbs
# ``longform_metadata._strip_leading_verb`` strips) — it is the operator's own
# phrasing, kept for the burned OST/title and stripped everywhere else.
_TITLE_PREFIX_RE = re.compile(
    r"^(?:to|through|up\s+around|around|the\s+road\s+to|toward(?:s)?)\s+",
    re.IGNORECASE,
)


def is_title_prefixed(label: str) -> bool:
    """True when ``label`` opens with a curated title word.

    The single predicate for "this label's leading word is the operator's
    own title phrasing, not a place noun" -- every other code path that
    needs to answer the same question (the label-quality WARN gate, the
    late grammar gate) calls this instead of re-declaring its own copy of
    the word list. The list was once copied into the quality gate and,
    narrower, into the grammar gate; three hand-maintained regexes for one
    word-class is
    exactly the mechanism that let the same prefix-word bug resurface.
    """
    return bool(_TITLE_PREFIX_RE.match((label or "").strip()))


@dataclass(frozen=True)
class LfLabel:
    """A per-LF label parsed into its title vs. place-noun components.

    ``display`` is the label verbatim — the OST burn and video title, where
    the operator's title shape is wanted. ``place`` strips a leading curated
    title word (``To``/``Through``/``Up around``/``Around``/``The road to``/
    ``Toward(s)``) so every other noun slot (hook, shorts titles, captions,
    tags, hashtags, ``social_location``) never renders the operator's verb.
    ``region`` is the tail after the last comma of ``place``, when present.
    A plain place label (no leading title word) leaves ``place`` identical to
    ``display`` — parsing is a no-op for the common GPS-derived case.
    """

    display: str
    place: str
    region: str


def parse_lf_label(label: str) -> LfLabel:
    """Parse an operator/derived per-LF label into ``LfLabel``."""
    display = (label or "").strip()
    place = _TITLE_PREFIX_RE.sub("", display, count=1).strip()
    region = place.rsplit(",", 1)[-1].strip() if "," in place else ""
    return LfLabel(display=display, place=place, region=region)
