"""Playlist resolver for YouTube auto-assignment.

Construction-correct: ``resolve_playlist`` is total — every input maps
to exactly one playlist via the config's rule list or the ``default``.
The uploader has no path that uploads without assigning a playlist;
this resolver makes that contract representable.

Region derivation order:

1. Explicit ``metadata["region"]`` (any case) — operator override.
2. Country derived from the LAST comma-delimited component of
   ``metadata["location"]``, lowercased, with any trailing
   ``" — Part N"`` series suffix stripped.
3. Neither present → no rule key → fall through to ``default``.

Series-suffix stripping mirrors ``social_generator._strip_series_suffix``
so a single-source content surface (title, tags, hashtags, playlist) all
agree on what "country" means for ``"X — Part 1"``.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

# country / US-ness is derived ONCE in the shared ``region_resolver``
# helper. The old ``_derive_country`` duplicate is deleted.
from .longform_metadata import HIGHLIGHTS_TRIM_THRESHOLD
from .region_resolver import derive_region

# Compass prefixes on a stateless Irish district label ("West Cork",
# "North Tipperary"): the remainder is the county, which ``derive_region``
# already knows.
_COMPASS_PREFIXES = ("north", "south", "east", "west", "mid")

# The only countries that have a Shorts playlist. Anything else is "not
# derivable" for a short: it is skipped rather than filed in the default.
_SHORTS_COUNTRIES = frozenset({"ireland", "usa"})


def _district_country(part: str) -> str:
    """``derive_region`` of one label part, retried without a compass prefix."""
    country = derive_region(part)
    if country not in _SHORTS_COUNTRIES:
        head, _, rest = part.partition(" ")
        if rest and head.lower() in _COMPASS_PREFIXES:
            country = derive_region(rest)
    return country


def short_region(sampled_regions: Iterable[str]) -> str:
    """Country key (``"ireland"`` / ``"usa"``) for a short's place labels, or ``""``.

    A published short carries only PLACE labels (``sampled_regions``: the
    clip's own ``"County, State"`` votes — ``"Unicoi, Tennessee"``, or a bare
    stateless Irish district such as ``"West Cork"``), never a country. The
    first label that resolves to a country with a Shorts playlist wins;
    ``derive_region`` does the label parse (so US-ness agrees with LF routing),
    plus one fallback it lacks: a compass-prefixed bare Irish district.
    ``""`` means the country cannot be derived — the caller must not file the
    short under the config default.
    """
    for label in sampled_regions:
        text = (label or "").strip()
        if not text:
            continue
        country = derive_region(text)
        if country not in _SHORTS_COUNTRIES:
            # An Irish county can carry its PROVINCE as the "state"
            # (``"County Kerry, Munster"``): ``derive_region`` reads the last
            # part ("munster") and finds no country, so retry on the first
            # part (the county / district), compass prefix stripped.
            country = _district_country(text.split(",")[0].strip())
        if country in _SHORTS_COUNTRIES:
            return country
    return ""


def resolve_playlist(metadata: Mapping[str, Any], config: Any) -> str:
    """Return the playlist name to assign for ``metadata``.

    Reads ``config.drive.youtube_playlists.rules`` in order. The first
    rule whose ``match`` dict is fully satisfied by the derived metadata
    keys (``region`` and, when available, ``format``) wins. Falls back
    to ``default`` if no rule matches — the function is total.

    Explicit ``metadata["region"]`` overrides the country-derived value.
    Both are normalised to lowercase before matching, so rules can be
    written naturally as ``{ region: "ireland" }``. Country / US-ness comes
    from the shared ``derive_region`` — authoritative off
    ``metadata["resolved_location"]`` when present, label-parsed otherwise.

    ``format`` is derived from ``metadata["trim_ratio"]`` (same key
    ``phase_d_metadata`` stamps onto the upload metadata) using the same
    strictly-below band as ``longform_metadata._select_verb``'s
    "Highlights from" / "Driving" title choice: ``"highlights"`` when
    the ratio is truthy and below ``HIGHLIGHTS_TRIM_THRESHOLD``, else
    ``"full"``. When ``trim_ratio`` is absent or not a number, no
    ``format`` key is derived at all, so a rule that requires ``format``
    simply can't match and region-only rules still decide — the
    behavior is unchanged for metadata that carries no trim signal.

    An explicit ``metadata["format"]`` overrides the trim-derived value,
    mirroring the explicit-``region`` override above (e.g. superlong
    parts, which carry no ``trim_ratio``, stamp ``format: "superlong"``
    directly).
    """
    # Build the derived key dict the rules match against.
    derived: dict[str, str] = {}
    country = derive_region(
        str(metadata.get("location") or ""),
        resolved=metadata.get("resolved_location"),
    )
    if country:
        derived["region"] = country
    explicit = metadata.get("region")
    if isinstance(explicit, str) and explicit.strip():
        derived["region"] = explicit.strip().lower()

    trim_ratio = metadata.get("trim_ratio")
    if isinstance(trim_ratio, (int, float)):
        is_highlights = bool(trim_ratio) and trim_ratio < HIGHLIGHTS_TRIM_THRESHOLD
        derived["format"] = "highlights" if is_highlights else "full"
    explicit_format = metadata.get("format")
    if isinstance(explicit_format, str) and explicit_format.strip():
        derived["format"] = explicit_format.strip().lower()

    yt = config.drive.youtube_playlists
    for rule in yt.rules:
        match = rule.match
        if all(derived.get(k) == str(v).lower() for k, v in match.items()):
            return rule.playlist_name
    return yt.default
