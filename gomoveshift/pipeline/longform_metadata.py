"""
Long Form Metadata Generator — YouTube title, description, and tags.

Uses the SEO Manager library for analytics-driven metadata generation.
Learns from historical performance data to improve titles, descriptions,
and tags over time.

The SEO Manager provides:
- NLP keyword extraction from location/topic
- YouTube-optimized title/description formatting
- Performance pattern analysis from past videos
- Quality scoring and validation
- Event sourcing for tracking what was generated

Before generating metadata for a new video, syncs the latest YouTube
analytics so the optimizer has fresh performance data to learn from.

"""

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from gomoveshift.config import get_config as _get_config

from . import schedule_constants as _schedule_constants
from ._series_utils import _strip_series_suffix, parse_lf_label
from .content_validator import ContentValidator
from .schedule_constants import (
    EST,
    LONG_FORM_DAYS,  # noqa: F401 — re-exported for test_schedule_constants single-source contract
    LONG_FORM_TIME,  # noqa: F401 — re-exported for test_schedule_constants single-source contract
)
from .title_builder import _road_passes_quality

logger = logging.getLogger(__name__)


def _derive_country(location: str) -> str:
    """Return the country component of a location string (lowercased).

    Country = last comma-delimited component of the location after
    stripping the ``— Part N`` series suffix, lowercased and stripped of
    surrounding whitespace. Returns ``""`` for empty or single-component
    inputs. Single convention shared with ``_is_us_location`` and the
    regions helper (sibling) — never duplicated. It is threaded per-LF
    into the upstream region-detector so the admin
    level (state vs county) is selected correctly per LF, not per drive.
    """
    if not location:
        return ""
    parts = [p.strip() for p in _strip_series_suffix(location).split(",") if p.strip()]
    if not parts:
        return ""
    return parts[-1].lower()


def _is_us_location(location: str, resolved: Optional[object] = None) -> bool:
    """True when the LF resolves to the US (miles/feet units).

    Routed through the shared ``region_resolver.is_us_region`` —
    authoritative off the canonical ``ResolvedLocation`` (``resolved``) when
    supplied, label-parsed otherwise. The duplicate parse formerly inlined here
    (and in ``playlist_resolver``) is deleted.
    """
    from .region_resolver import is_us_region

    return is_us_region(location, resolved=resolved)


def _format_distance_line(
    distance_km, location: str, resolved: Optional[object] = None, full_drive: bool = False
) -> str:
    """Country-aware ``🛣️ Distance:`` line for the long-form description.

    - US drives → ``🛣️ Distance: 29 mi`` (no km, no parenthetical)
    - non-US drives → ``🛣️ Distance: 47 km (29 mi)``
    - ``distance_km`` falsy (None / 0 / negative) → empty string (line omitted)
    - ``full_drive=True`` → the label reads ``Full drive:`` instead
      of ``Distance:``, satisfying the ``check_description_redundancy``
      scope rule for a distance computed from the whole (pre-cut) trace
      rather than the LF's kept segments.

    Returns the full line including trailing newline so callers can splice
    it into the description without additional spacing logic. Distance is
    rounded to the nearest integer in both units; thousands are
    comma-separated for readability on long drives.
    """
    if distance_km is None or distance_km <= 0:
        return ""
    km_int = round(float(distance_km))
    mi_int = round(float(distance_km) * 0.621371)
    label = "Full drive" if full_drive else "Distance"
    if _is_us_location(location, resolved=resolved):
        return f"🛣️ {label}: {mi_int:,} mi\n"
    return f"🛣️ {label}: {km_int:,} km ({mi_int:,} mi)\n"


def _format_insert_line(count: int) -> str:
    """``📸 N photo/clip inserts`` for a cut carrying inserts; empty (line omitted) for none.

    ``count`` comes from ``Deliverables.inserts`` (the drive run's per-LF insert records).
    """
    n = int(count or 0)
    if n <= 0:
        return ""
    return f"\U0001f4f8 {n} photo/clip insert{'s' if n != 1 else ''}\n"


# Elevation description lines (scalar ⛰️ gain + 📈 profile range) are DROPPED
# — low SEO value and GPS-noise-inflated on flat drives. Their
# formatters (`_format_elevation_line` / `_format_elevation_profile_line`) and
# suppression-threshold constants were deleted along with the rendered lines.
# The underlying `elevation_gain_m` / `elevation_profile` state is still
# computed + persisted (route-stats sidecar + invariants consume it).


def _speed_character(avg_kmh: float, std_kmh: float) -> str:
    """Human pace descriptor from avg speed + per-step speed std.

    The speed *spread* (std) distinguishes a constant-speed highway from a
    curvy/town drive at the same average: high std = lots of accel/brake =
    bends and settlements. Thresholds are in km/h regardless of display unit.
    """
    if avg_kmh < 35:
        return "town & village pace"
    if std_kmh >= 25:
        return "winding roads"
    if avg_kmh >= 80 and std_kmh < 15:
        return "highway cruise"
    return "country roads"


def _format_speed_line(
    speed, location: str, resolved: Optional[object] = None, full_drive: bool = False
) -> str:
    """Country-aware ``🚗 Pace:`` line (speed/character).

    - US → ``🚗 Pace: 48 mph avg · highway cruise`` (mph)
    - non-US → ``🚗 Pace: 78 km/h avg · winding roads`` (km/h)
    - ``speed`` falsy / non-positive ``avg_kmh`` → "" (line omitted)
    - ``full_drive=True`` → the label reads ``Full drive:`` instead
      of ``Pace:``, satisfying the ``check_description_redundancy``
      scope rule for a speed computed from the whole (pre-cut) trace rather
      than the LF's kept segments.

    ``speed`` is the ``state.per_lf_speed`` dict (``avg_kmh``/``max_kmh``/
    ``std_kmh``). Character via ``_speed_character``.
    """
    if not speed:
        return ""
    avg_kmh = speed.get("avg_kmh")
    if not avg_kmh or avg_kmh <= 0:
        return ""
    character = _speed_character(float(avg_kmh), float(speed.get("std_kmh") or 0.0))
    label = "Full drive" if full_drive else "Pace"
    if _is_us_location(location, resolved=resolved):
        return f"🚗 {label}: {round(avg_kmh * 0.621371):,} mph avg · {character}\n"
    return f"🚗 {label}: {round(avg_kmh):,} km/h avg · {character}\n"


# 8-point compass → human "…bound" heading word.
_BEARING_WORD = {
    "N": "Northbound",
    "NE": "Northeastbound",
    "E": "Eastbound",
    "SE": "Southeastbound",
    "S": "Southbound",
    "SW": "Southwestbound",
    "W": "Westbound",
    "NW": "Northwestbound",
}


def _format_bearing_line(bearing, location: str, resolved: Optional[object] = None) -> str:
    """``🧭 Heading:`` line (net travel direction).

    ``bearing`` is the ``state.per_lf_bearing`` 8-point compass string
    (``"N"``/``"NE"``/…) wired from ``_compass_from_waypoints``. Renders
    ``🧭 Heading: Northbound``; unknown / unmapped value → "" (omitted).
    Direction is unit-independent so ``location``/``resolved`` are unused
    (kept for formatter-signature symmetry).
    """
    word = _BEARING_WORD.get(bearing) if bearing else None
    if not word:
        return ""
    return f"🧭 Heading: {word}\n"


# Cap the regions list at this many entries in the description. It was
# lifted from 4 → 12: a long multi-county drive should list EVERY county
# it crossed (the old 4-cap dropped the long-tail names the SEO/discovery value
# lives in). The ceiling is kept only as a wall-of-text backstop for a
# pathological trace; realistic drives (≤12 regions) now list all of them.
# Excess beyond the ceiling is still summarised as ``… +N more``.
_REGIONS_DESCRIPTION_CAP = 12


def _format_regions_line(regions, location: str) -> str:
    """``🗺️ Through:`` line for the long-form description.

    Single label across countries (Ireland and USA both use ``🗺️ Through:``). Returns the full line
    with trailing newline so callers can splice it into the description.

    Behaviour:

    - ``regions`` falsy (None / empty list) → empty string (line omitted).
    - One or more regions → ``🗺️ Through: County Kerry, County Cork``.
    - Beyond ``_REGIONS_DESCRIPTION_CAP`` entries → first N items joined,
      then ``" … +M more"`` where ``M`` is the overflow count.

    Region names are read from each dict's ``name`` field. Adjacent
    dedup is the upstream helper's responsibility; this formatter is
    cosmetic-only.

    ``location`` is the LF's own label, and it is FILTERED OUT of the
    list: "Through" means the regions the route passed through, and the
    LF's own region is not information there. It is also a third
    byte-identical occurrence of a string the hook and the ``Location:``
    line already carry, which breaches the 2x location budget in
    ``check_description_redundancy``. When nothing else remains, the
    line is omitted entirely.
    """
    if not regions:
        return ""
    names = [r.get("name") for r in regions if r and r.get("name")]
    # EXACT equality, never substring: a town-form label like
    # "Dingle, County Kerry, Ireland" legitimately contains the region
    # "County Kerry", and that line must still render.
    own = (location or "").strip()
    names = [n for n in names if n.strip() != own]
    if not names:
        return ""
    if len(names) <= _REGIONS_DESCRIPTION_CAP:
        joined = ", ".join(names)
        return f"🗺️ Through: {joined}\n"
    overflow = len(names) - _REGIONS_DESCRIPTION_CAP
    head = ", ".join(names[:_REGIONS_DESCRIPTION_CAP])
    return f"🗺️ Through: {head} … +{overflow} more\n"


# Town-level route sequence cap for the ``🛤️ Route:`` line. A long drive's
# full town list is wall-of-text; show head → … → tail beyond this.
_ROUTE_PLACES_CAP = 8


def _format_route_line(resolved: Optional[object]) -> str:
    """``🛤️ Route:`` line — the route's town-level place sequence.

    Reads ``resolved.ordered_place_sequence`` (route-ordered, consecutively
    deduped town/village labels populated by the builder via
    ``dominant_location.ordered_places``). Complements the county-level
    ``🗺️ Through:`` line at finer granularity:

    - ``Killarney → Kenmare → Sneem`` joined with ``→``.
    - ≥2 places required (a single point is no "route") → "" otherwise.
    - Beyond ``_ROUTE_PLACES_CAP`` places → head ``→ … →`` tail elision so the
      arc's endpoints always survive.
    """
    seq = list(getattr(resolved, "ordered_place_sequence", None) or []) if resolved else []
    seq = [s for s in seq if isinstance(s, str) and s.strip()]
    if len(seq) < 2:
        return ""
    if len(seq) <= _ROUTE_PLACES_CAP:
        body = " → ".join(seq)
    else:
        head = _ROUTE_PLACES_CAP // 2
        body = " → ".join(seq[:head]) + " → … → " + " → ".join(seq[-(_ROUTE_PLACES_CAP - head) :])
    return f"🛤️ Route: {body}\n"


# Defaults sourced from the user's config.yaml ``seo:`` block at import time.
# Previously these read from ``_SEOConfig()`` bare defaults (cwd-relative
# ``./credentials`` and ``./data/seo``), so any LongFormMetadataGenerator
# constructed no-arg silently bypassed the user's configured paths. The
# YouTube credential loader's SOPS lookup at
# ``creds_dir.parent / "youtube_upload_token.json.enc"`` then resolved to
# ``./youtube_upload_token.json.enc`` (a path that doesn't exist), surfacing
# as ``exists=False`` in pipeline logs. Tests still monkeypatch these
# constants directly via the autouse
# ``conftest.isolate_seo_storage_from_workspace`` fixture.
DEFAULT_SEO_STORAGE = Path(_get_config().seo.storage_path)
DEFAULT_CREDENTIALS_DIR = Path(_get_config().seo.credentials_dir)


# Distinguishable outcomes of reading the published-titles ledger. The review
# gate treats UNREADABLE as a hard fail (dedup can't be verified → don't ship an
# unverified title) but MISSING as a skip (a fresh channel legitimately has no
# history yet). ``ensure_unique_titles`` consumers degrade to the empty set on
# both — robust-by-construction, they never need to raise.
LEDGER_OK = "ok"
LEDGER_MISSING = "missing"
LEDGER_UNREADABLE = "unreadable"

# Skip test uploads from the dedup ledger. Word-boundary, NOT a bare substring:
# ``"test" in title`` also dropped real titles like "Testament Creek" and
# "Contest Winner". ``\btest\b`` matches only the standalone word ("Test Drive",
# "TEST upload").
_TEST_TITLE_RE = re.compile(r"\btest\b", re.IGNORECASE)


def read_published_titles() -> tuple[set[str], str]:
    """Read the channel published-titles ledger as ``(titles, status)``.

    ``status`` is one of :data:`LEDGER_OK`, :data:`LEDGER_MISSING` (no file —
    legitimate on a fresh channel) or :data:`LEDGER_UNREADABLE` (OSError /
    decode error — the ledger exists but can't be parsed, so dedup protection
    cannot be verified). On UNREADABLE a greppable WARN is logged and an empty
    set returned; callers that hard-gate on title uniqueness must treat that
    status as a failure rather than as "no known titles".

    Reads ``<DEFAULT_SEO_STORAGE>/gomoveshift/events.jsonl`` — the analytics
    event ledger :meth:`LongFormMetadataGenerator._batch_sync_stats` fills with
    every channel video's ``data.title``. Blank titles and the standalone word
    ``"test"`` (case-insensitive, word-boundary) are skipped. A single
    malformed JSON line is skipped, not treated as unreadable.
    """
    events = DEFAULT_SEO_STORAGE / "gomoveshift" / "events.jsonl"
    if not events.exists():
        return set(), LEDGER_MISSING
    titles: set[str] = set()
    try:
        with open(events, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(event, dict):
                    continue
                data = event.get("data")
                title = data.get("title", "") if isinstance(data, dict) else ""
                if not title or _TEST_TITLE_RE.search(title):
                    continue
                titles.add(title)
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(
            "published-titles ledger unreadable at %s (%s) — duplicate-title "
            "protection cannot be verified",
            events,
            exc,
        )
        return set(), LEDGER_UNREADABLE
    return titles, LEDGER_OK


def load_published_titles() -> set[str]:
    """Return the set of YouTube titles already live on the channel.

    Thin wrapper over :func:`read_published_titles` for robust-by-construction
    consumers (``title_builder.ensure_unique_titles``) that only need the set
    and degrade to "no known published titles" on a missing/unreadable ledger.
    An unreadable ledger still emits the greppable WARN (via
    ``read_published_titles``). The review gate calls ``read_published_titles``
    directly so it can hard-gate on the UNREADABLE status before the
    irreversible OST burn.
    """
    titles, _status = read_published_titles()
    return titles


def live_catalog_title_ids() -> dict[str, str]:
    """``{normalized_title: video_id}`` from the local live-catalog snapshot
    that ``catalog-fetch --known-ids-from-queue`` writes beside the queue DB
    (``<DEFAULT_SEO_STORAGE>/live_catalog.latest.json``).

    Names the live video behind a duplicate-title refusal when the run's
    channel refresh gave no id (degraded run, or a site that never had one)
    and the on-disk ledger, which carries titles only, cannot. Any age is
    fine — this names a video, it does not gate one. Never fetches; a
    missing or unreadable snapshot is ``{}``.
    """
    from ..cli.handlers.pub_status import live_catalog_snapshot_path

    try:
        catalog = json.loads(
            live_catalog_snapshot_path(DEFAULT_SEO_STORAGE).read_text(encoding="utf-8")
        )
        videos = catalog.get("videos") or {}
    except (OSError, ValueError, AttributeError):
        return {}
    ids: dict[str, str] = {}
    for video_id, video in videos.items() if isinstance(videos, dict) else ():
        title = video.get("title") if isinstance(video, dict) else None
        if title and title.strip():
            ids.setdefault(title.strip().lower(), video_id)
    return ids


def refresh_channel_title_ledger(credentials_dir: Optional[Path] = None) -> dict[str, str]:
    """One ``playlistItems.list`` call on the channel's uploads
    playlist, returning ``{normalized_title: video_id}`` for every live
    upload. ``events.jsonl`` (:func:`load_published_titles`) is only
    refreshed as a side effect of Phase D's full analytics sync, so a Phase
    BC that starts before the previous upload's Phase D dedups against a
    stale ledger (the bug behind two live videos both titled "Pennsylvania
    Scenic Drive in 4K | Gettysburg, PA"). Phase BC calls this right before
    ``load_published_titles`` so the dedup check sees the channel as of now.

    Paginated, ``part=snippet`` only (~3 quota units); reuses the credential
    source and client construction of ``_sync_youtube_analytics`` /
    ``_fetch_channel_videos``. Missing credentials or any API failure
    degrade to ``{}`` with a WARNING — never a hard failure (an
    offline/credential-less run must still dry-run Phase BC).
    """
    from ._yt_creds import load_youtube_credentials

    try:
        # This module's DEFAULT_CREDENTIALS_DIR (monkeypatched by tests),
        # matching ``LongFormMetadataGenerator.credentials_dir``'s fallback.
        yt_creds = load_youtube_credentials(credentials_dir or DEFAULT_CREDENTIALS_DIR)
    except FileNotFoundError as exc:
        logger.warning(
            "YouTube credentials unavailable for the Phase BC ledger refresh "
            "(%s) — duplicate-title protection falls back to the on-disk "
            "ledger, which may be stale since the last Phase D analytics sync",
            exc,
        )
        return {}

    try:
        import os
        import tempfile

        from seo_manager.analytics.youtube_sync import YouTubeAnalyticsSync

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="yt_creds_", delete=True
        ) as creds_file:
            os.chmod(creds_file.name, 0o600)
            json.dump(yt_creds, creds_file)
            creds_file.flush()
            sync = YouTubeAnalyticsSync(creds_file.name)

            channels_response = (
                sync.youtube_data.channels().list(part="contentDetails", mine=True).execute()
            )
            items = channels_response.get("items") or []
            if not items:
                return {}
            uploads_playlist = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

            titles: dict[str, str] = {}
            next_page = None
            while True:
                resp = (
                    sync.youtube_data.playlistItems()
                    .list(
                        part="snippet",
                        playlistId=uploads_playlist,
                        maxResults=50,
                        pageToken=next_page,
                    )
                    .execute()
                )
                for item in resp.get("items", []):
                    snippet = item.get("snippet", {})
                    title = snippet.get("title", "")
                    video_id = snippet.get("resourceId", {}).get("videoId", "")
                    if title and video_id:
                        titles[title.strip().lower()] = video_id
                next_page = resp.get("nextPageToken")
                if not next_page:
                    break
            return titles
    except Exception as exc:
        logger.warning(
            "Phase BC ledger refresh failed (%s) — falling back to the on-disk ledger",
            exc,
        )
        return {}


# Trim-ratio threshold below which an LF is framed as "Highlights from …"
# rather than "Driving …". Single source of truth — the real
# trim_ratio is threaded through phase_bc and the same constant is used in description and
# first-comment band selection.
HIGHLIGHTS_TRIM_THRESHOLD = 0.5

# Minimum source-minus-final delta (seconds) for the "🎬 Final cut: X
# (edited from Y)" honesty disclosure to fire. Gating on numeric seconds — not
# formatted-string equality — prevents the round-collapse where two materially
# different durations format identically (e.g. "30 min") and hide a real edit,
# and suppresses trivial sub-30s edits that aren't worth disclosing.
_TRIM_DISCLOSURE_MIN_DELTA_SEC = 30.0

# Verb-led title template. The verb is structural, not optional — every
# generated title starts with a curated verb.
TITLE_TEMPLATE = "{verb} {location}"

# Maximum YouTube title length we accept (YouTube hard limit is 100 chars).
_TITLE_MAX_LEN = 100


def _select_verb(trim_ratio: Optional[float]) -> str:
    """Choose the verb for an LF title given its trim ratio.

    A caller may pass ``trim_ratio=None`` (returns ``"Driving"``); the real
    ratio is threaded through phase_bc — the function is
    already total over both cases so no further generator change is needed
    when the ratio becomes available.

    A literal-zero ratio is "unmeasured" (no segments / ffprobe gap), NOT a
    100%-trimmed highlight — ``if trim_ratio`` treats both ``0`` and ``None`` as
    unmeasured so a full-length LF never gets the "Highlights from" verb. A real
    heavy edit (e.g. 0.3) is truthy and still bands to "Highlights from".
    """
    if trim_ratio and trim_ratio < HIGHLIGHTS_TRIM_THRESHOLD:
        return "Highlights from"
    return "Driving"


def _clamp_title_len(title: str, max_len: int = _TITLE_MAX_LEN) -> str:
    """Clamp a fully-assembled YouTube title to ``max_len`` chars.

    ``_enforce_length_simple`` (``title_builder``) clamps only the verb-less
    base inside ``build_lf_title``'s fallback cascade; the verb prepend AND
    the dedup ``"(Part N of M)"`` suffix are appended downstream with no
    length awareness, so the assembled ``"{verb} {base} (Part N of M)"`` could
    exceed YouTube's 100-char hard limit. This is the single post-assembly
    clamp — ``_build_title`` is the one site both the SEO-manager and basic
    generators route through. It preserves the verb prefix and any trailing
    ``"(Part N of M)"`` suffix, truncating the middle on a word boundary.
    """
    if len(title) <= max_len:
        return title
    m = _PART_SUFFIX_RE.search(title)
    suffix = m.group(0).strip() if m else ""
    head = title[: m.start()].rstrip() if m else title
    budget = max_len - (len(suffix) + 1 if suffix else 0)
    if budget <= 0:
        # Pathological: the suffix alone blows the budget — hard slice.
        return title[:max_len].rstrip()
    if len(head) > budget:
        cut = head[:budget]
        sp = cut.rfind(" ")
        head = (cut[:sp] if sp > 0 else cut).rstrip(" ,—-")
    return f"{head} {suffix}" if suffix else head


# Route-only SEO title shape. "Not all footage is scenic drives" — the title
# makes NO genre claim; it carries only factual tokens (route + "in 4K"). The
# route string already carries the region via its state/country suffix
# ("Asheville to Boone, NC"). The via-road is demoted to tags/description; the
# trim-honesty verb moves to the description's 🎬 Final cut line. Targets the
# 55-65 char SERP window (clamped to 100 by ``_clamp_title_len``; never padded).
# NOTE: this drops the former front-loaded-region optimization (region no
# longer leads); the region survives in the route tail and the description hook.
_SEO_TITLE_TEMPLATE = "{route} in 4K"
# Drop a "via {Road}" clause (up to the next comma / end) from the route.
_VIA_ROAD_RE = re.compile(r"\s+via\s+[^,]+", re.IGNORECASE)
# The dedup suffix appended by ``title_builder.ensure_unique_titles`` — either
# "(Part N of M)" (in-batch collisions) or bare "(Part N)" (channel-collision
# bump loop). Single source of truth for ``_clamp_title_len`` (which preserves
# it inside the 100-char budget) and ``_build_title`` (detach it before the
# front-loaded rebuild and re-append it to the final string, so the rebuild
# can't parse it into the region noun and two same-county LFs stay distinct).
# The bare form must match too — previously it was folded into the
# route noun or truncated away, silently restoring the duplicate.
_PART_SUFFIX_RE = re.compile(r"\s*\(Part \d+(?: of \d+)?\)\s*$")


def _front_loaded_title(location: str, resolved: Optional[object] = None) -> str:
    """Reshape a canonical LF label into the route-only SEO title.

    ``"Asheville to Boone via US-221, NC"`` → ``"Asheville to Boone, NC in
    4K"``; ``"Killarney to Dingle via N86, Kerry"`` → ``"Killarney to Dingle,
    Kerry in 4K"``. The route keeps its region tail (the state/country suffix)
    so no region context is lost; no genre word ("Scenic Drive") is ever
    emitted. US-ness comes from ``resolved`` (the SAME verdict as units/
    playlist/tags) when present, else from the label, and only normalizes the
    SERP-friendly state abbr in the route tail.
    """
    from .title_builder import is_us_state, normalize_state_abbr

    base = _strip_series_suffix(_strip_leading_verb((location or "").strip()))
    if not base:
        # Degenerate: no route, no region — emit a minimal factual format token,
        # never the unverifiable "Scenic Drive" genre label.
        return "4K POV"
    parts = [p.strip() for p in _VIA_ROAD_RE.sub("", base).split(",") if p.strip()]
    trailing = parts[-1] if len(parts) >= 2 else ""
    if _is_us_location(location, resolved=resolved) and trailing and is_us_state(trailing):
        # Keep the SERP-friendly ", NC" abbr in the route tail.
        parts[-1] = normalize_state_abbr(trailing) or trailing
    route = ", ".join(parts)
    return _SEO_TITLE_TEMPLATE.format(route=route)


def assemble_final_title(label: str, resolved: Optional[object] = None) -> str:
    """Map a canonical LF label to its assembled final YouTube title.

    Pure, side-effect-free wrapper over ``LongFormMetadataGenerator
    ._build_title`` — the single reshape site (verb strip → via-clause drop →
    "{route} in 4K" → 100-char clamp, part suffix detached/re-appended). The
    channel ledger (``load_published_titles``) holds titles in THIS namespace,
    while dedup inputs are pre-reshape labels; every comparison between the
    two must go through this helper. Never duplicates the templates.
    """
    return LongFormMetadataGenerator._build_title(label, resolved=resolved)


# The above-the-fold description SEO search line. YouTube renders
# only ~the first two lines of a description before the "…more" fold — the
# SEO-critical zone. Line 1 stays the trim-banded hook (POV drive / Highlights
# from); line 2 is THIS line, which guarantees the evergreen "4K drive"
# term + the region noun + the GPS-verified town arc + the via-road all sit
# above the fold. The road is demoted from the title precisely so the
# description is where it earns its keep. Robust-by-construction: an LF with no
# usable location data yields "" and the line is simply omitted (line 1 hook +
# blank-line back-compat preserved).
_SEO_HOOK_ARC_REGION = "{region} 4K drive — {arc}."
_SEO_HOOK_ARC_ONLY = "{arc} — 4K drive."
_SEO_HOOK_REGION_ONLY = "4K drive through {region}."


def _seo_arc(arc_part: str, resolved: Optional[object] = None) -> str:
    """The searchable ``A to B`` town arc for the description SEO line.

    Prefers the label's own canonical ``"A to B"`` (so the description arc stays
    consistent with the title's arc — both are vote-derived by ``build_lf_title``);
    falls back to the resolver's vote-derived ``ordered_place_sequence`` endpoints
    when the label carries no arc (a bare-region / GPS-less label). Returns "" when
    no two distinct endpoints exist. The vote-derived sources mean a wrong county
    can't leak a wrong town into the arc.
    """
    if " to " in arc_part:
        a, _, b = arc_part.partition(" to ")
        a, b = a.strip(), b.strip()
        if a and b and a.lower() != b.lower():
            return f"{a} to {b}"
    seq = list(getattr(resolved, "ordered_place_sequence", None) or []) if resolved else []
    seq = [s for s in seq if isinstance(s, str) and s.strip()]
    if len(seq) >= 2 and seq[0].lower() != seq[-1].lower():
        return f"{seq[0]} to {seq[-1]}"
    return ""


def _seo_search_line(
    location: str,
    resolved: Optional[object] = None,
    named_routes: Optional[list[str]] = None,
    road_coverage_measured: bool = False,
) -> str:
    """Build the above-the-fold description SEO search line.

    Shape ``"{Region} 4K drive — {A} to {B} via {Road}."`` Region is the
    FULL state name for US drives (SEO value: "North Carolina" > "NC") and the
    trailing region verbatim for non-US — mirroring ``_front_loaded_title`` so
    the two surfaces agree. The via-road (dropped from the title) is restored
    here only when ``road_coverage_measured``: untimed votes
    never back a road claim, so by default the clause is stripped, not
    restored. Returns the line WITH a trailing newline, or "" when no usable
    location data exists (the line is then omitted, preserving back-compat).
    """
    from .title_builder import state_full_name

    base = _strip_series_suffix(_strip_leading_verb((location or "").strip()))
    if not base:
        return ""

    # Restore the via-road the title demoted — the description is where
    # the road belongs. Prefer the label's "via {Road}" clause, else a named route.
    via_road = ""
    m_via = _VIA_ROAD_RE.search(base)
    if m_via:
        via_road = m_via.group(0).strip()[len("via") :].strip().rstrip(",").strip()
    if not via_road and named_routes:
        via_road = next((r.strip() for r in named_routes if r and r.strip()), "")
    if not road_coverage_measured:
        via_road = ""

    parts = [p.strip() for p in _VIA_ROAD_RE.sub("", base).split(",") if p.strip()]
    # The label is "<arc-or-town>[, <region>]" where arc = "A to B".
    region_raw = ""
    arc_part = ""
    if len(parts) >= 2:
        region_raw, arc_part = parts[-1], parts[0]
    elif len(parts) == 1:
        if " to " in parts[0]:
            arc_part = parts[0]  # bare arc, no region
        else:
            region_raw = parts[0]  # bare region / single town

    if _is_us_location(location, resolved=resolved):
        dom_state = getattr(resolved, "dominant_state", None) if resolved else None
        region = state_full_name(dom_state or region_raw) if (dom_state or region_raw) else ""
    else:
        region = region_raw

    arc = _seo_arc(arc_part, resolved)
    if arc and via_road:
        arc = f"{arc} via {via_road}"

    if arc and region:
        body = _SEO_HOOK_ARC_REGION.format(region=region, arc=arc)
    elif arc:
        body = _SEO_HOOK_ARC_ONLY.format(arc=arc)
    elif region:
        body = _SEO_HOOK_REGION_ONLY.format(region=region)
    else:
        return ""
    return f"{body}\n"


def _strip_leading_verb(text: str) -> str:
    """Drop a leading curated verb so a re-prefix can't double it.

    Title overrides (``per_lf_ost_text``) are verb-less by construction,
    but a future caller could pass an already-verbed string. Stripping
    first makes ``_build_title`` idempotent over the verb prefix
    (``f(f(x)) == f(x)``). The verb set is closed — exactly the two
    strings ``_select_verb`` can return.
    """
    for verb in ("Highlights from ", "Driving "):
        if text.startswith(verb):
            return text[len(verb) :]
    return text


# Month + northern-hemisphere season tags. GMS films Ireland + the US —
# both N. hemisphere — so a flat calendar-month → season map is correct for
# every drive the pipeline produces (revisit only if a S.-hemisphere drive is
# ever added). Seasonal search demand is real ("autumn drive", "spring
# drive"); the year alone (the prior behaviour) captured none of it.
_MONTH_NAMES = (
    "January February March April May June July August September October November December"
).split()
_SEASON_BY_MONTH = {
    m: s
    for s, months in {
        "winter": (12, 1, 2),
        "spring": (3, 4, 5),
        "summer": (6, 7, 8),
        "autumn": (9, 10, 11),
    }.items()
    for m in months
}
_ISO_MONTH_RE = re.compile(r"^\s*\d{4}-(\d{2})")


def _seasonal_tags(recording_date: Optional[str]) -> list[str]:
    """Month + season tags from an ISO ``YYYY-MM-DD`` recording date.

    Returns ``[month_name, season, "<season> drive"]`` (e.g. ``["September",
    "autumn", "autumn drive"]``) — whole-phrase, never fragmented. Empty when
    the date is missing or unparseable, so a bad value never fabricates a tag.
    Season is northern-hemisphere by calendar month (see ``_SEASON_BY_MONTH``).
    """
    m = _ISO_MONTH_RE.match(recording_date or "")
    if not m:
        return []
    month = int(m.group(1))
    if not 1 <= month <= 12:
        return []
    season = _SEASON_BY_MONTH[month]
    return [_MONTH_NAMES[month - 1], season, f"{season} drive"]


def _location_tags(location: str, recording_date: Optional[str]) -> list[str]:
    """Build whole-phrase YouTube tags from a location string + recording date.

    Construction-correct: never word-splits the location. Every comma-
    delimited component contributes a tag verbatim. When the
    location has multiple components, two phrase tags are added (``"<first>
    drive"`` and ``"<last> drive"``). When ``recording_date`` is
    present, only the year (``YYYY``) is emitted as a tag — never the full
    date, which would fragment the tag pool.

    De-duplicates while preserving insertion order so the most-specific tag
    (the full first component) appears first.
    """
    tags: list[str] = []
    # An operator ``--per-lf-location`` title ("To Clonakilty, County
    # Cork") is not a place noun — tags read the parsed place, never the
    # leading title word.
    base = _strip_series_suffix(parse_lf_label(location).place)
    parts = [p.strip() for p in base.split(",") if p.strip()]
    for part in parts:
        tags.append(part)
    if len(parts) >= 2:
        tags.append(f"{parts[0]} drive")
        tags.append(f"{parts[-1]} drive")
    if recording_date:
        tags.append(recording_date[:4])
        tags.extend(_seasonal_tags(recording_date))  # month + season
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# Broken-geocode patterns the description entry gate rejects. Whole-text
# (case-insensitive where relevant) so real place names pass unharmed. Mirrors
# the shapes ``ost_validator`` rejects for the burned overlay; kept local to
# avoid importing that module's private regexes.
_BROKEN_LITERAL_PATTERN = re.compile(r"^\s*(none|null)(\s*,\s*(none|null))*\s*$", re.IGNORECASE)
_COORDINATE_PATTERN = re.compile(
    r"^\s*-?\d{1,3}(?:\.\d+)?\s*°?\s*[,\s]\s*-?\d{1,3}(?:\.\d+)?\s*°?\s*$"
)
_DIGITS_ONLY_PATTERN = re.compile(r"^\s*\d+\s*$")


def _clean_hashtag(text: str) -> str:
    """Reduce a label component to a clean alphanumeric hashtag body.

    Strips every non-``[A-Za-z0-9]`` char (spaces, parens, commas, periods,
    dashes) so a road-disambiguated label like ``"Colleton County (Blocker
    Run Road)"`` yields ``"ColletonCountyBlockerRunRoad"`` instead of a
    syntactically broken ``"#ColletonCounty(BlockerRunRoad)"``.
    """
    return re.sub(r"[^A-Za-z0-9]", "", text)


def _label_components(location: str) -> list[str]:
    """Clean comma components of an LF label — the shared root for the LF
    description's region noun and its geo hashtags (kills the ``parts[0]``
    route-form spam). Strips a leading curated verb, the ``(Part N of M)`` /
    ``— Part N`` series suffix, and the ``via {Road}`` clause, then splits on
    commas and drops empties. The ``A to B`` arc is left intact here; the
    hashtag caller splits it so each town becomes its own component::

        "Killarney to Dingle via N86, County Kerry" -> ["Killarney to Dingle", "County Kerry"]
        "County Kerry, Ireland"                     -> ["County Kerry", "Ireland"]
    """
    base = _VIA_ROAD_RE.sub("", _strip_series_suffix(_strip_leading_verb((location or "").strip())))
    return [p.strip() for p in base.split(",") if p.strip()]


# GoMoveShift-specific base tags (always included)
GMS_BASE_TAGS = [
    "POV driving",
    "drive",
    "road trip",
    "driving video",
    "4K driving",
    "relaxing drive",
    "virtual travel",
    "dashcam",
    "drive with me",
    "driving route",
    "POV",
    "long drive",
]


def _optimizer_body(
    location: str,
    duration_str: str,
    regions: Optional[list] = None,
    distance_km: Optional[float] = None,
    resolved: Optional[object] = None,
) -> str:
    """Fact-rich body fed to the SEO optimizer for keyword extraction.

    Replaces the fixed boilerplate (which produced generic, drive-agnostic
    keywords) with THIS LF's real route facts — the GPS-verified place
    sequence, the regions crossed, and the distance — so the optimizer's
    analytics-derived tags reflect the actual drive. Every clause is optional;
    a GPS-less LF degrades to the evergreen sentences alone.
    """
    sentences = [f"POV driving video through {location}, {duration_str} of 4K drive footage."]
    seq = list(getattr(resolved, "ordered_place_sequence", None) or []) if resolved else []
    seq = [s for s in seq if isinstance(s, str) and s.strip()]
    if len(seq) >= 2:
        sentences.append(f"The route passes through {', '.join(seq[:8])}.")
    names = [r.get("name") for r in (regions or []) if r and r.get("name")]
    if names:
        sentences.append(f"Driving through {', '.join(names[:6])}.")
    if distance_km and distance_km > 0:
        sentences.append(f"A {round(float(distance_km))} km road trip.")
    sentences.append("Relaxing dashcam road trip footage, perfect for virtual travel.")
    return " ".join(sentences)


# Administrative marker that identifies a phrase tag as naming a county-level
# region. ``resolved.regions`` labels ("Nelson County, VA", "County Kerry")
# always carry it. Kept for reference/back-compat; the cross-region filter now
# discriminates by proper-noun vocabulary (below), which subsumes it.
_REGION_MARKER_RE = re.compile(r"\b(?:county|parish)\b", re.IGNORECASE)

# Word tokenizer for tag phrases — alphanumerics only (drops punctuation).
_TAG_WORD_RE = re.compile(r"[a-z0-9]+")

# Generic drive/scenic/production vocabulary that legitimately appears in the
# optimizer's channel-wide top-performing phrases REGARDLESS of where an LF
# drives ("mountain pass drive", "relaxing coastal views", "autumn road trip").
# These words are place-agnostic, so a phrase built only from them is NEVER a
# cross-region leak. The discriminator for a leak (below) is a phrase word that
# is NONE of: generic vocab, a common descriptor/function word, or one of THIS
# LF's own resolved place names — i.e. a proper-noun place from another region
# ("kerry", "dingle", "kenmare"). Kept deliberately broad on the generic side so
# real generic tags survive; a residual place named entirely from generic words
# ("ocean city") is an accepted miss.
_GENERIC_TAG_WORDS: frozenset[str] = frozenset(
    {
        # driving / roads / journey
        "pov",
        "driving",
        "drive",
        "drives",
        "road",
        "roads",
        "roadway",
        "roadways",
        "route",
        "routes",
        "highway",
        "highways",
        "freeway",
        "motorway",
        "expressway",
        "street",
        "streets",
        "lane",
        "lanes",
        "avenue",
        "boulevard",
        "trip",
        "trips",
        "roadtrip",
        "roadtrips",
        "journey",
        "journeys",
        "tour",
        "tours",
        "touring",
        "ride",
        "rides",
        "cruising",
        "cruise",
        "commute",
        "backroad",
        "backroads",
        "byway",
        "byways",
        "loop",
        # scenic / landscape (place-agnostic)
        "scenic",
        "scenery",
        "view",
        "views",
        "viewpoint",
        "vista",
        "vistas",
        "panorama",
        "panoramic",
        "landscape",
        "landscapes",
        "nature",
        "natural",
        "wilderness",
        "outdoors",
        "outdoor",
        # geographic-generic (appear in generic scenic tags)
        "mountain",
        "mountains",
        "mountainous",
        "alpine",
        "hill",
        "hills",
        "hilly",
        "highland",
        "highlands",
        "valley",
        "valleys",
        "canyon",
        "canyons",
        "gorge",
        "pass",
        "passes",
        "summit",
        "ridge",
        "cliff",
        "cliffs",
        "coastal",
        "coast",
        "coastline",
        "seaside",
        "ocean",
        "sea",
        "beach",
        "beaches",
        "shore",
        "shores",
        "shoreline",
        "lake",
        "lakes",
        "lakeside",
        "river",
        "rivers",
        "riverside",
        "waterfall",
        "waterfalls",
        "forest",
        "forests",
        "woodland",
        "woodlands",
        "woods",
        "countryside",
        "country",
        "rural",
        "farmland",
        "farm",
        "farms",
        "desert",
        "prairie",
        "plains",
        "meadow",
        "meadows",
        "island",
        "islands",
        "isle",
        "peninsula",
        "bay",
        "harbor",
        "harbour",
        "point",
        "cape",
        # production
        "4k",
        "uhd",
        "8k",
        "hd",
        "60fps",
        "fps",
        "footage",
        "video",
        "videos",
        "film",
        "cinematic",
        "dashcam",
        "dashboard",
        "dash",
        "cam",
        "gopro",
        "camera",
        "aerial",
        "drone",
        "timelapse",
        "hyperlapse",
        # relaxation / ambience
        "relaxing",
        "relax",
        "relaxation",
        "calm",
        "calming",
        "peaceful",
        "soothing",
        "ambient",
        "ambience",
        "meditation",
        "meditative",
        "asmr",
        "chill",
        "chillout",
        "zen",
        "tranquil",
        "serene",
        "serenity",
        "mindful",
        "unwind",
        "cozy",
        "comfy",
        # travel
        "virtual",
        "travel",
        "traveling",
        "travelling",
        "wanderlust",
        "explore",
        "exploring",
        "exploration",
        "adventure",
        "adventures",
        "getaway",
        "vacation",
        "holiday",
        "destination",
        "destinations",
        "sightseeing",
        # season / weather / time
        "autumn",
        "fall",
        "winter",
        "spring",
        "summer",
        "seasonal",
        "foliage",
        "leaves",
        "snow",
        "snowy",
        "rain",
        "rainy",
        "sunny",
        "sunshine",
        "sun",
        "cloud",
        "cloudy",
        "fog",
        "foggy",
        "mist",
        "misty",
        "storm",
        "stormy",
        "weather",
        "morning",
        "afternoon",
        "evening",
        "night",
        "nighttime",
        "midnight",
        "sunset",
        "sunrise",
        "dusk",
        "dawn",
        "twilight",
        "golden",
        "hour",
        "day",
        "daytime",
        "daylight",
        "noon",
        # descriptors / marketing adjectives (protect legit non-place phrases)
        "beautiful",
        "gorgeous",
        "stunning",
        "breathtaking",
        "amazing",
        "epic",
        "incredible",
        "spectacular",
        "picturesque",
        "pretty",
        "lovely",
        "quiet",
        "hidden",
        "gem",
        "gems",
        "secret",
        "best",
        "top",
        "ultimate",
        "perfect",
        "favorite",
        "favourite",
        "must",
        "see",
        "bucket",
        "list",
        "slow",
        "fast",
        "empty",
        "open",
        "endless",
        "winding",
        "curvy",
        "hairpin",
        "twisty",
        "narrow",
        "wide",
        "steep",
        "long",
        "longest",
        "short",
        "quick",
        "easy",
        "smooth",
        "rough",
        "remote",
        "wild",
        "untouched",
        "pristine",
        "famous",
        "iconic",
        "popular",
        "classic",
        "vintage",
        "historic",
        "old",
        "new",
        "great",
        "good",
        "big",
        "little",
        # directions
        "north",
        "south",
        "east",
        "west",
        "northern",
        "southern",
        "eastern",
        "western",
        "central",
        "upper",
        "lower",
        # colors (common in generic tags)
        "blue",
        "green",
        "red",
        "white",
        "black",
        "orange",
        "yellow",
        "purple",
        "pink",
        "grey",
        "gray",
        "brown",
        "silver",
        # function words / stopwords
        "the",
        "and",
        "for",
        "with",
        "without",
        "your",
        "you",
        "our",
        "this",
        "that",
        "through",
        "along",
        "near",
        "over",
        "under",
        "across",
        "around",
        "between",
        "into",
        "onto",
        "off",
        "out",
        "per",
        "via",
        "are",
        "was",
        "were",
        "from",
        "its",
        "day",
        "way",
    }
)


def _own_place_tokens(resolved: Optional[object]) -> set[str]:
    """Lowercased WORD tokens for every place name THIS LF actually drives.

    Union of the resolved model's place fields — ``regions``,
    ``ordered_place_sequence``, endpoint labels, the via-road, the destination
    POI, and the dominant county/state — split into individual words. These are
    the ONLY proper-noun place tokens a channel-history phrase may carry and
    still be in-region ("Nelson County drive" for a Nelson LF). Empty when the
    LF is GPS-less/legacy, which turns the cross-region filter into a no-op.
    """
    fields: list = list(getattr(resolved, "regions", None) or [])
    fields += list(getattr(resolved, "ordered_place_sequence", None) or [])
    for attr in (
        "start_label",
        "end_label",
        "road",
        "destination",
        "dominant_county",
        "dominant_state",
    ):
        value = getattr(resolved, attr, None)
        if value:
            fields.append(value)
    tokens: set[str] = set()
    for field_value in fields:
        if isinstance(field_value, str):
            tokens.update(_TAG_WORD_RE.findall(field_value.lower()))
    return tokens


def _drop_cross_region_location_tags(
    candidate_tags: Optional[list], resolved: Optional[object]
) -> list[str]:
    """Filter channel-history phrase tags that name a DIFFERENT LF's region.

    The SEO optimizer merges the channel's "top-performing tags" into
    ``metadata.tags`` with no topical filter, so once the best performers are
    (say) Irish, every USA LF inherits Irish location phrases on its tag surface
    — a cross-region location leak. The original filter only caught phrases
    carrying a ``county``/``parish`` marker, so bare town/road phrases
    ("Kenmare drive", "Slea Head Road") from another region still leaked. Now a
    phrase is dropped when it contains a proper-noun WORD that is neither generic
    drive/scenic vocabulary nor one of THIS LF's own resolved place names — i.e.
    a place from somewhere the LF never drove. Generic phrases ("mountain pass
    drive") and in-region location phrases ("Nelson County road trip") survive;
    when the LF's places are unknown (GPS-less/legacy), nothing is dropped.
    """
    own = _own_place_tokens(resolved)
    if not own:
        return list(candidate_tags or [])
    out: list[str] = []
    for tag in candidate_tags or []:
        if isinstance(tag, str) and _has_foreign_place_word(tag, own):
            continue  # cross-region location tag → drop
        out.append(tag)
    return out


def _has_foreign_place_word(tag: str, own_tokens: set[str]) -> bool:
    """True if ``tag`` carries a place word from OUTSIDE this LF's route.

    A word triggers the leak verdict when it is 3+ chars, not purely numeric
    (road numbers are generic), not in the generic drive/scenic vocabulary, and
    not one of this LF's own resolved place tokens — the signature of a proper
    noun naming another region ("kerry", "dingle").
    """
    for word in _TAG_WORD_RE.findall(tag.lower()):
        if len(word) < 3 or word.isdigit():
            continue
        if word in _GENERIC_TAG_WORDS or word in own_tokens:
            continue
        return True
    return False


def _phrase_tags(candidate_tags: Optional[list], existing_lower: set[str]) -> list[str]:
    """Multi-word optimizer tags worth keeping (consume, don't discard).

    The optimizer word-splits the topic into single-word fragments
    (``"dingle"``, ``"toward"``) that violate tag integrity — those
    stay dropped. But its analytics-derived MULTI-WORD phrases (``"mountain
    pass drive"``) carry real search value and were previously discarded
    wholesale. Keep only space-bearing phrases not already present, mutating
    ``existing_lower`` so the caller's dedup stays consistent.
    """
    out: list[str] = []
    for t in candidate_tags or []:
        if not isinstance(t, str):
            continue
        t = t.strip()
        if " " in t and t.lower() not in existing_lower:
            out.append(t)
            existing_lower.add(t.lower())
    return out


def _quoted_tag_width(tags: list[str]) -> int:
    """YouTube/Metricool comma-joined tag width (a spaced tag is quoted, +2)."""
    return sum(len(t) + (2 if " " in t else 0) for t in tags) + max(0, len(tags) - 1)


def _cap_youtube_tags(tags: list[str], *, max_count: int = 30, max_chars: int = 480) -> list[str]:
    """Cap a long-form YouTube tag list to YouTube's count + 500-char limits.

    YouTube rejects a tag set whose comma-joined width exceeds 500 chars, and
    counts a tag CONTAINING A SPACE as quoted (+2); Metricool's import counts
    slightly higher, so keep a margin at 480 — mirroring the shorts guard
    (``social_generator._build_tags._yt_tag_len``). The LF generate paths cap
    COUNT only, and ``_enrich_with_route_profile`` appends GPS towns/roads/
    features AFTER that cap with NO length check, so an enriched LF could ship
    an over-limit tag set that YouTube/Metricool silently reject. Trim to
    ``max_count`` then drop lowest-priority (trailing) tags until under the
    width budget.
    """
    out = [t for t in (tags or []) if isinstance(t, str)][:max_count]
    while out and _quoted_tag_width(out) > max_chars:
        out.pop()
    return out


class LongFormMetadataGenerator:
    """
    Generate YouTube metadata for long form POV driving videos.

    Uses SEO Manager for analytics-driven optimization. Falls back to
    basic generation if SEO Manager is unavailable.
    """

    PUBLISH_DAYS = {
        "Monday": "mon",
        "Tuesday": "tue",
        "Wednesday": "wed",
        "Thursday": "thu",
        "Friday": "fri",
        "Saturday": "sat",
        "Sunday": "sun",
    }

    def __init__(
        self,
        seo_storage_path: Optional[Path] = None,
        credentials_dir: Optional[Path] = None,
        site_id: Optional[str] = None,
    ):
        """
        Args:
            seo_storage_path: Path to SEO data storage (for analytics/event store).
                              Defaults to ``config.seo.storage_path``.
            credentials_dir: Path to brand credentials directory.
                             Defaults to ``config.seo.credentials_dir``.
            site_id: SEO site identifier events/queries are tagged with.
                     None -> config.seo.site_id.
        """
        self.storage_path = seo_storage_path or DEFAULT_SEO_STORAGE
        self.credentials_dir = credentials_dir or DEFAULT_CREDENTIALS_DIR
        self.site_id = site_id if site_id is not None else _get_config().seo.site_id
        self._manager = None
        self._optimizer = None
        self._ledger = None
        self._seo_degraded = False
        self._init_seo_manager()
        self._init_ledger()

    def _init_seo_manager(self):
        """Initialize SEO Manager with YouTube adapter for GoMoveShift."""
        try:
            from seo_manager import SEOManager
            from seo_manager.analytics.seo_optimizer import SEOOptimizer
            from seo_manager.platforms.youtube import YouTubeAdapter

            self._manager = SEOManager(
                site_id=self.site_id,
                platform=YouTubeAdapter(),
                storage_path=self.storage_path,
            )

            # Verify the event store is readable (catches corrupt JSON)
            try:
                self._manager.store.query_by_site(self.site_id)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"  SEO event store corrupted ({e}) — METADATA QUALITY DEGRADED")
                self._manager = None
                self._optimizer = None
                self._seo_degraded = True
                return

            self._optimizer = SEOOptimizer(self._manager)
            logger.info("  SEO Manager initialized for gomoveshift")
        except ImportError:
            logger.error("  seo_manager not installed — METADATA QUALITY DEGRADED")
            self._manager = None
            self._optimizer = None
            self._seo_degraded = True
        except PermissionError as e:
            logger.error(
                "  SEO Manager blocked by permission error (check .lock file ownership): %s", e
            )
            self._manager = None
            self._optimizer = None
            self._seo_degraded = True
        except Exception as e:
            import traceback

            logger.error(f"  SEO Manager init FAILED: {e}")
            logger.error(f"  Traceback: {traceback.format_exc()}")
            logger.error("  Pipeline will use basic fallback — METADATA QUALITY DEGRADED")
            self._manager = None
            self._optimizer = None
            self._seo_degraded = True

    def generate(
        self,
        location: str,
        publish_day: str = "",
        video_duration: float = 0.0,
        project_name: str = "",
        title_format_index: int = 0,
        recording_date: Optional[str] = None,
        title_override: Optional[str] = None,
        route_profile: Optional[dict] = None,
        trim_ratio: Optional[float] = None,
        source_duration: Optional[float] = None,
        distance_km: Optional[float] = None,
        elevation_gain_m: Optional[float] = None,
        elevation_profile: Optional[dict] = None,
        speed: Optional[dict] = None,
        bearing: Optional[str] = None,
        regions: Optional[list] = None,
        has_gps: Optional[bool] = None,
        resolved_location: Optional[object] = None,
        full_drive_stats: Optional[list] = None,
        insert_count: int = 0,
        road_coverage_measured: bool = False,
        location_evidence: Optional[dict] = None,
        title_evidence: Optional[object] = None,
        title_part: Optional[tuple] = None,
        preselected_title_decision: Optional[dict] = None,
    ) -> dict:
        """
        Generate complete YouTube metadata for a long form video.

        Uses SEO Manager for analytics-driven generation when available,
        falls back to basic templates otherwise.

        Args:
            location: Location name (e.g., "County Tipperary, Ireland")
            publish_day: Day name (e.g., "Wednesday")
            video_duration: Video duration in seconds
            project_name: Project name for reference
            title_format_index: Which title format to use (0 = best performing)
            location_evidence: This LF's ``LocationEvidence`` record (dict form),
                the facts Waypost's title decision may claim.
            title_evidence: Waypost ``EvidenceSnapshot`` for the decision; none
                means no observations, so the route-only baseline ships.
            title_part: ``(part_index, part_count)`` of a split full-drive LF;
                the part suffix is part of the decided title, never appended
                after it.
            preselected_title_decision: Waypost decision approved against the
                retained route before encoding; its baseline must still match
                the metadata title assembled here.

        Returns:
            Dict with: title, description, tags, category, publish_day,
                       project_name, quality_score, content_id, generated_at

        Raises:
            ValueError: If `location` is empty, whitespace-only, None, or a
                placeholder literal (UNKNOWN / LOCATION_PLACEHOLDER / TODO /
                TBD / PLACEHOLDER, case-insensitive). The non-negotiable
                GPS/SEO correctness rule forbids rendering blank or
                sentinel location titles.
        """
        # Hard gate: reject empty / placeholder locations at entry so no
        # garbage title ever makes it to disk or YouTube. Keep the check
        # here (vs deeper in _generate_basic/_generate_with_seo_manager)
        # so all code paths share the same contract.
        if location is None or not str(location).strip():
            raise ValueError(
                "LongFormMetadataGenerator.generate: location is empty — "
                "refusing to produce placeholder metadata. Upstream must "
                "supply a real, GPS-verifiable location string."
            )
        _placeholder_literals = {
            "unknown",
            "location_placeholder",
            "placeholder",
            "todo",
            "tbd",
        }
        if str(location).strip().lower() in _placeholder_literals:
            raise ValueError(
                f"LongFormMetadataGenerator.generate: location {location!r} "
                f"is a placeholder literal — refusing to render. Upstream "
                f"must resolve the location before calling generate()."
            )
        # The placeholder set above catches only literal sentinels. A
        # broken geocode can also surface a coordinate pair ("34.1, -80.4"),
        # a null-island fix ("0.0, 0.0"), a literal None/null, or a pure-digit
        # string — none of which are placeholders but all of which would burn
        # garbage into the title/description. Reject them at the same entry
        # gate (whole-string match, so real names like "Todos Santos" pass).
        _loc_clean = str(location).strip()
        if (
            _BROKEN_LITERAL_PATTERN.match(_loc_clean)
            or _COORDINATE_PATTERN.match(_loc_clean)
            or _DIGITS_ONLY_PATTERN.match(_loc_clean)
        ):
            raise ValueError(
                f"LongFormMetadataGenerator.generate: location {location!r} is "
                f"a coordinate / null-island / geocoder-error string — refusing "
                f"to render. Upstream must resolve a real label first."
            )

        if self._manager:
            result = self._generate_with_seo_manager(
                location,
                publish_day,
                video_duration,
                project_name,
                recording_date=recording_date,
                title_override=title_override,
                trim_ratio=trim_ratio,
                source_duration=source_duration,
                distance_km=distance_km,
                elevation_gain_m=elevation_gain_m,
                elevation_profile=elevation_profile,
                speed=speed,
                bearing=bearing,
                regions=regions,
                resolved_location=resolved_location,
                full_drive_stats=full_drive_stats,
                insert_count=insert_count,
                road_coverage_measured=road_coverage_measured,
            )
        else:
            logger.warning(
                "⚠ longform_metadata: SEO Manager unavailable "
                "(seo_degraded=%s) — preferred path (analytics-driven metadata) "
                "unavailable; falling back to _generate_basic for location=%r "
                "(quality_score will be None; metadata quality degraded)",
                self._seo_degraded,
                location,
            )
            result = self._generate_basic(
                location,
                publish_day,
                video_duration,
                project_name,
                recording_date=recording_date,
                title_override=title_override,
                trim_ratio=trim_ratio,
                source_duration=source_duration,
                distance_km=distance_km,
                elevation_gain_m=elevation_gain_m,
                elevation_profile=elevation_profile,
                speed=speed,
                bearing=bearing,
                regions=regions,
                resolved_location=resolved_location,
                full_drive_stats=full_drive_stats,
                insert_count=insert_count,
                road_coverage_measured=road_coverage_measured,
            )

        # Enrich with route profile data (GPS-verified towns, roads, features)
        if route_profile:
            result = self._enrich_with_route_profile(
                result, route_profile, road_coverage_measured=road_coverage_measured
            )

        # Re-cap tags at the single choke point every generate path passes
        # through. The optimized/basic paths cap COUNT (:30) BEFORE the
        # enrichment above appended GPS place tags, and neither ever
        # length-checked — YouTube rejects a tag set over 500 chars (the shorts
        # path already guards this). Enforce the 30-count + <500-char budget on
        # the FINAL list so an enriched LF can never ship over-limit.
        if isinstance(result, dict) and result.get("tags"):
            result["tags"] = _cap_youtube_tags(result["tags"])

        # Stamp the GPS-presence flag onto the result so downstream
        # gates (notably ``check_metadata_quality``'s chapter-density
        # branch in ``validation/content_gates.py``) can skip checks
        # that are unsatisfiable on GPS-less drives. Single source of
        # truth: ``state["per_lf_gpx_path"][lf.index] is not None``,
        # threaded in by Phase D — NOT a heuristic on description
        # text. ``None`` means "caller did not assert either way" and
        # downstream falls through to the strict legacy behavior.
        if has_gps is not None:
            result["has_gps"] = bool(has_gps)

        # Stamp which stat(s) (if any) reflect the full (pre-cut)
        # drive rather than the LF's kept segments, so
        # ``check_description_redundancy``'s scope rule can require —
        # and, since the line above is rendered with the ``Full drive:``
        # prefix, always find — the matching prefix. Absent/empty (the
        # kept-segment default) omits the key entirely, matching the
        # gate's ``metadata_json.get("full_drive_stats") or []`` read.
        if full_drive_stats:
            result["full_drive_stats"] = list(full_drive_stats)

        # Waypost decides the title from verified facts; the record rides
        # with the metadata and the upload refuses any other title.
        if isinstance(result, dict) and result.get("title"):
            self._apply_title_decision(
                result,
                location_evidence,
                title_evidence,
                title_part,
                preselected_title_decision=preselected_title_decision,
            )

        # Enforce content safety — pipeline stops if any text is unsafe
        validator = ContentValidator()
        validator.enforce(
            titles=[result.get("title", "")],
            descriptions=[result.get("description", "")],
        )

        return result

    @staticmethod
    def _apply_title_decision(
        result, location_evidence, title_evidence, title_part, *, preselected_title_decision=None
    ) -> None:
        from .title_builder import append_full_drive_part_suffix
        from .title_decision import PreRenderTitleError, decide_title

        baseline = result["title"]
        if title_part:
            baseline = append_full_drive_part_suffix(baseline, *title_part)
        if preselected_title_decision is not None:
            if preselected_title_decision.get("baseline_title") != baseline:
                raise PreRenderTitleError(
                    f"pre-render title baseline {preselected_title_decision.get('baseline_title')!r} "
                    f"does not match metadata baseline {baseline!r}"
                )
            chosen = preselected_title_decision.get("chosen_title")
            if not isinstance(chosen, str) or not chosen.strip():
                raise PreRenderTitleError("pre-render title decision has no chosen title")
            result["title"] = chosen
            result["title_decision"] = preselected_title_decision
            return
        cut = baseline.rfind(" in 4K")
        decision = decide_title(
            "longform",
            baseline,
            location_evidence,
            suffix=baseline[cut:] if cut >= 0 else "",
            snapshot=title_evidence,
        )
        result["title"] = decision["chosen_title"]
        result["title_decision"] = decision

    @staticmethod
    def _enrich_with_route_profile(
        result: dict, route_profile: dict, road_coverage_measured: bool = False
    ) -> dict:
        """Enrich metadata with GPS-verified route profile data.

        Surfaces start/end-town journey arc, multi-state spans, and
        top POIs (features) in the description body — in addition to the
        towns/roads paragraph already produced. Roads cap is raised from 3
        to 5. All values come from ``RouteProfile`` fields the pipeline
        already computes; no new GPS work is done here.

        All enhancements are additive — existing metadata is preserved.
        The body line(s) added here include GPS-verified nouns (towns
        and/or roads and/or features), which are also threaded back into
        the metadata dict as ``chapter_towns`` / ``named_routes`` so the
        Phase D ``check_description_redundancy`` gate can validate the
        body line is not a pure restatement.
        """
        towns = list(route_profile.get("towns_traversed") or [])
        # Gate every road through the SAME quality check the title's
        # via-road slot uses (``_road_passes_quality`` — ref-numbered or
        # named through-road only). A street/lane/town road that a naive
        # share/distance floor let through must never reach prose or tags.
        roads = [r for r in (route_profile.get("roads_traversed") or []) if _road_passes_quality(r)]
        # Road coverage is not measured over retained time, so
        # by default no road reaches prose, ``named_routes`` or tags.
        if not road_coverage_measured:
            roads = []
        features = list(route_profile.get("features") or [])
        states = list(route_profile.get("states_traversed") or [])
        start_town = route_profile.get("start_town") or ""
        end_town = route_profile.get("end_town") or ""

        # ── Build the body lines ──────────────────────────────────────
        body_lines: list[str] = []

        # Line 1: journey arc (preferred) OR towns/roads paragraph (fallback)
        if start_town and end_town and start_town != end_town:
            # Journey-arc sentence. Intermediate towns = towns_traversed
            # minus the endpoints, each named ONCE in first-visit order.
            # towns_traversed keeps the true sequence (a loop re-enters a
            # town — a battlefield loop read "passing
            # through Freedom Township, Gettysburg, Gettysburg"); the 🛤️
            # Route line carries that sequence, the prose must not.
            seen_lower = {start_town.lower(), end_town.lower()}
            intermediate: list[str] = []
            for t in towns:
                if not t or t.lower() in seen_lower:
                    continue
                seen_lower.add(t.lower())
                intermediate.append(t)
            intermediate = intermediate[:5]
            line = f"Started in {start_town}, ended in {end_town}"
            if intermediate:
                line += f", passing through {', '.join(intermediate)}"
            line += "."
            body_lines.append(line)
            # Roads, if any — surface up to 5
            if roads:
                body_lines.append(f"Roads traveled: {', '.join(roads[:5])}.")
        elif towns:
            # Fallback to legacy "Driving through {towns} on {roads}." form.
            line = f"Driving through {', '.join(towns)}"
            if roads:
                line += f" on {', '.join(roads[:5])}"
            line += "."
            body_lines.append(line)
        elif roads:
            # Towns missing but roads known — still mention the roads.
            body_lines.append(f"Roads traveled: {', '.join(roads[:5])}.")

        # Multi-state surfacing
        if len(states) > 1:
            body_lines.append(f"Spans {len(states)} states: {', '.join(states)}.")

        # POI line (features in body, not just tags)
        if features:
            body_lines.append(f"Highlights along the route: {', '.join(features[:5])}.")

        # ── Insert body into description ──────────────────────────────
        if body_lines:
            desc_body = "\n".join(body_lines)
            description = result.get("description", "")
            if description:
                loc_marker = "\n📍 Location:"
                idx = description.find(loc_marker)
                if idx != -1:
                    description = description[:idx] + f"\n{desc_body}\n" + description[idx:]
                else:
                    description += f"\n\n{desc_body}"
            else:
                description = desc_body
            result["description"] = description

            # Thread GPS-verified nouns back into metadata so the Phase D
            # description-redundancy gate recognises the body's new nouns.
            # Don't clobber existing values that downstream stages may have
            # set with higher-fidelity data (chapter generator output).
            existing_chapter_towns = result.get("chapter_towns") or []
            if not existing_chapter_towns:
                # Combine endpoints + intermediates + features (all GPS-verified).
                new_nouns: list[str] = []
                seen_lower: set[str] = set()
                for t in [start_town, end_town, *towns, *features]:
                    if t and t.lower() not in seen_lower:
                        new_nouns.append(t)
                        seen_lower.add(t.lower())
                if new_nouns:
                    result["chapter_towns"] = new_nouns
            existing_named_routes = result.get("named_routes") or []
            if not existing_named_routes and roads:
                result["named_routes"] = roads[:5]

        # --- Tag enrichment (GPS-verified place names) ---
        existing_tags = result.get("tags", [])
        existing_lower = {t.lower() for t in existing_tags}

        profile_tags: list[str] = []
        for town in towns:
            if town.lower() not in existing_lower:
                profile_tags.append(town)
                existing_lower.add(town.lower())
        for road in roads[:5]:
            if road.lower() not in existing_lower:
                profile_tags.append(road)
                existing_lower.add(road.lower())
        for feat in features:
            if feat.lower() not in existing_lower:
                profile_tags.append(feat)
                existing_lower.add(feat.lower())

        result["tags"] = existing_tags + profile_tags

        return result

    def _init_ledger(self):
        """Load the content ledger and sync any new Metricool CSVs."""
        from .content_ledger import ContentLedger

        # Derive ledger path from storage_path so tests can redirect it
        self._ledger_path = self.storage_path / ".content_ledger.json"
        try:
            self._ledger = ContentLedger.load(self._ledger_path)
        except Exception as e:
            # The ledger file EXISTS but couldn't be read/parsed —
            # ContentLedger.load() already treats a genuinely missing file
            # as a fresh install (no exception). Degrading here silently
            # drops collision data and get_next_publish_slot() falls back to
            # its naive, calendar-blind calculation, which can double-book
            # an already-taken publish slot. Refuse instead.
            raise RuntimeError(
                f"Content ledger unreadable at {self._ledger_path}: {e} — "
                "refusing to schedule without collision data"
            ) from e
        try:
            ingested = self._ledger.ingest_all_metricool_csvs()
            if ingested > 0:
                self._ledger.save(self._ledger_path)
                logger.info(f"  Ledger: ingested {ingested} new entries from Metricool CSVs")
            else:
                logger.info(f"  Ledger: {len(self._ledger.entries)} entries loaded")
        except Exception as e:
            logger.error(f"  Ledger Metricool CSV sync failed: {e} — continuing with loaded ledger")

    def _sync_youtube_analytics(self):
        """
        Sync latest YouTube analytics before generating metadata.

        Fetches all channel videos and their performance stats, storing
        them in the event store so the optimizer has fresh data.

        MANDATORY: YouTube analytics must be synced for every publish.
        Credential sources (tried in order):
        1. SOPS-encrypted token ``youtube_upload_token.json.enc`` beside the credentials directory
        2. BrandCredentials (Fernet-encrypted in credentials/gomoveshift/)
        """
        if self._manager is None:
            logger.info("  SEO Manager not available — skipping analytics sync")
            return

        try:
            yt_creds = self._load_youtube_credentials()
        except FileNotFoundError:
            logger.warning("  YouTube credentials not available — skipping analytics sync")
            return

        try:
            import json
            import os
            import tempfile

            from seo_manager.analytics.youtube_sync import YouTubeAnalyticsSync

            # NamedTemporaryFile with delete=True guarantees unlink even on
            # SIGKILL between write + use, closing the leak window that the
            # prior mkstemp + finally-unlink path left open. YouTubeAnalyticsSync
            # requires a filesystem path (passes it to
            # Credentials.from_authorized_user_file), so in-memory dict-passing
            # would require a change to the SEO library —
            # out of scope here.
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                prefix="yt_creds_",
                delete=True,
            ) as creds_file:
                os.chmod(creds_file.name, 0o600)
                json.dump(yt_creds, creds_file)
                creds_file.flush()

                sync = YouTubeAnalyticsSync(creds_file.name)

                # List all channel videos
                video_mapping = self._fetch_channel_videos(sync)

                if not video_mapping:
                    logger.info("  No channel videos found to sync")
                    return

                logger.info(f"  Found {len(video_mapping)} channel videos")

                # Batch fetch stats (50 per API call) instead of 1-by-1
                self._batch_sync_stats(sync, video_mapping)

        except ImportError as e:
            logger.warning(f"  YouTube sync unavailable (missing dependency): {e}")
        except Exception as e:
            logger.warning(f"  YouTube analytics sync failed: {e}")
            logger.debug("  Sync error details:", exc_info=True)

    def _load_youtube_credentials(self) -> dict:
        """
        Load YouTube OAuth credentials.

        Delegates to the shared `_yt_creds.load_youtube_credentials` helper
        so analytics sync and YouTube upload use a single source of truth.
        """
        from ._yt_creds import load_youtube_credentials

        return load_youtube_credentials(self.credentials_dir)

    def _fetch_channel_videos(self, sync) -> dict:
        """
        Fetch all videos from the GoMoveShift YouTube channel.

        Uses the YouTube Data API to list uploads, returning a mapping
        of {video_id: content_id} for the sync.
        """
        try:
            # Get the channel's uploads playlist
            channels_response = (
                sync.youtube_data.channels()
                .list(
                    part="contentDetails",
                    mine=True,
                )
                .execute()
            )

            if not channels_response.get("items"):
                return {}

            uploads_playlist = channels_response["items"][0]["contentDetails"]["relatedPlaylists"][
                "uploads"
            ]

            # Fetch all videos from the uploads playlist
            video_mapping = {}
            next_page = None

            while True:
                playlist_response = (
                    sync.youtube_data.playlistItems()
                    .list(
                        part="snippet",
                        playlistId=uploads_playlist,
                        maxResults=50,
                        pageToken=next_page,
                    )
                    .execute()
                )

                for item in playlist_response.get("items", []):
                    video_id = item["snippet"]["resourceId"]["videoId"]
                    title = item["snippet"]["title"]
                    # Use a slugified title as content_id
                    content_id = (
                        title.lower()
                        .replace(" ", "-")
                        .replace("|", "")
                        .replace("—", "-")
                        .replace(":", "")
                        .strip("-")[:60]
                    )
                    video_mapping[video_id] = content_id

                next_page = playlist_response.get("nextPageToken")
                if not next_page:
                    break

            return video_mapping

        except Exception as e:
            logger.warning(f"  Failed to list channel videos: {e}")
            return {}

    # Analytics API v2 metrics pulled per eligible video (ANALYTICS_CTR_LIMITATION
    # applies here too — cardClickRate is the click-through proxy, never a fake
    # impression CTR). Quota-capped: reports.query is one call per video, so we
    # only spend it on videos likely to matter for the optimizer.
    _ANALYTICS_METRICS = (
        "views,estimatedMinutesWatched,averageViewDuration,"
        "averageViewPercentage,cardImpressions,cardClickRate,subscribersGained"
    )
    _ANALYTICS_MAX_VIDEOS = 200
    _ANALYTICS_RECENT_DAYS = 90

    def _is_analytics_eligible(self, index: int, published_at, now) -> bool:
        """Newest `_ANALYTICS_MAX_VIDEOS` by channel order, plus anything
        published within `_ANALYTICS_RECENT_DAYS` days — keeps the per-video
        Analytics API quota bounded on large channels."""
        if index < self._ANALYTICS_MAX_VIDEOS:
            return True
        if not published_at:
            return False
        try:
            published = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        except ValueError:
            return False
        return (now - published) <= timedelta(days=self._ANALYTICS_RECENT_DAYS)

    def _fetch_video_analytics_metrics(self, sync, video_id: str) -> Optional[dict]:
        """
        Fetch `_ANALYTICS_METRICS` from the YouTube Analytics API v2 for one
        video. Never raises — a missing scope, 403, or quota error just means
        no analytics fields get added to this video's event; the Data API
        sync path is unaffected.
        """
        from seo_manager.analytics.youtube_sync import analytics_report_window

        start_date, end_date = analytics_report_window()
        try:
            response = (
                sync.youtube_analytics.reports()
                .query(
                    ids="channel==MINE",
                    startDate=start_date,
                    endDate=end_date,
                    metrics=self._ANALYTICS_METRICS,
                    filters=f"video=={video_id}",
                )
                .execute()
            )
            rows = response.get("rows") or []
            if not rows:
                return None
            headers = [h["name"] for h in response.get("columnHeaders", [])]
            return dict(zip(headers, rows[0]))
        except Exception as e:
            logger.warning(f"  Analytics fetch failed for {video_id}: {e}")
            return None

    def _batch_sync_stats(self, sync, video_mapping: dict):
        """
        Fetch stats for all videos in batches of 50 (YouTube API limit)
        and store as performance events.

        Much more efficient than 1 API call per video. Also pulls per-video
        Analytics API v2 metrics (watch time / CTR proxies) for videos within
        the quota cap — see `_is_analytics_eligible`.
        """
        from datetime import datetime, timezone

        from seo_manager.analytics import metric_source
        from seo_manager.core.events import EventType, SEOEvent

        video_ids = list(video_mapping.keys())
        index_by_id = {vid: idx for idx, vid in enumerate(video_ids)}
        synced = 0
        now = datetime.now(timezone.utc)

        for i in range(0, len(video_ids), 50):
            batch = video_ids[i : i + 50]
            batch_str = ",".join(batch)

            try:
                response = (
                    sync.youtube_data.videos()
                    .list(
                        part="statistics,snippet",
                        id=batch_str,
                    )
                    .execute()
                )

                for item in response.get("items", []):
                    vid = item["id"]
                    content_id = video_mapping.get(vid, vid)
                    stats = item.get("statistics", {})
                    snippet = item.get("snippet", {})
                    published_at = snippet.get("publishedAt")

                    data = {
                        "video_id": vid,
                        "title": snippet.get("title", ""),
                        "views": int(stats.get("viewCount", 0)),
                        "likes": int(stats.get("likeCount", 0)),
                        "comments": int(stats.get("commentCount", 0)),
                        "favorites": int(stats.get("favoriteCount", 0)),
                    }
                    # Every ctr row says what it is a rate OF. cardClickRate is
                    # in-player card clicks among viewers already watching, not
                    # thumbnail impressions; unlabelled it read as "unknown".
                    data["ctr_source"] = metric_source.UNKNOWN
                    if published_at:
                        data["published_at"] = published_at

                    if self._is_analytics_eligible(index_by_id.get(vid, i), published_at, now):
                        analytics = self._fetch_video_analytics_metrics(sync, vid)
                        if analytics:
                            data["estimated_minutes_watched"] = analytics.get(
                                "estimatedMinutesWatched"
                            )
                            data["average_view_duration"] = analytics.get("averageViewDuration")
                            data["average_view_percentage"] = analytics.get("averageViewPercentage")
                            data["card_impressions"] = analytics.get("cardImpressions")
                            data["ctr"] = analytics.get("cardClickRate")
                            if data["ctr"] is not None:
                                data["ctr_source"] = metric_source.YOUTUBE_CARD_CLICK_RATE
                            data["subscribers_gained"] = analytics.get("subscribersGained")

                    event = SEOEvent(
                        event_id=f"yt_sync_{vid}",
                        event_type=EventType.PERFORMANCE_UPDATED,
                        site_id=self.site_id,
                        content_id=content_id,
                        timestamp=datetime.now().isoformat(),
                        data=data,
                    )
                    self._manager.store.upsert_event(event)
                    synced += 1

            except Exception as e:
                logger.warning(f"  Batch sync error (offset {i}): {e}")

        logger.info(f"  Synced stats for {synced}/{len(video_ids)} videos")

    def _generate_with_seo_manager(
        self,
        location: str,
        publish_day: str,
        video_duration: float,
        project_name: str,
        recording_date: Optional[str] = None,
        title_override: Optional[str] = None,
        trim_ratio: Optional[float] = None,
        source_duration: Optional[float] = None,
        distance_km: Optional[float] = None,
        elevation_gain_m: Optional[float] = None,
        elevation_profile: Optional[dict] = None,
        speed: Optional[dict] = None,
        bearing: Optional[str] = None,
        regions: Optional[list] = None,
        resolved_location: Optional[object] = None,
        full_drive_stats: Optional[list] = None,
        insert_count: int = 0,
        road_coverage_measured: bool = False,
    ) -> dict:
        """Generate metadata using SEO Manager with analytics optimization."""
        from seo_manager import ContentType

        # Sync latest YouTube analytics before generating
        logger.info("  Syncing YouTube analytics...")
        self._sync_youtube_analytics()

        # Format duration for description
        duration_str = self._format_duration(video_duration)

        # Feed the optimizer THIS LF's real route facts (place sequence,
        # regions, distance) instead of fixed boilerplate, so its analytics-
        # derived keywords reflect the actual drive.
        body = _optimizer_body(
            location,
            duration_str,
            regions=regions,
            distance_km=distance_km,
            resolved=resolved_location,
        )

        # Try optimizer first (uses analytics patterns)
        metadata = None
        if self._optimizer:
            try:
                self._optimizer.analyze_patterns(min_views=0)
                metadata = self._optimizer.optimize_metadata(
                    content_type=ContentType.VIDEO,
                    topic=location,
                    body=body,
                    base_tags=GMS_BASE_TAGS,
                    category="Travel & Events",
                )
                logger.info("  Using analytics-optimized metadata")
            except Exception as e:
                logger.debug(f"  Optimizer failed ({e}), falling back to standard generation")

        # Fall back to standard SEO Manager generation
        if metadata is None:
            metadata = self._manager.generate(
                content_type=ContentType.VIDEO,
                topic=location,
                body=body,
                base_tags=GMS_BASE_TAGS,
                category="Travel & Events",
            )

        # Save to event store for future analytics
        event_id = None
        try:
            event_id = self._manager.save(metadata)
            logger.info(f"  Saved to event store: {event_id}")
        except Exception as e:
            logger.warning(f"  Failed to save to event store: {e}")

        # Calculate next publish slot (ledger-aware) — needed for description
        next_slot = self.get_next_publish_slot("longform")

        # Enrich description with GMS-specific structure
        description = self._build_gms_description(
            location=location,
            seo_description=metadata.description,
            duration_str=duration_str,
            publish_day=publish_day,
            publish_date=next_slot["date"] if next_slot else "",
            recording_date=recording_date,
            trim_ratio=trim_ratio,
            source_duration=source_duration,
            video_duration=video_duration,
            distance_km=distance_km,
            elevation_gain_m=elevation_gain_m,
            elevation_profile=elevation_profile,
            speed=speed,
            bearing=bearing,
            regions=regions,
            resolved_location=resolved_location,
            full_drive_stats=full_drive_stats,
            insert_count=insert_count,
            road_coverage_measured=road_coverage_measured,
        )

        # Transactional booking: NO ledger write here (metadata-gen is
        # pre-upload → phantom slots). The LF entry books after YouTube
        # confirms; ``next_slot`` above feeds description text only.

        # The route-only baseline title (the optimizer's title is not an
        # input; ``generate`` hands this baseline to Waypost's select_title).
        if title_override:
            # An explicit override is the canonical (verb-less)
            # location string (per_lf_ost_text). Route it through
            # _build_title so the trim-based verb ("Highlights from" /
            # "Driving") is still applied — don't bypass it. Strip any
            # leading verb first to stay idempotent.
            title = self._build_title(
                location=_strip_leading_verb(title_override),
                trim_ratio=trim_ratio,
                resolved=resolved_location,
            )
        else:
            title = self._build_title(
                location=location,
                trim_ratio=trim_ratio,
                resolved=resolved_location,
            )

        # Curated tags — base tags + whole-phrase location tags. The SEO Manager's single-word fragments (``"dingle"``,
        # ``"peninsula"``, ``"toward"``) stay discarded — they violate content
        # integrity on the YouTube tag surface. This reverses the WHOLESALE
        # discard for the optimizer's MULTI-WORD analytics phrases only: those
        # carry real search value and now survive (``_phrase_tags``). The basic
        # fallback path uses the same curated core; the only intended
        # divergence is the optimizer's surviving multi-word phrases.
        tags = list(GMS_BASE_TAGS)
        for tag in _location_tags(location, recording_date):
            if tag.lower() not in [t.lower() for t in tags]:
                tags.append(tag)
        # Consume the optimizer's analytics-derived multi-word phrases.
        # First region-filter them — the optimizer merges the channel's
        # "top-performing tags" with no topical filter, so a cross-region
        # location phrase ("County Kerry drive") would otherwise leak onto
        # every LF regardless of where THIS one drives.
        _existing_lower = {t.lower() for t in tags}
        _region_safe_tags = _drop_cross_region_location_tags(
            getattr(metadata, "tags", None), resolved_location
        )
        tags.extend(_phrase_tags(_region_safe_tags, _existing_lower))
        tags = tags[:30]

        # Recalculate quality score on final title/description/tags
        # (the optimizer scored its own output, but we override title +
        # description + tags)
        from seo_manager.generation.base import BaseSEOGenerator

        quality_score = BaseSEOGenerator()._calculate_quality_score(title, description, tags)

        return {
            "title": title,
            "description": description,
            "tags": tags,
            "category": "Travel & Events",
            "publish_day": publish_day,
            "project_name": project_name,
            "quality_score": quality_score,
            "keyword_density": metadata.keyword_density,
            "readability_score": metadata.readability_score,
            "content_id": metadata.content_id,
            "event_id": event_id,
            "generated_at": datetime.now().isoformat(),
            "next_publish": next_slot,
        }

    @staticmethod
    def _build_title(
        location: str,
        trim_ratio: Optional[float] = None,
        resolved: Optional[object] = None,
    ) -> str:
        """Build a route-only, SEO-optimized YouTube title for one LF.

        DECIDED shape ``"{A} to {B}, {ST} in 4K"`` (55-65 char SERP target) —
        route + "in 4K", no genre word. The trim-honesty verb moved OUT of the
        description's 🎬 Final cut line, so ``trim_ratio`` stays in the signature
        for compat but no longer alters the title. The optimizer's title is no
        input at all: this is the route-only baseline, and Waypost's
        ``select_title`` decides the published title. The burned-OST/
        thumbnail text (``per_lf_ost_text``) is a separate string — unaffected.
        ``_clamp_title_len`` still guards the 100-char limit + ``(Part N of M)``.

        ``ensure_unique_titles`` appends the ``(Part N of M)`` dedup suffix
        to ``location`` BEFORE this rebuild (``phase_d_metadata`` →
        ``title_override``). ``_front_loaded_title`` reshapes the label into the
        SEO template and would otherwise parse the trailing ``(Part N of M)``
        into the route noun (``"… (Part 1 of 2) in 4K"`` mangled) — or drop it
        — so two same-county LFs collapse to one mangled/identical
        title. Detach the suffix here, rebuild from the clean base, then
        re-append so it lands at the tail where ``_clamp_title_len`` preserves
        it within the 100-char budget.
        """
        m = _PART_SUFFIX_RE.search(location or "")
        part_suffix = m.group(0).strip() if m else ""
        base = location[: m.start()].rstrip() if m else location
        title = _front_loaded_title(base, resolved)
        if part_suffix:
            title = f"{title} {part_suffix}"
        return _clamp_title_len(title)

    @staticmethod
    def _build_factual_body(
        location: str,
        duration_str: str,
        terrain_type: Optional[str] = None,
        chapter_towns: Optional[list[str]] = None,
        named_routes: Optional[list[str]] = None,
        road_coverage_measured: bool = False,
    ) -> str:
        """Build a description body line using only verifiable facts.

        Only references data that the pipeline can prove: GPS-derived terrain
        class, Overpass-verified town names, Overpass-verified route names,
        and measured duration. Never describes visual content — we can't see
        the video.
        """
        # GPS-provable terrain adjectives (only high-confidence classes)
        _terrain_adj = {
            "coastal": "coastal",
            "mountain": "mountain",
            "winding": "winding",
        }

        if not road_coverage_measured:
            named_routes = None  # an unmeasured road is not a verifiable fact
        # Without GPS-derived context, the body would only restate
        # facts already in the 📍 / 🎬 / 📅 block. Return empty so the
        # caller omits the body block entirely.
        if not chapter_towns and not named_routes and terrain_type not in _terrain_adj:
            return ""

        # Build the opening phrase
        if named_routes:
            route_str = " and ".join(named_routes[:2])
            opener = f"A {duration_str} drive along {route_str} through {location}"
        elif terrain_type in _terrain_adj:
            adj = _terrain_adj[terrain_type]
            opener = f"A {duration_str} {adj} drive through {location}"
        else:
            opener = f"A {duration_str} drive through {location}"

        # Append town names if available (GPS + Overpass verified)
        if chapter_towns and len(chapter_towns) >= 2:
            if len(chapter_towns) == 2:
                town_str = f"{chapter_towns[0]} and {chapter_towns[1]}"
            else:
                town_str = f"{', '.join(chapter_towns[:-1])}, and {chapter_towns[-1]}"
            opener += f", passing through {town_str}"

        return f"{opener}."

    @staticmethod
    def rebuild_description_body(
        description: str,
        location: str,
        duration_str: str,
        terrain_type: Optional[str] = None,
        chapter_towns: Optional[list[str]] = None,
        named_routes: Optional[list[str]] = None,
        road_coverage_measured: bool = False,
    ) -> str:
        """Replace the body paragraph in a full description with updated facts.

        The body paragraph sits between the hook (first line) and the
        '📍 Location:' line, separated by blank lines. If no towns or routes
        are provided, returns the description unchanged.
        """
        if not chapter_towns and not named_routes:
            return description

        import re

        new_body = LongFormMetadataGenerator._build_factual_body(
            location=location,
            duration_str=duration_str,
            terrain_type=terrain_type,
            chapter_towns=chapter_towns,
            named_routes=named_routes,
            road_coverage_measured=road_coverage_measured,
        )

        # Replace body paragraph between hook and Location line:
        # the generator now omits the body paragraph when no GPS context
        # was available — in that case there's only ``\n\n📍 Location:``
        # after the hook with no prior body to match. Two-pass: first try
        # to replace an existing body; if no replacement happened, insert
        # a fresh body paragraph immediately before ``📍 Location:``.
        replaced, n_subs = re.subn(
            r"(?<=\n\n).+?(?=\n\n📍 Location:)",
            new_body,
            description,
            count=1,
            flags=re.DOTALL,
        )
        if n_subs == 0:
            replaced = description.replace(
                "\n\n📍 Location:",
                f"\n\n{new_body}\n\n📍 Location:",
                1,
            )

        # ── Hook enforcement: fix generic/subjective hooks ────────────
        _subjective_words = {
            "stunning",
            "beautiful",
            "amazing",
            "breathtaking",
            "gorgeous",
            "relaxing",
            "incredible",
            "spectacular",
            "magnificent",
            "picturesque",
        }
        _generic_starts = [
            "join me",
            "come along",
            "experience the",
            "discover the",
            "dive into",
            "insights, and discussions",
            "latest developments",
        ]
        lines = replaced.split("\n")
        if lines:
            _hook_lower = lines[0].lower()
            _is_generic = any(_hook_lower.startswith(gp) for gp in _generic_starts)
            _has_subjective = bool(_subjective_words.intersection(_hook_lower.split()))
            _towns = chapter_towns or []
            _town_in_hook = any(t.lower() in _hook_lower for t in _towns) if _towns else False
            if _is_generic or _has_subjective or (_towns and not _town_in_hook):
                region = location.split(",")[0].strip()
                if len(_towns) >= 2:
                    lines[0] = f"POV drive from {_towns[0]} to {_towns[-1]} through {region}."
                else:
                    lines[0] = f"POV drive through {region}."
                replaced = "\n".join(lines)

        return replaced

    def _build_gms_description(
        self,
        location: str,
        seo_description: str,
        duration_str: str,
        publish_day: str,
        publish_date: str = "",
        recording_date: Optional[str] = None,
        chapters: Optional[list] = None,
        terrain_type: Optional[str] = None,
        chapter_towns: Optional[list[str]] = None,
        named_routes: Optional[list[str]] = None,
        trim_ratio: Optional[float] = None,
        source_duration: Optional[float] = None,
        video_duration: Optional[float] = None,
        distance_km: Optional[float] = None,
        elevation_gain_m: Optional[float] = None,
        elevation_profile: Optional[dict] = None,
        speed: Optional[dict] = None,
        bearing: Optional[str] = None,
        regions: Optional[list] = None,
        resolved_location: Optional[object] = None,
        full_drive_stats: Optional[list] = None,
        insert_count: int = 0,
        road_coverage_measured: bool = False,
    ) -> str:
        """
        Build GoMoveShift-specific YouTube description.

        Combines SEO manager's optimized hook with GMS branding.
        When chapters are provided and the hook mentions "stunning roads",
        replaces it with a route-specific "POV drive from X to Y" hook.
        When terrain_type/chapter_towns are provided, builds a fact-based body.
        """
        # An operator ``--per-lf-location`` title ("To Clonakilty,
        # County Cork") is not a place noun — the DESCRIPTION (hook, SEO
        # line, tags, hashtags, factual body, "📍 Location:") reads the
        # parsed place everywhere below. The video TITLE/OST burn is built
        # from the separate ``per_lf_ost_text``/``title_override`` source and
        # keeps the operator's title shape untouched.
        location = parse_lf_label(location).place
        # Extract the first line from SEO description as the hook
        seo_lines = seo_description.split("\n")
        hook = seo_lines[0] if seo_lines else f"POV drive through {location}"

        # When ``trim_ratio`` is provided, the hook is
        # banded (Highlights vs POV drive) and overrides any SEO-derived
        # hook so framing is deterministic from the trim band — viewers
        # can't be misled by SEO boilerplate.
        if trim_ratio is not None:
            source_pretty = (
                self._format_duration(source_duration)
                if source_duration is not None and source_duration > 0
                else duration_str
            )
            if trim_ratio < HIGHLIGHTS_TRIM_THRESHOLD:
                hook = f"Highlights from {source_pretty} of driving through {location}."
            else:
                hook = f"POV drive through {location} ({duration_str})."

        # ── Hook quality gate: no generic/subjective hooks ──────────
        _subjective_words = {
            "stunning",
            "beautiful",
            "amazing",
            "breathtaking",
            "gorgeous",
            "relaxing",
            "incredible",
            "spectacular",
            "magnificent",
            "picturesque",
        }
        _generic_starts = [
            "join me",
            "come along",
            "experience the",
            "discover the",
            "dive into",
            "insights, and discussions",
            "latest developments",
        ]

        _hook_lower = hook.lower()
        _is_generic = any(_hook_lower.startswith(gp) for gp in _generic_starts)
        _has_subjective = bool(_subjective_words.intersection(_hook_lower.split()))

        # Collect GPS-verified towns from chapters and chapter_towns param
        import re as _re_ch

        _hook_towns: list[str] = []
        if chapter_towns:
            _hook_towns = list(chapter_towns)
        elif chapters:
            for ch in chapters:
                m = _re_ch.match(
                    r"^Entering\s+(.+?),\s+(?:County\s+\w+|\w+\s+County)$",
                    ch.get("title", ""),
                )
                if m:
                    _hook_towns.append(m.group(1))

        _town_in_hook = any(t.lower() in _hook_lower for t in _hook_towns) if _hook_towns else False

        # Cleaned label components (via-road/series/verb stripped) — the shared
        # root for both the region noun below and the geo hashtags further down.
        _comp = _label_components(location)

        # When trim_ratio is provided, the banded hook is the
        # source of truth — skip the legacy generic/subjective rewrite
        # so the trim band's framing survives.
        if trim_ratio is None and (
            _is_generic or _has_subjective or (_hook_towns and not _town_in_hook)
        ):
            # Region noun = the trailing region component, NOT parts[0]. For a
            # route-form label parts[0] is the "A to B via Road" arc, which
            # published a garbled "…through Killarney to Dingle via N86." Use
            # the first component only when it is not itself an arc; else the
            # trailing region. Same cleaned root the geo hashtags use.
            if _comp and " to " not in _comp[0]:
                region = _comp[0]
            elif len(_comp) >= 2:
                region = _comp[-1]
            else:
                region = ""
            if len(_hook_towns) >= 2:
                _arc = f"from {_hook_towns[0]} to {_hook_towns[-1]}"
                hook = f"POV drive {_arc} through {region}." if region else f"POV drive {_arc}."
            elif region:
                hook = f"POV drive through {region}."
            else:
                hook = (
                    f"POV drive through {_comp[0]}."
                    if _comp
                    else f"POV drive through {location.strip()}."
                )

        # The SEO-critical above-the-fold line 2. The trim-banded
        # hook (line 1) is preserved verbatim; this line guarantees the
        # evergreen "4K drive" term + the region noun + the GPS-verified
        # town arc + the via-road sit in the first two lines YouTube renders
        # before the "…more" fold. Empty (line omitted) when there's no usable
        # location data.
        seo_line = _seo_search_line(
            location,
            resolved=resolved_location,
            named_routes=named_routes,
            road_coverage_measured=road_coverage_measured,
        )

        # Geo hashtags: one per GPS-clean location component (town-arc endpoints
        # + region), reusing the shorts' _location_hashtags so a route-form
        # label "A to B via Road, Region" yields #A #B #Region — never the
        # smashed "#AtoBviaRoad" the raw parts[0] split produced. Splitting the
        # arc on " to " (via-road already stripped by _label_components) makes
        # each town its own tag. A single-component label emits one geo hashtag
        # and no meaningless "#Travel" filler.
        from .social_generator import _location_hashtags

        _geo_label = ", ".join(t for c in _comp for t in c.split(" to "))
        geo_hashtags = _location_hashtags(_geo_label)

        # Build recording date line if available.
        # Standardized format: "📅 Filmed: <date>" matches legacy Ireland
        # production format and the resolve-pipeline metricool_exporter
        # template — single canonical format across the codebase. The per-LF
        # date survives a resume.
        # Route through the SHARED formatter so long-form matches the
        # child shorts' "Mon YYYY" shape (was raw ISO here) and an
        # unparseable value is rejected, not burned. Single source of truth
        # is ``social_generator.format_recording_date``.
        from .social_generator import format_recording_date

        _filmed = format_recording_date(recording_date)
        recording_line = f"📅 Filmed: {_filmed}\n" if _filmed else ""

        # Always-on disclosure line. When the source
        # clip duration is known, surface "🎬 Final cut: X (edited from Y
        # of footage)" so the trim ratio is explicit on the page —
        # viewers can't be misled even if the title-band logic ever
        # drifts. When ``source_duration`` is unknown (legacy callers
        # not yet threading the value), fall back to the legacy
        # "🎬 Duration:" line; the disclosure regex won't match but no
        # honesty claim is implied.
        if source_duration is not None and source_duration > 0:
            _final_pretty = (
                self._format_duration(video_duration)
                if video_duration is not None and video_duration > 0
                else duration_str
            )
            _source_pretty = self._format_duration(source_duration)
            # Honesty contract: when there was no effective edit, emitting "edited
            # from <X> of footage" is a false claim AND produces a 3× duration
            # redundancy (hook + Final cut + edited-from). Decide on NUMERIC
            # seconds, not formatted-string equality — two materially different
            # durations can format identically (e.g. "30 min") and the legacy
            # string compare would then HIDE a real edit. Only fall back to the
            # string compare when no numeric final duration is available.
            if video_duration is not None and video_duration > 0:
                _edited = (source_duration - video_duration) >= _TRIM_DISCLOSURE_MIN_DELTA_SEC
            else:
                _edited = _final_pretty != _source_pretty
            if _edited:
                duration_line = (
                    f"🎬 Final cut: {_final_pretty} (edited from {_source_pretty} of footage)\n"
                )
            else:
                duration_line = f"🎬 Duration: {_final_pretty}\n"
        else:
            duration_line = f"🎬 Duration: {duration_str}\n"

        # GPS-derived distance.
        # Country-aware unit selection — US uses mi only, non-US uses
        # ``km (mi)``. Empty string when distance is unknown so the line
        # is simply omitted. Inserts between ``🎬 Final cut`` and
        # ``📅 Filmed``.
        _full_drive_stats = full_drive_stats or []
        distance_line = _format_distance_line(
            distance_km,
            location,
            resolved=resolved_location,
            full_drive="distance" in _full_drive_stats,
        )

        # Elevation lines (scalar ⛰️ gain + 📈 profile range) DROPPED:
        # low SEO value and GPS-noise-inflated on flat drives
        # (a flat coastal drive reported a fabricated ~1,600 ft of gain).
        # The underlying ``elevation_gain_m`` / ``elevation_profile`` state is
        # still computed + persisted (route-stats sidecar + invariants consume
        # it), just no longer rendered into the description.

        # Avg-speed/character ``🚗 Pace:`` line DROPPED from the
        # long-form description — a bare average speed over an
        # unwindowed GPS trace reads as a false precision claim, and the
        # derived "character" (highway cruise / winding roads / town pace)
        # duplicates the terrain descriptor already in the title/body. The
        # ``_format_speed_line`` / ``_speed_character`` helpers and the
        # underlying ``speed`` state are unchanged — they're just no longer
        # spliced into the rendered description.

        # Net travel heading ``🧭 Heading:`` line (wired from
        # _compass_from_waypoints). Empty for loop drives / GPS-less LFs.
        bearing_line = _format_bearing_line(bearing, location, resolved=resolved_location)

        # GPS-derived regions
        # (counties / states) the route passes through. Single ``🗺️ Through:``
        # label across countries; admin level (state vs county) is selected
        # upstream per-LF based on country derivation. Empty when ``regions``
        # is None / [] (GPS-less drives, geocoder unavailable, or no admin
        # match). Capped at ``_REGIONS_DESCRIPTION_CAP`` entries with overflow
        # shown as ``… +N more``; the LF's own label is filtered out first.
        # Inserts between ``⛰️ Elevation`` and ``📅 Filmed``.
        regions_line = _format_regions_line(regions, location)

        # Town-level ``🛤️ Route:`` sequence from the canonical resolver
        # (ordered_place_sequence). Finer granularity than the county-level
        # ``🗺️ Through:`` line above; empty for GPS-less / single-point LFs.
        route_line = _format_route_line(resolved_location)
        # The ``📸`` line names how many photo/clip inserts the cut carries.
        insert_line = _format_insert_line(insert_count)

        # Build factual body line from verifiable data only
        body_line = self._build_factual_body(
            location=location,
            duration_str=duration_str,
            terrain_type=terrain_type,
            chapter_towns=chapter_towns,
            named_routes=named_routes,
            road_coverage_measured=road_coverage_measured,
        )

        # Build hashtags from GMS_BASE_TAGS + location/country (not hardcoded)
        def _tag_to_hashtag(tag: str) -> str:
            return "#" + "".join(word.capitalize() for word in tag.split())

        # The per-component geo hashtags lead so they SURVIVE the 12-hashtag
        # slice below. They were previously appended AFTER the 12 base hashtags,
        # so the [:12] cap sliced the only drive-specific discovery hashtags
        # right off. Geo first → evergreen base fills the remainder; YouTube
        # shows the first 3 above the title.
        hashtag_parts = list(geo_hashtags)
        hashtag_parts += [_tag_to_hashtag(t) for t in GMS_BASE_TAGS]
        # Deduplicate while preserving order
        seen_ht: set[str] = set()
        unique_hashtags: list[str] = []
        for ht in hashtag_parts:
            if ht.lower() not in seen_ht:
                seen_ht.add(ht.lower())
                unique_hashtags.append(ht)
        hashtag_line = " ".join(unique_hashtags[:12])

        # When ``body_line`` is empty (no GPS-derived context), omit
        # the body block entirely so we go straight from the hook to
        # 📍 Location with a single blank-line separator.
        body_block = f"{body_line}\n\n" if body_line else ""

        # Config-driven footer (playlist link / cadence / site), keyed by
        # the SAME derived region as playlist auto-assignment — the shared
        # ``region_resolver.derive_region`` off the canonical
        # ResolvedLocation, never a fresh GPX/label re-derive. All config
        # keys default empty → empty block → byte-identical description.
        # Renders between the CTA block and the hashtag divider, so
        # ``_embed_chapters`` (inserts before the FIRST hashtag) still
        # lands chapters after the footer and before the hashtags.
        from .region_resolver import derive_region

        _footer_cfg = _get_config().drive
        _footer_region = derive_region(location, resolved=resolved_location)
        _footer_lines: list[str] = []
        _playlist_url = _footer_cfg.playlist_url_by_region.get(_footer_region, "")
        if _playlist_url:
            _region_display = "USA" if _footer_region == "usa" else _footer_region.title()
            _footer_lines.append(f"▶ Full {_region_display} playlist: {_playlist_url}")
        _cadence = _footer_cfg.cadence_line_by_region.get(_footer_region, "")
        if _cadence:
            _footer_lines.append(_cadence)
        if _footer_cfg.site_url:
            _footer_lines.append(f"🌍 {_footer_cfg.site_url}")
        footer_block = ("\n".join(_footer_lines) + "\n\n") if _footer_lines else ""

        return (
            f"{hook}\n"
            f"{seo_line}"
            f"\n"
            f"{body_block}"
            f"📍 Location: {location}\n"
            f"{duration_line}"
            f"{distance_line}"
            f"{bearing_line}"
            f"{regions_line}"
            f"{route_line}"
            f"{insert_line}"
            f"{recording_line}"
            f"\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"\n"
            f"🔔 Subscribe for more POV driving videos!\n"
            f"👍 Like this video if you enjoyed the drive\n"
            f"💬 Comment your favorite moment!\n"
            f"\n"
            f"{footer_block}"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"\n"
            f"{hashtag_line}"
        )

    @staticmethod
    def _dedup_chapters(description: str) -> str:
        """Strip duplicate '📍 CHAPTERS' blocks, keeping only the last one."""
        marker = "📍 CHAPTERS"
        parts = description.split(marker)
        if len(parts) <= 2:
            return description
        # Keep everything before the first marker + the last marker + its content
        before_first = parts[0]
        last_block = parts[-1]
        return before_first.rstrip("\n") + f"\n\n{marker}{last_block}"

    @staticmethod
    def _embed_chapters(description: str, chapter_text: str) -> str:
        """Replace any existing chapter block in description, or insert before hashtags."""
        import re

        marker = "📍 CHAPTERS"
        # Strip existing chapter block(s) — everything from marker to next double-newline or end
        pattern = re.compile(r"\n*" + re.escape(marker) + r".*?(?=\n\n[^\d]|\Z)", re.DOTALL)
        cleaned = pattern.sub("", description)

        chapter_block = f"\n\n{marker}\n{chapter_text}"

        # Insert chapters BEFORE hashtag block (hashtags must be last in YouTube descriptions)
        hashtag_match = re.search(r"\n\n(#\w+\s)", cleaned)
        if hashtag_match:
            insert_pos = hashtag_match.start()
            return cleaned[:insert_pos] + chapter_block + cleaned[insert_pos:]

        return cleaned.rstrip("\n") + chapter_block

    def _generate_basic(
        self,
        location: str,
        publish_day: str,
        video_duration: float,
        project_name: str,
        recording_date: Optional[str] = None,
        title_override: Optional[str] = None,
        trim_ratio: Optional[float] = None,
        source_duration: Optional[float] = None,
        distance_km: Optional[float] = None,
        elevation_gain_m: Optional[float] = None,
        elevation_profile: Optional[dict] = None,
        speed: Optional[dict] = None,
        bearing: Optional[str] = None,
        regions: Optional[list] = None,
        resolved_location: Optional[object] = None,
        full_drive_stats: Optional[list] = None,
        insert_count: int = 0,
        road_coverage_measured: bool = False,
    ) -> dict:
        """Fallback: basic metadata generation without SEO Manager."""
        duration_str = self._format_duration(video_duration)

        if title_override:
            # Prepend the trim-based verb to the canonical override
            # (idempotent via _strip_leading_verb) instead of bypassing
            # _build_title.
            title = self._build_title(
                location=_strip_leading_verb(title_override),
                trim_ratio=trim_ratio,
                resolved=resolved_location,
            )
        else:
            title = self._build_title(
                location=location, trim_ratio=trim_ratio, resolved=resolved_location
            )

        next_slot = self.get_next_publish_slot("longform")

        description = self._build_gms_description(
            location=location,
            seo_description=f"Join me on a relaxing POV drive through {location}!",
            duration_str=duration_str,
            publish_day=publish_day,
            publish_date=next_slot["date"] if next_slot else "",
            recording_date=recording_date,
            trim_ratio=trim_ratio,
            source_duration=source_duration,
            video_duration=video_duration,
            distance_km=distance_km,
            elevation_gain_m=elevation_gain_m,
            elevation_profile=elevation_profile,
            speed=speed,
            bearing=bearing,
            regions=regions,
            resolved_location=resolved_location,
            full_drive_stats=full_drive_stats,
            insert_count=insert_count,
            road_coverage_measured=road_coverage_measured,
        )

        # Curated tags — base tags + whole-phrase location tags.
        # Never word-splits the location string; fragments like "dingle" or
        # "peninsula" are unrepresentable here.
        tags = list(GMS_BASE_TAGS)
        for tag in _location_tags(location, recording_date):
            if tag.lower() not in [t.lower() for t in tags]:
                tags.append(tag)

        return {
            "title": title,
            "description": description,
            "tags": tags[:30],
            "category": "Travel & Events",
            "publish_day": publish_day,
            "project_name": project_name,
            "quality_score": None,
            "keyword_density": None,
            "readability_score": None,
            "content_id": project_name,
            "event_id": None,
            "seo_degraded": True,
            "generated_at": datetime.now().isoformat(),
        }

    def get_next_publish_slot(self, content_type: str = "longform") -> dict:
        """
        Calculate the next available publish slot based on the GMS schedule,
        accounting for already-scheduled content in the ledger.

        Args:
            content_type: "longform" or "shorts"

        Returns:
            Dict with: date, time, day_name, datetime_est, content_type
        """
        if self._ledger:
            slot = self._ledger.get_next_available_slot(content_type)
            if slot:
                return slot

        # No ledger: the same seo_manager search over an empty occupancy.
        if content_type == "longform":
            from .seo_slots import next_cadence_day

            publish_dt = next_cadence_day(
                _schedule_constants.LONG_FORM_DAYS,
                _schedule_constants.LONG_FORM_TIME,
                datetime.now(EST),
                include_today=True,
            )
            return {
                "date": publish_dt.strftime("%Y-%m-%d"),
                "time": publish_dt.strftime("%H:%M:%S"),
                "day_name": publish_dt.strftime("%A"),
                "datetime_est": publish_dt.isoformat(),
                "content_type": "longform",
            }

        return {}

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if minutes >= 60:
            hours = minutes // 60
            mins = minutes % 60
            return f"{hours}h {mins}m"
        return f"{minutes}m {secs}s"
