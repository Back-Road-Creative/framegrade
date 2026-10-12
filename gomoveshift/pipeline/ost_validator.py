"""Validation for derived OST text before rendering.

Catches known failure modes from the derivation pipeline:
  * empty / whitespace-only output
  * overly long text that won't fit the overlay
  * literal "None"/"null" strings from broken geocoder data
  * pure-digit or coordinate strings (broken geocode fallback)
  * orphaned trailing connector left by a title chop ("Killarney to")
  * HOA / subdivision leakage (belt-and-suspenders with RouteAnalyzer's filter)

Caller uses the (bool, reason) return to log a warning and fall back to a
safer default (state-only text, or empty) rather than burning the render
on garbage.
"""

import re
from dataclasses import dataclass

from gps_journey.analysis.route_analyzer import HOA_SUFFIX_PATTERNS

from .content_validator import ContentValidator
from .ffmpeg_filters import (
    FONT_PATH,
    TITLE_CONNECTOR_TOKENS,
    measure_text_width,
    wrap_text_for_ffmpeg,
)

_MAX_LENGTH_BY_FORMAT = {
    "lf": 60,
    "shorts": 40,
    # The burned per-LF title doubles as the YouTube title, which
    # ``title_builder`` enforces at MAX_TITLE_LEN=100. Gating it at the
    # 60-char OST-overlay limit would false-reject legitimate long titles,
    # so the burned-title gate uses this 100-char format for the length
    # check while still running every garbage-pattern check below.
    "lf_title": 100,
}
_DEFAULT_MAX_LENGTH = 60


@dataclass(frozen=True)
class _PixelSpec:
    """Rendered-pixel fit budget for one OST surface.

    Codepoint counts are only a fair proxy for narrow Latin glyphs; a run of
    wide glyphs (capital ``W``s, CJK) overflows the frame while staying under
    the codepoint cap. Each surface wraps the burned text against ``max_width_px``
    at ``font_size`` (PIL ``textbbox`` pinned to the exact ffmpeg ``font_path``
    — DejaVuSans-Bold, NOT a Liberation/Ubuntu fallback) and is
    considered un-fittable when it needs more than ``max_lines`` lines OR any
    single wrapped line still exceeds ``max_width_px`` (an unspaceable word).
    """

    font_size: int
    max_width_px: int
    max_lines: int
    font_path: str = FONT_PATH


# Long-form overlay: fontsize 48 at the 1920px reference width (``ost_applier``
# ``_LOCATION_FONTSIZE`` / ``_REF_WIDTH``), text area = width × ``_LOCATION_WIDTH_FRAC``
# (0.85), wrapped over at most ``_LOCATION_MAX_LINES`` (3). ``lf`` and ``lf_title``
# burn into the SAME overlay, so they share one pixel spec (their codepoint caps
# differ only because ``lf_title`` doubles as the 100-char YouTube title).
_LF_PIXEL_SPEC = _PixelSpec(font_size=48, max_width_px=int(1920 * 0.85), max_lines=3)
# Shorts overlay: fontsize 48 at 1080px width, 40px side margins (``ffmpeg_filters``
# ``LOC_FONT_SIZE`` / ``LOC_MAX_WIDTH``); 2-line cap for mobile readability.
_SHORTS_PIXEL_SPEC = _PixelSpec(font_size=48, max_width_px=1080 - 80, max_lines=2)

_FORMAT_PIXEL_SPEC = {
    "lf": _LF_PIXEL_SPEC,
    "lf_title": _LF_PIXEL_SPEC,
    "shorts": _SHORTS_PIXEL_SPEC,
}


def measure_wrapped_lines(text: str, format: str) -> int:
    """Number of lines ``text`` wraps to in ``format``'s overlay (pixel-based).

    Uses the same PIL/``FONT_PATH`` wrap machinery the render path uses, so the
    measurement matches the burned pixels. Unknown formats fall back to ``lf``.
    """
    spec = _FORMAT_PIXEL_SPEC.get(format, _LF_PIXEL_SPEC)
    lines = wrap_text_for_ffmpeg(text, spec.font_size, spec.max_width_px, spec.font_path)
    return len(lines)


def text_fits_overlay(text: str, format: str) -> bool:
    """True when ``text`` fits ``format``'s overlay after pixel word-wrapping.

    Fails when the wrapped block needs more than ``max_lines`` lines OR any
    single wrapped line still exceeds ``max_width_px`` (an unspaceable wide
    word — e.g. a long run of capital ``W``s). Degrades to ``True`` when PIL is unavailable
    (``measure_text_width`` returns ``-1``); the codepoint cap remains as the
    coarse backstop in that case.
    """
    spec = _FORMAT_PIXEL_SPEC.get(format, _LF_PIXEL_SPEC)
    lines = wrap_text_for_ffmpeg(text, spec.font_size, spec.max_width_px, spec.font_path)
    if len(lines) > spec.max_lines:
        return False
    for line in lines:
        w = measure_text_width(line, spec.font_size, spec.font_path)
        if w > spec.max_width_px:  # -1 (PIL missing) degrades to fit
            return False
    return True


# Literal words that indicate a missing/broken upstream value. Matched as
# whole-text (case-insensitive), not substring, so "Noneville" stays valid.
_BROKEN_LITERAL_PATTERN = re.compile(
    r"^\s*(none|null)(\s*,\s*(none|null))*\s*$",
    re.IGNORECASE,
)

# Placeholder / sentinel literals the pipeline must NEVER render.
# Whole-text match (case-insensitive) so real place names like "Unknowngate"
# or "Todos Santos" pass through unharmed.
_PLACEHOLDER_LITERALS = {
    "unknown",
    "location_placeholder",
    "placeholder",
    "todo",
    "tbd",
}
_PLACEHOLDER_PATTERN = re.compile(
    r"^\s*(" + "|".join(re.escape(lit) for lit in _PLACEHOLDER_LITERALS) + r")"
    r"(\s*,\s*(" + "|".join(re.escape(lit) for lit in _PLACEHOLDER_LITERALS) + r"))*\s*$",
    re.IGNORECASE,
)

# Pure coordinate strings — two signed decimal numbers with optional units.
_COORDINATE_PATTERN = re.compile(
    r"^\s*-?\d{1,3}(?:\.\d+)?\s*°?\s*[,\s]\s*-?\d{1,3}(?:\.\d+)?\s*°?\s*$"
)

# Pure digits only (no letters).
_DIGITS_ONLY_PATTERN = re.compile(r"^\s*\d+\s*$")

# Orphaned trailing connector — "Killarney to" — left when an upstream chop
# cut an "A to B via {road}" arc right after a joiner token. The chop sites
# strip these by construction (``ffmpeg_filters.strip_trailing_connectors``);
# this rule is the pre-burn backstop. Whole-token match (must follow
# whitespace/comma or start the text) so "Toronto" stays valid.
_TRAILING_CONNECTOR_PATTERN = re.compile(
    r"(?:^|[\s,])(" + "|".join(re.escape(t) for t in sorted(TITLE_CONNECTOR_TOKENS)) + r")[\s,]*$",
    re.IGNORECASE,
)

# Reuse the metadata safety validator on burned OST text too. The
# garbage/length/HOA checks above never look at hype or unsafe claims, so a
# hype phrase reaching the overlay via an OST override or a POI name
# ("Hidden Gem Rd") would otherwise burn unchallenged — the same claim the
# title/description layer already rejects. One shared instance; ContentValidator
# pre-compiles its patterns in __init__.
_CONTENT_VALIDATOR = ContentValidator()


def validate_ost(text: str, format: str) -> tuple[bool, str]:
    """Return (is_valid, reason). reason is empty when valid.

    Args:
        text: The derived OST string.
        format: "lf" or "shorts" — controls length limit. Unknown formats
            fall back to the long-form limit.
    """
    if text is None or not text.strip():
        return False, "empty or whitespace-only text"

    max_length = _MAX_LENGTH_BY_FORMAT.get(format, _DEFAULT_MAX_LENGTH)
    if len(text) > max_length:
        return False, f"length {len(text)} exceeds {format} limit of {max_length}"

    # Rendered-pixel fit: codepoints are a fair proxy only for narrow Latin
    # glyphs — a wide CJK/fullwidth run overflows the frame under the codepoint
    # cap. Measure with PIL against the exact ffmpeg FONT_PATH.
    if not text_fits_overlay(text, format):
        return False, f"text overflows {format} overlay frame width (rendered pixels)"

    if _BROKEN_LITERAL_PATTERN.match(text):
        return False, "broken geocoder output: literal None/null"

    if _PLACEHOLDER_PATTERN.match(text):
        return False, "placeholder literal (UNKNOWN/TODO/TBD/LOCATION_PLACEHOLDER)"

    if _DIGITS_ONLY_PATTERN.match(text):
        return False, "broken geocoder output: pure digits"

    if _COORDINATE_PATTERN.match(text):
        return False, "broken geocoder output: coordinate string"

    if _TRAILING_CONNECTOR_PATTERN.search(text):
        return False, "orphaned trailing connector token (chopped arc title)"

    for suffix in HOA_SUFFIX_PATTERNS:
        if suffix.lower() in text.lower():
            return False, f"contains HOA/subdivision pattern: {suffix.strip()!r}"

    # Hype / unsafe-claim gate (parity with metadata): reject before the irreversible
    # burn so an override/POI-borne hype word can't reach pixels the metadata
    # layer would have blocked.
    content_result = _CONTENT_VALIDATOR.validate(text, field_name="ost")
    if not content_result.valid:
        return False, f"unsafe content: {content_result.violations[0]}"

    return True, ""


def validate_ost_uniqueness(titles: list[str]) -> dict[str, list[int]]:
    """Check for duplicate or near-duplicate OST titles across episodes.

    Near-duplicates are titles that differ only by whitespace (leading,
    trailing, internal runs of spaces/tabs). Empty/whitespace-only titles
    are ignored entirely.

    Returns:
        A dict mapping the canonical (first-occurrence) title text to the
        list of indices where that title appears. Only entries with 2+
        occurrences are included. An empty dict means all titles are unique.
    """

    # Normalize: strip + collapse internal whitespace
    def _normalize(text: str) -> str:
        return " ".join(text.split())

    # Map normalized → (first-occurrence original text, list of indices)
    seen: dict[str, tuple[str, list[int]]] = {}

    for i, title in enumerate(titles):
        normed = _normalize(title)
        if not normed:
            continue
        if normed in seen:
            seen[normed][1].append(i)
        else:
            seen[normed] = (title.strip(), [i])

    return {
        original: indices for original, indices in (v for v in seen.values()) if len(indices) >= 2
    }
