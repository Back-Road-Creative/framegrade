"""OCR engine with EasyOCR and Claude fallback."""

import logging
import re
from typing import Optional

import cv2
import numpy as np

from ..config import Config
from ..models import ConfidenceLevel, DetectedLocation, LocationType
from .sign_evidence import SignClaim

logger = logging.getLogger(__name__)

# Captured tokens that describe infrastructure, not a place. Guards regex
# false-positives like "County Road 12" -> "Road" -> a bogus "Road County".
_CAPTURE_STOP_WORDS = frozenset(
    {
        "road",
        "rd",
        "route",
        "rte",
        "line",
        "street",
        "st",
        "avenue",
        "ave",
        "highway",
        "hwy",
        "motorway",
        "council",
        "lane",
        "drive",
        "way",
    }
)

# Trailing sign phrases that follow the placename; a greedy capture is truncated
# at the first one so "Welcome to Clifden Twinned with Boston" yields "Clifden",
# not the whole sign.
_SIGN_SUFFIX_PHRASES = (
    "twinned with",
    "twin town",
    "please drive",
    "drive carefully",
    "drive safely",
    "county council",
    "welcomes you",
    "welcomes careful",
)

# Text -> (capture, LocationType) table. Module-level so the multi-claim parser
# and ``OCREngine._parse_location`` read the SAME patterns; ``__init__`` keeps
# ``self.location_patterns`` pointing at this list.
_LOCATION_PATTERNS = [
    # "Welcome to X" or "Fáilte go X"
    (r"welcome\s+to\s+(.+)", LocationType.TOWN_ENTRY),
    (r"fáilte\s+go\s+(.+)", LocationType.TOWN_ENTRY),
    (r"failte\s+go\s+(.+)", LocationType.TOWN_ENTRY),
    # "X County" or "County X" or "Co. X"
    (r"(\w+)\s+county", LocationType.COUNTY_BOUNDARY),
    (r"county\s+(\w+)", LocationType.COUNTY_BOUNDARY),
    # ``co``/``co.`` must be its own token — a bare ``co\.?`` matched
    # inside words ("tesco express" -> "Express County").
    (r"(?:^|\s)co\.?\s+(\w+)", LocationType.COUNTY_BOUNDARY),
    # River/bridge crossings
    (r"(\w+)\s+river", LocationType.RIVER_CROSSING),
    (r"river\s+(\w+)", LocationType.RIVER_CROSSING),
    (r"(\w+)\s+bridge", LocationType.RIVER_CROSSING),
    # Distance signs (X 5km)
    (r"(\w+)\s+\d+\s*km", LocationType.ROAD_SIGN),
    # Town names with population
    (r"(\w+)\s+pop\.?\s*\d+", LocationType.TOWN_ENTRY),
]

# Claim kinds emitted by ``parse_sign_claims`` (plain ``LocationType`` values —
# ``SignClaim`` stays JSON-native).
_CLAIM_COUNTY = LocationType.COUNTY_BOUNDARY.value
_CLAIM_STATE = LocationType.STATE_BOUNDARY.value
_CLAIM_TOWN = LocationType.TOWN_ENTRY.value
_CLAIM_ROUTE = LocationType.ROUTE_MARKER.value

# A "Welcome to X" capture is greedy and swallows the rest of a multi-line sign
# ("Exampleville — Example County, SC"). The placename is the first segment; cut
# at the first list/clause delimiter. A spaced dash separates clauses, a bare
# hyphen does not ("Winston-Salem" survives).
_SIGN_SEGMENT_SPLIT_RE = re.compile(r"[,;|/\n·•]|\s[—–]\s|\s-\s")

# Tokens that mark a capture as a county sign rather than a town sign, so
# "Welcome to County Kerry" yields only the county claim.
_COUNTY_MARKER_TOKENS = frozenset({"county", "co", "co."})

# Route-marker shapes, matched CASE-SENSITIVELY against the raw OCR text:
# route shields are uppercase, and a case-insensitive match turns ordinary sign
# words into routes ("AHEAD in 2 mi" -> "IN-2"). A missed claim is far cheaper
# than a false one (a false claim costs the operator a bogus conflict warning).
#   US    — "I-95", "I 95", "US 17", "SC 64"; hyphenated when normalised.
#   Irish/UK — "N59", "M50", "R336", "A4"; stay unhyphenated.
_EURO_ROUTE_RE = re.compile(r"\b[NMRA]\d{1,4}\b")

# Lazily-built lookup tables (see ``_sign_tables``).
_SIGN_TABLES: dict = {}


def _sign_tables() -> dict:
    """State-name/abbreviation lookups + the US route regex, built once.

    ``pipeline.geo_constants.US_STATE_ABBREVS`` is the single source of truth
    for US state names and postal abbreviations. It is imported lazily because
    ``gomoveshift.pipeline``'s package init imports ``gomoveshift.analysis``,
    so a module-level import here would add an import edge back into a package
    that is mid-initialisation whenever analysis is imported first.
    """
    if _SIGN_TABLES:
        return _SIGN_TABLES

    from ..pipeline.geo_constants import US_STATE_ABBREVS

    names = list(US_STATE_ABBREVS)
    abbrevs = sorted(set(US_STATE_ABBREVS.values()))
    _SIGN_TABLES["name_by_lower"] = {n.lower(): n for n in names}
    _SIGN_TABLES["name_by_abbrev"] = {a: n for n, a in US_STATE_ABBREVS.items()}
    # Longest-first alternation so "West Virginia" wins over "Virginia".
    _SIGN_TABLES["name_re"] = re.compile(
        r"\b(?:"
        + "|".join(re.escape(n.lower()) for n in sorted(names, key=len, reverse=True))
        + r")\b"
    )
    _SIGN_TABLES["abbrev_re"] = re.compile(r"\b(?:" + "|".join(abbrevs) + r")\b")
    _SIGN_TABLES["us_route_re"] = re.compile(
        r"\b(I|US|" + "|".join(abbrevs) + r")[\s-]?(\d{1,3}[A-Z]?)\b"
    )
    return _SIGN_TABLES


def _normalize_route_token(token: str) -> str:
    """Normalise a sign's route token to the form the via-road vote uses.

    ``title_builder._road_ref_members`` is the source of truth for route-ref
    normalisation (``"I 95"`` -> ``"I-95"``, an unhyphenated ``"N59"``
    unchanged), so a route claim compares directly against
    ``ResolvedLocation.road`` / ``road_refs``. Imported lazily for the same
    package-cycle reason as ``_sign_tables``.
    """
    from ..pipeline.title_builder import _road_ref_members

    members = _road_ref_members(token.upper())
    return members[0] if members else token.upper()


def _first_sign_segment(captured: str) -> str:
    """First clause of a greedy sign capture ("walterboro — colleton county")."""
    return _SIGN_SEGMENT_SPLIT_RE.split(captured, maxsplit=1)[0].strip()


def parse_sign_claims(text: Optional[str]) -> list[SignClaim]:
    """Every distinct location claim a sign's OCR text asserts.

    Unlike :meth:`OCREngine._parse_location` — which returns the FIRST matching
    pattern as one ``(name, LocationType)`` pair and is left untouched — this
    returns every county / state / town / route claim found, deduplicated on
    ``(kind, value)``. Values are normalised for direct comparison against a
    ``ResolvedLocation``: counties BARE (``"Example"``, not
    ``"Example County"``), states as full names, routes in via-road vote form.

    Rivers, bridges and distance signs are matched by the shared pattern table
    but yield no claim — they are not location authorities.
    """
    if not text or not text.strip():
        return []

    tables = _sign_tables()
    lowered = text.lower()

    # State-name spans first: a county capture landing inside one is the
    # state's own words, not a place ("Example County South Carolina" must
    # not also yield a "South" county).
    state_spans: list[tuple[int, int]] = []
    state_names: list[str] = []
    for match in tables["name_re"].finditer(lowered):
        state_spans.append(match.span())
        state_names.append(tables["name_by_lower"][match.group(0)])

    claims: list[SignClaim] = []
    seen: set[tuple[str, str]] = set()

    def _add(kind: str, value: str) -> None:
        value = value.strip()
        if not value or (kind, value) in seen:
            return
        seen.add((kind, value))
        claims.append(SignClaim(kind=kind, value=value, raw_text=text))

    for pattern, loc_type in _LOCATION_PATTERNS:
        if loc_type not in (LocationType.TOWN_ENTRY, LocationType.COUNTY_BOUNDARY):
            continue
        for match in re.finditer(pattern, lowered):
            # Greedy captures swallow trailing sign text; truncate at the first
            # sign-trailer phrase, then at the first clause delimiter.
            captured = _first_sign_segment(
                OCREngine._truncate_at_sign_suffix(match.group(1).strip())
            )
            tokens = captured.split()
            # A road/infrastructure descriptor is never a placename
            # ("County Road 12" captured "road" -> a bogus "Road County").
            if not tokens or tokens[0] in _CAPTURE_STOP_WORDS:
                continue

            if loc_type == LocationType.COUNTY_BOUNDARY:
                start, end = match.span(1)
                if any(s < end and start < e for s, e in state_spans):
                    continue
                # "Kerry County Council" is an authority, not a boundary: the
                # capture is clean but a blocklisted sign-suffix phrase starts
                # right after it.
                tail = lowered[match.end(1) :].lstrip()
                if any(tail.startswith(phrase) for phrase in _SIGN_SUFFIX_PHRASES):
                    continue
                _add(_CLAIM_COUNTY, captured.title())
            else:
                # "Welcome to South Carolina" is a state sign, and
                # "Welcome to County Kerry" is a county sign — neither is a town.
                if captured in tables["name_by_lower"]:
                    continue
                if any(tok in _COUNTY_MARKER_TOKENS for tok in tokens):
                    continue
                _add(_CLAIM_TOWN, captured.title())

    has_place = bool(seen)

    for name in state_names:
        _add(_CLAIM_STATE, name)
    # A bare postal abbreviation is only a state claim alongside a place claim
    # ("Example County, SC") — on its own it is two uppercase letters.
    if has_place:
        for match in tables["abbrev_re"].finditer(text):
            _add(_CLAIM_STATE, tables["name_by_abbrev"][match.group(0)])

    for match in tables["us_route_re"].finditer(text):
        _add(_CLAIM_ROUTE, _normalize_route_token(f"{match.group(1)} {match.group(2)}"))
    for match in _EURO_ROUTE_RE.finditer(text):
        _add(_CLAIM_ROUTE, _normalize_route_token(match.group(0)))

    return claims


class OCREngine:
    """
    OCR engine combining EasyOCR for fast local processing
    and Claude for ambiguous cases.
    """

    def __init__(self, config: Config):
        self.config = config
        self._easyocr_reader = None

        # Patterns for location detection (shared with ``parse_sign_claims``)
        self.location_patterns = _LOCATION_PATTERNS

    @property
    def easyocr_reader(self):
        """Lazy load EasyOCR reader.

        ``easyocr`` ships in the optional ``signs`` extra, so a default install
        does not have it. Translate the bare ModuleNotFoundError into the
        install command, otherwise the only clue an operator gets for a
        ``--sign-verify`` run is a missing-module traceback.
        """
        if self._easyocr_reader is None:
            try:
                import easyocr
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Road-sign OCR needs the optional 'signs' extra, which is not "
                    "installed. Install it with: pip install -e '.[signs]'  "
                    "(it pulls easyocr and torch, ~4.5 GB). Runs without "
                    "--sign-verify do not need it."
                ) from exc

            self._easyocr_reader = easyocr.Reader(
                self.config.ocr.languages,
                gpu=False,  # Set True if GPU available
                verbose=False,
            )
        return self._easyocr_reader

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_path: Optional[str] = None,
    ) -> Optional[DetectedLocation]:
        """
        Process a frame through OCR pipeline.

        1. Try EasyOCR first
        2. If confidence low, try Claude
        3. Parse result to extract location info
        """
        # Run EasyOCR
        ocr_result = self._run_easyocr(frame)

        if ocr_result is None:
            return None

        text, confidence = ocr_result

        # Claude OCR fallback disabled — requires migration to claude CLI
        source = "easyocr"

        # Parse the text to extract location info
        location_name, location_type = self._parse_location(text)

        # Determine confidence level
        if confidence >= 0.8:
            conf_level = ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            conf_level = ConfidenceLevel.MEDIUM
        else:
            conf_level = ConfidenceLevel.LOW

        return DetectedLocation(
            timestamp=timestamp,
            text=text,
            location_name=location_name,
            location_type=location_type,
            confidence=conf_level,
            confidence_score=confidence,
            frame_path=frame_path,
            source=source,
        )

    def _run_easyocr(self, frame: np.ndarray) -> Optional[tuple[str, float]]:
        """Run EasyOCR on frame, return (text, confidence) or None."""
        try:
            # Convert BGR to RGB for EasyOCR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.easyocr_reader.readtext(rgb_frame)

            if not result:
                return None

            # Combine all detected text
            texts = []
            confidences = []

            for bbox, text, conf in result:
                if conf >= 0.3 and len(text) >= 2:
                    texts.append(text)
                    confidences.append(conf)

            if not texts:
                return None

            combined_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences)

            # Filter out very short results
            if len(combined_text.strip()) < 3:
                return None

            return combined_text, avg_confidence

        except Exception as e:
            logger.error("EasyOCR error: %s", e, exc_info=True)
            return None

    @staticmethod
    def _truncate_at_sign_suffix(captured: str) -> str:
        """Cut a capture at the first trailing sign phrase (twinning, county
        council, drive-safely text that follows the placename on a real sign)."""
        earliest = len(captured)
        for phrase in _SIGN_SUFFIX_PHRASES:
            idx = captured.find(phrase)
            if idx != -1:
                earliest = min(earliest, idx)
        return captured[:earliest].strip()

    def _parse_location(self, text: str) -> tuple[str, LocationType]:
        """Parse OCR text to extract location name and type."""
        text_lower = text.lower()

        for pattern, loc_type in self.location_patterns:
            match = re.search(pattern, text_lower)
            if not match:
                continue

            # Greedy captures ("welcome to X …") swallow trailing sign text;
            # truncate at the first sign-trailer phrase.
            captured = self._truncate_at_sign_suffix(match.group(1).strip())

            # A road/infrastructure descriptor is never a placename
            # ("County Road 12" captured "road" -> a bogus "Road County").
            first_token = captured.split()[0] if captured.split() else ""
            if not captured or first_token in _CAPTURE_STOP_WORDS:
                continue

            # Preserve a county-first sign's own form ("County Kerry") so the
            # assembler's US template doesn't flip it to "Kerry County".
            if loc_type == LocationType.COUNTY_BOUNDARY and match.group(0).strip().startswith(
                "county "
            ):
                return f"County {captured.title()}", loc_type

            return captured.title(), loc_type

        # No pattern matched - use the full text as location name
        # Clean up common OCR artifacts
        clean_text = re.sub(r"[^\w\s]", "", text)
        clean_text = " ".join(clean_text.split())

        if clean_text:
            return clean_text.title(), LocationType.UNKNOWN

        return text, LocationType.UNKNOWN


class LocationDeduplicator:
    """Removes duplicate location detections."""

    def __init__(self, time_threshold: float = 3.0, similarity_threshold: float = 0.8):
        self.time_threshold = time_threshold
        self.similarity_threshold = similarity_threshold

    def deduplicate(self, locations: list[DetectedLocation]) -> list[DetectedLocation]:
        """Remove duplicate detections of the same sign."""
        if not locations:
            return []

        # Sort by timestamp
        locations = sorted(locations, key=lambda x: x.timestamp)

        result = []
        for loc in locations:
            # Check if this is a duplicate of a recent detection
            is_duplicate = False
            for existing in reversed(result):
                # Only check recent detections
                if loc.timestamp - existing.timestamp > self.time_threshold:
                    break

                # Check if same location
                if self._is_same_location(loc, existing):
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if loc.confidence_score > existing.confidence_score:
                        result.remove(existing)
                        result.append(loc)
                    break

            if not is_duplicate:
                result.append(loc)

        return result

    def _is_same_location(self, loc1: DetectedLocation, loc2: DetectedLocation) -> bool:
        """Check if two detections refer to the same location."""
        # Same location name
        if loc1.location_name.lower() == loc2.location_name.lower():
            return True

        # Similar text (fuzzy match)
        similarity = self._text_similarity(loc1.text, loc2.text)
        return similarity >= self.similarity_threshold

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple character-based similarity."""
        text1 = text1.lower()
        text2 = text2.lower()

        if not text1 or not text2:
            return 0.0

        # Longest common subsequence ratio
        shorter = text1 if len(text1) < len(text2) else text2
        longer = text2 if len(text1) < len(text2) else text1

        matches = sum(1 for c in shorter if c in longer)
        return matches / len(longer)
