"""Burned OST / title is GPS-derivable ONLY.

No prior test asserted that the text the pipeline burns on-screen (and reuses as
the YouTube title) contains ONLY tokens traceable to the GPS geocode (place
names, via-road, region) plus a small allow-list of factual format tokens
(4K / POV / Part / in / to / via / resolution labels). A future change that
slipped an unverifiable hype/gimmick adjective into the derived title would ship
unchecked.

This test drives the REAL title generators — ``build_lf_title`` (the canonical
burned per-LF OST, which doubles as the YouTube title) and ``_front_loaded_title``
(the SEO YouTube variant) — over a representative set of geocoded inputs and
asserts, for both surfaces:

  (a) policy-level: the output passes ``ContentValidator`` with NO
      hype/superlative/unverifiable-adjective flags, and the burned OST passes
      ``validate_ost`` (structural GPS-derivable gate). ``ContentValidator`` is
      the existing superlative/hype validator — reused here, not re-implemented.

  (b) token-level: every substantive word of the output is traceable to the
      input geocode strings (place names / via-road / region, in full OR
      abbreviated form) OR to an explicit factual-format allow-list. Any leftover
      word is, by construction, an unverifiable token and fails the test.

The geocoder is the only mock — it stands in for the external reverse-geocode
service and feeds the real generators deterministic place data (the same seam
every other title test uses). The generators themselves are exercised for real.
"""

import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline.content_validator import ContentValidator
from gomoveshift.pipeline.longform_metadata import _front_loaded_title
from gomoveshift.pipeline.ost_validator import validate_ost
from gomoveshift.pipeline.title_builder import build_lf_title, normalize_state_abbr

# Factual format tokens that are always allowed — they encode format/route
# structure, not an unverifiable claim about the drive.
# Allow-list: {4K, POV, Part, in, to, via, |, digits, resolution labels}. The
# pipe "|" and other punctuation normalize away in the tokenizer, so only word
# tokens need listing here.
_FORMAT_TOKENS = {
    "to",
    "via",
    "in",
    "of",
    "part",
    "pov",
    "4k",
    "hd",
    "uhd",
    "2160p",
    "1440p",
    "1080p",
    "720p",
}

# A pure number ("2" in "Part 2") or a resolution label ("1080p").
_NUMERIC_OR_RES = re.compile(r"\d+p?$")


def _tokens(text: str) -> list[str]:
    """Lower-cased alphanumeric word cores of ``text`` (punctuation stripped).

    Applied identically to inputs and outputs so "Lady's" → "ladys" on both
    sides, "N86" → "n86", "(Sea" → "sea", and "|" → "" (dropped). This uniform
    normalization is what makes the subset check meaningful.
    """
    out = []
    for word in text.split():
        core = re.sub(r"[^a-z0-9]", "", word.lower())
        if core:
            out.append(core)
    return out


def _state_abbrs(location: str) -> set[str]:
    """State abbreviations the generator legitimately derives from ``location``.

    ``build_lf_title`` abbreviates a GPS-derived US state ("South Carolina" →
    "SC"). "SC" is therefore GPS-derivable even though it isn't literally in the
    input, so the allow-list must include it. Derived via the SAME real helper
    the generator uses (``normalize_state_abbr``); parenthetical road suffixes
    are stripped first so the region part parses.
    """
    abbrs: set[str] = set()
    for part in re.sub(r"\([^)]*\)", "", location).split(","):
        abbr = normalize_state_abbr(part.strip())
        if abbr:
            abbrs.update(_tokens(abbr))
    return abbrs


def _seg(lat: float, lon: float):
    return SimpleNamespace(gps_center_lat=lat, gps_center_lon=lon)


def _loc(**kwargs):
    defaults = dict(
        city=None, locality=None, suburb=None, name=None, town=None, state=None, country=None
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _geocoder(places):
    """A reverse-geocoder returning ``places`` (a town name or None) in order."""
    mock = MagicMock()
    mock.reverse_geocode.side_effect = [_loc(city=p) if p else _loc() for p in places]
    return mock


class _Case:
    """One geocoded input + the GPS strings its output is allowed to draw from."""

    def __init__(self, cid, segs, places, per_lf_location, expected_title, **kwargs):
        self.id = cid
        self.segs = segs
        self.places = places
        self.per_lf_location = per_lf_location
        self.expected_title = expected_title
        self.kwargs = kwargs  # e.g. resolved_destination

    def build(self) -> str:
        return build_lf_title(
            self.segs,
            _geocoder(self.places),
            per_lf_location=self.per_lf_location,
            **self.kwargs,
        )

    def allowed_tokens(self) -> set[str]:
        allowed = set(_FORMAT_TOKENS)
        # Input geocode strings: town names, the per-LF location (region + road),
        # and any destination POI — all GPS/geocoder-derived facts.
        sources = [p for p in self.places if p]
        sources.append(self.per_lf_location)
        if self.kwargs.get("resolved_destination"):
            sources.append(self.kwargs["resolved_destination"])
        for src in sources:
            allowed.update(_tokens(src))
        allowed.update(_state_abbrs(self.per_lf_location))
        return allowed


# Representative geocoded inputs. Coordinates + place data mirror the proven
# fixtures in test_ireland_title.py so the derivation is deterministic. Cover:
# US/non-US, with/without via-road, single-region fallback, and a destination
# anchor (a POI resolved from the footage, not an endpoint arc).
CASES = [
    _Case(
        "ireland_via_road",
        [_seg(52.06, -9.51), _seg(52.14, -9.70)],
        ["Killarney", "Dingle"],
        "Killarney, Kerry (N86)",
        # The geocoded road has no measured coverage, so it is omitted.
        "Killarney to Dingle, Kerry",
    ),
    _Case(
        "ireland_no_road",
        [_seg(52.06, -9.51), _seg(52.14, -10.27)],
        ["Killarney", "Dingle"],
        "Killarney, Kerry",
        "Killarney to Dingle, Kerry",
    ),
    _Case(
        "us_via_road",
        [_seg(32.3, -80.6), _seg(32.4, -80.7)],
        ["Port Royal", "Lady's Island"],
        "Beaufort County, South Carolina (Sea Island Parkway)",
        "Port Royal to Lady's Island, SC",
    ),
    _Case(
        "us_no_road",
        [_seg(32.3, -80.6), _seg(32.4, -80.7)],
        ["Port Royal", "Lady's Island"],
        "Beaufort County, South Carolina",
        "Port Royal to Lady's Island, SC",
    ),
    _Case(
        "single_region_fallback",
        [_seg(52.06, -9.51), _seg(52.06, -9.51)],
        [None, None],
        "Kerry, Ireland",
        "Kerry, Ireland",
    ),
    _Case(
        "destination_anchor",
        [_seg(32.3, -80.6), _seg(32.4, -80.7)],
        [None, None],
        "Beaufort County, South Carolina",
        "Hunting Island State Park, SC",
        resolved_destination="Hunting Island State Park",
    ),
]

_CASE_IDS = [c.id for c in CASES]


@pytest.fixture(scope="module")
def validator():
    return ContentValidator()


@pytest.mark.parametrize("case", CASES, ids=_CASE_IDS)
def test_burned_title_matches_expected(case):
    """Sanity anchor: the generator produces the deterministic expected title."""
    assert case.build() == case.expected_title


@pytest.mark.parametrize("case", CASES, ids=_CASE_IDS)
def test_burned_ost_passes_policy_validators(case, validator):
    """(a) Burned OST passes ContentValidator (no hype) AND validate_ost (structural)."""
    title = case.build()

    result = validator.validate(title, field_name="burned_ost")
    assert result.valid, (
        f"[{case.id}] burned OST flagged by ContentValidator: '{title}'\n"
        f"Violations: {result.violations}"
    )

    ok, reason = validate_ost(title, "lf")
    assert ok, f"[{case.id}] burned OST failed structural validate_ost: '{title}' — {reason}"


@pytest.mark.parametrize("case", CASES, ids=_CASE_IDS)
def test_seo_title_passes_content_validator(case, validator):
    """(a) The SEO YouTube title variant also carries no hype/superlative flags."""
    seo = _front_loaded_title(case.build())
    result = validator.validate(seo, field_name="seo_title")
    assert result.valid, (
        f"[{case.id}] SEO title flagged by ContentValidator: '{seo}'\n"
        f"Violations: {result.violations}"
    )


@pytest.mark.parametrize("case", CASES, ids=_CASE_IDS)
def test_every_output_token_is_gps_derivable(case):
    """(b) Every substantive word of BOTH surfaces traces to input geocode data.

    The allow-list is the input place names + per-LF location (region + road) +
    destination POI + derived state abbreviation + the factual format tokens.
    Any output token outside that set is an unverifiable word (a hype/gimmick
    adjective the geocode can't back) and fails the test.
    """
    allowed = case.allowed_tokens()
    burned = case.build()
    seo = _front_loaded_title(burned)

    for surface, text in [("burned_ost", burned), ("seo_title", seo)]:
        leftover = [
            tok for tok in _tokens(text) if tok not in allowed and not _NUMERIC_OR_RES.match(tok)
        ]
        assert not leftover, (
            f"[{case.id}] {surface} '{text}' contains non-GPS-derivable "
            f"token(s) {leftover}. Every substantive word must trace to the "
            f"input geocode (place/road/region) or an allowed factual format "
            f"token; a leftover word is an unverifiable claim."
        )
