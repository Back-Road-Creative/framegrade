"""String-fixture tests for the multi-claim road-sign parser.

``parse_sign_claims`` is the OCR-text -> typed-claim half of road-sign GPS
verification. These tests are pure string fixtures — no video, no OCR engine,
no network.

The two properties under test are:

* **normalisation** — a claim's ``value`` must land in the exact shape the
  downstream comparator matches against (bare county names for
  ``ResolvedLocation.regions``/vote members, hyphenated route refs for
  ``ResolvedLocation.road``/``road_refs``, full state names).
* **noise discipline** — a sign is noisy, and a false claim costs an operator
  a bogus conflict warning, so the infrastructure/stop-word and sign-suffix
  guards must survive the move from single-match to multi-match parsing.
"""

from gomoveshift.analysis.ocr import OCREngine, parse_sign_claims
from gomoveshift.models import LocationType

COUNTY = LocationType.COUNTY_BOUNDARY.value
STATE = LocationType.STATE_BOUNDARY.value
TOWN = LocationType.TOWN_ENTRY.value
ROUTE = LocationType.ROUTE_MARKER.value


def _pairs(text):
    """(kind, value) pairs for a sign's text — order-insensitive comparisons."""
    return {(c.kind, c.value) for c in parse_sign_claims(text)}


def _values(text, kind):
    return sorted(c.value for c in parse_sign_claims(text) if c.kind == kind)


# ── new LocationType members ───────────────────────────────────────────


def test_new_location_types_exist_with_expected_values():
    assert LocationType.STATE_BOUNDARY.value == "state_boundary"
    assert LocationType.ROUTE_MARKER.value == "route_marker"
    # Existing members are untouched.
    assert LocationType.COUNTY_BOUNDARY.value == "county_boundary"
    assert LocationType.TOWN_ENTRY.value == "town_entry"


# ── county normalisation: all three sign forms -> BARE county name ─────


def test_county_suffix_form_yields_bare_name():
    assert _values("Colleton County", COUNTY) == ["Colleton"]


def test_county_prefix_form_yields_bare_name():
    assert _values("County Kerry", COUNTY) == ["Kerry"]


def test_county_abbreviated_form_yields_bare_name():
    assert _values("Co. Galway", COUNTY) == ["Galway"]


def test_county_claim_never_carries_the_county_filler_word():
    for text in ("Colleton County", "County Kerry", "Co. Galway"):
        for claim in parse_sign_claims(text):
            if claim.kind == COUNTY:
                assert "county" not in claim.value.lower()


def test_parse_location_county_first_form_is_unchanged():
    """``_parse_location`` keeps its own "County X" shape — claims are additive."""
    from gomoveshift.config import Config

    engine = OCREngine(Config())
    name, loc_type = engine._parse_location("County Kerry")
    assert (name, loc_type) == ("County Kerry", LocationType.COUNTY_BOUNDARY)


# ── noise discipline ───────────────────────────────────────────────────


def test_county_council_sign_yields_no_county_claim():
    assert _values("Kerry County Council", COUNTY) == []


def test_county_road_sign_yields_no_county_claim():
    assert _values("County Road 12", COUNTY) == []


def test_river_and_bridge_signs_yield_no_claims():
    assert parse_sign_claims("Shannon River") == []
    assert parse_sign_claims("Ashley River Bridge") == []


def test_distance_sign_yields_no_claims():
    assert parse_sign_claims("Clifden 5 km") == []


def test_empty_and_garbage_text_yields_no_claims():
    assert parse_sign_claims("") == []
    assert parse_sign_claims("   ") == []
    assert parse_sign_claims("!!! ###") == []
    assert parse_sign_claims(None) == []


def test_twinning_suffix_is_truncated_off_the_town_claim():
    assert _values("Welcome to Clifden Twinned with Boston", TOWN) == ["Clifden"]


# ── town vs state ──────────────────────────────────────────────────────


def test_welcome_to_town_is_a_town_claim_and_not_a_state_claim():
    claims = _pairs("Welcome to Walterboro")
    assert (TOWN, "Walterboro") in claims
    assert not [k for k, _ in claims if k == STATE]


def test_welcome_to_state_is_a_state_claim_and_not_a_town_claim():
    claims = _pairs("Welcome to South Carolina")
    assert (STATE, "South Carolina") in claims
    assert not [k for k, _ in claims if k == TOWN]


def test_irish_welcome_form_yields_a_town_claim():
    assert _values("Fáilte go Clifden", TOWN) == ["Clifden"]


def test_population_sign_yields_a_town_claim():
    assert _values("Walterboro Pop. 5398", TOWN) == ["Walterboro"]


# ── state forms ────────────────────────────────────────────────────────


def test_state_line_sign_yields_a_state_claim():
    assert _values("South Carolina State Line", STATE) == ["South Carolina"]


def test_entering_state_sign_yields_a_state_claim():
    assert _values("Entering North Carolina", STATE) == ["North Carolina"]


def test_longer_state_name_wins_over_its_substring():
    assert _values("Welcome to West Virginia", STATE) == ["West Virginia"]


def test_postal_abbreviation_alongside_a_county_yields_the_full_state_name():
    assert _pairs("Colleton County, SC") == {
        (COUNTY, "Colleton"),
        (STATE, "South Carolina"),
    }


def test_bare_postal_abbreviation_without_a_place_claim_yields_nothing():
    assert parse_sign_claims("SC") == []


def test_state_name_beside_a_county_does_not_leak_a_second_county_claim():
    assert _values("Colleton County South Carolina", COUNTY) == ["Colleton"]


# ── route markers ──────────────────────────────────────────────────────


def test_us_route_shapes_normalise_to_hyphenated_refs():
    assert _values("I-95", ROUTE) == ["I-95"]
    assert _values("I 95", ROUTE) == ["I-95"]
    assert _values("US 17", ROUTE) == ["US-17"]
    assert _values("US-17", ROUTE) == ["US-17"]
    assert _values("SC 64", ROUTE) == ["SC-64"]


def test_irish_and_uk_route_shapes_stay_unhyphenated():
    assert _values("N59", ROUTE) == ["N59"]
    assert _values("M50", ROUTE) == ["M50"]
    assert _values("R336", ROUTE) == ["R336"]
    assert _values("A4", ROUTE) == ["A4"]


def test_route_normalisation_matches_the_via_road_vote_form():
    """Same normaliser the via-road vote uses, so claims compare directly."""
    from gomoveshift.pipeline.title_builder import _road_ref_members

    for raw, expected in (("I 95", "I-95"), ("US 17", "US-17"), ("N59", "N59")):
        assert _values(raw, ROUTE) == _road_ref_members(expected)


def test_multiple_routes_on_one_sign_are_all_claimed():
    assert _values("JCT I 26 US 17", ROUTE) == ["I-26", "US-17"]


# ── multi-claim extraction + duplicate collapse ────────────────────────


def test_one_sign_yields_town_county_and_state_claims():
    assert _pairs("Welcome to Walterboro — Colleton County, SC") == {
        (TOWN, "Walterboro"),
        (COUNTY, "Colleton"),
        (STATE, "South Carolina"),
    }


def test_duplicate_kind_value_pairs_collapse_to_one():
    claims = parse_sign_claims("Colleton County  Colleton County")
    assert [(c.kind, c.value) for c in claims] == [(COUNTY, "Colleton")]


def test_claims_carry_the_raw_sign_text_for_operator_review():
    text = "Welcome to Walterboro — Colleton County, SC"
    for claim in parse_sign_claims(text):
        assert claim.raw_text == text


def test_claim_kinds_are_plain_location_type_value_strings():
    for claim in parse_sign_claims("Welcome to Walterboro — Colleton County, SC I 95"):
        assert isinstance(claim.kind, str)
        assert claim.kind in {t.value for t in LocationType}
