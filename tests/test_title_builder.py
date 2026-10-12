"""Tests for title_builder — endpoint-places + via-road format.

Canonical format:
    "Port Royal to Lady's Island via Sea Island Parkway, SC"

Fallback cascade covered here:
    1. start to end via road (canonical)
    2. start to end (no road)
    3. start Loop via road (start == end, road present)
    3b. start Loop (start == end, no road)
    4. county fallback (place, region — no genre word)
    5. per_lf_location verbatim (total geocode failure)
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline.title_builder import (
    MAX_TITLE_LEN,
    _evenly_sample,
    _evenly_sample_weighted,
    _first_last_centroids_from_gps,
    _geocode_town,
    _looks_like_county,
    _looks_like_region,
    _point_latlon,
    build_lf_title,
    build_superlong_part_title,
    dominant_location,
    ensure_unique_titles,
    is_us_state,
    normalize_state_abbr,
)
from gomoveshift.pipeline.validation.place_corrections import PlaceCorrections


def _seg(lat: float, lon: float):
    return SimpleNamespace(gps_center_lat=lat, gps_center_lon=lon)


def _loc(**kwargs):
    """Return a location object with the given attributes."""
    defaults = dict(city=None, locality=None, suburb=None, name=None, town=None)
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ── normalize_state_abbr ─────────────────────────────────────────────


class TestNormalizeStateAbbr:
    def test_full_name_mapped(self):
        assert normalize_state_abbr("South Carolina") == "SC"
        assert normalize_state_abbr("north carolina") == "NC"
        assert normalize_state_abbr("new york") == "NY"

    def test_already_abbr_passes_through(self):
        assert normalize_state_abbr("SC") == "SC"
        assert normalize_state_abbr("nc") == "NC"

    def test_unknown_returns_empty(self):
        assert normalize_state_abbr("Wakanda") == ""
        assert normalize_state_abbr("") == ""
        assert normalize_state_abbr(None) == ""


class TestIsUsState:
    def test_full_state_name(self):
        assert is_us_state("South Carolina") is True
        assert is_us_state("pennsylvania") is True

    def test_two_letter_abbr(self):
        assert is_us_state("SC") is True
        assert is_us_state("pa") is True

    def test_non_us_two_letter_token_rejected(self):
        # Country codes / arbitrary 2-letter tokens are not US states.
        assert is_us_state("IE") is False
        assert is_us_state("UK") is False

    def test_country_and_junk_rejected(self):
        assert is_us_state("Ireland") is False
        assert is_us_state("County Kerry") is False
        assert is_us_state("") is False
        assert is_us_state(None) is False


# ── build_lf_title — canonical (level 1) ─────────────────────────────


class TestCanonicalFormat:
    """Level 1: start to end via road, state."""

    def test_endpoints_differ_road_present(self):
        segs = [_seg(32.3, -80.6), _seg(32.4, -80.7)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Port Royal"),
            _loc(city="Lady's Island"),
        ]

        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Beaufort County, South Carolina (Sea Island Parkway)",
        )
        assert title == "Port Royal to Lady's Island via Sea Island Parkway, SC"

    def test_state_abbr_override_wins_over_parsed(self):
        """When caller passes state_abbr explicitly, skip parsing."""
        segs = [_seg(32.3, -80.6), _seg(32.4, -80.7)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Port Royal"),
            _loc(city="Lady's Island"),
        ]

        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Beaufort County, South Carolina (Sea Island Parkway)",
            state_abbr="SC",
        )
        assert title == "Port Royal to Lady's Island via Sea Island Parkway, SC"


# ── Via-road span guard ────────────────────────────────


class TestViaRoadSpanGuard:
    """A 'via {road}' suffix is dropped when the endpoints are >50 km apart."""

    def test_long_span_drops_via_road(self):
        # Lake Lure → Oakland, ~180 km apart, both NC. The dominant road is a
        # coastal-SC road far from both endpoints — must NOT appear.
        segs = [_seg(35.43, -81.90), _seg(36.00, -80.00)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Lake Lure"),
            _loc(city="Oakland"),
        ]
        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Rutherford County, North Carolina (Hunting Island Drive)",
        )
        assert "via" not in title
        assert title == "Lake Lure to Oakland, NC"

    def test_short_span_keeps_via_road(self):
        # ~14 km apart — a legitimate local road, suffix survives (no regression).
        segs = [_seg(32.30, -80.60), _seg(32.40, -80.70)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Port Royal"),
            _loc(city="Lady's Island"),
        ]
        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Beaufort County, South Carolina (Sea Island Parkway)",
        )
        assert title == "Port Royal to Lady's Island via Sea Island Parkway, SC"

    def test_long_span_drops_multi_road(self):
        segs = [_seg(35.43, -81.90), _seg(36.00, -80.00)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Lake Lure"),
            _loc(city="Oakland"),
        ]
        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Rutherford County, North Carolina",
            roads=["US-74", "NC-9"],
        )
        assert "via" not in title
        assert title == "Lake Lure to Oakland, NC"

    def test_long_span_drops_county_via_road_level4(self):
        # No endpoint towns resolve → county fallback (Level 4). On a
        # long crossing the county's via-road must drop too (not just the
        # A-to-B form), even when it arrives via the per_lf_location suffix.
        segs = [_seg(32.40, -80.40), _seg(35.60, -82.90)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [_loc(), _loc()]  # no city → no towns
        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Transylvania County, North Carolina (Hunting Island Drive)",
        )
        assert "via" not in title
        assert title == "Transylvania County, NC"

    def test_long_span_keeps_interstate_via_road(self):
        # An interstate is a numbered through-route that legitimately
        # describes a long crossing (Johnson City, TN → Seven Mile Ford,
        # VA stays on I 81). The 50 km span guard exists for LOCAL roads; a
        # route-ref road must survive it. ~180 km apart, dominant road "I 81".
        segs = [_seg(35.43, -81.90), _seg(36.00, -80.00)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Bristol"),
            _loc(city="Marion"),
        ]
        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Washington County, Virginia (I 81)",
        )
        assert title == "Bristol to Marion via I 81, VA"

    def test_long_span_still_drops_named_through_road(self):
        # Targeted: the interstate exemption is route-ref-only. A named
        # through-road ("Sea Island Parkway") passes the via-road quality
        # filter but carries NO route number, so the span guard must still
        # drop it on a long crossing — proving the exemption is not a blanket
        # "any quality road survives" hole.
        segs = [_seg(35.43, -81.90), _seg(36.00, -80.00)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Lake Lure"),
            _loc(city="Oakland"),
        ]
        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Beaufort County, South Carolina (Sea Island Parkway)",
        )
        assert "via" not in title
        assert title == "Lake Lure to Oakland, SC"


# ── Level 2: no road ─────────────────────────────────────────────────


class TestNoRoadFallback:
    def test_endpoints_differ_no_road(self):
        segs = [_seg(32.3, -80.6), _seg(32.4, -80.7)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Port Royal"),
            _loc(city="Lady's Island"),
        ]
        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Beaufort County, South Carolina",
        )
        assert title == "Port Royal to Lady's Island, SC"


# ── Level 3: Loop (start == end) ─────────────────────────────────────


class TestLoopFormat:
    def test_loop_with_road(self):
        segs = [_seg(32.7, -80.6), _seg(32.7, -80.6)]  # same start/end
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Colleton"),
            _loc(city="Colleton"),
        ]
        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Colleton County, South Carolina (Sea Island Parkway)",
        )
        assert title == "Colleton Loop via Sea Island Parkway, SC"

    def test_loop_without_road(self):
        segs = [_seg(32.7, -80.6), _seg(32.7, -80.6)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Colleton"),
            _loc(city="Colleton"),
        ]
        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Colleton County, South Carolina",
        )
        assert title == "Colleton Loop, SC"


# ── Level 4: county/town fallback ────────────────────────────────────


class TestCountyFallback:
    def test_no_endpoint_data_uses_county(self):
        """Segments with (0,0) centroid → no endpoint lookup → county fallback."""
        segs = [_seg(0.0, 0.0), _seg(0.0, 0.0)]
        geocoder = MagicMock()
        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Colleton County, South Carolina",
        )
        assert title == "Colleton County, SC"
        geocoder.reverse_geocode.assert_not_called()

    def test_no_geocoder_uses_county(self):
        segs = [_seg(32.7, -80.6), _seg(32.7, -80.7)]
        title = build_lf_title(
            segs,
            geocoder=None,
            per_lf_location="Macon County, North Carolina",
        )
        assert title == "Macon County, NC"

    def test_county_with_road_suffix(self):
        """No endpoint data but a QUALITY road present → county via road.

        (a junk suffix road like "Blocker Run Road" is now filtered out;
        a named through-route such as a Parkway still surfaces.)
        """
        segs = [_seg(0.0, 0.0), _seg(0.0, 0.0)]
        title = build_lf_title(
            segs,
            geocoder=None,
            per_lf_location="Colleton County, South Carolina (Sea Island Parkway)",
        )
        assert title == "Colleton County via Sea Island Parkway, SC"


# ── Level 5: total fallback ──────────────────────────────────────────


class TestFinalFallback:
    def test_non_us_region_uses_level4_place_region(self):
        """Non-US trailing region used verbatim at Level 4
        rather than falling through to the raw Level 5 return."""
        segs = []
        title = build_lf_title(
            segs,
            geocoder=None,
            per_lf_location="Some Place, Wakanda",
        )
        assert title == "Some Place, Wakanda"

    def test_no_region_at_all_returns_per_lf_location(self):
        """No comma → no county/region split → genuine Level 5 return."""
        segs = []
        title = build_lf_title(
            segs,
            geocoder=None,
            per_lf_location="Somewhere",
        )
        assert title == "Somewhere"

    def test_empty_input_returns_empty(self):
        assert build_lf_title([], None, per_lf_location="") == ""


class TestEnforceLengthRegionPreserving:
    """_enforce_length's last-resort fallback drops the trailing region
    clause before any char-chop, so an over-budget fallback keeps the town
    intact instead of a mangled "County Kerry Ga" fragment."""

    def test_fallback_drops_county_clause_before_charchop(self):
        from gomoveshift.pipeline.title_builder import _enforce_length

        town = "Dingle Peninsula toward Gortmore and Ballydavid and Dunquin and Feohanagh Coast"
        fallback = f"{town}, County Kerry Gaeltacht"
        assert len(fallback) > MAX_TITLE_LEN - len("SC") - 2  # forces the fallback branch
        result = _enforce_length("X" * (MAX_TITLE_LEN + 1), fallback, "SC")
        assert result == f"{town}, SC"
        assert "County" not in result
        assert len(result) <= MAX_TITLE_LEN


# ── Level 0: destination anchor ────────────────────────────────


class TestDestinationAnchor:
    """A resolved destination POI anchors the title AHEAD of the A→B cascade.

    The county still lives in the ResolvedLocation the far-road location invariant gate reads, so a
    destination title never desyncs that gate (verified in test_resolver_populate
    + the gate's own suite — it parses dominant_county/regions, never the title).
    """

    def test_destination_wins_over_endpoint_cascade(self):
        """A destination POI replaces what would otherwise be an A→B via-road
        title — and a real geocoder is never consulted (short-circuit)."""
        segs = [_seg(32.3, -80.6), _seg(32.4, -80.7)]
        geocoder = MagicMock()
        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Colleton County, South Carolina (Donnelley Drive)",
            resolved_destination="Donnelley Wildlife Management Area",
        )
        assert title == "Donnelley Wildlife Management Area, SC"
        geocoder.reverse_geocode.assert_not_called()

    def test_non_us_region_appended_verbatim(self):
        """Non-US: region_label is the trailing region (matches the rest of the
        cascade — ``normalize_state_abbr`` yields "" for a non-US region, so the
        trailing token is used verbatim)."""
        segs = [_seg(52.0, -9.5), _seg(52.1, -9.6)]
        title = build_lf_title(
            segs,
            geocoder=None,
            per_lf_location="County Kerry, Ireland",
            resolved_destination="Killarney National Park",
        )
        assert title == "Killarney National Park, Ireland"

    def test_destination_already_carrying_region_not_doubled(self):
        title = build_lf_title(
            [_seg(32.3, -80.6), _seg(32.4, -80.7)],
            geocoder=None,
            per_lf_location="Colleton County, South Carolina",
            resolved_destination="Botany Bay Beach, SC",
        )
        assert title == "Botany Bay Beach, SC"

    def test_no_destination_is_regression_safe(self):
        """An LF with no resolvable destination produces the SAME title as
        before the wiring — the A→B cascade is untouched."""
        segs = [_seg(32.3, -80.6), _seg(32.4, -80.7)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Port Royal"),
            _loc(city="Lady's Island"),
        ]
        per_lf_location = "Beaufort County, South Carolina (Sea Island Parkway)"
        without_kwarg = build_lf_title(segs, geocoder, per_lf_location=per_lf_location)

        geocoder2 = MagicMock()
        geocoder2.reverse_geocode.side_effect = [
            _loc(city="Port Royal"),
            _loc(city="Lady's Island"),
        ]
        with_none = build_lf_title(
            segs, geocoder2, per_lf_location=per_lf_location, resolved_destination=None
        )
        assert (
            without_kwarg == with_none == "Port Royal to Lady's Island via Sea Island Parkway, SC"
        )

    def test_full_drive_county_only_unaffected_without_destination(self):
        """The GPS-less / county-only (full_drive) path keeps its shape when no
        destination is supplied."""
        segs = [_seg(0.0, 0.0), _seg(0.0, 0.0)]
        title = build_lf_title(
            segs,
            geocoder=None,
            per_lf_location="Macon County, North Carolina",
            resolved_destination=None,
        )
        assert title == "Macon County, NC"


# ── Footage-honest shape for non-contiguous (highlight) LFs ────


def _seg_c(lat: float, lon: float, contiguous: bool):
    """Segment carrying the ``contiguous_with_previous`` flag (LFSegment-like).

    The first segment's flag is irrelevant to the LF-level non-contiguity test
    (it has no predecessor); a False on ANY later segment marks the LF a
    best-moments montage.
    """
    return SimpleNamespace(
        gps_center_lat=lat, gps_center_lon=lon, contiguous_with_previous=contiguous
    )


class TestNonContiguousTitleShape:
    """A best-moments LF (scattered, non-contiguous clips) must NOT assert an
    "A to B via {road}" continuity the footage lacks. It degrades to the
    feature/region anchor ("{place}, {region}"). A genuinely
    contiguous (full_drive) LF keeps its A→B / via-road shape — no regression."""

    def test_noncontiguous_drops_a_to_b_via_road(self):
        # Same short-span endpoints as the canonical Level-1 test — but the
        # montage is non-contiguous, so the A→B via-road claim is suppressed.
        segs = [
            _seg_c(32.30, -80.60, True),
            _seg_c(32.40, -80.70, False),  # a gap → highlight montage
        ]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Port Royal"),
            _loc(city="Lady's Island"),
        ]
        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Beaufort County, South Carolina (Sea Island Parkway)",
        )
        assert " to " not in title
        assert "via" not in title
        assert title == "Beaufort County, SC"

    def test_contiguous_full_drive_keeps_a_to_b_via_road(self):
        # REGRESSION GUARD: every segment contiguous → genuine continuous drive,
        # so the canonical A→B via-road shape is preserved.
        segs = [
            _seg_c(32.30, -80.60, True),
            _seg_c(32.40, -80.70, True),
        ]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Port Royal"),
            _loc(city="Lady's Island"),
        ]
        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Beaufort County, South Carolina (Sea Island Parkway)",
        )
        assert title == "Port Royal to Lady's Island via Sea Island Parkway, SC"

    def test_legacy_segments_without_flag_keep_a_to_b(self):
        # Backward-compat: segments lacking ``contiguous_with_previous`` default
        # to contiguous (full_drive / pre-flag call sites) — A→B shape preserved.
        segs = [_seg(32.30, -80.60), _seg(32.40, -80.70)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Port Royal"),
            _loc(city="Lady's Island"),
        ]
        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Beaufort County, South Carolina (Sea Island Parkway)",
        )
        assert title == "Port Royal to Lady's Island via Sea Island Parkway, SC"

    def test_noncontiguous_drops_plain_a_to_b_no_road(self):
        # No road in the per-LF location → the plain "A to B" (Level 2) shape is
        # ALSO a continuity claim and is suppressed for a montage. The gap is on
        # a LATER segment (segment[0]'s flag has no predecessor, so it's ignored).
        segs = [
            _seg_c(32.30, -80.60, True),
            _seg_c(32.40, -80.70, False),
        ]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Port Royal"),
            _loc(city="Lady's Island"),
        ]
        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Beaufort County, South Carolina",
        )
        assert " to " not in title
        assert title == "Beaufort County, SC"

    def test_resolved_destination_still_wins_for_noncontiguous(self):
        # The destination anchor (Level 0) is itself footage-honest (it names
        # what the highlights are about) and still wins over the new shape.
        segs = [
            _seg_c(32.30, -80.60, True),
            _seg_c(32.40, -80.70, False),
        ]
        geocoder = MagicMock()
        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Colleton County, South Carolina (Donnelley Drive)",
            resolved_destination="Donnelley Wildlife Management Area",
        )
        assert title == "Donnelley Wildlife Management Area, SC"
        geocoder.reverse_geocode.assert_not_called()


# ── Part-N suffix numeric disambig must NOT be used as road ──────────


class TestPartSuffixIgnored:
    def test_part_n_not_treated_as_road(self):
        """``(Part 2)`` is a numeric disambiguator from
        ``derive_per_lf_locations`` — must NOT end up as a road name."""
        segs = [_seg(32.7, -80.6), _seg(32.7, -80.6)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Colleton"),
            _loc(city="Colleton"),
        ]
        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Colleton County, South Carolina (Part 2)",
        )
        # Level 3b (loop without road) — not "Loop via Part 2"
        assert "Part 2" not in title
        assert title == "Colleton Loop, SC"


# ── Road follows the voted county, not the centroid ──────────────


class _CrossCountyGeo:
    """Geocoder whose ``reverse_geocode_detailed`` returns the nearest of two
    cached county points (each with its own road). ``reverse_geocode`` (the
    endpoint town lookup) is keyed off the same nearest-point split. Mirrors a
    sub-50km crossing where the arithmetic-mean centroid lands in the MINORITY
    county while the footage plurality (the vote) is in the other."""

    _FRIPP = (32.3095, -80.6332)  # Beaufort County, SC — the plurality
    _EDISTO = (32.6591, -80.6646)  # Colleton County, SC — the minority
    _BEAUFORT = {
        "county": "Beaufort County",
        "state": "South Carolina",
        "town": "Port Royal",
        "road": "Sea Island Parkway",
    }
    _COLLETON = {
        "county": "Colleton County",
        "state": "South Carolina",
        "town": "Edisto Beach",
        "road": "Blocker Run Road",
    }

    def _nearest(self, lat, lon):
        df = abs(lat - self._FRIPP[0]) + abs(lon - self._FRIPP[1])
        de = abs(lat - self._EDISTO[0]) + abs(lon - self._EDISTO[1])
        return self._BEAUFORT if df <= de else self._COLLETON

    def reverse_geocode_detailed(self, lat, lon):
        return dict(self._nearest(lat, lon))

    def reverse_geocode(self, lat, lon):
        return _loc(city=self._nearest(lat, lon)["town"])


class TestRoadFromVotedCounty:
    """The ``(road)`` disambiguator must come from the winning (voted)
    county. The per_lf_location suffix carries the CENTROID's road, which on a
    boundary crossing is the wrong county's road."""

    def test_via_road_uses_voted_county_road_over_centroid_suffix(self):
        # 8 Fripp (Beaufort) + 2 Edisto (Colleton) → Beaufort wins the vote.
        # The endpoints differ (Port Royal → Edisto Beach) so the via-road
        # cascade fires. The suffix below carries Colleton's road (as
        # derive_per_lf_locations would write from the centroid); the title
        # must instead burn Beaufort's road.
        gps = [{"lat": 32.3095, "lon": -80.6332}] * 8 + [{"lat": 32.6591, "lon": -80.6646}] * 2
        title = build_lf_title(
            [_seg(32.3095, -80.6332), _seg(32.6591, -80.6646)],
            _CrossCountyGeo(),
            per_lf_location="Beaufort County, South Carolina (Blocker Run Road)",
            gps_points=gps,
        )
        assert "Sea Island Parkway" in title
        assert "Blocker Run Road" not in title


# ── Length enforcement ───────────────────────────────────────────────


class TestTitleLengthEnforcement:
    def test_title_under_100_chars_unchanged(self):
        segs = [_seg(32.3, -80.6), _seg(32.4, -80.7)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Port Royal"),
            _loc(city="Lady's Island"),
        ]
        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Beaufort County, South Carolina (Sea Island Parkway)",
        )
        assert len(title) <= 100

    def test_excessive_title_degrades(self):
        """A very long start/end/road combo should degrade to fit 100 chars."""
        segs = [_seg(32.3, -80.6), _seg(32.4, -80.7)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="A" * 50),  # absurdly long town name
            _loc(city="B" * 50),
        ]
        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Beaufort County, South Carolina (C" + "c" * 40 + " Road)",
        )
        assert len(title) <= 100


# ── Cross-LF uniqueness ──────────────────────────────────────────────


class TestEnsureUniqueTitles:
    def test_unique_titles_unchanged(self):
        titles = {1: "A", 2: "B", 3: "C"}
        assert ensure_unique_titles(titles) == {1: "A", 2: "B", 3: "C"}

    def test_collision_gets_part_suffix(self):
        """LF 2 and LF 4 both 'Colleton County, SC' — and LF 3 differs only
        by a via-road, which the final-title reshape strips: all three
        collapse to one assembled title, so all three need Part suffixes."""
        titles = {
            1: "Beaufort County, SC",
            2: "Colleton County, SC",
            3: "Colleton County via Blocker Run Road, SC",
            4: "Colleton County, SC",  # collides with LF 2
            5: "Transylvania County, NC",
        }
        result = ensure_unique_titles(titles)
        # Unique ones unchanged
        assert result[1] == "Beaufort County, SC"
        assert result[5] == "Transylvania County, NC"
        # Collisions in the final-title namespace get (Part N of M) suffixes
        assert result[2] == "Colleton County, SC (Part 1 of 3)"
        assert result[3] == "Colleton County via Blocker Run Road, SC (Part 2 of 3)"
        assert result[4] == "Colleton County, SC (Part 3 of 3)"
        # All 5 titles distinct after dedup
        assert len(set(result.values())) == 5

    def test_empty_input(self):
        assert ensure_unique_titles({}) == {}

    def test_three_way_collision(self):
        titles = {1: "X", 2: "X", 3: "X"}
        result = ensure_unique_titles(titles)
        assert result[1] == "X (Part 1 of 3)"
        assert result[2] == "X (Part 2 of 3)"
        assert result[3] == "X (Part 3 of 3)"

    def test_published_none_is_unchanged(self):
        """Omitting published_titles preserves the in-batch-only behaviour."""
        titles = {1: "A", 2: "B"}
        assert ensure_unique_titles(titles, published_titles=None) == {1: "A", 2: "B"}
        assert ensure_unique_titles(titles, published_titles=set()) == {1: "A", 2: "B"}

    def test_exact_channel_collision_raises_naming_video_id(self):
        """A label unique in-batch whose bare assembled form exactly
        matches a live channel title is a hard failure, not a silent
        "(Part N)" rename."""
        from gomoveshift.pipeline.exceptions import PhaseBCExitContractError

        with pytest.raises(PhaseBCExitContractError, match="yt_abc123"):
            ensure_unique_titles(
                {1: "Foothills Parkway, TN", 2: "Cherohala Skyway, NC"},
                published_titles={"foothills parkway, tn in 4k"},
                published_title_ids={"foothills parkway, tn in 4k": "yt_abc123"},
            )

    def test_in_batch_and_channel_collision_combined(self):
        """Two in-batch dupes get (Part N of M); if the deduped form still
        matches a published title it is bumped further."""
        result = ensure_unique_titles(
            {1: "Skyline Drive, VA", 2: "Skyline Drive, VA"},
            published_titles={"Skyline Drive, VA in 4K (Part 1 of 2)"},
        )
        # Part 1 of 2 collides with the channel → bumped; Part 2 of 2 is free.
        assert result[1] == "Skyline Drive, VA (Part 2)"
        assert result[2] == "Skyline Drive, VA (Part 2 of 2)"
        assert len(set(result.values())) == 2


# ── build_superlong_part_title ───────────────────────────────────────


class TestBuildSuperlongPartTitle:
    def test_single_part_no_suffix(self):
        title = build_superlong_part_title("Charleston to Savannah, SC", 1, 1)
        assert title == "Driving Charleston to Savannah, SC"
        assert "Part" not in title

    def test_two_part_day_explicit_suffixes(self):
        base = "Charleston to Savannah, SC"
        t1 = build_superlong_part_title(base, 1, 2)
        t2 = build_superlong_part_title(base, 2, 2)
        assert t1 == "Driving Charleston to Savannah, SC — Part 1 of 2"
        assert t2 == "Driving Charleston to Savannah, SC — Part 2 of 2"
        assert t1 != t2

    def test_trim_ratio_selects_the_same_honesty_verb_as_full_drive_lfs(self):
        """Reuses longform_metadata._select_verb's threshold rather than
        hardcoding "Driving" — a stops-only cut (high kept ratio, the
        normal superlong case) reads as Driving; a hypothetical heavy
        edit still reads as Highlights, never mislabeled."""
        assert build_superlong_part_title("X", 1, 1, trim_ratio=0.95).startswith("Driving ")
        assert build_superlong_part_title("X", 1, 1, trim_ratio=0.2).startswith("Highlights from ")

    def test_idempotent_no_double_verb_or_suffix(self):
        assert build_superlong_part_title("Driving Route X, NC", 1, 1) == "Driving Route X, NC"
        once = build_superlong_part_title("Route X, NC", 1, 2)
        assert build_superlong_part_title(once, 1, 2) == once
        assert once.count("Part 1 of 2") == 1


class TestSuperlongTitleSurvivesDedup:
    def test_two_part_day_titles_pass_through_dedup_unchanged(self):
        """The explicit per-part suffix already makes the two titles
        distinct, so ``ensure_unique_titles`` must not touch them."""
        base = "Charleston to Savannah, SC"
        titles = {
            1: build_superlong_part_title(base, 1, 2),
            2: build_superlong_part_title(base, 2, 2),
        }
        result = ensure_unique_titles(titles)
        assert result == titles
        assert "(Part" not in result[1] and "(Part" not in result[2]

    def test_explicit_suffix_not_double_suffixed_by_dedup(self):
        """Contrived residual collision: two already-explicitly-suffixed
        titles that still collide must not get the dedup Counter's
        ``(Part N of M)`` stacked on top of the em-dash suffix."""
        titles = {1: "Test Loop, NC — Part 1 of 2", 2: "Test Loop, NC — Part 1 of 2"}
        result = ensure_unique_titles(titles)
        assert "(Part 1 of 2) (Part 1 of 2)" not in result[1]
        assert result[1].count("Part 1 of 2") == 1


# ── Geocoder exception handling ──────────────────────────────────────


class TestGeocoderExceptions:
    def test_reverse_geocode_exception_degrades_gracefully(self):
        """Geocoder raising must not crash the title build — fall back to county."""
        segs = [_seg(32.3, -80.6), _seg(32.4, -80.7)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = RuntimeError("network down")

        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Beaufort County, South Carolina",
        )
        # Should NOT raise; falls to county/state fallback
        assert title == "Beaufort County, SC"

    def test_reverse_geocode_returns_none_attrs(self):
        """Geocoder returning an object with no usable attrs → fall back."""
        segs = [_seg(32.3, -80.6), _seg(32.4, -80.7)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [_loc(), _loc()]  # all None attrs

        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Beaufort County, South Carolina",
        )
        # No start/end town available → Level 4
        assert title == "Beaufort County, SC"


# ── Titles adopt per_lf_ost_text directly (single source) ────
# When the caller has already computed the canonical OST text per LF
# (`state["per_lf_ost_text"]`, computed upstream), `build_lf_title` must
# adopt it verbatim — no geocoding detour. This keeps OST / title /
# thumbnail aligned from one source.


class TestPerLFOSTTextParameter:
    """Tests for the ``per_lf_ost_text`` passthrough parameter.

    Contract: when ``per_lf_ost_text`` is a non-empty string, the
    builder returns it verbatim (after length enforcement) and does
    NOT call the geocoder. When it's None or empty, the builder falls
    back to the existing geocoding cascade for backward compatibility.
    """

    def test_title_uses_ost_text_when_provided(self):
        """Passing a canonical ost_text string → returned verbatim,
        geocoder never called."""
        segs = [_seg(32.3, -80.6), _seg(32.4, -80.7)]
        geocoder = MagicMock()
        # Explicitly signal we do NOT expect a geocoder call:
        geocoder.reverse_geocode.side_effect = AssertionError(
            "geocoder must not be called when per_lf_ost_text is set"
        )

        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Beaufort County, South Carolina (Sea Island Parkway)",
            per_lf_ost_text="Port Royal to Lady's Island via Sea Island Parkway, SC",
        )
        assert title == "Port Royal to Lady's Island via Sea Island Parkway, SC"
        geocoder.reverse_geocode.assert_not_called()

    def test_title_falls_back_to_geocoding_when_no_ost_text(self):
        """Without ost_text → legacy cascade (backward compat)."""
        segs = [_seg(32.3, -80.6), _seg(32.4, -80.7)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Port Royal"),
            _loc(city="Lady's Island"),
        ]

        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Beaufort County, South Carolina (Sea Island Parkway)",
            per_lf_ost_text=None,
        )
        assert title == "Port Royal to Lady's Island via Sea Island Parkway, SC"
        # Legacy cascade actually hit the geocoder
        assert geocoder.reverse_geocode.call_count == 2

    def test_title_falls_back_when_ost_text_is_empty_string(self):
        """Empty string ost_text is treated as unset (falls through)."""
        segs = [_seg(32.3, -80.6), _seg(32.4, -80.7)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Port Royal"),
            _loc(city="Lady's Island"),
        ]

        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Beaufort County, South Carolina (Sea Island Parkway)",
            per_lf_ost_text="",
        )
        assert title == "Port Royal to Lady's Island via Sea Island Parkway, SC"

    def test_title_enforces_length_on_ost_text(self):
        """Very long ost_text gets the same length enforcement as the
        legacy cascade — returned title must be ≤ MAX_TITLE_LEN."""
        over_the_limit = "A" * (MAX_TITLE_LEN + 25)
        title = build_lf_title(
            segments=[],
            geocoder=None,
            per_lf_location="Beaufort County, South Carolina",
            per_lf_ost_text=over_the_limit,
        )
        assert len(title) <= MAX_TITLE_LEN, (
            f"ost_text longer than MAX_TITLE_LEN must be trimmed; got {len(title)} chars"
        )

    def test_title_within_limit_ost_text_unchanged(self):
        """A normal-length canonical string is returned verbatim."""
        canonical = "Port Royal to Lady's Island via Sea Island Parkway, SC"
        title = build_lf_title(
            segments=[],
            geocoder=None,
            per_lf_location="Beaufort County, South Carolina (Sea Island Parkway)",
            per_lf_ost_text=canonical,
        )
        assert title == canonical

    def test_title_ost_text_no_geocoder_no_segments(self):
        """ost_text is authoritative even when the legacy cascade
        couldn't produce a title (no geocoder, no segments)."""
        title = build_lf_title(
            segments=[],
            geocoder=None,
            per_lf_location="",  # legacy path would return ""
            per_lf_ost_text="Port Royal to Lady's Island via Sea Island Parkway, SC",
        )
        assert title == "Port Royal to Lady's Island via Sea Island Parkway, SC"


class TestUniqueTitlesIdenticalOSTText:
    """Dedup contract: ``ensure_unique_titles`` still fires on
    identical TITLES (including when they came from identical ost_text),
    but per_lf_ost_text in the caller's state must stay clean — the
    dedup suffix is a TITLE-only policy."""

    def test_unique_titles_with_identical_ost_text(self):
        """Two LFs sharing the same canonical ost_text collide on the
        title map → ensure_unique_titles appends (Part N of M)."""
        shared = "Port Royal to Lady's Island via Sea Island Parkway, SC"
        titles = {1: shared, 2: shared, 3: "Different Title"}
        result = ensure_unique_titles(titles)
        assert result[1] == f"{shared} (Part 1 of 2)"
        assert result[2] == f"{shared} (Part 2 of 2)"
        assert result[3] == "Different Title"

    def test_ensure_unique_titles_does_not_mutate_input(self):
        """Contract: ``ensure_unique_titles`` returns a NEW dict — it
        must never mutate the caller's input dict. The caller stores
        ``state["per_lf_ost_text"] = dict(per_lf_ost_text)`` BEFORE
        passing it in, so mutation there would corrupt the canonical
        OST text source."""
        shared = "Port Royal to Lady's Island via Sea Island Parkway, SC"
        ost_text_source = {1: shared, 2: shared}
        snapshot = dict(ost_text_source)

        _result = ensure_unique_titles(ost_text_source)
        # Input must be untouched.
        assert ost_text_source == snapshot, (
            f"ensure_unique_titles mutated its input — canonical OST text "
            f"would be corrupted. Before: {snapshot}. After: {ost_text_source}"
        )


# ── GPS-quality filter + multi-road titles ────────────────────
#
# ``build_lf_title`` now accepts optional ``gps_points`` and ``roads``
# kwargs. When ``gps_points`` is provided, the builder runs
# ``filter_high_quality_points`` before computing start/end centroids
# so noisy / stationary GPS can't drag the centroid into a
# neighbouring town. When ``roads`` is provided, the builder emits
# the multi-road format ``"A to B via R1 + R2 + R3, SC"`` up to
# ``max_count=3`` roads, collapsing to the single-road format when
# the multi-road title would exceed ``MAX_TITLE_LEN``.


class TestFilteredGPSForCentroids:
    def test_title_uses_filtered_gps_for_centroids(self):
        """gps_points is filtered before the centroid reducer — the
        geocoder should see the CLEAN centroid, not one dragged by the
        noisy/stationary cluster.
        """
        segs = [_seg(32.3, -80.6), _seg(32.4, -80.7)]
        clean = [
            {"lat": 32.30, "lon": -80.60, "hdop": 1.0, "speed_mps": 12.0, "time": 0.0},
            {"lat": 32.40, "lon": -80.70, "hdop": 1.0, "speed_mps": 12.0, "time": 120.0},
        ]
        noise = [
            {"lat": 33.0, "lon": -79.0, "hdop": 8.0, "speed_mps": 12.0, "time": 10.0},
            {"lat": 33.0, "lon": -79.0, "hdop": 1.0, "speed_mps": 0.0, "time": 20.0},
        ]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Port Royal"),
            _loc(city="Lady's Island"),
        ]

        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Beaufort County, South Carolina (Sea Island Parkway)",
            gps_points=clean + noise,
        )
        assert title.startswith("Port Royal to Lady's Island")
        first_call_args = geocoder.reverse_geocode.call_args_list[0][0]
        assert abs(first_call_args[0] - 32.30) < 0.1


class TestMultiRoadTitleFormat:
    def _roads(self, *names):
        return [SimpleNamespace(name=n, duration_sec=300.0, length_m=8000.0) for n in names]

    def test_title_multi_road_two(self):
        segs = [_seg(32.3, -80.6), _seg(32.4, -80.7)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Port Royal"),
            _loc(city="Beaufort"),
        ]
        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Beaufort County, South Carolina",
            roads=self._roads("US-17", "SC-170"),
        )
        assert title == "Port Royal to Beaufort via US-17 + SC-170, SC"

    def test_title_multi_road_three(self):
        segs = [_seg(32.3, -80.6), _seg(32.4, -80.7)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="A"),
            _loc(city="B"),
        ]
        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="X County, South Carolina",
            roads=self._roads("R1", "R2", "R3"),
        )
        assert title == "A to B via R1 + R2 + R3, SC"

    def test_title_degrades_to_single_road_when_too_long(self):
        """Over MAX_TITLE_LEN → collapse to single-road format."""
        segs = [_seg(32.3, -80.6), _seg(32.4, -80.7)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Longtownnameheregoesverylong"),
            _loc(city="Anotherlongtownnameforthetail"),
        ]
        long_roads = self._roads(
            "ReallyLongInterstateNameOne",
            "ReallyLongInterstateNameTwo",
            "ReallyLongInterstateNameThree",
        )
        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Some County, South Carolina (ReallyLongInterstateNameOne)",
            roads=long_roads,
        )
        assert len(title) <= MAX_TITLE_LEN
        # Single-road fallback — no "+ <road>" in the output.
        assert "+ ReallyLongInterstateNameTwo" not in title


class TestLoopTitleRequiresCircularGPS:
    """Level 3/3b 'Loop' titles must only fire when GPS confirms the drive
    is actually circular (start and end centroids within 5 km).  A linear
    drive through a small town produces start_town == end_town by geocoding
    but MUST NOT receive a 'Loop' title."""

    def _geocoder_same_town(self, town="Mars Hill"):
        gc = MagicMock()
        gc.reverse_geocode.return_value = _loc(city=town, state="North Carolina")
        return gc

    def _gps_linear(self):
        """GPS points spanning ~25 km — clearly not a loop."""
        return [
            SimpleNamespace(lat=35.87, lon=-82.57),
            SimpleNamespace(lat=35.89, lon=-82.60),
            SimpleNamespace(lat=36.00, lon=-82.65),
            SimpleNamespace(lat=36.09, lon=-82.70),
        ]

    def _gps_circular(self):
        """GPS points within 2 km — consistent with a small loop."""
        return [
            SimpleNamespace(lat=35.870, lon=-82.570),
            SimpleNamespace(lat=35.872, lon=-82.572),
            SimpleNamespace(lat=35.871, lon=-82.571),
        ]

    def test_linear_drive_same_town_no_loop_title(self):
        """When GPS centroids are far apart, 'Loop' must not appear in the title
        even if the geocoder returns the same town for start and end."""
        segs = [_seg(35.87, -82.57), _seg(35.89, -82.60)]
        title = build_lf_title(
            segs,
            self._geocoder_same_town(),
            per_lf_location="Madison County, NC",
            gps_points=self._gps_linear(),
        )
        assert "Loop" not in title, f"Linear drive must not produce Loop title: {title!r}"

    def test_actual_loop_gps_allows_loop_title(self):
        """When GPS centroids are within 5 km, level 3b may fire."""
        segs = [_seg(35.870, -82.570), _seg(35.872, -82.572)]
        title = build_lf_title(
            segs,
            self._geocoder_same_town(),
            per_lf_location="Mars Hill, NC",
            gps_points=self._gps_circular(),
        )
        assert "Loop" in title, f"Circular GPS drive should produce Loop title: {title!r}"

    def test_no_gps_no_loop_title(self):
        """Without GPS points the fallback centroids come from segment gps_center_*,
        which may put start and end in the same town. No Loop title must appear."""
        segs = [_seg(35.87, -82.57), _seg(35.88, -82.58)]
        title = build_lf_title(
            segs,
            self._geocoder_same_town(),
            per_lf_location="Madison County, NC",
        )
        assert "Loop" not in title, f"No-GPS same-town drive must not produce Loop title: {title!r}"

    def test_loop_road_name_filtered_from_level4(self):
        """When the road name contains 'Loop', level 4 must drop it and use
        the county-only fallback rather than 'county via Loop, state'."""
        segs = [_seg(35.87, -82.57)]
        gc = MagicMock()
        gc.reverse_geocode.return_value = None  # geocoder gives nothing
        title = build_lf_title(
            segs,
            gc,
            per_lf_location="Madison County Scenic Loop, NC",
            gps_points=self._gps_linear(),
        )
        assert "Loop" not in title, (
            f"Loop road name must be stripped from level-4 fallback: {title!r}"
        )


# ── dominant_location: deterministic, direction-independent vote ──


class TestDominantLocationVoteDeterminism:
    """The county vote must not depend on drive direction. A tie between two
    counties (equal footage) must resolve the SAME way regardless of which
    end of the trace the drive started from — last-resort tie-break is
    alphabetical, never insertion/sample order."""

    @staticmethod
    def _geocoder():
        """Reverse-geocode by latitude band: lat>=35 → Buncombe/NC,
        lat<35 → Avery/NC. Detailed dict shape per reverse_geocode_detailed."""
        gc = MagicMock()

        def _detailed(lat, lon):
            if lat >= 35.0:
                return {"county": "Buncombe", "state": "NC", "town": "Asheville"}
            return {"county": "Avery", "state": "NC", "town": "Banner Elk"}

        gc.reverse_geocode_detailed.side_effect = _detailed
        return gc

    def test_tie_resolves_alphabetically_both_directions(self):
        # Two coords per county => exact tie on sample count.
        northbound = [(34.0, -82.0), (34.5, -82.1), (35.5, -82.5), (36.0, -82.6)]
        southbound = list(reversed(northbound))

        north = dominant_location(northbound, self._geocoder(), max_samples=4)
        south = dominant_location(southbound, self._geocoder(), max_samples=4)

        assert north is not None and south is not None
        # Same winner regardless of direction.
        assert (north.county, north.state) == (south.county, south.state)
        # Alphabetical last-resort tie-break: Avery < Buncombe.
        assert north.county == "Avery"

    def test_footage_plurality_still_wins_over_alpha(self):
        # Buncombe holds 3 of 4 samples — plurality must beat the alpha rule.
        coords = [(34.0, -82.0), (35.2, -82.5), (35.5, -82.5), (36.0, -82.6)]
        res = dominant_location(coords, self._geocoder(), max_samples=4)
        assert res is not None
        assert res.county == "Buncombe"


class TestEvenlySampleDivByZero:
    """``_evenly_sample(seq, 1)`` must not div-by-zero (``k - 1`` == 0)."""

    def test_k1_returns_median_element(self):
        # n > 1, k == 1 hit the ``step = (n - 1) / (k - 1)`` divide before the guard.
        seq = list(range(7))
        assert _evenly_sample(seq, 1) == [seq[len(seq) // 2]]

    def test_k1_single_element(self):
        assert _evenly_sample([42], 1) == [42]

    def test_k1_empty(self):
        assert _evenly_sample([], 1) == []

    def test_weighted_k1_no_div_by_zero(self):
        # The weighted variant has a parallel ``len(sampled) - 1`` divide.
        seq = list(range(7))
        out = _evenly_sample_weighted(seq, 1)
        assert len(out) == 1
        item, weight = out[0]
        assert item == seq[len(seq) // 2]
        assert weight == len(seq)  # single sample stands for the whole sequence

    def test_k2_endpoints_preserved(self):
        # Regression: the k==1 guard must not perturb the k>=2 spread.
        seq = list(range(10))
        out = _evenly_sample(seq, 2)
        assert out[0] == seq[0] and out[-1] == seq[-1]


class _RegionGeo:
    """Endpoint centroids geocode to a broad REGION name (no town/village); the
    detailed geocode is empty so the resolved sequence is the only settlement
    source. Mirrors a drive through a wildlife management area resolving to "ACE Basin".
    """

    def reverse_geocode(self, lat, lon):
        return SimpleNamespace(
            city=None,
            locality=None,
            suburb=None,
            name="ACE Basin",
            town=None,
            place=None,
            hamlet=None,
            village=None,
        )

    def reverse_geocode_detailed(self, lat, lon):
        return {}


# Two fixes ~5 km apart inside the basin, under the via-road span.
_REGION_GPS = [{"lat": 32.601, "lon": -80.521}, {"lat": 32.575, "lon": -80.498}]


class TestLooksLikeRegion:
    def test_region_names_flagged(self):
        assert _looks_like_region("ACE Basin") is True
        assert _looks_like_region("ACE Basin, SC") is True  # with state suffix
        assert _looks_like_region("Donnelley Wildlife Management Area") is True
        assert _looks_like_region("Francis Marion National Forest") is True
        assert _looks_like_region("the Lowcountry") is True

    def test_park_and_conservation_tokens_flagged(self):
        # Broad park/protected-area names are regions, not settlements — they
        # must not win the endpoint/anchor slots.
        assert _looks_like_region("Congaree National Park") is True
        assert _looks_like_region("Table Rock State Park") is True
        assert _looks_like_region("Green River Conservation Area") is True

    def test_real_settlements_not_flagged(self):
        # Ordinary towns in the same drive must survive the guard.
        assert _looks_like_region("Beaufort") is False
        assert _looks_like_region("Walterboro") is False
        # A town whose name merely contains a region token as a substring is
        # safe — the word-boundary regex only matches the standalone word.
        assert _looks_like_region("Basinger") is False  # "Basin" is a substring, not a word
        assert _looks_like_region("Forest City") is False  # "Forest" alone isn't a region token

    def test_none_and_empty_not_flagged(self):
        assert _looks_like_region(None) is False
        assert _looks_like_region("") is False

    def test_county_still_handled_by_county_guard(self):
        # Region guard is orthogonal — a county name is NOT a region, the
        # existing county guard owns it.
        assert _looks_like_region("Colleton County") is False
        assert _looks_like_county("Colleton County") is True


class TestBuildLfTitleRegionDemoted:
    def test_region_endpoints_surface_sequence_town(self):
        # "ACE Basin" endpoints are demoted from the town slot so the
        # recognizable-settlement backfill surfaces the real town.
        title = build_lf_title(
            [],
            _RegionGeo(),
            "Colleton County, South Carolina",
            gps_points=_REGION_GPS,
            resolved_places=["ACE Basin", "Walterboro", "ACE Basin"],
        )
        assert title == "Walterboro, SC"

    def test_all_region_sequence_falls_to_county(self):
        # No settlement anywhere — the title must NOT burn the broad region; it
        # degrades honestly to the county.
        title = build_lf_title(
            [],
            _RegionGeo(),
            "Colleton County, South Carolina",
            gps_points=_REGION_GPS,
            resolved_places=["ACE Basin", "ACE Basin", "ACE Basin"],
        )
        assert title == "Colleton County, SC"


# ── Endpoint derivation: head/tail windows, POI reject, cross-state ──


class TestEndpointWindowCentroids:
    def test_head_tail_windows_not_half_centroids(self):
        # 20 fixes along a line. A HALF-trace mean sits ~¼-route inland
        # (lat ≈ 32.045 / 32.145); the head/tail windows must hug the true
        # origin/destination instead.
        pts = [{"lat": 32.0 + 0.01 * i, "lon": -80.0 - 0.01 * i} for i in range(20)]
        first, last = _first_last_centroids_from_gps(pts)
        assert abs(first[0] - 32.00) < 0.005, f"start window drifted mid-route: {first}"
        assert abs(last[0] - 32.19) < 0.005, f"end window drifted mid-route: {last}"

    def test_single_point_and_empty(self):
        c = _first_last_centroids_from_gps([{"lat": 32.0, "lon": -80.0}])
        assert c == ((32.0, -80.0), (32.0, -80.0))
        assert _first_last_centroids_from_gps([]) == (None, None)


class TestGeocodeTownRejectsPoiName:
    def test_name_field_not_used(self):
        # ``name`` prefers poi_name (a business the vehicle passed near) — it
        # must never become a burned endpoint town.
        geocoder = MagicMock()
        geocoder.reverse_geocode.return_value = _loc(name="Dickie's Pub")
        assert _geocode_town(geocoder, 32.3, -80.6) is None

    def test_settlement_fields_still_used(self):
        geocoder = MagicMock()
        geocoder.reverse_geocode.return_value = _loc(town="Walterboro", name="Dickie's Pub")
        assert _geocode_town(geocoder, 32.3, -80.6) == "Walterboro"


class TestGeocodeTownRankOrder:
    """``_geocode_town`` must prefer the most significant settlement
    class — city/town over village/hamlet — not hamlet-over-city."""

    def test_city_beats_hamlet(self):
        geocoder = MagicMock()
        geocoder.reverse_geocode.return_value = _loc(hamlet="Cline", city="Wytheville")
        assert _geocode_town(geocoder, 36.9, -81.1) == "Wytheville"

    def test_city_beats_town(self):
        geocoder = MagicMock()
        geocoder.reverse_geocode.return_value = _loc(town="Fort Chiswell", city="Wytheville")
        assert _geocode_town(geocoder, 36.9, -81.1) == "Wytheville"

    def test_hamlet_alone_still_used(self):
        geocoder = MagicMock()
        geocoder.reverse_geocode.return_value = _loc(hamlet="Cline")
        assert _geocode_town(geocoder, 36.9, -81.1) == "Cline"


class _WytheGeo:
    """Wythe County trace: southern fixes geocode to hamlet Cline, northern
    fixes to town Wytheville. Endpoint ``reverse_geocode`` resolves NOTHING so
    the place-sequence ladder supplies every settlement."""

    def reverse_geocode(self, lat, lon):
        return _loc()

    def reverse_geocode_detailed(self, lat, lon):
        if lat < 36.90:
            return {"county": "Wythe County", "state": "Virginia", "hamlet": "Cline"}
        return {"county": "Wythe County", "state": "Virginia", "town": "Wytheville"}


_WYTHE_GPS = [{"lat": 36.85 + 0.01 * i, "lon": -81.15 + 0.01 * i} for i in range(10)]


class TestSettlementAnchorRankAware:
    def test_level4_anchor_prefers_town_over_first_sequence_hamlet(self):
        """A non-contiguous LF's Level-4 anchor must name the highest-
        class settlement (Wytheville), not the travel-order-first hamlet."""
        segs = [_seg_c(36.85, -81.15, True), _seg_c(36.94, -81.06, False)]
        title = build_lf_title(
            segs,
            _WytheGeo(),
            "Wythe County, Virginia",
            gps_points=_WYTHE_GPS,
        )
        assert title == "Wytheville, VA"


class _CrossStateGeo:
    """SC→NC crossing whose central third sits in a THIRD county, so the
    flake guard keeps both endpoints (genuine crossing)."""

    def _band(self, lat):
        if lat <= 35.10:
            return ({"county": "Spartanburg County", "state": "South Carolina"}, "Landrum")
        if lat >= 35.20:
            return ({"county": "Polk County", "state": "North Carolina"}, "Saluda")
        return ({"county": "Greenville County", "state": "South Carolina"}, None)

    def reverse_geocode(self, lat, lon):
        return _loc(town=self._band(lat)[1])

    def reverse_geocode_detailed(self, lat, lon):
        return dict(self._band(lat)[0])


_CROSS_STATE_GPS = [
    {"lat": lat, "lon": -82.2}
    for lat in (35.05, 35.06, 35.12, 35.13, 35.14, 35.15, 35.16, 35.17, 35.24, 35.25)
]


class TestCrossStateShape:
    def test_each_endpoint_carries_own_state(self):
        # Shape A: never "Landrum to Saluda, SC" — each town gets its own state.
        title = build_lf_title(
            [],
            _CrossStateGeo(),
            "Spartanburg County, South Carolina",
            gps_points=_CROSS_STATE_GPS,
        )
        assert title == "Landrum, SC to Saluda, NC"

    def test_cross_state_with_road(self):
        title = build_lf_title(
            [],
            _CrossStateGeo(),
            "Spartanburg County, South Carolina",
            gps_points=_CROSS_STATE_GPS,
            resolved_road="US-176",
        )
        assert title == "Landrum, SC to Saluda, NC via US-176"


# ── Stateless vote-key serialization ────────────────────────────


class _StatelessGeo:
    """Winning county with NO state/province — the state slot is Python
    ``None`` (Irish counties, independent cities like Washington DC)."""

    def reverse_geocode_detailed(self, lat, lon):
        return {"county": "County Kerry", "state": None, "country": "Ireland"}


class TestStatelessVoteKey:
    """A None state must serialize to an EMPTY slot (``"County|"``), never
    the literal string ``"None"`` — otherwise the consumer partition yields the
    truthy ``"None"`` and a malformed ``"County, None"`` region that false-fails
    the cross-surface location invariant."""

    def test_none_state_serializes_to_empty_slot_not_literal_none(self):
        dom = dominant_location([{"lat": 52.1409, "lon": -10.2683}] * 8, _StatelessGeo())
        assert dom is not None
        assert all("None" not in key for key in dom.votes)
        assert "County Kerry|" in dom.votes


# ── Prime-meridian longitude ───────────────────────────────────


class TestPointLatlonPrimeMeridian:
    """A model object on the prime meridian (lon == 0.0 — UK/France/West
    Africa) must NOT be treated as missing. The falsy ``or`` chain that fell
    back to ``.lng`` dropped these valid points."""

    def test_object_lon_zero_is_valid_not_dropped(self):
        point = SimpleNamespace(lat=51.5, lon=0.0)  # Greenwich; no .lng attr
        assert _point_latlon(point) == (51.5, 0.0)

    def test_object_lon_zero_does_not_fall_through_to_lng(self):
        # ``.lon`` is present and 0.0 → must win over a stray ``.lng``.
        point = SimpleNamespace(lat=51.5, lon=0.0, lng=99.9)
        assert _point_latlon(point) == (51.5, 0.0)

    def test_object_null_island_still_returned_by_helper(self):
        # (0.0, 0.0) is returned here; the null-island guard drops it later.
        point = SimpleNamespace(lat=0.0, lon=0.0)
        assert _point_latlon(point) == (0.0, 0.0)


# ── _geocode_town field widening ─────────────────────────────────────


def test_geocode_town_widens_to_village_hamlet_place():
    """``_geocode_town`` anchors endpoints on village/hamlet/place labels, not
    just city/town."""
    segs = [_seg(52.10, -10.27), _seg(52.14, -10.33)]
    for attr in ("village", "hamlet", "place"):
        g = MagicMock()
        g.reverse_geocode.side_effect = [
            _loc(**{attr: "Ballyferriter"}),
            _loc(**{attr: "Dunquin"}),
        ]
        title = build_lf_title(
            segs,
            g,
            per_lf_location="County Kerry, Ireland (Slea Head Drive)",
            state_abbr="Kerry",
        )
        assert "Ballyferriter" in title and "Dunquin" in title, f"{attr} attr missed"


# ── Corrections ledger feeds the title's place-name map/denylist ──


class TestPlaceCorrections:
    """A viewer-flagged place-name mistake, once corrected in the ledger,
    must not be regenerated the next time the pipeline drives the same
    area."""

    def _title(self, ost_text, corrections=None, location="Beaufort County, South Carolina"):
        return build_lf_title(
            segments=[],
            geocoder=None,
            per_lf_location=location,
            per_lf_ost_text=ost_text,
            place_corrections=corrections,
        )

    def test_mapped_wrong_name_is_substituted(self):
        corrections = PlaceCorrections(mapping={"Ridgeway": "Ridgeville"})
        title = self._title("Charleston to Ridgeway via US-17, SC", corrections)
        assert title == "Charleston to Ridgeville via US-17, SC"

    def test_denylisted_name_is_refused_and_falls_back(self):
        corrections = PlaceCorrections(denylist=("Frogmore Depot",))
        title = self._title("Beaufort to Frogmore Depot, SC", corrections)
        assert title == "Beaufort County, South Carolina"

    def test_empty_corrections_is_a_strict_noop(self):
        title = self._title("Beaufort to Port Royal, SC", PlaceCorrections())
        assert title == "Beaufort to Port Royal, SC"

    def test_missing_ledger_defaults_to_noop(self, monkeypatch, tmp_path):
        """No ``place_corrections`` kwarg -> loads the (missing) default
        ledger path, which must be a strict no-op."""
        monkeypatch.setenv("GMS_CORRECTIONS_PATH", str(tmp_path / "absent.yaml"))
        assert self._title("Beaufort to Port Royal, SC") == "Beaufort to Port Royal, SC"
