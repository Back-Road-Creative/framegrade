"""Tests for ``dominant_location`` — the county-vote reducer that replaces
single-centroid reverse-geocoding on the drive day-first label path.

Root cause it guards: a drive's GPS centroid (arithmetic mean) lands on the
wrong side of a county/state line when the route crosses one. A route
that films both Beaufort County and Colleton County can have a mean that
geocodes to the minority county.
``dominant_location`` votes the modal county across sampled points instead,
so the label always reflects the county holding the plurality of footage.
"""

from gomoveshift.pipeline.title_builder import DominantLocation, dominant_location

# Example points.
_FRIPP = (32.31, -80.63)  # Beaufort County; city/town null, hamlet="Fort Fremont"
_FRIPP2 = (32.30, -80.64)  # Beaufort County
_EDISTO = (32.66, -80.66)  # Colleton County


def _make_geocoder(by_point):
    """Mock geocoder: nearest cached point (within ~0.1 deg) wins."""

    class _Geo:
        def reverse_geocode_detailed(self, lat, lon):
            best = None
            best_d = 0.15
            for (clat, clon), data in by_point.items():
                d = abs(lat - clat) + abs(lon - clon)
                if d < best_d:
                    best_d, best = d, data
            return dict(best) if best else {"county": None, "state": None}

    return _Geo()


_BEAUFORT_DETAIL = {
    "county": "Beaufort County",
    "state": "South Carolina",
    "hamlet": "Fort Fremont",
    "city": None,
    "town": None,
    "village": None,
}
_COLLETON_DETAIL = {
    "county": "Colleton County",
    "state": "South Carolina",
    "town": "Edisto Beach",
}


def _geo_sc():
    return _make_geocoder(
        {
            _FRIPP: _BEAUFORT_DETAIL,
            _FRIPP2: _BEAUFORT_DETAIL,
            _EDISTO: _COLLETON_DETAIL,
        }
    )


class TestDominantLocation:
    def test_beaufort_plurality_beats_colleton_minority(self):
        """A Fripp-dominant LF with a few Edisto points votes Beaufort."""
        points = [{"lat": _FRIPP[0], "lon": _FRIPP[1]}] * 8 + [
            {"lat": _EDISTO[0], "lon": _EDISTO[1]}
        ] * 2
        result = dominant_location(points, _geo_sc())
        assert isinstance(result, DominantLocation)
        assert result.county == "Beaufort County"
        assert result.state == "South Carolina"

    def test_pure_colleton_track_stays_colleton(self):
        """No false demotion — a genuine Colleton-only drive labels Colleton."""
        points = [{"lat": _EDISTO[0], "lon": _EDISTO[1]}] * 6
        result = dominant_location(points, _geo_sc())
        assert result.county == "Colleton County"

    def test_hamlet_surfaces_when_city_town_null(self):
        """Fripp Island has no city/town — the name falls back to hamlet."""
        points = [{"lat": _FRIPP[0], "lon": _FRIPP[1]}] * 5
        result = dominant_location(points, _geo_sc())
        assert result.county == "Beaufort County"
        assert result.name == "Fort Fremont"

    def test_road_comes_from_winning_county_not_minority(self):
        """``road`` must be a road sampled in the WINNING (voted) county,
        never the runner-up's. A sub-50km crossing whose centroid lands in the
        minority county would otherwise stamp the wrong-county road."""
        beaufort = dict(_BEAUFORT_DETAIL, road="Sea Island Parkway")
        colleton = dict(_COLLETON_DETAIL, road="Blocker Run Road")
        geo = _make_geocoder({_FRIPP: beaufort, _FRIPP2: beaufort, _EDISTO: colleton})
        points = [{"lat": _FRIPP[0], "lon": _FRIPP[1]}] * 8 + [
            {"lat": _EDISTO[0], "lon": _EDISTO[1]}
        ] * 2
        result = dominant_location(points, geo)
        assert result.county == "Beaufort County"
        # The road must be Beaufort's road, NOT Colleton's (the minority).
        assert result.road == "Sea Island Parkway"

    def test_handles_tuple_points_shorts_shape(self):
        """Shorts pass (t, lat, lon) tuples — must be accepted too."""
        points = [(0.0, _FRIPP[0], _FRIPP[1]), (1.0, _FRIPP2[0], _FRIPP2[1])] * 4
        result = dominant_location(points, _geo_sc())
        assert result.county == "Beaufort County"

    def test_vote_breakdown_exposed_for_dry_run(self):
        points = [{"lat": _FRIPP[0], "lon": _FRIPP[1]}] * 8 + [
            {"lat": _EDISTO[0], "lon": _EDISTO[1]}
        ] * 2
        result = dominant_location(points, _geo_sc())
        # winning bucket present with the larger count
        beaufort = result.votes.get("Beaufort County|South Carolina", 0)
        colleton = result.votes.get("Colleton County|South Carolina", 0)
        assert beaufort > colleton >= 1

    def test_none_when_no_geocoder(self):
        assert dominant_location([{"lat": 1.0, "lon": 2.0}], None) is None

    def test_none_when_no_points(self):
        assert dominant_location([], _geo_sc()) is None

    def test_rejects_null_island(self):
        """(0,0) GPS no-lock sentinels are never geocoded."""
        result = dominant_location([{"lat": 0.0, "lon": 0.0}], _geo_sc())
        assert result is None


# ── Geocode quorum floor ──────────────────────────────────────────────
# A far-road weak plurality off too few geocoded samples must NOT publish.
# ``_geo_sc`` returns {"county": None, "state": None} for any point not within
# ~0.15 deg of a cached coord, so a point just outside that radius produces NO
# vote — it models a reverse-geocode miss. ``_BLANK`` sits ~25 km from Fripp:
# close enough to survive the spatial-outlier quality filter (so the lone Fripp
# fix is NOT stripped as the outlier), but outside the cache-match radius so it
# never geocodes to a county — leaving only the genuinely located samples to
# count toward the quorum.
_BLANK = (32.10, -80.40)  # off the cached coords; geocodes to no county/state


# ── Rank-aware name pick (Wytheville beats Cline) ───────────────────
# The winning-county name must be picked by settlement class across ALL
# winning-county samples — a town beats a hamlet regardless of which was
# sampled first (travel order). If the first Wythe County sample geocodes the
# hamlet "Cline" and the town Wytheville appears only in later samples, the
# label must still be the town, not "Cline, Wythe County, Virginia".
_CLINE = (36.85, -81.15)  # Wythe County hamlet
_WYTHEVILLE = (36.95, -81.08)  # Wythe County town

_CLINE_DETAIL = {
    "county": "Wythe County",
    "state": "Virginia",
    "hamlet": "Cline",
    "place_rank": 20,
    "addresstype": "hamlet",
}
_WYTHEVILLE_DETAIL = {
    "county": "Wythe County",
    "state": "Virginia",
    "town": "Wytheville",
    "place_rank": 18,
    "addresstype": "town",
}


def _wythe_points(first, second, n_first=4, n_second=4):
    return [{"lat": first[0], "lon": first[1]}] * n_first + [
        {"lat": second[0], "lon": second[1]}
    ] * n_second


class TestRankAwareNamePick:
    def test_town_beats_hamlet_sampled_first(self):
        """Hamlet sampled FIRST (travel order) must still lose to the town."""
        geo = _make_geocoder({_CLINE: _CLINE_DETAIL, _WYTHEVILLE: _WYTHEVILLE_DETAIL})
        result = dominant_location(_wythe_points(_CLINE, _WYTHEVILLE), geo)
        assert result.county == "Wythe County"
        assert result.name == "Wytheville"

    def test_town_beats_hamlet_reversed_order(self):
        """Direction independence: town first gives the same answer."""
        geo = _make_geocoder({_CLINE: _CLINE_DETAIL, _WYTHEVILLE: _WYTHEVILLE_DETAIL})
        result = dominant_location(_wythe_points(_WYTHEVILLE, _CLINE), geo)
        assert result.name == "Wytheville"

    def test_legacy_cache_without_rank_fields_still_ranks_by_field(self):
        """Legacy cached dicts lack place_rank/addresstype entirely — the
        field-derived class (town field > hamlet field) must still decide."""
        cline = {"county": "Wythe County", "state": "Virginia", "hamlet": "Cline"}
        wytheville = {"county": "Wythe County", "state": "Virginia", "town": "Wytheville"}
        geo = _make_geocoder({_CLINE: cline, _WYTHEVILLE: wytheville})
        result = dominant_location(_wythe_points(_CLINE, _WYTHEVILLE), geo)
        assert result.name == "Wytheville"

    def test_addresstype_refines_generic_name_field_pick(self):
        """A sample whose settlement arrived only in the generic ``name`` slot
        is promoted by the ``addresstype`` ("town") above a hamlet."""
        wytheville_name_only = {
            "county": "Wythe County",
            "state": "Virginia",
            "name": "Wytheville",
            "addresstype": "town",
            "place_rank": 18,
        }
        geo = _make_geocoder({_CLINE: _CLINE_DETAIL, _WYTHEVILLE: wytheville_name_only})
        result = dominant_location(_wythe_points(_CLINE, _WYTHEVILLE), geo)
        assert result.name == "Wytheville"

    def test_unranked_name_field_poi_loses_to_hamlet(self):
        """A bare ``name``-field string with NO rank fields (POI-prone) stays
        least significant — the hamlet wins even when the POI is sampled first."""
        poi = {"county": "Wythe County", "state": "Virginia", "name": "Big Walker Lookout"}
        geo = _make_geocoder({_WYTHEVILLE: poi, _CLINE: _CLINE_DETAIL})
        result = dominant_location(_wythe_points(_WYTHEVILLE, _CLINE), geo)
        assert result.name == "Cline"

    def test_single_sample_hamlet_only_county_keeps_hamlet(self):
        """A one-settlement county still surfaces its hamlet
        (better than the bare county) — rank awareness never blanks the name."""
        geo = _make_geocoder({_CLINE: _CLINE_DETAIL})
        result = dominant_location([{"lat": _CLINE[0], "lon": _CLINE[1]}] * 5, geo)
        assert result.county == "Wythe County"
        assert result.name == "Cline"


class TestGeocodeQuorumFloor:
    def test_single_geocoded_sample_misses_quorum(self):
        """Only ONE sample resolves to a county; the rest geocode to nothing.
        With ``min_votes=2`` the weak plurality is rejected (returns None) so
        the caller degrades to its centroid/fallback label path."""
        points = [{"lat": _FRIPP[0], "lon": _FRIPP[1]}] + [{"lat": _BLANK[0], "lon": _BLANK[1]}] * 8
        assert dominant_location(points, _geo_sc(), min_votes=2) is None

    def test_single_geocoded_sample_still_wins_at_default(self):
        """Default ``min_votes=1`` preserves the earlier behaviour — a lone
        located sample still produces a winner for the many existing callers."""
        points = [{"lat": _FRIPP[0], "lon": _FRIPP[1]}] + [{"lat": _BLANK[0], "lon": _BLANK[1]}] * 8
        result = dominant_location(points, _geo_sc())
        assert result is not None
        assert result.county == "Beaufort County"

    def test_healthy_plurality_clears_quorum(self):
        """A multi-sample plurality has ≥2 geocoded votes and still wins under
        the floor — the quorum only kills the thin-evidence case."""
        points = [{"lat": _FRIPP[0], "lon": _FRIPP[1]}] * 8 + [
            {"lat": _EDISTO[0], "lon": _EDISTO[1]}
        ] * 2
        result = dominant_location(points, _geo_sc(), min_votes=2)
        assert result is not None
        assert result.county == "Beaufort County"


# ── One real county split across two "state" variants ───────────────────
# Nominatim returns an Irish county once WITH its province as ``state``
# ("Munster") and once STATELESS (state None). Keying the vote on the
# ``(county, state)`` tuple splits that ONE real county across two buckets, so a
# genuinely-minority county (fewer TOTAL samples) can win the plurality and
# stamp the WRONG county. The vote must be county-first: each county is
# collapsed onto its modal state BEFORE the tally so the real dominant wins.
_KERRY_MUNSTER = (52.14, -10.27)  # County Kerry, province-as-state
_KERRY_STATELESS = (52.05, -9.50)  # County Kerry, no state — the SAME real county
_CORK = (51.90, -8.47)  # County Cork — the true minority

_KERRY_M_DETAIL = {"county": "County Kerry", "state": "Munster", "town": "Dingle"}
_KERRY_S_DETAIL = {"county": "County Kerry", "state": None, "town": "Killarney"}
_CORK_DETAIL = {"county": "County Cork", "state": "Munster", "town": "Cork"}


class TestSplitStateCountyVote:
    def _geo_ie(self):
        return _make_geocoder(
            {
                _KERRY_MUNSTER: _KERRY_M_DETAIL,
                _KERRY_STATELESS: _KERRY_S_DETAIL,
                _CORK: _CORK_DETAIL,
            }
        )

    @staticmethod
    def _pt(coord):
        return {"lat": coord[0], "lon": coord[1]}

    def test_split_state_county_not_beaten_by_true_minority(self):
        """County Kerry (2 Munster + 2 stateless = 4 real samples) must beat
        County Cork (3 samples). Keying the vote on ``(county, state)`` splits
        Kerry into two 2-vote buckets and Cork's 3 wins the plurality — the
        WRONG county. County-first voting collapses Kerry to one bucket."""
        points = (
            [self._pt(_KERRY_MUNSTER)] * 2
            + [self._pt(_KERRY_STATELESS)] * 2
            + [self._pt(_CORK)] * 3
        )
        result = dominant_location(points, self._geo_ie())
        assert result is not None
        assert result.county == "County Kerry"

    def test_normal_single_state_vote_unchanged(self):
        """Guard: a single-state drive is unaffected — Cork alone wins when it
        genuinely holds the plurality and Kerry is not split across states."""
        points = [self._pt(_CORK)] * 5 + [self._pt(_KERRY_MUNSTER)] * 2
        result = dominant_location(points, self._geo_ie())
        assert result.county == "County Cork"
        assert result.state == "Munster"
