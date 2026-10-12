"""``ResolvedLocation`` builder + shared parse-time GPS-offset helper.

Covers the parse-time offset correction (``effective_offset`` corrects leading null-island
and stale-nonzero locks), the tuple→dict conversion before the quality filter,
geocode quorum floor, and vote/regions population for the ``location_invariant`` gate.
"""

import logging
from types import SimpleNamespace

from gomoveshift.analysis.shorts_models import GPSData, GPSTrackpoint, effective_offset
from gomoveshift.pipeline.gps_enrichment import (
    GEOCODE_QUORUM_MIN,
    _coerce_gps_points_to_dicts,
    _journey_endpoint_loss_by_lf,
    build_resolved_location,
    derive_per_lf_locations,
)
from gomoveshift.models import ResolvedLocation
from gomoveshift.pipeline.validation.content_gates import check_location_invariant

# Real cached coords (mirrors test_dominant_location.py).
_FRIPP = (32.3095, -80.6332)  # Beaufort County, SC
_EDISTO = (32.6591, -80.6646)  # Colleton County, SC
_BEAUFORT = {
    "county": "Beaufort County",
    "state": "South Carolina",
    "town": "St. Helena",
    "road": "Sea Island Parkway",
}
_COLLETON = {"county": "Colleton County", "state": "South Carolina", "town": "Edisto Beach"}


def _make_geocoder(by_point):
    class _Geo:
        def reverse_geocode_detailed(self, lat, lon):
            best, best_d = None, 0.15
            for (clat, clon), data in by_point.items():
                d = abs(lat - clat) + abs(lon - clon)
                if d < best_d:
                    best_d, best = d, data
            return dict(best) if best else {"county": None, "state": None}

    return _Geo()


def _geo_sc():
    return _make_geocoder({_FRIPP: _BEAUFORT, _EDISTO: _COLLETON})


# A non-US (Ireland) LF. Nominatim returns the province as ``state`` and a
# real ``country`` string; the builder must carry that country, not None.
_DINGLE = (52.1409, -10.2683)  # Dingle, County Kerry, Munster, Ireland
_KERRY = {
    "county": "County Kerry",
    "state": "Munster",
    "town": "Dingle",
    "road": "N86",
    "country": "Ireland",
}


def _geo_ie():
    return _make_geocoder({_DINGLE: _KERRY})


def _gps(*tps):
    return GPSData(trackpoints=[GPSTrackpoint(lat, lon, None, ts) for lat, lon, ts in tps])


class TestEffectiveOffset:
    def test_clean_trace_keeps_stored_offset(self):
        assert (
            effective_offset(_gps((32.3, -80.6, 1000.0), (32.31, -80.61, 1001.0)), 1000.0) == 1000.0
        )

    def test_leading_null_island_corrected(self):
        """A leading (0,0) cold-start run shifts the offset to the first fix."""
        gps = _gps((0.0, 0.0, 1000.0), (0.0, 0.0, 1001.0), (32.3, -80.6, 1005.0))
        assert effective_offset(gps, 1000.0) == 1005.0

    def test_leading_stale_nonzero_lock_corrected(self):
        """A leading repeated nonzero fix (stale clock lock) is skipped."""
        gps = _gps(
            (40.0, -75.0, 1000.0),  # stale lock, repeated
            (40.0, -75.0, 1001.0),
            (40.0, -75.0, 1002.0),
            (32.3, -80.6, 1050.0),  # real route begins
            (32.31, -80.61, 1051.0),
        )
        assert effective_offset(gps, 1000.0) == 1050.0

    def test_mid_trace_null_in_stale_run_not_returned(self):
        """A (0,0) null embedded AFTER a leading stale-lock run must be
        skipped, never returned as the offset — the null-island and stale-lock
        skips are independent (null-island is always invalid)."""
        gps = _gps(
            (40.0, -75.0, 1000.0),  # leading stale lock
            (40.0, -75.0, 1001.0),
            (0.0, 0.0, 1002.0),  # mid-trace null embedded in the stale run
            (32.3, -80.6, 1050.0),  # first genuine non-null, non-stale fix
        )
        assert effective_offset(gps, 1000.0) == 1050.0

    def test_parked_start_keeps_base_offset(self):
        """A genuine PARKED start — identical leading fixes at the TRUE
        start (a car sat ~5 min before pulling away), NEAR the subsequent
        trace and on the current epoch — must keep the base offset. The
        stale-lock override must NOT treat it as a teleport and jump the
        offset ~300 s forward to the first moving fix (which would drag every
        LF window + shorts trim across a county line)."""
        parked = _FRIPP  # (32.3095, -80.6332) Beaufort County, SC
        gps = _gps(
            (parked[0], parked[1], 1000.0),
            (parked[0], parked[1], 1150.0),
            (parked[0], parked[1], 1300.0),  # ~5 min parked, identical fixes
            (32.3096, -80.6330, 1305.0),  # pulls away ~25 m — NEAR, not a teleport
            (32.3100, -80.6325, 1306.0),
        )
        assert effective_offset(gps, 1000.0) == 1000.0

    def test_stale_lock_far_from_trace_still_overrides(self):
        """The genuine teleport-clock case (a long identical run whose first
        real fix is hundreds of km away) must STILL override the base offset —
        the parked-start guard only spares NEAR runs, never far ones."""
        gps = _gps(
            (40.0, -75.0, 1000.0),  # stale lock coordinate (~600 km away)
            (40.0, -75.0, 1001.0),
            (40.0, -75.0, 1002.0),
            (32.3, -80.6, 1050.0),  # real route begins — FAR
            (32.31, -80.61, 1051.0),
        )
        assert effective_offset(gps, 1000.0) == 1050.0

    def test_handles_none_and_dict_gps(self):
        assert effective_offset(None, 7.0) == 7.0
        tps = [
            {"lat": 0.0, "lon": 0.0, "timestamp": 5.0},
            {"lat": 32.3, "lon": -80.6, "timestamp": 9.0},
        ]
        assert effective_offset({"trackpoints": tps}, 5.0) == 9.0


# Tuple → dict conversion arms the (no-op-on-tuple) filter
def test_tuple_null_island_dropped_by_filter():
    coerced = _coerce_gps_points_to_dicts([(0.0, 0.0, 0.0), (1.0, 32.3, -80.6)])
    assert all(isinstance(p, dict) for p in coerced)
    from gomoveshift.pipeline.gps_quality import filter_high_quality_points

    survivors = filter_high_quality_points(coerced)
    assert all(not (p["lat"] == 0.0 and p["lon"] == 0.0) for p in survivors)


# Builder — vote / regions / quorum
_FR = {"lat": _FRIPP[0], "lon": _FRIPP[1]}
_ED = {"lat": _EDISTO[0], "lon": _EDISTO[1]}
# Sits off every cached coord → geocodes to no county (a reverse-geocode
# miss) yet survives the spatial-outlier filter (~25 km from Fripp), so a long
# trace of these plus ONE Fripp fix models the far-road single-flaky-geocode case.
_BLANK = {"lat": 32.10, "lon": -80.40}
_KE = {"lat": _DINGLE[0], "lon": _DINGLE[1]}


class TestBuildResolvedLocation:
    def test_votes_county_regions_vote_endpoints_country(self):
        rl = build_resolved_location([_FR] * 8 + [_ED] * 2, geocoder=_geo_sc())
        assert isinstance(rl, ResolvedLocation)
        assert (rl.dominant_county, rl.dominant_state) == ("Beaufort County", "South Carolina")
        assert rl.is_us is True and rl.country_code == "US"
        assert (
            "Beaufort County, South Carolina" in rl.regions
        )  # regions/vote feed the location_invariant gate
        assert "Colleton County, South Carolina" in rl.regions
        assert (
            rl.vote["Beaufort County, South Carolina"] > rl.vote["Colleton County, South Carolina"]
        )
        assert rl.endpoints["start"][0] == _FRIPP[0]

    def test_non_us_carries_geocoded_country_code(self):
        """A non-US LF must carry the geocoded COUNTRY in ``country_code``
        (not None) so every region surface routes off the ResolvedLocation
        instead of the label parse. ``is_us`` stays False — a US flip still
        needs a trailing US state, so a non-US country never flips the units."""
        rl = build_resolved_location([_KE] * 8, geocoder=_geo_ie())
        assert (rl.dominant_county, rl.dominant_state) == ("County Kerry", "Munster")
        assert rl.is_us is False
        assert rl.country_code == "Ireland"

    def test_non_us_resolved_routes_region_playlist_tags(self):
        """End-to-end: the built ResolvedLocation drives region/playlist/tags
        off the geocoded country, so a real ``", Kerry"`` production label routes
        to Ireland — not the label parse (which yields ``"kerry"`` / All Drives)."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.playlist_resolver import resolve_playlist
        from gomoveshift.pipeline.region_resolver import derive_region
        from gomoveshift.pipeline.social_generator import _region_tags

        rl = build_resolved_location([_KE] * 8, geocoder=_geo_ie())
        label = "Killarney to Dingle via N86, Kerry"  # documented real Ireland shape
        assert derive_region(label, resolved=rl) == "ireland"
        md = {"location": label, "resolved_location": rl}
        assert resolve_playlist(md, Config()) == "Ireland Drives"
        assert "Ireland road trip" in _region_tags(label, resolved=rl)

    def test_road_populated_from_winning_county(self):
        """``build_resolved_location`` carries ``dom.road`` (the winning
        county's road) so callers need not re-geocode to recover it."""
        rl = build_resolved_location([_FR] * 8 + [_ED] * 2, geocoder=_geo_sc())
        assert rl.road == "Sea Island Parkway"

    def test_road_vote_tally_carried_onto_resolved_location(self):
        """The winning county's full road-vote tally is threaded onto the
        persisted ResolvedLocation so the provenance sidecar can record it."""
        rl = build_resolved_location([_FR] * 8 + [_ED] * 2, geocoder=_geo_sc())
        assert rl.road_votes.get("Sea Island Parkway")

    def test_tuple_null_island_does_not_pollute_vote(self):
        """A (t,0,0) tuple in the trace must not be geocoded."""
        pts = [(float(i), _FRIPP[0], _FRIPP[1]) for i in range(6)] + [(99.0, 0.0, 0.0)]
        assert build_resolved_location(pts, geocoder=_geo_sc()).dominant_county == "Beaufort County"

    def test_quorum_floor_degrades_when_too_few_samples(self):
        """Below GEOCODE_QUORUM_MIN successful samples → no dominant county."""
        assert GEOCODE_QUORUM_MIN >= 2
        rl = build_resolved_location([_FR], geocoder=_geo_sc())  # one sample only
        assert rl.dominant_county is None and rl.dominant_state is None

    def test_quorum_gates_on_sample_count_not_weighted_mass(self):
        """The quorum is a COUNT of geocoded samples, never the footage-
        weighted vote MASS. A long trace with exactly ONE sample resolving to
        the winning county accrues a catchment-span mass >= GEOCODE_QUORUM_MIN
        while the true geocoded-sample count is 1 (< the floor). It must
        DEGRADE the WHOLE evidence surface: road / ordered_place_sequence /
        regions / vote are emptied along with dominant_*, so a 1-sample flake
        cannot corroborate wrong titles or hard-fail correct ones downstream.
        Trace-derived endpoints stay (no geocode)."""
        assert GEOCODE_QUORUM_MIN >= 2
        pts = [dict(_BLANK) for _ in range(45)]
        pts[22] = dict(_FR)  # a single evenly-sampled index resolves to Beaufort
        rl = build_resolved_location(pts, geocoder=_geo_sc())
        # Degrade: one flaky geocode must not stamp the dominant county/state.
        assert rl.dominant_county is None
        assert rl.dominant_state is None
        assert rl.is_us is False and rl.country_code is None
        # Degrade contract: NO sub-floor evidence survives the gate.
        assert rl.road is None
        assert rl.ordered_place_sequence == []
        assert rl.regions == []
        assert rl.vote == {}
        # Endpoints derive from the raw trace, not the geocoder — kept.
        assert rl.endpoints

    def test_quorum_degraded_model_skips_location_invariant_gate(self):
        """The ``location_invariant`` gate must SKIP a quorum-degraded model, never
        reconcile against its 1-sample region set. Otherwise a correct
        quorum-derived title ("Colleton County") would hard-fail against
        regions=["Beaufort County, ..."] (the single flake) post-encode."""
        pts = [dict(_BLANK) for _ in range(45)]
        pts[22] = dict(_FR)  # single flaky geocode → Beaufort
        rl = build_resolved_location(pts, geocoder=_geo_sc())
        # Serialized exactly as phase_bc_day_first / phase_d_metadata thread it.
        gate = check_location_invariant(
            {
                "title": "Colleton County, South Carolina",
                "location": "Colleton County, South Carolina",
                "resolved_location": {
                    "dominant_county": rl.dominant_county,
                    "dominant_state": rl.dominant_state,
                    "regions": list(rl.regions),
                    "vote": dict(rl.vote),
                    "ordered_place_sequence": list(rl.ordered_place_sequence),
                    "destination": rl.destination,
                },
            }
        )
        assert gate.passed is True
        assert "skipped" in gate.message

    def test_quorum_met_gate_still_reconciles(self):
        """A quorum-MET model keeps the full evidence
        surface and the ``location_invariant`` gate still catches the genuine far-road class
        (title county in zero sampled regions)."""
        rl = build_resolved_location([_FR] * 8 + [_ED] * 2, geocoder=_geo_sc())
        assert rl.road == "Sea Island Parkway"
        assert rl.regions and rl.vote and rl.ordered_place_sequence
        meta = {
            "title": "Aiken County, South Carolina",  # never sampled → far-road
            "location": "Aiken County, South Carolina",
            "resolved_location": {
                "dominant_county": rl.dominant_county,
                "dominant_state": rl.dominant_state,
                "regions": list(rl.regions),
                "vote": dict(rl.vote),
                "ordered_place_sequence": list(rl.ordered_place_sequence),
                "destination": rl.destination,
            },
        }
        assert check_location_invariant(meta).passed is False

    def test_empty_trace_and_no_geocoder_return_empty_resolved(self):
        assert build_resolved_location([], geocoder=_geo_sc()).dominant_county is None
        rl = build_resolved_location([_FR] * 4, geocoder=None)
        assert isinstance(rl, ResolvedLocation) and rl.regions == []


# ── a stateless winning county (state is None) ─────────────────────────
# Irish counties and US independent cities (Washington DC) reverse-geocode to a
# county with NO state/province. The vote key must serialize the None state to
# an EMPTY slot so the region label is the bare county ("County Kerry"), never
# the malformed "County Kerry, None" that false-fails the location invariant
# (title_label is built from the real Python-None state → "County Kerry", which
# is then NOT a member of a ["County Kerry, None"] region set → HARD FAIL).
_KERRY_STATELESS = {"county": "County Kerry", "state": None, "town": "Dingle", "country": "Ireland"}
_DC = (38.9072, -77.0369)  # Washington, DC — an independent city, no state
_WASHINGTON = {"county": "Washington", "state": None, "country": "United States"}


def _invariant_meta(rl):
    return {
        "resolved_location": {
            "dominant_county": rl.dominant_county,
            "dominant_state": rl.dominant_state,
            "regions": rl.regions,
            "vote": rl.vote,
        }
    }


class TestStatelessCountyRegionLabel:
    def test_irish_stateless_county_labels_bare_county_and_passes_invariant(self):
        rl = build_resolved_location(
            [_KE] * 8, geocoder=_make_geocoder({_DINGLE: _KERRY_STATELESS})
        )
        assert rl.dominant_county == "County Kerry"
        assert rl.dominant_state is None
        assert "County Kerry" in rl.regions
        assert "County Kerry, None" not in rl.regions
        assert check_location_invariant(_invariant_meta(rl)).passed is True

    def test_independent_city_stateless_labels_bare_name_and_passes_invariant(self):
        dc = {"lat": _DC[0], "lon": _DC[1]}
        rl = build_resolved_location([dc] * 8, geocoder=_make_geocoder({_DC: _WASHINGTON}))
        assert rl.dominant_county == "Washington"
        assert rl.dominant_state is None
        assert rl.regions == ["Washington"]
        assert check_location_invariant(_invariant_meta(rl)).passed is True

    def test_stateless_irish_county_still_carries_country_code(self):
        """A stateless (no province-as-state) Irish county must STILL carry
        its geocoded country even when no state is present, since ``country`` is known.
        ``is_us`` stays False (no US state), so a non-US country never flips
        the units."""
        rl = build_resolved_location(
            [_KE] * 8, geocoder=_make_geocoder({_DINGLE: _KERRY_STATELESS})
        )
        assert rl.dominant_state is None
        assert rl.is_us is False
        assert rl.country_code == "Ireland"


# ── resume centroid-fallback must VOTE, not bare-mean ──────────────────
# A county line at lat 32.5: below = Beaufort, above = Colleton. An LF whose
# segment centroids are mostly in Beaufort but whose ARITHMETIC MEAN drifts
# across the line into Colleton is the cross-county-line far-road bug
# on the resume path: on resume ``gps_points_by_lf`` is empty (no in-memory
# trace), so ``derive_per_lf_locations`` must not geocode the bare-mean
# segment centroid (it would stamp the wrong county); it votes the modal county
# across the per-segment centroids instead.
def _geo_county_line():
    class _Geo:
        def _county(self, lat):
            return (
                ("Colleton County", "South Carolina")
                if lat >= 32.5
                else ("Beaufort County", "South Carolina")
            )

        def reverse_geocode_detailed(self, lat, lon):
            county, state = self._county(lat)
            return {"county": county, "state": state, "town": county.split()[0]}

        def reverse_geocode(self, lat, lon):
            county, state = self._county(lat)
            return SimpleNamespace(city=None, name=None, county=county, state=state)

    return _Geo()


def _seg(lat, lon):
    return SimpleNamespace(gps_center_lat=lat, gps_center_lon=lon)


def _lf(idx, *segs):
    return SimpleNamespace(index=idx, segments=list(segs))


class TestResumeCentroidFallbackVotes:
    # Three Beaufort segments (lat 32.45) + two Colleton (lat 32.70).
    # Plurality county = Beaufort (3 vs 2). Arithmetic mean lat =
    # (3*32.45 + 2*32.70)/5 = 32.55 → > 32.5 → geocodes to Colleton (WRONG).
    _CROSS_LINE_LF = None

    def _cross_line_lf(self):
        return _lf(
            1,
            _seg(32.45, -80.60),
            _seg(32.45, -80.61),
            _seg(32.45, -80.62),
            _seg(32.70, -80.66),
            _seg(32.70, -80.67),
        )

    def test_bare_mean_would_pick_wrong_county(self):
        """Guard the fixture: a bare-mean centroid would geocode to Colleton."""
        from gomoveshift.pipeline.gps_enrichment import _centroid_for_segments

        lat, _lon = _centroid_for_segments(self._cross_line_lf().segments)
        assert lat >= 32.5  # mean is on the WRONG (Colleton) side of the line

    def test_resume_fallback_votes_plurality_county(self):
        """No in-memory ``gps_points_by_lf`` (resume) → vote over segment
        centroids, landing on the plurality county (Beaufort), NOT the
        bare-mean centroid (Colleton)."""
        labels = derive_per_lf_locations(
            [self._cross_line_lf()],
            geocoder=_geo_county_line(),
            gps_points_by_lf=None,  # resume: state round-tripped, no in-memory trace
        )
        assert labels[1] == "Beaufort County, South Carolina"


# ── per-segment head/tail GPS loss must warn + withhold the A→B endpoint ─
# The journey's ``start`` is supplied by the FIRST segment yielding in-range
# trackpoints, its ``end`` by the LAST. When that segment drops its head (or
# tail) waypoints, ``get_points_in_range`` returns a trace clipped inward, so
# ``build_resolved_location`` derives a SHIFTED origin/destination from
# ``coords[0]`` / ``coords[-1]`` and burns it as if authoritative — silently.
# The detector must WARN naming the endpoint and the builder must WITHHOLD it.
def _moving_gps(timestamps):
    """A real MOVING trace (varying coords ⇒ never a stale-lock, so
    ``effective_offset`` leaves the window at the stored offset) sampled at the
    given timestamps, starting near Fripp and drifting north."""
    tps = [
        GPSTrackpoint(32.30 + 0.001 * i, -80.63, None, float(ts)) for i, ts in enumerate(timestamps)
    ]
    return GPSData(trackpoints=tps)


def _one_seg_plan(gps_data, *, start_time=0.0, end_time=100.0):
    """A one-LF, one-segment drive_plan + ``gps_per_episode`` whose segment
    requests the window ``[start_time, end_time]`` from episode 0."""
    seg = SimpleNamespace(source_file_index=0, start_time=start_time, end_time=end_time)
    lf = SimpleNamespace(index=1, segments=[seg])
    return SimpleNamespace(regular_lfs=[lf]), [(gps_data, 0.0)]


class TestPerSegmentEndpointLoss:
    def test_head_loss_warns_and_flags_start(self, caplog):
        # Head waypoints (t=0..59) dropped: the trace opens 60s into the window.
        plan, gpe = _one_seg_plan(_moving_gps(range(60, 101)))
        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.gps_enrichment"):
            loss = _journey_endpoint_loss_by_lf(plan, gpe)
        assert loss.get(1) == {"start"}
        assert any(
            r.levelno == logging.WARNING and "start" in r.getMessage().lower()
            for r in caplog.records
        )

    def test_tail_loss_warns_and_flags_end(self, caplog):
        # Tail waypoints (t=41..100) dropped: the trace ends 60s before the edge.
        plan, gpe = _one_seg_plan(_moving_gps(range(0, 41)))
        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.gps_enrichment"):
            loss = _journey_endpoint_loss_by_lf(plan, gpe)
        assert loss.get(1) == {"end"}
        assert any(
            r.levelno == logging.WARNING and "end" in r.getMessage().lower() for r in caplog.records
        )

    def test_fully_covered_segment_has_no_loss_and_no_warn(self, caplog):
        plan, gpe = _one_seg_plan(_moving_gps(range(0, 101)))
        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.gps_enrichment"):
            loss = _journey_endpoint_loss_by_lf(plan, gpe)
        assert loss == {}
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    def test_builder_withholds_flagged_start_endpoint(self):
        # A shifted ``start`` must NOT be emitted; ``end`` (uncorrupted) stays.
        rl = build_resolved_location([_FR] * 8, geocoder=None, withhold_endpoints={"start"})
        assert "start" not in rl.endpoints
        assert rl.endpoints["end"][0] == _FRIPP[0]

    def test_builder_withholds_both_when_flagged(self):
        rl = build_resolved_location([_FR] * 8, geocoder=None, withhold_endpoints={"start", "end"})
        assert rl.endpoints == {}

    def test_builder_default_leaves_both_endpoints_unchanged(self):
        # Regression: the fully-covered path (no withhold) is exactly as before.
        rl = build_resolved_location([_FR] * 8, geocoder=None)
        assert rl.endpoints["start"][0] == _FRIPP[0]
        assert rl.endpoints["end"][0] == _FRIPP[0]
