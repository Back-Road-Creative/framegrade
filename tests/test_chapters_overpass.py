"""Tests for Overpass-based POI discovery in chapters.py."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.chapters import (
    ChapterGenerator,
    NominatimChapterSource,
    OverpassChapterSource,
    _bbox_from_waypoints,
    _haversine,
    _nominatim_search,
    _query_overpass,
)


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_waypoint(lat, lon, ts):
    """Create a mock waypoint."""
    wp = MagicMock()
    wp.latitude = lat
    wp.longitude = lon
    wp.timestamp = ts
    wp.location = None
    return wp


def _make_journey(waypoints):
    """Create a mock Journey from waypoints."""
    j = MagicMock()
    j.waypoints = tuple(waypoints)
    j.start_date = waypoints[0].timestamp
    j.end_date = waypoints[-1].timestamp
    j.statistics = MagicMock()
    return j


OVERPASS_RESPONSE = {
    "elements": [
        {
            "type": "node",
            "id": 1,
            "lat": 51.88,
            "lon": -9.90,
            "tags": {"name": "Kenmare", "place": "town"},
        },
        {
            "type": "node",
            "id": 2,
            "lat": 51.82,
            "lon": -10.02,
            "tags": {"name": "Sneem", "place": "village"},
        },
        {
            "type": "node",
            "id": 3,
            "lat": 51.77,
            "lon": -10.10,
            "tags": {"name": "Moll's Gap", "tourism": "viewpoint"},
        },
        {
            "type": "way",
            "id": 100,
            "center": {"lat": 51.85, "lon": -9.95},
            "geometry": [{"lat": 51.85, "lon": -9.93}, {"lat": 51.84, "lon": -9.94}],
            "tags": {"name": "Ring of Kerry", "highway": "primary"},
        },
        {
            "type": "way",
            "id": 101,
            "center": {"lat": 51.80, "lon": -10.05},
            "geometry": [{"lat": 51.80, "lon": -9.98}, {"lat": 51.79, "lon": -9.99}],
            "tags": {"name": "Wild Atlantic Way", "highway": "secondary"},
        },
    ]
}


@pytest.fixture
def config():
    from gomoveshift.config import Config

    return Config()


@pytest.fixture
def mock_journey():
    """Kerry-like GPS journey along the route."""
    start = datetime(2025, 9, 10, 8, 0, 0)
    wps = []
    # ~60 min drive with waypoints every 30s
    for i in range(120):
        ts = start + timedelta(seconds=i * 30)
        lat = 51.88 - i * 0.001  # Moving south
        lon = -9.90 - i * 0.001  # Moving west
        wps.append(_make_waypoint(lat, lon, ts))
    return _make_journey(wps)


# ── Tests ─────────────────────────────────────────────────────────────


class TestHaversine:
    def test_same_point_is_zero(self):
        assert _haversine(51.88, -9.90, 51.88, -9.90) == 0.0

    def test_known_distance(self):
        # Dublin to Cork ≈ 220 km
        d = _haversine(53.35, -6.26, 51.90, -8.47)
        assert 200_000 < d < 250_000

    def test_nearby_points(self):
        # ~111m apart (0.001 degree latitude)
        d = _haversine(51.880, -9.900, 51.881, -9.900)
        assert 100 < d < 120


class TestBboxFromWaypoints:
    def test_computes_bbox_with_margin(self, mock_journey):
        bbox = _bbox_from_waypoints(mock_journey.waypoints)
        s, w, n, e = bbox
        # Check margin is applied (waypoints span ~0.12 degrees)
        assert s < 51.76  # south of southernmost point
        assert n > 51.88  # north of northernmost point


class TestOverpassPOIDiscovery:
    @patch("gomoveshift.pipeline.chapters._query_overpass")
    def test_discovers_towns_and_pois(self, mock_overpass, config, mock_journey):
        mock_overpass.return_value = OVERPASS_RESPONSE
        gen = ChapterGenerator(config)
        gen._keep_segments = None

        gps_duration = 3600.0
        pois = gen._discover_area_pois(mock_journey, gps_duration, gps_duration, proximity_m=5000.0)

        # Should find at least Kenmare and Sneem (towns near route)
        poi_names = [p["name"] for _, p in pois]
        assert len(pois) >= 2, f"Expected >= 2 POIs, got {len(pois)}: {poi_names}"

    @patch("gomoveshift.pipeline.chapters._query_overpass")
    def test_detects_named_routes(self, mock_overpass, config, mock_journey):
        mock_overpass.return_value = OVERPASS_RESPONSE
        gen = ChapterGenerator(config)
        gen._keep_segments = None

        gen._discover_area_pois(mock_journey, 3600.0, 3600.0, proximity_m=5000.0)

        assert "Ring of Kerry" in gen.named_routes or "Wild Atlantic Way" in gen.named_routes

    @patch("gomoveshift.pipeline.chapters._query_overpass")
    def test_caches_overpass_response(self, mock_overpass, config, mock_journey, tmp_path):
        mock_overpass.return_value = OVERPASS_RESPONSE
        gen = ChapterGenerator(config)
        gen._keep_segments = None

        gen._discover_area_pois(
            mock_journey, 3600.0, 3600.0, stages_dir=tmp_path, proximity_m=5000.0
        )

        # Cache file should exist
        cache_files = list(tmp_path.glob("overpass_cache_*.json"))
        assert len(cache_files) == 1

        # Second call should use cache (mock not called again)
        mock_overpass.reset_mock()
        gen._discover_area_pois(
            mock_journey, 3600.0, 3600.0, stages_dir=tmp_path, proximity_m=5000.0
        )
        mock_overpass.assert_not_called()


class TestOverpassFallback:
    @patch("gomoveshift.pipeline.chapters._query_overpass")
    def test_fallback_to_geocoding_on_overpass_failure(self, mock_overpass, config, mock_journey):
        """When Overpass returns None, _extract_towns falls back to geocoding."""
        mock_overpass.return_value = None
        gen = ChapterGenerator(config)
        gen._keep_segments = None
        gen._stages_dir = None

        # Mock the geocoder to avoid real network calls
        mock_loc = MagicMock()
        mock_loc.city = "Kenmare"
        mock_loc.locality = None
        mock_loc.name = "Kenmare"
        mock_loc.suburb = None
        mock_loc.county = "Kerry"
        mock_loc.country = "Ireland"
        mock_loc.latitude = 51.88
        mock_loc.longitude = -9.90
        gen.geocoder.reverse_geocode = MagicMock(return_value=mock_loc)

        gen._extract_towns(mock_journey, 3600.0, 3600.0)

        # Should have used geocoder as fallback
        assert gen.geocoder.reverse_geocode.called
        # And should be capped at MAX_GEOCODE_FALLBACK samples
        assert gen.geocoder.reverse_geocode.call_count <= 200


class TestOverpassTotalOutageDegradesToGeocoding:
    """Every Overpass mirror down must NOT abort chapters.

    `post_overpass_query` raises `RuntimeError` when all endpoints fail; that
    exception escaped `discover_area_pois` and killed the whole chapter pass
    ("CHAPTER-GATE: LF 1 chapter generation failed: Overpass query failed on
    every endpoint" → 0 chapters → metadata-quality HARD FAIL on a 50-min LF),
    while the designed Nominatim fallback in `_extract_towns` never ran.
    """

    _OUTAGE = RuntimeError(
        "Overpass query failed on every endpoint — "
        "https://overpass-api.de/api/interpreter: HTTP 504; "
        "https://overpass.kumi.systems/api/interpreter: The read operation timed out. "
        "Check network connectivity or Overpass API status."
    )

    @patch("gomoveshift.pipeline.chapters._query_overpass")
    def test_discover_area_pois_returns_empty_and_logs_the_fallback(
        self, mock_overpass, mock_journey, caplog
    ):
        from gomoveshift.pipeline.chapters.overpass_source import OverpassChapterSource

        mock_overpass.side_effect = self._OUTAGE
        src = OverpassChapterSource()
        with caplog.at_level("ERROR", logger="gomoveshift.pipeline.chapters.overpass_source"):
            matched, route_pois, named_routes = src.discover_area_pois(
                journey=mock_journey,
                gps_duration=3600.0,
                original_duration=3600.0,
                scale_gps_to_video=lambda t, g, o: t,
                map_to_edited_time=lambda t: t,
            )
        assert (matched, route_pois, named_routes) == ([], [], [])
        msgs = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
        assert any("Overpass" in m and "Nominatim" in m and "falling back" in m for m in msgs), msgs
        assert any("every endpoint" in m for m in msgs), msgs

    @patch("gomoveshift.pipeline.chapters._query_overpass")
    def test_extract_towns_falls_back_to_geocoding_when_every_endpoint_is_down(
        self, mock_overpass, config, mock_journey
    ):
        mock_overpass.side_effect = self._OUTAGE
        gen = ChapterGenerator(config)
        gen._keep_segments = None
        gen._stages_dir = None
        mock_loc = MagicMock()
        mock_loc.city = "Gettysburg"
        mock_loc.locality = None
        mock_loc.name = "Gettysburg"
        mock_loc.suburb = None
        mock_loc.county = "Adams County"
        mock_loc.country = "United States"
        mock_loc.latitude = 39.83
        mock_loc.longitude = -77.23
        gen.geocoder.reverse_geocode = MagicMock(return_value=mock_loc)

        towns = gen._extract_towns(mock_journey, 3600.0, 3600.0)

        assert gen.geocoder.reverse_geocode.called, "Nominatim fallback must run on an outage"
        assert towns, "an outage must still yield chapters from the geocoding fallback"


class TestChaptersRequiredWithGPS:
    """Chapters must be generated when GPS file is available and parseable."""

    @patch("gomoveshift.pipeline.chapters._query_overpass")
    def test_generates_chapters_from_overpass(self, mock_overpass, config, mock_journey):
        mock_overpass.return_value = OVERPASS_RESPONSE
        gen = ChapterGenerator(config)

        # Mock the end-location geocode to return a valid location
        mock_loc = MagicMock()
        mock_loc.city = "Sneem"
        mock_loc.locality = None
        mock_loc.name = "Sneem"
        mock_loc.suburb = None
        mock_loc.county = "Kerry"
        mock_loc.country = "Ireland"
        gen.geocoder.reverse_geocode = MagicMock(return_value=mock_loc)

        chapters = gen.generate(
            gpx_path=Path("fake.gpx"),
            video_duration=3600.0,
            start_location="Kenmare, Kerry",
            _mock_journey=mock_journey,
        )

        assert len(chapters) >= 2, "Must have at least start and finish chapters"


class TestGPSDateAutoDetection:
    @patch("gomoveshift.pipeline.chapters._query_overpass")
    def test_recording_date_from_journey(self, mock_overpass, config, mock_journey):
        mock_overpass.return_value = OVERPASS_RESPONSE
        gen = ChapterGenerator(config)

        mock_loc = MagicMock()
        mock_loc.city = "Sneem"
        mock_loc.locality = None
        mock_loc.name = "Sneem"
        mock_loc.suburb = None
        mock_loc.county = "Kerry"
        mock_loc.country = "Ireland"
        gen.geocoder.reverse_geocode = MagicMock(return_value=mock_loc)

        gen.generate(
            gpx_path=Path("fake.gpx"),
            video_duration=3600.0,
            start_location="Kenmare, Kerry",
            _mock_journey=mock_journey,
        )

        assert gen.recording_date is not None
        assert gen.recording_date.year == 2025
        assert gen.recording_date.month == 9


class TestOverpassQueryCoastalPOITypes:
    """Overpass query must include coastal POI types (beach, bay, bridge, river)."""

    def _capture_query(self, bbox=(32.0, -81.0, 33.0, -80.0)):
        """Call _query_overpass and return the query string sent to Overpass."""
        import urllib.error
        import urllib.parse

        captured_query = []

        def fake_urlopen(req, timeout=None):
            body = urllib.parse.unquote(req.data.decode())
            captured_query.append(body)
            raise urllib.error.URLError("no network in tests")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with patch("time.sleep"):  # suppress retry delays
                with pytest.raises(RuntimeError):  # all 3 retries fail → raises
                    _query_overpass(bbox)

        assert captured_query, "urlopen was not called"
        return captured_query[0]

    def test_query_includes_beach_bay_cape_island(self):
        """Query must fetch natural beach/bay/cape/island nodes."""
        query = self._capture_query()
        assert "beach|bay|cape|island" in query, (
            f"Query missing beach/bay/cape/island types:\n{query}"
        )

    def test_query_includes_bridge(self):
        """Query must fetch named bridge nodes."""
        query = self._capture_query()
        assert "bridge" in query, f"Query missing bridge type:\n{query}"

    def test_query_includes_named_waterways(self):
        """Query must fetch named river/waterway nodes."""
        query = self._capture_query()
        assert "waterway" in query or "river" in query, (
            f"Query missing waterway/river types:\n{query}"
        )


class TestOverpassRetryBehavior:
    """Overpass must retry 3 times with exponential backoff, then raise."""

    def test_raises_after_three_failures(self):
        """After 3 consecutive network errors, RuntimeError must be raised."""
        import urllib.error

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
            with patch("time.sleep"):  # don't actually sleep
                with pytest.raises(RuntimeError, match="Overpass query failed"):
                    _query_overpass((51.0, -10.0, 52.0, -9.0))

    def test_uses_exponential_backoff(self):
        """Sleep durations between retries must double each attempt."""
        import urllib.error

        sleep_calls = []
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
            with patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
                with pytest.raises(RuntimeError):
                    _query_overpass((51.0, -10.0, 52.0, -9.0))
        # Two sleeps between 3 attempts; each must be >= the previous
        assert len(sleep_calls) >= 2
        assert sleep_calls[1] >= sleep_calls[0], (
            f"Second sleep ({sleep_calls[1]}s) should be >= first ({sleep_calls[0]}s)"
        )

    def test_succeeds_on_second_attempt(self):
        """A transient failure followed by success must return data, not raise."""
        import urllib.error
        import json

        good_response = {"elements": []}
        call_count = [0]

        def flaky_urlopen(req, timeout=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise urllib.error.URLError("transient")
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.status = 200
            mock_resp.read.return_value = json.dumps(good_response).encode()
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=flaky_urlopen):
            with patch("time.sleep"):
                result = _query_overpass((51.0, -10.0, 52.0, -9.0))
        assert result == good_response
        assert call_count[0] == 2


class TestNominatimSearchErrorHandling:
    """_nominatim_search must not use bare except; parse errors must propagate."""

    def test_network_error_returns_none(self):
        """URLError (network) should be caught and return None."""
        import urllib.error

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no net")):
            with patch("time.sleep"):
                result = _nominatim_search("Kenmare, Kerry, Ireland")
        assert result is None

    def test_malformed_response_raises(self):
        """A response missing lat/lon keys must propagate — not return None silently."""
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'[{"display_name": "Kenmare"}]'  # missing lat/lon
        with patch("urllib.request.urlopen", return_value=mock_resp):
            with patch("time.sleep"):
                with pytest.raises((KeyError, ValueError)):
                    _nominatim_search("Kenmare, Kerry")

    def test_empty_results_returns_none(self):
        """Empty list from Nominatim (location not found) must return None."""
        import json

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps([]).encode()
        with patch("urllib.request.urlopen", return_value=mock_resp):
            with patch("time.sleep"):
                result = _nominatim_search("Nonexistent Place XYZ")
        assert result is None


class TestNominatimSessionBudget:
    """NominatimChapterSource must enforce a per-session request cap."""

    def test_51st_request_returns_none(self):
        """The 51st geocode request must return None (budget exhausted)."""
        mock_geocoder = MagicMock()
        mock_geocoder.reverse_geocode.return_value = MagicMock(city="FakeCity")

        source = NominatimChapterSource(geocoder=mock_geocoder)
        assert source.MAX_SESSION_REQUESTS == 50

        # Make 50 successful requests
        for i in range(50):
            result = source.reverse_geocode(52.0 + i * 0.01, -9.0)
            assert result is not None, f"Request {i + 1} should succeed"

        assert source.session_request_count == 50
        assert mock_geocoder.reverse_geocode.call_count == 50

        # 51st request: budget exhausted, returns None
        result = source.reverse_geocode(53.0, -8.0)
        assert result is None, "51st request should return None (budget exhausted)"
        # Underlying geocoder must NOT have been called for the 51st
        assert mock_geocoder.reverse_geocode.call_count == 50

    def test_budget_exhausted_falls_back_to_cache(self):
        """When budget is exhausted and cache is available, use cache."""
        mock_geocoder = MagicMock()
        mock_geocoder.reverse_geocode.return_value = MagicMock(city="Cached")

        mock_cache = MagicMock()
        cached_loc = MagicMock(city="CachedCity")
        mock_cache.get.return_value = cached_loc

        source = NominatimChapterSource(geocoder=mock_geocoder, cache=mock_cache)
        # Exhaust budget
        source._session_request_count = 50

        result = source.reverse_geocode(52.0, -9.0)
        assert result == cached_loc
        mock_cache.get.assert_called_once_with(52.0, -9.0)

    def test_detailed_geocode_returns_empty_when_exhausted(self):
        """reverse_geocode_detailed must return {} when budget exhausted."""
        mock_geocoder = MagicMock()
        source = NominatimChapterSource(geocoder=mock_geocoder)
        source._session_request_count = 50

        result = source.reverse_geocode_detailed(52.0, -9.0)
        assert result == {}
        mock_geocoder.reverse_geocode_detailed.assert_not_called()


class TestOverpassQueryReturnsGeometry:
    """The query must request real geometry (``out geom``),
    not element centers (``out center``). Under ``out center``, way/relation
    elements carry only a zero-area center point (no ``bounds``/``geometry``),
    so the destination resolver degenerates every area POI to a point (never
    encloses footage) and route detection has no vertices to measure against.
    """

    def _capture_query(self, bbox=(32.0, -81.0, 33.0, -80.0)):
        import urllib.error
        import urllib.parse

        captured = []

        def fake_urlopen(req, timeout=None):
            captured.append(urllib.parse.unquote(req.data.decode()))
            raise urllib.error.URLError("no network in tests")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with patch("time.sleep"):
                with pytest.raises(RuntimeError):
                    _query_overpass(bbox)
        assert captured, "urlopen was not called"
        return captured[0]

    def test_query_requests_geometry_not_center(self):
        query = self._capture_query()
        assert "out geom" in query, f"query must request geometry:\n{query}"
        assert "out center" not in query, (
            f"`out center` returns no bounds/geometry — area POIs degenerate to points:\n{query}"
        )


class TestClassifyElementOutGeom:
    """An ``out geom`` way/relation carries ``bounds`` (plus
    geometry/members) but NO ``center``. classify_element must derive the
    element point from the bounds center instead of dropping the element for
    lack of a ``center`` (which silently discarded every area POI)."""

    def test_way_with_bounds_no_center_is_classified(self):
        elem = {
            "type": "way",
            "tags": {"name": "Killarney National Park", "leisure": "park"},
            "bounds": {"minlat": 51.9, "minlon": -9.6, "maxlat": 52.1, "maxlon": -9.4},
            "geometry": [{"lat": 52.0, "lon": -9.5}],
        }
        wps = [(0.0, 52.0, -9.5)]  # at the bounds center
        result = OverpassChapterSource.classify_element(elem, wps, 5000.0, [], set())
        assert result is not None, "out-geom way dropped for missing center"
        _t, elat, elon, poi_type, _d = result
        assert poi_type == "park"
        assert abs(elat - 52.0) < 1e-6 and abs(elon + 9.5) < 1e-6

    def test_relation_with_members_no_center_is_classified(self):
        elem = {
            "type": "relation",
            "tags": {"name": "Killarney National Park", "boundary": "national_park"},
            "bounds": {"minlat": 51.9, "minlon": -9.6, "maxlat": 52.1, "maxlon": -9.4},
            "members": [{"type": "way", "role": "outer", "geometry": [{"lat": 52.0, "lon": -9.5}]}],
        }
        wps = [(0.0, 52.0, -9.5)]
        result = OverpassChapterSource.classify_element(elem, wps, 5000.0, [], set())
        assert result is not None
        assert result[3] == "park"


class TestRouteGeometryDetection:
    """A named route is detected only where its real geometry passes within
    the sane radius of a waypoint — not from an over-approximating centroid."""

    def test_route_detected_when_geometry_passes_waypoint(self):
        elem = {
            "type": "way",
            "tags": {"name": "Ring of Kerry", "highway": "primary"},
            "bounds": {"minlat": 51.9, "minlon": -10.0, "maxlat": 52.1, "maxlon": -9.0},
            "geometry": [{"lat": 52.0, "lon": -9.5}, {"lat": 52.02, "lon": -9.52}],
        }
        wps = [(0.0, 52.0, -9.5)]  # a route vertex sits on the waypoint
        detected: set[str] = set()
        OverpassChapterSource.classify_element(elem, wps, 5000.0, ["Ring of Kerry"], detected)
        assert "Ring of Kerry" in detected

    def test_route_not_detected_when_geometry_far_though_centroid_near(self):
        # Vertices ~34 km either side of the waypoint; their midpoint (the naive
        # centroid, and the bounds center) sits on the waypoint. A ``center`` is
        # placed at that midpoint to prove geometry — not the centroid — governs.
        elem = {
            "type": "way",
            "tags": {"name": "Ring of Kerry", "highway": "primary"},
            "center": {"lat": 52.0, "lon": -9.5},
            "bounds": {"minlat": 52.0, "minlon": -10.0, "maxlat": 52.0, "maxlon": -9.0},
            "geometry": [{"lat": 52.0, "lon": -10.0}, {"lat": 52.0, "lon": -9.0}],
        }
        wps = [(0.0, 52.0, -9.5)]  # on the centroid, ~34 km from each vertex
        detected: set[str] = set()
        OverpassChapterSource.classify_element(elem, wps, 5000.0, ["Ring of Kerry"], detected)
        assert "Ring of Kerry" not in detected


class TestOverpassUserAgent:
    """Overpass API rejects Python-urllib's default User-Agent with
    HTTP 406, so every chapter query fails without one. The
    request must include a descriptive User-Agent identifying the
    project and a contact.

    Regression: without a UA header, Overpass returns 406 →
    ``_query_overpass`` raises RuntimeError → every ChapterGenerator
    call silently produces 0 chapters → every LF description has an
    orphan empty ``📍 CHAPTERS`` block and the SEO manager falls back
    to generic titles instead of using Overpass-verified town names.
    """

    def test_overpass_request_includes_user_agent(self):
        """``_query_overpass`` must set a ``User-Agent`` header on the
        urllib Request object. urlopen is mocked so no network is
        touched."""
        from unittest.mock import MagicMock, patch

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"version":0.6,"elements":[]}'
        # urlopen returns a context manager
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_response)
        mock_cm.__exit__ = MagicMock(return_value=False)

        captured_request = {}

        def _capture(req, timeout=None):
            # Copy headers before returning — urlopen doesn't expose
            # req later in our code path.
            captured_request["headers"] = dict(req.headers)
            captured_request["data"] = req.data
            return mock_cm

        with patch("urllib.request.urlopen", side_effect=_capture):
            _query_overpass(bbox=(32.0, -81.0, 32.5, -80.5))

        # Normalize headers — urllib lowercases the first letter of each
        # header name on the Request object's .headers dict.
        headers_ci = {k.lower(): v for k, v in captured_request["headers"].items()}
        assert "user-agent" in headers_ci, (
            f"Overpass request missing User-Agent header; headers were "
            f"{list(captured_request['headers'].keys())}. Default "
            f"Python-urllib UA causes HTTP 406."
        )
        ua = headers_ci["user-agent"]
        assert "gomoveshift" in ua.lower() or "video" in ua.lower(), (
            f"User-Agent should identify the caller; got {ua!r}"
        )
        assert "python-urllib" not in ua.lower(), (
            f"User-Agent must not be Python-urllib default (triggers HTTP 406); got {ua!r}"
        )

    def test_overpass_user_agent_constant_format(self):
        """Constant-level check that ``OVERPASS_USER_AGENT`` identifies
        the project and includes a contact/URL per OSM usage policy."""
        from gomoveshift.pipeline.chapters.overpass_source import (
            OVERPASS_USER_AGENT,
        )

        assert "gomoveshift" in OVERPASS_USER_AGENT.lower(), (
            f"UA must identify the project; got {OVERPASS_USER_AGENT!r}"
        )
        # OSM policy requests a contact URL or email. Accept either.
        assert "http" in OVERPASS_USER_AGENT.lower() or "@" in OVERPASS_USER_AGENT, (
            f"UA should include a contact (URL or email) per OSM usage "
            f"policy; got {OVERPASS_USER_AGENT!r}"
        )

    def test_overpass_user_agent_contact_env_override(self, monkeypatch):
        """``GMS_OVERPASS_CONTACT`` appends an operator contact; unset keeps
        the identifying default."""
        from gomoveshift.pipeline.chapters.overpass_source import (
            OVERPASS_USER_AGENT,
            overpass_user_agent,
        )

        monkeypatch.delenv("GMS_OVERPASS_CONTACT", raising=False)
        assert overpass_user_agent() == OVERPASS_USER_AGENT

        monkeypatch.setenv("GMS_OVERPASS_CONTACT", "ops@example.invalid")
        ua = overpass_user_agent()
        assert ua.startswith(OVERPASS_USER_AGENT[:-1])
        assert "ops@example.invalid" in ua
        assert ua.endswith(")")
