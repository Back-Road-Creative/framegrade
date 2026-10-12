"""Silent-degradation regression tests for chapters + geocoding sources.

Each test forces a fallback path in the chapter generator or geocoding
helpers and asserts that (a) the fallback behavior ran and (b) a WARNING
was emitted naming both the preferred path and the fallback, so an
operator watching the console can see the degradation.

These guard against the "silent fallback" class of bug surfaced in the
silent-degradation review.

Findings covered (log-only fixes):
  * NominatimChapterSource: cache.get() exception after session budget
    exhaustion now emits a WARNING instead of silently returning None.
  * NominatimChapterSource.geocode_fallback: per-waypoint geocode
    exceptions now log waypoint coordinates in the WARNING.
  * OverpassChapterSource.discover_area_pois: corrupt/unreadable
    cache files now emit a WARNING before falling back to a fresh
    Overpass API query.
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest


NOMINATIM_LOGGER = "gomoveshift.pipeline.chapters.nominatim_source"
OVERPASS_LOGGER = "gomoveshift.pipeline.chapters.overpass_source"


# ---------------------------------------------------------------------------
# NominatimChapterSource.reverse_geocode — cache lookup failure after budget
# ---------------------------------------------------------------------------


class TestReverseGeocodeCacheFallbackWarns:
    """Cache lookup failure after budget exhaustion must WARN."""

    def test_cache_exception_returns_none_and_warns(self, caplog):
        from gomoveshift.pipeline.chapters.nominatim_source import (
            NominatimChapterSource,
        )

        geocoder = MagicMock()
        cache = MagicMock()
        cache.get.side_effect = RuntimeError("cache read corrupt")

        source = NominatimChapterSource(geocoder=geocoder, cache=cache)
        # Force budget exhaustion.
        source._session_request_count = NominatimChapterSource.MAX_SESSION_REQUESTS

        with caplog.at_level("WARNING", logger=NOMINATIM_LOGGER):
            result = source.reverse_geocode(10.0, 20.0)

        # Fallback ran — returned None because cache raised.
        assert result is None
        # Cache was consulted (preferred path).
        cache.get.assert_called_once_with(10.0, 20.0)
        # Underlying geocoder was NOT called (budget exhausted).
        geocoder.reverse_geocode.assert_not_called()

        messages = [r.getMessage() for r in caplog.records if r.name == NOMINATIM_LOGGER]
        # First WARNING: budget exhausted (existing, kept for context).
        assert any("budget exhausted" in m for m in messages), messages
        # New WARNING: cache lookup failure.
        assert any("cache lookup failed" in m and "cache read corrupt" in m for m in messages), (
            f"expected cache-fallback warning, got: {messages!r}"
        )


# ---------------------------------------------------------------------------
# NominatimChapterSource.geocode_fallback — per-waypoint exception
# ---------------------------------------------------------------------------


class TestGeocodeFallbackWaypointExceptionWarns:
    """Per-waypoint geocode exceptions must WARN with coordinates."""

    def test_geocode_exception_skips_waypoint_and_warns(self, caplog):
        from gomoveshift.pipeline.chapters.nominatim_source import (
            NominatimChapterSource,
        )

        # Build a minimal journey with waypoints lacking .location so
        # geocode_fallback is forced to call reverse_geocode.
        wp = MagicMock()
        wp.latitude = 37.5
        wp.longitude = -122.5
        wp.location = None
        # Real timestamp: the fallback maps each sample by the waypoint's actual
        # time, so mocks must carry a datetime like a production Waypoint does.
        wp.timestamp = datetime(2025, 1, 1, 0, 0, 0)

        journey = MagicMock()
        journey.waypoints = [wp] * 60  # enough for sampling

        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = RuntimeError("nominatim 503")

        source = NominatimChapterSource(geocoder=geocoder, cache=None)

        scale = lambda g, gd, od: g  # noqa: E731
        map_edit = lambda t: t  # noqa: E731

        with caplog.at_level("WARNING", logger=NOMINATIM_LOGGER):
            results = source.geocode_fallback(
                journey,
                gps_duration=60.0,
                original_duration=60.0,
                scale_gps_to_video=scale,
                map_to_edited_time=map_edit,
                is_valid_placename_fn=lambda n: True,
            )

        # Fallback ran — exception was swallowed, no results produced.
        assert results == []

        messages = [r.getMessage() for r in caplog.records if r.name == NOMINATIM_LOGGER]
        # Must mention waypoint coordinates (not just generic "Geocoding failed").
        assert any(
            "Geocoding failed" in m and f"{wp.latitude:.5f}" in m and f"{wp.longitude:.5f}" in m
            for m in messages
        ), f"expected waypoint-coord warning, got: {messages!r}"


# ---------------------------------------------------------------------------
# OverpassChapterSource.discover_area_pois — corrupt cache file
# ---------------------------------------------------------------------------


class TestOverpassCacheLoadFailureWarns:
    """Corrupt Overpass cache must WARN before falling back to fresh query."""

    def test_corrupt_cache_warns_and_requeries(self, tmp_path, monkeypatch, caplog):
        from gomoveshift.pipeline import chapters as chapters_pkg
        from gomoveshift.pipeline.chapters.overpass_source import (
            OverpassChapterSource,
            _bbox_from_waypoints,
            _cache_path_for_bbox,
        )

        # Build a tiny journey.
        wp = MagicMock()
        wp.latitude = 40.0
        wp.longitude = -105.0
        wp.timestamp = MagicMock()
        journey = MagicMock()
        journey.waypoints = [wp]

        # Pre-populate the bbox-derived cache file with invalid JSON so
        # json.load raises and triggers the fallback warning.
        bbox = _bbox_from_waypoints(journey.waypoints)
        cache_file = _cache_path_for_bbox(tmp_path, bbox)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("{ not valid json")

        # Capture whether the fresh API path was invoked. Return minimal
        # valid data so discover_area_pois returns cleanly.
        call_log: list[bool] = []

        def fake_query(bbox_arg, named_road_patterns_arg):
            call_log.append(True)
            return {"elements": []}

        monkeypatch.setattr(chapters_pkg, "_query_overpass", fake_query)

        source = OverpassChapterSource()

        with caplog.at_level("WARNING", logger=OVERPASS_LOGGER):
            matched, route_pois, named_routes = source.discover_area_pois(
                journey=journey,
                gps_duration=60.0,
                original_duration=60.0,
                scale_gps_to_video=lambda g, gd, od: g,
                map_to_edited_time=lambda t: t,
                stages_dir=tmp_path,
            )

        # Fallback ran — fresh Overpass query was made after the cache
        # load failed.
        assert call_log == [True], "expected fallback to fresh Overpass query"
        # Empty elements -> empty results, but no crash.
        assert matched == []
        assert route_pois == []
        assert named_routes == []

        messages = [r.getMessage() for r in caplog.records if r.name == OVERPASS_LOGGER]
        assert any(
            "Failed to load Overpass cache" in m and "fresh Overpass API query" in m
            for m in messages
        ), f"expected cache-load-failure warning, got: {messages!r}"


# ---------------------------------------------------------------------------
# Sanity: the existing budget-exhausted warning still fires when cache is
# None (no cache configured).
# ---------------------------------------------------------------------------


class TestBudgetExhaustedServesWarmDetailedCache:
    """After the session budget is spent, both budgeted reverse-geocode paths
    must serve a warm entry from the SAME cache namespace the roads geocoder
    writes (`detailed`) instead of silently degrading to None / {}.

    `reverse_geocode` serves via a namespace-aware `GeocodingCache.get()` that
    reconstructs a Location from the warm detailed entry; `reverse_geocode_detailed`
    serves the raw dict via `get_detailed()`. Neither may hit the network once
    the budget is exhausted.
    """

    def _warm_source(self, tmp_path):
        from gomoveshift.pipeline.chapters.nominatim_source import (
            NominatimChapterSource,
        )
        from gps_journey.geocoding.cache import GeocodingCache

        cache = GeocodingCache(cache_path=tmp_path / "cache.json", ttl_days=90)
        lat, lon = 55.0, -8.3
        cache.set_detailed(
            lat,
            lon,
            {
                "road": "N56",
                "village": "Rossbeg",
                "town": "Dunfanaghy",
                "hamlet": "Maghera",
                "locality": "Ardbane",
                "county": "County Donegal",
                "state": "Ulster",
                "country": "Ireland",
                "full_address": "N56, Rossbeg, County Donegal, Ireland",
            },
        )
        geocoder = MagicMock()
        source = NominatimChapterSource(geocoder=geocoder, cache=cache)
        # Force budget exhaustion.
        source._session_request_count = NominatimChapterSource.MAX_SESSION_REQUESTS
        return source, geocoder, lat, lon

    def test_reverse_geocode_serves_location_from_warm_detailed(self, tmp_path):
        source, geocoder, lat, lon = self._warm_source(tmp_path)

        loc = source.reverse_geocode(lat, lon)

        assert loc is not None, "warm detailed cache must yield a Location, not None"
        assert loc.name == "Rossbeg"
        assert loc.town == "Dunfanaghy"
        assert loc.county == "County Donegal"
        assert loc.country == "Ireland"
        # Budget exhausted — underlying geocoder must NOT be hit.
        geocoder.reverse_geocode.assert_not_called()

    def test_reverse_geocode_detailed_serves_warm_dict(self, tmp_path):
        source, geocoder, lat, lon = self._warm_source(tmp_path)

        detailed = source.reverse_geocode_detailed(lat, lon)

        assert detailed, "warm detailed cache must yield the dict, not {}"
        assert detailed["village"] == "Rossbeg"
        assert detailed["road"] == "N56"
        # Budget exhausted — no API hit, budget preserved.
        geocoder.reverse_geocode_detailed.assert_not_called()


class TestReverseGeocodeBudgetExhaustedStillWarns:
    def test_no_cache_budget_exhausted_warns(self, caplog):
        from gomoveshift.pipeline.chapters.nominatim_source import (
            NominatimChapterSource,
        )

        geocoder = MagicMock()
        source = NominatimChapterSource(geocoder=geocoder, cache=None)
        source._session_request_count = NominatimChapterSource.MAX_SESSION_REQUESTS

        with caplog.at_level("WARNING", logger=NOMINATIM_LOGGER):
            result = source.reverse_geocode(1.0, 2.0)

        assert result is None
        geocoder.reverse_geocode.assert_not_called()

        messages = [r.getMessage() for r in caplog.records if r.name == NOMINATIM_LOGGER]
        assert any("budget exhausted" in m for m in messages), messages
