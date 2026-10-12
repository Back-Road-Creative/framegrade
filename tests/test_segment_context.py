"""Tests for GPS-derived segment context — terrain, time-of-day, POI lookup."""

import pytest

from gomoveshift.pipeline.segment_context import (
    SegmentContext,
    classify_terrain,
    classify_time_of_day,
    find_nearest_poi,
)


class TestTerrainClassification:
    """Terrain classification from GPS elevation and turn metrics."""

    def test_mountain_high_gain_high_elevation(self):
        """Mountain needs gain rate + absolute height + a real net range."""
        assert classify_terrain(25.0, 350.0, net_elevation_range_m=400.0) == "mountain"

    def test_mountain_needs_net_range_floor(self):
        """A high flat plateau with only summed GPS jitter is NOT a mountain.

        get_elevation_gain_rate sums positive deltas, so noise above 200 m can
        clear the gain cut; without a net-range floor it shipped '#MountainPass'
        on flat drives.
        """
        assert classify_terrain(25.0, 300.0, net_elevation_range_m=15.0) == "countryside"

    def test_winding_high_turn_rate(self):
        assert classify_terrain(5.0, 100.0, turn_rate=50.0) == "winding"

    def test_countryside_default(self):
        assert classify_terrain(8.0, 100.0, turn_rate=10.0) == "countryside"

    def test_mountain_takes_priority_over_winding(self):
        """High elevation + high gain + net range = mountain even with turns."""
        assert (
            classify_terrain(25.0, 300.0, turn_rate=50.0, net_elevation_range_m=400.0) == "mountain"
        )

    def test_coastal_requires_coastline_signal(self):
        """'coastal' fires only on a verified near_coast signal, never from low elevation.

        Regression test for the fix: a low-lying INLAND clip (river valley, fen)
        must not ship '#OceanRoad'. Low elevation alone → countryside.
        """
        assert classify_terrain(2.0, 10.0) == "countryside"
        assert classify_terrain(2.0, 10.0, near_coast=True) == "coastal"

    def test_low_elevation_winding_road_is_winding_not_coastal(self):
        """A twisty low inland road is 'winding', not falsely 'coastal'.

        Note: the old code inferred coastal from elevation_m < 30 BEFORE the
        winding check, so this misclassified. With near_coast gating, low
        elevation no longer preempts winding.
        """
        assert classify_terrain(5.0, 10.0, turn_rate=50.0) == "winding"

    def test_coastal_signal_takes_priority_over_winding(self):
        """A verified coastal road stays 'coastal' even when turns would score winding."""
        assert classify_terrain(5.0, 10.0, turn_rate=50.0, near_coast=True) == "coastal"


class TestTimeOfDay:
    """Time-of-day classification from GPS timestamps."""

    def test_morning(self):
        # 2025-09-13 09:00 UTC
        ts = 1726214400.0
        assert classify_time_of_day(ts) == "morning"

    def test_midday(self):
        # 2025-09-13 12:00 UTC
        ts = 1726225200.0
        assert classify_time_of_day(ts) == "midday"

    def test_afternoon(self):
        # 2025-09-13 15:00 UTC
        ts = 1726236000.0
        assert classify_time_of_day(ts) == "afternoon"

    def test_golden_hour(self):
        # 2025-09-13 18:00 UTC
        ts = 1726246800.0
        assert classify_time_of_day(ts) == "golden hour"

    def test_dawn(self):
        # 2025-09-13 05:00 UTC
        ts = 1726200000.0
        assert classify_time_of_day(ts) == "dawn"

    def test_evening(self):
        # 2025-09-13 21:00 UTC
        ts = 1726257600.0
        assert classify_time_of_day(ts) == "evening"


class TestFindNearestPOI:
    """POI lookup from video timestamp."""

    def test_finds_closest_poi(self):
        pois = [
            {"gps_time": 1000.0, "name": "Kenmare"},
            {"gps_time": 1200.0, "name": "Sneem"},
            {"gps_time": 1500.0, "name": "Caherdaniel"},
        ]
        # Video time 50s with scale 1x maps to GPS time 1050 — closest to Kenmare
        result = find_nearest_poi(50.0, pois, gps_start_time=1000.0, time_scale=1.0)
        assert result == "Kenmare"

    def test_returns_none_if_too_far(self):
        pois = [{"gps_time": 10000.0, "name": "Distant Town"}]
        result = find_nearest_poi(0.0, pois, gps_start_time=0.0, time_scale=1.0)
        assert result is None

    def test_returns_none_for_empty_pois(self):
        result = find_nearest_poi(50.0, [], gps_start_time=0.0, time_scale=1.0)
        assert result is None


class TestSegmentContextProperties:
    """SegmentContext dataclass behavior."""

    def test_location_detail_prefers_poi(self):
        ctx = SegmentContext(nearest_poi="Kenmare", named_route="Ring of Kerry")
        assert ctx.location_detail == "Kenmare"

    def test_location_detail_falls_back_to_route(self):
        ctx = SegmentContext(named_route="Ring of Kerry")
        assert ctx.location_detail == "Ring of Kerry"

    def test_location_detail_empty_when_no_context(self):
        ctx = SegmentContext()
        assert ctx.location_detail == ""

    def test_has_context_true_with_terrain(self):
        ctx = SegmentContext(terrain_type="coastal")
        assert ctx.has_context is True

    def test_has_context_false_when_empty(self):
        ctx = SegmentContext()
        assert ctx.has_context is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
