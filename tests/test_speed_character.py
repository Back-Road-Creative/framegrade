"""Per-LF speed/character (avg/max/std km/h + derived character).

Covers the compute (``_compute_per_lf_speed`` — per-step speeds from waypoint
pairs via the public ``calculate_distance_km``, glitch-filtered), the
``_speed_character`` band logic, the country-aware ``🚗 Pace:`` formatter, the
int-keyed ``per_lf_speed`` state field, and the ``generate()`` splice.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline.drive_state_model import _INT_KEY_DICT_FIELDS, PipelineState
from gomoveshift.pipeline.longform_metadata import (
    LongFormMetadataGenerator,
    _format_speed_line,
    _speed_character,
)
from gomoveshift.pipeline.post_encoding_invariants import _compute_per_lf_speed

_T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _journey(points):
    """points: list of (lat, lon, seconds_from_start) → mock Journey."""
    j = MagicMock()
    wps = []
    for lat, lon, sec in points:
        wp = MagicMock()
        wp.latitude, wp.longitude = lat, lon
        wp.timestamp = (_T0 + timedelta(seconds=sec)) if sec is not None else None
        wps.append(wp)
    j.waypoints = wps
    return j


def _compute_one(points, key=0):
    return _compute_per_lf_speed({key: "/a.gpx"}, parse_fn=MagicMock(return_value=_journey(points)))


# ~0.01 deg latitude ≈ 1.11 km. Over 60 s → ~66.6 km/h.
class TestComputePerLfSpeed:
    def test_empty_returns_empty(self):
        assert _compute_per_lf_speed({}, parse_fn=MagicMock()) == {}

    def test_avg_and_max_computed(self):
        # two ~1.11 km steps over 60 s each → ~66.6 km/h
        pts = [(40.0, -75.0, 0), (40.01, -75.0, 60), (40.02, -75.0, 120)]
        s = _compute_one(pts)[0]
        assert s["avg_kmh"] == pytest.approx(66.7, abs=1.0)
        assert s["max_kmh"] == pytest.approx(66.7, abs=1.0)
        assert s["std_kmh"] == pytest.approx(0.0, abs=0.5)

    def test_glitch_step_excluded(self):
        # second step jumps 1 deg (~111 km) in 1 s → ~400,000 km/h, dropped.
        pts = [(40.0, -75.0, 0), (40.01, -75.0, 60), (41.01, -75.0, 61)]
        s = _compute_one(pts)[0]
        assert s["max_kmh"] < 200.0  # glitch not counted

    def test_nonpositive_dt_skipped(self):
        pts = [(40.0, -75.0, 0), (40.01, -75.0, 0), (40.02, -75.0, 60)]
        s = _compute_one(pts)[0]
        assert s["avg_kmh"] > 0  # the zero-dt step skipped, the 60 s step counts

    def test_missing_timestamps_omit_lf(self):
        assert _compute_one([(40.0, -75.0, None), (40.01, -75.0, None)]) == {}

    def test_single_waypoint_omits_lf(self):
        assert _compute_one([(40.0, -75.0, 0)]) == {}

    def test_parse_failure_and_str_key(self):
        def boom(_):
            raise OSError("x")

        assert _compute_per_lf_speed({0: "/a.gpx"}, parse_fn=boom) == {}
        pts = [(40.0, -75.0, 0), (40.01, -75.0, 60)]
        assert next(iter(_compute_one(pts, key="0"))) == 0


class TestSpeedCharacter:
    def test_town_pace(self):
        assert _speed_character(20.0, 5.0) == "town & village pace"

    def test_winding_high_variance(self):
        assert _speed_character(55.0, 30.0) == "winding roads"

    def test_highway_cruise(self):
        assert _speed_character(95.0, 8.0) == "highway cruise"

    def test_country_roads_default(self):
        assert _speed_character(60.0, 18.0) == "country roads"


class TestFormatSpeedLine:
    def test_us_uses_mph(self):
        line = _format_speed_line({"avg_kmh": 96.5, "std_kmh": 8.0}, "Asheville, NC, USA")
        assert line == "🚗 Pace: 60 mph avg · highway cruise\n"

    def test_non_us_uses_kmh(self):
        line = _format_speed_line({"avg_kmh": 78.0, "std_kmh": 30.0}, "Kerry, Ireland")
        assert line == "🚗 Pace: 78 km/h avg · winding roads\n"

    def test_none_or_zero_avg_returns_empty(self):
        assert _format_speed_line(None, "X, Ireland") == ""
        assert _format_speed_line({"avg_kmh": 0.0}, "X, Ireland") == ""


class TestStateRoundTrip:
    def test_registered_and_default_none(self):
        assert "per_lf_speed" in _INT_KEY_DICT_FIELDS
        assert PipelineState().per_lf_speed is None

    def test_int_keys_survive_roundtrip(self):
        state = PipelineState()
        state.per_lf_speed = {0: {"avg_kmh": 78.0, "max_kmh": 110.0, "std_kmh": 22.0}}
        restored = PipelineState.from_dict(state.to_dict())
        assert set(restored.per_lf_speed) == {0}
        assert isinstance(next(iter(restored.per_lf_speed)), int)
        assert restored.per_lf_speed[0]["avg_kmh"] == pytest.approx(78.0)


class TestGenerateIntegration:
    def test_pace_line_never_rendered(self):
        """The 🚗 Pace/avg-speed line was dropped from the long-form
        description entirely — present or absent ``speed``, the
        description never carries it."""
        gen = LongFormMetadataGenerator()
        with_speed = gen.generate(
            location="Kerry, Ireland",
            video_duration=1800.0,
            speed={"avg_kmh": 78.0, "max_kmh": 110.0, "std_kmh": 30.0},
        )
        assert "🚗 Pace:" not in with_speed["description"]
        without = gen.generate(location="Kerry, Ireland", video_duration=1800.0)
        assert "🚗 Pace:" not in without["description"]
