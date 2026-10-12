"""Per-LF GPS numeric facts route through one cleaned/teleport-stripped
trace + a physical-plausibility gate.

Real-data-shaped regression for the SC "315 mi" class: a per-LF GPX trace that
opens with a cold-start teleport spike (a sub-second jump of tens of km from a
stale-locked fix to the real route) plus normal driving. The compute fns now
clean the trace via ``_cleaned_waypoints`` (reusing
``gps_quality.filter_high_quality_points`` — null-island + spatial + shared
km/SEC temporal-teleport strip) before summing, and gate the result on a
physical ceiling, so the inflated raw value is never published.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from gomoveshift.pipeline.post_encoding_invariants import (
    _assert_per_lf_distance_complete,
    _compute_per_lf_distance,
    _compute_per_lf_elevation_gain,
    _compute_per_lf_speed,
)

_T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _journey(points):
    """points: list of (lat, lon, seconds_from_start, elevation_m) → mock Journey."""
    j = MagicMock()
    wps = []
    for lat, lon, sec, elev in points:
        wp = MagicMock()
        wp.latitude, wp.longitude = lat, lon
        wp.timestamp = (_T0 + timedelta(seconds=sec)) if sec is not None else None
        wp.elevation = elev
        wps.append(wp)
    j.waypoints = wps
    return j


def _normal_drive(n=12, dt=60, elev0=100.0, delev=2.0):
    """Clean ~realistic trace: ~1.11 km steps over ``dt`` s (~66 km/h), gentle climb."""
    return [(40.0 + i * 0.01, -75.0, i * dt, elev0 + i * delev) for i in range(n)]


def _with_cold_start_teleport():
    """A stale lock ~111 km north jumped to the real route in 0.5 s — the 315-mi
    shape; the spike also carries a 5000 m garbage altitude."""
    return [(41.0, -75.0, -0.5, 5000.0)] + _normal_drive()


def _parse(points):
    return MagicMock(return_value=_journey(points))


class TestDistance:
    def test_teleport_spike_stripped(self):
        # Raw sum would add the ~111 km jump; cleaning leaves only the ~12 km body.
        result = _compute_per_lf_distance(
            {0: "/a.gpx"}, parse_fn=_parse(_with_cold_start_teleport())
        )
        assert 0 in result
        assert 10.0 < result[0] < 16.0  # real driving distance, nowhere near 200 mi

    def test_clean_trace_unchanged(self):
        result = _compute_per_lf_distance({0: "/a.gpx"}, parse_fn=_parse(_normal_drive()))
        assert 10.0 < result[0] < 16.0

    def test_implausible_avg_speed_omits_and_invariant_tolerates(self):
        # ~12 km over a ~2 s span → ~22,000 km/h avg: gated out, line omitted.
        fast = [(40.0 + i * 0.01, -75.0, i * 0.2, 100.0) for i in range(12)]
        rejected: set = set()
        result = _compute_per_lf_distance(
            {0: "/a.gpx"}, parse_fn=_parse(fast), plausibility_rejected=rejected
        )
        assert result == {}
        assert 0 in rejected
        # A plausibility-omitted LF is tolerated, not a false completeness failure.
        _assert_per_lf_distance_complete({0: "/a.gpx"}, result, tolerated_indices=rejected)


class TestStationaryOmission:
    """A mostly-stationary (parked-with-jitter) LF
    whose cleaned trace collapses to ~one coordinate → 0 km distance. That is a
    LEGITIMATE omission of the optional distance/elevation line, NOT a parse
    failure — the compute fns add the index to ``plausibility_rejected`` and
    drop the line instead of aborting the whole multi-hour run."""

    def test_stationary_distance_omitted_and_tolerated(self):
        stationary = [(40.0, -75.0, i * 30, 100.0) for i in range(12)]
        rejected: set = set()
        result = _compute_per_lf_distance(
            {0: "/a.gpx"}, parse_fn=_parse(stationary), plausibility_rejected=rejected
        )
        assert result == {}  # no fabricated "0 km" line
        assert 0 in rejected
        # The omitted LF is tolerated, not a false completeness failure.
        _assert_per_lf_distance_complete({0: "/a.gpx"}, result, tolerated_indices=rejected)

    def test_stationary_does_not_abort_elevation(self):
        # A stationary trace cleans to a non-empty but zero-displacement trace:
        # the elevation gain is a legitimate 0 (no climb), never a parse failure
        # that aborts the run.
        stationary = [(40.0, -75.0, i * 30, 100.0) for i in range(12)]
        result = _compute_per_lf_elevation_gain({0: "/a.gpx"}, parse_fn=_parse(stationary))
        assert result.get(0, 0.0) == 0.0


class TestElevation:
    def test_teleport_spike_stripped(self):
        # The 5000 m cold-start altitude is dropped with the teleport head.
        result = _compute_per_lf_elevation_gain(
            {0: "/a.gpx"}, parse_fn=_parse(_with_cold_start_teleport())
        )
        assert 0 in result and result[0] < 60.0  # body climbs only 22 m

    def test_clean_trace_unchanged(self):
        result = _compute_per_lf_elevation_gain({0: "/a.gpx"}, parse_fn=_parse(_normal_drive()))
        assert result[0] == 22.0  # 11 steps × 2 m

    def test_noise_inflated_gain_per_km_omitted(self):
        # ~2.2 km trace with ±500 m sawtooth noise → impossible m/km, gated out.
        noisy = [
            (40.0 + i * 0.002, -75.0, i * 20, 100.0 + (500.0 if i % 2 else 0.0)) for i in range(12)
        ]
        rejected: set = set()
        _compute_per_lf_elevation_gain(
            {0: "/a.gpx"}, parse_fn=_parse(noisy), plausibility_rejected=rejected
        )
        assert 0 in rejected


class TestSpeed:
    def test_teleport_step_never_counts(self):
        result = _compute_per_lf_speed({0: "/a.gpx"}, parse_fn=_parse(_with_cold_start_teleport()))
        assert 0 in result and result[0]["max_kmh"] < 200.0  # body is uniform ~66 km/h

    def test_clean_trace_unchanged(self):
        result = _compute_per_lf_speed({0: "/a.gpx"}, parse_fn=_parse(_normal_drive(dt=30)))
        assert 0 in result and result[0]["avg_kmh"] > 0
