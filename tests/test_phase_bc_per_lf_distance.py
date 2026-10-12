"""Per-LF distance computation in phase_bc_day_first.

Pattern mirrors ``test_phase_bc_trim_ratio.py`` / per_lf_recording_date:
- Compute path: ``_compute_per_lf_distance`` parses each LF's GPX and
  sums Haversine waypoints via the public ``calculate_distance_km``.
- Resume backstop: same compute is idempotent on resume because GPX
  paths persist in ``state["per_lf_gpx_path"]`` (added alongside
  the distance map).
- Invariant: ``_assert_per_lf_distance_complete`` raises listing
  missing indices when ``per_lf_gpx_path`` has entries the distance
  map lacks. LFs without a GPX path are excluded (graceful — GPS-less
  drive → no distance line in the description).
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    _assert_per_lf_distance_complete,
    _compute_per_lf_distance,
)


def _make_journey(distances_pattern):
    """Build a mock Journey-like object whose waypoints span the given km pattern.

    distances_pattern: list of (lat, lon) tuples — Haversine sum across
    consecutive points gives the trajectory distance.
    """
    journey = MagicMock()
    waypoints = []
    for lat, lon in distances_pattern:
        wp = MagicMock()
        wp.latitude = lat
        wp.longitude = lon
        wp.timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        wp.elevation = 0.0
        waypoints.append(wp)
    journey.waypoints = waypoints
    return journey


class TestComputePerLfDistance:
    def test_empty_per_lf_gpx_path_returns_empty(self):
        result = _compute_per_lf_distance({}, parse_fn=MagicMock())
        assert result == {}

    def test_single_lf_with_gpx_computes_distance(self):
        # 1 degree latitude ≈ 111 km
        journey = _make_journey([(40.0, -75.0), (41.0, -75.0)])
        parse_fn = MagicMock(return_value=journey)
        result = _compute_per_lf_distance({0: "/fake/path.gpx"}, parse_fn=parse_fn)
        assert 0 in result
        assert 110.0 < result[0] < 112.0
        parse_fn.assert_called_once()

    def test_multiple_lfs_each_distance_computed(self):
        # LF 0: 1° lat span (~111 km); LF 1: 0.5° lat span (~55 km)
        def fake_parse(path):
            if "lf0" in str(path):
                return _make_journey([(40.0, -75.0), (41.0, -75.0)])
            return _make_journey([(40.0, -75.0), (40.5, -75.0)])

        result = _compute_per_lf_distance(
            {0: "/fake/lf0.gpx", 1: "/fake/lf1.gpx"}, parse_fn=fake_parse
        )
        assert 110.0 < result[0] < 112.0
        assert 54.0 < result[1] < 56.0

    def test_zero_distance_lf_omitted_from_map(self):
        """Stationary waypoints (no movement) → omitted, not zero entry.
        Surfacing "0 km" in the description would be more misleading than
        omitting the line."""

        def fake_parse(path):
            return _make_journey([(40.0, -75.0), (40.0, -75.0), (40.0, -75.0)])

        result = _compute_per_lf_distance({0: "/stationary.gpx"}, parse_fn=fake_parse)
        assert result == {}

    def test_gpx_parse_failure_omits_lf(self):
        """A GPX that fails to parse leaves the LF out of the map — the
        invariant will fire if the operator expects coverage; otherwise the
        description line is simply omitted for that LF."""

        def fake_parse(path):
            raise OSError("simulated parse failure")

        result = _compute_per_lf_distance({0: "/broken.gpx"}, parse_fn=fake_parse)
        assert result == {}

    def test_journey_with_no_waypoints_omits_lf(self):
        journey = MagicMock()
        journey.waypoints = []
        result = _compute_per_lf_distance(
            {0: "/empty.gpx"}, parse_fn=MagicMock(return_value=journey)
        )
        assert result == {}

    def test_string_key_normalized_to_int(self):
        """State-resume tolerance: per_lf_gpx_path may come back from JSON
        with string keys. The compute function must normalize."""
        journey = _make_journey([(40.0, -75.0), (41.0, -75.0)])
        result = _compute_per_lf_distance(
            {"0": "/fake/path.gpx"}, parse_fn=MagicMock(return_value=journey)
        )
        assert 0 in result


class TestAssertPerLfDistanceComplete:
    def test_full_coverage_does_not_raise(self):
        per_lf_gpx_path = {0: "/a.gpx", 1: "/b.gpx"}
        mapping = {0: 47.0, 1: 32.5}
        _assert_per_lf_distance_complete(per_lf_gpx_path, mapping)  # no raise

    def test_missing_entry_raises_with_indices(self):
        per_lf_gpx_path = {0: "/a.gpx", 1: "/b.gpx"}
        mapping = {0: 47.0}  # LF 1 missing
        with pytest.raises(RuntimeError) as exc:
            _assert_per_lf_distance_complete(per_lf_gpx_path, mapping)
        assert "[1]" in str(exc.value) or "1" in str(exc.value)
        assert "per_lf_distance_km" in str(exc.value)

    def test_empty_gpx_path_tolerated(self):
        """GPS-less drive: no GPX paths populated → no distance map needed.
        Invariant must not fire (else legacy WNC drives would break)."""
        _assert_per_lf_distance_complete({}, {})  # no raise

    def test_extra_entries_in_map_do_not_raise(self):
        """Defensive — if the map carries LFs not in per_lf_gpx_path (e.g.
        stale state from a previous run), don't error. Only missing
        coverage matters."""
        per_lf_gpx_path = {0: "/a.gpx"}
        mapping = {0: 47.0, 99: 5.0}
        _assert_per_lf_distance_complete(per_lf_gpx_path, mapping)  # no raise

    def test_string_keyed_per_lf_gpx_path_supported(self):
        """State resume may surface str-keyed paths from JSON. Match by int."""
        per_lf_gpx_path = {"0": "/a.gpx"}
        mapping = {0: 47.0}
        _assert_per_lf_distance_complete(per_lf_gpx_path, mapping)  # no raise
