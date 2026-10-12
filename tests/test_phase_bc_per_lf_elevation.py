"""Per-LF elevation-gain computation in phase_bc_day_first.

Pattern mirrors ``test_phase_bc_per_lf_distance.py``:
- Compute path: ``_compute_per_lf_elevation_gain`` parses each LF's GPX and
  sums positive elevation deltas via the public ``calculate_elevation_gain_m``.
- Invariant: ``_assert_per_lf_elevation_gain_complete`` raises listing
  missing indices when ``per_lf_gpx_path`` has entries the elevation map
  lacks. LFs without a GPX path are excluded (graceful — GPS-less drive →
  no elevation line in the description).
- Flat / zero-gain trajectories are still recorded as ``0.0`` in the map
  (the formatter handles the "suppress < 50 m" framing — phase_bc records
  the measurement; the description layer decides what is interesting).
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    _assert_per_lf_elevation_gain_complete,
    _compute_per_lf_elevation_gain,
)


def _make_journey(elevations):
    """Build a mock Journey-like object whose waypoints carry the given elevation series."""
    journey = MagicMock()
    waypoints = []
    for idx, elev in enumerate(elevations):
        wp = MagicMock()
        wp.latitude = 40.0
        wp.longitude = -75.0
        wp.timestamp = datetime(2026, 1, 1, 12, idx, tzinfo=timezone.utc)
        wp.elevation = elev
        waypoints.append(wp)
    journey.waypoints = waypoints
    return journey


class TestComputePerLfElevationGain:
    def test_empty_per_lf_gpx_path_returns_empty(self):
        result = _compute_per_lf_elevation_gain({}, parse_fn=MagicMock())
        assert result == {}

    def test_single_lf_with_climb(self):
        journey = _make_journey([100.0, 150.0, 200.0])  # +50 +50 = 100
        parse_fn = MagicMock(return_value=journey)
        result = _compute_per_lf_elevation_gain({0: "/fake/path.gpx"}, parse_fn=parse_fn)
        assert result[0] == pytest.approx(100.0)

    def test_multiple_lfs_each_gain_computed(self):
        def fake_parse(path):
            if "lf0" in str(path):
                return _make_journey([100.0, 350.0])  # +250
            return _make_journey([200.0, 280.0])  # +80

        result = _compute_per_lf_elevation_gain(
            {0: "/fake/lf0.gpx", 1: "/fake/lf1.gpx"}, parse_fn=fake_parse
        )
        assert result[0] == pytest.approx(250.0)
        assert result[1] == pytest.approx(80.0)

    def test_flat_trajectory_records_zero(self):
        """Unlike distance (where 0 is misleading and the entry is dropped),
        elevation gain of 0 is a real, honest measurement — record it. The
        50 m suppression threshold lives at the formatter, not here."""
        journey = _make_journey([100.0, 100.0, 100.0])
        result = _compute_per_lf_elevation_gain(
            {0: "/flat.gpx"}, parse_fn=MagicMock(return_value=journey)
        )
        assert result == {0: 0.0}

    def test_pure_descent_records_zero(self):
        journey = _make_journey([300.0, 200.0, 100.0])
        result = _compute_per_lf_elevation_gain(
            {0: "/descent.gpx"}, parse_fn=MagicMock(return_value=journey)
        )
        assert result == {0: 0.0}

    def test_mixed_up_down_sums_positive_only(self):
        # 100 → 200 (+100) → 150 (-50, ignored) → 300 (+150) = 250
        journey = _make_journey([100.0, 200.0, 150.0, 300.0])
        result = _compute_per_lf_elevation_gain(
            {0: "/mixed.gpx"}, parse_fn=MagicMock(return_value=journey)
        )
        assert result[0] == pytest.approx(250.0)

    def test_gpx_parse_failure_omits_lf(self):
        def fake_parse(path):
            raise OSError("simulated parse failure")

        result = _compute_per_lf_elevation_gain({0: "/broken.gpx"}, parse_fn=fake_parse)
        assert result == {}

    def test_journey_with_no_waypoints_omits_lf(self):
        journey = MagicMock()
        journey.waypoints = []
        result = _compute_per_lf_elevation_gain(
            {0: "/empty.gpx"}, parse_fn=MagicMock(return_value=journey)
        )
        assert result == {}

    def test_string_key_normalized_to_int(self):
        journey = _make_journey([100.0, 200.0])
        result = _compute_per_lf_elevation_gain(
            {"0": "/fake/path.gpx"}, parse_fn=MagicMock(return_value=journey)
        )
        assert 0 in result
        assert result[0] == pytest.approx(100.0)

    def test_none_elevations_silently_skipped(self):
        """Matches upstream semantics — None-on-either-side deltas are
        excluded from the sum. A trajectory of all-Nones records 0.0."""
        journey = _make_journey([100.0, None, 200.0, 250.0])
        # 100→None: skip. None→200: skip. 200→250: +50. Total = 50.
        result = _compute_per_lf_elevation_gain(
            {0: "/with-nones.gpx"}, parse_fn=MagicMock(return_value=journey)
        )
        assert result[0] == pytest.approx(50.0)


class TestAssertPerLfElevationGainComplete:
    def test_full_coverage_does_not_raise(self):
        per_lf_gpx_path = {0: "/a.gpx", 1: "/b.gpx"}
        mapping = {0: 250.0, 1: 0.0}  # zero gain still counts as coverage
        _assert_per_lf_elevation_gain_complete(per_lf_gpx_path, mapping)  # no raise

    def test_missing_entry_raises_with_indices(self):
        per_lf_gpx_path = {0: "/a.gpx", 1: "/b.gpx"}
        mapping = {0: 250.0}
        with pytest.raises(RuntimeError) as exc:
            _assert_per_lf_elevation_gain_complete(per_lf_gpx_path, mapping)
        assert "1" in str(exc.value)
        assert "per_lf_elevation_gain_m" in str(exc.value)

    def test_empty_gpx_path_tolerated(self):
        _assert_per_lf_elevation_gain_complete({}, {})  # no raise

    def test_extra_entries_in_map_do_not_raise(self):
        per_lf_gpx_path = {0: "/a.gpx"}
        mapping = {0: 250.0, 99: 5.0}
        _assert_per_lf_elevation_gain_complete(per_lf_gpx_path, mapping)  # no raise

    def test_string_keyed_per_lf_gpx_path_supported(self):
        per_lf_gpx_path = {"0": "/a.gpx"}
        mapping = {0: 250.0}
        _assert_per_lf_elevation_gain_complete(per_lf_gpx_path, mapping)  # no raise
