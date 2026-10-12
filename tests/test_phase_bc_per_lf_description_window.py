"""Per-LF description GPS stats are windowed to the LF's INCLUDED segments.

The windowed per-LF description facts (distance / elevation / speed / bearing)
parse the representative episode's whole GPX, keyed only on
``per_lf_gpx_path``. A best-moments LF (trim ~0.3) keeps ~30% of that episode,
so before this fix its ``🛣️ Distance:`` shipped the FULL episode distance (the
surviving "315 mi"-class inflation the plausibility ceiling doesn't catch —
the implied average speed stays plausible). (The ``🗺️ Through:`` line is now
sourced from ``resolved.regions``, not a windowed GPX pass.)

Fix: window the parsed trace to the LF's included source-timeline segments
(reusing ``phase_d._build_included_segments_for_lf`` +
``gps_quality.trim_gps_to_clip_timeline``) BEFORE computing any per-LF stat.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from gomoveshift.pipeline.post_encoding_invariants import _compute_per_lf_distance


def _line_journey(n_points: int, total_lat_deg: float, total_seconds: float):
    """Mock Journey whose waypoints move due-north at a constant cadence.

    Waypoint ``i`` is at ``lat 40 + total_lat_deg*i/(n-1)`` and timestamp
    ``base + total_seconds*i/(n-1)`` — so the source-second offset from the
    first waypoint equals its position along the trace.
    """
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    journey = MagicMock()
    waypoints = []
    for i in range(n_points):
        frac = i / (n_points - 1)
        wp = MagicMock()
        wp.latitude = 40.0 + total_lat_deg * frac
        wp.longitude = -75.0
        wp.timestamp = base + timedelta(seconds=total_seconds * frac)
        wp.elevation = 0.0
        waypoints.append(wp)
    journey.waypoints = waypoints
    return journey


class TestDistanceWindowing:
    def test_full_episode_distance_without_window(self):
        """No included-segments → whole-episode distance (prior behaviour)."""
        result = _compute_per_lf_distance(
            {0: "/ep.gpx"},
            parse_fn=MagicMock(return_value=_line_journey(11, 1.0, 7200.0)),
        )
        assert 108.0 < result[0] < 114.0  # ~111 km over the whole 1° span

    def test_distance_windowed_to_included_segments(self):
        """Trim ~0.3 LF → distance ≈ the included 30%, NOT the full episode."""
        result = _compute_per_lf_distance(
            {0: "/ep.gpx"},
            parse_fn=MagicMock(return_value=_line_journey(11, 1.0, 7200.0)),
            per_lf_included_segments={0: [(0.0, 2160.0)]},
        )
        # first 2160 s of a 7200 s / 1° trace ≈ 0.3° ≈ 33 km, well under the
        # full-episode ~111 km the bug shipped.
        assert 30.0 < result[0] < 36.0


class TestDistanceSegmentIsolation:
    """Two disjoint included segments must NOT be bridged.

    ``per_lf_included_segments`` can carry several non-contiguous ranges (a
    best-moments LF keeps several separate clips out of one long episode).
    Concatenating their waypoints into one flat trace before summing
    consecutive-point distance folds the straight-line gap BETWEEN the kept
    clips — footage that was CUT — into the reported ``🛣️ Distance:``. On a
    real run this shipped 21.9 km for a 9m19s cut whose own kept footage
    measures 6.4 km; two cross-segment jumps (2.5 km, 12.9 km) accounted
    for the other 15.4 km.
    """

    def test_disjoint_segments_do_not_bridge_the_gap(self):
        from gomoveshift.pipeline.post_encoding_invariants import (
            _compute_per_lf_distance,
        )

        # Two 1-step (720s) windows at opposite ends of a 7200s / 1deg trace,
        # each carrying ~11.1 km of real (in-window) movement. A correct,
        # segment-isolated sum reports ~22 km. Bridging the ~6480s gap between
        # them (8 steps of ~11.1 km each) would report ~111 km — the whole
        # episode's distance in disguise.
        result = _compute_per_lf_distance(
            {0: "/ep.gpx"},
            parse_fn=MagicMock(return_value=_line_journey(11, 1.0, 7200.0)),
            per_lf_included_segments={0: [(0.0, 720.0), (6480.0, 7200.0)]},
        )
        assert 20.0 < result[0] < 25.0


def _four_point_journey():
    """Two 120 km/h, 2-km, 60s windows separated by a plausible 100 km/h,
    50 km, 1800s gap — the gap step is itself physically plausible (under the
    200 km/h glitch ceiling) so only segment isolation, not the existing
    teleport filter, can keep it out of the average."""
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    specs = [(0, 40.00000), (60, 40.01798), (1860, 40.46758), (1920, 40.48556)]
    journey = MagicMock()
    waypoints = []
    for sec, lat in specs:
        wp = MagicMock()
        wp.latitude = lat
        wp.longitude = -75.0
        wp.timestamp = base + timedelta(seconds=sec)
        wp.elevation = 0.0
        waypoints.append(wp)
    journey.waypoints = waypoints
    return journey


class TestSpeedSegmentIsolation:
    """Mirror of the distance case for ``🚗 Pace:`` (avg_kmh)."""

    def test_disjoint_segments_do_not_bridge_the_gap(self):
        from gomoveshift.pipeline.post_encoding_invariants import (
            _compute_per_lf_speed,
        )

        result = _compute_per_lf_speed(
            {0: "/ep.gpx"},
            parse_fn=MagicMock(return_value=_four_point_journey()),
            per_lf_included_segments={0: [(0.0, 60.0), (1860.0, 1920.0)]},
        )
        # Both in-window steps are ~120 km/h. Bridging the gap folds in the
        # ~100 km/h cross-gap step, pulling avg_kmh down to ~101 km/h.
        assert 115.0 < result[0]["avg_kmh"] < 125.0
