"""Per-LF GPS distance/elevation must SUM across every source
episode an LF spans, not just its single representative episode.

Before the fix the computes parsed only the ONE representative GPX
(``per_lf_gpx_path[lf]``), so a two-episode LF shipped only the representative
episode's distance/climb. The single-rep map stays unchanged; a new
``per_lf_gpx_episode_paths`` (``{lf: {gpx_path: episode_local_ranges}}``) lets
the computes sum across all episodes. Each test asserts BOTH the old
single-rep witness (understated) and the new summed value (correct).
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from gomoveshift.pipeline.post_encoding_invariants import (
    _compute_per_lf_distance,
    _compute_per_lf_elevation_gain,
)

_T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
# 1 degree latitude ~= 111.195 km; deltas land on clean ~km values.
_DELTA_3KM, _DELTA_2KM, _DELTA_4KM = 0.027, 0.018, 0.036


def _journey(lat_delta, elev_gain=0.0):
    """Two-waypoint journey: ``lat_delta`` deg north over 300 s, +``elev_gain`` m."""
    wps = []
    for i, (lat, elev) in enumerate([(40.0, 100.0), (40.0 + lat_delta, 100.0 + elev_gain)]):
        wp = MagicMock()
        wp.latitude, wp.longitude, wp.elevation = lat, -75.0, elev
        wp.timestamp = _T0 + timedelta(seconds=i * 300)
        wps.append(wp)
    journey = MagicMock()
    journey.waypoints = wps
    return journey


def _fake_parse(path):
    return {
        "/A.gpx": _journey(_DELTA_3KM, 30.0),
        "/B.gpx": _journey(_DELTA_2KM, 20.0),
        "/C.gpx": _journey(_DELTA_4KM, 40.0),
    }[str(path)]


# LF 0 spans episodes 0 (GPX A) + 1 (GPX B); representative = A. LF 1 = single (C).
_PATH = {0: "/A.gpx", 1: "/C.gpx"}
_INCLUDED = {0: [(0.0, 400.0)], 1: [(0.0, 400.0)]}
_EPISODE_MAP = {
    0: {"/A.gpx": [(0.0, 400.0)], "/B.gpx": [(0.0, 400.0)]},
    1: {"/C.gpx": [(0.0, 400.0)]},
}


def test_multi_episode_distance_sums_across_episodes():
    old = _compute_per_lf_distance(_PATH, parse_fn=_fake_parse, per_lf_included_segments=_INCLUDED)
    assert 2.9 < old[0] < 3.1  # witness: representative episode only (understated)
    new = _compute_per_lf_distance(
        _PATH,
        parse_fn=_fake_parse,
        per_lf_included_segments=_INCLUDED,
        per_lf_gpx_episode_paths=_EPISODE_MAP,
    )
    assert 4.9 < new[0] < 5.1  # 3 km + 2 km summed across both episodes
    assert 3.9 < new[1] < 4.1  # single-episode LF unchanged (no regression)


def test_multi_episode_elevation_sums_across_episodes():
    old = _compute_per_lf_elevation_gain(
        _PATH, parse_fn=_fake_parse, per_lf_included_segments=_INCLUDED
    )
    assert 29.0 < old[0] < 31.0  # witness: episode 0's 30 m climb only
    new = _compute_per_lf_elevation_gain(
        _PATH,
        parse_fn=_fake_parse,
        per_lf_included_segments=_INCLUDED,
        per_lf_gpx_episode_paths=_EPISODE_MAP,
    )
    assert 49.0 < new[0] < 51.0  # 30 m + 20 m summed
    assert 39.0 < new[1] < 41.0  # single-episode LF unchanged
