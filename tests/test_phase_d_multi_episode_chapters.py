"""A full-drive LF spanning two GPX-bearing episodes charts BOTH traces.

Regression: a full-drive LF covered a 1800 s episode and a
701 s episode, but chapters were built from the 701 s episode's GPX alone
(segment-count representative) and linearly stretched 3.66x over the 41.5 min
LF — one intermediate chapter, placed on the wrong roads.
"""

from types import SimpleNamespace

from gomoveshift.pipeline.drive_phases.phase_d_metadata import (
    _build_lf_multi_episode_journey,
)


def _tp(ts, lat, lon):
    return SimpleNamespace(timestamp=ts, lat=lat, lon=lon)


def _seg(idx, start, end, ep_global_start):
    return SimpleNamespace(
        source_file_index=idx,
        start_time=start,
        end_time=end,
        original_start_time=ep_global_start + start,
        original_end_time=ep_global_start + end,
    )


def _gps(points):
    return SimpleNamespace(trackpoints=points)


def test_two_episode_lf_gets_one_journey_on_the_lf_timeline():
    # ep0: 100 s long, GPX absolute ts 1000..1100 (offset 1000)
    # ep1: 50 s long, GPX absolute ts 5000..5050 (offset 5000)
    ep0 = _gps([_tp(1000 + t, 51.5, -9.3 - t / 1000) for t in range(0, 101, 10)])
    ep1 = _gps([_tp(5000 + t, 51.5, -9.6 - t / 1000) for t in range(0, 51, 10)])
    gps_per_episode = [(ep0, 1000.0), (ep1, 5000.0)]
    lf = SimpleNamespace(
        segments=[_seg(0, 0.0, 100.0, 0.0), _seg(1, 0.0, 20.0, 100.0), _seg(1, 30.0, 50.0, 100.0)]
    )

    built = _build_lf_multi_episode_journey(lf, gps_per_episode, gpx_files=["a.gpx", "b.gpx"])

    assert built is not None
    journey, ranges = built
    lons = [wp.longitude for wp in journey.waypoints]
    assert min(lons) < -9.6 and max(lons) > -9.31  # both episodes' traces present
    span = (journey.end_date - journey.start_date).total_seconds()
    assert span == 150.0  # ep1 lands after ep0 on the LF's own timeline
    assert ranges == [(0.0, 100.0), (100.0, 120.0), (130.0, 150.0)]


def test_single_episode_lf_keeps_the_per_episode_gpx_path():
    ep0 = _gps([_tp(1000 + t, 51.5, -9.3) for t in range(0, 101, 10)])
    lf = SimpleNamespace(segments=[_seg(0, 0.0, 100.0, 0.0)])
    assert _build_lf_multi_episode_journey(lf, [(ep0, 1000.0)], gpx_files=["a.gpx"]) is None


def test_episode_without_gpx_is_ignored():
    ep0 = _gps([_tp(1000 + t, 51.5, -9.3) for t in range(0, 101, 10)])
    lf = SimpleNamespace(segments=[_seg(0, 0.0, 100.0, 0.0), _seg(1, 0.0, 20.0, 100.0)])
    assert (
        _build_lf_multi_episode_journey(lf, [(ep0, 1000.0), None], gpx_files=["a.gpx", None])
        is None
    )
