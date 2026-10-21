"""The identity-keyed per-leg caches must verify the object, not trust ``id()``.

``dashboard._INDEX_CACHE``, ``dashboard._TOTAL_CACHE``,
``frame_state._ORDERED_CACHE`` and ``hud._SEGMENT_CAPABILITY_CACHE`` key on
``id(segments)`` so a renderer that reuses one segments object for every frame
of a leg pays for the index/scan once. CPython reuses the address of a freed
object for the next allocation, so a bare ``id()`` key can hand a *new* leg the
*previous* leg's cached entry -- observed as ``current_heart_rate_bpm is None``
for a track that carries heart rate (an order-dependent full-suite
flake under xdist). A cache hit is valid only when the cached entry still
refers to the very same object.
"""

from __future__ import annotations

import pytest

from gomoveshift.pipeline.route_story import dashboard, frame_state, hud
from gomoveshift.pipeline.route_story.dashboard import dashboard_state_at
from gomoveshift.pipeline.route_story.frame_state import frame_state_at
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.timeline import Scene, StoryTimeline


def _leg(*, heart_rate: bool) -> list[list[RoutePoint]]:
    hr = (120.0, 140.0) if heart_rate else (None, None)
    return [
        [
            RoutePoint(
                time_s=0, latitude=0.0, longitude=0.0, elevation_m=100.0, heart_rate_bpm=hr[0]
            ),
            RoutePoint(
                time_s=10, latitude=0.0, longitude=0.001, elevation_m=110.0, heart_rate_bpm=hr[1]
            ),
        ]
    ]


_TIMELINE = StoryTimeline(fps=10, scenes=(Scene("route_follow", 0, 10, 0.0, 10.0),))


_CACHES = (
    dashboard._INDEX_CACHE,
    dashboard._TOTAL_CACHE,
    frame_state._ORDERED_CACHE,
    hud._SEGMENT_CAPABILITY_CACHE,
)


@pytest.fixture(autouse=True)
def _clear_caches():
    for cache in _CACHES:
        cache.clear()
    yield
    for cache in _CACHES:
        cache.clear()


def test_dashboard_index_cache_ignores_an_entry_left_by_a_different_object():
    """A stale entry under this leg's ``id()`` must be rebuilt, not reused."""
    without_hr = _leg(heart_rate=False)
    dashboard_state_at(_TIMELINE, 5, without_hr, moving_speed_mps=1.0)
    stale_key = next(iter(dashboard._INDEX_CACHE))
    stale_entry = dashboard._INDEX_CACHE.pop(stale_key)

    with_hr = _leg(heart_rate=True)
    # Simulate CPython handing the new leg the freed leg's address.
    dashboard._INDEX_CACHE[(id(with_hr), 1.0)] = stale_entry

    state = dashboard_state_at(_TIMELINE, 5, with_hr, moving_speed_mps=1.0)

    assert state.current_heart_rate_bpm == pytest.approx(130.0)


def test_dashboard_index_cache_still_hits_for_the_same_object():
    leg = _leg(heart_rate=True)
    dashboard_state_at(_TIMELINE, 5, leg, moving_speed_mps=1.0)
    (key,) = dashboard._INDEX_CACHE
    first = dashboard._INDEX_CACHE[key]
    dashboard_state_at(_TIMELINE, 6, leg, moving_speed_mps=1.0)
    assert dashboard._INDEX_CACHE[key] is first


def test_hud_capability_cache_ignores_an_entry_left_by_a_different_object():
    without_hr = _leg(heart_rate=False)
    assert hud._segment_capabilities(without_hr) == (False, False, False)
    stale_key = next(iter(hud._SEGMENT_CAPABILITY_CACHE))
    stale_entry = hud._SEGMENT_CAPABILITY_CACHE.pop(stale_key)

    with_hr = _leg(heart_rate=True)
    hud._SEGMENT_CAPABILITY_CACHE[id(with_hr)] = stale_entry

    assert hud._segment_capabilities(with_hr) == (True, False, False)


def test_hud_capability_cache_still_hits_for_the_same_object():
    leg = _leg(heart_rate=True)
    hud._segment_capabilities(leg)
    (key,) = hud._SEGMENT_CAPABILITY_CACHE
    first = hud._SEGMENT_CAPABILITY_CACHE[key]
    hud._segment_capabilities(leg)
    assert hud._SEGMENT_CAPABILITY_CACHE[key] is first


def _two_point_leg(longitude_end: float) -> list[list[RoutePoint]]:
    return [
        [
            RoutePoint(time_s=0, latitude=0.0, longitude=0.0),
            RoutePoint(time_s=10, latitude=0.0, longitude=longitude_end),
        ]
    ]


def test_dashboard_total_is_summarized_once_per_leg_not_once_per_frame(monkeypatch):
    """A leg's total distance is the same for every frame; re-summarizing the
    whole track per frame made a 420-frame leg over a 14,001-point track spend
    ~0.17 s/frame in ``summarize_track`` (route-story suite profile, 2026-10-12).
    """
    calls = []
    real = dashboard.summarize_track
    monkeypatch.setattr(
        dashboard, "summarize_track", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    leg = _leg(heart_rate=True)
    states = [dashboard_state_at(_TIMELINE, f, leg, moving_speed_mps=1.0) for f in range(10)]
    assert len(calls) == 1
    assert {s.distance_total_m for s in states} == {real(leg, moving_speed_mps=1.0).distance_m}


def test_dashboard_total_cache_ignores_an_entry_left_by_a_different_object():
    short = _two_point_leg(0.001)
    short_total = dashboard_state_at(_TIMELINE, 5, short, moving_speed_mps=1.0).distance_total_m
    stale_entry = dashboard._TOTAL_CACHE.pop(next(iter(dashboard._TOTAL_CACHE)))

    long = _two_point_leg(0.002)
    dashboard._TOTAL_CACHE[id(long)] = stale_entry

    long_total = dashboard_state_at(_TIMELINE, 5, long, moving_speed_mps=1.0).distance_total_m
    assert long_total == pytest.approx(2 * short_total)


def test_frame_state_orders_a_leg_once_not_once_per_frame(monkeypatch):
    calls = []
    real = frame_state._segment_distance_offsets
    monkeypatch.setattr(
        frame_state, "_segment_distance_offsets", lambda s: calls.append(1) or real(s)
    )
    leg = _leg(heart_rate=False)
    for f in range(10):
        frame_state_at(_TIMELINE, f, leg)
    assert len(calls) == 1


def test_frame_state_cache_ignores_an_entry_left_by_a_different_object():
    short = _two_point_leg(0.001)
    short_distance = frame_state_at(_TIMELINE, 10, short).distance_m
    stale_entry = frame_state._ORDERED_CACHE.pop(next(iter(frame_state._ORDERED_CACHE)))

    long = _two_point_leg(0.002)
    frame_state._ORDERED_CACHE[id(long)] = stale_entry

    assert frame_state_at(_TIMELINE, 10, long).distance_m == pytest.approx(2 * short_distance)
