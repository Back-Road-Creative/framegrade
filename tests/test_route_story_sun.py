"""Tests for the Route Story sun model: real light, not a fixed 315° constant."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gomoveshift.pipeline.route_story.sun import (
    SceneLight,
    hillshade_exaggeration,
    light_at,
    light_for_pose,
    sun_sweep,
)


def test_light_at_requires_tz_aware_datetime():
    with pytest.raises(ValueError):
        light_at(40.0, -105.0, datetime(2026, 3, 20, 12, 0, 0))


def test_light_at_equinox_noon_matches_expected_geometry():
    # 2026-03-20 is the March equinox; lon=0 keeps solar noon near 12:00 UTC.
    when = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
    light = light_at(40.0, 0.0, when)
    assert isinstance(light, SceneLight)
    assert not light.night
    # Elevation ~= 90 - latitude at solar noon on an equinox.
    assert 45.0 <= light.elevation_deg <= 55.0
    # Sun roughly due south for a northern mid-latitude observer at solar noon.
    assert 170.0 <= light.azimuth_deg <= 190.0


def test_light_at_midnight_is_night_with_low_clamped_intensity():
    when = datetime(2026, 3, 20, 0, 0, 0, tzinfo=timezone.utc)
    light = light_at(40.0, 0.0, when)
    assert light.night is True
    assert light.elevation_deg == 0.0  # MapLibre wants elevation >= 0
    assert light.intensity == pytest.approx(0.15)


def test_scene_light_elevation_never_negative():
    when = datetime(2026, 6, 21, 3, 0, 0, tzinfo=timezone.utc)
    light = light_at(51.5, -0.13, when)  # London near summer solstice, deep night
    assert light.elevation_deg >= 0.0


def test_scene_light_intensity_clamped_between_floor_and_one():
    for hour in range(24):
        when = datetime(2026, 6, 21, hour, 0, 0, tzinfo=timezone.utc)
        light = light_at(35.0, -106.0, when)
        assert 0.15 <= light.intensity <= 1.0


@pytest.mark.parametrize("zoom,expected", [(10, 0.4), (12, 0.4), (13, 0.2), (14, 0.0), (16, 0.0)])
def test_hillshade_exaggeration_zoom_curve(zoom, expected):
    assert hillshade_exaggeration(zoom) == pytest.approx(expected)


def test_sun_sweep_length_and_endpoints():
    start = datetime(2026, 3, 20, 10, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 3, 20, 16, 0, 0, tzinfo=timezone.utc)
    frames = sun_sweep(40.0, 0.0, start, end, frames=5)
    assert len(frames) == 5
    assert frames[0] == light_at(40.0, 0.0, start)
    assert frames[-1] == light_at(40.0, 0.0, end)


def test_sun_sweep_clock_is_monotonic_via_elevation_trend():
    start = datetime(2026, 3, 20, 6, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
    frames = sun_sweep(0.0, 0.0, start, end, frames=4)
    elevations = [f.elevation_deg for f in frames]
    # Sun climbing from morning toward solar noon at the equator: non-decreasing.
    assert elevations == sorted(elevations)


def test_sun_sweep_requires_at_least_two_frames():
    start = datetime(2026, 3, 20, 6, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    with pytest.raises(ValueError):
        sun_sweep(0.0, 0.0, start, end, frames=1)


def test_light_for_pose_uses_route_time_when_present():
    pose_time = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
    fallback = datetime(2026, 3, 20, 0, 0, 0, tzinfo=timezone.utc)
    light = light_for_pose(pose_time, 40.0, 0.0, fallback=fallback)
    assert light == light_at(40.0, 0.0, pose_time)


def test_light_for_pose_uses_fallback_when_pose_time_is_none():
    fallback = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
    light = light_for_pose(None, 40.0, 0.0, fallback=fallback)
    assert light == light_at(40.0, 0.0, fallback)


def test_scene_light_to_scene_serializes_for_scene_js():
    light = SceneLight(azimuth_deg=270.0, elevation_deg=30.0, intensity=0.8, night=False)
    assert light.to_scene(zoom=11.0) == {
        "azimuth": 270.0,
        "elevation": 30.0,
        "intensity": 0.8,
        "night": False,
        "hillshadeExaggeration": hillshade_exaggeration(11.0),
    }


def test_scene_light_to_scene_night_keeps_the_given_low_intensity():
    light = SceneLight(azimuth_deg=10.0, elevation_deg=0.0, intensity=0.15, night=True)
    scene_dict = light.to_scene(zoom=14.0)
    assert scene_dict["night"] is True
    assert scene_dict["intensity"] == 0.15
