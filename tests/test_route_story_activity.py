"""The explicit Route Story activity contract."""

import dataclasses

import pytest

from gomoveshift.pipeline.route_story.activity import (
    ACTIVITY_PROFILES,
    MAPPING_VERSION,
    ActivityProfile,
    region_units,
    resolve_activity,
    resolve_units,
    track_centroid,
)
from gomoveshift.pipeline.route_story.metrics import RoutePoint


def test_explicit_activity_wins_regardless_of_observed_speed():
    profile = resolve_activity("hiking", observed_max_speed_mps=40.0)
    assert profile == ACTIVITY_PROFILES["hiking"]


def test_auto_falls_back_to_hiking_speed_band():
    profile = resolve_activity("auto", observed_max_speed_mps=2.0)
    assert profile.name == "hiking"


def test_none_activity_falls_back_to_cycling_speed_band():
    profile = resolve_activity(None, observed_max_speed_mps=7.0)
    assert profile.name == "cycling"


def test_no_observed_speed_falls_back_to_driving():
    profile = resolve_activity(None, observed_max_speed_mps=None)
    assert profile.name == "driving"


def test_high_observed_speed_falls_back_to_driving():
    profile = resolve_activity("auto", observed_max_speed_mps=25.0)
    assert profile.name == "driving"


def test_unknown_explicit_activity_raises():
    with pytest.raises(ValueError):
        resolve_activity("unicycling", observed_max_speed_mps=5.0)


def test_profile_carries_hud_cap_and_grade_cap_but_no_units_default():
    profile = ACTIVITY_PROFILES["cycling"]
    assert isinstance(profile, ActivityProfile)
    assert profile.hud_speed_cap_mps > profile.max_speed_mps
    assert profile.grade_cap_percent > 0
    # Units come from resolve_units (explicit value or region), never the profile.
    assert "default_units" not in {f.name for f in dataclasses.fields(ActivityProfile)}
    assert not hasattr(profile, "default_units")


def test_mapping_version_is_a_non_empty_string():
    assert isinstance(MAPPING_VERSION, str)
    assert MAPPING_VERSION


def test_mapping_version_bumped_after_profile_field_removal():
    # ACTIVITY_PROFILES changed shape (default_units dropped), so the receipt
    # version must differ from the one that produced the old table.
    assert MAPPING_VERSION != "2026-09-25.1"


def test_region_units_dublin_is_metric():
    assert region_units(53.35, -6.26) == "metric"


def test_region_units_boulder_is_imperial():
    assert region_units(40.0, -105.3) == "imperial"


def test_region_units_london_is_imperial():
    assert region_units(51.5, -0.1) == "imperial"


def test_resolve_units_explicit_wins_over_region():
    units, source = resolve_units("metric", centroid=(40.0, -105.3))
    assert (units, source) == ("metric", "explicit")


def test_resolve_units_auto_uses_region():
    units, source = resolve_units("auto", centroid=(53.35, -6.26))
    assert (units, source) == ("metric", "region")


def test_resolve_units_none_uses_region():
    units, source = resolve_units(None, centroid=(40.0, -105.3))
    assert (units, source) == ("imperial", "region")


def test_resolve_units_no_centroid_defaults_to_metric():
    units, source = resolve_units("auto", centroid=None)
    assert (units, source) == ("metric", "default")


def test_resolve_units_unknown_explicit_raises():
    with pytest.raises(ValueError):
        resolve_units("furlongs", centroid=(40.0, -105.3))


def test_track_centroid_averages_all_points_across_segments():
    segments = [
        [RoutePoint(0, 40.0, -105.0), RoutePoint(10, 42.0, -105.0)],
        [RoutePoint(20, 41.0, -103.0)],
    ]
    centroid = track_centroid(segments)
    assert centroid == pytest.approx((41.0, -104.33333333), abs=1e-6)


def test_track_centroid_empty_segments_is_none():
    assert track_centroid([]) is None
    assert track_centroid([[]]) is None
