"""Route start/end privacy trim: drop points near the route's endpoints along
the actual travelled track, never by wall-clock/index alone. See
``docs/pipeline/150-route-story-route-privacy.md``.

Wired into ``route-story-film``; see ``tests/test_route_story_film.py``'s
trim-flag tests and ``tests/test_route_story_public_receipt.py``.
"""

import pytest

from gomoveshift.pipeline.route_story.route_privacy import (
    PrivacyZone,
    RouteTrim,
    RouteTrimError,
    mask_zones,
    receipt_fragment,
    trim_route,
)
from route_story_fixtures import driving_points, segments_with_gap


def test_no_op_trim_returns_input_unchanged():
    segments = [driving_points(5)]
    result = trim_route(segments, RouteTrim())
    assert result.segments is segments
    assert result.removed_start_points == 0
    assert result.removed_end_points == 0
    assert result.removed_start_m == 0.0
    assert result.removed_end_m == 0.0


def test_start_trim_removes_points_inside_radius():
    segment = driving_points(10, speed_mps=20.0, dt=1.0)
    segments = [segment]

    result = trim_route(segments, RouteTrim(start_m=25.0))

    assert result.removed_start_points == 2
    assert result.removed_end_points == 0
    assert result.removed_start_m == 40.0
    first_remaining = result.segments[0][0]
    assert first_remaining is segment[2]


def test_end_trim_removes_points_inside_radius():
    segment = driving_points(10, speed_mps=20.0, dt=1.0)
    segments = [segment]

    result = trim_route(segments, RouteTrim(end_m=25.0))

    assert result.removed_end_points == 2
    assert result.removed_start_points == 0
    assert result.removed_end_m == 40.0
    last_remaining = result.segments[-1][-1]
    assert last_remaining is segment[7]


def test_gapped_first_segment_entirely_inside_start_dropped_whole():
    segments = segments_with_gap(100.0, speed_mps=20.0, n=5)

    result = trim_route(segments, RouteTrim(start_m=90.0))

    assert len(result.segments) == 1
    assert result.segments[0] == segments[1][1:]
    assert result.removed_start_points == 6


def test_over_trim_raises_with_route_length_in_message():
    segments = [driving_points(5, speed_mps=20.0, dt=1.0)]

    with pytest.raises(RouteTrimError, match=r"80"):
        trim_route(segments, RouteTrim(start_m=100.0))


def test_result_never_contains_a_point_absent_from_input():
    segment = driving_points(10, speed_mps=20.0, dt=1.0)
    segments = [segment]
    known_ids = {id(point) for point in segment}

    result = trim_route(segments, RouteTrim(start_m=25.0, end_m=25.0))

    for kept_segment in result.segments:
        for point in kept_segment:
            assert id(point) in known_ids


def test_receipt_fragment_shape():
    segment = driving_points(10, speed_mps=20.0, dt=1.0)
    result = trim_route([segment], RouteTrim(start_m=25.0, end_m=25.0))

    fragment = receipt_fragment(result)

    assert fragment == {
        "route_trim": {
            "start_m": 25.0,
            "end_m": 25.0,
            "removed_start_points": result.removed_start_points,
            "removed_end_points": result.removed_end_points,
            "removed_start_m": result.removed_start_m,
            "removed_end_m": result.removed_end_m,
        }
    }


def test_mid_route_zone_splits_one_segment_into_two_kept_segments():
    segment = driving_points(10, speed_mps=20.0, dt=1.0)
    # Points 4 and 5 are 20m apart; a zone centered between them with a 15m
    # radius reaches both (10m away) but not point 3 or point 6 (30m away).
    zone = PrivacyZone(
        center_lat=(segment[4].latitude + segment[5].latitude) / 2,
        center_lon=segment[4].longitude,
        radius_m=15.0,
    )

    result = mask_zones([segment], [zone])

    assert len(result.segments) == 2
    assert result.segments[0] == segment[0:4]
    assert result.segments[1] == segment[6:10]
    assert result.removed_points == 2
    assert result.removed_m == pytest.approx(20.0)
    assert result.zone_count == 1


def test_zone_spanning_an_entire_short_segment_drops_it_whole():
    short_segment = driving_points(2, speed_mps=20.0, dt=1.0, start_lat=0.0, start_lon=0.0)
    far_segment = driving_points(5, speed_mps=20.0, dt=1.0, start_lat=10.0, start_lon=0.0)
    zone = PrivacyZone(
        center_lat=short_segment[0].latitude,
        center_lon=short_segment[0].longitude,
        radius_m=25.0,
    )

    result = mask_zones([short_segment, far_segment], [zone])

    assert len(result.segments) == 1
    assert result.segments[0] == far_segment
    assert result.removed_points == 2


def test_overlapping_zones_dont_double_count():
    segment = driving_points(10, speed_mps=20.0, dt=1.0)
    zone_a = PrivacyZone(
        center_lat=(segment[3].latitude + segment[4].latitude) / 2,
        center_lon=segment[3].longitude,
        radius_m=15.0,
    )
    zone_b = PrivacyZone(
        center_lat=(segment[4].latitude + segment[5].latitude) / 2,
        center_lon=segment[4].longitude,
        radius_m=15.0,
    )

    result = mask_zones([segment], [zone_a, zone_b])

    # Points 3, 4, and 5 are each within at least one zone -- three points
    # removed, never four (which double-counting point 4 would produce).
    assert result.removed_points == 3
    assert len(result.segments) == 2
    assert result.segments[0] == segment[0:3]
    assert result.segments[1] == segment[6:10]


def test_zone_covering_the_whole_route_raises_with_route_length_in_message():
    segment = driving_points(5, speed_mps=20.0, dt=1.0)
    zone = PrivacyZone(
        center_lat=segment[2].latitude, center_lon=segment[2].longitude, radius_m=1000.0
    )

    with pytest.raises(RouteTrimError, match=r"80"):
        mask_zones([segment], [zone])


def test_no_zones_returns_input_unchanged():
    segments = [driving_points(5)]

    result = mask_zones(segments, [])

    assert result.segments is segments
    assert result.removed_points == 0
    assert result.removed_m == 0.0
    assert result.zone_count == 0


def test_receipt_fragment_zones_carries_counts_only_never_center_or_radius():
    segment = driving_points(10, speed_mps=20.0, dt=1.0)
    trim_result = trim_route([segment], RouteTrim(start_m=25.0, end_m=25.0))
    zone = PrivacyZone(
        center_lat=(segment[4].latitude + segment[5].latitude) / 2,
        center_lon=segment[4].longitude,
        radius_m=15.0,
    )
    zone_result = mask_zones(trim_result.segments, [zone])

    fragment = receipt_fragment(trim_result, zone_result)

    assert fragment["route_trim"]["zones"] == {
        "count": 1,
        "removed_points": zone_result.removed_points,
        "removed_m": zone_result.removed_m,
    }
    for forbidden in ("center_lat", "center_lon", "radius_m", "centers", "radii"):
        assert forbidden not in fragment["route_trim"]["zones"]


@pytest.mark.parametrize(
    ("lat", "lon", "radius_m"),
    [(95.0, 10.0, 100.0), (10.0, 200.0, 100.0), (10.0, 10.0, 0.0), (10.0, 10.0, -5.0)],
)
def test_privacy_zone_refuses_an_impossible_centre_or_radius(lat, lon, radius_m):
    # A swapped LAT,LON or a zero radius would otherwise mask nothing, silently.
    with pytest.raises(ValueError):
        PrivacyZone(center_lat=lat, center_lon=lon, radius_m=radius_m)
