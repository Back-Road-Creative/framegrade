"""Route Story media anchors stay evidence-backed and privacy-aware."""

from datetime import datetime, timezone

import pytest

from gomoveshift.pipeline.route_story.anchors import (
    MediaEvidence,
    anchor_media,
    anchor_media_to_segments,
    calibrate_clock_offset_s,
    capture_start_from_filename,
    captured_at_route_seconds,
    place_media,
)
from gomoveshift.pipeline.route_story.metrics import RoutePoint


def _point(t, lat, lon):
    return RoutePoint(time_s=t, latitude=lat, longitude=lon, elevation_m=100.0)


def test_embedded_coordinates_win_over_timestamp_matching():
    anchored = anchor_media(
        MediaEvidence("photo.jpg", captured_at_s=10.0, latitude=35.0, longitude=-80.0),
        [_point(0, 0.0, 0.0), _point(20, 0.0, 0.02)],
    )

    assert anchored.status == "embedded_coordinates"
    assert (anchored.latitude, anchored.longitude) == (35.0, -80.0)
    assert anchored.track_time_s is None


def test_timestamp_anchor_interpolates_only_a_continuous_track_edge():
    anchored = anchor_media(
        MediaEvidence("clip.mp4", captured_at_s=5.0),
        [_point(0, 0.0, 0.0), _point(10, 0.0, 0.01)],
    )

    assert anchored.status == "timestamp_interpolated"
    assert anchored.track_time_s == 5.0
    assert anchored.latitude == 0.0
    assert anchored.longitude == 0.005


def test_timestamp_in_an_unobserved_gap_is_not_fabricated_as_a_location():
    anchored = anchor_media(
        MediaEvidence("clip.mp4", captured_at_s=30.0),
        [_point(0, 0.0, 0.0), _point(60, 0.0, 0.06)],
        max_interpolation_gap_s=15.0,
    )

    assert anchored.status == "unanchored_gap"
    assert anchored.latitude is None
    assert anchored.longitude is None


def test_timestamp_outside_the_recording_is_unanchored():
    anchored = anchor_media(MediaEvidence("photo.jpg", captured_at_s=99.0), [_point(0, 0.0, 0.0)])

    assert anchored.status == "unanchored_outside_track"
    assert anchored.latitude is None


def test_private_anchor_omits_coordinates_for_any_export_surface():
    anchored = anchor_media(
        MediaEvidence("photo.jpg", captured_at_s=5.0),
        [_point(0, 0.0, 0.0), _point(10, 0.0, 0.01)],
        include_coordinates=False,
    )

    assert anchored.status == "timestamp_interpolated"
    assert anchored.latitude is None
    assert anchored.longitude is None


def test_capture_time_with_explicit_offset_maps_to_the_route_source_clock():
    source_start = datetime(2026, 9, 16, 14, 40, 49, tzinfo=timezone.utc)
    capture = datetime.fromisoformat("2026-09-16T11:19:39-04:00")

    assert captured_at_route_seconds(capture, source_start) == 2330.0


def test_capture_time_without_an_explicit_timezone_is_refused_for_route_matching():
    source_start = datetime(2026, 9, 16, 14, 40, 49, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="timezone"):
        captured_at_route_seconds(datetime(2026, 9, 16, 11, 19, 39), source_start)


def test_timestamp_anchor_searches_all_selected_segments():
    anchored = anchor_media_to_segments(
        MediaEvidence("clip.mp4", captured_at_s=105.0),
        [
            [_point(0, 0.0, 0.0), _point(10, 0.0, 0.01)],
            [_point(100, 1.0, 1.0), _point(110, 1.0, 1.01)],
        ],
        max_interpolation_gap_s=15,
    )
    assert anchored.status == "timestamp_interpolated"
    assert anchored.track_time_s == 105.0
    assert anchored.latitude == 1.0


def test_a_physically_impossible_embedded_fix_is_rejected_for_the_clock_position():
    # 40,-70 is thousands of km from this track in 5 s of travel -- no longer
    # merely flagged for review, it is rejected outright for the clock.
    anchored = anchor_media_to_segments(
        MediaEvidence("clip.mp4", captured_at_s=5.0, latitude=40.0, longitude=-70.0),
        [[_point(0, 0.0, 0.0), _point(10, 0.0, 0.01)]],
        max_interpolation_gap_s=15,
        coordinate_conflict_distance_m=1000,
    )
    assert anchored.status == "timestamp_interpolated"
    assert (anchored.latitude, anchored.longitude) == (0.0, 0.005)
    assert anchored.track_time_s == 5.0
    assert (anchored.rejected_latitude, anchored.rejected_longitude) == (40.0, -70.0)
    assert anchored.required_speed_mps is not None and anchored.required_speed_mps > 1000.0


def test_segments_refuse_a_non_positive_coordinate_conflict_distance():
    with pytest.raises(ValueError, match="coordinate_conflict_distance_m"):
        anchor_media_to_segments(
            MediaEvidence("clip.mp4", latitude=40.0, longitude=-70.0),
            [[_point(0, 0.0, 0.0), _point(10, 0.0, 0.01)]],
            coordinate_conflict_distance_m=0,
        )


def test_segments_keep_searching_past_a_gap_and_report_it_when_nothing_else_matches():
    anchored = anchor_media_to_segments(
        MediaEvidence("clip.mp4", captured_at_s=50.0),
        [[_point(0, 0.0, 0.0), _point(100, 0.0, 0.1)]],
        max_interpolation_gap_s=15,
    )

    assert anchored.status == "unanchored_gap"
    assert anchored.latitude is None
    assert anchored.track_time_s is None


def test_placement_keeps_certain_time_for_a_photo_stranded_in_a_gap_with_an_isolated_fix():
    # A DSC-numbered camera file: the capture time falls in a 300 s gap far wider than the
    # interpolation edge cap, but an isolated real fix 5 s later still gives a
    # (uncertain) position without ever discarding the certain capture time.
    track = [_point(0, 0.0, 0.0), _point(300, 0.0, 0.3)]
    isolated = _point(155, 0.0, 0.1505)
    placement = place_media(
        MediaEvidence("DSC_0001.NEF", captured_at_s=150.0),
        track,
        observations=[*track, isolated],
        max_interpolation_gap_s=15.0,
        nearest_observation_radius_s=30.0,
    )

    assert placement.timeline_s == 150.0
    assert placement.position_source == "nearest_observation"
    assert (placement.latitude, placement.longitude) == (0.0, 0.1505)
    assert placement.radius_s == 5.0
    assert placement.conflict is False


def test_placement_resolves_a_stale_embedded_fix_to_the_corroborated_clock_position():
    # Phone-clip-shaped: embedded GPS disagrees with the clock-derived track
    # position by far more than the safety radius while the clock is
    # corroborated by a nearby observed edge, so WHERE defers to the clock and
    # the conflict is recorded rather than silently dropped.
    track = [_point(0, 0.0, 0.0), _point(10, 0.0, 0.01)]
    placement = place_media(
        MediaEvidence(
            "PXL_20250115_120319456.mp4", captured_at_s=5.0, latitude=40.0, longitude=-70.0
        ),
        track,
        coordinate_conflict_distance_m=1_000.0,
        conflict_window_s=600.0,
    )

    assert placement.timeline_s == 5.0
    assert placement.position_source == "clock_interpolated"
    assert (placement.latitude, placement.longitude) == (0.0, 0.005)
    assert placement.conflict is True
    assert (placement.conflict_latitude, placement.conflict_longitude) == (40.0, -70.0)


def test_placement_reports_no_conflict_when_embedded_gps_agrees_with_the_clock():
    track = [_point(0, 0.0, 0.0), _point(10, 0.0, 0.01)]
    placement = place_media(
        MediaEvidence("agrees.jpg", captured_at_s=5.0, latitude=0.0, longitude=0.005),
        track,
    )

    assert placement.conflict is False
    assert placement.position_source == "embedded_gps"
    assert (placement.latitude, placement.longitude) == (0.0, 0.005)


def test_calibrate_clock_offset_from_one_corroborated_phone_still():
    track = [_point(0, 0.0, 0.0), _point(50, 0.0, 0.05), _point(100, 0.0, 0.1)]
    still = MediaEvidence("phone.jpg", captured_at_s=13.0, latitude=0.0, longitude=0.05)

    assert calibrate_clock_offset_s(still, track) == 37.0


def test_calibrate_clock_offset_refuses_an_uncorroborated_still():
    track = [_point(0, 0.0, 0.0), _point(100, 0.0, 0.1)]
    still = MediaEvidence("phone.jpg", captured_at_s=13.0, latitude=5.0, longitude=5.0)

    with pytest.raises(ValueError, match="corroborate"):
        calibrate_clock_offset_s(still, track)


def test_placement_applies_a_calibrated_body_clock_offset():
    track = [_point(0, 0.0, 0.0), _point(50, 0.0, 0.05), _point(100, 0.0, 0.1)]
    offset_s = calibrate_clock_offset_s(
        MediaEvidence("phone.jpg", captured_at_s=13.0, latitude=0.0, longitude=0.05), track
    )

    placement = place_media(
        MediaEvidence("DSC_body.NEF", captured_at_s=63.0),
        track,
        max_interpolation_gap_s=60.0,
        clock_offset_s=offset_s,
        clock_source="phone.jpg",
    )

    assert offset_s == 37.0
    assert placement.timeline_s == 100.0
    assert placement.clock_offset_s == 37.0
    assert placement.clock_source == "phone.jpg"


def test_capture_start_from_pixel_filename_is_utc_from_the_filename_not_the_container():
    # Container creation_time is file-finalisation time (12:03:40 here); the
    # Pixel filename carries the actual recording start (12:03:19.456 UTC).
    started = capture_start_from_filename("PXL_20250115_120319456.mp4")

    assert started == datetime(2025, 1, 15, 12, 3, 19, 456000, tzinfo=timezone.utc)


def test_capture_start_from_filename_is_none_for_a_non_matching_name():
    assert capture_start_from_filename("DSC_0001.NEF") is None


def test_stationary_edge_rule_interpolates_a_wide_but_short_rest_stop_gap():
    # A 246 s edge over 30 m of ground -- a rest stop, not a device outage.
    track = [_point(1000.0, 0.0, 0.0), _point(1246.0, 0.0, 0.0002698)]
    anchored = anchor_media(
        MediaEvidence("photo.jpg", captured_at_s=1242.0),
        track,
        max_interpolation_gap_s=60.0,
        max_interpolation_gap_m=150.0,
    )

    assert anchored.status == "timestamp_interpolated"
    assert anchored.interpolation_method == "stationary_edge"
    assert anchored.latitude is not None


def test_stationary_edge_rule_does_not_rescue_a_wide_edge_on_the_ground_too():
    # Same 246 s edge, but 800 m of real ground -- a genuine device outage.
    track = [_point(1000.0, 0.0, 0.0), _point(1246.0, 0.0, 0.007194)]
    anchored = anchor_media(
        MediaEvidence("photo.jpg", captured_at_s=1242.0),
        track,
        max_interpolation_gap_s=60.0,
        max_interpolation_gap_m=150.0,
    )

    assert anchored.status == "unanchored_gap"
    assert anchored.interpolation_method is None


def test_segments_reject_an_impossible_embedded_gps_fix_for_a_certain_clock_time():
    # Walking-speed track; embedded GPS ~600 km away would need ~23 km/s to
    # reach from the nearest track fix 24 s away -- physically impossible.
    track = [[_point(0.0, 0.0, 0.0), _point(48.0, 0.0, 0.0005), _point(83.0, 0.0, 0.0009)]]
    anchored = anchor_media_to_segments(
        MediaEvidence("PXL_stale.mp4", captured_at_s=24.0, latitude=0.0, longitude=5.0),
        track,
        max_interpolation_gap_s=15.0,
        max_interpolation_gap_m=150.0,
        coordinate_conflict_distance_m=1_000.0,
    )

    assert anchored.status == "timestamp_interpolated"
    assert anchored.track_time_s == 24.0
    assert anchored.latitude == pytest.approx(0.0)
    assert anchored.longitude == pytest.approx(0.00025)
    assert anchored.interpolation_method == "stationary_edge"
    assert (anchored.rejected_latitude, anchored.rejected_longitude) == (0.0, 5.0)
    assert anchored.required_speed_mps is not None
    assert anchored.required_speed_mps > 1000.0


def test_segments_keep_a_plausible_conflict_on_a_driving_speed_track():
    # 25 m/s driving track; a phone fix ~1.5 km off only needs ~55 m/s from the
    # nearest fix -- reachable for a car, so the embedded GPS is kept and
    # merely flagged for review, exactly like before.
    track = [[_point(0.0, 0.0, 0.0), _point(100.0, 0.0, 0.022486)]]
    anchored = anchor_media_to_segments(
        MediaEvidence("phone.jpg", captured_at_s=50.0, latitude=0.0, longitude=0.024732),
        track,
        max_interpolation_gap_s=100.0,
        max_interpolation_gap_m=150.0,
        coordinate_conflict_distance_m=1_000.0,
    )

    assert anchored.status == "coordinate_time_conflict"
    assert (anchored.latitude, anchored.longitude) == (0.0, 0.024732)
    assert anchored.rejected_latitude is None
    assert anchored.required_speed_mps is None


def test_calibrate_clock_offset_is_not_measurable_without_embedded_gps():
    track = [_point(0, 0.0, 0.0), _point(100, 0.0, 0.1)]
    still = MediaEvidence("no_gps.jpg", captured_at_s=13.0)

    with pytest.raises(ValueError, match="embedded GPS"):
        calibrate_clock_offset_s(still, track)


def test_placement_uses_the_stationary_edge_rule_for_a_rest_stop_gap():
    track = [_point(1000.0, 0.0, 0.0), _point(1246.0, 0.0, 0.0002698)]
    placement = place_media(
        MediaEvidence("photo.jpg", captured_at_s=1242.0),
        track,
        max_interpolation_gap_s=60.0,
        max_interpolation_gap_m=150.0,
    )

    assert placement.position_source == "clock_interpolated"
    assert placement.interpolation_method == "stationary_edge"
    assert placement.timeline_s == 1242.0


def test_placement_rejects_an_impossible_embedded_gps_fix():
    track = [_point(0.0, 0.0, 0.0), _point(48.0, 0.0, 0.0005), _point(83.0, 0.0, 0.0009)]
    placement = place_media(
        MediaEvidence("PXL_stale.mp4", captured_at_s=24.0, latitude=0.0, longitude=5.0),
        track,
        max_interpolation_gap_s=15.0,
        max_interpolation_gap_m=150.0,
        coordinate_conflict_distance_m=1_000.0,
    )

    assert placement.conflict is True
    assert placement.conflict_rejected is True
    assert placement.position_source == "clock_interpolated"
    assert placement.interpolation_method == "stationary_edge"
    assert (placement.conflict_latitude, placement.conflict_longitude) == (0.0, 5.0)
    assert placement.required_speed_mps is not None
    assert placement.required_speed_mps > 1000.0
