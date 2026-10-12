"""USA satellite opening camera and timeline contracts."""

import pytest

from gomoveshift.pipeline.route_story.opening import CameraPose, SatelliteOpening
from gomoveshift.pipeline.route_story.timeline import Scene


def test_overview_scene_freezes_source_recording_time():
    with pytest.raises(ValueError, match="freeze"):
        Scene("overview", 0, 300, 10.0, 20.0)


def test_us_satellite_opening_eases_from_national_view_to_visible_route_start():
    opening = SatelliteOpening.for_route_start(
        latitude=35.123,
        longitude=-80.456,
        start_frame=0,
        end_frame=300,
        final_zoom=12.0,
    )

    assert opening.pose_at_frame(0) == opening.start
    assert opening.pose_at_frame(300) == CameraPose(-80.456, 35.123, 12.0, 0.0, 0.0)
    middle = opening.pose_at_frame(150)
    assert opening.start.zoom < middle.zoom < opening.end.zoom
    assert opening.start.longitude < middle.longitude < opening.end.longitude


def test_opening_rejects_invalid_target_coordinates_or_duration():
    with pytest.raises(ValueError, match="latitude"):
        SatelliteOpening.for_route_start(
            latitude=91.0, longitude=-80.0, start_frame=0, end_frame=300, final_zoom=12.0
        )

    with pytest.raises(ValueError, match="positive"):
        SatelliteOpening.for_route_start(
            latitude=35.0, longitude=-80.0, start_frame=10, end_frame=10, final_zoom=12.0
        )
