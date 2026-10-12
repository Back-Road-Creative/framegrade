"""Frame-addressable Route Story timeline contract."""

import pytest

from gomoveshift.pipeline.route_story.timeline import Scene, StoryTimeline


def test_photo_hold_freezes_recording_time_while_presentation_frames_advance():
    timeline = StoryTimeline(
        fps=10,
        scenes=(
            Scene("route_follow", 0, 10, 0.0, 10.0),
            Scene("photo_hold", 10, 20, 10.0, 10.0),
            Scene("route_follow", 20, 30, 10.0, 20.0),
        ),
    )

    assert timeline.route_time_at_frame(0) == 0.0
    assert timeline.route_time_at_frame(9) == 9.0
    assert timeline.route_time_at_frame(10) == 10.0
    assert timeline.route_time_at_frame(19) == 10.0
    assert timeline.route_time_at_frame(20) == 10.0
    assert timeline.route_time_at_frame(29) == 19.0


def test_moving_cutaway_advances_to_its_true_route_end_time():
    timeline = StoryTimeline(
        fps=30,
        scenes=(Scene("source_video", 0, 60, 100.0, 112.0),),
    )

    assert timeline.route_time_at_frame(0) == 100.0
    assert timeline.route_time_at_frame(30) == 106.0
    assert timeline.route_time_at_frame(59) == pytest.approx(111.8)
    assert timeline.route_time_at_frame(60) == 112.0


def test_scene_boundaries_are_half_open_and_cannot_overlap_or_leave_frame_holes():
    with pytest.raises(ValueError, match="contiguous"):
        StoryTimeline(
            fps=30,
            scenes=(
                Scene("route_follow", 0, 10, 0.0, 1.0),
                Scene("route_follow", 11, 20, 1.0, 2.0),
            ),
        )

    with pytest.raises(ValueError, match="positive"):
        Scene("route_follow", 10, 10, 0.0, 1.0)


def test_frame_after_story_returns_final_recording_time_for_recap_export():
    timeline = StoryTimeline(
        fps=10,
        scenes=(Scene("recap", 0, 10, 50.0, 50.0),),
    )

    assert timeline.route_time_at_frame(10) == 50.0


def test_overview_freezes_source_time_before_route_follow_begins():
    timeline = StoryTimeline(
        fps=30,
        scenes=(
            Scene("overview", 0, 300, 0.0, 0.0),
            Scene("route_follow", 300, 600, 0.0, 60.0),
        ),
    )

    assert timeline.route_time_at_frame(299) == 0.0
    assert timeline.route_time_at_frame(300) == 0.0
