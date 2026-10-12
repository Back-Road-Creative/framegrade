"""Film map legs rendered through the offline MapLibre terrain scene."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.route_story import film as film_module
from gomoveshift.pipeline.route_story.layers import PillowMapRenderer, _point_at_time
from gomoveshift.pipeline.route_story.metrics import RoutePoint, distance_m
from gomoveshift.pipeline.route_story.opening import CameraPose
from gomoveshift.pipeline.route_story.planner import (
    DEFAULT_MOTION_LIMITS,
    MotionReport,
    Shot,
    ShotPlan,
    _shortest_angle_delta,
    assert_smooth,
    motion_report,
)
from gomoveshift.pipeline.route_story import scene_renderer as scene_renderer_module
from gomoveshift.pipeline.route_story.satellite import SatelliteRaster
from gomoveshift.pipeline.route_story.scene_renderer import (
    SceneMapRenderer,
    _chord_bearing,
    _target_bearing,
)

WIDTH, HEIGHT = 320, 180
MAP_HEIGHT = int(HEIGHT * 0.855)
TRIP_EPOCH = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


class _FakeCapture:
    """Records every pose it is asked to render and returns a solid frame."""

    def __init__(
        self,
        width: int,
        height: int,
        color: tuple[int, int, int] = (80, 90, 100),
    ):
        self.width = width
        self.height = height
        self.color = color
        self.poses: list[dict] = []

    def render(self, pose, light=None):
        self.poses.append(pose)
        array = np.full((self.height, self.width, 3), self.color, dtype=np.uint8)
        return array, []


def _straight_segments(count: int = 60, step_s: float = 1.0) -> list[list[RoutePoint]]:
    return [
        [
            RoutePoint(i * step_s, 35.0 + i * 0.0004, -80.0 + i * 0.0006, 100.0 + i)
            for i in range(count)
        ]
    ]


def _out_and_back_segments(
    total_s: float = 14_000.0, step_s: float = 1.0, jitter_deg: float = 15.0, seed: int = 3
) -> list[list[RoutePoint]]:
    """A real-trip-scale out-and-back route: a 180 deg reversal at the midpoint,
    plus per-point GPS jitter -- the shape that exposed the uncapped bearing bug."""
    rng = np.random.default_rng(seed)
    count = int(total_s / step_s) + 1
    lon, lat = -80.0, 35.0
    points = []
    for i in range(count):
        t = i * step_s
        base_bearing = 90.0 if t <= total_s / 2 else 270.0
        bearing = base_bearing + rng.uniform(-jitter_deg, jitter_deg)
        lat += math.cos(math.radians(bearing)) * 0.00003
        lon += math.sin(math.radians(bearing)) * 0.00003
        points.append(RoutePoint(t, lat, lon, 100.0))
    return [points]


def _shot_plan_for(poses: list[dict]) -> ShotPlan:
    camera_poses = [
        CameraPose(0.0, 0.0, pose["zoom"], pose["pitch"], pose["bearing"]) for pose in poses
    ]
    shot = Shot("follow", 0, len(camera_poses), lambda frame: camera_poses[frame])
    return ShotPlan((shot,))


def _motion_report_for(poses: list[dict], fps: float) -> MotionReport:
    return motion_report(_shot_plan_for(poses), fps)


def test_pose_centre_tracks_route_position_at_time():
    segments = _straight_segments()
    capture = _FakeCapture(WIDTH, MAP_HEIGHT)
    renderer = SceneMapRenderer(
        capture=capture, trip_epoch=TRIP_EPOCH, max_zoom=16.0, follow_zoom=14.0
    )

    for at_time in (0.0, 10.5, 30.0, 58.9):
        renderer.render(segments, at_time, WIDTH, HEIGHT)

    assert len(capture.poses) == 4
    for pose, at_time in zip(capture.poses, (0.0, 10.5, 30.0, 58.9)):
        expected = _point_at_time(segments, at_time)
        assert expected is not None
        assert pose["longitude"] == pytest.approx(expected.longitude, abs=1e-6)
        assert pose["latitude"] == pytest.approx(expected.latitude, abs=1e-6)


def test_render_leg_follows_the_route_within_every_motion_limit_on_a_compressed_real_trip_shape():
    # 14000 s of route compressed into a 14 s / 30 fps map leg (1000x), on an
    # out-and-back route with GPS jitter -- the shape that drives an uncapped follow
    # camera far past the bearing speed cap, and a speed-only limiter into pinning at
    # the cap.
    total_s = 14_000.0
    segments = _out_and_back_segments(total_s=total_s)
    fps = 30.0
    frame_count = 420
    times = [total_s * (index + 1) / frame_count for index in range(frame_count)]

    capture = _FakeCapture(WIDTH, MAP_HEIGHT)
    renderer = SceneMapRenderer(
        capture=capture, trip_epoch=TRIP_EPOCH, max_zoom=16.0, follow_zoom=14.0, fps=fps
    )

    list(renderer.render_leg(segments, times, WIDTH, HEIGHT))

    report = _motion_report_for(capture.poses, fps)
    assert_smooth(_shot_plan_for(capture.poses), fps)

    speed_cap = DEFAULT_MOTION_LIMITS["bearing_speed_deg_s"]
    at_cap = sum(1 for value in report.bearing_speed_deg_s if abs(abs(value) - speed_cap) < 1e-3)
    frac_at_cap = at_cap / len(report.bearing_speed_deg_s)
    assert frac_at_cap < 0.20, f"{frac_at_cap:.2%} of frames pinned at the speed cap"

    window_s = max(
        scene_renderer_module._MIN_CHORD_WINDOW_S,
        scene_renderer_module._CHORD_WINDOW_FRACTION * (times[-1] - times[0]),
    )
    chord_direction = scene_renderer_module._target_bearing(
        segments, times[0], scene_renderer_module._MIN_BEARING_WINDOW_S
    )
    lags = []
    for at_time, bearing in zip(times, (pose["bearing"] for pose in capture.poses)):
        chord_direction = _chord_bearing(
            segments, at_time, window_s, times[0], times[-1], chord_direction
        )
        lags.append(abs(_shortest_angle_delta(bearing, chord_direction)))
    lags.sort()
    median_lag = lags[len(lags) // 2]
    assert median_lag < 30.0, f"median heading lag {median_lag:.1f} deg vs the chord direction"


def test_render_leg_carries_bearing_continuity_across_legs():
    total_s = 14_000.0
    segments = _out_and_back_segments(total_s=total_s)
    fps = 30.0
    frame_count = 210
    first_times = [total_s * (index + 1) / (2 * frame_count) for index in range(frame_count)]
    second_start = first_times[-1]
    second_times = [
        second_start + (total_s - second_start) * (index + 1) / frame_count
        for index in range(frame_count)
    ]

    capture = _FakeCapture(WIDTH, MAP_HEIGHT)
    renderer = SceneMapRenderer(
        capture=capture, trip_epoch=TRIP_EPOCH, max_zoom=16.0, follow_zoom=14.0, fps=fps
    )

    list(renderer.render_leg(segments, first_times, WIDTH, HEIGHT))
    joined_velocity = renderer._last_bearing_velocity
    list(renderer.render_leg(segments, second_times, WIDTH, HEIGHT))

    # The second leg starts from the first leg's own end bearing AND angular
    # velocity -- carrying only the bearing would let the join instantly zero
    # the velocity, which is itself an unbounded (infinite) deceleration.
    assert joined_velocity != 0.0
    assert_smooth(_shot_plan_for(capture.poses), fps)


def test_rendered_poses_accumulate_across_legs_in_render_order():
    total_s = 14_000.0
    segments = _out_and_back_segments(total_s=total_s)
    fps = 30.0
    frame_count = 20
    first_times = [total_s * (index + 1) / (2 * frame_count) for index in range(frame_count)]
    second_start = first_times[-1]
    second_times = [
        second_start + (total_s - second_start) * (index + 1) / frame_count
        for index in range(frame_count)
    ]

    capture = _FakeCapture(WIDTH, MAP_HEIGHT)
    renderer = SceneMapRenderer(
        capture=capture, trip_epoch=TRIP_EPOCH, max_zoom=16.0, follow_zoom=14.0, fps=fps
    )

    list(renderer.render_leg(segments, first_times, WIDTH, HEIGHT))
    list(renderer.render_leg(segments, second_times, WIDTH, HEIGHT))

    assert len(renderer.rendered_poses) == 2 * frame_count
    assert [state.bearing_deg for state in renderer.rendered_poses] == [
        pose["bearing"] for pose in capture.poses
    ]


def test_dashboard_is_composited_over_the_captured_frame():
    segments = _straight_segments()
    color = (80, 90, 100)
    capture = _FakeCapture(WIDTH, MAP_HEIGHT, color=color)
    renderer = SceneMapRenderer(
        capture=capture, trip_epoch=TRIP_EPOCH, max_zoom=16.0, follow_zoom=14.0
    )

    frame = renderer.render(segments, 15.0, WIDTH, HEIGHT)

    assert isinstance(frame, Image.Image)
    assert frame.size == (WIDTH, HEIGHT)
    dashboard_pixel = frame.getpixel((WIDTH // 2, HEIGHT - 5))
    assert isinstance(dashboard_pixel, tuple)
    assert dashboard_pixel[:3] != color


def test_profile_cursor_is_drawn_in_the_profile_box():
    segment = [RoutePoint(i * 1.0, 35.0 + i * 0.0004, -80.0, 100.0 + i) for i in range(30)]
    segments = [segment]
    capture = _FakeCapture(WIDTH, MAP_HEIGHT)
    renderer = SceneMapRenderer(
        capture=capture, trip_epoch=TRIP_EPOCH, max_zoom=16.0, follow_zoom=14.0
    ).measuring(segments)

    frame = renderer.render(segments, 15.0, WIDTH, HEIGHT)

    layout = scene_renderer_module.hud_layout(WIDTH, HEIGHT)
    pixels = frame.load()
    cursor_color = (245, 183, 59)
    found = any(
        pixels[x, y][:3] == cursor_color
        for x in range(layout.profile_left, layout.profile_right)
        for y in range(int(layout.profile_top), int(layout.profile_top + layout.profile_height))
    )
    assert found


def test_zoom_never_exceeds_the_bundle_cap():
    segments = _straight_segments()
    capture = _FakeCapture(WIDTH, MAP_HEIGHT)
    renderer = SceneMapRenderer(
        capture=capture, trip_epoch=TRIP_EPOCH, max_zoom=11.5, follow_zoom=18.0
    )

    renderer.render(segments, 5.0, WIDTH, HEIGHT)

    assert capture.poses[0]["zoom"] <= 11.5


def test_render_leg_holds_dashboard_digits_steady_within_a_cadence_block(monkeypatch):
    # A long GPS clock compressed hundreds of times into a short map leg: without a
    # held sample instant, every dashboard digit changes on every output frame.
    segments = _straight_segments(count=900, step_s=1.0)
    fps = 30.0
    frame_count = 60
    trip_seconds = 700.0
    times = [trip_seconds * (index + 1) / frame_count for index in range(frame_count)]

    capture = _FakeCapture(WIDTH, MAP_HEIGHT)
    renderer = SceneMapRenderer(
        capture=capture, trip_epoch=TRIP_EPOCH, max_zoom=16.0, follow_zoom=14.0, fps=fps
    )

    recorded = []
    monkeypatch.setattr(
        scene_renderer_module,
        "_draw_dashboard",
        lambda canvas, dashboard, credit: recorded.append(dashboard),
    )

    list(renderer.render_leg(segments, times, WIDTH, HEIGHT))

    cadence_frames = round(scene_renderer_module._DASHBOARD_CADENCE_S * fps)
    assert cadence_frames > 1
    assert len(recorded) == frame_count
    # Every frame within the first cadence block reads identical digits...
    for index in range(1, cadence_frames):
        assert recorded[index] == recorded[0]
    # ...and the next block moves on.
    assert recorded[cadence_frames] != recorded[0]


def test_pillow_render_leg_matches_per_frame_render(tmp_path):
    raster_path = tmp_path / "sat.jpg"
    Image.new("RGB", (200, 200), "#335577").save(raster_path)
    raster = SatelliteRaster(raster_path, -80.2, 34.8, -79.8, 35.2, "Test")
    renderer = PillowMapRenderer(rasters=(raster,))
    segments = _straight_segments()
    times = [5.0, 10.0, 15.0]

    looped = [renderer.render(segments, at_time, WIDTH, HEIGHT) for at_time in times]
    via_leg = list(renderer.render_leg(segments, times, WIDTH, HEIGHT))

    assert len(via_leg) == len(looped)
    for one, other in zip(looped, via_leg):
        assert list(one.getdata()) == list(other.getdata())


def test_film_cli_requires_bundle_with_scene_renderer(tmp_path):
    argv = [
        "--gpx",
        str(tmp_path / "track.gpx"),
        "--recorded-on",
        "2026-06-01",
        "--photos-dir",
        str(tmp_path),
        "--national-raster",
        str(tmp_path / "national.jpg"),
        "--national-credit",
        "Test",
        "--local-raster",
        str(tmp_path / "local.jpg"),
        "--local-credit",
        "Test",
        "--local-bounds",
        "-80.1",
        "34.9",
        "-79.9",
        "35.1",
        "--output",
        str(tmp_path / "out.mp4"),
        "--renderer",
        "scene",
    ]
    with pytest.raises(SystemExit) as excinfo:
        film_module.main(argv)
    assert excinfo.value.code != 0


class TestPoseTravelled:
    """The bright travelled line is geometry supplied by the renderer, cut at
    the same points the marker and camera already agree on -- never a
    separate fraction applied to a different geometry (the source of the
    travelled-line-runs-ahead-of-the-marker defect)."""

    def test_travelled_ends_at_the_marker_across_fifty_sampled_times(self):
        segments = _straight_segments(count=80)
        capture = _FakeCapture(WIDTH, MAP_HEIGHT)
        renderer = SceneMapRenderer(
            capture=capture, trip_epoch=TRIP_EPOCH, max_zoom=16.0, follow_zoom=14.0
        )
        total_time = segments[0][-1].time_s
        times = [total_time * index / 49.0 for index in range(50)]

        for at_time in times:
            renderer._render_frame(segments, at_time, bearing=0.0, width=WIDTH, height=HEIGHT)

        assert len(capture.poses) == 50
        for pose in capture.poses:
            assert "travelled" in pose
            assert pose["travelled"][-1][-1] == pose["marker"]

    def test_no_pose_carries_a_progress_key(self):
        segments = _straight_segments()
        capture = _FakeCapture(WIDTH, MAP_HEIGHT)
        renderer = SceneMapRenderer(
            capture=capture, trip_epoch=TRIP_EPOCH, max_zoom=16.0, follow_zoom=14.0
        )

        for at_time in (0.0, 10.0, 30.0, 59.0):
            renderer.render(segments, at_time, WIDTH, HEIGHT)

        assert all("progress" not in pose for pose in capture.poses)

    def test_travelled_has_one_part_per_started_segment_with_no_cross_gap_edge(self):
        # A GPS gap the display filter already split the track on: the
        # travelled line must be cut the same way, never bridged straight
        # across it.
        segments = [
            [RoutePoint(0.0, 35.0, -80.0, 100.0), RoutePoint(10.0, 35.001, -80.0, 100.0)],
            [RoutePoint(50.0, 35.002, -80.0, 100.0), RoutePoint(60.0, 35.003, -80.0, 100.0)],
        ]
        capture = _FakeCapture(WIDTH, MAP_HEIGHT)
        renderer = SceneMapRenderer(
            capture=capture, trip_epoch=TRIP_EPOCH, max_zoom=16.0, follow_zoom=14.0
        )

        renderer._render_frame(segments, 55.0, bearing=0.0, width=WIDTH, height=HEIGHT)

        pose = capture.poses[0]
        travelled = pose["travelled"]
        assert len(travelled) == 2
        assert travelled[-1][-1] == pose["marker"]
        for part in travelled:
            for first, second in zip(part, part[1:]):
                first_point = RoutePoint(0.0, first[1], first[0], 0.0)
                second_point = RoutePoint(0.0, second[1], second[0], 0.0)
                assert distance_m(first_point, second_point) < 200.0

    def test_marker_inside_a_gap_still_terminates_the_travelled_geometry(self):
        segments = [
            [RoutePoint(0.0, 35.0, -80.0, 100.0), RoutePoint(10.0, 35.001, -80.0, 100.0)],
            [RoutePoint(50.0, 35.002, -80.0, 100.0), RoutePoint(60.0, 35.003, -80.0, 100.0)],
        ]
        capture = _FakeCapture(WIDTH, MAP_HEIGHT)
        renderer = SceneMapRenderer(
            capture=capture, trip_epoch=TRIP_EPOCH, max_zoom=16.0, follow_zoom=14.0
        )

        renderer._render_frame(segments, 30.0, bearing=0.0, width=WIDTH, height=HEIGHT)

        pose = capture.poses[0]
        assert len(pose["travelled"]) == 1
        assert pose["travelled"][-1][-1] == pose["marker"]

    def test_the_whole_route_is_sent_on_every_pose_identically(self):
        # A pose must fully describe its own frame: a frame cache keys on the
        # pose dict and can look a pose up BEFORE it ever reaches the page
        # (a cache hit on frame 1 must never leave frame 2's page without a
        # route because "send once" skipped it). Every pose carries the same
        # route geometry; scene.js is what dedupes the repeated setData.
        segments = _straight_segments()
        capture = _FakeCapture(WIDTH, MAP_HEIGHT)
        renderer = SceneMapRenderer(
            capture=capture, trip_epoch=TRIP_EPOCH, max_zoom=16.0, follow_zoom=14.0
        )

        renderer.render(segments, 5.0, WIDTH, HEIGHT)
        renderer.render(segments, 15.0, WIDTH, HEIGHT)

        assert "route" in capture.poses[0]
        assert "route" in capture.poses[1]
        assert capture.poses[0]["route"] == capture.poses[1]["route"]

    def test_a_dataclasses_replace_clone_still_carries_the_route_on_its_pose(self):
        # The per-worker clone `_capture_leg_parallel` makes via
        # `dataclasses.replace` must not rely on any "already sent" flag
        # surviving the clone -- each worker's own page needs the route too.
        from dataclasses import replace

        segments = _straight_segments()
        capture = _FakeCapture(WIDTH, MAP_HEIGHT)
        renderer = SceneMapRenderer(
            capture=capture, trip_epoch=TRIP_EPOCH, max_zoom=16.0, follow_zoom=14.0
        )
        renderer.render(segments, 5.0, WIDTH, HEIGHT)

        worker_capture = _FakeCapture(WIDTH, MAP_HEIGHT)
        worker = replace(renderer, capture=worker_capture)
        worker.render(segments, 5.0, WIDTH, HEIGHT)

        assert "route" in worker_capture.poses[0]
        assert worker_capture.poses[0]["route"] == capture.poses[0]["route"]


class TestPoseMarkerHeading:
    def test_marker_heading_is_travel_direction_independent_of_camera_bearing(self):
        segments = _out_and_back_segments()
        capture = _FakeCapture(WIDTH, MAP_HEIGHT)
        renderer = SceneMapRenderer(
            capture=capture, trip_epoch=TRIP_EPOCH, max_zoom=16.0, follow_zoom=14.0
        )

        at_time = 2000.0
        renderer._render_frame(segments, at_time, bearing=17.0, width=WIDTH, height=HEIGHT)

        pose = capture.poses[0]
        assert "marker_heading" in pose
        assert pose["marker_heading"] == pytest.approx(
            _target_bearing(segments, at_time, scene_renderer_module._MIN_BEARING_WINDOW_S)
        )
        assert pose["marker_heading"] != pytest.approx(pose["bearing"])

    def test_marker_heading_is_continuous_across_a_frame_pair_on_a_straight_leg(self):
        segments = _straight_segments()
        capture = _FakeCapture(WIDTH, MAP_HEIGHT)
        renderer = SceneMapRenderer(
            capture=capture, trip_epoch=TRIP_EPOCH, max_zoom=16.0, follow_zoom=14.0
        )

        renderer.render(segments, 10.0, WIDTH, HEIGHT)
        renderer.render(segments, 10.5, WIDTH, HEIGHT)

        first, second = capture.poses[0]["marker_heading"], capture.poses[1]["marker_heading"]
        assert abs(second - first) < 5.0


class _LabelledCapture(_FakeCapture):
    """A good map frame carrying enough placed labels to cover >0.5% in dark pills."""

    def render(self, pose, light=None):
        array, _ = super().render(pose, light)
        labels = [
            {"name": f"Knob {i}", "visible": True, "x": 20 + (i % 4) * 75, "y": 15 + (i // 4) * 30}
            for i in range(16)
        ]
        return array, labels


def test_an_unloaded_raw_capture_fails_frame_qa_with_the_pose_that_made_it():
    from gomoveshift.pipeline.route_story.frame_qa import FrameQaError

    renderer = SceneMapRenderer(
        capture=_FakeCapture(WIDTH, MAP_HEIGHT, color=(16, 20, 24)),
        trip_epoch=TRIP_EPOCH,
        max_zoom=16.0,
    )

    with pytest.raises(FrameQaError, match="unloaded-tile") as excinfo:
        renderer.render(_straight_segments(), 10.0, WIDTH, HEIGHT)

    assert excinfo.value.pose is not None and "bearing" in excinfo.value.pose
    assert excinfo.value.rgb.shape == (MAP_HEIGHT, WIDTH, 3)


def test_a_page_white_raw_capture_fails_inside_a_map_leg_too():
    from gomoveshift.pipeline.route_story.frame_qa import FrameQaError

    renderer = SceneMapRenderer(
        capture=_FakeCapture(WIDTH, MAP_HEIGHT, color=(255, 255, 255)),
        trip_epoch=TRIP_EPOCH,
        max_zoom=16.0,
    )

    with pytest.raises(FrameQaError, match="page-white"):
        list(renderer.render_leg(_straight_segments(), [1.0, 2.0, 3.0], WIDTH, HEIGHT))


def test_render_leg_refuses_a_camera_jump_across_a_gps_gap_before_any_capture():
    # Two segments far apart with a time gap between them: a leg time that
    # lands past the gap's midpoint snaps the marker straight from segment
    # one's endpoint to segment two's start -- a real camera jump the
    # planned-camera continuity check must catch, and it must catch it
    # before a single frame is captured.
    from gomoveshift.pipeline.route_story.continuity import CameraContinuityError

    segment_one = [RoutePoint(float(i), 35.0, -80.0, 100.0) for i in range(6)]
    segment_two = [RoutePoint(20.0 + i, 36.0, -75.0, 100.0) for i in range(6)]
    segments = [segment_one, segment_two]
    times = [float(i) for i in range(14)]

    capture = _FakeCapture(WIDTH, MAP_HEIGHT)
    renderer = SceneMapRenderer(
        capture=capture, trip_epoch=TRIP_EPOCH, max_zoom=16.0, follow_zoom=14.0
    )

    with pytest.raises(CameraContinuityError):
        list(renderer.render_leg(segments, times, WIDTH, HEIGHT))

    assert capture.poses == []


def test_frame_qa_judges_the_raw_map_not_the_label_pills_or_dashboard_drawn_over_it():
    # Dark label pills alone can cover enough of a clean frame to read as "unloaded
    # tile", while the raw capture under them is clean, so QA must judge the raw map.
    renderer = SceneMapRenderer(
        capture=_LabelledCapture(WIDTH, MAP_HEIGHT), trip_epoch=TRIP_EPOCH, max_zoom=16.0
    )

    frame = renderer.render(_straight_segments(), 10.0, WIDTH, HEIGHT)

    assert frame.size == (WIDTH, HEIGHT)
    assert renderer.checks_raw_frames is True


_WILDES_KNOB = {
    "name": "Wildes Knob Overlook",
    "kind": "poi",
    "visible": True,
    "x": 100,
    "y": 80,
}
# Anchored at the same point, but a shorter name -- its pill's right edge
# never reaches the incumbent's own far-right corner, so that corner is only
# ever filled when "Wildes Knob Overlook" itself is the one drawn there.
_NEW_PEAK = {"name": "New Peak", "kind": "poi", "visible": True, "x": 100, "y": 80}


class _RankFlipCapture(_FakeCapture):
    """Wildes Knob is in view every frame of the leg; on frame 2 only, a
    same-spot newcomer is listed FIRST -- the shape of a real rank flip that
    lets a stateless ``place_labels`` collision-drop the incumbent for one
    frame while it is still on screen."""

    def render(self, pose, light=None):
        array, _ = super().render(pose, light)
        index = len(self.poses) - 1
        if index == 2:
            return array, [_NEW_PEAK, _WILDES_KNOB]
        return array, [_WILDES_KNOB]


def _wildes_knob_far_corner(width: int, height: int, full_height: int = HEIGHT) -> tuple[int, int]:
    from gomoveshift.pipeline.route_story.layout import hud_layout
    from gomoveshift.pipeline.route_story.proof import place_labels

    # Predict the same, possibly-relocated position the renderer actually
    # draws: place_labels now gets the dashboard/profile keep-clear boxes
    # scene_renderer.py resolves from hud_layout(width, full_height), the
    # frame's full (not map-cropped) height -- an unconstrained call here
    # would still predict the pre-keep-clear position and go stale.
    layout = hud_layout(width, full_height)
    dashboard_box = (
        layout.dashboard_left,
        layout.dashboard_panel_top,
        layout.dashboard_right,
        layout.dashboard_panel_top + layout.dashboard_panel_height,
    )
    profile_box = (
        layout.profile_left,
        layout.profile_top,
        layout.profile_right,
        layout.profile_top + layout.profile_height,
    )
    (pill,) = place_labels(
        [_WILDES_KNOB], width, height, extra_keep_clear=(dashboard_box, profile_box)
    )
    return int(pill.box[2]) - 2, int((pill.box[1] + pill.box[3]) / 2)


def _pill_drawn_at(frame: Image.Image, x: int, y: int) -> bool:
    from gomoveshift.pipeline.route_story.proof import _LABEL_FILL

    region = frame.crop((x - 1, y - 1, x + 2, y + 2))
    return _LABEL_FILL in {pixel[:3] for pixel in region.getdata()}


def test_render_leg_keeps_an_in_view_label_through_a_rank_flip_without_popping():
    renderer = SceneMapRenderer(
        capture=_RankFlipCapture(WIDTH, MAP_HEIGHT), trip_epoch=TRIP_EPOCH, max_zoom=16.0
    )

    frames = list(
        renderer.render_leg(_straight_segments(), [0.0, 1.0, 2.0, 3.0, 4.0], WIDTH, HEIGHT)
    )
    corner = _wildes_knob_far_corner(WIDTH, MAP_HEIGHT)

    assert len(frames) == 5
    for index, frame in enumerate(frames):
        assert _pill_drawn_at(frame, *corner), f"Wildes Knob pill missing on frame {index}"


class _ParallelRankFlipCapture:
    """Same rank-flip shape as ``_RankFlipCapture`` (a same-spot newcomer
    listed first on one frame), but keyed off the pose's own marker
    longitude rather than a per-instance call count -- a parallel worker
    only ever sees ITS OWN striped subset of a leg's frames, never all of
    them in order, so a call-count-based fake would answer a different
    question on every worker. Every fresh capture the factory hands a
    worker still answers the SAME frame identically."""

    def __init__(
        self,
        width: int,
        height: int,
        flip_index: int,
        origin_lon: float = -80.0,
        lon_step: float = 0.0006,
    ):
        self.width = width
        self.height = height
        self.flip_index = flip_index
        self.origin_lon = origin_lon
        self.lon_step = lon_step

    def __enter__(self) -> "_ParallelRankFlipCapture":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def render(self, pose, light=None):
        array = np.full((self.height, self.width, 3), (80, 90, 100), dtype=np.uint8)
        index = round((pose["marker"][0] - self.origin_lon) / self.lon_step)
        if index == self.flip_index:
            return array, [_NEW_PEAK, _WILDES_KNOB]
        return array, [_WILDES_KNOB]


def test_parallel_leg_render_keeps_the_same_label_persistence_as_serial():
    """A leg rendered with workers>1 must carry the same never-evict
    incumbent rule ``render_leg``'s serial branch already applies -- an
    in-view label popping for one frame under a rank flip is the same bug
    regardless of which capture path rendered it.
    """
    flip_index = 2
    segments = _straight_segments(count=6)
    times = [0.0, 1.0, 2.0, 3.0, 4.0]

    serial_renderer = SceneMapRenderer(
        capture=_ParallelRankFlipCapture(WIDTH, MAP_HEIGHT, flip_index),
        trip_epoch=TRIP_EPOCH,
        max_zoom=16.0,
    )
    serial_frames = list(serial_renderer.render_leg(segments, times, WIDTH, HEIGHT))

    parallel_renderer = SceneMapRenderer(
        capture=_ParallelRankFlipCapture(WIDTH, MAP_HEIGHT, flip_index),
        trip_epoch=TRIP_EPOCH,
        max_zoom=16.0,
        workers=4,
        capture_factory=lambda: _ParallelRankFlipCapture(WIDTH, MAP_HEIGHT, flip_index),
    )
    parallel_frames = list(parallel_renderer.render_leg(segments, times, WIDTH, HEIGHT))

    assert len(serial_frames) == len(parallel_frames) == len(times)
    corner = _wildes_knob_far_corner(WIDTH, MAP_HEIGHT)
    for index, (serial_frame, parallel_frame) in enumerate(zip(serial_frames, parallel_frames)):
        assert _pill_drawn_at(serial_frame, *corner), f"serial frame {index} missing incumbent pill"
        assert _pill_drawn_at(parallel_frame, *corner), (
            f"parallel frame {index} missing incumbent pill (dropped without persistence)"
        )
        assert np.array(serial_frame).tolist() == np.array(parallel_frame).tolist()
