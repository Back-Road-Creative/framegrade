"""Tests for the Route Story terrain-proof shot plan and render loop.

The render loop (``FfmpegFrameSink``/``render_frames``/``render_proof_clip``)
uses fakes for ``capture``/``sink`` everywhere except the dedicated
``FfmpegFrameSink`` deadlock test, so this suite runs fast and
deterministically. Real ``SceneCapture``/ffmpeg wiring is exercised only by
the real CLI run against a real bundle.
"""

from __future__ import annotations

import math
import sys
import threading
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from gomoveshift.pipeline.route_story import planner, proof
from gomoveshift.pipeline.route_story.frame_qa import (
    TRAVELLED_LINE_RGB,
    FrameQaError,
    UNLOADED_RGB,
)
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.scene import capture
from gomoveshift.pipeline.route_story.sun import SceneLight, light_at

FPS = 30.0
WIDTH = 1920.0
HEIGHT = 1080.0
TRIP_EPOCH = datetime(2026, 6, 21, 12, 0, 0, tzinfo=timezone.utc)


def _straight_route(n: int = 60, step_deg: float = 0.001) -> list[RoutePoint]:
    """A gently curving route heading roughly east, climbing then descending."""
    points = []
    for i in range(n):
        lat = 35.20 + 0.02 * math.sin(i / 20.0)
        lon = -84.00 + i * step_deg
        points.append(RoutePoint(time_s=float(i * 10), latitude=lat, longitude=lon))
    return points


def _hill_height_sampler(
    points: list[RoutePoint],
    peak_index: int,
    base: float = 500.0,
    peak: float = 1800.0,
    ramp_half_width: int = 3,
):
    """A flat-baseline route with one narrow triangular spike at ``peak_index``.

    A single unambiguous local maximum (rather than a route-spanning linear
    ramp, where any window on the ramp has the same local elevation range as
    any other -- a tie an unrelated float-rounding artifact then resolves
    arbitrarily) so ``choose_relief_section`` has exactly one correct answer.
    """
    heights = {}
    for i, point in enumerate(points):
        distance = abs(i - peak_index)
        if distance >= ramp_half_width:
            height = base
        else:
            height = base + (peak - base) * (1.0 - distance / ramp_half_width)
        heights[(round(point.longitude, 6), round(point.latitude, 6))] = height

    def sample(lon: float, lat: float) -> float | None:
        return heights.get((round(lon, 6), round(lat, 6)))

    return sample


# ---------------------------------------------------------------------------
# choose_relief_section
# ---------------------------------------------------------------------------


def test_choose_relief_section_picks_the_peak() -> None:
    points = _straight_route()
    sampler = _hill_height_sampler(points, peak_index=30)
    relief = proof.choose_relief_section(points, sampler, window_points=10)
    assert abs(relief.index - 30) <= 5
    assert relief.relief_range_m > 0


def test_choose_relief_section_forward_fills_missing_samples() -> None:
    points = _straight_route(n=20)

    def sampler(lon, lat):
        # Only every other point has a DEM sample.
        idx = round((lon + 84.00) / 0.001)
        return 1000.0 if idx % 2 == 0 else None

    relief = proof.choose_relief_section(points, sampler, window_points=6)
    assert relief.relief_range_m == pytest.approx(0.0)


def test_choose_relief_section_rejects_all_missing() -> None:
    points = _straight_route(n=10)
    with pytest.raises(ValueError, match="not enough terrain samples"):
        proof.choose_relief_section(points, lambda lon, lat: None)


# ---------------------------------------------------------------------------
# select_chase_points
# ---------------------------------------------------------------------------


def test_select_chase_points_walks_forward() -> None:
    points = _straight_route(n=40)
    selected = proof.select_chase_points(points, start_index=5, target_distance_m=500.0)
    assert selected[0] == points[5]
    assert len(selected) >= 2
    assert selected == points[5 : 5 + len(selected)]


def test_select_chase_points_falls_back_to_backward_at_route_end() -> None:
    points = _straight_route(n=40)
    selected = proof.select_chase_points(points, start_index=39, target_distance_m=500.0)
    assert selected[-1] == points[39]
    assert len(selected) >= 2


def test_select_chase_points_rejects_bad_index() -> None:
    points = _straight_route(n=5)
    with pytest.raises(ValueError):
        proof.select_chase_points(points, start_index=99, target_distance_m=100.0)


def test_select_chase_points_rejects_single_point_route() -> None:
    with pytest.raises(ValueError):
        proof.select_chase_points(
            [RoutePoint(0.0, 35.0, -84.0)], start_index=0, target_distance_m=100.0
        )


def _jittered_bend_route() -> list[RoutePoint]:
    """A chase start with a 0.5 m GPS hop, then ~25 m legs
    bending from 213 to 89 degrees."""
    legs = [(0.5, 211.0), (24.3, 213.0), (25.0, 201.0), (11.4, 121.0), (25.1, 98.0), (31.0, 89.0)]
    lat, lon = 35.2366, -83.5631
    points = [RoutePoint(time_s=0.0, latitude=lat, longitude=lon)]
    for i, (dist, bearing) in enumerate(legs, start=1):
        lat += dist * math.cos(math.radians(bearing)) / 111_320.0
        lon += dist * math.sin(math.radians(bearing)) / (111_320.0 * math.cos(math.radians(lat)))
        points.append(RoutePoint(time_s=float(i * 3), latitude=lat, longitude=lon))
    return points


def test_select_chase_points_drops_points_closer_than_gps_accuracy() -> None:
    points = _jittered_bend_route()

    selected = proof.select_chase_points(points, start_index=0, target_distance_m=500.0)

    assert selected[0] == points[0]
    assert points[1] not in selected
    assert selected[1:] == points[2:]


def test_chase_over_a_jittered_start_stays_within_the_bearing_speed_cap() -> None:
    """A sub-metre GPS hop must not get a sub-frame share of the chase: the
    smoothed heading would step many degrees in one frame, far past planner's
    25 deg/s cap."""
    points = _jittered_bend_route()
    selected = proof.select_chase_points(
        points, start_index=0, target_distance_m=proof.CHASE_SPEED_MPS * proof.CHASE_SECONDS
    )

    shot = proof.chase(selected, zoom=16.0, pitch=60.0, fps=FPS, seconds=proof.CHASE_SECONDS)
    bearings = [shot.pose_at(frame).bearing for frame in range(shot.end_frame)]
    rates = [abs(((b - a + 180.0) % 360.0 - 180.0) * FPS) for a, b in zip(bearings, bearings[1:])]

    assert max(rates) <= planner.DEFAULT_MOTION_LIMITS["bearing_speed_deg_s"]


# ---------------------------------------------------------------------------
# chase() shot
# ---------------------------------------------------------------------------


def test_chase_shot_starts_and_ends_on_the_real_route() -> None:
    points = _straight_route(n=20)[:8]
    shot = proof.chase(points, zoom=15.0, pitch=42.0, fps=FPS, seconds=2.0)
    start_pose = shot.pose_at(0)
    assert start_pose.longitude == pytest.approx(points[0].longitude)
    assert start_pose.latitude == pytest.approx(points[0].latitude)
    end_pose = shot._boundary_pose(shot.end_frame)
    assert end_pose.longitude == pytest.approx(points[-1].longitude)
    assert end_pose.latitude == pytest.approx(points[-1].latitude)
    assert end_pose.zoom == 15.0
    assert end_pose.pitch == 42.0


def test_chase_shot_heading_has_no_frame_to_frame_jitter() -> None:
    points = _straight_route(n=30)[:15]
    shot = proof.chase(points, zoom=15.0, pitch=42.0, fps=FPS, seconds=4.0)
    bearings = [shot.pose_at(f).bearing for f in range(shot.start_frame, shot.end_frame)]
    deltas = [abs(((b - a + 180.0) % 360.0) - 180.0) for a, b in zip(bearings, bearings[1:])]
    max_deg_per_frame = 25.0 / FPS  # matches planner.DEFAULT_MOTION_LIMITS bearing speed
    assert max(deltas) <= max_deg_per_frame + 1e-6


def test_chase_rejects_too_few_points() -> None:
    with pytest.raises(ValueError):
        proof.chase([RoutePoint(0.0, 35.0, -84.0)], zoom=15.0, pitch=42.0, fps=FPS, seconds=2.0)


def test_chase_rejects_nonpositive_duration() -> None:
    points = _straight_route(n=5)
    with pytest.raises(ValueError):
        proof.chase(points, zoom=15.0, pitch=42.0, fps=FPS, seconds=0.0)


# ---------------------------------------------------------------------------
# build_shot_plan
# ---------------------------------------------------------------------------


def _plan(n: int = 200, peak_index: int = 100):
    points = _straight_route(n=n)
    sampler = _hill_height_sampler(points, peak_index=peak_index)
    return points, proof.build_shot_plan(points, sampler, FPS, WIDTH, HEIGHT)


def test_build_shot_plan_produces_five_shots_in_order() -> None:
    _, proof_plan = _plan()
    kinds = [shot.kind for shot in proof_plan.plan.shots]
    assert kinds == ["descent", "reveal", "chase", "arrival", "pullback"]


def test_build_shot_plan_total_duration_in_20_to_30_seconds() -> None:
    _, proof_plan = _plan()
    duration_s = proof_plan.plan.frame_count / FPS
    assert 20.0 <= duration_s <= 30.0


def test_build_shot_plan_validates_and_is_smooth() -> None:
    # build_shot_plan already calls plan.validate() and assert_smooth();
    # a successful return is the assertion. Re-run explicitly for clarity.
    _, proof_plan = _plan()
    proof_plan.plan.validate()
    from gomoveshift.pipeline.route_story.planner import assert_smooth

    assert_smooth(proof_plan.plan, FPS)


def test_build_shot_plan_chase_starts_at_relief_point() -> None:
    points, proof_plan = _plan(n=200, peak_index=100)
    relief_point = points[proof_plan.relief.index]
    assert proof_plan.chase_points[0].longitude == pytest.approx(relief_point.longitude)
    assert proof_plan.chase_points[0].latitude == pytest.approx(relief_point.latitude)


def test_build_shot_plan_rejects_single_point_route() -> None:
    with pytest.raises(ValueError):
        proof.build_shot_plan(
            [RoutePoint(0.0, 35.0, -84.0)], lambda lo, la: 500.0, FPS, WIDTH, HEIGHT
        )


# ---------------------------------------------------------------------------
# marker_position
# ---------------------------------------------------------------------------


def test_marker_position_tracks_camera_during_chase_and_holds_elsewhere() -> None:
    points, proof_plan = _plan()
    plan = proof_plan.plan
    relief_point = proof_plan.chase_points[0]
    chase_end_point = proof_plan.chase_points[-1]

    chase_shot = next(s for s in plan.shots if s.kind == "chase")
    mid_chase_frame = (chase_shot.start_frame + chase_shot.end_frame) // 2
    marker = proof.marker_position(plan, mid_chase_frame, relief_point, chase_end_point)
    pose = plan.pose_at(mid_chase_frame)
    assert marker == (pose.longitude, pose.latitude)

    descent_shot = next(s for s in plan.shots if s.kind == "descent")
    marker_descent = proof.marker_position(
        plan, descent_shot.start_frame, relief_point, chase_end_point
    )
    assert marker_descent == (relief_point.longitude, relief_point.latitude)

    pullback_shot = next(s for s in plan.shots if s.kind == "pullback")
    marker_pullback = proof.marker_position(
        plan, pullback_shot.start_frame, relief_point, chase_end_point
    )
    assert marker_pullback == (chase_end_point.longitude, chase_end_point.latitude)


# ---------------------------------------------------------------------------
# Zoom floor: the plan may never ask for a zoom the bundle cannot serve
# ---------------------------------------------------------------------------


def test_build_shot_plan_keeps_every_pose_at_or_above_the_zoom_floor() -> None:
    points = _straight_route(n=200)
    sampler = _hill_height_sampler(points, peak_index=100)
    floor = 13.0

    proof_plan = proof.build_shot_plan(points, sampler, FPS, WIDTH, HEIGHT, min_zoom=floor)

    zooms = [proof_plan.plan.pose_at(f).zoom for f in range(proof_plan.plan.frame_count)]
    assert min(zooms) >= floor - 1e-9
    # Without a floor the wide shots dip below it — otherwise this proves nothing.
    unfloored = proof.build_shot_plan(points, sampler, FPS, WIDTH, HEIGHT)
    assert min(unfloored.plan.pose_at(f).zoom for f in range(unfloored.plan.frame_count)) < floor


def test_build_shot_plan_clamps_the_arrival_push_in_to_the_zoom_cap() -> None:
    """Default imagery stops at z15, so max_shot_zoom is ~15.32, but the
    arrival pushed in to FINAL_ZOOM + 0.6 = 15.6 and failed the render's zoom
    check for every default bundle."""
    points = _straight_route(n=200)
    sampler = _hill_height_sampler(points, peak_index=100)
    cap = planner.max_shot_zoom(15.0)

    proof_plan = proof.build_shot_plan(points, sampler, FPS, WIDTH, HEIGHT, max_zoom=cap)

    proof_plan.plan.assert_within_zoom(cap)
    arrival = proof_plan.plan.shots[3]
    assert arrival._boundary_pose(arrival.end_frame).zoom == pytest.approx(cap)


def test_build_shot_plan_with_a_floor_is_still_smooth() -> None:
    from gomoveshift.pipeline.route_story.planner import assert_smooth

    points = _straight_route(n=200)
    sampler = _hill_height_sampler(points, peak_index=100)

    proof_plan = proof.build_shot_plan(points, sampler, FPS, WIDTH, HEIGHT, min_zoom=13.0)

    proof_plan.plan.validate()
    assert_smooth(proof_plan.plan, FPS)


# ---------------------------------------------------------------------------
# render_frames with a fake capture + sink
# ---------------------------------------------------------------------------


class FakeCapture:
    """Renders a flat color frame and one fixed label; records every pose/light seen."""

    def __init__(self) -> None:
        self.poses: list[dict] = []
        self.lights: list[SceneLight | None] = []
        self.render_calls = 0

    def render(self, pose, light=None):
        self.poses.append(pose)
        self.lights.append(light)
        self.render_calls += 1
        array = np.full((HEIGHT_PX, WIDTH_PX, 3), 40, dtype=np.uint8)
        # inside the fake's own WIDTH_PX x HEIGHT_PX (64x36) canvas so the
        # drawn label rectangle actually lands on a pixel this test checks.
        # Short enough that its pill fits this 64 px canvas: place_labels drops
        # a name too wide for the frame rather than drawing it cut in half, so
        # a longer fixture name would leave nothing on the pixels to check.
        labels = [{"name": "Peak", "kind": "peak", "x": 20.0, "y": 20.0, "visible": True}]
        return array, labels


WIDTH_PX, HEIGHT_PX = 64, 36

FIXED_LIGHT = SceneLight(azimuth_deg=200.0, elevation_deg=30.0, intensity=0.6, night=False)


def test_render_frames_writes_every_frame_and_draws_labels() -> None:
    _, proof_plan = _plan()
    plan = proof_plan.plan
    capture = FakeCapture()
    sink = proof.ListFrameSink()
    relief_point = proof_plan.chase_points[0]
    chase_end_point = proof_plan.chase_points[-1]

    stats = proof.render_frames(
        plan,
        capture,
        sink,
        lambda f: proof.marker_position(plan, f, relief_point, chase_end_point),
        light_fn=lambda f: FIXED_LIGHT,
    )

    assert stats.frame_count == plan.frame_count
    assert len(sink.frames) == plan.frame_count
    assert stats.determinism_ok is True
    # the receipt checks determinism on 2 poses (frame 0 and the midpoint);
    # determinism_check re-renders each checked pose twice on top of the
    # frame's own render, so 2 checked frames add 4 extra render calls.
    assert capture.render_calls == plan.frame_count + 4
    # label text was drawn: the frame is no longer a flat 40 everywhere
    assert not np.all(sink.frames[0] == 40)
    # every main-loop call (excluding the light-less determinism-check calls)
    # carried the light_fn's own light.
    non_none_lights = [light for light in capture.lights if light is not None]
    assert non_none_lights == [FIXED_LIGHT] * plan.frame_count


def test_render_frames_raises_when_sink_close_fails() -> None:
    _, proof_plan = _plan(n=40, peak_index=20)
    plan = proof_plan.plan

    class FailingSink(proof.ListFrameSink):
        def close(self) -> bool:
            return False

    capture = FakeCapture()
    sink = FailingSink()
    relief_point = proof_plan.chase_points[0]
    chase_end_point = proof_plan.chase_points[-1]
    with pytest.raises(RuntimeError):
        proof.render_frames(
            plan,
            capture,
            sink,
            lambda f: proof.marker_position(plan, f, relief_point, chase_end_point),
            light_fn=lambda f: FIXED_LIGHT,
        )


def test_render_frames_frame_qa_failure_raises_and_still_closes_sink() -> None:
    _, proof_plan = _plan(n=40, peak_index=20)
    plan = proof_plan.plan

    class UnloadedCapture:
        def render(self, pose, light=None):
            array = np.full((HEIGHT_PX, WIDTH_PX, 3), UNLOADED_RGB, dtype=np.uint8)
            return array, []

    class TrackingSink(proof.ListFrameSink):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        def close(self) -> bool:
            self.closed = True
            return super().close()

    capture = UnloadedCapture()
    sink = TrackingSink()
    relief_point = proof_plan.chase_points[0]
    chase_end_point = proof_plan.chase_points[-1]

    with pytest.raises(FrameQaError):
        proof.render_frames(
            plan,
            capture,
            sink,
            lambda f: proof.marker_position(plan, f, relief_point, chase_end_point),
            light_fn=lambda f: FIXED_LIGHT,
        )

    assert sink.closed is True


def test_render_frames_fails_a_hairline_travelled_line_near_the_capture_marker() -> None:
    """capture.render's own "marker" label carries the real on-screen marker
    position; render_frames must thread it into check_frame's travelled-line
    floor rather than never checking it (marker_xy_from_labels wiring)."""
    _, proof_plan = _plan(n=40, peak_index=20)
    plan = proof_plan.plan

    class HairlineCapture:
        def render(self, pose, light=None):
            array = np.full((HEIGHT_PX, WIDTH_PX, 3), 40, dtype=np.uint8)
            marker_x, marker_y = 32.0, 18.0
            array[:, int(marker_x)] = TRAVELLED_LINE_RGB
            labels = [
                {"kind": "marker", "x": marker_x, "y": marker_y, "visible": True},
            ]
            return array, labels

    capture = HairlineCapture()
    sink = proof.ListFrameSink()
    relief_point = proof_plan.chase_points[0]
    chase_end_point = proof_plan.chase_points[-1]

    with pytest.raises(FrameQaError, match="travelled"):
        proof.render_frames(
            plan,
            capture,
            sink,
            lambda f: proof.marker_position(plan, f, relief_point, chase_end_point),
            light_fn=lambda f: FIXED_LIGHT,
        )


# ---------------------------------------------------------------------------
# render_frames with workers > 1: parallel capture, strict frame order
# ---------------------------------------------------------------------------


class _TinyPlan:
    """A minimal ``ShotPlan``-shaped stand-in: no shots/marker/light logic,
    just enough for ``render_frames`` -- the frame's own index is stashed on
    the pose's ``pitch`` field so a fake capture can identify which frame a
    render call is for without any real camera geometry.
    """

    def __init__(self, frame_count: int) -> None:
        self.frame_count = frame_count

    def pose_at(self, frame: int):
        from gomoveshift.pipeline.route_story.opening import CameraPose

        return CameraPose(
            longitude=-84.0 + frame * 0.0001,
            latitude=35.0,
            zoom=10.0,
            pitch=float(frame),
            bearing=0.0,
        )


def _light_for_frame(frame: int) -> SceneLight:
    """A light that actually varies by frame, so light_summary min/max are meaningful."""
    return SceneLight(
        azimuth_deg=float(frame % 360),
        elevation_deg=float((frame * 3) % 90),
        intensity=0.5,
        night=frame % 7 == 0,
    )


class FrameOrderCapture:
    """Stamps the frame index (from ``pose['pitch']``) into every pixel so a
    test can read back, from the sink alone, which frame each write was.
    """

    def __init__(self) -> None:
        self.render_calls = 0
        self.none_light_calls = 0
        self.frames_seen: list[int] = []

    def render(self, pose, light=None):
        self.render_calls += 1
        frame = int(pose["pitch"])
        if light is None:
            # a determinism_check() re-render of an already-seen frame, not a
            # new main-loop frame -- counted separately, never in frames_seen.
            self.none_light_calls += 1
        else:
            self.frames_seen.append(frame)
        # +50 keeps every value well clear of frame_qa's flat-black (<=8),
        # unloaded-tile (~16,20,24) and page-void (255,255,255) bands.
        array = np.full((HEIGHT_PX, WIDTH_PX, 3), 50 + frame, dtype=np.uint8)
        return array, []


class _CaptureContext:
    """Context manager around a fresh capture; records enter/exit for tests."""

    def __init__(self, capture, registry: list["_CaptureContext"]) -> None:
        self.capture = capture
        self.entered = False
        self.exited = False
        registry.append(self)

    def __enter__(self):
        self.entered = True
        return self.capture

    def __exit__(self, *exc_info) -> bool:
        self.exited = True
        return False


def _order_capture_factory(registry: list[_CaptureContext]):
    def factory():
        return _CaptureContext(FrameOrderCapture(), registry)

    return factory


@pytest.mark.parametrize(
    ("workers", "frame_count"),
    [
        (1, 7),
        (2, 7),  # not evenly divisible
        (3, 10),
        (5, 3),  # workers > frame_count
    ],
)
def test_render_frames_parallel_preserves_frame_order(workers, frame_count) -> None:
    plan = _TinyPlan(frame_count)
    sink = proof.ListFrameSink()
    registry: list[_CaptureContext] = []

    if workers == 1:
        capture = FrameOrderCapture()
        stats = proof.render_frames(
            plan,
            capture,
            sink,
            lambda f: (0.0, 0.0),
            light_fn=_light_for_frame,
            workers=workers,
        )
    else:
        stats = proof.render_frames(
            plan,
            None,
            sink,
            lambda f: (0.0, 0.0),
            light_fn=_light_for_frame,
            workers=workers,
            capture_factory=_order_capture_factory(registry),
        )
        # one capture per non-empty range, all entered and exited
        assert len(registry) == min(workers, frame_count)
        assert all(ctx.entered and ctx.exited for ctx in registry)

    observed = [int(frame[0, 0, 0]) - 50 for frame in sink.frames]
    assert observed == list(range(frame_count))
    assert stats.frame_count == frame_count


def test_render_frames_parallel_partitions_frames_disjointly_across_workers() -> None:
    frame_count = 11
    workers = 3
    plan = _TinyPlan(frame_count)
    sink = proof.ListFrameSink()
    registry: list[_CaptureContext] = []

    proof.render_frames(
        plan,
        None,
        sink,
        lambda f: (0.0, 0.0),
        light_fn=_light_for_frame,
        workers=workers,
        capture_factory=_order_capture_factory(registry),
    )

    per_worker_frames = [ctx.capture.frames_seen for ctx in registry]
    all_frames = [f for frames in per_worker_frames for f in frames]
    assert sorted(all_frames) == list(range(frame_count))
    # striped, not contiguous: each worker's own frames are exactly one
    # residue class mod the worker count -- i, i+N, i+2N, ...
    num_workers = len(registry)
    seen_residues: set[int] = set()
    for frames in per_worker_frames:
        assert frames == sorted(frames)
        residues = {f % num_workers for f in frames}
        assert len(residues) == 1, f"worker's own frames span multiple residues: {frames}"
        residue = residues.pop()
        assert residue not in seen_residues, "two workers claimed the same residue"
        seen_residues.add(residue)
        assert frames == list(range(residue, frame_count, num_workers))
    assert seen_residues == set(range(num_workers))


def test_render_frames_parallel_determinism_frames_checked_exactly_once_each() -> None:
    frame_count = 10
    plan = _TinyPlan(frame_count)
    sink = proof.ListFrameSink()
    registry: list[_CaptureContext] = []

    proof.render_frames(
        plan,
        None,
        sink,
        lambda f: (0.0, 0.0),
        light_fn=_light_for_frame,
        workers=3,
        capture_factory=_order_capture_factory(registry),
    )

    # determinism_check calls capture.render(pose) with no light kwarg, so a
    # call with light is None marks a determinism-check render; each of the
    # two checked frames (0 and frame_count // 2) triggers exactly 2 such
    # calls (determinism_check renders twice), 4 in total across all workers.
    total_none_light_calls = sum(ctx.capture.none_light_calls for ctx in registry)
    assert total_none_light_calls == 4


def test_render_frames_parallel_matches_serial_light_summary_and_frame_qa() -> None:
    frame_count = 9
    plan = _TinyPlan(frame_count)

    serial_sink = proof.ListFrameSink()
    serial_stats = proof.render_frames(
        plan,
        FrameOrderCapture(),
        serial_sink,
        lambda f: (0.0, 0.0),
        light_fn=_light_for_frame,
        workers=1,
    )

    parallel_sink = proof.ListFrameSink()
    registry: list[_CaptureContext] = []
    parallel_stats = proof.render_frames(
        plan,
        None,
        parallel_sink,
        lambda f: (0.0, 0.0),
        light_fn=_light_for_frame,
        workers=3,
        capture_factory=_order_capture_factory(registry),
    )

    assert parallel_stats.light_summary == serial_stats.light_summary
    assert parallel_stats.frame_qa == serial_stats.frame_qa
    assert parallel_stats.determinism_ok == serial_stats.determinism_ok is True


def test_render_frames_parallel_failure_closes_sink_exits_all_captures_and_raises_original() -> (
    None
):
    frame_count = 9
    bad_frame = 4  # 4 % 3 == 1: owned by worker 1's stride under workers=3
    plan = _TinyPlan(frame_count)

    class SelectivelyUnloadedCapture:
        def __init__(self) -> None:
            self.render_calls = 0

        def render(self, pose, light=None):
            self.render_calls += 1
            frame = int(pose["pitch"])
            if frame == bad_frame:
                array = np.full((HEIGHT_PX, WIDTH_PX, 3), UNLOADED_RGB, dtype=np.uint8)
            else:
                array = np.full((HEIGHT_PX, WIDTH_PX, 3), 40, dtype=np.uint8)
            return array, []

    registry: list[_CaptureContext] = []

    def factory():
        return _CaptureContext(SelectivelyUnloadedCapture(), registry)

    class TrackingSink(proof.ListFrameSink):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        def close(self) -> bool:
            self.closed = True
            return super().close()

    sink = TrackingSink()

    with pytest.raises(FrameQaError):
        proof.render_frames(
            plan,
            None,
            sink,
            lambda f: (0.0, 0.0),
            light_fn=_light_for_frame,
            workers=3,
            capture_factory=factory,
        )

    assert sink.closed is True
    assert len(registry) == 3
    assert all(ctx.entered and ctx.exited for ctx in registry)


def test_render_frames_parallel_keeps_every_worker_busy() -> None:
    """Would FAIL under contiguous frame ranges: the consumer drains worker 0's
    whole block before ever touching worker 1's queue, so worker 0 races ahead
    to ~frame_count/workers while 1 and 2 sit blocked on their own (maxsize=4)
    queue almost the entire render. Round-robin striding keeps every worker's
    next frame within one queue-depth of being consumed, so none starves.
    """
    workers = 3
    frame_count = 60
    plan = _TinyPlan(frame_count)
    sink = proof.ListFrameSink()

    counters = [0] * workers
    index_iter = iter(range(workers))
    index_lock = threading.Lock()

    class BusyTrackingCapture:
        def __init__(self, my_index: int) -> None:
            self._my_index = my_index

        def __enter__(self):
            return self

        def __exit__(self, *exc_info) -> bool:
            return False

        def render(self, pose, light=None):
            if light is not None:  # exclude determinism_check's extra re-renders
                counters[self._my_index] += 1
                my_count = counters[self._my_index]
                if my_count >= 8:
                    others = [c for i, c in enumerate(counters) if i != self._my_index]
                    assert all(c >= my_count - 8 for c in others), (
                        f"worker {self._my_index} reached {my_count} renders while "
                        f"the others are only at {others} -- a worker is starved"
                    )
            array = np.full((HEIGHT_PX, WIDTH_PX, 3), 60, dtype=np.uint8)
            return array, []

    def factory():
        with index_lock:
            idx = next(index_iter)
        return BusyTrackingCapture(idx)

    proof.render_frames(
        plan,
        None,
        sink,
        lambda f: (0.0, 0.0),
        light_fn=_light_for_frame,
        workers=workers,
        capture_factory=factory,
    )

    assert len(sink.frames) == frame_count


def test_render_frames_requires_capture_factory_when_workers_greater_than_one() -> None:
    plan = _TinyPlan(5)
    sink = proof.ListFrameSink()
    with pytest.raises(ValueError, match="capture_factory"):
        proof.render_frames(
            plan,
            None,
            sink,
            lambda f: (0.0, 0.0),
            light_fn=_light_for_frame,
            workers=3,
        )


# ---------------------------------------------------------------------------
# default_render_workers: pure sizing helper
# ---------------------------------------------------------------------------


def test_default_render_workers_is_half_the_cpus_capped_at_6_floored_at_1() -> None:
    assert proof.default_render_workers(cpu_count=1) == 1
    assert proof.default_render_workers(cpu_count=2) == 1
    assert proof.default_render_workers(cpu_count=4) == 2
    assert proof.default_render_workers(cpu_count=12) == 6
    assert proof.default_render_workers(cpu_count=24) == 6


# ---------------------------------------------------------------------------
# One scene browser at a time: the ceiling _resolve_workers may never exceed
# ---------------------------------------------------------------------------


def test_requested_workers_are_clamped_to_the_concurrent_capture_ceiling(monkeypatch) -> None:
    """An operator asking for 6 capture workers gets the number that can WORK.

    At the film's capture size one scene browser reaches ready in a couple of
    seconds, but a second one -- in the same process or a separate one -- can fail
    to reach ready at all, staying `loadFired: false` with style, tiles and labels
    all loaded. Without the clamp a render asked for six dies this way only after
    its prepare and preflight.

    So `--workers 6` is not merely slow, it cannot produce a film at all, and
    the cheapest place to refuse it is here: the one helper every scene render
    path resolves its worker count through.
    """
    monkeypatch.delenv(capture.MAX_CONCURRENT_CAPTURES_ENV, raising=False)

    assert proof._resolve_workers(6) == 1
    assert proof._resolve_workers(2) == 1
    assert proof._resolve_workers(1) == 1


def test_the_capture_ceiling_is_an_operator_override_not_a_hard_wire(monkeypatch) -> None:
    """A box where browsers DO coexist must not be held to the default limit.

    The clamp exists because of an observed property of some hosts, not a
    law about headless Chromium, so it has to be liftable without a code
    change -- otherwise a future GPU box silently renders six times slower.
    """
    monkeypatch.setenv(capture.MAX_CONCURRENT_CAPTURES_ENV, "4")

    assert proof._resolve_workers(6) == 4
    assert proof._resolve_workers(3) == 3


def test_a_junk_capture_ceiling_falls_back_to_one_rather_than_crashing(monkeypatch) -> None:
    monkeypatch.setenv(capture.MAX_CONCURRENT_CAPTURES_ENV, "banana")

    assert capture.max_concurrent_captures() == 1
    assert proof._resolve_workers(6) == 1


# ---------------------------------------------------------------------------
# route_light_fn: per-frame light (real chase time, sun_sweep pullback, fallback)
# ---------------------------------------------------------------------------


def _center(proof_plan) -> tuple[float, float]:
    lon = (proof_plan.route_bbox[0] + proof_plan.route_bbox[2]) / 2.0
    lat = (proof_plan.route_bbox[1] + proof_plan.route_bbox[3]) / 2.0
    return lat, lon


def test_route_light_fn_chase_time_advances_with_route_position() -> None:
    points, proof_plan = _plan()
    plan = proof_plan.plan
    chase_shot = next(s for s in plan.shots if s.kind == "chase")
    center_lat, center_lon = _center(proof_plan)

    light_fn = proof.route_light_fn(
        plan, points, proof_plan.chase_points, TRIP_EPOCH, center_lat, center_lon
    )

    start_light = light_fn(chase_shot.start_frame)
    end_light = light_fn(chase_shot.end_frame - 1)
    assert start_light.azimuth_deg != end_light.azimuth_deg

    start_pose = plan.pose_at(chase_shot.start_frame)
    expected_start = light_at(
        start_pose.latitude,
        start_pose.longitude,
        TRIP_EPOCH + timedelta(seconds=proof_plan.chase_points[0].time_s),
    )
    assert start_light.azimuth_deg == pytest.approx(expected_start.azimuth_deg)
    assert start_light.elevation_deg == pytest.approx(expected_start.elevation_deg)


def test_route_light_fn_pullback_sweeps_from_chase_end_to_trip_end() -> None:
    points, proof_plan = _plan()
    plan = proof_plan.plan
    pullback_shot = next(s for s in plan.shots if s.kind == "pullback")
    center_lat, center_lon = _center(proof_plan)

    light_fn = proof.route_light_fn(
        plan, points, proof_plan.chase_points, TRIP_EPOCH, center_lat, center_lon
    )

    chase_end_time = TRIP_EPOCH + timedelta(seconds=proof_plan.chase_points[-1].time_s)
    trip_end_time = TRIP_EPOCH + timedelta(seconds=max(p.time_s for p in points))
    expected_start = light_at(center_lat, center_lon, chase_end_time)
    expected_end = light_at(center_lat, center_lon, trip_end_time)

    got_start = light_fn(pullback_shot.start_frame)
    got_end = light_fn(pullback_shot.end_frame - 1)
    assert got_start.azimuth_deg == pytest.approx(expected_start.azimuth_deg)
    assert got_end.azimuth_deg == pytest.approx(expected_end.azimuth_deg)


def test_route_light_fn_opening_shots_use_trip_start_fallback() -> None:
    points, proof_plan = _plan()
    plan = proof_plan.plan
    descent_shot = next(s for s in plan.shots if s.kind == "descent")
    center_lat, center_lon = _center(proof_plan)

    light_fn = proof.route_light_fn(
        plan, points, proof_plan.chase_points, TRIP_EPOCH, center_lat, center_lon
    )

    trip_start_time = TRIP_EPOCH + timedelta(seconds=min(p.time_s for p in points))
    expected = light_at(center_lat, center_lon, trip_start_time)
    got = light_fn(descent_shot.start_frame)
    assert got.azimuth_deg == pytest.approx(expected.azimuth_deg)
    assert got.elevation_deg == pytest.approx(expected.elevation_deg)


# ---------------------------------------------------------------------------
# FfmpegFrameSink: close() must never deadlock on undrained stderr
# ---------------------------------------------------------------------------


def test_ffmpeg_frame_sink_close_does_not_deadlock_on_noisy_stderr(tmp_path, monkeypatch) -> None:
    """A fake ffmpeg that writes > the 64 KB pipe buffer to stderr, then reads
    stdin to EOF and exits 0. Before the fix, ``stderr=subprocess.PIPE`` with
    nothing draining it made the fake process block inside its own stderr
    write; ``close()``'s ``proc.wait()`` then never returned.
    """
    fake_ffmpeg = tmp_path / "fake_ffmpeg.py"
    fake_ffmpeg.write_text(
        "import sys\n"
        "sys.stderr.write('x' * 200_000)\n"
        "sys.stderr.flush()\n"
        "sys.stdin.buffer.read()\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        proof,
        "_ffmpeg_argv",
        lambda output_path, width, height, fps: [sys.executable, str(fake_ffmpeg)],
    )

    output_path = tmp_path / "out.mp4"
    sink = proof.FfmpegFrameSink(output_path, 64, 36, 30.0)
    sink.write(np.zeros((36, 64, 3), dtype=np.uint8))

    result: dict[str, object] = {}

    def _close() -> None:
        result["ok"] = sink.close()

    # The deadlock this guards never returns, so the budget only has to outlast
    # the fake ffmpeg's interpreter startup on a heavily loaded machine (it can
    # take more than 5 s just to start).
    thread = threading.Thread(target=_close, daemon=True)
    thread.start()
    thread.join(timeout=60.0)
    still_alive = thread.is_alive()
    if still_alive:
        sink._proc.kill()
        thread.join(timeout=2.0)

    assert not still_alive, "FfmpegFrameSink.close() deadlocked on undrained stderr"
    assert result.get("ok") is True


# ---------------------------------------------------------------------------
# render_proof_clip (full orchestration, fakes injected)
# ---------------------------------------------------------------------------


def test_render_proof_clip_end_to_end_with_fakes(tmp_path, monkeypatch) -> None:
    import json

    gpx_text = "\n".join(
        [
            '<?xml version="1.0"?>',
            "<gpx><trk><trkseg>",
        ]
        + [
            f'<trkpt lat="{35.20 + 0.02 * math.sin(i / 20.0):.6f}" '
            f'lon="{-84.00 + i * 0.001:.6f}">'
            f"<time>2026-09-16T14:{i // 60:02d}:{i % 60:02d}Z</time></trkpt>"
            for i in range(200)
        ]
        + ["</trkseg></trk></gpx>"]
    )
    gpx_path = tmp_path / "trip.gpx"
    gpx_path.write_text(gpx_text, encoding="utf-8")

    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    (bundle_root / "manifest.json").write_text(
        json.dumps(
            {
                "imagery": {"sources": [{"name": "NAIP imagery"}], "maxzoom": 16},
                "terrain": {"sources": [{"name": "USGS 3DEP DEM"}], "maxzoom": 13},
                "labels": {"sources": [{"name": "OpenStreetMap Overpass"}]},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        proof, "terrain_height_sampler", lambda root, zoom: lambda lo, la: 1000.0 + abs(lo) * 10
    )

    captures: list[FakeCapture] = []

    def capture_factory(root, w, h):
        class _Ctx:
            def __enter__(self):
                cap = FakeCapture()
                captures.append(cap)
                return cap

            def __exit__(self, *exc):
                return False

        return _Ctx()

    sinks: list[proof.ListFrameSink] = []

    def sink_factory(path, w, h, f):
        sink = proof.ListFrameSink()
        sinks.append(sink)
        return sink

    output_path = tmp_path / "out.mp4"
    receipt = proof.render_proof_clip(
        gpx_path,
        bundle_root,
        output_path,
        recorded_on=date(2026, 9, 16),
        fps=FPS,
        width=1920,
        height=1080,
        capture_factory=capture_factory,
        sink_factory=sink_factory,
    )

    assert receipt["schema"] == "route-story-proof-receipt/2"
    assert receipt["frame_count"] == len(sinks[0].frames)
    assert 20.0 <= receipt["duration_seconds"] <= 30.0
    assert receipt["determinism_ok"] is True
    assert [s["kind"] for s in receipt["shots"]] == [
        "descent",
        "reveal",
        "chase",
        "arrival",
        "pullback",
    ]
    assert receipt["sources"]["terrain"] == [{"name": "USGS 3DEP DEM"}]
    assert receipt["git_head"]  # non-empty, real sha or "unknown"
    assert receipt["render_fps"] > 0
    assert receipt["x_realtime"] > 0

    light = receipt["light"]
    assert light["source"] == "sun_track"
    assert light["azimuth_deg_min"] <= light["azimuth_deg_max"]
    assert light["elevation_deg_min"] >= 0.0
    assert light["elevation_deg_max"] >= light["elevation_deg_min"]
    assert isinstance(light["night_frames"], int)

    zoom_cap = receipt["zoom_cap"]
    assert zoom_cap["imagery_maxzoom"] == 16
    assert zoom_cap["max_zoom"] > zoom_cap["imagery_maxzoom"]
    assert zoom_cap["enlargement"] == pytest.approx(1.25)

    frame_qa = receipt["frame_qa"]
    assert frame_qa["frames_checked"] == receipt["frame_count"]
    assert frame_qa["frames_failed"] == 0


def test_render_proof_clip_receipt_carries_workers_and_readback(tmp_path, monkeypatch) -> None:
    import json

    gpx_text = "\n".join(
        [
            '<?xml version="1.0"?>',
            "<gpx><trk><trkseg>",
        ]
        + [
            f'<trkpt lat="{35.20 + 0.02 * math.sin(i / 20.0):.6f}" '
            f'lon="{-84.00 + i * 0.001:.6f}">'
            f"<time>2026-09-16T14:{i // 60:02d}:{i % 60:02d}Z</time></trkpt>"
            for i in range(200)
        ]
        + ["</trkseg></trk></gpx>"]
    )
    gpx_path = tmp_path / "trip.gpx"
    gpx_path.write_text(gpx_text, encoding="utf-8")

    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    (bundle_root / "manifest.json").write_text(
        json.dumps(
            {
                "imagery": {"sources": [{"name": "NAIP imagery"}], "maxzoom": 16},
                "terrain": {"sources": [{"name": "USGS 3DEP DEM"}], "maxzoom": 13},
                "labels": {"sources": [{"name": "OpenStreetMap Overpass"}]},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        proof, "terrain_height_sampler", lambda root, zoom: lambda lo, la: 1000.0 + abs(lo) * 10
    )

    def capture_factory(root, w, h):
        class _Ctx:
            def __enter__(self):
                return FakeCapture()

            def __exit__(self, *exc):
                return False

        return _Ctx()

    def sink_factory(path, w, h, f):
        return proof.ListFrameSink()

    output_path = tmp_path / "out.mp4"
    receipt = proof.render_proof_clip(
        gpx_path,
        bundle_root,
        output_path,
        recorded_on=date(2026, 9, 16),
        fps=FPS,
        width=1920,
        height=1080,
        capture_factory=capture_factory,
        sink_factory=sink_factory,
        workers=1,
        readback="png",
    )

    assert receipt["workers"] == 1
    assert receipt["readback"] == "png"


def test_render_proof_clip_zoom_cap_fails_before_capture_or_sink(tmp_path, monkeypatch) -> None:
    import json

    gpx_text = "\n".join(
        [
            '<?xml version="1.0"?>',
            "<gpx><trk><trkseg>",
        ]
        + [
            f'<trkpt lat="{35.20 + 0.02 * math.sin(i / 20.0):.6f}" '
            f'lon="{-84.00 + i * 0.001:.6f}">'
            f"<time>2026-09-16T14:{i // 60:02d}:{i % 60:02d}Z</time></trkpt>"
            for i in range(200)
        ]
        + ["</trkseg></trk></gpx>"]
    )
    gpx_path = tmp_path / "trip.gpx"
    gpx_path.write_text(gpx_text, encoding="utf-8")

    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    (bundle_root / "manifest.json").write_text(
        json.dumps(
            {
                # far too low to hold the proof's own descent/arrival zooms
                "imagery": {"sources": [{"name": "NAIP imagery"}], "maxzoom": 5},
                "terrain": {"sources": [{"name": "USGS 3DEP DEM"}], "maxzoom": 13},
                "labels": {"sources": [{"name": "OpenStreetMap Overpass"}]},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        proof, "terrain_height_sampler", lambda root, zoom: lambda lo, la: 1000.0 + abs(lo) * 10
    )

    def capture_factory(root, w, h):
        raise AssertionError("capture_factory must not run for an over-zoomed plan")

    def sink_factory(path, w, h, f):
        raise AssertionError("sink_factory must not run for an over-zoomed plan")

    output_path = tmp_path / "out.mp4"
    with pytest.raises(ValueError, match="exceeds max_shot_zoom"):
        proof.render_proof_clip(
            gpx_path,
            bundle_root,
            output_path,
            recorded_on=date(2026, 9, 16),
            fps=FPS,
            width=1920,
            height=1080,
            capture_factory=capture_factory,
            sink_factory=sink_factory,
        )


def test_render_proof_clip_selects_only_the_recorded_day(tmp_path, monkeypatch) -> None:
    """A GPX carrying another year's track (a real-world mixed-history export)
    must never widen the plan's route bbox or move its trip clock -- both the
    route points and ``trip_epoch`` come from ``recorded_on``'s own selected
    day, exactly as ``prepare_bundle`` selects it, never from every segment
    in the file."""
    import json

    trip_points = "\n".join(
        f'<trkpt lat="{35.20 + 0.02 * math.sin(i / 20.0):.6f}" '
        f'lon="{-84.00 + i * 0.001:.6f}">'
        f"<time>2026-09-16T14:{i // 60:02d}:{i % 60:02d}Z</time></trkpt>"
        for i in range(200)
    )
    gpx_text = "\n".join(
        [
            '<?xml version="1.0"?>',
            "<gpx>",
            '<trk><trkseg><trkpt lat="40.000000" lon="-70.000000">'
            "<time>2022-08-09T09:00:00Z</time></trkpt></trkseg></trk>",
            "<trk><trkseg>",
            trip_points,
            "</trkseg></trk>",
            "</gpx>",
        ]
    )
    gpx_path = tmp_path / "trip.gpx"
    gpx_path.write_text(gpx_text, encoding="utf-8")

    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    (bundle_root / "manifest.json").write_text(
        json.dumps(
            {
                "imagery": {"sources": [{"name": "NAIP imagery"}], "maxzoom": 16},
                "terrain": {"sources": [{"name": "USGS 3DEP DEM"}], "maxzoom": 13},
                "labels": {"sources": [{"name": "OpenStreetMap Overpass"}]},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        proof, "terrain_height_sampler", lambda root, zoom: lambda lo, la: 1000.0 + abs(lo) * 10
    )

    captured: dict = {}
    original_build_shot_plan = proof.build_shot_plan

    def spy_build_shot_plan(points, *args, **kwargs):
        captured["points"] = points
        return original_build_shot_plan(points, *args, **kwargs)

    monkeypatch.setattr(proof, "build_shot_plan", spy_build_shot_plan)

    original_route_light_fn = proof.route_light_fn

    def spy_route_light_fn(plan, points, chase_points, trip_epoch, center_lat, center_lon):
        captured["trip_epoch"] = trip_epoch
        return original_route_light_fn(
            plan, points, chase_points, trip_epoch, center_lat, center_lon
        )

    monkeypatch.setattr(proof, "route_light_fn", spy_route_light_fn)

    def capture_factory(root, w, h):
        class _Ctx:
            def __enter__(self):
                return FakeCapture()

            def __exit__(self, *exc):
                return False

        return _Ctx()

    def sink_factory(path, w, h, f):
        return proof.ListFrameSink()

    output_path = tmp_path / "out.mp4"
    proof.render_proof_clip(
        gpx_path,
        bundle_root,
        output_path,
        recorded_on=date(2026, 9, 16),
        fps=FPS,
        width=1920,
        height=1080,
        capture_factory=capture_factory,
        sink_factory=sink_factory,
    )

    lats = [p.latitude for p in captured["points"]]
    lons = [p.longitude for p in captured["points"]]
    assert min(lats) > 30.0  # excludes the 2022-08-09 point at lat 40.0
    assert max(lons) < -80.0  # excludes the 2022-08-09 point at lon -70.0
    assert captured["trip_epoch"] == datetime(2026, 9, 16, 14, 0, 0, tzinfo=timezone.utc)
